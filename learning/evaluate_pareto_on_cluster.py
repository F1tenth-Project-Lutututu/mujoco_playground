"""Evaluate a Pareto manifest directly against checkpoints on Eagle.

This worker is intended to run inside one GPU Slurm allocation. It evaluates
all selected policies in one process so compatible policies reuse JAX
executables and each rollout uses a large batch of random tasks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Sequence


SAVE_FULL_SIGNALS = False


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--manifest", type=Path)
  source.add_argument("--environment")
  parser.add_argument("--models-root", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--run-date", type=int)
  parser.add_argument(
      "--min-checkpoint-step",
      type=int,
      default=400_000_000,
      help=(
          "Minimum progress required for eligibility. Checkpoint selection "
          "targets 400M, or 1B for Go1JoystickFlatTerrain25."
      ),
  )
  parser.add_argument("--num-random-tasks", type=int, default=1024)
  parser.add_argument("--task-seed", type=int, default=0)
  parser.add_argument("--episode-length", type=int, default=1000)
  parser.add_argument(
      "--continue-on-error",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  if args.num_random_tasks <= 0:
    raise ValueError("--num-random-tasks must be positive.")

  # These must be set before importing JAX through the evaluation modules.
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
  os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
  os.environ.setdefault("MUJOCO_GL", "egl")

  from learning import evaluate_all_models  # pylint: disable=g-import-not-at-top
  from learning import evaluate_policy  # pylint: disable=g-import-not-at-top
  from learning import pareto_policy_pipeline  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top

  devices = jax.devices()
  if jax.default_backend() != "gpu" or not any(
      device.platform == "gpu" for device in devices
  ):
    raise RuntimeError(
        "Cluster Pareto evaluation requires a visible CUDA GPU; "
        f"JAX backend is {jax.default_backend()!r}, devices={devices}. "
        "Check the Slurm GPU allocation and CUDA_VISIBLE_DEVICES."
    )
  evaluator_options = {
      option
      for action in evaluate_policy._build_parser()._actions
      for option in action.option_strings
  }
  required_option = "--no-torque_highpass_normalize_by_capacity"
  if required_option not in evaluator_options:
    raise RuntimeError(
        "The Eagle checkout is older than the cluster Pareto worker: "
        f"evaluate_policy.py does not support {required_option}. Update the "
        "remote repository to the same revision as localhost before "
        "submitting."
    )
  required_evaluator_version = 3
  evaluator_version = getattr(
      evaluate_policy, "EVALUATOR_COMPATIBILITY_VERSION", 0
  )
  if evaluator_version < required_evaluator_version:
    raise RuntimeError(
        "The Eagle checkout is older than the cluster Pareto worker: "
        "evaluate_policy.py lacks full trajectory archive support "
        f"(version {evaluator_version}, required "
        f"{required_evaluator_version}). Update the remote repository to "
        "the same revision as localhost before submitting."
    )

  if args.manifest is not None:
    manifest_path = args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = str(manifest["environment"])
  else:
    environment = str(args.environment)
    model_environment = args.models_root / environment
    selected = pareto_policy_pipeline.select_runs(
        [
            path.name
            for path in model_environment.iterdir()
            if path.is_dir()
        ],
        run_date=args.run_date,
        training_steps_millions=(
            1000
            if environment
            == pareto_policy_pipeline.LONG_HORIZON_ENVIRONMENT
            else None
        ),
    )
    default_target_step = pareto_policy_pipeline.evaluation_target_step(
        environment
    )
    checkpoint_paths = {}
    checkpoint_steps = {}
    for run in selected:
      checkpoints = model_environment / run.run_name / "checkpoints"
      numeric = (
          [
              path
              for path in checkpoints.iterdir()
              if path.is_dir() and path.name.isdigit()
          ]
          if checkpoints.is_dir()
          else []
      )
      checkpoint_paths[run.run_name] = numeric
      checkpoint_steps[run.run_name] = [int(path.name) for path in numeric]
    target_step = pareto_policy_pipeline.mixed_horizon_evaluation_target(
        selected, checkpoint_steps, default_target_step
    )
    if target_step != default_target_step:
      print(
          "Mixed 400M/1000M cohort: selecting checkpoints nearest "
          f"the common 400M terminal step {target_step:012d}."
      )
    completed = []
    skipped = []
    for run in selected:
      numeric = checkpoint_paths[run.run_name]
      if not numeric:
        skipped.append({
            **asdict(run),
            "reason": "no numeric checkpoint exists",
        })
        continue
      latest_step = max(int(path.name) for path in numeric)
      required_step = max(args.min_checkpoint_step, target_step)
      if latest_step < required_step:
        skipped.append({
            **asdict(run),
            "reason": (
                f"latest checkpoint {latest_step:012d} is below "
                f"{required_step:012d}"
            ),
        })
        continue
      checkpoint_name = pareto_policy_pipeline.select_checkpoint_name(
          [path.name for path in numeric], target_step
      )
      completed.append(
          pareto_policy_pipeline.PolicyRun(
              **{
                  **asdict(run),
                  "checkpoint": checkpoint_name,
              }
          )
      )
    if not completed:
      raise ValueError(f"No complete Pareto runs found for {environment}.")
    pareto_policy_pipeline.validate_sweeps(
        model_environment, completed, environment
    )
    environment_output = args.output_root / environment
    manifest_path = (
        environment_output / pareto_policy_pipeline.MANIFEST_NAME
    )
    pareto_policy_pipeline._write_manifest(
        manifest_path, environment, completed, selected, skipped
    )
    pareto_policy_pipeline._print_download_report(completed, skipped)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

  runs = manifest.get("runs", [])
  if not runs:
    raise ValueError(f"Manifest contains no evaluable runs: {manifest_path}")

  evaluate_all_models.MODEL_NAMES = frozenset(
      str(run["run_name"]) for run in runs
  )
  evaluate_all_models.MODELS_DIRECTORY = args.models_root
  evaluate_all_models.OUTPUT_DIRECTORY = args.output_root
  evaluate_all_models.NUM_RANDOM_TASKS = args.num_random_tasks
  evaluate_all_models.TASK_SEED = args.task_seed
  evaluate_all_models.EPISODE_LENGTH = args.episode_length
  evaluate_all_models.REQUIRE_CUDA = True
  evaluate_all_models.RENDER_VIDEO = False
  # Metrics are computed from the in-memory rollout tensors. Persisting the
  # full batched trajectories produces multi-gigabyte archives per policy
  # and makes large sweeps storage- and compression-bound.
  evaluate_all_models.SAVE_SIGNALS = SAVE_FULL_SIGNALS
  evaluate_all_models.USE_SAVED_ENVIRONMENT_CONFIG = True
  evaluate_all_models.TORQUE_NORMALIZATION_MODES = {"raw_torque": False}
  evaluate_all_models.CONTINUE_ON_ERROR = args.continue_on_error

  environment_output = args.output_root / environment
  environment_output.mkdir(parents=True, exist_ok=True)
  copied_manifest = environment_output / "pareto_manifest.json"
  if manifest_path.resolve() != copied_manifest.resolve():
    shutil.copy2(manifest_path, copied_manifest)
  worker_config = {
      "environment": environment,
      "manifest": str(manifest_path.resolve()),
      "models_root": str(args.models_root.resolve()),
      "output_root": str(args.output_root.resolve()),
      "num_random_tasks": args.num_random_tasks,
      "task_seed": args.task_seed,
      "episode_length": args.episode_length,
      "save_signals": SAVE_FULL_SIGNALS,
      "continue_on_error": args.continue_on_error,
      "xla_python_client_preallocate": os.environ[
          "XLA_PYTHON_CLIENT_PREALLOCATE"
      ],
      "xla_python_client_mem_fraction": os.environ[
          "XLA_PYTHON_CLIENT_MEM_FRACTION"
      ],
  }
  (environment_output / "cluster_evaluation_config.json").write_text(
      json.dumps(worker_config, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  evaluate_all_models.main(environment)


if __name__ == "__main__":
  main()
