"""Evaluate a Pareto manifest directly against checkpoints on Eagle.

This worker is intended to run inside one GPU Slurm allocation. It evaluates
all selected policies in one process so compatible policies reuse JAX
executables and each rollout uses a large batch of random tasks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--models-root", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--num-random-tasks", type=int, default=2048)
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

  manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
  environment = str(manifest["environment"])
  runs = manifest.get("runs", [])
  if not runs:
    raise ValueError(f"Manifest contains no evaluable runs: {args.manifest}")

  from learning import evaluate_all_models  # pylint: disable=g-import-not-at-top

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
  evaluate_all_models.SAVE_SIGNALS = False
  evaluate_all_models.USE_SAVED_ENVIRONMENT_CONFIG = True
  evaluate_all_models.TORQUE_NORMALIZATION_MODES = {"raw_torque": False}
  evaluate_all_models.CONTINUE_ON_ERROR = args.continue_on_error

  environment_output = args.output_root / environment
  environment_output.mkdir(parents=True, exist_ok=True)
  shutil.copy2(args.manifest, environment_output / "pareto_manifest.json")
  worker_config = {
      "environment": environment,
      "manifest": str(args.manifest.resolve()),
      "models_root": str(args.models_root.resolve()),
      "output_root": str(args.output_root.resolve()),
      "num_random_tasks": args.num_random_tasks,
      "task_seed": args.task_seed,
      "episode_length": args.episode_length,
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
