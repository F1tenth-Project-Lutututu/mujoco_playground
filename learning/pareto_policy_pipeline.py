# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Download and evaluate penalty sweeps used for Pareto-front comparisons.

The default workflow selects the newest Eagle run for every
(method, penalty scale, seed), downloads its latest checkpoint, and evaluates
all policies on the same reproducible random tasks:

  python -m learning.pareto_policy_pipeline all

Download and evaluation can also be run separately with the ``download`` and
``evaluate`` subcommands.  Plot the stored reports with
``python -m learning.plot_policy_pareto``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Sequence

from learning import download_models_to_evaluate as downloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = "Go1JoystickFlatTerrain"
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "eagle" / "pareto"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluations" / "pareto"
MANIFEST_NAME = "pareto_manifest.json"

# The date prefix is captured so repeated sweeps can be deduplicated in favor
# of the newest run.  The high-pass family fixes f=5 Hz, order=1, and
# difference order=0 ("m10"); only its hp penalty scale varies.
RUN_PATTERNS = {
    "baseline": re.compile(
        r"(?P<date>\d{6})-baseline-400M-ar(?P<scale>[0-9]+e[mp][0-9]+)"
        r"-seed(?P<seed>\d+)"
    ),
    "torque_rate": re.compile(
        r"(?P<date>\d{6})-torquerate-400M-tr(?P<scale>[0-9]+e[mp][0-9]+)"
        r"-seed(?P<seed>\d+)"
    ),
    "high_pass": re.compile(
        r"(?P<date>\d{6})-highpass-400M-hp(?P<scale>[0-9]+e[mp][0-9]+)"
        r"-f5o1m10-seed(?P<seed>\d+)"
    ),
}


@dataclass(frozen=True)
class PolicyRun:
  method: str
  scale_tag: str
  scale: float
  seed: int
  date: int
  run_name: str
  checkpoint: str | None = None


def decode_scale(tag: str) -> float:
  """Decodes run-name scientific notation such as 2em3 as 2e-3."""
  match = re.fullmatch(r"([0-9]+)e([mp])([0-9]+)", tag)
  if match is None:
    raise ValueError(f"Invalid penalty scale tag: {tag!r}")
  sign = "-" if match.group(2) == "m" else "+"
  return float(f"{match.group(1)}e{sign}{match.group(3)}")


def select_runs(run_names: Sequence[str]) -> list[PolicyRun]:
  """Selects the newest run for every method, scale, and seed."""
  selected: dict[tuple[str, str, int], PolicyRun] = {}
  for run_name in run_names:
    for method, pattern in RUN_PATTERNS.items():
      match = pattern.fullmatch(run_name)
      if match is None:
        continue
      run = PolicyRun(
          method=method,
          scale_tag=match.group("scale"),
          scale=decode_scale(match.group("scale")),
          seed=int(match.group("seed")),
          date=int(match.group("date")),
          run_name=run_name,
      )
      key = (run.method, run.scale_tag, run.seed)
      if key not in selected or run.date > selected[key].date:
        selected[key] = run
      break
  return sorted(
      selected.values(),
      key=lambda run: (run.method, run.scale, run.seed),
  )


def _write_manifest(
    path: Path, environment: str, runs: Sequence[PolicyRun]
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "environment": environment,
      "selection": {
          "baseline": "*baseline-400M-ar*-seed*",
          "torque_rate": "*torquerate-400M-tr*-seed*",
          "high_pass": "*highpass-400M-hp*-f5o1m10-seed*",
          "duplicate_policy": (
              "newest date for each method, scale, and seed"
          ),
      },
      "runs": [asdict(run) for run in runs],
  }
  path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )


def _comparable_run_config(
    config: dict, method: str
) -> dict:
  """Removes seed/provenance and the swept scale from a run config."""
  result = copy.deepcopy(config)
  result.pop("created_at", None)
  result.pop("seed", None)
  result.pop("command", None)
  result.get("ppo_config", {}).pop("seed", None)
  reward_name = {
      "baseline": "action_rate",
      "torque_rate": "torque_rate",
      "high_pass": "torque_high_freq",
  }[method]
  result["environment_config"]["reward_config"]["scales"][reward_name] = (
      "<swept>"
  )
  override_name = f"reward_config.scales.{reward_name}"
  result["environment_config_overrides"][override_name] = "<swept>"
  return result


def validate_sweeps(local_environment: Path, runs: Sequence[PolicyRun]) -> None:
  """Checks that runs within each method differ only by scale and seed."""
  references: dict[str, tuple[dict, str]] = {}
  for run in runs:
    config_path = (
        local_environment / run.run_name / "checkpoints" / "run_config.json"
    )
    if not config_path.is_file():
      raise FileNotFoundError(f"Downloaded run config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    comparable = _comparable_run_config(config, run.method)
    if run.method not in references:
      references[run.method] = (comparable, run.run_name)
      continue
    reference, reference_name = references[run.method]
    if comparable != reference:
      raise ValueError(
          f"{run.method} sweep configurations differ beyond penalty scale "
          f"and seed: {reference_name!r} versus {run.run_name!r}"
      )


def download(
    environment: str,
    *,
    host: str,
    remote_logs: PurePosixPath,
    local_root: Path,
    min_checkpoint_step: int,
) -> Path:
  """Downloads selected checkpoints and returns the manifest path."""
  environment_root = remote_logs / environment
  local_environment = local_root / environment
  runs = select_runs(downloader._remote_run_names(host, environment_root))
  if not runs:
    raise ValueError(f"No requested Pareto sweep runs found at {environment_root}")

  completed = []
  print(f"Selected {len(runs)} policies from {environment_root}.", flush=True)
  for index, run in enumerate(runs, start=1):
    remote_run = environment_root / run.run_name
    checkpoint, configs = downloader._latest_checkpoint(host, remote_run)
    if int(checkpoint) < min_checkpoint_step:
      print(
          f"[{index}/{len(runs)}] {run.run_name}: skipped because latest "
          f"checkpoint {checkpoint} is below {min_checkpoint_step:012d}",
          flush=True,
      )
      continue
    local_run = local_environment / run.run_name
    local_checkpoint = local_run / "checkpoints" / checkpoint
    print(
        f"[{index}/{len(runs)}] {run.run_name}: checkpoint {checkpoint}",
        flush=True,
    )
    # Orbax checkpoints end with _sharding.  An interrupted legacy scp may
    # leave the directory in place without that file; download it again.
    checkpoint_complete = (
        local_checkpoint.is_dir()
        and (local_checkpoint / "_sharding").is_file()
    )
    if not checkpoint_complete:
      if local_checkpoint.exists():
        print(
            f"[{index}/{len(runs)}] {run.run_name}: replacing incomplete "
            "local checkpoint",
            flush=True,
        )
      downloader._copy_remote(
          host,
          remote_run / "checkpoints" / checkpoint,
          local_run / "checkpoints",
      )
    for remote_config in configs:
      relative_parent = remote_config.relative_to(remote_run).parent
      local_config = local_run / Path(relative_parent) / remote_config.name
      if not local_config.is_file():
        downloader._copy_remote(
            host, remote_config, local_run / Path(relative_parent)
        )
    completed.append(
        PolicyRun(**{**asdict(run), "checkpoint": checkpoint})
    )

  manifest = local_environment / MANIFEST_NAME
  validate_sweeps(local_environment, completed)
  _write_manifest(manifest, environment, completed)
  print(f"Download manifest: {manifest}", flush=True)
  return manifest


def evaluate(
    environment: str,
    *,
    local_root: Path,
    output_root: Path,
    num_random_tasks: int,
    task_seed: int,
    require_cuda: bool,
) -> None:
  """Evaluates downloaded policies using evaluate_all_models.py."""
  # Imported lazily so run discovery and download do not require evaluation
  # dependencies such as JAX, MuJoCo, and media codecs.
  from learning import evaluate_all_models  # pylint: disable=g-import-not-at-top

  manifest_path = local_root / environment / MANIFEST_NAME
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  evaluate_all_models.MODEL_NAMES = frozenset(
      str(run["run_name"]) for run in manifest["runs"]
  )
  evaluate_all_models.MODELS_DIRECTORY = local_root
  evaluate_all_models.OUTPUT_DIRECTORY = output_root
  evaluate_all_models.NUM_RANDOM_TASKS = num_random_tasks
  evaluate_all_models.TASK_SEED = task_seed
  evaluate_all_models.REQUIRE_CUDA = require_cuda
  # These policy families have different causal reward-memory observations.
  # Restore each saved environment config so its observation structure matches
  # the checkpoint; evaluation still uses common random tasks and reports
  # reward_without_regularization for the Pareto x-axis.
  evaluate_all_models.USE_SAVED_ENVIRONMENT_CONFIG = True
  # Raw torque is the common physical signal for policies trained with
  # different regularizers.  Capacity-normalized evaluation can be requested
  # later directly through evaluate_all_models when needed.
  evaluate_all_models.TORQUE_NORMALIZATION_MODES = {"raw_torque": False}
  evaluate_all_models.main(environment)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=("download", "evaluate", "all"))
  parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
  parser.add_argument("--host", default=downloader.DEFAULT_HOST)
  parser.add_argument(
      "--remote-logs",
      type=PurePosixPath,
      default=downloader.DEFAULT_REMOTE_LOGS,
  )
  parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
  parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
  parser.add_argument("--num-random-tasks", type=int, default=1024)
  parser.add_argument("--task-seed", type=int, default=0)
  parser.add_argument(
      "--min-checkpoint-step",
      type=int,
      default=400_000_000,
      help="Skip incomplete nominal 400M runs (default: %(default)s).",
  )
  parser.add_argument(
      "--require-cuda",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  if args.num_random_tasks <= 0:
    raise ValueError("--num-random-tasks must be positive.")
  if args.command in ("download", "all"):
    download(
        args.environment,
        host=args.host,
        remote_logs=args.remote_logs,
        local_root=args.local_root,
        min_checkpoint_step=args.min_checkpoint_step,
    )
  if args.command in ("evaluate", "all"):
    evaluate(
        args.environment,
        local_root=args.local_root,
        output_root=args.output_root,
        num_random_tasks=args.num_random_tasks,
        task_seed=args.task_seed,
        require_cuda=args.require_cuda,
    )


if __name__ == "__main__":
  main()
