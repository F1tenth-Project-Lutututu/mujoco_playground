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
(method, penalty scale, seed), downloads its target-aligned checkpoint, and
evaluates all policies on the same reproducible random tasks:

  python -m learning.pareto_policy_pipeline all

Download and evaluation can also be run separately with the ``download`` and
``evaluate`` subcommands.  Plot the stored reports with
``python -m learning.plot_policy_pareto``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from learning import download_models_to_evaluate as downloader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = "Go1JoystickFlatTerrain"
DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "eagle" / "pareto"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluations" / "pareto"
MANIFEST_NAME = "pareto_manifest.json"
EVALUATION_COVERAGE_NAME = "pareto_seed_coverage.json"
DEFAULT_ARCHIVE_PARTITION = "standard"
DEFAULT_EVALUATION_STEP = 400_000_000
LONG_HORIZON_ENVIRONMENT = "Go1JoystickFlatTerrain25"
LONG_HORIZON_EVALUATION_STEP = 1_000_000_000

# The date prefix is captured so repeated sweeps can be deduplicated in favor
# of the newest run.  High-pass cutoff/difference-order combinations are
# treated as separate methods; the historical f=5 Hz, m=1 combination keeps
# the plain ``high_pass`` name for manifest and plotting compatibility.
RUN_PATTERNS = {
    "action_smoothness": re.compile(
        r"(?P<date>\d{6})-actionsmoothness-(?P<steps>\d+)M-"
        r"as(?P<scale>[0-9]+e[mp][0-9]+)-seed(?P<seed>\d+)"
    ),
    "baseline": re.compile(
        r"(?P<date>\d{6})-baseline-(?P<steps>\d+)M-"
        r"ar(?P<scale>[0-9]+e[mp][0-9]+)"
        r"-seed(?P<seed>\d+)"
    ),
    "torque_rate": re.compile(
        r"(?P<date>\d{6})-torquerate-(?P<steps>\d+)M-"
        r"tr(?P<scale>[0-9]+e[mp][0-9]+)"
        r"-seed(?P<seed>\d+)"
    ),
    "high_pass": re.compile(
        r"(?P<date>\d{6})-highpass-(?P<steps>\d+)M-"
        r"hp(?P<scale>[0-9]+e[mp][0-9]+)"
        r"-f(?P<cutoff>[0-9]+(?:p[0-9]+)?)"
        r"o(?P<filter_order>[0-9]+)"
        r"m(?P<difference_order>[0-9]+(?:p[0-9]+)?)"
        r"-seed(?P<seed>\d+)"
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
  cutoff_hz: float | None = None
  filter_order: int | None = None
  difference_order: float | None = None
  training_steps_millions: int | None = None


def _decode_decimal_tag(tag: str) -> float:
  """Decodes an unambiguous decimal tag such as ``2p5``."""
  return float(tag.replace("p", "."))


def _decode_legacy_difference_order(tag: str) -> float:
  """Decodes the launcher's historical m tags (m10=1.0, m15=1.5)."""
  if "p" in tag:
    return _decode_decimal_tag(tag)
  return int(tag) / 10


def high_pass_method(
    cutoff_hz: float, difference_order: float, filter_order: int = 1
) -> str:
  """Returns the stable series name for one high-pass configuration."""
  if cutoff_hz == 5.0 and difference_order == 1.0:
    return "high_pass"
  cutoff = format(cutoff_hz, "g").replace(".", "p")
  difference = format(difference_order, "g").replace(".", "p")
  order = "" if filter_order == 1 else f"_o{filter_order}"
  return f"high_pass_f{cutoff}{order}_m{difference}"


def base_method(method: str) -> str:
  """Returns the reward family for a possibly configured method name."""
  return "high_pass" if method.startswith("high_pass_f") else method


def decode_scale(tag: str) -> float:
  """Decodes run-name scientific notation such as 2em3 as 2e-3."""
  match = re.fullmatch(r"([0-9]+)e([mp])([0-9]+)", tag)
  if match is None:
    raise ValueError(f"Invalid penalty scale tag: {tag!r}")
  sign = "-" if match.group(2) == "m" else "+"
  return float(f"{match.group(1)}e{sign}{match.group(3)}")


def select_runs(
    run_names: Sequence[str],
    run_date: int | None = None,
    training_steps_millions: int | None = None,
) -> list[PolicyRun]:
  """Selects runs for one date, or the newest per method, scale, and seed."""
  selected: dict[tuple[str, str, int], PolicyRun] = {}
  for run_name in run_names:
    for method, pattern in RUN_PATTERNS.items():
      match = pattern.fullmatch(run_name)
      if match is None:
        continue
      date = int(match.group("date"))
      if run_date is not None and date != run_date:
        break
      run_steps_millions = int(match.group("steps"))
      if (
          training_steps_millions is not None
          and run_steps_millions != training_steps_millions
      ):
        break
      cutoff_hz = None
      filter_order = None
      difference_order = None
      configured_method = method
      if method == "high_pass":
        cutoff_hz = _decode_decimal_tag(match.group("cutoff"))
        filter_order = int(match.group("filter_order"))
        difference_order = _decode_legacy_difference_order(
            match.group("difference_order")
        )
        configured_method = high_pass_method(
            cutoff_hz, difference_order, filter_order
        )
      run = PolicyRun(
          method=configured_method,
          scale_tag=match.group("scale"),
          scale=decode_scale(match.group("scale")),
          seed=int(match.group("seed")),
          date=date,
          run_name=run_name,
          cutoff_hz=cutoff_hz,
          filter_order=filter_order,
        difference_order=difference_order,
        training_steps_millions=run_steps_millions,
      )
      key = (run.method, run.scale_tag, run.seed)
      if key not in selected or run.date > selected[key].date:
        selected[key] = run
      break
  return sorted(
      selected.values(),
      key=lambda run: (run.method, run.scale, run.seed),
  )


def evaluation_target_step(environment: str) -> int:
  """Returns the common checkpoint step used to compare one environment."""
  if environment == LONG_HORIZON_ENVIRONMENT:
    return LONG_HORIZON_EVALUATION_STEP
  return DEFAULT_EVALUATION_STEP


def select_checkpoint_name(
    checkpoint_names: Sequence[str], target_step: int
) -> str:
  """Selects the numeric checkpoint nearest a target, preferring later ties."""
  numeric = [name for name in checkpoint_names if name.isdigit()]
  if not numeric:
    raise ValueError("No numeric checkpoints exist.")
  return min(numeric, key=lambda name: (abs(int(name) - target_step), -int(name)))


def _write_manifest(
    path: Path,
    environment: str,
    runs: Sequence[PolicyRun],
    selected_runs: Sequence[PolicyRun],
    skipped_runs: Sequence[dict],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "environment": environment,
      "selection": {
          "action_smoothness": "*actionsmoothness-*M-as*-seed*",
          "baseline": "*baseline-*M-ar*-seed*",
          "torque_rate": "*torquerate-*M-tr*-seed*",
          "high_pass": "*highpass-*M-hp*-f*o*m*-seed*",
          "duplicate_policy": (
              "newest date for each method, scale, and seed"
          ),
      },
      "runs": [asdict(run) for run in runs],
      "skipped_runs": list(skipped_runs),
      "seed_coverage": _seed_coverage(selected_runs, runs, skipped_runs),
  }
  path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )


def _seed_coverage(
    selected_runs: Sequence[PolicyRun],
    completed_runs: Sequence[PolicyRun],
    skipped_runs: Sequence[dict],
) -> list[dict]:
  """Summarizes expected, available, and skipped seeds per sweep point."""
  completed = {
      (run.method, run.scale_tag, run.seed) for run in completed_runs
  }
  skipped_by_key = {
      (item["method"], item["scale_tag"], item["seed"]): item["reason"]
      for item in skipped_runs
  }
  groups: dict[tuple[str, str], dict] = {}
  for run in selected_runs:
    group = groups.setdefault(
        (run.method, run.scale_tag),
        {
            "method": run.method,
            "scale": run.scale,
            "scale_tag": run.scale_tag,
            "expected_seeds": [],
            "completed_seeds": [],
            "failed_seeds": [],
        },
    )
    group["expected_seeds"].append(run.seed)
    key = (run.method, run.scale_tag, run.seed)
    if key in completed:
      group["completed_seeds"].append(run.seed)
    elif key in skipped_by_key:
      group["failed_seeds"].append({
          "seed": run.seed,
          "run_name": run.run_name,
          "reason": skipped_by_key[key],
      })
  result = []
  for group in groups.values():
    group["expected_seeds"].sort()
    group["completed_seeds"].sort()
    group["failed_seeds"].sort(key=lambda item: item["seed"])
    group["all_seeds_available"] = (
        group["expected_seeds"] == group["completed_seeds"]
    )
    result.append(group)
  return sorted(result, key=lambda item: (item["method"], item["scale"]))


def _print_download_report(
    completed: Sequence[PolicyRun], skipped: Sequence[dict]
) -> None:
  """Prints a concise post-download success and failure report."""
  print(
      f"Download report: {len(completed)} completed, {len(skipped)} skipped.",
      flush=True,
  )
  for item in skipped:
    print(
        f"  {item['run_name']} (seed {item['seed']}): {item['reason']}",
        flush=True,
    )


def _fill_missing_defaults(value: dict, defaults: dict) -> dict:
  """Recursively fills missing dictionary fields without replacing saved data."""
  result = copy.deepcopy(value)
  for key, default_value in defaults.items():
    if key not in result:
      result[key] = copy.deepcopy(default_value)
    elif isinstance(result[key], dict) and isinstance(default_value, dict):
      result[key] = _fill_missing_defaults(result[key], default_value)
  return result


def _comparable_run_config(
    config: dict,
    method: str,
    environment_defaults: dict | None = None,
) -> dict:
  """Removes seed/provenance and the swept scale from a run config."""
  result = copy.deepcopy(config)
  if environment_defaults is not None:
    saved_environment = result.get("environment_config", {})
    result["environment_config"] = _fill_missing_defaults(
        saved_environment, environment_defaults
    )
  result.pop("created_at", None)
  result.pop("seed", None)
  result.pop("command", None)
  result.get("ppo_config", {}).pop("seed", None)
  reward_name = {
      "action_smoothness": "action_rate",
      "baseline": "action_rate",
      "torque_rate": "torque_rate",
      "high_pass": "torque_high_freq",
  }[base_method(method)]
  result["environment_config"]["reward_config"]["scales"][reward_name] = (
      "<swept>"
  )
  override_name = f"reward_config.scales.{reward_name}"
  result["environment_config_overrides"][override_name] = "<swept>"
  return result


def validate_sweeps(
    local_environment: Path,
    runs: Sequence[PolicyRun],
    environment: str,
) -> None:
  """Checks that runs within each method differ only by scale and seed."""
  # Imported lazily so importing the download pipeline itself stays lightweight.
  from mujoco_playground._src import registry  # pylint: disable=g-import-not-at-top

  environment_defaults = registry.get_default_config(environment).to_dict()
  references: dict[str, tuple[dict, str]] = {}
  for run in runs:
    config_path = (
        local_environment / run.run_name / "checkpoints" / "run_config.json"
    )
    if not config_path.is_file():
      raise FileNotFoundError(f"Downloaded run config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    comparable = _comparable_run_config(
        config, run.method, environment_defaults
    )
    if run.method not in references:
      references[run.method] = (comparable, run.run_name)
      continue
    reference, reference_name = references[run.method]
    if comparable != reference:
      raise ValueError(
          f"{run.method} sweep configurations differ beyond penalty scale "
          f"and seed: {reference_name!r} versus {run.run_name!r}"
      )


def _download_archive(
    *,
    host: str,
    environment_root: PurePosixPath,
    local_environment: Path,
    members: Sequence[PurePosixPath],
    partition: str,
) -> None:
  """Packs selected files in a Slurm worker job and downloads one archive."""
  if not members:
    return
  transfer_id = uuid.uuid4().hex
  remote_transfer_root = environment_root / f".pareto-transfer-{transfer_id}"
  remote_list = remote_transfer_root / "members.txt"
  remote_archive = remote_transfer_root / "policies.tar.gz"
  remote_log = remote_transfer_root / "tar.log"
  quoted_transfer_root = shlex.quote(str(remote_transfer_root))
  member_text = "".join(f"{member.as_posix()}\n" for member in members)

  try:
    subprocess.run(
        [
            "ssh",
            host,
            f"mkdir -p {quoted_transfer_root} && "
            f"cat > {shlex.quote(str(remote_list))}",
        ],
        input=member_text,
        text=True,
        check=True,
    )
    tar_command = shlex.join([
        "tar",
        "-czf",
        str(remote_archive),
        "-C",
        str(environment_root),
        "-T",
        str(remote_list),
    ])
    print(
        f"Packaging {len(members)} paths on an Eagle {partition} worker...",
        flush=True,
    )
    submit_command = shlex.join([
        "sbatch",
        "--wait",
        "--parsable",
        f"--partition={partition}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=01:00:00",
        "--job-name=pareto_pack",
        f"--output={remote_log}",
        f"--wrap={tar_command}",
    ])
    downloader._run(("ssh", host, submit_command))

    local_environment.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pareto-archive-", dir=local_environment
    ) as temporary_directory:
      temporary_path = Path(temporary_directory)
      downloader._copy_remote(host, remote_archive, temporary_path)
      local_archive = temporary_path / remote_archive.name
      extraction_root = temporary_path / "extracted"
      extraction_root.mkdir()
      with tarfile.open(local_archive, "r:gz") as archive:
        archive.extractall(extraction_root, filter="data")
      for source in extraction_root.iterdir():
        target = local_environment / source.name
        if source.is_dir():
          shutil.copytree(source, target, dirs_exist_ok=True)
        else:
          shutil.copy2(source, target)
    print("Archive downloaded and unpacked.", flush=True)
  finally:
    # This directory is uniquely generated by this invocation.
    subprocess.run(
        ["ssh", host, f"rm -rf -- {quoted_transfer_root}"],
        text=True,
        check=False,
    )


def _checkpoint_complete(checkpoint: Path) -> bool:
  """Returns whether an Orbax PPO checkpoint has all evaluator inputs."""
  return (
      checkpoint.is_dir()
      and (checkpoint / "_sharding").is_file()
      and (checkpoint / "ppo_network_config.json").is_file()
  )


def _delete_remote_run_without_checkpoint(
    host: str,
    environment_root: PurePosixPath,
    run: PolicyRun,
) -> None:
  """Deletes one validated remote run known to have no numeric checkpoint."""
  if not any(
      pattern.fullmatch(run.run_name) for pattern in RUN_PATTERNS.values()
  ):
    raise ValueError(f"Refusing to delete unexpected run name: {run.run_name!r}")
  remote_run = environment_root / run.run_name
  if remote_run.parent != environment_root:
    raise ValueError(f"Refusing to delete unsafe remote path: {remote_run}")
  downloader._run((
      "ssh",
      host,
      f"rm -rf -- {shlex.quote(str(remote_run))}",
  ))


def download(
    environment: str,
    *,
    host: str,
    remote_logs: PurePosixPath,
    local_root: Path,
    min_checkpoint_step: int,
    archive_partition: str,
    run_date: int | None = None,
    delete_runs_without_checkpoints: bool = False,
) -> Path:
  """Downloads selected checkpoints and returns the manifest path."""
  environment_root = remote_logs / environment
  local_environment = local_root / environment
  runs = select_runs(
      downloader._remote_run_names(host, environment_root),
      run_date=run_date,
      training_steps_millions=(
          1000 if environment == LONG_HORIZON_ENVIRONMENT else None
      ),
  )
  if not runs:
    date_description = f" for date {run_date:06d}" if run_date else ""
    raise ValueError(
        "No requested Pareto sweep runs found"
        f"{date_description} at {environment_root}"
    )

  completed = []
  skipped: list[dict] = []
  archive_members: list[PurePosixPath] = []
  print(f"Selected {len(runs)} policies from {environment_root}.", flush=True)
  target_step = evaluation_target_step(environment)
  for index, run in enumerate(runs, start=1):
    remote_run = environment_root / run.run_name
    try:
      checkpoint_names, configs = downloader._checkpoint_inventory(
          host, remote_run
      )
    except ValueError as error:
      if "No numeric checkpoints found" not in str(error):
        raise
      print(
          f"[{index}/{len(runs)}] {run.run_name}: skipped because no numeric "
          "checkpoint exists",
          flush=True,
      )
      deleted = False
      if delete_runs_without_checkpoints:
        _delete_remote_run_without_checkpoint(host, environment_root, run)
        deleted = True
        print(
            f"[{index}/{len(runs)}] {run.run_name}: deleted remote run "
            "directory",
            flush=True,
        )
      skipped.append({
          **asdict(run),
          "reason": "no numeric checkpoint exists",
          "remote_directory_deleted": deleted,
      })
      continue
    latest_step = max(map(int, checkpoint_names))
    required_step = max(min_checkpoint_step, target_step)
    if latest_step < required_step:
      print(
          f"[{index}/{len(runs)}] {run.run_name}: skipped because latest "
          f"checkpoint {latest_step:012d} is below {required_step:012d}",
          flush=True,
      )
      skipped.append({
          **asdict(run),
          "reason": (
              f"latest checkpoint {latest_step:012d} is below "
              f"{required_step:012d}"
          ),
      })
      continue
    checkpoint = select_checkpoint_name(checkpoint_names, target_step)
    local_run = local_environment / run.run_name
    local_checkpoint = local_run / "checkpoints" / checkpoint
    print(
        f"[{index}/{len(runs)}] {run.run_name}: checkpoint {checkpoint}",
        flush=True,
    )
    # An interrupted transfer can leave a structurally valid Orbax checkpoint
    # without the separate network configuration required by the evaluator.
    checkpoint_complete = _checkpoint_complete(local_checkpoint)
    if not checkpoint_complete:
      if local_checkpoint.exists():
        print(
            f"[{index}/{len(runs)}] {run.run_name}: replacing incomplete "
            "local checkpoint",
            flush=True,
        )
      archive_members.append(
          (remote_run / "checkpoints" / checkpoint).relative_to(
              environment_root
          )
      )
    for remote_config in configs:
      relative_parent = remote_config.relative_to(remote_run).parent
      local_config = local_run / Path(relative_parent) / remote_config.name
      if not local_config.is_file():
        archive_members.append(
            remote_config.relative_to(environment_root)
        )
    completed.append(
        PolicyRun(**{**asdict(run), "checkpoint": checkpoint})
    )

  unique_members = list(dict.fromkeys(archive_members))
  _download_archive(
      host=host,
      environment_root=environment_root,
      local_environment=local_environment,
      members=unique_members,
      partition=archive_partition,
  )
  incomplete = [
      str(
          local_environment
          / run.run_name
          / "checkpoints"
          / str(run.checkpoint)
      )
      for run in completed
      if not _checkpoint_complete(
          local_environment
          / run.run_name
          / "checkpoints"
          / str(run.checkpoint)
      )
  ]
  if incomplete:
    _print_download_report(completed, skipped)
    preview = "\n".join(f"  {path}" for path in incomplete[:10])
    remainder = (
        f"\n  ... and {len(incomplete) - 10} more" if len(incomplete) > 10 else ""
    )
    raise FileNotFoundError(
        "Downloaded checkpoints are missing _sharding or "
        f"ppo_network_config.json:\n{preview}{remainder}"
    )
  manifest = local_environment / MANIFEST_NAME
  validate_sweeps(local_environment, completed, environment)
  _write_manifest(manifest, environment, completed, runs, skipped)
  _print_download_report(completed, skipped)
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
  evaluation_environment = output_root / environment
  evaluation_environment.mkdir(parents=True, exist_ok=True)
  coverage_path = evaluation_environment / EVALUATION_COVERAGE_NAME
  coverage_path.write_text(
      json.dumps(
          {
              "environment": environment,
              "source_manifest": str(manifest_path.resolve()),
              "seed_coverage": manifest.get("seed_coverage", []),
              "skipped_runs": manifest.get("skipped_runs", []),
          },
          indent=2,
          sort_keys=True,
      )
      + "\n",
      encoding="utf-8",
  )
  incomplete_coverage = [
      item
      for item in manifest.get("seed_coverage", [])
      if not item.get("all_seeds_available", True)
  ]
  print(f"Evaluation seed coverage: {coverage_path}", flush=True)
  for item in incomplete_coverage:
    missing = [
        failure["seed"] for failure in item.get("failed_seeds", [])
    ]
    print(
        f"  WARNING {item['method']} scale {item['scale']:g}: not all seeds "
        f"will be considered; missing {missing}",
        flush=True,
    )
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
      "--run-date",
      type=int,
      help=(
          "Restrict downloads to one YYMMDD run-name prefix. Use this when "
          "different dates contain incompatible experiment configurations."
      ),
  )
  parser.add_argument(
      "--delete-runs-without-checkpoints",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Permanently delete matched remote run directories when they have "
          "no numeric checkpoint, allowing the run name to be submitted "
          "again (default: false)."
      ),
  )
  parser.add_argument(
      "--min-checkpoint-step",
      type=int,
      default=400_000_000,
      help="Skip incomplete nominal 400M runs (default: %(default)s).",
  )
  parser.add_argument(
      "--archive-partition",
      default=DEFAULT_ARCHIVE_PARTITION,
      help="Eagle CPU partition used to build the transfer archive.",
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
      archive_partition=args.archive_partition,
      run_date=args.run_date,
      delete_runs_without_checkpoints=args.delete_runs_without_checkpoints,
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
