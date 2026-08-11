#!/usr/bin/env python3
"""Find incomplete training runs on Eagle and resubmit their seed indices.

A run is incomplete when it has no numeric checkpoint at or above
``--min-checkpoint-step``. A complete sibling from the same run family supplies
the original training configuration. The command is a dry run unless
``--execute`` is supplied.

Examples:

  python -m learning.recover_incomplete_cluster_runs \
      Go1JoystickRoughTerrainPushesAndDomainRandomization
  python -m learning.recover_incomplete_cluster_runs \
      Go1JoystickRoughTerrainPushesAndDomainRandomization \
      --run-pattern '260728-*' --execute
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
from pathlib import PurePosixPath
import re
import shlex
import subprocess
from typing import Sequence

from learning import download_models_to_evaluate as downloader
from learning import recover_missing_cluster_seeds as seed_recovery


@dataclass(frozen=True)
class IncompleteRun:
  family: str
  run_name: str
  seed: int
  latest_checkpoint: int | None
  template_run: str


def _checkpoint_inventory(
    host: str, environment_root: PurePosixPath
) -> dict[str, tuple[int, ...]]:
  """Returns numeric checkpoint steps for every immediate run directory."""
  quoted_root = shlex.quote(str(environment_root))
  lines = downloader._ssh_lines(
      host,
      f"find {quoted_root} -mindepth 1 -maxdepth 1 -type d "
      "-printf 'R\\t%f\\n'; "
      f"find {quoted_root} -mindepth 3 -maxdepth 3 -type d "
      f"-path {shlex.quote(str(environment_root / '*' / 'checkpoints' / '*'))} "
      "-printf 'C\\t%P\\n'",
  )
  inventory: dict[str, list[int]] = {}
  checkpoint_pattern = re.compile(r"(.+)/checkpoints/([0-9]+)")
  for line in lines:
    kind, separator, value = line.partition("\t")
    if not separator:
      continue
    if kind == "R":
      inventory.setdefault(value, [])
    elif kind == "C":
      match = checkpoint_pattern.fullmatch(value)
      if match is not None:
        inventory.setdefault(match.group(1), []).append(
            int(match.group(2))
        )
  return {
      run_name: tuple(sorted(steps))
      for run_name, steps in inventory.items()
  }


def _discover(
    inventory: dict[str, tuple[int, ...]],
    min_checkpoint_step: int,
    run_pattern: str,
) -> list[IncompleteRun]:
  """Finds incomplete seeded runs and chooses a configuration template."""
  families: dict[str, dict[int, tuple[str, int | None]]] = {}
  for run_name, checkpoints in inventory.items():
    match = seed_recovery.SEEDED_RUN.fullmatch(run_name)
    if match is None or not fnmatch.fnmatch(
        match.group("family"), run_pattern
    ):
      continue
    latest = max(checkpoints) if checkpoints else None
    families.setdefault(match.group("family"), {})[
        int(match.group("seed"))
    ] = (run_name, latest)

  result = []
  for family, runs in sorted(families.items()):
    templates = [
        (seed, run_name)
        for seed, (run_name, latest) in runs.items()
        if latest is not None and latest >= min_checkpoint_step
    ]
    for seed, (run_name, latest) in sorted(runs.items()):
      if latest is None or latest < min_checkpoint_step:
        # Prefer a completed sibling, but an incomplete run's run_config.json
        # still records the exact original command and handles families where
        # every seed failed.
        template_run = min(templates)[1] if templates else run_name
        result.append(
            IncompleteRun(
                family, run_name, seed, latest, template_run
            )
        )
  return result


def _archive_and_submit_command(
    project_root: PurePosixPath,
    environment_root: PurePosixPath,
    run: IncompleteRun,
    launcher_arguments: Sequence[str],
) -> str:
  archive_root = environment_root / ".incomplete-runs"
  source = environment_root / run.run_name
  destination = archive_root / run.run_name
  submission = shlex.join([
      "sbatch",
      f"--array={run.seed}",
      "slurm.sh",
      *launcher_arguments,
  ])
  return (
      "set -e; "
      f"mkdir -p {shlex.quote(str(archive_root))}; "
      f"test ! -e {shlex.quote(str(destination))}; "
      f"mv -- {shlex.quote(str(source))} "
      f"{shlex.quote(str(destination))}; "
      f"cd {shlex.quote(str(project_root))}; "
      f"{submission}"
  )


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("environment")
  parser.add_argument("--host", default=downloader.DEFAULT_HOST)
  parser.add_argument(
      "--remote-logs",
      type=PurePosixPath,
      default=downloader.DEFAULT_REMOTE_LOGS,
  )
  parser.add_argument(
      "--min-checkpoint-step",
      type=int,
      default=400_000_000,
      help="Minimum usable checkpoint step (default: 400000000).",
  )
  parser.add_argument(
      "--run-pattern",
      default="*",
      help="Shell-style run-family filter, for example '260728-*'.",
  )
  parser.add_argument(
      "--execute",
      action="store_true",
      help="Archive incomplete directories and submit replacements.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  if args.min_checkpoint_step <= 0:
    raise ValueError("--min-checkpoint-step must be positive")
  environment = downloader._validate_name(
      args.environment, "environment name"
  )
  host = downloader._validate_name(args.host, "SSH host")
  environment_root = args.remote_logs / environment
  inventory = _checkpoint_inventory(host, environment_root)
  incomplete = _discover(
      inventory, args.min_checkpoint_step, args.run_pattern
  )
  if not incomplete:
    print(
        f"No recoverable incomplete runs found for {environment} matching "
        f"{args.run_pattern!r}."
    )
    return 0

  prepared = []
  for run in incomplete:
    latest = (
        "none"
        if run.latest_checkpoint is None
        else f"{run.latest_checkpoint:012d}"
    )
    try:
      config = seed_recovery._saved_run_config(
          host, environment_root, run.template_run
      )
      launcher_arguments = seed_recovery._launcher_arguments(config)
      seed_recovery._validate_launcher_family(
          run.family, launcher_arguments
      )
    except (
        KeyError,
        ValueError,
        seed_recovery.json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
      print(f"{run.run_name}: cannot reconstruct launcher: {error}")
      continue
    command = _archive_and_submit_command(
        args.remote_logs.parent,
        environment_root,
        run,
        launcher_arguments,
    )
    prepared.append((run, command))
    print(
        f"{run.run_name}: latest checkpoint {latest}; "
        f"template {run.template_run}"
    )
    print(
        "  sbatch "
        f"--array={run.seed} slurm.sh "
        f"{shlex.join(launcher_arguments)}"
    )

  if not args.execute:
    print(
        "\nDry run only. Add --execute to archive these incomplete "
        "directories and submit their replacement seeds."
    )
    return 0

  for run, command in prepared:
    output = seed_recovery._ssh(host, command).strip()
    print(f"{run.run_name}: {output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
