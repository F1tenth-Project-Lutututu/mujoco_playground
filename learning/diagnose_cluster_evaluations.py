#!/usr/bin/env python3
"""Identify incomplete Eagle evaluations and explain each failure.

Unlike ``report_cluster_evaluations.py``, this command inspects the latest
numeric checkpoint of every saved run and validates the expected evaluation
artifacts.

Example:

  python -m learning.diagnose_cluster_evaluations
  python -m learning.diagnose_cluster_evaluations --environment-pattern '*Go1*'
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import shlex
from typing import Any, Sequence

from learning import download_models_to_evaluate as downloader
from learning import pareto_cluster
from learning import report_cluster_evaluations as coverage_report


_REMOTE_INSPECTOR = r"""
import csv
import json
import pathlib
import sys

logs = pathlib.Path(sys.argv[1])
evaluations = pathlib.Path(sys.argv[2])
environment = sys.argv[3]
saved_root = logs / environment
evaluation_root = evaluations / environment / "raw_torque"
rows = []

def directories(path, numeric_only=False):
  found = []
  errors = []
  try:
    children = path.iterdir()
    for child in children:
      if numeric_only and not child.name.isdigit():
        continue
      try:
        if child.is_dir():
          found.append(child)
      except OSError as error:
        errors.append("{}: {}".format(child, error))
  except FileNotFoundError:
    pass
  except OSError as error:
    errors.append("{}: {}".format(path, error))
  return found, errors

saved_runs, root_errors = directories(saved_root)
if root_errors:
  print(json.dumps({"fatal_errors": root_errors}, separators=(",", ":")))
  raise SystemExit()

for run in sorted(
    (run for run in saved_runs if not run.name.startswith(".")),
    key=lambda p: p.name,
):
  checkpoints_root = run / "checkpoints"
  checkpoints, checkpoint_errors = directories(checkpoints_root, numeric_only=True)
  latest = max(checkpoints, key=lambda p: int(p.name)).name if checkpoints else None
  run_output = evaluation_root / run.name
  outputs, output_errors = directories(run_output)
  outputs = sorted(
      outputs,
      key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name),
  )
  output_names = [p.name for p in outputs]
  target = run_output / latest if latest is not None else None
  artifacts = {}
  try:
    target_exists = target is not None and target.is_dir()
  except OSError as error:
    target_exists = False
    output_errors.append("{}: {}".format(target, error))
  if target_exists:
    for name in ("rollouts.csv", "summary.json"):
      path = target / name
      try:
        exists = path.is_file()
        artifacts[name] = {
            "exists": exists,
            "size": path.stat().st_size if exists else 0,
        }
      except OSError as error:
        artifacts[name] = {"exists": False, "size": 0}
        output_errors.append("{}: {}".format(path, error))
    rollouts = target / "rollouts.csv"
    if artifacts["rollouts.csv"]["exists"] and artifacts["rollouts.csv"]["size"]:
      try:
        with rollouts.open(newline="", encoding="utf-8") as stream:
          reader = csv.reader(stream)
          header = next(reader)
          first_row = next(reader)
        artifacts["rollouts.csv"]["valid"] = bool(header and first_row)
      except Exception as error:
        artifacts["rollouts.csv"]["valid"] = False
        artifacts["rollouts.csv"]["error"] = str(error)
    summary = target / "summary.json"
    if artifacts["summary.json"]["exists"] and artifacts["summary.json"]["size"]:
      try:
        value = json.loads(summary.read_text(encoding="utf-8"))
        artifacts["summary.json"]["valid"] = isinstance(value, dict)
      except Exception as error:
        artifacts["summary.json"]["valid"] = False
        artifacts["summary.json"]["error"] = str(error)
  rows.append({
      "run": run.name,
      "latest_checkpoint": latest,
      "evaluation_checkpoints": output_names,
      "artifacts": artifacts,
      "filesystem_errors": checkpoint_errors + output_errors,
  })

print(json.dumps(rows, separators=(",", ":")))
"""


@dataclass(frozen=True)
class Diagnosis:
  environment: str
  run: str
  checkpoint: str | None
  status: str
  detail: str


def _remote_inventory(
    host: str,
    remote_logs: PurePosixPath,
    evaluation_root: PurePosixPath,
    environment: str,
) -> list[dict[str, Any]]:
  command = shlex.join([
      "python3",
      "-c",
      _REMOTE_INSPECTOR,
      str(remote_logs),
      str(evaluation_root),
      environment,
  ])
  lines = downloader._ssh_lines(host, command)
  if len(lines) != 1:
    raise RuntimeError(
        f"Unexpected inventory response for {environment}: {len(lines)} lines"
    )
  value = json.loads(lines[0])
  if isinstance(value, dict) and value.get("fatal_errors"):
    raise RuntimeError(
        f"Cannot inspect {environment}: " + "; ".join(value["fatal_errors"])
    )
  if not isinstance(value, list):
    raise RuntimeError(f"Invalid inventory response for {environment}")
  return value


def _diagnose_row(environment: str, row: dict[str, Any]) -> Diagnosis:
  run = str(row["run"])
  checkpoint = row.get("latest_checkpoint")
  outputs = [str(value) for value in row.get("evaluation_checkpoints", [])]
  artifacts = row.get("artifacts", {})
  filesystem_errors = row.get("filesystem_errors", [])
  if filesystem_errors:
    return Diagnosis(
        environment, run, checkpoint, "FILESYSTEM_ERROR",
        "; ".join(str(error) for error in filesystem_errors),
    )
  if checkpoint is None:
    return Diagnosis(
        environment, run, None, "NO_CHECKPOINT",
        "saved run has no numeric checkpoint",
    )
  if checkpoint not in outputs:
    if outputs:
      return Diagnosis(
          environment, run, checkpoint, "STALE_CHECKPOINT",
          "latest checkpoint is not evaluated; existing outputs: "
          + ", ".join(outputs),
      )
    return Diagnosis(
        environment, run, checkpoint, "NOT_EVALUATED",
        "no evaluation output directory exists",
    )

  problems = []
  for filename, description in (
      ("rollouts.csv", "rollout report"),
      ("summary.json", "summary"),
  ):
    artifact = artifacts.get(filename, {})
    if not artifact.get("exists"):
      problems.append(f"missing {filename}")
    elif not artifact.get("size"):
      problems.append(f"empty {filename}")
    elif not artifact.get("valid", False):
      suffix = (
          f": {artifact['error']}" if artifact.get("error") else ""
      )
      problems.append(f"invalid {description}{suffix}")
  if problems:
    return Diagnosis(
        environment, run, checkpoint, "PARTIAL_OUTPUT", "; ".join(problems)
    )

  return Diagnosis(environment, run, checkpoint, "COMPLETE", "reports are valid")


def collect_diagnoses(
    host: str,
    remote_logs: PurePosixPath,
    evaluation_root: PurePosixPath,
    pattern: str = "*",
) -> list[Diagnosis]:
  """Returns a diagnosis for every saved run in matching environments."""
  result = []
  for environment in coverage_report._environment_names(
      host, remote_logs, pattern
  ):
    result.extend(
        _diagnose_row(environment, row)
        for row in _remote_inventory(
            host, remote_logs, evaluation_root, environment
        )
    )
  return result


def format_diagnoses(
    rows: Sequence[Diagnosis], *, include_complete: bool = False
) -> str:
  shown = [
      row for row in rows if include_complete or row.status != "COMPLETE"
  ]
  if not shown:
    return "No evaluation problems found."
  lines = []
  current_environment = None
  for row in shown:
    if row.environment != current_environment:
      if lines:
        lines.append("")
      lines.append(row.environment)
      current_environment = row.environment
    checkpoint = f" @ {row.checkpoint}" if row.checkpoint else ""
    lines.append(
        f"  {row.status:<20} {row.run}{checkpoint}\n"
        f"    {row.detail}"
    )
  return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default=downloader.DEFAULT_HOST)
  parser.add_argument(
      "--remote-logs",
      type=PurePosixPath,
      default=downloader.DEFAULT_REMOTE_LOGS,
  )
  parser.add_argument(
      "--remote-evaluations",
      type=PurePosixPath,
      default=pareto_cluster.DEFAULT_REMOTE_OUTPUT_ROOT,
  )
  parser.add_argument("--environment-pattern", default="*")
  parser.add_argument(
      "--include-complete",
      action="store_true",
      help="Also print runs whose latest checkpoint has every artifact.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  host = downloader._validate_name(args.host, "SSH host")
  rows = collect_diagnoses(
      host,
      args.remote_logs,
      args.remote_evaluations,
      args.environment_pattern,
  )
  print(format_diagnoses(rows, include_complete=args.include_complete))
  problems = sum(row.status != "COMPLETE" for row in rows)
  print(f"\nTotal: {problems} problematic, {len(rows) - problems} complete.")
  return 1 if problems else 0


if __name__ == "__main__":
  raise SystemExit(main())
