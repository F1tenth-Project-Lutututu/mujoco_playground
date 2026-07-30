#!/usr/bin/env python3
"""Report cluster evaluation coverage for quadruped locomotion environments.

The report compares unique saved training-run directories below the Eagle logs
root with unique runs having a completed top-level ``rollouts.csv`` below the
cluster Pareto evaluation root.

Example:

  python -m learning.report_cluster_evaluations
  python -m learning.report_cluster_evaluations --environment-pattern '*Spot*'
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
from pathlib import PurePosixPath
import shlex
from typing import Sequence

from learning import download_models_to_evaluate as downloader
from learning import pareto_cluster


QUADRUPED_PREFIXES = (
    "Barkour",
    "G1",
    "Go1",
    "SilverBadger",
    "Spot",
)


@dataclass(frozen=True)
class Coverage:
  environment: str
  evaluated_runs: int
  saved_runs: int

  @property
  def missing_runs(self) -> int:
    return self.saved_runs - self.evaluated_runs


def _is_quadruped_locomotion(environment: str, pattern: str) -> bool:
  return (
      "Joystick" in environment
      and environment.startswith(QUADRUPED_PREFIXES)
      and fnmatch.fnmatch(environment, pattern)
  )


def _environment_names(
    host: str, remote_logs: PurePosixPath, pattern: str
) -> list[str]:
  root = shlex.quote(str(remote_logs))
  names = downloader._ssh_lines(
      host,
      f"find {root} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n'",
  )
  return sorted(
      name for name in names if _is_quadruped_locomotion(name, pattern)
  )


def _evaluated_run_names(
    host: str,
    evaluation_root: PurePosixPath,
    environment: str,
) -> set[str]:
  raw_torque = evaluation_root / environment / "raw_torque"
  quoted_root = shlex.quote(str(raw_torque))
  paths = downloader._ssh_lines(
      host,
      f"if test -d {quoted_root}; then "
      f"find {quoted_root} -mindepth 3 -maxdepth 3 -type f "
      "-name rollouts.csv -printf '%p\\n'; fi",
  )
  result = set()
  for value in paths:
    path = PurePosixPath(value)
    try:
      relative = path.relative_to(raw_torque)
    except ValueError:
      continue
    if len(relative.parts) == 3:
      result.add(relative.parts[0])
  return result


def collect_coverage(
    host: str,
    remote_logs: PurePosixPath,
    evaluation_root: PurePosixPath,
    pattern: str = "*",
) -> list[Coverage]:
  """Collects saved-versus-evaluated unique run counts from the cluster."""
  rows = []
  for environment in _environment_names(host, remote_logs, pattern):
    saved = set(
        downloader._remote_run_names(host, remote_logs / environment)
    )
    evaluated = _evaluated_run_names(
        host, evaluation_root, environment
    )
    rows.append(
        Coverage(
            environment=environment,
            evaluated_runs=len(saved & evaluated),
            saved_runs=len(saved),
        )
    )
  return rows


def format_table(rows: Sequence[Coverage]) -> str:
  """Formats coverage rows as a compact aligned text table."""
  headers = ("Environment", "Reports", "Saved runs", "Missing", "Status")
  values = [
      (
          row.environment,
          str(row.evaluated_runs),
          str(row.saved_runs),
          str(row.missing_runs),
          "COMPLETE" if row.missing_runs == 0 else "INCOMPLETE",
      )
      for row in rows
  ]
  widths = [
      max(len(headers[index]), *(len(row[index]) for row in values))
      if values
      else len(headers[index])
      for index in range(len(headers))
  ]
  lines = [
      "  ".join(
          value.ljust(widths[index])
          if index in (0, 4)
          else value.rjust(widths[index])
          for index, value in enumerate(headers)
      ),
      "  ".join("-" * width for width in widths),
  ]
  lines.extend(
      "  ".join(
          value.ljust(widths[index])
          if index in (0, 4)
          else value.rjust(widths[index])
          for index, value in enumerate(row)
      )
      for row in values
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
  parser.add_argument(
      "--environment-pattern",
      default="*",
      help="Shell-style filter applied after quadruped discovery.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  host = downloader._validate_name(args.host, "SSH host")
  rows = collect_coverage(
      host,
      args.remote_logs,
      args.remote_evaluations,
      args.environment_pattern,
  )
  if not rows:
    print("No matching quadruped locomotion environments found.")
    return 0
  print(format_table(rows))
  print(
      f"\nTotal: {sum(row.evaluated_runs for row in rows)}/"
      f"{sum(row.saved_runs for row in rows)} evaluated; "
      f"{sum(row.missing_runs for row in rows)} missing."
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
