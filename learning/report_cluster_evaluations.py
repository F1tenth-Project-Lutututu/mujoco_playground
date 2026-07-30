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
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import time
from typing import Sequence

from learning import download_models_to_evaluate as downloader
from learning import pareto_cluster


QUADRUPED_PREFIXES = (
    "Barkour",
    "Go1",
    "SilverBadger",
    "Spot",
)
CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_MAX_AGE_SECONDS = 300.0


def _default_cache_file() -> Path:
  cache_root = Path(
      os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
  )
  return cache_root / "mujoco_playground" / "cluster_evaluations.json"


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


def _cache_key(
    host: str,
    remote_logs: PurePosixPath,
    evaluation_root: PurePosixPath,
    pattern: str,
) -> str:
  return json.dumps(
      [host, str(remote_logs), str(evaluation_root), pattern],
      separators=(",", ":"),
  )


def _read_cached_coverage(
    cache_file: Path,
    key: str,
    max_age_seconds: float,
    now: float,
) -> list[Coverage] | None:
  try:
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    if not isinstance(cache, dict):
      return None
    entry = cache["entries"][key]
    if (
        cache["cache_format_version"] != CACHE_FORMAT_VERSION
        or now - float(entry["created_at"]) > max_age_seconds
    ):
      return None
    return [Coverage(**row) for row in entry["rows"]]
  except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
    return None


def _write_cached_coverage(
    cache_file: Path,
    key: str,
    rows: Sequence[Coverage],
    now: float,
) -> None:
  try:
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    if (
        not isinstance(cache, dict)
        or cache.get("cache_format_version") != CACHE_FORMAT_VERSION
    ):
      cache = {}
  except (OSError, ValueError, json.JSONDecodeError):
    cache = {}
  entries = cache.get("entries", {})
  if not isinstance(entries, dict):
    entries = {}
  entries[key] = {
      "created_at": now,
      "rows": [
          {
              "environment": row.environment,
              "evaluated_runs": row.evaluated_runs,
              "saved_runs": row.saved_runs,
          }
          for row in rows
      ],
  }
  payload = {
      "cache_format_version": CACHE_FORMAT_VERSION,
      "entries": entries,
  }
  try:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_file)
  except OSError:
    # Reporting should still succeed when the cache location is not writable.
    return


def collect_coverage_cached(
    host: str,
    remote_logs: PurePosixPath,
    evaluation_root: PurePosixPath,
    pattern: str = "*",
    *,
    cache_file: Path | None = None,
    max_age_seconds: float = DEFAULT_CACHE_MAX_AGE_SECONDS,
    refresh: bool = False,
) -> list[Coverage]:
  """Collects coverage, reusing a recent result for the same remote query."""
  cache_file = cache_file or _default_cache_file()
  key = _cache_key(host, remote_logs, evaluation_root, pattern)
  now = time.time()
  if not refresh:
    cached = _read_cached_coverage(cache_file, key, max_age_seconds, now)
    if cached is not None:
      return cached
  rows = collect_coverage(host, remote_logs, evaluation_root, pattern)
  _write_cached_coverage(cache_file, key, rows, time.time())
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
  parser.add_argument(
      "--cache-file",
      type=Path,
      default=None,
      help="Persistent cache path (default: the user cache directory).",
  )
  parser.add_argument(
      "--cache-max-age",
      type=float,
      default=DEFAULT_CACHE_MAX_AGE_SECONDS,
      metavar="SECONDS",
      help="Reuse cached results this old or newer (default: 300).",
  )
  parser.add_argument(
      "--refresh-cache",
      action="store_true",
      help="Ignore cached results and query the cluster.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)
  host = downloader._validate_name(args.host, "SSH host")
  if args.cache_max_age < 0:
    parser.error("--cache-max-age must be non-negative")
  rows = collect_coverage_cached(
      host,
      args.remote_logs,
      args.remote_evaluations,
      args.environment_pattern,
      cache_file=args.cache_file,
      max_age_seconds=args.cache_max_age,
      refresh=args.refresh_cache,
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
