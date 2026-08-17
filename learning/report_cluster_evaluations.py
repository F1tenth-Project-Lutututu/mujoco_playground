#!/usr/bin/env python3
"""Report live cluster evaluation progress for quadruped environments.

When available, each evaluation's ``pareto_manifest.json`` defines the exact
deduplicated workload. Output directories and top-level ``rollouts.csv`` files
then provide read-only started and completed counters, respectively.

Example:

  python -m learning.report_cluster_evaluations
  python -m learning.report_cluster_evaluations --environment-pattern '*Spot*'
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from learning import download_models_to_evaluate as downloader
from learning import pareto_cluster

QUADRUPED_PREFIXES = (
    "Barkour",
    "Go1",
    "SilverBadger",
    "Spot",
)
CACHE_FORMAT_VERSION = 4


def _default_cache_file() -> Path:
  cache_root = Path(
      os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
  )
  return cache_root / "mujoco_playground" / "cluster_evaluations.json"


@dataclass(frozen=True)
class Coverage:
  environment: str
  evaluated_runs: int
  planned_runs: int
  active_runs: int = 0
  skipped_runs: int = 0
  manifest_available: bool = True

  @property
  def missing_runs(self) -> int:
    return self.planned_runs - self.evaluated_runs

  @property
  def progress_percent(self) -> float:
    return (
        100.0
        if self.planned_runs == 0
        else 100.0 * self.evaluated_runs / self.planned_runs
    )


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


def _started_run_names(
    host: str,
    evaluation_root: PurePosixPath,
    environment: str,
) -> set[str]:
  """Returns runs for which an evaluator created a checkpoint output dir."""
  raw_torque = evaluation_root / environment / "raw_torque"
  quoted_root = shlex.quote(str(raw_torque))
  paths = downloader._ssh_lines(
      host,
      f"if test -d {quoted_root}; then "
      f"find {quoted_root} -mindepth 2 -maxdepth 2 -type d "
      "-printf '%p\\n'; fi",
  )
  result = set()
  for value in paths:
    try:
      relative = PurePosixPath(value).relative_to(raw_torque)
    except ValueError:
      continue
    if len(relative.parts) == 2:
      result.add(relative.parts[0])
  return result


def _evaluation_manifest(
    host: str,
    evaluation_root: PurePosixPath,
    environment: str,
) -> dict | None:
  path = evaluation_root / environment / "pareto_manifest.json"
  quoted_path = shlex.quote(str(path))
  lines = downloader._ssh_lines(
      host,
      f"if test -f {quoted_path}; then cat -- {quoted_path}; fi",
  )
  if not lines:
    return None
  try:
    value = json.loads("\n".join(lines))
  except json.JSONDecodeError:
    return None
  return value if isinstance(value, dict) else None


def _changed_environment_names(
    host: str,
    evaluation_root: PurePosixPath,
    since: float,
) -> set[str]:
  """Returns environments with a changed plan or completed report."""
  quoted_root = shlex.quote(str(evaluation_root))
  paths = downloader._ssh_lines(
      host,
      f"if test -d {quoted_root}; then "
      f"find {quoted_root} -type f "
      "\\( -name rollouts.csv -o -name pareto_manifest.json \\) "
      f"-newermt {shlex.quote(f'@{since}')} -printf '%p\\n' "
      "2>/dev/null || true; fi",
  )
  changed = set()
  for value in paths:
    try:
      relative = PurePosixPath(value).relative_to(evaluation_root)
    except ValueError:
      continue
    if len(relative.parts) == 2 and relative.name == "pareto_manifest.json":
      changed.add(relative.parts[0])
    elif len(relative.parts) >= 3 and relative.parts[1] == "raw_torque":
      changed.add(relative.parts[0])
  return changed


def _collect_environment_coverage(
    host: str,
    remote_logs: PurePosixPath,
    evaluation_root: PurePosixPath,
    environment: str,
) -> Coverage:
  manifest = _evaluation_manifest(host, evaluation_root, environment)
  if manifest is None:
    # Before submission, retain the legacy saved-run estimate while clearly
    # marking that no exact evaluation plan exists yet.
    planned = {
        name
        for name in downloader._remote_run_names(
            host, remote_logs / environment
        )
        if not name.startswith(".")
    }
    skipped_count = 0
    manifest_available = False
  else:
    planned = {
        str(run["run_name"])
        for run in manifest.get("runs", [])
        if isinstance(run, dict) and "run_name" in run
    }
    skipped_count = len(manifest.get("skipped_runs", []))
    manifest_available = True
  evaluated = _evaluated_run_names(host, evaluation_root, environment)
  started = _started_run_names(host, evaluation_root, environment)
  return Coverage(
      environment=environment,
      evaluated_runs=len(planned & evaluated),
      planned_runs=len(planned),
      active_runs=len((planned & started) - evaluated),
      skipped_runs=skipped_count,
      manifest_available=manifest_available,
  )


def collect_coverage(
    host: str,
    remote_logs: PurePosixPath,
    evaluation_root: PurePosixPath,
    pattern: str = "*",
) -> list[Coverage]:
  """Collects saved-versus-evaluated unique run counts from the cluster."""
  return [
      _collect_environment_coverage(
          host, remote_logs, evaluation_root, environment
      )
      for environment in _environment_names(host, remote_logs, pattern)
  ]


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
) -> tuple[float, list[Coverage]] | None:
  try:
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    if not isinstance(cache, dict):
      return None
    entry = cache["entries"][key]
    if cache["cache_format_version"] != CACHE_FORMAT_VERSION:
      return None
    return (
        float(entry["checked_at"]),
        [Coverage(**row) for row in entry["rows"]],
    )
  except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
    return None


def _write_cached_coverage(
    cache_file: Path,
    key: str,
    rows: Sequence[Coverage],
    checked_at: float,
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
      "checked_at": checked_at,
      "rows": [
          {
              "environment": row.environment,
              "evaluated_runs": row.evaluated_runs,
            "planned_runs": row.planned_runs,
            "active_runs": row.active_runs,
            "skipped_runs": row.skipped_runs,
            "manifest_available": row.manifest_available,
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
    refresh: bool = False,
) -> list[Coverage]:
  """Refreshes only environments changed since the previous invocation."""
  cache_file = cache_file or _default_cache_file()
  key = _cache_key(host, remote_logs, evaluation_root, pattern)
  scan_started_at = time.time()
  cached = None if refresh else _read_cached_coverage(cache_file, key)
  if cached is None:
    rows = collect_coverage(host, remote_logs, evaluation_root, pattern)
  else:
    checked_at, cached_rows = cached
    environments = _environment_names(host, remote_logs, pattern)
    cached_by_environment = {row.environment: row for row in cached_rows}
    changed = _changed_environment_names(
        host, evaluation_root, checked_at
    )
    refresh_environments = changed | (
        set(environments) - cached_by_environment.keys()
    )
    refresh_environments |= {
        row.environment for row in cached_rows if row.missing_runs > 0
    }
    rows = [
        _collect_environment_coverage(
            host, remote_logs, evaluation_root, environment
        )
        if environment in refresh_environments
        else cached_by_environment[environment]
        for environment in environments
    ]
  _write_cached_coverage(cache_file, key, rows, scan_started_at)
  return rows


def format_table(rows: Sequence[Coverage]) -> str:
  """Formats coverage rows as a compact aligned text table."""
  headers = (
      "Environment",
      "Done",
      "Active",
      "Planned",
      "Skipped",
      "Remaining",
      "Progress",
      "Status",
  )
  values = [
      (
          row.environment,
          str(row.evaluated_runs),
          str(row.active_runs),
          str(row.planned_runs),
          str(row.skipped_runs),
          str(row.missing_runs),
          f"{row.progress_percent:5.1f}%",
          (
              "COMPLETE"
              if row.missing_runs == 0
              else "RUNNING"
              if row.active_runs > 0 or row.evaluated_runs > 0
              else "PLANNED"
              if row.manifest_available
              else "UNPLANNED"
          ),
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
          if index in (0, len(headers) - 1)
          else value.rjust(widths[index])
          for index, value in enumerate(headers)
      ),
      "  ".join("-" * width for width in widths),
  ]
  lines.extend(
      "  ".join(
          value.ljust(widths[index])
          if index in (0, len(headers) - 1)
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
      "--refresh-cache",
      action="store_true",
      help="Ignore cached results and query the cluster.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)
  host = downloader._validate_name(args.host, "SSH host")
  rows = collect_coverage_cached(
      host,
      args.remote_logs,
      args.remote_evaluations,
      args.environment_pattern,
      cache_file=args.cache_file,
      refresh=args.refresh_cache,
  )
  if not rows:
    print("No matching quadruped locomotion environments found.")
    return 0
  print(format_table(rows))
  print(
      f"\nTotal: {sum(row.evaluated_runs for row in rows)}/"
      f"{sum(row.planned_runs for row in rows)} planned evaluations complete; "
      f"{sum(row.active_runs for row in rows)} active, "
      f"{sum(row.skipped_runs for row in rows)} skipped before evaluation, "
      f"{sum(row.missing_runs for row in rows)} remaining."
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
