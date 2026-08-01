#!/usr/bin/env python3
"""Recompute rollout metrics from saved batched trajectory archives.

The manifest supplies exact policy/checkpoint paths, avoiding recursive
filesystem discovery.  Each ``signals.npz`` is loaded once and every selected
metric is computed over all rollouts with vectorized NumPy/SciPy operations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from scipy import signal as scipy_signal


JOINT_VELOCITY_MSSD = (
    "smoothness/joint_velocity/"
    "mssd_mean_squared_second_difference_per_dof"
)
JOINT_VELOCITY_MSGFD = (
    "smoothness/joint_velocity/"
    "msgfd_mean_absolute_savgol_filter_deviation_per_dof"
)
DEFAULT_METRICS = (JOINT_VELOCITY_MSSD, JOINT_VELOCITY_MSGFD)
METRIC_ALIASES = {
    "joint_velocity_mssd": JOINT_VELOCITY_MSSD,
    "joint_velocity_msgfd": JOINT_VELOCITY_MSGFD,
}


def _safe_name(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "scenario"


def _mssd(signal: np.ndarray, active: np.ndarray) -> np.ndarray:
  """Computes MSSD for every rollout in one vectorized batch."""
  second_delta = np.diff(signal, n=2, axis=0)
  valid = active[:-2] & active[1:-1] & active[2:]
  squared = np.square(second_delta)
  numerator = np.sum(squared * valid[..., None], axis=(0, 2))
  denominator = np.sum(valid, axis=0) * signal.shape[-1]
  return np.divide(
      numerator,
      denominator,
      out=np.zeros(signal.shape[1], dtype=np.float64),
      where=denominator > 0,
  )


def _msgfd(
    signal: np.ndarray,
    active: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
  """Computes MSGFD in batches grouped by active trajectory length."""
  lengths = np.sum(active, axis=0)
  result = np.zeros(signal.shape[1], dtype=np.float64)
  for length in np.unique(lengths):
    indices = np.flatnonzero(lengths == length)
    effective_window = min(window_length, int(length))
    if effective_window % 2 == 0:
      effective_window -= 1
    if effective_window < 1:
      continue
    effective_polyorder = min(polyorder, effective_window - 1)
    values = signal[:length, indices]
    filtered = scipy_signal.savgol_filter(
        values,
        window_length=effective_window,
        polyorder=effective_polyorder,
        axis=0,
        mode="interp",
    )
    result[indices] = np.mean(np.abs(values - filtered), axis=(0, 2))
  return result


def compute_metrics(
    signals: Mapping[str, np.ndarray],
    metrics: Sequence[str],
    *,
    savgol_window_length: int,
    savgol_polyorder: int,
) -> dict[str, np.ndarray]:
  """Computes selected metrics for all archived rollouts."""
  active = np.asarray(signals["active"], dtype=bool)
  joint_velocity = np.asarray(signals["qvel"])[..., 6:]
  unknown = sorted(set(metrics) - set(DEFAULT_METRICS))
  if unknown:
    raise ValueError("Unknown metrics: " + ", ".join(unknown))
  result = {}
  if JOINT_VELOCITY_MSSD in metrics:
    result[JOINT_VELOCITY_MSSD] = _mssd(joint_velocity, active)
  if JOINT_VELOCITY_MSGFD in metrics:
    result[JOINT_VELOCITY_MSGFD] = _msgfd(
        joint_velocity,
        active,
        savgol_window_length,
        savgol_polyorder,
    )
  return result


def _read_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as file:
    return list(csv.DictReader(file))


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  fieldnames = sorted(set().union(*(row.keys() for row in rows)))
  with temporary.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def _summary_statistics(
    rows: Sequence[Mapping[str, str]], metrics: Sequence[str]
) -> dict[str, float]:
  result = {}
  for metric in metrics:
    values = np.asarray([float(row[metric]) for row in rows])
    for suffix, reducer in (
        ("mean", np.mean),
        ("std", np.std),
        ("median", np.median),
        ("min", np.min),
        ("max", np.max),
    ):
      result[f"{metric}/{suffix}"] = float(reducer(values))
  return result


def update_evaluation(
    evaluation_dir: Path, metrics: Sequence[str]
) -> None:
  """Updates one evaluation's CSV/JSON reports from its trajectory archive."""
  summary_path = evaluation_dir / "summary.json"
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  metadata = summary["metadata"]
  scenario_names = list(summary["scenarios"])
  all_rows = _read_rows(evaluation_dir / "rollouts.csv")
  row_offset = 0
  for scenario_name in scenario_names:
    scenario_dir = evaluation_dir / _safe_name(scenario_name)
    archive = scenario_dir / "signals.npz"
    if not archive.is_file():
      raise FileNotFoundError(
          f"Trajectory archive not found: {archive}. Re-evaluate once with "
          "--save_signals."
      )
    with np.load(archive) as signals:
      computed = compute_metrics(
          signals,
          metrics,
          savgol_window_length=int(metadata["savgol_window_length"]),
          savgol_polyorder=int(metadata["savgol_polyorder"]),
      )
    rollout_count = len(next(iter(computed.values())))
    scenario_rows = all_rows[row_offset:row_offset + rollout_count]
    if len(scenario_rows) != rollout_count:
      raise ValueError(
          f"Rollout table and archive disagree in {evaluation_dir}."
      )
    for metric, values in computed.items():
      for row, value in zip(scenario_rows, values):
        row[metric] = repr(float(value))
    statistics = _summary_statistics(scenario_rows, metrics)
    summary["scenarios"][scenario_name].update(statistics)
    scenario_summary = scenario_dir / "summary.json"
    if scenario_summary.is_file():
      value = json.loads(scenario_summary.read_text(encoding="utf-8"))
      value.update(statistics)
      scenario_summary.write_text(
          json.dumps(value, indent=2, sort_keys=True) + "\n",
          encoding="utf-8",
      )
    row_offset += rollout_count
  if row_offset != len(all_rows):
    raise ValueError(f"Unmatched rollout rows in {evaluation_dir}.")
  summary["overall"].update(_summary_statistics(all_rows, metrics))
  history = summary["metadata"].setdefault("postprocessed_metrics", [])
  for metric in metrics:
    if metric not in history:
      history.append(metric)
  _write_rows(evaluation_dir / "rollouts.csv", all_rows)
  summary_path.write_text(
      json.dumps(summary, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary_csv = _read_rows(evaluation_dir / "summary.csv")
  if summary_csv:
    summary_csv[0].update(summary["overall"])
    for scenario_name in scenario_names:
      summary_csv[0].update({
          f"scenario/{scenario_name}/{key}": value
          for key, value in summary["scenarios"][scenario_name].items()
      })
    _write_rows(evaluation_dir / "summary.csv", summary_csv)


def _evaluation_dirs(
    manifest: Path, evaluation_root: Path
) -> list[Path]:
  value = json.loads(manifest.read_text(encoding="utf-8"))
  return [
      evaluation_root
      / "raw_torque"
      / str(run["run_name"])
      / str(run["checkpoint"])
      for run in value["runs"]
  ]


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--evaluation-root", type=Path, required=True)
  parser.add_argument(
      "--metric",
      action="append",
      choices=(*METRIC_ALIASES, *DEFAULT_METRICS),
      dest="metrics",
      help="Metric to compute; repeat as needed. Defaults to all available.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  metrics = tuple(
      METRIC_ALIASES.get(metric, metric)
      for metric in (args.metrics or DEFAULT_METRICS)
  )
  directories = _evaluation_dirs(args.manifest, args.evaluation_root)
  for index, directory in enumerate(directories, start=1):
    print(f"[{index}/{len(directories)}] {directory}", flush=True)
    update_evaluation(directory, metrics)
  print(
      f"Updated {len(directories)} evaluations with {len(metrics)} metrics."
  )


if __name__ == "__main__":
  main()
