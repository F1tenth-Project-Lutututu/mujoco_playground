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
"""Plot reward-versus-cost Pareto fronts from random-task policy evaluations.

Example:

  python -m learning.plot_policy_pareto Go1JoystickFlatTerrain

Each point pools all random-task rollouts and seeds for one method/penalty
scale.  Reward is maximized and every configured y-axis metric is minimized.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from learning import pareto_policy_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = "Go1JoystickFlatTerrain"
DEFAULT_X_METRIC = "eval_reward_means/total_without_regularization"
MIN_X = 31.0
AGGREGATE_CACHE_NAME = "pareto_aggregates.csv"
AGGREGATE_CACHE_MANIFEST_NAME = "pareto_aggregates_cache.json"
AGGREGATE_CACHE_VERSION = 2
DEFAULT_Y_METRICS = (
    "smoothness/torque/mssd_mean_squared_second_difference_per_dof",
    "smoothness/torque/msgfd_mean_absolute_savgol_filter_deviation_per_dof",
    "torque_spectrum/eval/total_energy_per_step",
    "tracking/absolute_mechanical_energy",
)
METHOD_LABELS = {
    "baseline": "Action rate",
    "torque_rate": "Torque rate",
    "high_pass": "High-pass torque",
}
METHOD_COLORS = {
    "baseline": "#4C78A8",
    "torque_rate": "#F58518",
    "high_pass": "#54A24B",
}


@dataclass(frozen=True)
class Point:
  method: str
  scale: float
  scale_tag: str
  x: float
  y: float
  sample_count: int
  seed_count: int
  expected_seed_count: int
  missing_seeds: str


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Returns nondominated points when x is maximized and y is minimized."""
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  result = np.ones(x.shape, dtype=bool)
  for index in range(len(x)):
    dominates = (
        (x >= x[index])
        & (y <= y[index])
        & ((x > x[index]) | (y < y[index]))
    )
    dominates[index] = False
    result[index] = not np.any(dominates)
  return result


def _float(row: dict[str, str], metric: str) -> float:
  try:
    value = float(row[metric])
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError(f"Missing or invalid metric {metric!r}") from error
  if not math.isfinite(value):
    raise ValueError(f"Metric {metric!r} is not finite: {value}")
  return value


def _manifest_runs(path: Path) -> list[dict]:
  value = json.loads(path.read_text(encoding="utf-8"))
  runs = value.get("runs")
  if not isinstance(runs, list):
    raise ValueError(f"Invalid Pareto manifest: {path}")
  coverage = {
      (item["method"], item["scale_tag"]): item
      for item in value.get("seed_coverage", [])
  }
  enriched = []
  for run in runs:
    item = coverage.get((run["method"], run["scale_tag"]), {})
    enriched.append({
        **run,
        "_expected_seeds": item.get("expected_seeds", [run.get("seed")]),
    })
  return enriched


def _report_paths(
    manifest: Path, evaluation_root: Path
) -> list[tuple[dict, Path]]:
  reports = []
  for run in _manifest_runs(manifest):
    report = (
        evaluation_root
        / "raw_torque"
        / str(run["run_name"])
        / str(run["checkpoint"])
        / "rollouts.csv"
    )
    if not report.is_file():
      raise FileNotFoundError(f"Evaluation report not found: {report}")
    reports.append((run, report))
  return reports


def _input_signature(
    manifest: Path, reports: Sequence[tuple[dict, Path]]
) -> dict:
  """Returns a fast fingerprint that changes with any selected input file."""
  manifest_stat = manifest.stat()
  return {
      "version": AGGREGATE_CACHE_VERSION,
      "manifest": {
          "path": str(manifest.resolve()),
          "size": manifest_stat.st_size,
          "mtime_ns": manifest_stat.st_mtime_ns,
      },
      "reports": [
          {
              "path": str(report.resolve()),
              "size": report.stat().st_size,
              "mtime_ns": report.stat().st_mtime_ns,
          }
          for _, report in reports
      ],
  }


def _read_aggregate_cache(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as file:
    return list(csv.DictReader(file))


def _write_aggregate_cache(
    path: Path, rows: Sequence[dict[str, object]]
) -> None:
  if not rows:
    raise ValueError("Cannot cache an empty aggregate table.")
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  fieldnames = (
      "method",
      "scale",
      "scale_tag",
      "sample_count",
      *sorted(set().union(*(row.keys() for row in rows)) - {
          "method",
          "scale",
          "scale_tag",
          "sample_count",
      }),
  )
  with temporary.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def _build_aggregates(
    reports: Sequence[tuple[dict, Path]]
) -> list[dict[str, object]]:
  """Reads every rollout once and averages every numeric metric by sweep point."""
  grouped: dict[
      tuple[str, str], dict[str, object]
  ] = {}
  for run, report in reports:
    key = (str(run["method"]), str(run["scale_tag"]))
    group = grouped.setdefault(
        key,
        {
            "scale": float(run["scale"]),
            "sample_count": 0,
            "seeds": set(),
            "expected_seeds": set(run.get("_expected_seeds", [])),
            "sums": {},
            "counts": {},
        },
    )
    group["expected_seeds"].update(run.get("_expected_seeds", []))
    if run.get("seed") is not None:
      group["seeds"].add(int(run["seed"]))
    with report.open(newline="", encoding="utf-8") as file:
      for row in csv.DictReader(file):
        group["sample_count"] = int(group["sample_count"]) + 1
        sums = group["sums"]
        counts = group["counts"]
        assert isinstance(sums, dict)
        assert isinstance(counts, dict)
        for metric, raw_value in row.items():
          try:
            value = float(raw_value)
          except (TypeError, ValueError):
            continue
          if not math.isfinite(value):
            continue
          sums[metric] = sums.get(metric, 0.0) + value
          counts[metric] = counts.get(metric, 0) + 1

  rows = []
  for (method, scale_tag), group in sorted(grouped.items()):
    sums = group["sums"]
    counts = group["counts"]
    assert isinstance(sums, dict)
    assert isinstance(counts, dict)
    seeds = sorted(group["seeds"])
    expected_seeds = sorted(group["expected_seeds"])
    missing_seeds = sorted(set(expected_seeds) - set(seeds))
    rows.append({
        "method": method,
        "scale": group["scale"],
        "scale_tag": scale_tag,
        "sample_count": group["sample_count"],
        "seed_count": len(seeds),
        "expected_seed_count": len(expected_seeds),
        "missing_seeds": ",".join(str(seed) for seed in missing_seeds),
        "all_seeds_considered": not missing_seeds,
        **{
            metric: total / counts[metric]
            for metric, total in sums.items()
        },
    })
  return rows


def load_aggregates(
    manifest: Path, evaluation_root: Path
) -> list[dict[str, str]]:
  """Loads cached rollout aggregates or rebuilds them when inputs changed."""
  reports = _report_paths(manifest, evaluation_root)
  signature = _input_signature(manifest, reports)
  cache = evaluation_root / AGGREGATE_CACHE_NAME
  cache_manifest = evaluation_root / AGGREGATE_CACHE_MANIFEST_NAME
  if cache.is_file() and cache_manifest.is_file():
    try:
      cached_signature = json.loads(
          cache_manifest.read_text(encoding="utf-8")
      )
    except (OSError, json.JSONDecodeError):
      cached_signature = None
    if cached_signature == signature:
      print(f"Using unchanged rollout aggregate cache: {cache}")
      return _read_aggregate_cache(cache)

  print(f"Rebuilding rollout aggregate cache from {len(reports)} reports.")
  rows = _build_aggregates(reports)
  _write_aggregate_cache(cache, rows)
  temporary_manifest = cache_manifest.with_suffix(
      cache_manifest.suffix + ".tmp"
  )
  temporary_manifest.write_text(
      json.dumps(signature, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  temporary_manifest.replace(cache_manifest)
  return _read_aggregate_cache(cache)


def _points_from_aggregates(
    aggregates: Sequence[dict[str, str]],
    x_metric: str,
    y_metric: str,
) -> list[Point]:
  points = []
  for row in aggregates:
    points.append(
        Point(
            method=row["method"],
            scale=float(row["scale"]),
            scale_tag=row["scale_tag"],
            x=_float(row, x_metric),
            y=_float(row, y_metric),
            sample_count=int(row["sample_count"]),
            seed_count=int(row.get("seed_count", 0)),
            expected_seed_count=int(row.get("expected_seed_count", 0)),
            missing_seeds=row.get("missing_seeds", ""),
        )
    )
  return sorted(points, key=lambda point: (point.method, point.scale))


def load_points(
    manifest: Path,
    evaluation_root: Path,
    x_metric: str,
    y_metric: str,
) -> list[Point]:
  """Returns points from the persistent all-metric rollout aggregate cache."""
  return _points_from_aggregates(
      load_aggregates(manifest, evaluation_root), x_metric, y_metric
  )


def plot(
    manifest: Path,
    evaluation_root: Path,
    output: Path,
    x_metric: str,
    y_metrics: Sequence[str],
    xlim: tuple[float, float | None] | None = None,
) -> Path:
  columns = 2
  rows = math.ceil(len(y_metrics) / columns)
  figure, axes = plt.subplots(
      rows, columns, figsize=(7.0 * columns, 5.2 * rows), squeeze=False
  )
  aggregates = load_aggregates(manifest, evaluation_root)
  csv_rows = []
  for axis, y_metric in zip(axes.flat, y_metrics):
    points = list(
        _points_from_aggregates(aggregates, x_metric, y_metric)
    )
    if xlim is None:
      points = [point for point in points if point.x >= MIN_X]
    else:
      points = [
          point
          for point in points
          if point.x >= xlim[0]
          and (xlim[1] is None or point.x <= xlim[1])
      ]
    if not points:
      selected_range = (
          f">= {MIN_X:g}"
          if xlim is None
          else (
              f">= {xlim[0]:g}"
              if xlim[1] is None
              else f"in [{xlim[0]:g}, {xlim[1]:g}]"
          )
      )
      raise ValueError(
          f"No points for {y_metric!r} have {x_metric!r} {selected_range}."
      )
    for method in METHOD_LABELS:
      method_points = [point for point in points if point.method == method]
      if not method_points:
        continue
      x = np.asarray([point.x for point in method_points])
      y = np.asarray([point.y for point in method_points])
      front = pareto_mask(x, y)
      color = METHOD_COLORS[method]
      axis.scatter(
          x, y, color=color, s=54, alpha=0.85, label=METHOD_LABELS[method]
      )
      front_order = np.argsort(x[front])
      axis.plot(
          x[front][front_order],
          y[front][front_order],
          color=color,
          linewidth=2.2,
          marker="o",
      )
      for point, is_front in zip(method_points, front):
        coverage_suffix = (
            ""
            if point.seed_count == point.expected_seed_count
            else f"\n{point.seed_count}/{point.expected_seed_count} seeds"
        )
        axis.annotate(
            f"{point.scale:g}{coverage_suffix}",
            (point.x, point.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
        csv_rows.append({
            "method": method,
            "scale": point.scale,
            "scale_tag": point.scale_tag,
            "x_metric": x_metric,
            "x": point.x,
            "y_metric": y_metric,
            "y": point.y,
            "pareto": bool(is_front),
            "sample_count": point.sample_count,
            "seed_count": point.seed_count,
            "expected_seed_count": point.expected_seed_count,
            "missing_seeds": point.missing_seeds,
            "all_seeds_considered": (
                point.seed_count == point.expected_seed_count
            ),
        })
    axis.set_xlabel(x_metric)
    axis.set_ylabel(y_metric)
    if xlim is not None:
      axis.set_xlim(left=xlim[0], right=xlim[1])
    axis.grid(alpha=0.25)
  for axis in axes.flat[len(y_metrics):]:
    axis.set_visible(False)
  handles, labels = axes.flat[0].get_legend_handles_labels()
  figure.legend(handles, labels, loc="upper center", ncol=len(labels))
  figure.suptitle("Penalty-scale Pareto fronts", y=0.995)
  figure.tight_layout(rect=(0, 0, 1, 0.96))
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=180, bbox_inches="tight")
  plt.close(figure)

  csv_output = output.with_suffix(".csv")
  with csv_output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=tuple(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
  return output


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "environment",
      nargs="?",
      help=f"Environment name (default: {DEFAULT_ENVIRONMENT}).",
  )
  parser.add_argument(
      "--environment",
      dest="environment_option",
      help="Legacy/explicit alternative to the positional environment.",
  )
  parser.add_argument(
      "--manifest",
      type=Path,
      help="Defaults to the pipeline manifest for --environment.",
  )
  parser.add_argument(
      "--evaluation-root",
      type=Path,
      help="Defaults to evaluations/pareto/<environment>.",
  )
  parser.add_argument(
      "--cluster",
      action="store_true",
      help=(
          "Read the manifest and reports from "
          "evaluations/pareto_cluster/<environment>."
      ),
  )
  parser.add_argument(
      "--output",
      type=Path,
      help=(
          "Defaults to evaluations/pareto/<environment>/policy_pareto.png."
      ),
  )
  parser.add_argument("--x-metric", default=DEFAULT_X_METRIC)
  parser.add_argument(
      "--xlim",
      nargs="+",
      type=float,
      metavar="LIMIT",
      help=(
          "Set MIN, or MIN MAX, for the included and displayed x range. "
          f"By default points below {MIN_X:g} are excluded."
      ),
  )
  parser.add_argument(
      "--y-metric",
      action="append",
      dest="y_metrics",
      help="Metric to plot; repeat for multiple subplots.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  if (
      args.environment
      and args.environment_option
      and args.environment != args.environment_option
  ):
    raise ValueError(
        "Positional environment and --environment specify different values."
    )
  environment = (
      args.environment or args.environment_option or DEFAULT_ENVIRONMENT
  )
  if args.cluster:
    default_evaluation_root = (
        PROJECT_ROOT / "evaluations" / "pareto_cluster" / environment
    )
    default_manifest = (
        default_evaluation_root / pareto_policy_pipeline.MANIFEST_NAME
    )
  else:
    default_evaluation_root = (
        pareto_policy_pipeline.DEFAULT_OUTPUT_ROOT / environment
    )
    default_manifest = (
        pareto_policy_pipeline.DEFAULT_LOCAL_ROOT
        / environment
        / pareto_policy_pipeline.MANIFEST_NAME
    )
  manifest = args.manifest or default_manifest
  evaluation_root = args.evaluation_root or default_evaluation_root
  if args.xlim is not None and len(args.xlim) not in (1, 2):
    raise ValueError("--xlim accepts either MIN or MIN MAX.")
  xlim = (
      (args.xlim[0], args.xlim[1] if len(args.xlim) == 2 else None)
      if args.xlim is not None
      else None
  )
  if xlim is not None and xlim[1] is not None and xlim[0] >= xlim[1]:
    raise ValueError("--xlim MIN must be smaller than MAX.")
  output_path = args.output or (
      evaluation_root / "policy_pareto.png"
  )
  output = plot(
      manifest,
      evaluation_root,
      output_path,
      args.x_metric,
      tuple(args.y_metrics or DEFAULT_Y_METRICS),
      xlim,
  )
  print(f"Pareto plot: {output.resolve()}")
  print(f"Pareto table: {output.with_suffix('.csv').resolve()}")


if __name__ == "__main__":
  main()
