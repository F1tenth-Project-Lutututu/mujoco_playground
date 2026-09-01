"""Create one IQM Pareto figure across selected quadruped environments.

The first invocation pools every rollout for each environment/method/penalty
point and writes ``multi_environment_pareto_iqm.csv``.  That file is an
intentional cache: later invocations only read it.  Delete it explicitly when
the underlying evaluations change and a new aggregate is desired.

Example:
  MPLBACKEND=Agg python -m learning.plot_multi_environment_pareto
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from learning import plot_policy_pareto as pareto
from learning import pareto_policy_pipeline as pipeline


# Edit these two lists to select the paper panels.  Columns follow this order.
ENVIRONMENTS = (
    "BarkourJoystick",
    "Go1JoystickFlatTerrain",
    "Go1JoystickFlatTerrain25",
    "Go1JoystickRoughTerrain",
    "SilverBadgerJoystickFlatTerrain",
    "SilverBadgerJoystickRoughTerrain",
    "SpotFlatTerrainJoystick",
    "SpotJoystickGaitTracking",
)
SMOOTHNESS_METRICS = (
    (
        "smoothness/torque/mssd_mean_squared_second_difference_per_dof",
        "MSSD",
    ),
    #("smoothness/torque/total_variation", r"Torque total variation ($\times 10^4$)"),
    (
        "smoothness/torque/msgfd_w5_p2_"
        "mean_absolute_savgol_filter_deviation_per_dof",
        "Savitzky–Golay deviation \n (window 5, order 2)",
    ),
)
METHODS = (
    ("baseline", "Action rate"),
    ("torque_rate", "Torque rate"),
    ("action_smoothness", "Action smoothness"),
    ("high_pass", "TFR (ours)"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluations" / "pareto_cluster"
RESULTS_ROOT = PROJECT_ROOT / "evaluations" / "pareto_results"
CACHE_PATH = RESULTS_ROOT / "multi_environment_pareto_iqm.csv"
OUTPUT_PATH = RESULTS_ROOT / "multi_environment_pareto_iqm.png"
REWARD_METRIC = pareto.DEFAULT_X_METRIC
LOG_Y_AXIS = True
X_EXPONENTIAL_STRENGTH = 2.0
METRIC_DISPLAY_SCALES = {
    "smoothness/torque/total_variation": 1e4,
}


def _iqm(values: Iterable[float]) -> float:
  """Returns the interquartile mean, matching the single-environment plot."""
  values = np.sort(np.asarray(list(values), dtype=float))
  if not len(values):
    raise ValueError("Cannot compute an IQM from no values.")
  trim = int(0.25 * len(values))
  middle = values[trim : len(values) - trim] if trim else values
  return float(np.mean(middle))


def _cache_rows() -> list[dict[str, str]]:
  with CACHE_PATH.open(newline="", encoding="utf-8") as file:
    return list(csv.DictReader(file))


def _evaluated_reports(environment: str) -> list[tuple[dict, Path]]:
  """Selects the latest evaluated run for each method/scale/seed point.

  Manifests are submission snapshots and can predate a newly evaluated sweep.
  The raw-report directories are the durable source of truth for this plot.
  """
  root = EVALUATION_ROOT / environment / "raw_torque"
  if not root.is_dir():
    raise FileNotFoundError(
        f"Missing evaluated reports for {environment}: {root}. Fetch its "
        "results before building the multi-environment cache."
    )
  selected = pipeline.select_runs(
      [path.name for path in root.iterdir() if path.is_dir()]
  )
  reports = []
  for policy_run in selected:
    candidates = sorted(
        path for path in (root / policy_run.run_name).glob("*/rollouts.csv")
        if path.parent.name.isdigit()
    )
    if len(candidates) != 1:
      raise FileNotFoundError(
          f"Expected one evaluated report for {environment}/"
          f"{policy_run.run_name}, found {len(candidates)}."
      )
    run = asdict(policy_run)
    run["checkpoint"] = candidates[0].parent.name
    reports.append((run, candidates[0]))
  return reports


def _write_cache() -> None:
  """Pools the selected metrics for all requested environment sweep points."""
  metric_names = (REWARD_METRIC, *(metric for metric, _ in SMOOTHNESS_METRICS))
  values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
  sample_counts: dict[tuple[str, str, str], int] = defaultdict(int)
  seed_sets: dict[tuple[str, str, str], set[int]] = defaultdict(set)
  scales: dict[tuple[str, str, str], float] = {}

  for environment in ENVIRONMENTS:
    print(f"Aggregating IQM reports for {environment}...", flush=True)
    for run, report in _evaluated_reports(environment):
      method = str(run["method"])
      if method not in dict(METHODS):
        continue
      point = (environment, method, str(run["scale_tag"]))
      scales[point] = float(run["scale"])
      seed_sets[point].add(int(run["seed"]))
      with report.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
          sample_counts[point] += 1
          for metric in metric_names:
            try:
              value = float(row[metric])
            except (KeyError, TypeError, ValueError):
              continue
            if np.isfinite(value):
              values[(*point, metric)].append(value)

  missing = sorted(
      (*point, metric)
      for point in sample_counts
      for metric in metric_names
      if not values[(*point, metric)]
  )
  if missing:
    raise ValueError(f"Empty metric values while aggregating: {missing[:3]}")
  CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
  rows = []
  for environment, method, scale_tag in sorted(sample_counts):
    row = {
        "environment": environment,
        "method": method,
        "scale_tag": scale_tag,
        "scale": "",
        "sample_count": sample_counts[(environment, method, scale_tag)],
        "seed_count": len(seed_sets[(environment, method, scale_tag)]),
    }
    for metric in metric_names:
      point_values = values[(environment, method, scale_tag, metric)]
      row[metric] = _iqm(point_values)
    rows.append(row)

  for row in rows:
    row["scale"] = scales[(row["environment"], row["method"], row["scale_tag"])]

  fieldnames = (
      "environment", "method", "scale", "scale_tag", "sample_count",
      "seed_count", *metric_names,
  )
  with CACHE_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  print(f"Wrote IQM aggregate cache: {CACHE_PATH}")


def _load_or_build_cache() -> list[dict[str, str]]:
  if CACHE_PATH.is_file():
    print(f"Using IQM aggregate cache: {CACHE_PATH}")
  else:
    _write_cache()
  return _cache_rows()


def _plot(rows: Sequence[dict[str, str]]) -> None:
  figure, axes = plt.subplots(
      len(SMOOTHNESS_METRICS),
      len(ENVIRONMENTS),
      figsize=(3.35 * len(ENVIRONMENTS), 3.0 * len(SMOOTHNESS_METRICS)),
      squeeze=False,
      sharex=False,
      sharey=False,
  )
  method_labels = dict(METHODS)
  legend_handles = {}
  for column, environment in enumerate(ENVIRONMENTS):
    environment_rows = [row for row in rows if row["environment"] == environment]
    for row_index, (metric, metric_label) in enumerate(SMOOTHNESS_METRICS):
      axis = axes[row_index, column]
      visible_y = []
      xlim = pareto._configured_xlim(environment, pareto.DEFAULT_XLIM_CONFIG)
      for method, label in METHODS:
        method_rows = [
            row for row in environment_rows if row["method"] == method
        ]
        if not method_rows:
          continue
        x = np.asarray([float(row[REWARD_METRIC]) for row in method_rows])
        y = np.asarray([float(row[metric]) for row in method_rows])
        y /= METRIC_DISPLAY_SCALES.get(metric, 1.0)
        in_view = np.ones(x.shape, dtype=bool)
        if xlim is not None:
          in_view &= x >= xlim[0]
          if xlim[1] is not None:
            in_view &= x <= xlim[1]
        x, y = x[in_view], y[in_view]
        if not len(x):
          continue
        front = pareto.pareto_mask(x, y)
        visible_y.extend(y[front])
        color = pareto._method_color(method)
        scatter = axis.scatter(x[front], y[front], color=color, s=34, label=label)
        order = np.argsort(x[front])
        axis.plot(x[front][order], y[front][order], color=color, linewidth=1.8)
        legend_handles.setdefault(method, scatter)
      if row_index == 0:
        axis.set_title(environment.replace("Joystick", "\nJoystick"), fontsize=10)
      if column == 0:
        axis.set_ylabel(metric_label)
      if row_index == len(SMOOTHNESS_METRICS) - 1:
        axis.set_xlabel("Task reward")
      if LOG_Y_AXIS:
        axis.set_yscale("log")
        if visible_y:
          log_y = np.log(np.asarray(visible_y, dtype=float))
          lower, upper = float(np.min(log_y)), float(np.max(log_y))
          padding = max(0.04 * (upper - lower), 0.04)
          lower, upper = np.exp(lower - padding), np.exp(upper + padding)
          axis.set_ylim(lower, upper)
          axis.yaxis.set_major_locator(
              ticker.FixedLocator(np.geomspace(lower, upper, num=3))
          )
          axis.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2g}"))
          axis.yaxis.set_minor_locator(ticker.NullLocator())
          axis.tick_params(axis="y", labelsize=7)
      environment_x = np.asarray(
          [float(row[REWARD_METRIC]) for row in environment_rows], dtype=float
      )
      if (
          X_EXPONENTIAL_STRENGTH
          and len(environment_x)
          and np.ptp(environment_x) > 0
      ):
        axis.set_xscale(
            "function",
            functions=pareto._exponential_axis_functions(
                float(np.min(environment_x)),
                float(np.max(environment_x)),
                X_EXPONENTIAL_STRENGTH,
            ),
        )
      if xlim is not None:
        axis.set_xlim(left=xlim[0], right=xlim[1])
      axis.grid(alpha=0.25)

  figure.legend(
      [legend_handles[method] for method, _ in METHODS if method in legend_handles],
      [method_labels[method] for method, _ in METHODS if method in legend_handles],
      loc="lower center",
      ncol=len(METHODS),
      frameon=False,
      bbox_to_anchor=(0.5, -0.01),
  )
  figure.subplots_adjust(bottom=0.15, wspace=0.24, hspace=0.20)
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
  plt.close(figure)
  print(f"Pareto figure: {OUTPUT_PATH}")


def main() -> None:
  _plot(_load_or_build_cache())


if __name__ == "__main__":
  main()
