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
import subprocess
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
    #"SpotFlatTerrainJoystick",
    "SpotJoystickGaitTracking",
)
SMOOTHNESS_METRICS = (
    (
        "smoothness/torque/mssd_mean_squared_second_difference_per_dof",
        "Torque MSSD",
    ),
    #("smoothness/torque/total_variation", r"Torque total variation ($\times 10^4$)"),
    (
        "smoothness/torque/msgfd_w5_p2_"
        "mean_absolute_savgol_filter_deviation_per_dof",
        "Torque Savitzky–Golay deviation",
        #"Torque Savitzky–Golay deviation \n (window 5, order 2)",
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
PDF_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".pdf")
IMPROVEMENT_TABLE_PATH = RESULTS_ROOT / "multi_environment_pareto_improvements.csv"
REWARD_METRIC = pareto.DEFAULT_X_METRIC
LOG_Y_AXIS = True
X_EXPONENTIAL_STRENGTH = 1.0
# IEEE two-column text width is approximately 7.16 inches.  Keeping the
# Matplotlib canvas at its final publication size prevents LaTeX from scaling
# otherwise reasonable fonts down to unreadable sizes.
FIGURE_WIDTH_INCHES = 13.
PANEL_HEIGHT_INCHES = 2.2
TITLE_FONT_SIZE = 10.0
LABEL_FONT_SIZE = 10.0
TICK_FONT_SIZE = 8.0
LEGEND_FONT_SIZE = 10.0
METRIC_DISPLAY_SCALES = {
    "smoothness/torque/total_variation": 1e4,
}


def _plain_tick_label(value: float, _position: float | None = None) -> str:
  """Formats plot ticks without scientific notation or redundant zeros."""
  return np.format_float_positional(
      value, precision=3, unique=False, fractional=False, trim="-"
  )


def _plain_y_tick_label(value: float, _position: float | None = None) -> str:
  """Formats y ticks with two significant digits and no exponent notation."""
  return np.format_float_positional(
      value, precision=2, unique=False, fractional=False, trim="-"
  )


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


def _crop_pdf(path: Path) -> None:
  """Crops a generated PDF to its visible page content using pdfcrop."""
  cropped_path = path.with_name(f"{path.stem}.cropped{path.suffix}")
  cropped_path.unlink(missing_ok=True)
  try:
    subprocess.run(
        ["pdfcrop", "--margins", "0", str(path), str(cropped_path)],
        check=True,
    )
    cropped_path.replace(path)
  finally:
    cropped_path.unlink(missing_ok=True)


def _front_xy(
    rows: Sequence[dict[str, str]], metric: str, xlim: tuple[float, float | None] | None
) -> tuple[np.ndarray, np.ndarray]:
  """Returns one method's visible Pareto front, sorted by reward.

  Duplicate reward values are reduced to their least-cost point so that the
  resulting curve is suitable for interpolation.
  """
  x = np.asarray([float(row[REWARD_METRIC]) for row in rows], dtype=float)
  y = np.asarray([float(row[metric]) for row in rows], dtype=float)
  if xlim is not None:
    visible = x >= xlim[0]
    if xlim[1] is not None:
      visible &= x <= xlim[1]
    x, y = x[visible], y[visible]
  if not len(x):
    return x, y
  front = pareto.pareto_mask(x, y)
  x, y = x[front], y[front]
  order = np.argsort(x)
  x, y = x[order], y[order]
  unique_x, first = np.unique(x, return_index=True)
  return unique_x, np.minimum.reduceat(y, first)


def _mean_smoothness_improvement(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
) -> tuple[float, float, float] | None:
  """Returns candidate's reward-weighted percentage reduction over overlap."""
  if len(reference_x) < 2 or len(candidate_x) < 2:
    return None
  lower = max(float(np.min(reference_x)), float(np.min(candidate_x)))
  upper = min(float(np.max(reference_x)), float(np.max(candidate_x)))
  if lower >= upper:
    return None
  grid = np.unique(np.concatenate((
      np.asarray([lower, upper]),
      reference_x[(reference_x > lower) & (reference_x < upper)],
      candidate_x[(candidate_x > lower) & (candidate_x < upper)],
  )))
  reference = np.interp(grid, reference_x, reference_y)
  candidate = np.interp(grid, candidate_x, candidate_y)
  if np.any(reference <= 0):
    return None
  improvement = 100.0 * (reference - candidate) / reference
  mean_improvement = float(np.trapezoid(improvement, grid) / (upper - lower))
  return mean_improvement, lower, upper


def _write_improvement_table(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
  """Writes TFR's smoothness improvement relative to every configured baseline."""
  ours = "high_pass"
  table_rows = []
  for environment in ENVIRONMENTS:
    environment_rows = [row for row in rows if row["environment"] == environment]
    xlim = pareto._configured_xlim(environment, pareto.DEFAULT_XLIM_CONFIG)
    ours_rows = [row for row in environment_rows if row["method"] == ours]
    for metric, metric_label in SMOOTHNESS_METRICS:
      ours_x, ours_y = _front_xy(ours_rows, metric, xlim)
      for baseline, baseline_label in METHODS:
        if baseline == ours:
          continue
        baseline_rows = [
            row for row in environment_rows if row["method"] == baseline
        ]
        baseline_x, baseline_y = _front_xy(baseline_rows, metric, xlim)
        result = _mean_smoothness_improvement(
            baseline_x, baseline_y, ours_x, ours_y
        )
        row = {
            "environment": environment,
            "smoothness_metric": metric,
            "smoothness_label": metric_label,
            "reference_method": baseline,
            "reference_label": baseline_label,
            "candidate_method": ours,
            "candidate_label": dict(METHODS)[ours],
            "mean_smoothness_improvement_percent": "",
            "overlap_reward_min": "",
            "overlap_reward_max": "",
            "status": "insufficient_or_nonoverlapping_fronts",
        }
        if result is not None:
          improvement, lower, upper = result
          row.update({
              "mean_smoothness_improvement_percent": improvement,
              "overlap_reward_min": lower,
              "overlap_reward_max": upper,
              "status": "ok",
          })
        table_rows.append(row)
  IMPROVEMENT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
  with IMPROVEMENT_TABLE_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=tuple(table_rows[0]))
    writer.writeheader()
    writer.writerows(table_rows)
  print(f"Pareto smoothness-improvement table: {IMPROVEMENT_TABLE_PATH}")
  return table_rows


def _print_improvement_table(rows: Sequence[dict[str, object]]) -> None:
  """Prints the saved comparison table in a concise, readable layout."""
  headers = ("Environment", "Metric", "Reference", "TFR improvement", "Overlap")
  body = []
  for row in rows:
    if row["status"] == "ok":
      improvement = f"{float(row['mean_smoothness_improvement_percent']):+.2f}%"
      overlap = (
          f"{float(row['overlap_reward_min']):.2f}–"
          f"{float(row['overlap_reward_max']):.2f}"
      )
    else:
      improvement, overlap = "n/a", "n/a"
    body.append((
        str(row["environment"]),
        " ".join(str(row["smoothness_label"]).split()),
        str(row["reference_label"]),
        improvement,
        overlap,
    ))
  widths = [
      max(len(header), *(len(row[index]) for row in body))
      for index, header in enumerate(headers)
  ]
  separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
  def format_row(row: Sequence[str]) -> str:
    return "|" + "|".join(
        f" {value:<{width}} " for value, width in zip(row, widths)
    ) + "|"
  print("\nTFR smoothness improvement over overlapping Pareto-front ranges")
  print(separator)
  print(format_row(headers))
  print(separator)
  for row in body:
    print(format_row(row))
  print(separator)


def _print_average_improvements(rows: Sequence[dict[str, object]]) -> None:
  """Prints unweighted mean gains across the environments in the table."""
  grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
  for row in rows:
    if row["status"] != "ok":
      continue
    key = (str(row["smoothness_label"]), str(row["reference_label"]))
    grouped[key].append(float(row["mean_smoothness_improvement_percent"]))
  headers = ("Metric", "Reference", "Mean TFR improvement", "Environments")
  body = [
      (
          " ".join(metric.split()),
          reference,
          f"{np.mean(improvements):+.2f}%",
          str(len(improvements)),
      )
      for (metric, reference), improvements in grouped.items()
  ]
  widths = [
      max(len(header), *(len(row[index]) for row in body))
      for index, header in enumerate(headers)
  ]
  separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

  def format_row(row: Sequence[str]) -> str:
    return "|" + "|".join(
        f" {value:<{width}} " for value, width in zip(row, widths)
    ) + "|"

  print("\nMean TFR smoothness improvement across environments (unweighted)")
  print(separator)
  print(format_row(headers))
  print(separator)
  for row in body:
    print(format_row(row))
  print(separator)


def _plot(rows: Sequence[dict[str, str]]) -> None:
  figure, axes = plt.subplots(
      len(SMOOTHNESS_METRICS),
      len(ENVIRONMENTS),
      figsize=(
          FIGURE_WIDTH_INCHES,
          PANEL_HEIGHT_INCHES * len(SMOOTHNESS_METRICS),
      ),
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
        scatter = axis.scatter(x[front], y[front], color=color, s=10, label=label)
        order = np.argsort(x[front])
        axis.plot(x[front][order], y[front][order], color=color, linewidth=1.2)
        legend_handles.setdefault(method, scatter)
      if row_index == 0:
        axis.set_title(
            environment.replace("Joystick", "\nJoystick"),
            fontsize=TITLE_FONT_SIZE,
        )
      if column == 0:
        axis.set_ylabel(metric_label, fontsize=LABEL_FONT_SIZE)
      #if row_index == len(SMOOTHNESS_METRICS) - 1:
      #  axis.set_xlabel("Task reward", fontsize=LABEL_FONT_SIZE)
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
          axis.yaxis.set_major_formatter(
              ticker.FuncFormatter(_plain_y_tick_label)
          )
          axis.yaxis.set_minor_locator(ticker.NullLocator())
      axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
      axis.tick_params(axis="y", pad=1.0)
      environment_x = np.asarray(
          [float(row[REWARD_METRIC]) for row in environment_rows], dtype=float
      )
      if xlim is not None:
        environment_x = environment_x[environment_x >= xlim[0]]
        if xlim[1] is not None:
          environment_x = environment_x[environment_x <= xlim[1]]
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
      axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
      axis.xaxis.set_major_formatter(ticker.FuncFormatter(_plain_tick_label))
      axis.grid(alpha=0.25)

  figure.legend(
      [legend_handles[method] for method, _ in METHODS if method in legend_handles],
      [method_labels[method] for method, _ in METHODS if method in legend_handles],
      loc="lower center",
      ncol=len(METHODS),
      frameon=False,
      bbox_to_anchor=(0.5, -0.01),
      fontsize=LEGEND_FONT_SIZE,
  )
  figure.subplots_adjust(
      left=0.075,
      right=0.995,
      top=0.90,
      bottom=0.10,
      wspace=0.22,
      hspace=0.18,
  )
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
  figure.savefig(PDF_OUTPUT_PATH, bbox_inches="tight")
  plt.close(figure)
  _crop_pdf(PDF_OUTPUT_PATH)
  print(f"Pareto figures: {OUTPUT_PATH}, {PDF_OUTPUT_PATH}")


def main() -> None:
  rows = _load_or_build_cache()
  improvement_rows = _write_improvement_table(rows)
  _plot(rows)
  _print_improvement_table(improvement_rows)
  _print_average_improvements(improvement_rows)


if __name__ == "__main__":
  main()
