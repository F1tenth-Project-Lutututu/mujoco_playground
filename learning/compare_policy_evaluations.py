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
"""Visually compare reports produced by ``learning/evaluate_policy.py``.

The script creates a multi-panel plot with one panel per metric and grouped
bars for the evaluations. Bar heights are per-scenario rollout means and error
bars show one standard deviation. Configure the uppercase constants below,
then run the script. A second figure shows paired per-task differences from a
reference method, removing between-task variability. A third figure expresses
the paired changes as percentages, oriented so positive values always mean an
improvement. All figures have CSV companions.
"""

import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


# Map plot labels to evaluation directories or summary.json files. A run-level
# directory containing exactly one evaluated checkpoint is also accepted. A
# directory whose name ends in ``-seedN`` selects all sibling seed directories
# for that policy type; the configured seed only serves as the pattern template.
#METHODS = {
#    "baseline": "evaluations/260716-go1-baseline-ar1em2-seed0",
#    "baseline-ar3em2": "evaluations/260716-go1-baseline-ar3em2-seed0",
#    "baseline-ar5em2": "evaluations/260717-go1-baseline-ar5em2-seed0",
#    "baseline-ar1em1": "evaluations/260717-go1-baseline-ar1em1-seed0",
#    "newhf1em6-f5o2m30": "evaluations/260716-go1-newhf1em6-f5o2m30-seed0",
#    "newhf1em5-f5o2m20": "evaluations/260716-go1-newhf1em5-f5o2m20-seed0",
#    "newhf1em5-f5o3m20": "evaluations/260716-go1-newhf1em5-f5o3m20-seed0",
#    "newhf1em5-f5o4m20": "evaluations/260717-go1-newhf1em5-f5o4m20-seed0",
#    "newhf1em5-f5o2m15": "evaluations/260716-go1-newhf1em5-f5o2m15-seed0",
#    "newhf1em4-f5o2m05": "evaluations/260716-go1-newhf1em4-f5o2m05-seed0",
#}
#PAIRED_REFERENCE = "baseline"

#NORMALIZATION_TYPE = "capacity_normalized"
NORMALIZATION_TYPE = "raw_torque"
EVALUATION_DIR = Path("evaluations/Go1JoystickRoughTerrain") / NORMALIZATION_TYPE

METHODS = {
    "baseline-ar2em1": EVALUATION_DIR / "260721-baseline-400M-ar2em1-seed0",
    "newhfstate-f5o1m10": EVALUATION_DIR / "260722-newhfstate1em3-400M-f5o1m10-seed0",
    "newhf-f5o5m25": EVALUATION_DIR / "260718-newhf1em5-400M-f5o5m25-seed0",
    "newhf-f5o4m25": EVALUATION_DIR / "260718-newhf1em5-400M-f5o4m25-seed0",
    "newhf-f5o3m25": EVALUATION_DIR / "260718-newhf1em5-400M-f5o3m25-seed0",
    "newhf-f5o2m25": EVALUATION_DIR / "260721-newhf1em5-400M-f5o2m25-seed0",
}
PAIRED_REFERENCE = "baseline-ar2em1"

METRICS = (
    "eval_reward_means/total_without_regularization",
    "tracking/linear_velocity_vector_rmse",
    "tracking/yaw_rate_rmse",
    "smoothness/torque/mean_squared_delta_l2_per_step",
    "smoothness/torque/mssd_mean_squared_second_difference_per_dof",
    "smoothness/torque/msgfd_mean_absolute_savgol_filter_deviation_per_dof",
    "torque_spectrum/eval/fft_above_5hz_energy_per_step",
    "torque_spectrum/eval/fft_above_15hz_energy_per_step",
    "tracking/absolute_mechanical_energy",
    "tracking/total_absolute_torque_impulse",
    "tracking/orientation_error_rms_degrees",
    "tracking/roll_pitch_rate_rms",
    "tracking/feet_height_error_mean_mm",
    #"episode/fell",
)
OUTPUT = Path("evaluation_comparison.png")
# Set to None to use OUTPUT with a .csv suffix.
CSV_OUTPUT: Path | None = None
TITLE = "Policy evaluation comparison"
PAIRED_OUTPUT = Path("evaluation_comparison_paired.png")
# Set to None to use PAIRED_OUTPUT with a .csv suffix.
PAIRED_CSV_OUTPUT: Path | None = None
PAIRED_TITLE = "Paired per-task differences"
PAIRED_PERCENT_OUTPUT = Path("evaluation_comparison_paired_percent.png")
# Set to None to use PAIRED_PERCENT_OUTPUT with a .csv suffix.
PAIRED_PERCENT_CSV_OUTPUT: Path | None = None
PAIRED_PERCENT_TITLE = "Paired per-task improvement/deterioration"
PAIRED_SHOW_TASK_POINTS = True
PAIRED_POINT_ALPHA = 0.15
# Hide task dots outside the standard Tukey boxplot fences so a few extreme
# task deltas do not squash the boxes. Set to None to display every task.
PAIRED_OUTLIER_IQR_MULTIPLIER: float | None = 1.5
PAIRED_LEGEND_COLUMNS = 4
COLUMNS = 5
FIGURE_WIDTH_INCHES = 19.2
FIGURE_ASPECT_RATIO = 16.0 / 9.0
DPI = 160
SHOW = False

# Direction used to turn relative changes into improvement percentages.
# 1 means larger values are better; -1 means smaller values are better.
METRIC_IMPROVEMENT_DIRECTIONS = {
    "eval_reward_means/total_without_regularization": 1,
    "tracking/linear_velocity_vector_rmse": -1,
    "tracking/yaw_rate_rmse": -1,
    "smoothness/torque/mean_squared_delta_l2_per_step": -1,
    "smoothness/torque/mssd_mean_squared_second_difference_per_dof": -1,
    "smoothness/torque/msgfd_mean_absolute_savgol_filter_deviation_per_dof": -1,
    "torque_spectrum/eval/fft_above_5hz_energy_per_step": -1,
    "torque_spectrum/eval/fft_above_15hz_energy_per_step": -1,
    "tracking/absolute_mechanical_energy": -1,
    "tracking/total_absolute_torque_impulse": -1,
    "tracking/orientation_error_rms_degrees": -1,
    "tracking/roll_pitch_rate_rms": -1,
    "tracking/feet_height_error_mean_mm": -1,
    "episode/fell": -1,
}

PAIRED_METADATA_FIELDS = (
    "evaluation_mode",
    "task_seed",
    "num_random_tasks",
    "episode_length_environment_steps",
    "action_repeat",
    "deterministic",
    "perturbations_disabled",
    "savgol_window_length",
    "savgol_polyorder",
    "feet_height_target_meters",
)


def _summary_path(path: Path) -> Path:
  if not path.is_dir():
    return path
  direct_summary = path / "summary.json"
  if direct_summary.is_file():
    return direct_summary
  checkpoint_summaries = sorted(path.glob("*/summary.json"))
  if len(checkpoint_summaries) == 1:
    return checkpoint_summaries[0]
  if len(checkpoint_summaries) > 1:
    raise ValueError(
        f"{path} contains multiple evaluated checkpoints; configure METHODS "
        "with a specific checkpoint directory."
    )
  return direct_summary


def _load_summary(path: Path) -> dict[str, Any]:
  summary_path = _summary_path(path)
  if not summary_path.is_file():
    raise FileNotFoundError(f"Evaluation summary not found: {summary_path}")
  with summary_path.open(encoding="utf-8") as fp:
    summary = json.load(fp)
  if not isinstance(summary, dict) or not isinstance(
      summary.get("scenarios"), dict
  ):
    raise ValueError(
        f"{summary_path} is not an evaluate_policy.py summary report."
    )
  return summary


def _load_rollouts(path: Path) -> list[dict[str, str]]:
  rollouts_path = _summary_path(path).parent / "rollouts.csv"
  if not rollouts_path.is_file():
    raise FileNotFoundError(f"Evaluation rollouts not found: {rollouts_path}")
  with rollouts_path.open(newline="", encoding="utf-8") as fp:
    return list(csv.DictReader(fp))


def _policy_seed_paths(path: Path) -> list[Path]:
  """Expands a configured seed run to every sibling seed of that policy."""
  seed_path = next(
      (
          candidate
          for candidate in (path, *path.parents)
          if re.fullmatch(r"(.+-seed)\d+", candidate.name)
      ),
      None,
  )
  if seed_path is None:
    return [path]
  match = re.fullmatch(r"(.+-seed)\d+", seed_path.name)
  assert match is not None
  seed_paths = sorted(
      candidate
      for candidate in seed_path.parent.glob(f"{match.group(1)}*")
      if candidate.is_dir()
      and re.fullmatch(rf"{re.escape(match.group(1))}\d+", candidate.name)
  )
  relative_path = path.relative_to(seed_path)
  return [candidate / relative_path for candidate in seed_paths] or [path]


def _policy_seed(path: Path) -> str:
  """Returns the policy seed encoded in a run directory name."""
  for candidate in (path, *path.parents):
    match = re.fullmatch(r".+-seed(\d+)", candidate.name)
    if match is not None:
      return match.group(1)
  return path.name


def _aggregate_rollouts(
    paths: Sequence[Path], metrics: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
  """Pools rollout data across seeds and builds the required summary stats."""
  summaries = [_load_summary(path) for path in paths]
  tables = []
  for path in paths:
    seed = _policy_seed(path)
    for row in _load_rollouts(path):
      row = dict(row)
      if "task" in row:
        row["task"] = f"{seed}:{row['task']}"
      tables.append(row)

  scenarios = list(
      dict.fromkeys(row.get("scenario", "random_tasks") for row in tables)
  )
  aggregate: dict[str, Any] = {
      "metadata": summaries[0].get("metadata", {}),
      "environment_config": summaries[0].get("environment_config", {}),
      "scenarios": {},
  }
  for scenario in scenarios:
    scenario_rows = [
        row for row in tables if row.get("scenario", "random_tasks") == scenario
    ]
    stats = {}
    for metric in metrics:
      values = []
      for row in scenario_rows:
        try:
          value = float(row[metric])
        except (KeyError, TypeError, ValueError):
          continue
        if math.isfinite(value):
          values.append(value)
      if values:
        stats[f"{metric}/mean"] = float(np.mean(values))
        stats[f"{metric}/std"] = float(np.std(values))
    aggregate["scenarios"][scenario] = stats
  return aggregate, tables


def _unique_labels(labels: Sequence[str]) -> list[str]:
  counts: dict[str, int] = {}
  result = []
  for label in labels:
    counts[label] = counts.get(label, 0) + 1
    suffix = f" ({counts[label]})" if counts[label] > 1 else ""
    result.append(f"{label}{suffix}")
  return result


def _labels_with_seed_counts(
    labels: Sequence[str], seed_paths: Sequence[Sequence[Path]]
) -> list[str]:
  """Adds the number of aggregated seeds to each legend label."""
  return [
      f"{label} ({len(paths)} {'seed' if len(paths) == 1 else 'seeds'})"
      for label, paths in zip(labels, seed_paths, strict=True)
  ]


def _metric_stat(
    scenario_summary: Mapping[str, Any], metric: str, stat: str
) -> float:
  value = scenario_summary.get(f"{metric}/{stat}")
  if value is None and stat == "mean":
    value = scenario_summary.get(metric)
  if value is None:
    return math.nan
  try:
    result = float(value)
  except (TypeError, ValueError):
    return math.nan
  return result if math.isfinite(result) else math.nan


def _comparison_rows(
    summaries: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    metrics: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
  scenarios = list(
      dict.fromkeys(
          scenario
          for summary in summaries
          for scenario in summary["scenarios"]
      )
  )
  rows = []
  for label, summary in zip(labels, summaries, strict=True):
    for scenario in scenarios:
      scenario_summary = summary["scenarios"].get(scenario, {})
      for metric in metrics:
        rows.append({
            "evaluation": label,
            "scenario": scenario,
            "metric": metric,
            "mean": _metric_stat(scenario_summary, metric, "mean"),
            "std": _metric_stat(scenario_summary, metric, "std"),
        })
  return scenarios, rows


def _validate_paired_reports(
    summaries: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    reference_label: str,
) -> None:
  if reference_label not in labels:
    raise ValueError(f"PAIRED_REFERENCE {reference_label!r} is not in METHODS.")
  reference_index = labels.index(reference_label)
  reference_metadata = summaries[reference_index].get("metadata", {})
  if reference_metadata.get("evaluation_mode") != "random_tasks":
    raise ValueError("Paired comparison requires random-task evaluations.")
  for label, summary in zip(labels, summaries, strict=True):
    if label == reference_label:
      continue
    metadata = summary.get("metadata", {})
    mismatches = []
    for field in PAIRED_METADATA_FIELDS:
      reference_value = reference_metadata.get(field)
      value = metadata.get(field)
      if field == "feet_height_target_meters":
        reference_value = reference_value or summaries[reference_index].get(
            "environment_config", {}
        ).get("reward_config", {}).get("max_foot_height")
        value = value or summary.get("environment_config", {}).get(
            "reward_config", {}
        ).get("max_foot_height")
      if value != reference_value:
        mismatches.append(field)
    if mismatches:
      raise ValueError(
          f"Evaluation {label!r} is not paired with the reference; mismatched "
          f"metadata: {', '.join(mismatches)}."
      )


def _paired_delta_rows(
    rollout_tables: Sequence[Sequence[Mapping[str, str]]],
    labels: Sequence[str],
    metrics: Sequence[str],
    reference_label: str,
) -> list[dict[str, Any]]:
  """Returns method-minus-reference differences for every matched task."""
  if reference_label not in labels:
    raise ValueError(f"PAIRED_REFERENCE {reference_label!r} is not in METHODS.")

  indexed_tables = []
  for label, table in zip(labels, rollout_tables, strict=True):
    indexed = {}
    for row in table:
      if "task" not in row:
        raise ValueError(
            f"Evaluation {label!r} has no task column; rerun it with "
            "--num_random_tasks."
        )
      key = (row.get("scenario", "random_tasks"), row["task"])
      if key in indexed:
        raise ValueError(f"Evaluation {label!r} contains duplicate task {key}.")
      indexed[key] = row
    indexed_tables.append(indexed)

  reference_index = labels.index(reference_label)
  reference = indexed_tables[reference_index]
  paired_rows = []
  for label, indexed in zip(labels, indexed_tables, strict=True):
    if label == reference_label:
      continue
    if set(indexed) != set(reference):
      missing = len(set(reference) - set(indexed))
      extra = len(set(indexed) - set(reference))
      raise ValueError(
          f"Evaluation {label!r} does not contain the same tasks as "
          f"{reference_label!r} ({missing} missing, {extra} extra)."
      )
    for scenario, task in reference:
      for metric in metrics:
        if metric not in reference[(scenario, task)] or metric not in indexed[
            (scenario, task)
        ]:
          continue
        try:
          reference_value = float(reference[(scenario, task)][metric])
          value = float(indexed[(scenario, task)][metric])
        except (TypeError, ValueError):
          continue
        if not (math.isfinite(reference_value) and math.isfinite(value)):
          continue
        paired_rows.append({
            "evaluation": label,
            "reference": reference_label,
            "scenario": scenario,
            "task": task,
            "metric": metric,
            "value": value,
            "reference_value": reference_value,
            "delta": value - reference_value,
        })
  return paired_rows


def _display_name(metric: str) -> str:
  name = metric.split("/")[-1]
  if metric.startswith("smoothness/torque/"):
    torque_names = {
        "mean_squared_delta_l2_per_step": (
            "mean squared torque Δ L2 / step (N²·m²)"
        ),
        "mssd_mean_squared_second_difference_per_dof": (
            "torque MSSD (N²·m²)"
        ),
        "msgfd_mean_absolute_savgol_filter_deviation_per_dof": (
            "torque MSGFD (N·m)"
        ),
    }
    if name in torque_names:
      return torque_names[name]
  display_names = {
      "absolute_mechanical_energy": "total energy (J)",
      "total_absolute_torque_impulse": "total |torque| (N·m·s)",
      "orientation_error_rms_degrees": "orientation error RMS (deg)",
      "roll_pitch_rate_rms": "roll/pitch angular velocity RMS (rad/s)",
      "feet_height_error_mean_mm": "feet height error (mm)",
  }
  if name in display_names:
    return display_names[name]
  if name.startswith("mssd_"):
    return "MSSD"
  if name.startswith("msgfd_"):
    return "MSGFD"
  name = re.sub(r"_per_(second|step|dof)", r" / \1", name)
  return name.replace("_", " ").replace("hz", " Hz")


def _direction_label(metric: str, normalized: bool = False) -> str:
  """Returns the direction in which values in a metric panel improve."""
  if normalized:
    return "↑ better"
  direction = METRIC_IMPROVEMENT_DIRECTIONS.get(metric)
  if direction == 1:
    return "↑ better"
  if direction == -1:
    return "↓ better"
  return ""


def _panel_title(metric: str, normalized: bool = False) -> str:
  direction = _direction_label(metric, normalized)
  display_name = _display_name(metric)
  return f"{display_name}\n{direction}" if direction else display_name


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
  with path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(
        fp, fieldnames=("evaluation", "scenario", "metric", "mean", "std")
    )
    writer.writeheader()
    writer.writerows(rows)


def _write_paired_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
  with path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(
        fp,
        fieldnames=(
            "evaluation",
            "reference",
            "scenario",
            "task",
            "metric",
            "value",
            "reference_value",
            "delta",
        ),
    )
    writer.writeheader()
    writer.writerows(rows)


def _paired_percent_rows(
    rows: Sequence[Mapping[str, Any]],
    improvement_directions: Mapping[str, int],
) -> list[dict[str, Any]]:
  """Adds reference-relative percentages with positive meaning improvement."""
  percentage_rows = []
  for row in rows:
    reference_value = float(row["reference_value"])
    if reference_value == 0.0:
      continue
    direction = improvement_directions.get(str(row["metric"]))
    if direction not in (-1, 1):
      raise ValueError(
          f"Metric {row['metric']!r} needs an improvement direction of 1 "
          "(higher is better) or -1 (lower is better)."
      )
    percentage_rows.append({
        **row,
        "percent_improvement": (
            direction * float(row["delta"]) / abs(reference_value) * 100.0
        ),
    })
  return percentage_rows


def _write_paired_percent_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
  with path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(
        fp,
        fieldnames=(
            "evaluation",
            "reference",
            "scenario",
            "task",
            "metric",
            "value",
            "reference_value",
            "delta",
            "percent_improvement",
        ),
    )
    writer.writeheader()
    writer.writerows(rows)


def _plot(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    scenarios: Sequence[str],
    metrics: Sequence[str],
    output: Path,
    title: str,
    columns: int,
    dpi: int,
    show: bool,
) -> None:
  try:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError as error:
    raise ImportError(
        "matplotlib is required; install the project's notebooks extra."
    ) from error

  columns = min(columns, len(metrics))
  row_count = math.ceil(len(metrics) / columns)
  fig, axes = plt.subplots(
      row_count,
      columns,
      figsize=_figure_size(),
      squeeze=False,
  )
  x = np.arange(len(scenarios), dtype=float)
  group_width = 0.8
  bar_width = group_width / len(labels)
  offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * bar_width
  row_lookup = {
      (row["evaluation"], row["scenario"], row["metric"]): row for row in rows
  }
  first_handles = []
  first_legend_labels = []
  for metric_index, metric in enumerate(metrics):
    ax = axes.flat[metric_index]
    any_values = False
    for label_index, label in enumerate(labels):
      means = np.asarray([
          row_lookup[(label, scenario, metric)]["mean"]
          for scenario in scenarios
      ])
      stds = np.asarray([
          row_lookup[(label, scenario, metric)]["std"]
          for scenario in scenarios
      ])
      valid = np.isfinite(means)
      if not np.any(valid):
        continue
      any_values = True
      yerr = np.where(np.isfinite(stds[valid]), stds[valid], 0.0)
      ax.bar(
          x[valid] + offsets[label_index],
          means[valid],
          yerr=yerr,
          width=bar_width * 0.9,
          capsize=3,
          label=label,
      )
    if not first_handles:
      first_handles, first_legend_labels = ax.get_legend_handles_labels()
    ax.set_title(_panel_title(metric))
    ax.set_xticks(x, scenarios, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    if metric == "episode/fell":
      ax.set_ylim(-0.05, 1.05)
    if not any_values:
      ax.text(
          0.5,
          0.5,
          "metric unavailable",
          ha="center",
          va="center",
          transform=ax.transAxes,
      )
  for ax in axes.flat[len(metrics):]:
    ax.set_visible(False)
  if first_handles:
    fig.legend(
        first_handles,
        first_legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=len(labels),
    )
  fig.suptitle(title, y=0.945)
  fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
  fig.savefig(output, dpi=dpi, bbox_inches="tight")
  if show:
    plt.show()
  plt.close(fig)


def _values_within_iqr(
    values: np.ndarray, multiplier: float | None
) -> np.ndarray:
  """Returns finite values within Tukey's IQR fences for plotting."""
  values = np.asarray(values, dtype=float)
  values = values[np.isfinite(values)]
  if multiplier is None or len(values) < 2:
    return values
  q1, q3 = np.percentile(values, (25.0, 75.0))
  iqr = q3 - q1
  return values[
      (values >= q1 - multiplier * iqr)
      & (values <= q3 + multiplier * iqr)
  ]


def _figure_size() -> tuple[float, float]:
  return (FIGURE_WIDTH_INCHES, FIGURE_WIDTH_INCHES / FIGURE_ASPECT_RATIO)


def _plot_paired_distributions(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    metrics: Sequence[str],
    reference_label: str,
    output: Path,
    value_field: str,
    ylabel: str,
    title: str,
    normalized_direction: bool = False,
) -> None:
  try:
    from matplotlib.patches import Patch  # pylint: disable=g-import-not-at-top
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError as error:
    raise ImportError(
        "matplotlib is required; install the project's notebooks extra."
    ) from error

  method_labels = [label for label in labels if label != reference_label]
  columns = min(COLUMNS, len(metrics))
  row_count = math.ceil(len(metrics) / columns)
  fig, axes = plt.subplots(
      row_count,
      columns,
      figsize=_figure_size(),
      squeeze=False,
  )
  rng = np.random.default_rng(0)
  colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  for metric_index, metric in enumerate(metrics):
    ax = axes.flat[metric_index]
    distributions = [
        np.asarray([
            row[value_field]
            for row in rows
            if row["metric"] == metric and row["evaluation"] == label
        ])
        for label in method_labels
    ]
    if not all(len(values) for values in distributions):
      ax.text(
          0.5,
          0.5,
          "metric unavailable",
          ha="center",
          va="center",
          transform=ax.transAxes,
      )
      continue
    boxplot = ax.boxplot(
        distributions,
        tick_labels=[""] * len(method_labels),
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    for method_index, (box, values) in enumerate(
        zip(boxplot["boxes"], distributions, strict=True)
    ):
      color = colors[method_index % len(colors)]
      box.set_facecolor(color)
      box.set_alpha(0.45)
      position = method_index + 1
      plotted_values = _values_within_iqr(
          values, PAIRED_OUTLIER_IQR_MULTIPLIER
      )
      if PAIRED_SHOW_TASK_POINTS:
        jitter = rng.uniform(-0.13, 0.13, size=len(plotted_values))
        ax.scatter(
            position + jitter,
            plotted_values,
            s=8,
            alpha=PAIRED_POINT_ALPHA,
            color=color,
            edgecolors="none",
        )
      mean = float(np.mean(plotted_values))
      confidence_95 = (
          1.96
          * float(np.std(plotted_values, ddof=1))
          / math.sqrt(len(plotted_values))
          if len(plotted_values) > 1
          else 0.0
      )
      ax.errorbar(
          position,
          mean,
          yerr=confidence_95,
          fmt="D",
          color="black",
          markersize=4,
          capsize=4,
          zorder=4,
      )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(_panel_title(metric, normalized=normalized_direction))
    ax.set_ylabel(ylabel.format(reference=reference_label))
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)
  for ax in axes.flat[len(metrics):]:
    ax.set_visible(False)
  legend_handles = [
      Patch(
          facecolor=colors[index % len(colors)],
          edgecolor="black",
          alpha=0.45,
          label=label,
      )
      for index, label in enumerate(method_labels)
  ]
  fig.legend(
      handles=legend_handles,
      loc="lower center",
      bbox_to_anchor=(0.5, 0.005),
      ncols=min(PAIRED_LEGEND_COLUMNS, len(method_labels)),
  )
  fig.suptitle(
      f"{title}\n"
      "boxes: task distribution; diamonds: inlier mean ± 95% CI",
      y=0.985,
  )
  fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))
  fig.savefig(output, dpi=DPI, bbox_inches="tight")
  if SHOW:
    plt.show()
  plt.close(fig)


def _plot_paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    metrics: Sequence[str],
    reference_label: str,
    output: Path,
) -> None:
  _plot_paired_distributions(
      rows,
      labels,
      metrics,
      reference_label,
      output,
      "delta",
      "method − {reference}",
      PAIRED_TITLE,
  )


def _plot_paired_percentages(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    metrics: Sequence[str],
    reference_label: str,
    output: Path,
) -> None:
  _plot_paired_distributions(
      rows,
      labels,
      metrics,
      reference_label,
      output,
      "percent_improvement",
      "improvement vs {reference} (%)",
      PAIRED_PERCENT_TITLE,
      normalized_direction=True,
  )


def compare() -> tuple[Path, Path, Path, Path]:
  """Compares the evaluations configured by the module-level constants."""
  if len(METHODS) < 2:
    raise ValueError("Configure at least two entries in METHODS.")
  if not METRICS:
    raise ValueError("Configure at least one entry in METRICS.")
  invalid_directions = [
      metric
      for metric in METRICS
      if METRIC_IMPROVEMENT_DIRECTIONS.get(metric) not in (-1, 1)
  ]
  if invalid_directions:
    raise ValueError(
        "Configure METRIC_IMPROVEMENT_DIRECTIONS for: "
        + ", ".join(invalid_directions)
    )
  if COLUMNS <= 0:
    raise ValueError("COLUMNS must be positive.")
  if PAIRED_LEGEND_COLUMNS <= 0:
    raise ValueError("PAIRED_LEGEND_COLUMNS must be positive.")
  if FIGURE_WIDTH_INCHES <= 0 or FIGURE_ASPECT_RATIO <= 0:
    raise ValueError("Figure width and aspect ratio must be positive.")
  if DPI <= 0:
    raise ValueError("DPI must be positive.")
  if (
      PAIRED_OUTLIER_IQR_MULTIPLIER is not None
      and PAIRED_OUTLIER_IQR_MULTIPLIER < 0
  ):
    raise ValueError("PAIRED_OUTLIER_IQR_MULTIPLIER cannot be negative.")

  paths = [Path(path) for path in METHODS.values()]
  seed_paths = [_policy_seed_paths(path) for path in paths]
  aggregated = [_aggregate_rollouts(group, METRICS) for group in seed_paths]
  summaries = [summary for summary, _ in aggregated]
  base_labels = _unique_labels(list(METHODS))
  if PAIRED_REFERENCE not in METHODS:
    raise ValueError(
        f"PAIRED_REFERENCE {PAIRED_REFERENCE!r} is not in METHODS."
    )
  reference_label = base_labels[list(METHODS).index(PAIRED_REFERENCE)]
  labels = _labels_with_seed_counts(base_labels, seed_paths)
  paired_reference = labels[base_labels.index(reference_label)]
  scenarios, rows = _comparison_rows(summaries, labels, METRICS)
  if not scenarios:
    raise ValueError("The evaluation reports do not contain any scenarios.")

  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  csv_output = CSV_OUTPUT or OUTPUT.with_suffix(".csv")
  csv_output.parent.mkdir(parents=True, exist_ok=True)
  _write_csv(csv_output, rows)
  _plot(
      rows,
      labels,
      scenarios,
      METRICS,
      OUTPUT,
      TITLE,
      COLUMNS,
      DPI,
      SHOW,
  )
  _validate_paired_reports(summaries, labels, paired_reference)
  rollout_tables = [table for _, table in aggregated]
  paired_rows = _paired_delta_rows(
      rollout_tables, labels, METRICS, paired_reference
  )
  paired_methods = [label for label in labels if label != paired_reference]
  available_pairs = {
      (row["evaluation"], row["metric"]) for row in paired_rows
  }
  unavailable_metrics = [
      metric
      for metric in METRICS
      if any(
          (method, metric) not in available_pairs for method in paired_methods
      )
  ]
  if unavailable_metrics:
    print(
        "Warning: paired metrics unavailable in one or more older reports; "
        "rerun those evaluations to populate: "
        + ", ".join(unavailable_metrics)
    )
  PAIRED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  paired_csv_output = PAIRED_CSV_OUTPUT or PAIRED_OUTPUT.with_suffix(".csv")
  paired_csv_output.parent.mkdir(parents=True, exist_ok=True)
  _write_paired_csv(paired_csv_output, paired_rows)
  _plot_paired_deltas(
      paired_rows, labels, METRICS, paired_reference, PAIRED_OUTPUT
  )
  paired_percent_rows = _paired_percent_rows(
      paired_rows, METRIC_IMPROVEMENT_DIRECTIONS
  )
  PAIRED_PERCENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  paired_percent_csv_output = (
      PAIRED_PERCENT_CSV_OUTPUT or PAIRED_PERCENT_OUTPUT.with_suffix(".csv")
  )
  paired_percent_csv_output.parent.mkdir(parents=True, exist_ok=True)
  _write_paired_percent_csv(paired_percent_csv_output, paired_percent_rows)
  _plot_paired_percentages(
      paired_percent_rows,
      labels,
      METRICS,
      paired_reference,
      PAIRED_PERCENT_OUTPUT,
  )
  print(f"Comparison plot written to: {OUTPUT.resolve()}")
  print(f"Comparison data written to: {csv_output.resolve()}")
  print(f"Paired comparison plot written to: {PAIRED_OUTPUT.resolve()}")
  print(f"Paired comparison data written to: {paired_csv_output.resolve()}")
  print(
      "Paired percentage plot written to: "
      f"{PAIRED_PERCENT_OUTPUT.resolve()}"
  )
  print(
      "Paired percentage data written to: "
      f"{paired_percent_csv_output.resolve()}"
  )
  return OUTPUT, csv_output, PAIRED_OUTPUT, paired_csv_output


def main() -> None:
  compare()


if __name__ == "__main__":
  main()
