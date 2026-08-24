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
Local evaluations are preferred when complete; downloaded cluster evaluations
are used automatically otherwise.  Plots from either source share one results
directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker
from matplotlib import colors as matplotlib_colors

from learning import pareto_policy_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_ROOT = PROJECT_ROOT / "evaluations" / "pareto_cluster"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "evaluations" / "pareto_results"
DEFAULT_XLIM_CONFIG = Path(__file__).with_name("pareto_xlim.json")
DEFAULT_PLOT_CONFIG_ROOT = Path(__file__).with_name("pareto_plot_configs")
DEFAULT_ENVIRONMENT = "Go1JoystickFlatTerrain"
DEFAULT_X_METRIC = "eval_reward_means/total_without_regularization"
AGGREGATE_CACHE_NAME = "pareto_aggregates.csv"
AGGREGATE_CACHE_MANIFEST_NAME = "pareto_aggregates_cache.json"
AGGREGATE_CACHE_VERSION = 3
DEFAULT_Y_METRICS = (
    "smoothness/torque/mssd_mean_squared_second_difference_per_dof",
    "smoothness/torque/msgfd_w5_p2_mean_absolute_savgol_filter_deviation_per_dof",
    "smoothness/torque/msgfd_w11_p3_mean_absolute_savgol_filter_deviation_per_dof",
    "smoothness/torque/msgfd_w21_p3_mean_absolute_savgol_filter_deviation_per_dof",
    "smoothness/torque/msgfd_w31_p3_mean_absolute_savgol_filter_deviation_per_dof",
    "smoothness/torque/total_variation",
    "smoothness/torque/sign_changes_total",
    "torque_spectrum/eval/total_energy_per_step",
    "tracking/absolute_mechanical_energy",
)
METRIC_LABELS = {
    DEFAULT_X_METRIC: "Task reward without regularization\n(higher is better)",
    "smoothness/torque/mssd_mean_squared_second_difference_per_dof": (
        "Torque smoothness: mean squared second difference\n"
        "(N²·m² per DoF; lower is better)"
    ),
    "smoothness/torque/msgfd_mean_absolute_savgol_filter_deviation_per_dof": (
        "Torque smoothness: mean Savitzky–Golay deviation\n"
        "(N·m per DoF; lower is better)"
    ),
    "smoothness/torque/total_variation": (
        "Torque total variation per episode\n"
        "(N·m; lower is better)"
    ),
    "smoothness/torque/sign_changes_total": (
        "Total torque sign changes per episode\n"
        "(count; lower is better)"
    ),
    "torque_spectrum/eval/total_energy_per_step": (
        "Total torque spectral energy per step\n"
        "(N²·m²; lower is better)"
    ),
    "tracking/absolute_mechanical_energy": (
        "Absolute mechanical energy per episode\n(J; lower is better)"
    ),
}

# Methods included in every Pareto subplot, in legend order. Comment out a
# line to omit that method from a comparison. Add configured method IDs here
# when plotting a manifest that contains additional variants.
PLOTTED_METHODS = (
    "action_smoothness",
    "baseline",
    "torque_rate",
    "torque_smoothness",
    "high_pass",
    "high_pass_f4_m1",
    #"high_pass_f5_m0p5",
    "high_pass_f5_m1p5",
    #"high_pass_f6_m1",

)
METHOD_LABELS = {
    "action_smoothness": "Action smoothness",
    "baseline": "Action rate",
    "torque_rate": "Torque rate",
    "torque_smoothness": "Torque smoothness",
    "high_pass": "High-pass torque",
}
METHOD_COLORS = {
    "action_smoothness": "#B279A2",
    "baseline": "#4C78A8",
    "torque_rate": "#F58518",
    "torque_smoothness": "#E45756",
    "high_pass": "#54A24B",
    # Keep configured high-pass variants visually separable instead of
    # relying on independently hashed hues, which can land close together.
    "high_pass_f2_m1": "#9467BD",
    "high_pass_f3_m1": "#17BECF",
    "high_pass_f4_m1": "#8C564B",
    "high_pass_f5_m0p5": "#E377C2",
    "high_pass_f5_m1p5": "#BCBD22",
    "high_pass_f5_m2": "#7F7F7F",
    # f=5 uses the historical green above; use a deep purple for f=6 so the
    # adjacent cutoff settings remain unmistakable in lines and markers.
    "high_pass_f6_m1": "#7B2CBF",
}


def _method_label(method: str) -> str:
  """Returns a readable label, including configured high-pass variants."""
  if method in METHOD_LABELS:
    return METHOD_LABELS[method]
  # Configured method IDs use p as a filesystem-safe decimal point.
  configured = re.fullmatch(
      r"high_pass_f(?P<f>[0-9]+(?:p[0-9]+)?)"
      r"(?:_o(?P<o>[0-9]+))?_m(?P<m>[0-9]+(?:p[0-9]+)?)",
      method,
  )
  if configured is None:
    return method.replace("_", " ").title()
  cutoff = configured.group("f").replace("p", ".")
  difference = configured.group("m").replace("p", ".")
  order = configured.group("o")
  order_text = "" if order is None else f", order={order}"
  return f"High-pass torque (f={cutoff} Hz, m={difference}{order_text})"


def _metric_label(metric: str) -> str:
  """Returns a concise plot label while retaining support for custom metrics."""
  if metric in METRIC_LABELS:
    return METRIC_LABELS[metric]
  spectral_energy = re.fullmatch(
      r"torque_spectrum/eval/fft_above_(?P<cutoff>[0-9]+(?:\.[0-9]+)?)"
      r"hz_energy_per_step",
      metric,
  )
  if spectral_energy is not None:
    return (
        "Torque spectral energy above "
        f"{spectral_energy.group('cutoff')} Hz per step\n"
        "(N²·m²; lower is better)"
    )
  savgol = re.fullmatch(
      r"smoothness/torque/msgfd_w(?P<window>[0-9]+)_p(?P<order>[0-9]+)_"
      r"mean_absolute_savgol_filter_deviation_per_dof",
      metric,
  )
  if savgol is not None:
    return (
        "Torque Savitzky–Golay deviation "
        f"(window={savgol.group('window')}, order={savgol.group('order')})\n"
        "(N·m per DoF; lower is better)"
    )
  return metric.split("/")[-1].replace("_", " ").capitalize()


def _percent_above_minimum(
    values: np.ndarray, minimum: float, metric: str
) -> np.ndarray:
  """Expresses values as percentages above a positive plotted minimum."""
  if minimum <= 0:
    raise ValueError(
        f"Cannot scale {metric!r} relative to its lowest plotted value "
        f"({minimum:g}); the minimum must be positive."
    )
  return 100.0 * (values / minimum - 1.0)


def _exponential_axis_functions(
    minimum: float, maximum: float, strength: float
):
  """Returns an invertible transform that expands the high end of an axis."""
  span = maximum - minimum
  if not math.isfinite(strength) or strength <= 0:
    raise ValueError("Exponential x-axis strength must be positive and finite.")
  if span <= 0:
    raise ValueError("Exponential x-axis scaling requires distinct x values.")
  denominator = np.expm1(strength)

  def forward(values):
    normalized = (np.asarray(values) - minimum) / span
    return np.expm1(strength * normalized) / denominator

  def inverse(values):
    normalized = np.log1p(np.asarray(values) * denominator) / strength
    return minimum + span * normalized

  return forward, inverse


def _axis_scaling_tag(
    *,
    y_percent_above_minimum: bool,
    log_y_axis: bool,
    log_percentage_y: bool,
    shifted_log_percentage_y: bool,
    x_exponential_strength: float,
) -> str:
  """Returns a filename-safe description of both axis transformations."""
  x_scaling = (
      "x-linear"
      if not x_exponential_strength
      else "x-exp-" + f"{x_exponential_strength:g}".replace(".", "p")
  )
  if y_percent_above_minimum:
    if shifted_log_percentage_y:
      y_scaling = "y-percent-shifted-log"
    elif log_percentage_y:
      y_scaling = "y-percent-symlog"
    else:
      y_scaling = "y-percent-linear"
  else:
    y_scaling = "y-absolute-log" if log_y_axis else "y-absolute-linear"
  return f"{x_scaling}_{y_scaling}"


def _output_variant_path(
    base_output: Path,
    scale_encoding: str,
    *,
    y_percent_above_minimum: bool,
    log_y_axis: bool,
    log_percentage_y: bool,
    shifted_log_percentage_y: bool,
    x_exponential_strength: float,
) -> Path:
  """Adds axis-scaling and penalty-representation tags to an output path."""
  axis_tag = _axis_scaling_tag(
      y_percent_above_minimum=y_percent_above_minimum,
      log_y_axis=log_y_axis,
      log_percentage_y=log_percentage_y,
      shifted_log_percentage_y=shifted_log_percentage_y,
      x_exponential_strength=x_exponential_strength,
  )
  return base_output.with_name(
      f"{base_output.stem}_{axis_tag}_repr-{scale_encoding}"
      f"{base_output.suffix}"
  )


def _method_color(method: str) -> str:
  """Returns stable colors while preserving colors of historical methods."""
  if method in METHOD_COLORS:
    return METHOD_COLORS[method]
  digest = hashlib.blake2b(method.encode("utf-8"), digest_size=8).digest()
  hue = int.from_bytes(digest, "big") / float(1 << 64)
  return matplotlib_colors.to_hex(
      matplotlib_colors.hsv_to_rgb((hue, 0.62, 0.72))
  )


def _method_order(
    points: Sequence["Point"],
    all_methods: bool = False,
    methods: Sequence[str] = PLOTTED_METHODS,
) -> list[str]:
  present = {point.method for point in points}
  if all_methods:
    configured_order = [
        method for method in METHOD_LABELS if method in present
    ]
    return configured_order + sorted(present - set(configured_order))
  return [method for method in methods if method in present]


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


class MissingEvaluationMetricsError(ValueError):
  """Raised when reports predate metrics requested by the plotter."""


def _require_metrics(
    aggregates: Sequence[dict[str, str]], metrics: Sequence[str]
) -> None:
  missing = sorted({
      metric
      for metric in metrics
      if any(metric not in row or not row[metric] for row in aggregates)
  })
  if missing:
    formatted = "\n".join(f"  {metric}" for metric in missing)
    raise MissingEvaluationMetricsError(
        "The evaluation reports do not contain these requested metrics:\n"
        f"{formatted}\n"
        "Regenerate the policy evaluations with the current evaluator before "
        "plotting."
    )


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


def _normalized_log_scales(points: Sequence[Point]) -> np.ndarray:
  """Maps positive penalty scales to [0, 1] within one method."""
  scales = np.asarray([point.scale for point in points], dtype=float)
  if np.any(scales <= 0):
    values = scales
  else:
    values = np.log10(scales)
  span = np.ptp(values)
  if span == 0:
    return np.full(values.shape, 0.5)
  return (values - np.min(values)) / span


def _float(row: dict[str, str], metric: str) -> float:
  try:
    value = float(row[metric])
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError(f"Missing or invalid metric {metric!r}") from error
  if not math.isfinite(value):
    raise ValueError(f"Metric {metric!r} is not finite: {value}")
  return value


def _filter_points(
    points: Sequence[Point],
    xlim: tuple[float, float | None] | None,
) -> list[Point]:
  """Applies an x range, or returns every point when none is configured."""
  if xlim is None:
    return list(points)
  return [
      point
      for point in points
      if point.x >= xlim[0]
      and (xlim[1] is None or point.x <= xlim[1])
  ]


def _filter_methods(
    points: Sequence[Point], methods: Sequence[str] = PLOTTED_METHODS
) -> list[Point]:
  """Keeps only explicitly selected methods."""
  enabled = set(methods)
  return [point for point in points if point.method in enabled]


def _configured_methods(
    environment: str, config_root: Path = DEFAULT_PLOT_CONFIG_ROOT
) -> tuple[bool, tuple[str, ...]]:
  """Loads whether to plot all methods or an ordered environment subset."""
  path = config_root / f"{environment}.toml"
  try:
    with path.open("rb") as config_file:
      value = tomllib.load(config_file)
  except FileNotFoundError:
    return True, ()
  except (OSError, tomllib.TOMLDecodeError) as error:
    raise ValueError(f"Cannot read Pareto plot config {path}: {error}") from error
  if not isinstance(value, dict) or not isinstance(
      value.get("all_methods"), bool
  ):
    raise ValueError(
        f"Pareto plot config must contain boolean 'all_methods': {path}"
    )
  all_methods = value["all_methods"]
  methods = value.get("methods", [])
  if not isinstance(methods, list) or not all(
      isinstance(method, str) and method for method in methods
  ):
    raise ValueError(f"Pareto plot config 'methods' must be a string list: {path}")
  if len(methods) != len(set(methods)):
    raise ValueError(f"Pareto plot config contains duplicate methods: {path}")
  if all_methods and methods:
    raise ValueError(
        f"Pareto plot config cannot select methods when 'all_methods' is true: {path}"
    )
  if not all_methods and not methods:
    raise ValueError(
        f"Pareto plot config must select methods when 'all_methods' is false: {path}"
    )
  return all_methods, tuple(methods)


def _configured_xlim(environment: str, path: Path) -> tuple[float, None] | None:
  """Loads an optional environment-specific lower x bound."""
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"Cannot read Pareto x-limit config {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError(f"Pareto x-limit config must be a JSON object: {path}")
  lower = value.get(environment)
  if lower is None:
    return None
  if isinstance(lower, bool) or not isinstance(lower, (int, float)):
    raise ValueError(
        f"Pareto x-limit for {environment!r} must be a number, got {lower!r}"
    )
  lower = float(lower)
  if not math.isfinite(lower):
    raise ValueError(
        f"Pareto x-limit for {environment!r} must be finite, got {lower!r}"
    )
  return (lower, None)


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
  substituted = []
  for run in _manifest_runs(manifest):
    run_root = (
        evaluation_root
        / "raw_torque"
        / str(run["run_name"])
    )
    report = run_root / str(run["checkpoint"]) / "rollouts.csv"
    if not report.is_file():
      alternatives = sorted(
          candidate
          for candidate in run_root.glob("*/rollouts.csv")
          if candidate.parent.name.isdigit()
      )
      if len(alternatives) != 1:
        detail = (
            "no alternative report exists"
            if not alternatives
            else f"{len(alternatives)} alternative reports are ambiguous"
        )
        raise FileNotFoundError(
            f"Evaluation report not found: {report}; {detail}"
        )
      substituted.append((report, alternatives[0]))
      report = alternatives[0]
    reports.append((run, report))
  if substituted:
    print(
        f"Using the sole completed checkpoint for {len(substituted)} runs "
        "whose manifest checkpoint was not evaluated."
    )
  return reports


def _source_is_available(manifest: Path, evaluation_root: Path) -> bool:
  """Whether a source has a readable manifest and every referenced report."""
  if not manifest.is_file():
    return False
  try:
    _report_paths(manifest, evaluation_root)
  except (OSError, KeyError, TypeError, ValueError):
    return False
  return True


def _default_source(environment: str) -> tuple[Path, Path]:
  """Selects complete local evaluations first, then downloaded cluster data."""
  local_root = pareto_policy_pipeline.DEFAULT_OUTPUT_ROOT / environment
  local_manifest = (
      pareto_policy_pipeline.DEFAULT_LOCAL_ROOT
      / environment
      / pareto_policy_pipeline.MANIFEST_NAME
  )
  cluster_root = DEFAULT_CLUSTER_ROOT / environment
  cluster_manifest = cluster_root / pareto_policy_pipeline.MANIFEST_NAME
  candidates = (
      ("local", local_manifest, local_root),
      ("cluster", cluster_manifest, cluster_root),
  )
  for label, manifest, evaluation_root in candidates:
    if _source_is_available(manifest, evaluation_root):
      print(f"Using {label} Pareto evaluations: {evaluation_root}")
      return manifest, evaluation_root
  checked = "\n".join(
      f"  {label}: manifest={manifest}, reports={evaluation_root / 'raw_torque'}"
      for label, manifest, evaluation_root in candidates
  )
  raise FileNotFoundError(
      f"No complete Pareto evaluations found for {environment!r}. Checked:\n"
      f"{checked}"
  )


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
    scale_encoding: str = "labels",
    hide_non_pareto: bool = True,
    y_percent_above_minimum: bool = False,
    log_y_axis: bool = False,
    log_percentage_y: bool = True,
    shifted_log_percentage_y: bool = False,
    x_exponential_strength: float = 0.0,
    all_methods: bool = False,
    methods: Sequence[str] = PLOTTED_METHODS,
) -> Path:
  """Plots Pareto fronts with one of several penalty-scale encodings."""
  if scale_encoding not in {"labels", "size", "opacity", "arrows"}:
    raise ValueError(f"Unknown scale encoding: {scale_encoding!r}")
  if log_y_axis and y_percent_above_minimum:
    raise ValueError(
        "A logarithmic absolute y axis cannot be combined with percentage "
        "y values. Use the percentage-axis options instead."
    )
  columns = 4
  rows = math.ceil(len(y_metrics) / columns)
  figure, axes = plt.subplots(
      rows, columns, figsize=(7.0 * columns, 5.2 * rows), squeeze=False
  )
  aggregates = load_aggregates(manifest, evaluation_root)
  _require_metrics(aggregates, (x_metric, *y_metrics))
  csv_rows = []
  method_scale_ranges: dict[str, tuple[float, float]] = {}
  for axis, y_metric in zip(axes.flat, y_metrics):
    points = list(
        _points_from_aggregates(aggregates, x_metric, y_metric)
    )
    points = _filter_points(points, xlim)
    if not points:
      selected_range = (
          "in the available reports"
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
    available_methods = sorted({point.method for point in points})
    if not all_methods:
      points = _filter_methods(points, methods)
    if not points:
      raise ValueError(
          f"None of the selected methods {list(methods)} are available. "
          f"Report methods: {available_methods}."
      )
    plotted_y = []
    for method in _method_order(points, all_methods=all_methods, methods=methods):
      method_points = [point for point in points if point.method == method]
      method_x = np.asarray([point.x for point in method_points])
      method_y = np.asarray([point.y for point in method_points])
      visible = (
          pareto_mask(method_x, method_y)
          if hide_non_pareto
          else np.ones(method_y.shape, dtype=bool)
      )
      plotted_y.extend(method_y[visible])
    y_minimum = min(plotted_y)
    if log_y_axis and y_minimum <= 0:
      raise ValueError(
          f"Cannot use a logarithmic y axis for {y_metric!r}: its lowest "
          f"visible value is {y_minimum:g}, but log scaling requires "
          "strictly positive values."
      )
    for method in _method_order(points, all_methods=all_methods, methods=methods):
      method_points = [point for point in points if point.method == method]
      if not method_points:
        continue
      x = np.asarray([point.x for point in method_points])
      y = np.asarray([point.y for point in method_points])
      normalized_scales = _normalized_log_scales(method_points)
      method_scale_ranges[method] = (
          min(point.scale for point in method_points),
          max(point.scale for point in method_points),
      )
      front = pareto_mask(x, y)
      visible = front if hide_non_pareto else np.ones_like(front, dtype=bool)
      display_y = (
          _percent_above_minimum(y, y_minimum, y_metric)
          if y_percent_above_minimum
          else y
      )
      color = _method_color(method)
      sizes = (
          42.0 + 120.0 * normalized_scales
          if scale_encoding == "size"
          else np.full(x.shape, 54.0)
      )
      colors = color
      if scale_encoding == "opacity":
        rgb = matplotlib_colors.to_rgb(color)
        colors = [
            (*rgb, 0.25 + 0.75 * normalized)
            for normalized in normalized_scales
        ]
      axis.scatter(
          x[visible],
          display_y[visible],
          color=(
              np.asarray(colors)[visible]
              if scale_encoding == "opacity"
              else colors
          ),
          s=sizes[visible],
          alpha=None if scale_encoding == "opacity" else 0.85,
          label=_method_label(method),
      )
      front_order = np.argsort(x[front])
      axis.plot(
          x[front][front_order],
          display_y[front][front_order],
          color=color,
          linewidth=2.2,
      )
      if scale_encoding == "arrows" and len(method_points) > 1:
        visible_indices = np.flatnonzero(visible)
        scale_order = visible_indices[np.argsort(
            [method_points[index].scale for index in visible_indices]
        )]
        ordered_x = x[scale_order]
        ordered_y = display_y[scale_order]
        axis.plot(
            ordered_x,
            ordered_y,
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.55,
        )
        for start, end in zip(
            zip(ordered_x[:-1], ordered_y[:-1]),
            zip(ordered_x[1:], ordered_y[1:]),
        ):
          axis.annotate(
              "",
              xy=end,
              xytext=start,
              arrowprops={
                  "arrowstyle": "->",
                  "color": color,
                  "alpha": 0.65,
                  "linewidth": 1.2,
                  "shrinkA": 7,
                  "shrinkB": 7,
              },
          )
      for point, is_front, is_visible, displayed_y in zip(
          method_points, front, visible, display_y
      ):
        coverage_suffix = (
            ""
            if point.seed_count == point.expected_seed_count
            else f"\n{point.seed_count}/{point.expected_seed_count} seeds"
        )
        if scale_encoding == "labels" and is_visible:
          axis.annotate(
              f"{point.scale:g}{coverage_suffix}",
              (point.x, displayed_y),
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
    axis.set_xlabel(_metric_label(x_metric))
    axis.set_ylabel(
        f"{_metric_label(y_metric).splitlines()[0]}\n"
        "% above lowest plotted value (lower is better)"
        if y_percent_above_minimum
        else _metric_label(y_metric)
    )
    if y_percent_above_minimum:
      axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100.0))
      if shifted_log_percentage_y:
        axis.set_yscale(
            "function",
            functions=(
                lambda value: np.log1p(value / 100.0),
                lambda value: 100.0 * np.expm1(value),
            ),
        )
        tick_candidates = np.asarray(
            [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
            dtype=float,
        )
        percentage_max = 100.0 * (max(plotted_y) / y_minimum - 1.0)
        visible_ticks = tick_candidates[tick_candidates <= percentage_max]
        if len(visible_ticks) < 2 and percentage_max > 0:
          visible_ticks = np.asarray([0.0, percentage_max])
        elif not len(visible_ticks):
          visible_ticks = np.asarray([0.0])
        axis.yaxis.set_major_locator(ticker.FixedLocator(visible_ticks))
      elif log_percentage_y:
        # A strict log scale cannot represent the best point at 0%. Symlog
        # retains it and transitions to logarithmic spacing above 1%.
        axis.set_yscale("symlog", base=10, linthresh=1.0, linscale=1.0)
    elif log_y_axis:
      axis.set_yscale("log")
    if x_exponential_strength:
      plotted_x = np.asarray([point.x for point in points])
      axis.set_xscale(
          "function",
          functions=_exponential_axis_functions(
              float(np.min(plotted_x)),
              float(np.max(plotted_x)),
              x_exponential_strength,
          ),
      )
    if xlim is not None:
      axis.set_xlim(left=xlim[0], right=xlim[1])
    axis.grid(alpha=0.25)
  for axis in axes.flat[len(y_metrics):]:
    axis.set_visible(False)
  handles, labels = axes.flat[0].get_legend_handles_labels()
  figure.legend(
      handles,
      labels,
      loc="upper center",
      bbox_to_anchor=(0.5, 0.975),
      ncol=len(labels),
      title=(
          "Regularization method · y axis: shifted log of % above minimum"
          if y_percent_above_minimum and shifted_log_percentage_y
          else "Regularization method"
      ),
  )
  figure.suptitle("Penalty-scale Pareto fronts", y=0.995)
  encoding_notes = {
      "size": "Marker size increases with penalty scale within each method.",
      "opacity": "Marker darkness increases with penalty scale within each method.",
      "arrows": "Dotted arrows point toward increasing penalty scale.",
  }
  if scale_encoding in encoding_notes and scale_encoding != "opacity":
    figure.text(
        0.5,
        0.005,
        encoding_notes[scale_encoding],
        ha="center",
        fontsize=9,
    )
  figure.tight_layout(
      rect=(
          0,
          (
              0.13
              if scale_encoding == "opacity"
              else 0.025
              if scale_encoding in encoding_notes
              else 0
          ),
          1,
          0.925,
      )
  )
  if scale_encoding == "opacity":
    methods = [
        method
        for method in _method_order(
            points, all_methods=all_methods, methods=methods
        )
        if method in method_scale_ranges
    ]
    bar_width = 0.24
    gap = 0.055
    total_width = len(methods) * bar_width + (len(methods) - 1) * gap
    left = (1.0 - total_width) / 2.0
    for index, method in enumerate(methods):
      minimum, maximum = method_scale_ranges[method]
      rgb = matplotlib_colors.to_rgb(_method_color(method))
      opacity_map = matplotlib_colors.LinearSegmentedColormap.from_list(
          f"{method}_penalty_opacity",
          [(*rgb, 0.25), (*rgb, 1.0)],
      )
      if minimum > 0 and minimum != maximum:
        norm = matplotlib_colors.LogNorm(vmin=minimum, vmax=maximum)
        midpoint = math.sqrt(minimum * maximum)
      else:
        norm = matplotlib_colors.Normalize(vmin=minimum, vmax=maximum)
        midpoint = (minimum + maximum) / 2.0
      scalar_mappable = cm.ScalarMappable(norm=norm, cmap=opacity_map)
      scalar_mappable.set_array([])
      colorbar_axis = figure.add_axes(
          [left + index * (bar_width + gap), 0.035, bar_width, 0.012]
      )
      colorbar = figure.colorbar(
          scalar_mappable, cax=colorbar_axis, orientation="horizontal"
      )
      ticks = sorted(set((minimum, midpoint, maximum)))
      colorbar.set_ticks(ticks)
      colorbar.set_ticklabels([f"{tick:.2g}" for tick in ticks])
      colorbar.ax.set_title(
          f"{_method_label(method)} penalty scale", fontsize=9, pad=5
      )
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
      help=(
          "Evaluation report root. By default complete local evaluations are "
          "preferred, with downloaded cluster evaluations as fallback."
      ),
  )
  parser.add_argument(
      "--cluster",
      action="store_true",
      help=(
          "Force downloaded cluster evaluations instead of automatic "
          "local-first discovery."
      ),
  )
  method_selection = parser.add_mutually_exclusive_group()
  method_selection.add_argument(
      "--all-methods",
      action="store_true",
      help=(
          "Plot every method found in the manifest, including every "
          "high-pass cutoff/m configuration. Overrides the environment "
          "plot config."
      ),
  )
  method_selection.add_argument(
      "--method",
      action="append",
      dest="methods",
      help=(
          "Plot only this method; repeat to select multiple methods. "
          "Overrides the environment plot config."
      ),
  )
  parser.add_argument(
      "--output",
      type=Path,
      help=(
          "Base output path. Axis-scaling and representation tags are added "
          "to its filename. Defaults below evaluations/pareto_results/"
          "<environment>/."
      ),
  )
  parser.add_argument("--x-metric", default=DEFAULT_X_METRIC)
  parser.add_argument(
      "--x-exponential-strength",
      type=float,
      default=0.0,
      metavar="STRENGTH",
      help=(
          "Exponentially expand high-reward x values. 0 is linear; try 1 "
          "for mild, 2 for moderate, or 4 for strong expansion."
      ),
  )
  parser.add_argument(
      "--xlim",
      nargs="+",
      type=float,
      metavar="LIMIT",
      help=(
          "Set MIN, or MIN MAX, for the included and displayed x range. "
          "Overrides the environment entry in learning/pareto_xlim.json."
      ),
  )
  parser.add_argument(
      "--y-metric",
      action="append",
      dest="y_metrics",
      help="Metric to plot; repeat for multiple subplots.",
  )
  parser.add_argument(
      "--hide-non-pareto",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Hide dominated policy points (default). Use --no-hide-non-pareto "
          "to show every evaluated policy."
      ),
  )
  y_scale = parser.add_mutually_exclusive_group()
  y_scale.add_argument(
      "--y-percent-above-minimum",
      dest="y_percent_above_minimum",
      action="store_true",
      help=(
          "Scale each y axis as the percentage above its lowest visible "
          "value instead of showing absolute metric values."
      ),
  )
  y_scale.add_argument(
      "--absolute-y-values",
      dest="y_percent_above_minimum",
      action="store_false",
      default=False,
      help="Show absolute y-metric values (default).",
  )
  y_scale.add_argument(
      "--log-y-axis",
      action="store_true",
      help=(
          "Show absolute y-metric values on a logarithmic axis. All visible "
          "values must be strictly positive."
      ),
  )
  percentage_axis = parser.add_mutually_exclusive_group()
  percentage_axis.add_argument(
      "--linear-percentage-y-axis",
      dest="log_percentage_y",
      action="store_false",
      default=True,
      help=(
          "Use a linear y axis in percentage mode. By default it is linear "
          "through 1%% so 0%% remains visible, then logarithmic."
      ),
  )
  percentage_axis.add_argument(
      "--shifted-log-percentage-y-axis",
      dest="shifted_log_percentage_y",
      action="store_true",
      help=(
          "Transform percentage-mode y coordinates as log(1 + percentage / "
          "100), while labeling ticks in the original percentages."
      ),
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
  configured_all_methods, configured_methods = _configured_methods(environment)
  all_methods = args.all_methods or (
      not args.methods and configured_all_methods
  )
  methods = tuple(args.methods or configured_methods)
  if args.manifest is not None or args.evaluation_root is not None:
    if args.cluster:
      raise ValueError(
          "--cluster cannot be combined with --manifest or --evaluation-root."
      )
    if args.manifest is not None and args.evaluation_root is not None:
      manifest = args.manifest
      evaluation_root = args.evaluation_root
    elif args.evaluation_root is not None:
      evaluation_root = args.evaluation_root
      colocated_manifest = (
          evaluation_root / pareto_policy_pipeline.MANIFEST_NAME
      )
      manifest = (
          colocated_manifest
          if colocated_manifest.is_file()
          else pareto_policy_pipeline.DEFAULT_LOCAL_ROOT
          / environment
          / pareto_policy_pipeline.MANIFEST_NAME
      )
    else:
      assert args.manifest is not None
      manifest = args.manifest
      evaluation_root = (
          manifest.parent
          if (manifest.parent / "raw_torque").is_dir()
          else pareto_policy_pipeline.DEFAULT_OUTPUT_ROOT / environment
      )
  elif args.cluster:
    evaluation_root = DEFAULT_CLUSTER_ROOT / environment
    manifest = evaluation_root / pareto_policy_pipeline.MANIFEST_NAME
  else:
    manifest, evaluation_root = _default_source(environment)
  if args.xlim is not None and len(args.xlim) not in (1, 2):
    raise ValueError("--xlim accepts either MIN or MIN MAX.")
  xlim = (
      (args.xlim[0], args.xlim[1] if len(args.xlim) == 2 else None)
      if args.xlim is not None
      else None
  )
  if args.xlim is None:
    xlim = _configured_xlim(environment, DEFAULT_XLIM_CONFIG)
  if xlim is not None and xlim[1] is not None and xlim[0] >= xlim[1]:
    raise ValueError("--xlim MIN must be smaller than MAX.")
  if not math.isfinite(args.x_exponential_strength):
    raise ValueError("--x-exponential-strength must be finite.")
  if args.x_exponential_strength < 0:
    raise ValueError("--x-exponential-strength cannot be negative.")
  base_output_path = args.output or (
      DEFAULT_RESULTS_ROOT / environment / "policy_pareto.png"
  )
  try:
    outputs = []
    for scale_encoding in ("labels", "size", "opacity", "arrows"):
      variant_output = _output_variant_path(
          base_output_path,
          scale_encoding,
          y_percent_above_minimum=args.y_percent_above_minimum,
          log_y_axis=args.log_y_axis,
          log_percentage_y=args.log_percentage_y,
          shifted_log_percentage_y=args.shifted_log_percentage_y,
          x_exponential_strength=args.x_exponential_strength,
      )
      outputs.append(
          plot(
              manifest,
              evaluation_root,
              variant_output,
              args.x_metric,
              tuple(args.y_metrics or DEFAULT_Y_METRICS),
              xlim,
              scale_encoding=scale_encoding,
              hide_non_pareto=args.hide_non_pareto,
              y_percent_above_minimum=args.y_percent_above_minimum,
              log_y_axis=args.log_y_axis,
              log_percentage_y=args.log_percentage_y,
              shifted_log_percentage_y=args.shifted_log_percentage_y,
              x_exponential_strength=args.x_exponential_strength,
              all_methods=all_methods,
              methods=methods,
          )
        )
  except MissingEvaluationMetricsError as error:
    cluster_hint = (
        "\nFor cluster results, update the Eagle checkout, then run:\n"
        f"  python -m learning.pareto_cluster submit {environment}\n"
        "After the Slurm job completes, fetch the refreshed reports:\n"
        f"  python -m learning.pareto_cluster fetch {environment}"
    )
    raise SystemExit(f"{error}{cluster_hint}") from None
  for generated_output in outputs:
    print(f"Pareto plot: {generated_output.resolve()}")
    print(f"Pareto table: {generated_output.with_suffix('.csv').resolve()}")


if __name__ == "__main__":
  main()
