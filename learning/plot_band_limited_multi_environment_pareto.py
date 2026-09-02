"""Plot torque-MSSD Pareto fronts for the four band-limited environments.

This is the compact, 2-by-2 companion to
``learning.plot_multi_environment_pareto``.  Its aggregate CSV is kept
separate so it can be refreshed independently by deleting it.

Example:
  MPLBACKEND=Agg python -m learning.plot_band_limited_multi_environment_pareto
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from learning import plot_multi_environment_pareto as multi
from learning import plot_policy_pareto as pareto


# Panels are laid out left-to-right, top-to-bottom in this order.
ENVIRONMENTS = (
    ("Go1JoystickBandLimited", "Go1JoystickFlatTerrain"),
    ("Go1JoystickRoughTerrainBandLimited", "Go1JoystickRoughTerrain"),
    ("SilverBadgerJoystickBandLimited", "SilverBadgerJoystickFlatTerrain"),
    ("SilverBadgerJoystickRoughTerrainBandLimited", "SilverBadgerJoystickRoughTerrain"),
)
MSSD_METRIC = "smoothness/torque/mssd_mean_squared_second_difference_per_dof"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "evaluations" / "pareto_results"
CACHE_PATH = RESULTS_ROOT / "band_limited_multi_environment_pareto_iqm.csv"
OUTPUT_PATH = RESULTS_ROOT / "band_limited_multi_environment_pareto_iqm.png"
PDF_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".pdf")
IMPROVEMENT_TABLE_PATH = (
    RESULTS_ROOT / "band_limited_multi_environment_pareto_improvements.csv"
)
# Typography for the final two-column figure. Adjust these constants to
# change the plot without affecting the full multi-environment figure.
TITLE_FONT_SIZE = 13.0
LABEL_FONT_SIZE = 13.0
TICK_FONT_SIZE = 10.0
LEGEND_FONT_SIZE = 13.0
LEGEND_COLUMN_SPACING = 0.7
LEGEND_HANDLE_TEXT_PAD = 0.25


def _configure_aggregation() -> None:
  """Reuse the established IQM aggregation with this plot's own cache."""
  multi.ENVIRONMENTS = tuple(name for name, _ in ENVIRONMENTS)
  multi.SMOOTHNESS_METRICS = ((MSSD_METRIC, "Torque MSSD"),)
  multi.CACHE_PATH = CACHE_PATH
  multi.IMPROVEMENT_TABLE_PATH = IMPROVEMENT_TABLE_PATH


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


def _plot(rows: Sequence[dict[str, str]]) -> None:
  figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), squeeze=False)
  method_labels = dict(multi.METHODS)
  legend_handles = {}

  for axis, (environment, title) in zip(axes.flat, ENVIRONMENTS, strict=True):
    environment_rows = [row for row in rows if row["environment"] == environment]
    visible_y = []
    xlim = pareto._configured_xlim(environment, pareto.DEFAULT_XLIM_CONFIG)
    for method, label in multi.METHODS:
      method_rows = [row for row in environment_rows if row["method"] == method]
      if not method_rows:
        continue
      x = np.asarray([float(row[multi.REWARD_METRIC]) for row in method_rows])
      y = np.asarray([float(row[MSSD_METRIC]) for row in method_rows])
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
      scatter = axis.scatter(x[front], y[front], color=color, s=12, label=label)
      order = np.argsort(x[front])
      axis.plot(x[front][order], y[front][order], color=color, linewidth=1.2)
      legend_handles.setdefault(method, scatter)

    axis.set_title(title, fontsize=TITLE_FONT_SIZE)
    axis.set_yscale("log")
    if visible_y:
      log_y = np.log(np.asarray(visible_y, dtype=float))
      lower, upper = float(np.min(log_y)), float(np.max(log_y))
      padding = max(0.04 * (upper - lower), 0.04)
      lower, upper = np.exp(lower - padding), np.exp(upper + padding)
      axis.set_ylim(lower, upper)
      axis.yaxis.set_major_locator(ticker.FixedLocator(np.geomspace(lower, upper, 3)))
      axis.yaxis.set_major_formatter(ticker.FuncFormatter(multi._plain_y_tick_label))
      axis.yaxis.set_minor_locator(ticker.NullLocator())
    environment_x = np.asarray(
        [float(row[multi.REWARD_METRIC]) for row in environment_rows], dtype=float
    )
    if xlim is not None:
      environment_x = environment_x[environment_x >= xlim[0]]
      if xlim[1] is not None:
        environment_x = environment_x[environment_x <= xlim[1]]
    if (
        multi.X_EXPONENTIAL_STRENGTH
        and len(environment_x)
        and np.ptp(environment_x) > 0
    ):
      axis.set_xscale(
          "function",
          functions=pareto._exponential_axis_functions(
              float(np.min(environment_x)), float(np.max(environment_x)),
              multi.X_EXPONENTIAL_STRENGTH,
          ),
      )
    if xlim is not None:
      axis.set_xlim(left=xlim[0], right=xlim[1])
    axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(multi._plain_tick_label))
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    axis.tick_params(axis="y", pad=1.0)
    axis.grid(alpha=0.25)

  for axis in axes[:, 0]:
    axis.set_ylabel("Torque MSSD", fontsize=LABEL_FONT_SIZE)
  # The bottom panel has wider tick labels (notably ``0.95``), so offset its
  # y-label six points right to align it with the upper panel's label.
  axes[1, 0].set_ylabel(
      "Torque MSSD", fontsize=LABEL_FONT_SIZE, labelpad=-2.0
  )
  for axis in axes[-1, :]:
    axis.set_xlabel("Task reward", fontsize=LABEL_FONT_SIZE)
  figure.legend(
      [
          legend_handles[method]
          for method, _ in multi.METHODS
          if method in legend_handles
      ],
      [
          method_labels[method]
          for method, _ in multi.METHODS
          if method in legend_handles
      ],
      loc="lower center", ncol=len(multi.METHODS), frameon=False,
      bbox_to_anchor=(0.5, -0.03), fontsize=LEGEND_FONT_SIZE,
      columnspacing=LEGEND_COLUMN_SPACING,
      handletextpad=LEGEND_HANDLE_TEXT_PAD,
  )
  figure.subplots_adjust(
      left=0.10, right=0.99, top=0.91, bottom=0.15, wspace=0.12, hspace=0.33
  )
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
  figure.savefig(PDF_OUTPUT_PATH, bbox_inches="tight")
  plt.close(figure)
  _crop_pdf(PDF_OUTPUT_PATH)
  print(f"Pareto figures: {OUTPUT_PATH}, {PDF_OUTPUT_PATH}")


def main() -> None:
  _configure_aggregation()
  rows = multi._load_or_build_cache()
  improvement_rows = multi._write_improvement_table(rows)
  _plot(rows)
  multi._print_improvement_table(improvement_rows)
  multi._print_average_improvements(improvement_rows)


if __name__ == "__main__":
  main()
