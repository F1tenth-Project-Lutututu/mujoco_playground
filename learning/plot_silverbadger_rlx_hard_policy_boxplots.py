"""Compare selected SilverBadger RLXHard policies with rollout boxplots.

Example:
  MPLBACKEND=Agg python -m learning.plot_silverbadger_rlx_hard_policy_boxplots
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from learning import pareto_policy_pipeline as pipeline


ENVIRONMENT = "SilverBadgerJoystickFlatTerrainRLXHard"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "evaluations" / "pareto_cluster" / ENVIRONMENT / "raw_torque"
RESULTS_ROOT = PROJECT_ROOT / "evaluations" / "pareto_results" / ENVIRONMENT
OUTPUT_STEM = "selected_policy_boxplots"

TOTAL_REWARD_METRIC = "eval_reward_means/total_without_regularization"
DURATION_METRIC = "episode/duration_seconds"
SMOOTHNESS_METRIC = "smoothness/torque/mssd_mean_squared_second_difference_per_dof"
SIGNIFICANCE_LEVEL = 0.05
AXIS_LABEL_FONT_SIZE = 15
TICK_LABEL_FONT_SIZE = 12
ANNOTATION_FONT_SIZE = 14
FIGURE_WIDTH_INCHES = 8.0
FIGURE_HEIGHT_INCHES = 4.0


@dataclass(frozen=True)
class PolicyChoice:
  """One policy to compare; edit these values to choose another policy."""

  label: str
  method: str
  scale: float
  seed: int


# Edit this tuple to change the plotted policies.
POLICIES = (
    PolicyChoice("AR\n$\\lambda=0.4$", "baseline", 0.4, 0),
    PolicyChoice("AS\n$\\lambda=0.1$", "action_smoothness", 0.1, 0),
    PolicyChoice("TR\n$\\lambda=0.002$", "torque_rate", 0.002, 0),
    PolicyChoice("TFR (ours)\n$\\lambda=0.006$", "high_pass", 0.006, 0),
)
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


def _matching_run(choice: PolicyChoice) -> pipeline.PolicyRun:
  runs = pipeline.select_runs([path.name for path in RAW_ROOT.iterdir() if path.is_dir()])
  matches = [
      run
      for run in runs
      if run.method == choice.method
      and run.seed == choice.seed
      and np.isclose(run.scale, choice.scale, rtol=0.0, atol=1e-12)
  ]
  if len(matches) != 1:
    raise ValueError(
        f"Expected one run for {choice.method}, scale={choice.scale}, "
        f"seed={choice.seed}; found {len(matches)}."
    )
  return matches[0]


def _report_path(choice: PolicyChoice) -> Path:
  run = _matching_run(choice)
  reports = sorted((RAW_ROOT / run.run_name).glob("*/rollouts.csv"))
  if len(reports) != 1:
    raise FileNotFoundError(
        f"Expected one rollout report for {run.run_name}, found {len(reports)}."
    )
  return reports[0]


def _metric_by_rollout(report: Path, metric: str) -> dict[str, float]:
  """Loads finite metric values keyed by the shared random-task rollout ID."""
  values = {}
  with report.open(newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
      try:
        value = float(row[metric])
      except (KeyError, TypeError, ValueError):
        continue
      rollout = row.get("rollout")
      if rollout is not None and np.isfinite(value):
        values[rollout] = value
  if not len(values):
    raise ValueError(f"No finite {metric!r} values in {report}.")
  return values


def _reward_per_second_by_rollout(report: Path) -> dict[str, float]:
  """Loads unregularized task reward divided by simulated rollout duration."""
  values = {}
  with report.open(newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
      try:
        reward = float(row[TOTAL_REWARD_METRIC])
        duration = float(row[DURATION_METRIC])
      except (KeyError, TypeError, ValueError):
        continue
      rollout = row.get("rollout")
      if (
          rollout is not None
          and np.isfinite(reward)
          and np.isfinite(duration)
          and duration > 0.0
      ):
        values[rollout] = reward / duration
  if not len(values):
    raise ValueError(f"No finite reward-per-second values in {report}.")
  return values


def _aligned_values(
    baseline: dict[str, float], proposed: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
  """Returns identically ordered rollout values, requiring exact pairing."""
  if baseline.keys() != proposed.keys():
    raise ValueError(
        "Selected policies do not have identical rollout IDs; paired "
        "significance testing would be invalid."
    )
  rollout_ids = sorted(baseline)
  return (
      np.asarray([baseline[rollout] for rollout in rollout_ids]),
      np.asarray([proposed[rollout] for rollout in rollout_ids]),
  )


def _holm_adjusted_pvalues(pvalues: list[float]) -> list[float]:
  """Returns Holm-Bonferroni adjusted p-values in the original order."""
  adjusted = [0.0] * len(pvalues)
  running_max = 0.0
  for rank, index in enumerate(np.argsort(pvalues)):
    running_max = max(running_max, (len(pvalues) - rank) * pvalues[index])
    adjusted[index] = min(running_max, 1.0)
  return adjusted


def _crop_pdf(path: Path) -> None:
  cropped = path.with_name(f"{path.stem}.cropped.pdf")
  try:
    subprocess.run(["pdfcrop", "--margins", "0", str(path), str(cropped)], check=True)
    cropped.replace(path)
  finally:
    cropped.unlink(missing_ok=True)


def _print_improvement_and_significance_tables(
    rewards: list[dict[str, float]], smoothness: list[dict[str, float]]
) -> None:
  """Prints TFR's median reward gain and MSSD reduction per baseline."""
  proposed_reward = float(np.median(list(rewards[-1].values())))
  proposed_smoothness = float(np.median(list(smoothness[-1].values())))
  print("\nTFR (ours) improvement over selected baselines (rollout medians):")
  print(f"{'Baseline':<12} {'Reward gain':>12} {'MSSD reduction':>16}")
  tests = []
  for choice, reward, smoothness_values in zip(
      POLICIES[:-1], rewards[:-1], smoothness[:-1], strict=True
  ):
    reward_gain = 100.0 * (proposed_reward / np.median(list(reward.values())) - 1.0)
    mssd_reduction = 100.0 * (
        1.0 - proposed_smoothness / np.median(list(smoothness_values.values()))
    )
    baseline_reward, tfr_reward = _aligned_values(reward, rewards[-1])
    baseline_mssd, tfr_mssd = _aligned_values(smoothness_values, smoothness[-1])
    tests.extend((
        (choice.label.splitlines()[0], "Reward", tfr_reward - baseline_reward),
        (choice.label.splitlines()[0], "Torque MSSD", baseline_mssd - tfr_mssd),
    ))
    print(f"{choice.label.splitlines()[0]:<12} {reward_gain:>+11.1f}% {mssd_reduction:>+15.1f}%")

  pvalues = [
      1.0 if np.allclose(difference, 0.0) else float(
          stats.wilcoxon(difference, alternative="greater").pvalue
      )
      for _, _, difference in tests
  ]
  adjusted_pvalues = _holm_adjusted_pvalues(pvalues)
  print("\nPaired one-sided Wilcoxon tests (TFR better; Holm-corrected):")
  print(f"{'Baseline':<12} {'Metric':<14} {'p':>10} {'Holm p':>10} {'Significant':>13}")
  for (baseline, metric, _), pvalue, adjusted in zip(
      tests, pvalues, adjusted_pvalues, strict=True
  ):
    significant = "yes" if adjusted < SIGNIFICANCE_LEVEL else "no"
    print(f"{baseline:<12} {metric:<14} {pvalue:>10.2e} {adjusted:>10.2e} {significant:>13}")


def _style_boxplot(axis: plt.Axes, values: list[np.ndarray], labels: list[str]) -> None:
  """Draws consistently styled policy boxplots."""
  boxes = axis.boxplot(
      values,
      patch_artist=True,
      showfliers=False,
      medianprops={"color": "black", "linewidth": 1.4},
      whiskerprops={"linewidth": 1.1},
      capprops={"linewidth": 1.1},
  )
  for patch, color in zip(boxes["boxes"], COLORS, strict=True):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
  axis.set_xticks(
      range(1, len(labels) + 1), labels, fontsize=TICK_LABEL_FONT_SIZE
  )
  axis.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
  axis.grid(axis="y", alpha=0.25)


def _add_smoothness_improvement_arrows(
    axis: plt.Axes,
    smoothness: list[dict[str, float]],
    *,
    arcs: list[float] | None = None,
    xytext_offsets: list[tuple[float, float]] | None = None,
) -> None:
  """Annotates reductions from each baseline median to the TFR median."""
  tfr_median = float(np.median(list(smoothness[-1].values())))
  arcs = arcs or [0.32, 0.38, 0.1]
  xytext_offsets = xytext_offsets or [(30, -21), (29, -30), (28, -8)]
  if len(arcs) != len(POLICIES) - 1 or len(xytext_offsets) != len(POLICIES) - 1:
    raise ValueError("Specify one arrow and label position for each baseline.")
  for index, (choice, baseline) in enumerate(
      zip(POLICIES[:-1], smoothness[:-1], strict=True), start=1
  ):
    baseline_median = float(np.median(list(baseline.values())))
    reduction = 100.0 * (1.0 - tfr_median / baseline_median)
    color = COLORS[index - 1]
    # Curved trajectories keep the three baseline-to-TFR comparisons distinct.
    arc_radius = arcs[index - 1]
    axis.annotate(
        "",
        xy=(len(POLICIES), tfr_median),
        xytext=(index, baseline_median),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": 1.2,
            "connectionstyle": f"arc3,rad={arc_radius}",
        },
        zorder=4,
    )
    axis.annotate(
        f"↓{reduction:.0f}%",
        xy=(index, baseline_median),
        xytext=xytext_offsets[index - 1],
        textcoords="offset points",
        color=color,
        fontsize=ANNOTATION_FONT_SIZE,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=5,
    )


def _save_two_panel_figure(
    *,
    rewards: list[np.ndarray],
    smoothness: list[np.ndarray],
    smoothness_by_rollout: list[dict[str, float]],
    labels: list[str],
    smoothness_log_scale: bool,
    annotate_smoothness: bool,
    suffix: str,
) -> tuple[Path, Path]:
  """Saves one reward/MSSD two-panel candidate figure."""
  figure, axes = plt.subplots(
      1, 2, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
  )
  _style_boxplot(axes[0], rewards, labels)
  axes[0].set_title("Average task reward per second", fontsize=AXIS_LABEL_FONT_SIZE)

  _style_boxplot(axes[1], smoothness, labels)
  axes[1].set_title("Torque MSSD", fontsize=AXIS_LABEL_FONT_SIZE)
  if smoothness_log_scale:
    axes[1].set_yscale("log")
  if annotate_smoothness:
    _add_smoothness_improvement_arrows(
        axes[1], smoothness_by_rollout
    )

  figure.subplots_adjust(left=0.055, right=0.995, bottom=0.29, top=0.88, wspace=0.17)
  output_path = RESULTS_ROOT / f"{OUTPUT_STEM}_{suffix}.png"
  pdf_output_path = output_path.with_suffix(".pdf")
  figure.savefig(output_path, dpi=300)
  figure.savefig(pdf_output_path)
  plt.close(figure)
  _crop_pdf(pdf_output_path)
  return output_path, pdf_output_path


def main() -> None:
  reports = [_report_path(choice) for choice in POLICIES]
  rewards = [_reward_per_second_by_rollout(report) for report in reports]
  smoothness = [_metric_by_rollout(report, SMOOTHNESS_METRIC) for report in reports]
  labels = [choice.label for choice in POLICIES]
  _print_improvement_and_significance_tables(rewards, smoothness)

  reward_values = [np.asarray(list(values.values())) for values in rewards]
  smoothness_values = [np.asarray(list(values.values())) for values in smoothness]
  RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
  outputs = (
      _save_two_panel_figure(
          rewards=reward_values,
          smoothness=smoothness_values,
          smoothness_by_rollout=smoothness,
          labels=labels,
          smoothness_log_scale=False,
          annotate_smoothness=False,
          suffix="linear_mssd",
      ),
      _save_two_panel_figure(
          rewards=reward_values,
          smoothness=smoothness_values,
          smoothness_by_rollout=smoothness,
          labels=labels,
          smoothness_log_scale=True,
          annotate_smoothness=False,
          suffix="log_mssd",
      ),
      _save_two_panel_figure(
          rewards=reward_values,
          smoothness=smoothness_values,
          smoothness_by_rollout=smoothness,
          labels=labels,
          smoothness_log_scale=True,
          annotate_smoothness=True,
          suffix="log_mssd_annotated",
      ),
      _save_two_panel_figure(
          rewards=reward_values,
          smoothness=smoothness_values,
          smoothness_by_rollout=smoothness,
          labels=labels,
          smoothness_log_scale=False,
          annotate_smoothness=True,
          suffix="linear_mssd_annotated",
      ),
  )
  print("Boxplots:")
  for png_path, pdf_path in outputs:
    print(f"  {png_path}, {pdf_path}")


if __name__ == "__main__":
  main()
