#!/usr/bin/env python3
"""Plot simulated and real-policy metrics together for SilverBadger RLXHard.

The top row shows the selected simulated policies and the bottom row shows
the task-balanced real-robot suite.  Both MSSD panels use a logarithmic scale
and carry the relative reduction annotations for the high-pass policy.

Example:
  MPLBACKEND=Agg python3 \
    paper/sim_vs_real/plot_simulation_and_real_policy_metrics.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
RESULTS_ROOT = (
    PROJECT_ROOT
    / "evaluations"
    / "pareto_results"
    / "SilverBadgerJoystickFlatTerrainRLXHard"
)
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from learning import plot_silverbadger_rlx_hard_policy_boxplots as simulation


def _load_real_metrics_plotter():
  """Loads the companion real-robot plotter without requiring a package."""
  spec = importlib.util.spec_from_file_location(
      "real_policy_metrics", RESULTS_ROOT / "plot_real_policy_metrics.py"
  )
  if spec is None or spec.loader is None:
    raise ImportError("Could not load the real-policy metrics plotting script.")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


real = _load_real_metrics_plotter()


OUTPUT_STEM = "simulation_and_real_policy_boxplots"
COMMON_POLICY_LABELS = (
    "AR\n$\\lambda=0.4$",
    "AS\n$\\lambda=0.1$",
    "TR\n$\\lambda=0.002$",
    "TFR (ours)\n$\\lambda=0.006$",
)
# Edit these point offsets and arrow paths to manually tune annotations. The
# label offsets control the gap from an arrow's starting point to its percent
# label. The layouts are deliberately separate because linear and logarithmic
# axes need different visual spacing. Values are ordered AR, AS, TR.
ANNOTATION_LAYOUTS = {
    "linear": {
        "simulation": {
            "arcs": (0.32, 0.38, 0.10),
            "label_offsets": ((30, -21), (29, -30), (28, -8)),
        },
        "real": {
            "label_offsets": {"ar": (7, -25), "as": (7, -30), "tr": (10, -3)},
            "lane_scales": (0.70, 0.80, 1.30),
            "annotation_font_size": 14,
        },
    },
    "log": {
        "simulation": {
            "arcs": (0.32, 0.38, 0.10),
            "label_offsets": ((38, -22), (36, -30), (37, -10)),
        },
        "real": {
            "label_offsets": {"ar": (15, -60), "as": (10, -70), "tr": (13, -3)},
            "lane_scales": (0.70, 0.80, 1.30),
            "annotation_font_size": 14,
        },
    },
}


def _simulated_values() -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Loads reward-rate and MSSD rollout distributions for selected policies."""
  reports = [simulation._report_path(choice) for choice in simulation.POLICIES]
  rewards = [
      simulation._reward_per_second_by_rollout(report) for report in reports
  ]
  mssd = [
      simulation._metric_by_rollout(report, simulation.SMOOTHNESS_METRIC)
      for report in reports
  ]
  return (
      [np.asarray(list(values.values())) for values in rewards],
      [np.asarray(list(values.values())) for values in mssd],
  )


def _real_values(metrics_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Loads the task-balanced aggregate distributions from real trials."""
  trials = real.load_trials(metrics_dir, real.DEFAULT_TASKS)
  error = real.balanced_suite_error(trials, real.DEFAULT_TASKS)
  if error is not None:
    raise ValueError(error)
  rewards, mssd = zip(
      *(
          real.values_for(trials, None, policy)
          for policy in real.POLICIES
      ),
      strict=True,
  )
  return list(rewards), list(mssd)


def _plot_simulation_row(
    axes: np.ndarray,
    rewards: list[np.ndarray],
  mssd: list[np.ndarray],
  mssd_log_scale: bool,
) -> None:
  layout = ANNOTATION_LAYOUTS["log" if mssd_log_scale else "linear"]["simulation"]
  simulation._style_boxplot(axes[0], rewards, list(COMMON_POLICY_LABELS))
  axes[0].set_title("Task reward per second", fontsize=simulation.AXIS_LABEL_FONT_SIZE)
  axes[0].set_ylabel("Simulation", fontsize=simulation.AXIS_LABEL_FONT_SIZE)

  simulation._style_boxplot(axes[1], mssd, list(COMMON_POLICY_LABELS))
  axes[1].set_title("Torque MSSD", fontsize=simulation.AXIS_LABEL_FONT_SIZE)
  if mssd_log_scale:
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", which="minor", alpha=0.16, linestyle="-")
  simulation._add_smoothness_improvement_arrows(
      axes[1],
      [
          {str(index): float(value) for index, value in enumerate(values)}
          for values in mssd
      ],
      arcs=list(layout["arcs"]),
      xytext_offsets=list(layout["label_offsets"]),
  )
  for axis in axes:
    axis.tick_params(axis="x", bottom=False, labelbottom=False)


def _plot_real_row(
    axes: np.ndarray,
    rewards: list[np.ndarray],
  mssd: list[np.ndarray],
  mssd_log_scale: bool,
) -> None:
  layout = ANNOTATION_LAYOUTS["log" if mssd_log_scale else "linear"]["real"]
  real.style_boxplot(axes[0], rewards)
  axes[0].set_ylabel("Real world", fontsize=15)

  real.style_boxplot(axes[1], mssd)
  if mssd_log_scale:
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", which="minor", alpha=0.16, linestyle="-")
    axes[1].set_ylim(
        bottom=0.90 * min(float(np.min(values)) for values in mssd),
        top=1.2 * max(float(np.max(values)) for values in mssd),
    )
  real.annotate_hp_reductions(
      axes[1],
      mssd,
      label_offsets=layout["label_offsets"],
      lane_scales=layout["lane_scales"],
      annotation_font_size=layout["annotation_font_size"],
  )
  for axis in axes:
    axis.set_xticks(range(1, len(COMMON_POLICY_LABELS) + 1), COMMON_POLICY_LABELS,
                    fontsize=12)


def _save_figure(
    output: Path,
    simulated_rewards: list[np.ndarray],
    simulated_mssd: list[np.ndarray],
    real_rewards: list[np.ndarray],
    real_mssd: list[np.ndarray],
    mssd_log_scale: bool,
) -> None:
  """Saves one combined figure for the requested MSSD axis scale."""
  #figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
  figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))
  _plot_simulation_row(
      axes[0], simulated_rewards, simulated_mssd, mssd_log_scale
  )
  _plot_real_row(axes[1], real_rewards, real_mssd, mssd_log_scale)
  figure.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.94,
                          hspace=0.10, wspace=0.12)
  figure.savefig(output)
  figure.savefig(output.with_suffix(".png"), dpi=300)
  plt.close(figure)
  simulation._crop_pdf(output)


def _print_medians(
    label: str, rewards: list[np.ndarray], mssd: list[np.ndarray]
) -> None:
  """Prints the central values represented by one row of boxplots."""
  print(f"\n{label} rollout medians:")
  print(f"{'Policy':<12} {'Reward / s':>12} {'Torque MSSD':>14}")
  for policy, reward_values, mssd_values in zip(
      COMMON_POLICY_LABELS, rewards, mssd, strict=True
  ):
    policy_name = policy.splitlines()[0]
    print(
        f"{policy_name:<12} {np.median(reward_values):>12.3f} "
        f"{np.median(mssd_values):>14.3f}"
    )


def _print_tfr_relative_changes(
    label: str, rewards: list[np.ndarray], mssd: list[np.ndarray]
) -> None:
  """Prints TFR's median reward gain and MSSD reduction per baseline."""
  tfr_reward = float(np.median(rewards[-1]))
  tfr_mssd = float(np.median(mssd[-1]))
  print(f"\nTFR (ours) relative to {label.lower()} baselines (medians):")
  print(f"{'Baseline':<12} {'Reward gain':>12} {'MSSD reduction':>16}")
  for policy, reward_values, mssd_values in zip(
      COMMON_POLICY_LABELS[:-1], rewards[:-1], mssd[:-1], strict=True
  ):
    baseline_reward = float(np.median(reward_values))
    baseline_mssd = float(np.median(mssd_values))
    reward_gain = 100.0 * (tfr_reward / baseline_reward - 1.0)
    mssd_reduction = 100.0 * (1.0 - tfr_mssd / baseline_mssd)
    policy_name = policy.splitlines()[0]
    print(
        f"{policy_name:<12} {reward_gain:>+11.1f}% "
        f"{mssd_reduction:>+15.1f}%"
    )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--metrics-dir", type=Path, default=RESULTS_ROOT / "metrics")
  parser.add_argument("--output-dir", type=Path, default=SCRIPT_ROOT)
  args = parser.parse_args()

  simulated_rewards, simulated_mssd = _simulated_values()
  real_rewards, real_mssd = _real_values(args.metrics_dir)
  _print_medians("Simulation", simulated_rewards, simulated_mssd)
  _print_medians("Real world", real_rewards, real_mssd)
  _print_tfr_relative_changes("Simulation", simulated_rewards, simulated_mssd)
  _print_tfr_relative_changes("Real world", real_rewards, real_mssd)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  for mssd_log_scale, scale in ((False, "linear"), (True, "log")):
    output = args.output_dir / f"{OUTPUT_STEM}_{scale}_mssd_annotated.pdf"
    _save_figure(
        output,
        simulated_rewards,
        simulated_mssd,
        real_rewards,
        real_mssd,
        mssd_log_scale,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
  main()
