#!/usr/bin/env python3
"""Plot approximate reward and measured-torque MSSD from real policy trials.

The default suite contains straight, three freestyle, and wiggly command trajectories.
It creates one boxplot per task plus one task-balanced aggregate plot.

Example:
  MPLBACKEND=Agg python3 plot_real_policy_metrics.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath


POLICIES = ("ar", "as", "tr", "hp")
LABELS = ("AR", "AS", "TR", "HP (ours)")
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
# Manual label placement, in points relative to each baseline boxplot median.
# Adjust each (horizontal, vertical) pair independently as needed.
IMPROVEMENT_LABEL_OFFSETS = {
    "ar": (7, -60),
    "as": (7, -70),
    "tr": (10, -5),
}
REWARD_KEY = "reward/total"
DURATION_KEY = "duration_seconds"
MSSD_PER_DOF_KEY = "smoothness/measured_torque/mssd_mean_squared_second_difference_per_dof"
TORQUE_DOFS = 13
DEFAULT_TASKS = (
    "straight_01",
    "freestyle_01",
    "freestyle_02",
    "freestyle_03",
    "wiggly_01",
)
FILENAME = re.compile(r"^(?P<task>.+)_(?P<policy>ar|as|tr|hp)_(?P<repeat>\d+)_metrics\.json$")


@dataclass(frozen=True)
class Trial:
    task: str
    policy: str
    reward_per_second: float
    mssd: float


def load_trials(metrics_dir: Path, tasks: tuple[str, ...]) -> list[Trial]:
    trials = []
    for path in sorted(metrics_dir.rglob("*_metrics.json")):
        match = FILENAME.match(path.name)
        if match is None or match["task"] not in tasks:
            continue
        values = json.loads(path.read_text(encoding="utf-8"))
        try:
            reward = float(values[REWARD_KEY])
            duration = float(values[DURATION_KEY])
            # The saved metric is averaged across the robot's 13 torque DoFs.
            # Convert it to the total MSSD reported by this plot.
            mssd = TORQUE_DOFS * float(values[MSSD_PER_DOF_KEY])
        except KeyError as error:
            raise KeyError(f"{path} does not contain {error.args[0]!r}.") from error
        if not np.isfinite(reward) or not np.isfinite(duration) or duration <= 0 or not np.isfinite(mssd):
            raise ValueError(f"Invalid reward, duration, or MSSD in {path}.")
        trials.append(Trial(match["task"], match["policy"], reward / duration, mssd))
    if not trials:
        raise FileNotFoundError(f"No matching metrics files found under {metrics_dir}.")
    return trials


def values_for(trials: list[Trial], task: str | None, policy: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [trial for trial in trials if trial.policy == policy and (task is None or trial.task == task)]
    if not selected:
        location = "aggregate suite" if task is None else task
        raise ValueError(f"No {policy!r} trials for {location}.")
    return (
        np.asarray([trial.reward_per_second for trial in selected]),
        np.asarray([trial.mssd for trial in selected]),
    )


def style_boxplot(axis: plt.Axes, values: list[np.ndarray]) -> None:
    boxes = axis.boxplot(values, patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "linewidth": 1.4},
                         whiskerprops={"linewidth": 1.1}, capprops={"linewidth": 1.1})
    for patch, color in zip(boxes["boxes"], COLORS, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axis.set_xticks(range(1, len(LABELS) + 1), LABELS, fontsize=12)
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(axis="y", alpha=0.25)


def annotate_hp_reductions(
    axis: plt.Axes,
    mssd: list[np.ndarray],
    *,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    lane_scales: tuple[float, float, float] = (0.70, 0.80, 1.3),
    annotation_font_size: float = 11,
) -> None:
    hp_median = float(np.median(mssd[-1]))
    # Reserve three separate lanes beneath the boxes.  The order is deliberate:
    # blue is lowest, followed by orange, then green.
    minimum = min(float(np.min(values)) for values in mssd)
    if len(lane_scales) != len(POLICIES) - 1:
        raise ValueError("Specify one arrow lane for each baseline.")
    label_offsets = label_offsets or IMPROVEMENT_LABEL_OFFSETS
    arrow_lanes = tuple(scale * minimum for scale in lane_scales)
    for index, baseline in enumerate(mssd[:-1], start=1):
        baseline_median = float(np.median(baseline))
        reduction = 100.0 * (1.0 - hp_median / baseline_median)
        start_x = float(index)
        end_x = float(len(POLICIES))
        lane = arrow_lanes[index - 1]
        # A cubic Bézier path keeps the arrows visually smooth while its two
        # control points hold each colour in its assigned below-box lane.
        path = MplPath([(start_x, baseline_median), (start_x, lane),
                        (end_x - 0.35, hp_median), (end_x, hp_median)],
                       [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        axis.add_patch(FancyArrowPatch(path=path, transform=axis.transData, arrowstyle="-|>",
                                       mutation_scale=12, color=COLORS[index - 1], linewidth=1.2, zorder=4))
        axis.annotate(f"↓{reduction:.0f}%", xy=(start_x, baseline_median),
                      xytext=label_offsets[POLICIES[index - 1]],
                      textcoords="offset points", color=COLORS[index - 1],
                      fontsize=annotation_font_size,
                      fontweight="bold", ha="left", va="bottom", zorder=5)


def crop_pdf_to_content(output: Path) -> None:
    """Replace *output* with a zero-margin pdfcrop version."""
    cropped = output.with_name(f"{output.stem}.cropped.pdf")
    try:
        subprocess.run(["pdfcrop", "--margins", "0", str(output), str(cropped)], check=True)
        cropped.replace(output)
    except FileNotFoundError as error:
        raise RuntimeError("pdfcrop is required to crop generated PDF plots.") from error
    finally:
        if cropped.exists():
            cropped.unlink()


def save_figure(output: Path, title: str, rewards: list[np.ndarray], mssd: list[np.ndarray],
                reward_ylabel: str = "Task reward* per second") -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
    style_boxplot(axes[0], rewards)
    #axes[0].set_title(f"{title}: approximate reward", fontsize=15)
    #axes[0].set_ylabel(reward_ylabel, fontsize=12)

    style_boxplot(axes[1], mssd)
    #axes[1].set_title(f"{title}: measured torque MSSD", fontsize=15)
    #axes[1].set_ylabel("Torque MSSD", fontsize=12)
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.90 * min(float(np.min(values)) for values in mssd),
                     top=1.2 * max(float(np.max(values)) for values in mssd))
    annotate_hp_reductions(axes[1], mssd)

    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.88, wspace=0.35)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)
    crop_pdf_to_content(output)


def balanced_suite_error(trials: list[Trial], tasks: tuple[str, ...]) -> str | None:
    counts = {(task, policy): sum(trial.task == task and trial.policy == policy for trial in trials)
              for task in tasks for policy in POLICIES}
    unique_counts = set(counts.values())
    if 0 in unique_counts or len(unique_counts) != 1:
        details = ", ".join(f"{task}/{policy}={count}" for (task, policy), count in counts.items())
        return ("Cannot make a task-balanced aggregate: each task/policy pair needs the same "
                f"non-zero number of repetitions ({details}).")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    args = parser.parse_args()
    tasks = tuple(args.tasks)
    output_dir = args.output_dir or args.metrics_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials(args.metrics_dir, tasks)

    #for task in tasks:
    #    rewards, mssd = zip(*(values_for(trials, task, policy) for policy in POLICIES), strict=True)
    #    output = output_dir / f"real_policy_{task}_boxplots_linear_mssd_annotated.pdf"
    #    save_figure(output, task.replace("_", " "), list(rewards), list(mssd))
    #    print(f"Wrote {output}")

    error = balanced_suite_error(trials, tasks)
    if error is not None:
        print(f"SKIP aggregate plot: {error}")
        return
    rewards, mssd = zip(*(values_for(trials, None, policy) for policy in POLICIES), strict=True)
    output = output_dir / "real_policy_task_suite_boxplots_linear_mssd_annotated.pdf"
    save_figure(output, "task suite", list(rewards), list(mssd))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
