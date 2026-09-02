#!/usr/bin/env python3
"""
Front-page figure for Torque-Frequency Regularization (TFR).

Panels
------
(a) Smooth desired-position actions can still produce oscillatory physical torque.
(b) Torque-rate vs frequency-selective TFR spectral weighting.
(c) Experimental Pareto front: task reward (higher is better) vs torque MSSD
    (lower is better), read directly from the experiment-export CSV.

The experiment CSV used by the paper is expected to contain columns such as:
    method, x_metric, x, y_metric, y, pareto

where:
    x_metric = eval_reward_means/total_without_regularization
    y_metric = smoothness/torque/mssd_mean_squared_second_difference_per_dof

Only rows with pareto == True are plotted in panel (c).

Example
-------
python front_page_figure.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import signal


REWARD_METRIC = "eval_reward_means/total_without_regularization"
MSSD_METRIC = "smoothness/torque/mssd_mean_squared_second_difference_per_dof"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROBOT_PATH = SCRIPT_DIR / "robot.png"
DEFAULT_PARETO_CSV = SCRIPT_DIR / "pareto.csv"
DEFAULT_OUTPUT_PREFIX = SCRIPT_DIR / "fig1_tfr"
ROBOT_IMAGE_SIZE = 2.0
ROBOT_BLOCK_CENTER_X = 0.52
#ROBOT_IMAGE_Y = 0.35
ROBOT_IMAGE_Y = 0.28
ROBOT_IMAGE_BASE_WIDTH = 0.62
ROBOT_IMAGE_BASE_HEIGHT = 0.245
ROBOT_IMAGE_CAPTION_SPACING = 0.04
ROBOT_BLOCK_HORIZONTAL_PADDING = -0.15
ROBOT_BLOCK_FONT_PADDING_GROWTH = 0.40
ROBOT_BLOCK_VERTICAL_PADDING = 0.02
LEFT_PANEL_TOP_PLOT_Y = 0.90
LEFT_PANEL_BOTTOM_PLOT_Y = -0.08
LEFT_PANEL_PLOT_HEIGHT = 0.15
LEFT_PANEL_PLOT_CAPTION_SPACING = 0.04
LEFT_PANEL_BOTTOM_ARROW_CAPTION_SPACING = 0.01
LEFT_PANEL_BOTTOM_ARROW_BOX_SPACING = 0.02
LEFT_PANEL_TOP_ARROW_PLOT_SPACING = 0.02
LEFT_PANEL_TOP_ARROW_BOX_SPACING = 0.01
LEFT_PANEL_CONTENT_Y_OFFSET = 0.03
SHOW_GRID = True
FIGURE_FONT_SCALE = 1.4
FIGURE_WIDTH = 7.05
FIGURE_BASE_HEIGHT = 2.90
LEFT_PANEL_WIDTH = 1.20
CENTER_PANEL_WIDTH = 1.04
RIGHT_PANEL_WIDTH = 1.04
LEFT_CENTER_PANEL_SPACING = 0.45
CENTER_RIGHT_PANEL_SPACING = 0.30
FREQUENCY_PANEL_YLABEL_PAD = 1.0
FREQUENCY_PANEL_LEGEND_X = 1.03
FREQUENCY_CUTOFF_LABEL_Y = -0.02
PARETO_PANEL_YLABEL_PAD = 1.0
Y_TICK_LABEL_PAD = 1.0
PANEL_TITLE_Y = 1.28

METHOD_LABELS = {
    "baseline": "Action rate",
    "action_smoothness": "Action smooth.",
    "torque_rate": "Torque rate",
    "torque_smoothness": "Torque smoothness",
    "high_pass": "TFR (ours)",
}

METHOD_ORDER = [
    "baseline",
    "torque_rate",
    "action_smoothness",
    "high_pass",
]

# Fig. 7 color scheme
COLORS = {
    "baseline": "#1f77b4",           # Action rate: blue
    "torque_rate": "#ff7f0e",        # Torque rate: orange
    "high_pass": "#2ca02c",          # TFR: green
    "action_smoothness": "#9467bd",  # Action smoothness: purple
}


def _font(size):
    """Scale a base font size consistently across the complete figure."""
    return size * FIGURE_FONT_SCALE


def _layout_scale():
    """Return a bounded scale used for font-dependent figure geometry."""
    return min(max(FIGURE_FONT_SCALE, 0.75), 1.5)


def _configure_grid(ax):
    """Apply the shared optional grid without implicitly enabling it."""
    if SHOW_GRID:
        ax.grid(True, alpha=0.25, linewidth=0.5)
    else:
        ax.grid(False)


def tfr_weight(f, dt, fc, butter_order=2, diff_order=1):
    """Cutoff-normalized TFR spectral penalty weight."""
    fs = 1.0 / dt
    sos = signal.butter(
        butter_order, fc, btype="highpass", fs=fs, output="sos"
    )
    _, h = signal.sosfreqz(sos, worN=f, fs=fs)
    hp_power = np.abs(h) ** 2

    denom = np.sin(np.pi * fc * dt)
    diff = (np.sin(np.pi * f * dt) / denom) ** (2 * diff_order)
    w = hp_power * diff

    wc = np.interp(fc, f, w)
    return w / max(wc, 1e-12)


def torque_rate_weight(f, dt, fc):
    """Cutoff-normalized first-difference / torque-rate penalty."""
    w = 4.0 * np.sin(np.pi * f * dt) ** 2
    wc = 4.0 * np.sin(np.pi * fc * dt) ** 2
    return w / max(wc, 1e-12)


def _as_bool(series):
    """Robust conversion of CSV booleans."""
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def load_pareto_export(path):
    """
    Load the experiment-export CSV and return the MSSD-vs-reward Pareto rows.

    The CSV stores reward in column x and MSSD in column y.  We retain only:
      * requested reward metric,
      * requested torque MSSD metric,
      * rows flagged as Pareto-optimal.
    """
    df = pd.read_csv(path)

    required = {"method", "x_metric", "x", "y_metric", "y", "pareto"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Experiment CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df[
        (df["x_metric"] == REWARD_METRIC)
        & (df["y_metric"] == MSSD_METRIC)
    ].copy()

    if df.empty:
        raise ValueError(
            "No rows found for the requested task-reward / torque-MSSD metrics."
        )

    df["pareto"] = _as_bool(df["pareto"])
    df = df[df["pareto"]].copy()

    if df.empty:
        raise ValueError("No Pareto-optimal rows remain after filtering.")

    # Keep the original method identities but create paper-facing labels.
    df["label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])

    # Sorting by reward gives the same left-to-right line presentation used
    # for the Pareto curves in the paper-style plots.
    df = df.sort_values(["method", "x", "y"]).reset_index(drop=True)
    return df


def add_pipeline_panel(ax, robot_path=None):
    ax.set_axis_off()
    ax.set_title(
        r"(a) Smooth actions $\ne$ smooth torque",
        fontsize=_font(8.2),
        y=PANEL_TITLE_Y,
        pad=0,
        verticalalignment="top",
    )

    image_width = ROBOT_IMAGE_BASE_WIDTH * ROBOT_IMAGE_SIZE
    image_height = ROBOT_IMAGE_BASE_HEIGHT * ROBOT_IMAGE_SIZE
    image_x = ROBOT_BLOCK_CENTER_X - 0.5 * image_width
    top_plot_y = LEFT_PANEL_TOP_PLOT_Y + LEFT_PANEL_CONTENT_Y_OFFSET
    bottom_plot_y = LEFT_PANEL_BOTTOM_PLOT_Y + LEFT_PANEL_CONTENT_Y_OFFSET
    robot_image_y = ROBOT_IMAGE_Y + LEFT_PANEL_CONTENT_Y_OFFSET

    # Match both trace widths and centers to the scaled robot image.
    top = ax.inset_axes([
        image_x,
        top_plot_y,
        image_width,
        LEFT_PANEL_PLOT_HEIGHT,
    ])
    bottom = ax.inset_axes([
        image_x,
        bottom_plot_y,
        image_width,
        LEFT_PANEL_PLOT_HEIGHT,
    ])
    top_caption_y = (
        top_plot_y
        + LEFT_PANEL_PLOT_HEIGHT
        + LEFT_PANEL_PLOT_CAPTION_SPACING
    )
    bottom_caption_y = (
        bottom_plot_y
        + LEFT_PANEL_PLOT_HEIGHT
        + LEFT_PANEL_PLOT_CAPTION_SPACING
    )

    t = np.linspace(0, 0.8, 400)
    action = (
        0.75 * np.sin(2 * np.pi * 1.4 * t)
        + 0.15 * np.sin(2 * np.pi * 0.35 * t)
    )
    torque = (
        0.80 * np.sin(2 * np.pi * 2.5 * t)
        + 0.24 * np.sin(2 * np.pi * 15.0 * t)
        + 0.08 * np.sin(2 * np.pi * 22.0 * t)
    )

    for a in (top, bottom):
        a.set_yticks([])
        a.set_xlim(0.0, 0.8)
        a.tick_params(
            axis="x",
            labelsize=_font(5.9),
            pad=_font(1.0),
            length=_font(2.3),
        )
        _configure_grid(a)
        a.spines[["top", "right"]].set_visible(False)

    # Both traces retain a visible time axis.  The upper trace uses only the
    # endpoints so the signal-flow arrow has clear space at the center.
    top.set_xticks([0.0, 0.8])
    bottom.set_xticks([0.0, 0.4, 0.8])

    top.plot(t, action, lw=1.6)
    top.set_ylabel(
        r"$q^{des}$", rotation=0, labelpad=_font(12), fontsize=_font(8)
    )

    bottom.plot(t, torque, lw=1.25)
    bottom.set_ylabel(
        r"$\tau$", rotation=0, labelpad=_font(12), fontsize=_font(8)
    )
    bottom.set_xlabel(
        "Time [s]", fontsize=_font(6.8), labelpad=_font(1.0)
    )

    ax.text(
        ROBOT_BLOCK_CENTER_X,
        top_caption_y,
        "desired joint position",
        ha="center", va="center",
        transform=ax.transAxes, fontsize=_font(7.4),
    )

    # Size the physical-system block around the centered robot and caption.
    caption_y = robot_image_y - ROBOT_IMAGE_CAPTION_SPACING
    effective_horizontal_padding = (
        ROBOT_BLOCK_HORIZONTAL_PADDING
        + ROBOT_BLOCK_FONT_PADDING_GROWTH * (FIGURE_FONT_SCALE - 1.0)
    )
    box_w = image_width + 2 * effective_horizontal_padding
    box_x = ROBOT_BLOCK_CENTER_X - 0.5 * box_w
    box_y = caption_y - ROBOT_BLOCK_VERTICAL_PADDING
    box_top = (
        robot_image_y + image_height + ROBOT_BLOCK_VERTICAL_PADDING
    )
    box_h = box_top - box_y
    block = FancyBboxPatch(
        (box_x, box_y),
        box_w,
        box_h,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        transform=ax.transAxes,
        facecolor=COLORS["baseline"],
        edgecolor=COLORS["baseline"],
        alpha=0.10,
        linewidth=1.0,
        zorder=0,
        clip_on=False,
    )
    ax.add_patch(block)

    # Robot image is the main visual element of the transformation block.
    if robot_path is not None:
        img = plt.imread(robot_path)
        photo = ax.inset_axes(
            [
                image_x,
                robot_image_y,
                image_width,
                image_height,
            ],
            zorder=2,
        )
        photo.imshow(img)
        photo.set_xticks([])
        photo.set_yticks([])
        for spine in photo.spines.values():
            spine.set_visible(False)

    ax.text(
        ROBOT_BLOCK_CENTER_X, caption_y,
        "PD controller + robot dynamics",
        ha="center", va="center",
        transform=ax.transAxes, fontsize=_font(7.0),
        zorder=3,
    )

    # Arrows live entirely in the blank spaces between plots and the box.
    ax.annotate(
        "",
        xy=(
            ROBOT_BLOCK_CENTER_X,
            box_top + LEFT_PANEL_TOP_ARROW_BOX_SPACING,
        ),
        xytext=(
            ROBOT_BLOCK_CENTER_X,
            top_plot_y - LEFT_PANEL_TOP_ARROW_PLOT_SPACING,
        ),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0),
        annotation_clip=False,
    )

    ax.annotate(
        "",
        xy=(
            ROBOT_BLOCK_CENTER_X,
            bottom_caption_y + LEFT_PANEL_BOTTOM_ARROW_CAPTION_SPACING,
        ),
        xytext=(
            ROBOT_BLOCK_CENTER_X,
            box_y - LEFT_PANEL_BOTTOM_ARROW_BOX_SPACING,
        ),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0),
        annotation_clip=False,
    )

    ax.text(
        ROBOT_BLOCK_CENTER_X,
        bottom_caption_y,
        "applied actuator torque",
        ha="center", va="center",
        transform=ax.transAxes, fontsize=_font(7.0),
    )


def add_frequency_panel(ax, dt, fc, butter_order, diff_order):
    ax.set_title(
        "(b) Frequency-selective\nregularization",
        fontsize=_font(8.2),
        y=PANEL_TITLE_Y,
        pad=0,
        verticalalignment="top",
    )

    nyquist = 0.5 / dt
    f = np.linspace(1e-3, nyquist, 1400)
    w_tr = torque_rate_weight(f, dt, fc)
    w_tfr = tfr_weight(f, dt, fc, butter_order, diff_order)

    ax.semilogy(f, w_tr, lw=1.55, ls="--", color=COLORS["torque_rate"], label="Torque rate")
    ax.semilogy(f, w_tfr, lw=1.75, color=COLORS["high_pass"], label="TFR (ours)")

    ax.axvline(fc, lw=0.95, ls=":", color=COLORS["high_pass"])
    # The preserved gait band is encoded with the same green theme as TFR,
    # but at low opacity so the response curves remain dominant.
    ax.axvspan(
        0, fc,
        color=COLORS["high_pass"],
        alpha=0.075,
        zorder=0,
    )

    # Preserve-gait message outside the axes, with an arrow into the shaded
    # low-frequency region.
    ax.annotate(
        "preserve gait band",
        xy=(0.10, 0.78),
        xycoords="axes fraction",
        xytext=(0.15, 1.035),
        #xytext=(0.15, 0.88),
        textcoords="axes fraction",
        ha="center", va="bottom",
        fontsize=_font(6.5),
        color=COLORS["high_pass"],
        arrowprops=dict(
            arrowstyle="->",
            lw=0.8,
            color=COLORS["high_pass"],
        ),
        annotation_clip=False,
    )

    # Keep this message inside the plot, but place it in otherwise empty
    # lower-right space and point toward the high-frequency TFR response.
    ax.annotate(
        "suppress\nhigh-frequency\ntorque oscillations",
        xy=(0.77, 0.75),
        xycoords="axes fraction",
        xytext=(0.62, 0.71),
        textcoords="axes fraction",
        ha="center", va="center",
        fontsize=_font(6.5),
        color=COLORS["high_pass"],
        #arrowprops=dict(
        #    arrowstyle="->",
        #    lw=0.8,
        #    color=COLORS["high_pass"],
        #),
    )

    ax.text(
        fc,
        FREQUENCY_CUTOFF_LABEL_Y,
        r"$f_c$",
        color=COLORS["high_pass"],
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=_font(8),
        clip_on=False,
    )

    ax.set_xlim(0, nyquist)
    ax.set_ylim(1e-4, max(40, np.nanmax(w_tfr) * 1.15))
    ax.set_xlabel("Frequency [Hz]", fontsize=_font(7.5))
    ax.set_ylabel(
        "Penalty weight",
        fontsize=_font(7.5),
        labelpad=_font(FREQUENCY_PANEL_YLABEL_PAD),
    )
    ax.tick_params(labelsize=_font(6.8))
    ax.tick_params(axis="y", pad=_font(Y_TICK_LABEL_PAD))
    _configure_grid(ax)
    ax.legend(
        frameon=False,
        fontsize=_font(6.7),
        loc="lower right",
        bbox_to_anchor=(FREQUENCY_PANEL_LEGEND_X, 0.0),
        borderaxespad=0.0,
    )


def add_pareto_panel(ax, pareto_df, robot_path=None):
    ax.set_title(
        "(c) Performance–smoothness\ntrade-off",
        fontsize=_font(8.2),
        y=PANEL_TITLE_Y,
        pad=0,
        verticalalignment="top",
    )

    # Fig.-7-like presentation: solid Pareto curves with compact point markers,
    # logarithmic smoothness axis, and no artificial normalization.
    plotted = []
    for method in METHOD_ORDER:
        g = pareto_df[pareto_df["method"] == method].sort_values("x")
        if g.empty:
            continue

        label = METHOD_LABELS.get(method, method)
        ax.plot(
            g["x"].to_numpy(),
            g["y"].to_numpy(),
            "-o",
            lw=1.25,
            ms=3.2,
            color=COLORS[method],
            label=label,
        )
        plotted.append(g)

    all_data = pd.concat(plotted, ignore_index=True)
    xmin, xmax = all_data["x"].min(), all_data["x"].max()
    ymin, ymax = all_data["y"].min(), all_data["y"].max()

    xpad = 0.05 * max(xmax - xmin, 1e-6)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_yscale("log")
    ax.set_ylim(ymin / 1.25, ymax * 1.18)

    ax.set_xlabel("Task reward  ↑", fontsize=_font(7.5))
    ax.set_ylabel(
        "Torque MSSD  ↓",
        fontsize=_font(7.5),
        labelpad=_font(PARETO_PANEL_YLABEL_PAD),
    )
    ax.tick_params(labelsize=_font(6.8))
    ax.tick_params(axis="y", pad=_font(Y_TICK_LABEL_PAD))
    _configure_grid(ax)

    # Place legend where the Pareto curves leave the most whitespace.
    ax.legend(
        frameon=False,
        fontsize=_font(6.5),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=1.8,
        columnspacing=0.65,
        labelspacing=0.25,
        borderaxespad=0.0,
    )

    # The desirable direction is lower-right.
    ax.annotate(
        "better",
        xy=(0.95, 0.07), xytext=(0.83, 0.16),
        xycoords="axes fraction", textcoords="axes fraction",
        fontsize=_font(6.5),
        ha="center",
        arrowprops=dict(arrowstyle="->", lw=0.85),
    )




def build_figure(
    robot_path=None,
    pareto_csv=None,
    dt=0.02,
    fc=5.0,
    butter_order=2,
    diff_order=1,
):
    if pareto_csv is None:
        raise ValueError(
            "Panel (c) now requires --pareto-csv with the experimental export."
        )

    pareto_df = load_pareto_export(pareto_csv)

    plt.rcParams.update({
        "font.size": _font(7.5),
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    layout_scale = _layout_scale()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_BASE_HEIGHT * layout_scale)
    )
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[
            LEFT_PANEL_WIDTH,
            LEFT_CENTER_PANEL_SPACING * layout_scale,
            CENTER_PANEL_WIDTH,
            CENTER_RIGHT_PANEL_SPACING * layout_scale,
            RIGHT_PANEL_WIDTH,
        ],
        wspace=0.0,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[0, 4])

    add_pipeline_panel(ax0, robot_path=robot_path)
    add_frequency_panel(ax1, dt, fc, butter_order, diff_order)
    add_pareto_panel(ax2, pareto_df, robot_path=None)

    # Slightly more top margin is needed because panel (b)'s frequency-region
    # labels now live above the axes rather than on top of the data.
    fig.subplots_adjust(
        left=0.045, right=0.995,
        top=0.76 - 0.025 * (layout_scale - 1.0),
        bottom=0.16 + 0.02 * (layout_scale - 1.0),
    )
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--robot", type=Path, default=DEFAULT_ROBOT_PATH,
        help=f"Robot photo shown in panel (a) (default: {DEFAULT_ROBOT_PATH}).",
    )
    p.add_argument(
        "--pareto-csv", type=Path, default=DEFAULT_PARETO_CSV,
        help=(
            "Experiment-export CSV containing reward/MSSD Pareto rows "
            f"(default: {DEFAULT_PARETO_CSV})."
        ),
    )
    p.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX,
        help=f"Output filename prefix (default: {DEFAULT_OUTPUT_PREFIX}).",
    )
    p.add_argument(
        "--dt", type=float, default=0.02,
        help="Policy control period [s].",
    )
    p.add_argument(
        "--fc", type=float, default=5.0,
        help="TFR cutoff frequency [Hz].",
    )
    p.add_argument("--butter-order", type=int, default=2)
    p.add_argument("--diff-order", type=int, default=1)
    args = p.parse_args()

    fig = build_figure(
        robot_path=args.robot,
        pareto_csv=args.pareto_csv,
        dt=args.dt,
        fc=args.fc,
        butter_order=args.butter_order,
        diff_order=args.diff_order,
    )

    prefix = args.out_prefix
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".png"), dpi=400)
    plt.close(fig)

    print(f"Saved: {prefix.with_suffix('.pdf')}")
    print(f"Saved: {prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
