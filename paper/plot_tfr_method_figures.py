#!/usr/bin/env python3
"""
Generate method figures for Torque-Frequency Regularization (TFR).

The theoretical curves reproduce the frequency weighting implemented in
mujoco_playground/_src/locomotion/torque_penalty.py:

  1. causal digital Butterworth high-pass filter,
  2. post-filter finite differences,
  3. cutoff-normalized difference gain,
  4. integer-order finite-difference spectral shaping,
  5. white-spectrum normalization on a 16,384-point frequency grid.

No experimental numbers are fabricated. The synthetic signal figure is
explicitly illustrative. All spectral-response curves are normalized to one
at the selected cutoff, making their unity crossings directly comparable.
White-spectrum normalization is retained as a training feature. Empirical
plots are generated only when the user supplies CSV files.

Examples
--------
Generate all theory/synthetic figures:
    python plot_tfr_method_figures.py --out-dir figures

Use the policy rate from Go1 (dt=0.02 s => fs=50 Hz):
    python plot_tfr_method_figures.py --fs 50 --cutoff 5 --order 1 \
        --difference-order 1 --out-dir figures

Generate empirical Pareto plots from a summary CSV:
    python plot_tfr_method_figures.py --summary-csv results_summary.csv

Generate rollout PSD from a CSV:
    python plot_tfr_method_figures.py --rollout-csv rollout.csv

Expected summary CSV columns
----------------------------
Required:
    method
    tracking_error
    torque_rate_rms
    high_freq_energy_ratio

Recommended:
    penalty_weight
    seed

Expected rollout CSV columns
----------------------------
Required:
    time
    method
    tau_0, tau_1, ..., tau_N

Multiple methods may be concatenated in one CSV. Rows for each method should
form a regularly sampled rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


EPS = 1e-12

# These two panels are intended to be placed side-by-side in one IEEE column.
# Design them at close to their final aspect ratio and use deliberately large
# type, so that shrinking the PDFs to half-column width leaves 6--7 pt text.
SWEEP_FIGSIZE = (5., 5.)
SWEEP_TITLE_SIZE = 14
SWEEP_LABEL_SIZE = 14
SWEEP_TICK_SIZE = 14
SWEEP_LEGEND_SIZE = 14
SWEEP_CURVE_LINEWIDTH = 2.4
SWEEP_GUIDE_LINEWIDTH = 1.8


def _style_compact_sweep(ax: plt.Axes) -> None:
    """Apply IEEE-readable typography to the paired sweep panels."""
    ax.tick_params(axis="both", labelsize=SWEEP_TICK_SIZE)
    ax.xaxis.label.set_size(SWEEP_LABEL_SIZE)
    ax.yaxis.label.set_size(SWEEP_LABEL_SIZE)
    ax.title.set_size(SWEEP_TITLE_SIZE)


def butterworth_highpass_sos(
    cutoff_hz: float,
    order: int,
    sample_rate_hz: float,
) -> np.ndarray:
    """Match scipy.signal.butter(..., btype='highpass', output='sos')."""
    nyquist = 0.5 * sample_rate_hz
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(
            f"cutoff_hz must lie in (0, {nyquist:g}) Hz, got {cutoff_hz}"
        )
    if not 1 <= order <= 8:
        raise ValueError("order must be an integer in [1, 8]")
    return signal.butter(
        order,
        cutoff_hz,
        btype="highpass",
        fs=sample_rate_hz,
        output="sos",
    )


def difference_energy_weight(
    f_hz: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    difference_order: int,
) -> np.ndarray:
    """Frequency-domain weight of the normalized m-th finite difference."""
    if isinstance(difference_order, bool) or int(difference_order) != difference_order:
        raise ValueError("difference_order must be an integer")
    difference_order = int(difference_order)
    if not 0 <= difference_order <= 8:
        raise ValueError("difference_order must be in [0, 8]")

    dt = 1.0 / sample_rate_hz
    gain = 2.0 * np.sin(np.pi * f_hz * dt)
    gain_at_cutoff = 2.0 * np.sin(np.pi * cutoff_hz * dt)
    rho = gain / gain_at_cutoff
    return rho ** (2 * difference_order)


def _raw_tfr_frequency_weight(
    f_hz: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int,
    difference_order: int,
) -> np.ndarray:
    """Unnormalized squared spectral weighting of TFR."""
    sos = butterworth_highpass_sos(cutoff_hz, order, sample_rate_hz)
    _, h = signal.sosfreqz(
        sos,
        worN=2.0 * np.pi * np.asarray(f_hz) / sample_rate_hz,
    )
    weight = np.abs(h) ** 2
    weight *= difference_energy_weight(
        np.asarray(f_hz),
        sample_rate_hz,
        cutoff_hz,
        difference_order,
    )
    return weight


def tfr_white_spectrum_normalizer(
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int,
    difference_order: int,
    grid_size: int = 16_384,
) -> float:
    """Mean TFR weight for a unit-variance white spectrum.

    This mirrors the repository implementation, which evaluates the digital
    frequency response on a dense scipy.signal.sosfreqz grid.
    """
    sos = butterworth_highpass_sos(cutoff_hz, order, sample_rate_hz)
    frequencies, response = signal.sosfreqz(
        sos,
        worN=grid_size,
        fs=sample_rate_hz,
    )
    difference_weight = difference_energy_weight(
        frequencies,
        sample_rate_hz,
        cutoff_hz,
        difference_order,
    )
    normalizer = float(np.mean(np.abs(response) ** 2 * difference_weight))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError(
            f"White-spectrum normalizer must be finite and positive, got {normalizer}"
        )
    return normalizer


def tfr_frequency_weight(
    f_hz: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int,
    difference_order: int,
    normalization: str = "white_spectrum",
) -> np.ndarray:
    """Squared spectral weighting of TFR."""
    weight = _raw_tfr_frequency_weight(
        f_hz,
        sample_rate_hz,
        cutoff_hz,
        order,
        difference_order,
    )

    if normalization in ("white_spectrum", "mean"):
        z = tfr_white_spectrum_normalizer(
            sample_rate_hz,
            cutoff_hz,
            order,
            difference_order,
        )
        weight = weight / z
    elif normalization == "cutoff":
        raw_fc = _raw_tfr_frequency_weight(
            np.asarray([cutoff_hz]),
            sample_rate_hz,
            cutoff_hz,
            order,
            difference_order,
        )[0]
        weight = weight / max(float(raw_fc), EPS)
    elif normalization != "none":
        raise ValueError(
            "normalization must be one of: none, white_spectrum, mean, cutoff"
        )
    return weight


def _raw_finite_difference_weight(
    f_hz: np.ndarray,
    sample_rate_hz: float,
    include_second_difference: bool = False,
) -> np.ndarray:
    """Raw AR/TR or AS/TS spectral weighting."""
    dt = 1.0 / sample_rate_hz
    gain_sq = (2.0 * np.sin(np.pi * np.asarray(f_hz) * dt)) ** 2
    weight = gain_sq.copy()
    if include_second_difference:
        weight += gain_sq**2
    return weight


def finite_difference_weight(
    f_hz: np.ndarray,
    sample_rate_hz: float,
    include_second_difference: bool = False,
    normalization: str = "white_spectrum",
    cutoff_hz: float | None = None,
    grid_size: int = 16_384,
) -> np.ndarray:
    """Frequency weighting of AR/TR, or AS/TS when second difference is added."""
    weight = _raw_finite_difference_weight(
        f_hz,
        sample_rate_hz,
        include_second_difference,
    )

    if normalization in ("white_spectrum", "mean"):
        # Use the same dense [0, Nyquist) grid convention as sosfreqz.
        reference_f = np.arange(grid_size, dtype=float)
        reference_f *= (0.5 * sample_rate_hz) / grid_size
        reference_weight = _raw_finite_difference_weight(
            reference_f,
            sample_rate_hz,
            include_second_difference,
        )
        weight /= max(float(np.mean(reference_weight)), EPS)
    elif normalization == "cutoff":
        if cutoff_hz is None:
            raise ValueError("cutoff_hz is required for cutoff normalization")
        reference = _raw_finite_difference_weight(
            np.asarray([cutoff_hz]),
            sample_rate_hz,
            include_second_difference,
        )[0]
        weight /= max(float(reference), EPS)
    elif normalization != "none":
        raise ValueError(
            "normalization must be one of: none, white_spectrum, mean, cutoff"
        )
    return weight


def frequency_grid(sample_rate_hz: float, n: int = 4096) -> np.ndarray:
    """Avoid exactly zero for log-x visualization."""
    nyquist = 0.5 * sample_rate_hz
    return np.linspace(max(1e-3, nyquist / n), nyquist, n)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()



def plot_pipeline(out_dir: Path) -> None:
    """Conceptual signal-flow diagram for the paper."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.set_axis_off()

    boxes = {
        "policy": (0.04, 0.58, 0.15, 0.18, "Policy\n$a_t$"),
        "target": (0.25, 0.58, 0.17, 0.18, "Desired position\n$q_t^{des}$"),
        "controller": (0.49, 0.58, 0.16, 0.18, "Low-level\ncontroller"),
        "robot": (0.72, 0.58, 0.16, 0.18, "Robot +\ncontacts"),
        "action_reg": (0.20, 0.13, 0.22, 0.18, "AR / AS\n(action space)"),
        "torque_reg": (0.62, 0.13, 0.30, 0.18, "TR / TS or TFR\n(applied-torque space)"),
    }

    for x, y, w, h, label in boxes.values():
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015",
            fill=False,
            linewidth=1.3,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    def arrow(x0, y0, x1, y1, label=None):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.2),
        )
        if label:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + 0.03,
                label,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
            )

    arrow(0.19, 0.67, 0.25, 0.67)
    arrow(0.42, 0.67, 0.49, 0.67)
    arrow(0.65, 0.67, 0.72, 0.67, label=r"$\tau_t$")
    arrow(0.88, 0.67, 0.95, 0.67, label="state")
    arrow(0.95, 0.67, 0.95, 0.88)
    arrow(0.95, 0.88, 0.115, 0.88)
    arrow(0.115, 0.88, 0.115, 0.76)

    # Branches showing where regularization is applied.
    arrow(0.33, 0.58, 0.33, 0.31)
    arrow(0.68, 0.58, 0.76, 0.31)

    ax.text(
        0.77,
        0.04,
        "TFR: capacity normalization $\\rightarrow$ causal HPF "
        "$\\rightarrow$ order-$m$ spectral shaping",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    ax.set_title("Where smoothness regularization acts in the control pipeline")
    savefig(out_dir / "tfr_pipeline.pdf")

def _plot_positive_semilogy(
    f: np.ndarray,
    weight: np.ndarray,
    *,
    label: str,
    decibels: bool = False,
) -> None:
    """Plot a spectral weight on a logarithmic or decibel vertical scale."""
    if decibels:
        plt.plot(f, 10.0 * np.log10(np.maximum(weight, 1e-8)), label=label)
    else:
        plt.semilogy(f, np.maximum(weight, EPS), label=label)


def plot_frequency_comparison(
    out_dir: Path,
    fs: float,
    cutoff: float,
    order: int,
    m: int,
    decibels: bool = False,
) -> None:
    f = frequency_grid(fs)

    tr = finite_difference_weight(
        f, fs, False, normalization="cutoff", cutoff_hz=cutoff
    )
    ts = finite_difference_weight(
        f, fs, True, normalization="cutoff", cutoff_hz=cutoff
    )
    tfr = tfr_frequency_weight(
        f, fs, cutoff, order, m, normalization="cutoff"
    )

    plt.figure(figsize=(6.4, 4.0))
    _plot_positive_semilogy(
        f, tr, label="AR / TR: first difference", decibels=decibels
    )
    _plot_positive_semilogy(
        f, ts, label="AS / TS: first + second difference", decibels=decibels
    )
    _plot_positive_semilogy(
        f,
        tfr,
        label=fr"TFR: $f_c={cutoff:g}$ Hz, $n={order}$, $m={m}$",
        decibels=decibels,
    )
    plt.axvline(
        cutoff,
        linestyle="--",
        linewidth=1.0,
    )
    plt.axhline(0.0 if decibels else 1.0, color="0.35", linestyle=":", linewidth=1.0)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(
        "Penalty weight relative to cutoff [dB]"
        if decibels
        else "Cutoff-normalized penalty weight"
    )
    plt.title("Spectral weighting of smoothness regularizers")
    plt.grid(True, which="both", alpha=0.25)
    if not decibels:
        plt.ylim(bottom=1e-6)
    plt.legend()
    suffix = "_db" if decibels else ""
    savefig(out_dir / f"tfr_frequency_comparison{suffix}.pdf")


def plot_frequency_comparison_emphasized(
    out_dir: Path,
    fs: float,
    cutoff: float,
    order: int,
    m: int,
) -> None:
    """Show the normalized comparison in cutoff-relative decibels.

    All curves cross 0 dB at f / f_c = 1.  Logarithmic frequency and dB
    coordinates expose their distinct attenuation and high-frequency growth.
    """
    f = frequency_grid(fs)
    curves = {
        "AR / TR: first difference": finite_difference_weight(
            f, fs, False, normalization="cutoff", cutoff_hz=cutoff
        ),
        "AS / TS: first + second difference": finite_difference_weight(
            f, fs, True, normalization="cutoff", cutoff_hz=cutoff
        ),
        fr"TFR: $f_c={cutoff:g}$ Hz, $n={order}$, $m={m}$": tfr_frequency_weight(
            f, fs, cutoff, order, m, normalization="cutoff"
        ),
    }

    plt.figure(figsize=(6.4, 4.0))
    for label, weight in curves.items():
        relative_db = 10.0 * np.log10(np.maximum(weight, 1e-8))
        plt.semilogx(f / cutoff, relative_db, label=label)
    plt.axvline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    plt.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    plt.xlabel(r"Normalized frequency $f / f_c$")
    plt.ylabel(r"Penalty weight relative to cutoff [dB]")
    plt.title("Spectral weighting: cutoff-relative comparison")
    plt.grid(True, which="both", alpha=0.25)
    plt.xlim(left=0.01)
    plt.legend()
    savefig(out_dir / "tfr_frequency_comparison_emphasized.pdf")


def plot_cutoff_sweep(
    out_dir: Path,
    fs: float,
    order: int,
    m: int,
    decibels: bool = False,
) -> None:
    f = frequency_grid(fs)
    candidates = [2.0, 5.0, 10.0]
    candidates = [fc for fc in candidates if fc < 0.5 * fs]

    _, ax = plt.subplots(figsize=SWEEP_FIGSIZE)
    for fc in candidates:
        w = tfr_frequency_weight(
            f, fs, fc, order, m, normalization="cutoff"
        )
        y = 10.0 * np.log10(np.maximum(w, 1e-8)) if decibels else np.maximum(w, EPS)
        (curve,) = (ax.plot if decibels else ax.semilogy)(
            f, y, label=fr"$f_c={fc:g}$ Hz", linewidth=SWEEP_CURVE_LINEWIDTH
        )
        ax.axvline(
            fc, color=curve.get_color(), linestyle="--",
            linewidth=SWEEP_GUIDE_LINEWIDTH,
        )
    ax.axhline(
        0.0 if decibels else 1.0,
        color="0.35",
        linestyle=":",
        linewidth=SWEEP_GUIDE_LINEWIDTH,
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(
        "Penalty weight relative to cutoff [dB]"
        if decibels
        else "Cutoff-normalized penalty weight"
    )
    plt.xlim(right=15.)
    ax.set_title(fr"Effect of cutoff frequency ($n={order}$, $m={m}$)")
    _style_compact_sweep(ax)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=SWEEP_LEGEND_SIZE)
    suffix = "_db" if decibels else ""
    savefig(out_dir / f"tfr_cutoff_sweep{suffix}.pdf")


def plot_difference_order_sweep(
    out_dir: Path,
    fs: float,
    cutoff: float,
    order: int,
    decibels: bool = False,
) -> None:
    f = frequency_grid(fs)
    candidates = [0, 1, 2, 3]

    plt.figure(figsize=(6.4, 4.0))
    for m in candidates:
        w = tfr_frequency_weight(
            f, fs, cutoff, order, m, normalization="cutoff"
        )
        _plot_positive_semilogy(f, w, label=fr"$m={m}$", decibels=decibels)
    plt.axvline(cutoff, linestyle="--", linewidth=1.0)
    plt.axhline(0.0 if decibels else 1.0, color="0.35", linestyle=":", linewidth=1.0)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(
        "Penalty weight relative to cutoff [dB]"
        if decibels
        else "Cutoff-normalized penalty weight"
    )
    plt.title(
        fr"Effect of difference order ($f_c={cutoff:g}$ Hz, $n={order}$)"
    )
    plt.grid(True, which="both", alpha=0.25)
    if not decibels:
        plt.ylim(bottom=1e-6)
    plt.legend()
    suffix = "_db" if decibels else ""
    savefig(out_dir / f"tfr_difference_order_sweep{suffix}.pdf")


def plot_filter_order_sweep(
    out_dir: Path,
    fs: float,
    cutoff: float,
    m: int,
    decibels: bool = False,
) -> None:
    f = frequency_grid(fs)
    candidates = [1, 2, 4, 8]

    _, ax = plt.subplots(figsize=SWEEP_FIGSIZE)
    for n in candidates:
        w = tfr_frequency_weight(
            f, fs, cutoff, n, m, normalization="cutoff"
        )
        if decibels:
            ax.plot(
                f,
                10.0 * np.log10(np.maximum(w, 1e-8)),
                label=fr"$n={n}$",
                linewidth=SWEEP_CURVE_LINEWIDTH,
            )
        else:
            ax.semilogy(
                f, np.maximum(w, EPS), label=fr"$n={n}$",
                linewidth=SWEEP_CURVE_LINEWIDTH,
            )
    ax.axvline(
        cutoff, color="black", linestyle="--", linewidth=SWEEP_GUIDE_LINEWIDTH
    )
    ax.axhline(
        0.0 if decibels else 1.0,
        color="0.35",
        linestyle=":",
        linewidth=SWEEP_GUIDE_LINEWIDTH,
    )
    ax.set_xlabel("Frequency [Hz]")
    #plt.ylabel(
    #    "Penalty weight relative to cutoff [dB]"
    #    if decibels
    #    else "Cutoff-normalized penalty weight"
    #)
    ax.set_xlim(right=10.0)
    ax.set_title(
        fr"Effect of Butterworth order ($f_c={cutoff:g}$ Hz, $m={m}$)"
    )
    _style_compact_sweep(ax)
    ax.grid(True, which="both", alpha=0.25)
    if not decibels:
        ax.set_ylim(bottom=1e-10)
    ax.legend(fontsize=SWEEP_LEGEND_SIZE)
    suffix = "_db" if decibels else ""
    savefig(out_dir / f"tfr_filter_order_sweep{suffix}.pdf")


def plot_synthetic_signal(
    out_dir: Path,
    fs: float,
    cutoff: float,
    order: int,
) -> None:
    duration = 2.0
    t = np.arange(0.0, duration, 1.0 / fs)

    slow_hz = min(1.5, 0.25 * cutoff)
    chatter_hz = min(0.8 * (0.5 * fs), max(cutoff * 2.5, cutoff + 1.0))
    slow = np.sin(2.0 * np.pi * slow_hz * t)
    chatter = 0.22 * np.sin(2.0 * np.pi * chatter_hz * t)
    torque = slow + chatter

    sos = butterworth_highpass_sos(cutoff, order, fs)

    # Approximate the online initialization used in the repository:
    # initialize SOS state to the steady state associated with the first sample.
    zi = signal.sosfilt_zi(sos) * torque[0]
    hp, _ = signal.sosfilt(sos, torque, zi=zi)

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(t, torque, label="Synthetic applied torque")
    plt.plot(t, slow, label=f"Slow component ({slow_hz:g} Hz)")
    plt.plot(t, hp, label="Causal high-pass output")
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized torque")
    plt.title("Synthetic illustration of TFR frequency selectivity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(out_dir / "tfr_synthetic_signal.pdf")


def plot_adaptive_weight(out_dir: Path) -> None:
    disturbance = np.linspace(0.0, 2.0, 500)
    w_min = 0.1
    w_max = 1.0

    plt.figure(figsize=(6.4, 4.0))
    for sigma in [0.1, 0.25, 0.5, 1.0]:
        weight = w_min + (w_max - w_min) * np.exp(-disturbance / sigma)
        plt.plot(disturbance, weight, label=fr"$\sigma={sigma:g}$")
    plt.xlabel("Disturbance measure $d$")
    plt.ylabel("Adaptive TFR multiplier $w(d)$")
    plt.title("Optional disturbance-adaptive regularization")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(out_dir / "tfr_adaptive_weight.pdf")


def _aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "method",
        "tracking_error",
        "torque_rate_rms",
        "high_freq_energy_ratio",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV missing columns: {sorted(missing)}")

    group_cols = ["method"]
    if "penalty_weight" in df.columns:
        group_cols.append("penalty_weight")

    numeric_cols = [
        "tracking_error",
        "torque_rate_rms",
        "high_freq_energy_ratio",
    ]
    return df.groupby(group_cols, as_index=False)[numeric_cols].mean()


def plot_pareto(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    agg = _aggregate_summary(df)

    plt.figure(figsize=(6.4, 4.0))
    for method, part in agg.groupby("method"):
        part = part.sort_values("tracking_error")
        plt.plot(
            part["tracking_error"],
            part["high_freq_energy_ratio"],
            marker="o",
            label=str(method),
        )
    plt.xlabel("Command-tracking error")
    plt.ylabel("High-frequency torque-energy ratio")
    plt.title("Tracking vs. high-frequency torque energy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(out_dir / "tfr_pareto_tracking_vs_hf.pdf")

    plt.figure(figsize=(6.4, 4.0))
    for method, part in agg.groupby("method"):
        part = part.sort_values("tracking_error")
        plt.plot(
            part["tracking_error"],
            part["torque_rate_rms"],
            marker="o",
            label=str(method),
        )
    plt.xlabel("Command-tracking error")
    plt.ylabel("Torque-rate RMS")
    plt.title("Tracking vs. torque-rate RMS")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig(out_dir / "tfr_pareto_tracking_vs_torque_rate.pdf")


def _torque_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"tau_\d+", c)]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def plot_rollout_psd(rollout_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(rollout_csv)
    required = {"time", "method"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rollout CSV missing columns: {sorted(missing)}")

    tau_cols = _torque_columns(df)
    if not tau_cols:
        raise ValueError("Rollout CSV needs torque columns tau_0, tau_1, ...")

    plt.figure(figsize=(6.4, 4.0))
    for method, part in df.groupby("method"):
        part = part.sort_values("time")
        t = part["time"].to_numpy()
        if len(t) < 8:
            raise ValueError(f"Not enough samples for method {method!r}")
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt

        psds = []
        freqs = None
        for col in tau_cols:
            f, pxx = signal.welch(
                part[col].to_numpy(),
                fs=fs,
                nperseg=min(256, len(part)),
                detrend="constant",
            )
            freqs = f
            psds.append(pxx)
        mean_psd = np.mean(np.stack(psds, axis=0), axis=0)
        plt.semilogy(freqs, mean_psd + EPS, label=str(method))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Mean torque PSD")
    plt.title("Applied-torque power spectral density")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    savefig(out_dir / "tfr_rollout_psd.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--fs", type=float, default=50.0,
                        help="Control/sample rate in Hz.")
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="TFR cutoff frequency in Hz.")
    parser.add_argument("--order", type=int, default=1,
                        help="Butterworth high-pass order.")
    parser.add_argument("--difference-order", type=int, default=1,
                        help="Integer post-filter difference order m.")
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--rollout-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.fs <= 0:
        raise ValueError("--fs must be positive")
    if not 0 < args.cutoff < 0.5 * args.fs:
        raise ValueError("--cutoff must lie strictly below Nyquist")

    plot_pipeline(args.out_dir)

    plot_frequency_comparison(
        args.out_dir,
        args.fs,
        args.cutoff,
        args.order,
        args.difference_order,
    )
    plot_frequency_comparison(
        args.out_dir,
        args.fs,
        args.cutoff,
        args.order,
        args.difference_order,
        decibels=True,
    )
    plot_frequency_comparison_emphasized(
        args.out_dir,
        args.fs,
        args.cutoff,
        args.order,
        args.difference_order,
    )
    plot_cutoff_sweep(
        args.out_dir,
        args.fs,
        args.order,
        args.difference_order,
    )
    plot_cutoff_sweep(
        args.out_dir,
        args.fs,
        args.order,
        args.difference_order,
        decibels=True,
    )
    plot_difference_order_sweep(
        args.out_dir,
        args.fs,
        args.cutoff,
        args.order,
    )
    plot_difference_order_sweep(
        args.out_dir,
        args.fs,
        args.cutoff,
        args.order,
        decibels=True,
    )
    plot_filter_order_sweep(
        args.out_dir,
        args.fs,
        args.cutoff,
        args.difference_order,
    )
    plot_filter_order_sweep(
        args.out_dir,
        args.fs,
        args.cutoff,
        args.difference_order,
        decibels=True,
    )
    plot_synthetic_signal(
        args.out_dir,
        args.fs,
        args.cutoff,
        args.order,
    )
    plot_adaptive_weight(args.out_dir)

    if args.summary_csv is not None:
        plot_pareto(args.summary_csv, args.out_dir)
    if args.rollout_csv is not None:
        plot_rollout_psd(args.rollout_csv, args.out_dir)


if __name__ == "__main__":
    main()
