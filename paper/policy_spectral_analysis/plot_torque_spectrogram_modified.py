#!/usr/bin/env python3
"""Plot mean locomotion-policy torque spectra and normalized penalties.

Both reported costs use white-spectrum normalization: the mean squared
frequency weight over [0, Nyquist] is one. Consequently unit-variance white
torque has the same expected cost under torque rate and under TFR, independent
of how each penalty redistributes sensitivity across frequency.

When ``feet_contact`` is present in the saved rollout signals, the script also
estimates the dominant per-foot contact-cycle frequency and overlays it on the
torque PSD. This provides a direct check that the low-frequency torque peak
coincides with structured locomotion rather than an arbitrary oscillation.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import signal as scipy_signal

CUTOFF_HZ = 5.0
FILTER_ORDER = 1
DIFFERENCE_ORDER = 1
NORMALIZER_GRID_SIZE = 16_384
FIGURE_SIZE = (8.4, 3.8)
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 12
DEFAULT_GAIT_MIN_FREQUENCY_HZ = 0.5
DEFAULT_GAIT_MAX_FREQUENCY_HZ = 6.0
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "figures" / "locomotion_policy_torque_spectra.pdf"
)
DEFAULT_INPUTS = (
    Path(__file__).resolve().parent / "data" / "go1_default",
    Path(__file__).resolve().parent / "data" / "spot_default",
    Path(__file__).resolve().parent / "data" / "sb_default",
)
DEFAULT_LABELS = ("Go1", "Spot", "Silver Badger")


def _resolve_signals(path: Path) -> Path:
    for candidate in (
        path,
        path / "signals.npz",
        path / "random_tasks" / "signals.npz",
    ):
        if candidate.is_file() and candidate.name == "signals.npz":
            return candidate
    raise FileNotFoundError(f"Could not find signals.npz under {path}")


def _sample_period(signals_path: Path, override: float | None) -> float:
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise ValueError("--sample-period must be positive and finite.")
        return override
    for parent in signals_path.parents:
        summary_path = parent / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            sample_period = summary.get("metadata", {}).get("sample_period_seconds")
            if sample_period is not None:
                return float(sample_period)
    raise ValueError("No sample period metadata found; pass --sample-period.")


def _frequency_normalizers(sample_rate_hz: float) -> tuple[float, float]:
    """Return white-spectrum normalizers for TR and TFR(f=5,n=1,m=1)."""
    frequencies = np.arange(NORMALIZER_GRID_SIZE, dtype=np.float64)
    frequencies *= (0.5 * sample_rate_hz) / NORMALIZER_GRID_SIZE
    difference_gain = 2.0 * np.sin(np.pi * frequencies / sample_rate_hz)
    torque_rate = float(np.mean(difference_gain**2))

    sos = scipy_signal.butter(
        FILTER_ORDER, CUTOFF_HZ, btype="highpass", fs=sample_rate_hz, output="sos"
    )
    _, response = scipy_signal.sosfreqz(
        sos, worN=NORMALIZER_GRID_SIZE, fs=sample_rate_hz
    )
    gain_at_cutoff = 2.0 * np.sin(np.pi * CUTOFF_HZ / sample_rate_hz)
    tfr_weight = np.abs(response) ** 2 * (difference_gain / gain_at_cutoff) ** 2
    return torque_rate, float(np.mean(tfr_weight))


def normalized_penalties(
    torque: np.ndarray, active: np.ndarray, sample_period: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-task, per-step normalized TR and TFR costs (summed over DOFs)."""
    sample_rate = 1.0 / sample_period
    tr_normalizer, tfr_normalizer = _frequency_normalizers(sample_rate)
    sos = scipy_signal.butter(
        FILTER_ORDER, CUTOFF_HZ, btype="highpass", fs=sample_rate, output="sos"
    )
    torque_rate_costs = []
    tfr_costs = []
    for task in range(torque.shape[1]):
        values = np.asarray(torque[active[:, task], task], dtype=np.float64)
        if not len(values):
            continue
        # Match the environment's steady-state initialization and zero first
        # difference, avoiding an artificial episode-boundary impulse.
        zi = scipy_signal.sosfilt_zi(sos)[:, :, None] * values[0][None, None, :]
        highpass, _ = scipy_signal.sosfilt(sos, values, axis=0, zi=zi)
        torque_delta = np.diff(values, axis=0, prepend=values[:1])
        cutoff_delta_scale = 1.0 / (2.0 * np.sin(np.pi * CUTOFF_HZ / sample_rate))
        highpass_delta = np.diff(highpass, axis=0, prepend=highpass[:1])
        torque_rate_costs.append(
            np.mean(np.sum(torque_delta**2, axis=-1)) / tr_normalizer
        )
        tfr_costs.append(
            np.mean(np.sum((highpass_delta * cutoff_delta_scale) ** 2, axis=-1))
            / tfr_normalizer
        )
    return np.asarray(torque_rate_costs), np.asarray(tfr_costs)


def mean_spectrogram(
    torque: np.ndarray,
    active: np.ndarray,
    sample_period: float,
    window_length: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window_length < 2 or window_length > torque.shape[0]:
        raise ValueError("--window-length must be in [2, trajectory length].")
    if not 0 <= overlap < 1:
        raise ValueError("--overlap must be in [0, 1).")
    step = max(1, round(window_length * (1.0 - overlap)))
    starts = np.arange(0, torque.shape[0] - window_length + 1, step)
    window = np.hanning(window_length)
    scale = (1.0 / sample_period) * np.sum(window**2)
    frequencies = np.fft.rfftfreq(window_length, sample_period)
    power_sum = np.zeros((len(frequencies), len(starts)))
    counts = np.zeros(len(starts), dtype=int)
    for column, start in enumerate(starts):
        valid_tasks = np.all(active[start : start + window_length], axis=0)
        segment = torque[start : start + window_length, valid_tasks]
        if not segment.size:
            continue
        segment = segment - np.mean(segment, axis=0, keepdims=True)
        spectrum = np.fft.rfft(segment * window[:, None, None], axis=0)
        power = np.abs(spectrum) ** 2 / scale
        power[1 : (-1 if window_length % 2 == 0 else None)] *= 2.0
        power_sum[:, column] = np.sum(power, axis=(1, 2))
        counts[column] = segment.shape[1] * segment.shape[2]
    power = np.full_like(power_sum, np.nan)
    np.divide(power_sum, counts[None], out=power, where=counts[None] > 0)
    times = (starts + (window_length - 1) / 2.0) * sample_period
    return frequencies, times, power


def _dominant_frequency(
    frequencies: np.ndarray,
    power: np.ndarray,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> float:
    """Return the strongest spectral bin in a physically relevant band."""
    if not math.isfinite(min_frequency_hz) or min_frequency_hz < 0:
        raise ValueError("Minimum dominant-frequency bound must be finite and >= 0.")
    if not math.isfinite(max_frequency_hz) or max_frequency_hz <= min_frequency_hz:
        raise ValueError("Maximum dominant-frequency bound must exceed the minimum.")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    if frequencies.shape != power.shape:
        raise ValueError("Frequency and power arrays must have matching shapes.")
    selected = (
        (frequencies >= min_frequency_hz)
        & (frequencies <= max_frequency_hz)
        & np.isfinite(power)
        & (power > 0)
    )
    if not np.any(selected):
        raise ValueError(
            "No positive spectral power in the requested dominant-frequency band."
        )
    candidate_indices = np.flatnonzero(selected)
    peak_index = candidate_indices[int(np.argmax(power[selected]))]
    return float(frequencies[peak_index])


def _add_contact_frequency_overlay(
    ax,
    line,
    contact_frequency_hz: float,
) -> None:
    ax.axvline(
        contact_frequency_hz,
        color=line.get_color(),
        linestyle="--",
        linewidth=1.15,
        alpha=0.75,
    )


def _legend_with_contact_frequency(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle="--",
            linewidth=1.15,
            label="Dominant foot-contact frequency",
        )
    )
    labels.append("Dominant foot-contact frequency")
    ax.legend(handles=handles, labels=labels, fontsize=LEGEND_SIZE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input directories (defaults to the Go1, Spot, and Silver Badger data).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help=(
            "Plot labels in input order (defaults to Go1, Spot, and Silver Badger "
            "for the default inputs; otherwise input directory names)."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--linear-output",
        type=Path,
        help="Output for the raw-torque PSD with a linear y-axis.",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        help="Output for spectra normalized by each actuator's observed maximum.",
    )
    parser.add_argument(
        "--contact-output",
        type=Path,
        help="Output for the normalized foot-contact spectra used for gait-frequency estimation.",
    )
    parser.add_argument("--sample-period", type=float)
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--max-frequency", type=float, default=25.0)
    parser.add_argument(
        "--gait-min-frequency",
        type=float,
        default=DEFAULT_GAIT_MIN_FREQUENCY_HZ,
        help="Lower bound for dominant foot-contact/torque peak estimation [Hz].",
    )
    parser.add_argument(
        "--gait-max-frequency",
        type=float,
        default=DEFAULT_GAIT_MAX_FREQUENCY_HZ,
        help="Upper bound for dominant foot-contact/torque peak estimation [Hz].",
    )
    parser.add_argument(
        "--no-gait-frequency-overlay",
        action="store_true",
        help="Do not draw dominant foot-contact frequencies on torque PSD figures.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    inputs_provided = bool(args.inputs)
    inputs = args.inputs or list(DEFAULT_INPUTS)
    if args.labels is not None and len(args.labels) != len(inputs):
        raise ValueError("--labels must contain exactly one label per input.")
    labels = args.labels or (
        list(DEFAULT_LABELS) if not inputs_provided else [path.name for path in inputs]
    )
    if args.max_frequency <= 0:
        raise ValueError("--max-frequency must be positive.")
    gait_max_frequency = min(args.gait_max_frequency, args.max_frequency)
    if gait_max_frequency <= args.gait_min_frequency:
        raise ValueError(
            "The gait-frequency search band must overlap the plotted frequency range."
        )

    analyses = []
    for input_path, label in zip(inputs, labels, strict=True):
        signals_path = _resolve_signals(input_path)
        sample_period = _sample_period(signals_path, args.sample_period)
        with np.load(signals_path) as archive:
            torque = np.asarray(archive["actuator_force"], dtype=np.float64)
            active = np.asarray(archive["active"], dtype=bool)
            feet_contact = (
                np.asarray(archive["feet_contact"], dtype=np.float64)
                if "feet_contact" in archive
                else None
            )
        if torque.ndim != 3 or active.shape != torque.shape[:2]:
            raise ValueError(
                "Expected torque (time,tasks,dofs) and active (time,tasks)."
            )
        if feet_contact is not None and (
            feet_contact.ndim != 3 or feet_contact.shape[:2] != active.shape
        ):
            raise ValueError(
                "Expected feet_contact (time,tasks,feet) aligned with active masks."
            )
        if feet_contact is None and not args.no_gait_frequency_overlay:
            warnings.warn(
                f"{signals_path} does not contain feet_contact; gait-frequency "
                f"validation will be skipped for {label}. Re-run this rollout with "
                "the current learning/evaluate_policy.py for a complete Fig. 2.",
                stacklevel=2,
            )
        tr, tfr = normalized_penalties(torque, active, sample_period)
        frequencies, _, windowed_power = mean_spectrogram(
            torque, active, sample_period, args.window_length, args.overlap
        )
        if not len(tr):
            raise ValueError(f"{signals_path} contains no active torque samples.")
        mean_power = np.nanmean(windowed_power, axis=1)
        if feet_contact is not None:
            contact_frequencies, _, contact_windowed_power = mean_spectrogram(
                feet_contact, active, sample_period, args.window_length, args.overlap
            )
            if not np.allclose(contact_frequencies, frequencies):
                raise ValueError(
                    "Torque and foot-contact spectra use inconsistent frequency grids."
                )
            mean_contact_power = np.nanmean(contact_windowed_power, axis=1)
            dominant_contact_frequency = _dominant_frequency(
                frequencies,
                mean_contact_power,
                args.gait_min_frequency,
                gait_max_frequency,
            )
        else:
            mean_contact_power = None
            dominant_contact_frequency = float("nan")
        dominant_torque_frequency = _dominant_frequency(
            frequencies,
            mean_power,
            args.gait_min_frequency,
            gait_max_frequency,
        )
        maximum_torque = np.max(np.abs(torque[active]), axis=0)
        if np.any(maximum_torque <= 0):
            raise ValueError(f"{signals_path} contains an identically zero actuator.")
        scaled_torque = torque / maximum_torque[None, None, :]
        scaled_tr, scaled_tfr = normalized_penalties(
            scaled_torque, active, sample_period
        )
        _, _, scaled_windowed_power = mean_spectrogram(
            scaled_torque,
            active,
            sample_period,
            args.window_length,
            args.overlap,
        )
        analyses.append(
            (
                label,
                frequencies,
                mean_power,
                tr,
                tfr,
                np.nanmean(scaled_windowed_power, axis=1),
                scaled_tr,
                scaled_tfr,
                maximum_torque,
                mean_contact_power,
                dominant_contact_frequency,
                dominant_torque_frequency,
            )
        )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for analysis in analyses:
        label, frequencies, mean_power = analysis[:3]
        contact_frequency = analysis[10]
        selected = frequencies <= args.max_frequency
        positive = mean_power[np.isfinite(mean_power) & (mean_power > 0)]
        floor = float(np.min(positive)) if positive.size else np.finfo(float).tiny
        (line,) = ax.plot(
            frequencies[selected],
            10.0 * np.log10(np.maximum(mean_power[selected], floor)),
            linewidth=1.8,
            label=label,
        )
        if not args.no_gait_frequency_overlay and np.isfinite(contact_frequency):
            _add_contact_frequency_overlay(ax, line, contact_frequency)
    ax.set(
        xlabel="Frequency [Hz]",
        ylabel=r"Mean torque PSD [dB (N m)$^2$/Hz]",
    )
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.25)
    if args.no_gait_frequency_overlay:
        ax.legend(fontsize=LEGEND_SIZE)
    else:
        _legend_with_contact_frequency(ax)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    linear_output = args.linear_output or args.output.with_name(
        f"{args.output.stem}_linear{args.output.suffix}"
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for analysis in analyses:
        label, frequencies, mean_power = analysis[:3]
        contact_frequency = analysis[10]
        selected = frequencies <= args.max_frequency
        (line,) = ax.plot(
            frequencies[selected],
            mean_power[selected],
            linewidth=1.8,
            label=label,
        )
        if not args.no_gait_frequency_overlay and np.isfinite(contact_frequency):
            _add_contact_frequency_overlay(ax, line, contact_frequency)
    ax.set(
        xlabel="Frequency [Hz]",
        ylabel=r"Mean torque PSD [(N m)$^2$/Hz]",
    )
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    if args.no_gait_frequency_overlay:
        ax.legend(fontsize=LEGEND_SIZE)
    else:
        _legend_with_contact_frequency(ax)
    fig.tight_layout()
    linear_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(linear_output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    normalized_output = args.normalized_output or args.output.with_name(
        f"{args.output.stem}_max_normalized{args.output.suffix}"
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for analysis in analyses:
        label = analysis[0]
        frequencies = analysis[1]
        mean_power = analysis[5]
        contact_frequency = analysis[10]
        selected = frequencies <= args.max_frequency
        total_variance = np.trapezoid(mean_power, frequencies)
        if not np.isfinite(total_variance) or total_variance <= 0:
            raise ValueError(f"{label} has no positive spectral variance.")
        variance_fraction = mean_power / total_variance
        (line,) = ax.plot(
            frequencies[selected],
            variance_fraction[selected],
            linewidth=1.8,
            label=label,
        )
        if not args.no_gait_frequency_overlay and np.isfinite(contact_frequency):
            _add_contact_frequency_overlay(ax, line, contact_frequency)
    ax.set(
        xlabel="Frequency [Hz]",
        ylabel="Fraction of torque variance per Hz",
    )
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.25)
    if args.no_gait_frequency_overlay:
        ax.legend(fontsize=LEGEND_SIZE)
    else:
        _legend_with_contact_frequency(ax)
    fig.tight_layout()
    normalized_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(normalized_output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    contact_output = args.contact_output or args.output.with_name(
        f"{args.output.stem}_foot_contact{args.output.suffix}"
    )
    contact_analyses = [analysis for analysis in analyses if analysis[9] is not None]
    if contact_analyses:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for analysis in contact_analyses:
            label = analysis[0]
            frequencies = analysis[1]
            mean_contact_power = analysis[9]
            selected = frequencies <= args.max_frequency
            total_contact_variance = np.trapezoid(mean_contact_power, frequencies)
            if not np.isfinite(total_contact_variance) or total_contact_variance <= 0:
                raise ValueError(f"{label} has no positive foot-contact spectral variance.")
            normalized_contact_power = mean_contact_power / total_contact_variance
            ax.plot(
                frequencies[selected],
                normalized_contact_power[selected],
                linewidth=1.8,
                label=label,
            )
        ax.set(
            xlabel="Frequency [Hz]",
            ylabel="Normalized foot-contact PSD [1/Hz]",
        )
        ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=LEGEND_SIZE)
        fig.tight_layout()
        contact_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(contact_output, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    output_data = {
        "labels": np.asarray(labels),
        "gait_frequency_search_min_hz": np.asarray(args.gait_min_frequency),
        "gait_frequency_search_max_hz": np.asarray(gait_max_frequency),
    }
    normalized_output_data = {"labels": np.asarray(labels)}
    for index, analysis in enumerate(analyses):
        (
            label,
            frequencies,
            mean_power,
            tr,
            tfr,
            scaled_mean_power,
            scaled_tr,
            scaled_tfr,
            maximum_torque,
            mean_contact_power,
            dominant_contact_frequency,
            dominant_torque_frequency,
        ) = analysis
        prefix = f"policy_{index}"
        output_data[f"{prefix}_frequencies_hz"] = frequencies
        output_data[f"{prefix}_mean_torque_psd"] = mean_power
        if mean_contact_power is not None:
            output_data[f"{prefix}_mean_feet_contact_psd"] = mean_contact_power
            output_data[f"{prefix}_dominant_contact_frequency_hz"] = np.asarray(
                dominant_contact_frequency
            )
        output_data[f"{prefix}_dominant_torque_frequency_hz"] = np.asarray(
            dominant_torque_frequency
        )
        if np.isfinite(dominant_contact_frequency):
            output_data[f"{prefix}_contact_torque_peak_difference_hz"] = np.asarray(
                abs(dominant_contact_frequency - dominant_torque_frequency)
            )
        output_data[f"{prefix}_normalized_torque_rate_per_task"] = tr
        output_data[f"{prefix}_normalized_tfr_f5_n1_m1_per_task"] = tfr
        normalized_output_data[f"{prefix}_frequencies_hz"] = frequencies
        normalized_output_data[f"{prefix}_mean_normalized_torque_psd"] = (
            scaled_mean_power
        )
        total_variance = np.trapezoid(scaled_mean_power, frequencies)
        normalized_output_data[f"{prefix}_torque_variance_fraction_per_hz"] = (
            scaled_mean_power / total_variance
        )
        normalized_output_data[f"{prefix}_normalized_torque_rate_per_task"] = scaled_tr
        normalized_output_data[f"{prefix}_normalized_tfr_f5_n1_m1_per_task"] = (
            scaled_tfr
        )
        normalized_output_data[f"{prefix}_maximum_absolute_torque_per_actuator"] = (
            maximum_torque
        )
        if np.isfinite(dominant_contact_frequency):
            print(
                f"{label} dominant foot-contact frequency: "
                f"{dominant_contact_frequency:.4g} Hz; dominant torque frequency in "
                f"the same band: {dominant_torque_frequency:.4g} Hz; "
                f"|difference|={abs(dominant_contact_frequency - dominant_torque_frequency):.4g} Hz"
            )
        else:
            print(
                f"{label} dominant torque frequency in gait band: "
                f"{dominant_torque_frequency:.4g} Hz; foot-contact frequency unavailable"
            )
        print(
            f"{label} raw-torque, frequency-normalized torque-rate penalty: "
            f"{np.mean(tr):.8g} +/- {np.std(tr):.8g}"
        )
        print(
            f"{label} raw-torque, frequency-normalized high-pass penalty "
            f"(f=5, o=1, m=1): "
            f"{np.mean(tfr):.8g} +/- {np.std(tfr):.8g}"
        )
        print(
            f"{label} max-torque-normalized torque-rate penalty: "
            f"{np.mean(scaled_tr):.8g} +/- {np.std(scaled_tr):.8g}"
        )
        print(
            f"{label} max-torque-normalized high-pass penalty "
            f"(f=5, o=1, m=1): "
            f"{np.mean(scaled_tfr):.8g} +/- {np.std(scaled_tfr):.8g}"
        )
    np.savez_compressed(args.output.with_suffix(".npz"), **output_data)
    np.savez_compressed(normalized_output.with_suffix(".npz"), **normalized_output_data)
    print(f"Mean torque spectrum written to: {args.output.resolve()}")
    print(f"Linear-scale torque spectrum written to: {linear_output.resolve()}")
    print(f"Normalized torque spectrum written to: {normalized_output.resolve()}")
    if contact_analyses:
        print(f"Foot-contact spectrum written to: {contact_output.resolve()}")
    else:
        print("No foot-contact spectrum written: none of the inputs contained feet_contact.")


if __name__ == "__main__":
    main()
