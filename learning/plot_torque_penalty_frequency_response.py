"""Plots frequency weighting of the torque high-pass penalty modes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


SAMPLE_RATE_HZ = 50.0
CUTOFFS_HZ = (5.0, 7.0, 10.0)
FILTER_ORDER = 2
FILTER_ORDERS = (1, 2, 3, 4)
DIFFERENCE_ORDERS = (0.0, 0.5, 1.0, 1.5, 2.0)


def plot_response(cutoff_hz: float, output_path: Path) -> None:
  """Plots both high-pass modes and consecutive-torque differences."""
  frequencies_hz = np.linspace(0.0, SAMPLE_RATE_HZ / 2.0, 2000)
  sos = signal.butter(
      FILTER_ORDER,
      cutoff_hz,
      btype="highpass",
      fs=SAMPLE_RATE_HZ,
      output="sos",
  )
  _, highpass_response = signal.sosfreqz(
      sos, worN=frequencies_hz, fs=SAMPLE_RATE_HZ
  )

  energy_weight = np.abs(highpass_response) ** 2
  difference_weight = 4.0 * np.sin(
      np.pi * frequencies_hz / SAMPLE_RATE_HZ
  ) ** 2
  difference_weight_at_cutoff = 4.0 * np.sin(
      np.pi * cutoff_hz / SAMPLE_RATE_HZ
  ) ** 2
  normalized_difference_weight = (
      difference_weight / difference_weight_at_cutoff
  )
  rate_weight = energy_weight * normalized_difference_weight

  figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
  axis.axvspan(
      0.0,
      cutoff_hz,
      color="#4C78A8",
      alpha=0.08,
      label="Below cutoff",
  )
  axis.axvline(
      cutoff_hz,
      color="#555555",
      linestyle="--",
      linewidth=1.4,
      label=f"Cutoff: {cutoff_hz:g} Hz",
  )
  axis.plot(
      frequencies_hz,
      energy_weight,
      color="#4C78A8",
      linewidth=2.6,
      label=r"energy: $|H_{HP}(f)|^2$",
  )
  axis.plot(
      frequencies_hz,
      rate_weight,
      color="#E45756",
      linewidth=2.6,
      label=r"rate: $|H_{HP}(f)|^2"
      r"[\sin(\pi f/f_s)/\sin(\pi f_c/f_s)]^2$",
  )
  axis.plot(
      frequencies_hz,
      difference_weight,
      color="#54A24B",
      linewidth=2.6,
      linestyle="-.",
      label=r"consecutive torque difference: $4\sin^2(\pi f/f_s)$",
  )

  axis.set(
      xlim=(0.0, SAMPLE_RATE_HZ / 2.0),
      ylim=(0.0, max(np.max(rate_weight), np.max(difference_weight)) * 1.02),
      xlabel="Torque frequency (Hz)",
      ylabel="Penalty weight per unit torque energy",
      title=(
          "Torque regularization frequency weighting\n"
          f"Butterworth order {FILTER_ORDER}, cutoff {cutoff_hz:g} Hz, "
          f"sample rate {SAMPLE_RATE_HZ:g} Hz"
      ),
  )
  axis.grid(True, alpha=0.25)
  axis.legend(loc="upper left", frameon=True)

  figure.savefig(output_path, dpi=180)
  plt.close(figure)
  print(output_path)


def plot_filter_order_comparison(
    cutoff_hz: float, output_path: Path
) -> None:
  """Compares energy and rate penalty weighting across filter orders."""
  frequencies_hz = np.linspace(0.0, SAMPLE_RATE_HZ / 2.0, 2000)
  difference_weight = 4.0 * np.sin(
      np.pi * frequencies_hz / SAMPLE_RATE_HZ
  ) ** 2
  difference_weight /= 4.0 * np.sin(
      np.pi * cutoff_hz / SAMPLE_RATE_HZ
  ) ** 2
  colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")

  figure, axes = plt.subplots(
      1, 2, figsize=(14, 5.8), sharex=True, constrained_layout=True
  )
  for order, color in zip(FILTER_ORDERS, colors):
    sos = signal.butter(
        order,
        cutoff_hz,
        btype="highpass",
        fs=SAMPLE_RATE_HZ,
        output="sos",
    )
    _, highpass_response = signal.sosfreqz(
        sos, worN=frequencies_hz, fs=SAMPLE_RATE_HZ
    )
    energy_weight = np.abs(highpass_response) ** 2
    axes[0].plot(
        frequencies_hz,
        energy_weight,
        color=color,
        linewidth=2.4,
        label=f"Order {order}",
    )
    axes[1].plot(
        frequencies_hz,
        energy_weight * difference_weight,
        color=color,
        linewidth=2.4,
        label=f"Order {order}",
    )

  for axis in axes:
    axis.axvspan(0.0, cutoff_hz, color="#4C78A8", alpha=0.07)
    axis.axvline(
        cutoff_hz,
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label=f"Cutoff: {cutoff_hz:g} Hz",
    )
    axis.set_xlim(0.0, SAMPLE_RATE_HZ / 2.0)
    axis.set_xlabel("Torque frequency (Hz)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right", frameon=True)

  axes[0].set_ylim(0.0, 1.05)
  axes[0].set_ylabel("Penalty weight per unit torque energy")
  axes[0].set_title(r"Energy penalty: $|H_{HP}(f)|^2$")
  axes[1].set_title("Cutoff-normalized rate penalty")
  axes[1].set_ylim(0.0, np.max(difference_weight) * 1.02)
  figure.suptitle(
      "Effect of Butterworth filter order on torque regularization\n"
      f"Cutoff {cutoff_hz:g} Hz, sample rate {SAMPLE_RATE_HZ:g} Hz",
      fontsize=16,
  )
  figure.savefig(output_path, dpi=180)
  plt.close(figure)
  print(output_path)


def plot_difference_order_comparison(
    cutoff_hz: float, output_path: Path
) -> None:
  """Compares absolute and normalized weighting across difference orders."""
  frequencies_hz = np.linspace(0.0, SAMPLE_RATE_HZ / 2.0, 2000)
  sos = signal.butter(
      FILTER_ORDER,
      cutoff_hz,
      btype="highpass",
      fs=SAMPLE_RATE_HZ,
      output="sos",
  )
  _, highpass_response = signal.sosfreqz(
      sos, worN=frequencies_hz, fs=SAMPLE_RATE_HZ
  )
  energy_weight = np.abs(highpass_response) ** 2
  difference_weight = 4.0 * np.sin(
      np.pi * frequencies_hz / SAMPLE_RATE_HZ
  ) ** 2
  difference_weight /= 4.0 * np.sin(
      np.pi * cutoff_hz / SAMPLE_RATE_HZ
  ) ** 2
  colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")

  figure, axes = plt.subplots(
      1, 2, figsize=(14, 5.8), sharex=False, constrained_layout=True
  )
  for difference_order, color in zip(DIFFERENCE_ORDERS, colors):
    weight = energy_weight * difference_weight**difference_order
    label = f"m = {difference_order:g}"
    axes[0].plot(
        frequencies_hz,
        np.maximum(weight, 1e-12),
        color=color,
        linewidth=2.4,
        label=label,
    )
    axes[1].plot(
        frequencies_hz,
        weight,
        color=color,
        linewidth=2.4,
        label=label,
    )

  for axis in axes:
    axis.axvspan(0.0, cutoff_hz, color="#4C78A8", alpha=0.07)
    axis.axvline(
        cutoff_hz,
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label=f"Cutoff: {cutoff_hz:g} Hz",
    )
    axis.set_xlim(0.0, SAMPLE_RATE_HZ / 2.0)
    axis.set_xlabel("Torque frequency (Hz)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right", frameon=True)

  axes[0].set_yscale("log")
  axes[0].set_ylim(
      1e-8, np.max(difference_weight) ** max(DIFFERENCE_ORDERS) * 1.2
  )
  axes[0].set_xlim(0.0, 20.0)
  axes[0].set_ylabel("Penalty weight per unit torque energy")
  axes[0].set_title("Absolute weighting (log scale)")
  visible_frequencies = frequencies_hz <= 20.0
  maximum_visible_weight = max(
      np.max(
          energy_weight[visible_frequencies]
          * difference_weight[visible_frequencies] ** difference_order
      )
      for difference_order in DIFFERENCE_ORDERS
  )
  axes[1].set_xlim(0.0, 20.0)
  axes[1].set_ylim(0.0, maximum_visible_weight * 1.05)
  axes[1].set_title("Linear scale from 0 to 20 Hz")
  figure.suptitle(
      "Effect of difference order m on torque regularization\n"
      f"Butterworth order {FILTER_ORDER}, cutoff {cutoff_hz:g} Hz, "
      f"sample rate {SAMPLE_RATE_HZ:g} Hz",
      fontsize=16,
  )
  figure.savefig(output_path, dpi=180)
  plt.close(figure)
  print(output_path)


def main() -> None:
  output_directory = Path(__file__).parent
  for cutoff_hz in CUTOFFS_HZ:
    filename = (
        "torque_penalty_frequency_response.png"
        if cutoff_hz == 5.0
        else f"torque_penalty_frequency_response_{cutoff_hz:g}hz.png"
    )
    plot_response(cutoff_hz, output_directory / filename)
  plot_filter_order_comparison(
      cutoff_hz=5.0,
      output_path=(
          output_directory / "torque_penalty_filter_order_comparison_5hz.png"
      ),
  )
  plot_difference_order_comparison(
      cutoff_hz=5.0,
      output_path=(
          output_directory
          / "torque_penalty_difference_order_comparison_5hz.png"
      ),
  )


if __name__ == "__main__":
  main()
