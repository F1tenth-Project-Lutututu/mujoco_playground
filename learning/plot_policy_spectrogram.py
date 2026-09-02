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
"""Plots a mean spectrogram from random-task policy evaluation signals.

Run ``learning/evaluate_policy.py`` with ``--num_random_tasks`` and
``--save_signals`` first. Multiple inputs may be supplied to average policy
seeds together. Each input can be a signals.npz file, its random_tasks
directory, or the parent evaluation directory.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_SIGNAL = "actuator_force"
DEFAULT_WINDOW_LENGTH = 128
DEFAULT_OVERLAP = 0.75


def _signals_path(path: Path) -> Path:
  """Resolves a signals archive from common evaluation directory layouts."""
  candidates = (
      path,
      path / "signals.npz",
      path / "random_tasks" / "signals.npz",
  )
  for candidate in candidates:
    if candidate.is_file() and candidate.name == "signals.npz":
      return candidate
  raise FileNotFoundError(f"Could not find signals.npz under: {path}")


def _sample_period(path: Path) -> float:
  """Loads the evaluator sample period associated with a signals archive."""
  for parent in path.parents:
    summary_path = parent / "summary.json"
    if not summary_path.is_file():
      continue
    with summary_path.open(encoding="utf-8") as fp:
      summary = json.load(fp)
    value = summary.get("metadata", {}).get("sample_period_seconds")
    if value is not None:
      return float(value)
  raise ValueError(
      f"No sample_period_seconds metadata found for {path}; "
      "pass --sample_period_seconds."
  )


def _mean_spectrogram(
    archives: Sequence[tuple[np.ndarray, np.ndarray]],
    sample_period: float,
    window_length: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Returns frequencies, times, mean PSD, and contributing task counts."""
  if not archives:
    raise ValueError("At least one signal archive is required.")
  if not math.isfinite(sample_period) or sample_period <= 0:
    raise ValueError("sample_period must be positive and finite.")
  if window_length < 2:
    raise ValueError("window_length must be at least 2.")
  if not 0 <= overlap < 1:
    raise ValueError("overlap must be in [0, 1).")
  step = max(1, round(window_length * (1 - overlap)))
  time_steps = {signal.shape[0] for signal, _ in archives}
  if len(time_steps) != 1:
    raise ValueError("All archives must have the same trajectory length.")
  total_steps = time_steps.pop()
  if window_length > total_steps:
    raise ValueError("window_length exceeds the trajectory length.")

  starts = np.arange(0, total_steps - window_length + 1, step)
  window = np.hanning(window_length)
  scale = (1 / sample_period) * np.sum(np.square(window))
  frequencies = np.fft.rfftfreq(window_length, d=sample_period)
  power_sum = np.zeros((len(frequencies), len(starts)))
  sample_counts = np.zeros(len(starts), dtype=np.int64)

  for signal, active in archives:
    signal = np.asarray(signal, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if signal.ndim == 2:
      signal = signal[:, :, None]
    if signal.ndim != 3 or active.shape != signal.shape[:2]:
      raise ValueError(
          "Signals must have shape (time, tasks, channels) and active must "
          "have shape (time, tasks)."
      )
    for time_index, start in enumerate(starts):
      valid = np.all(active[start : start + window_length], axis=0)
      if not np.any(valid):
        continue
      segment = signal[start : start + window_length, valid]
      segment -= np.mean(segment, axis=0, keepdims=True)
      spectrum = np.fft.rfft(segment * window[:, None, None], axis=0)
      power = np.square(np.abs(spectrum)) / scale
      upper = -1 if window_length % 2 == 0 else None
      power[1:upper] *= 2
      power_sum[:, time_index] += np.sum(power, axis=(1, 2))
      sample_counts[time_index] += valid.sum() * signal.shape[2]

  mean_power = np.full_like(power_sum, np.nan)
  np.divide(
      power_sum,
      sample_counts[None, :],
      out=mean_power,
      where=sample_counts[None, :] > 0,
  )
  times = (starts + (window_length - 1) / 2) * sample_period
  return frequencies, times, mean_power, sample_counts


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("inputs", nargs="+", type=Path)
  parser.add_argument("--signal", default=DEFAULT_SIGNAL)
  parser.add_argument(
      "--output", type=Path, default=Path("mean_spectrogram.png")
  )
  parser.add_argument("--sample_period_seconds", type=float)
  parser.add_argument(
      "--window_length", type=int, default=DEFAULT_WINDOW_LENGTH
  )
  parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
  parser.add_argument("--max_frequency_hz", type=float)
  parser.add_argument("--title", default="Mean policy spectrogram")
  parser.add_argument("--dpi", type=int, default=160)
  parser.add_argument("--show", action="store_true")
  return parser


def main() -> None:
  args = _build_parser().parse_args()
  paths = [_signals_path(path) for path in args.inputs]
  sample_period = args.sample_period_seconds or _sample_period(paths[0])
  archives = []
  for path in paths:
    if args.sample_period_seconds is None:
      archive_period = _sample_period(path)
      if not math.isclose(archive_period, sample_period):
        raise ValueError("Input archives have different sample periods.")
    with np.load(path) as archive:
      if args.signal not in archive or "active" not in archive:
        raise KeyError(f"{path} lacks {args.signal!r} or 'active'.")
      archives.append((archive[args.signal], archive["active"]))

  frequencies, times, power, counts = _mean_spectrogram(
      archives, sample_period, args.window_length, args.overlap
  )
  selected = np.ones_like(frequencies, dtype=bool)
  if args.max_frequency_hz is not None:
    selected = frequencies <= args.max_frequency_hz
  if not np.any(selected):
    raise ValueError("--max_frequency_hz excludes every frequency bin.")

  try:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError as error:
    raise ImportError(
        "matplotlib is required to plot the spectrogram."
    ) from error
  finite_positive = power[np.isfinite(power) & (power > 0)]
  floor = float(np.min(finite_positive)) if finite_positive.size else 1e-30
  power_db = 10 * np.log10(np.maximum(power[selected], floor))
  fig, ax = plt.subplots(figsize=(12, 6.75))
  image = ax.pcolormesh(
      times, frequencies[selected], power_db, shading="auto", cmap="magma"
  )
  ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title=args.title)
  colorbar = fig.colorbar(image, ax=ax)
  colorbar.set_label(f"Mean {args.signal} PSD (dB / Hz)")
  fig.tight_layout()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
  if args.show:
    plt.show()
  plt.close(fig)

  data_output = args.output.with_suffix(".npz")
  np.savez_compressed(
      data_output,
      frequencies_hz=frequencies,
      times_seconds=times,
      mean_power_spectral_density=power,
      contributing_task_channels=counts,
      signal=args.signal,
      input_paths=np.asarray([str(path) for path in paths]),
  )
  print(f"Mean spectrogram written to: {args.output.resolve()}")
  print(f"Spectrogram data written to: {data_output.resolve()}")


if __name__ == "__main__":
  main()
