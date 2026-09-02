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
"""Tests for the random-task mean spectrogram plotter."""

import json
from pathlib import Path

from absl.testing import absltest
import numpy as np

from learning import plot_policy_spectrogram


class PlotPolicySpectrogramTest(absltest.TestCase):

  def test_mean_spectrogram_peaks_at_sine_frequency(self):
    sample_period = 0.02
    time = np.arange(128) * sample_period
    signal = np.sin(2 * np.pi * 5 * time)[:, None, None]
    signal = np.repeat(signal, 2, axis=1)
    active = np.ones((128, 2), dtype=bool)

    frequencies, _, power, counts = (
        plot_policy_spectrogram._mean_spectrogram(
            [(signal, active)], sample_period, 100, 0.5
        )
    )

    self.assertAlmostEqual(frequencies[np.nanargmax(power[:, 0])], 5.0)
    self.assertEqual(counts[0], 2)

  def test_mean_spectrogram_ignores_inactive_tasks(self):
    signal = np.zeros((8, 2, 1))
    signal[:, 1] = 100
    active = np.ones((8, 2), dtype=bool)
    active[3, 1] = False

    _, _, power, counts = plot_policy_spectrogram._mean_spectrogram(
        [(signal, active)], 1.0, 8, 0
    )

    self.assertEqual(counts[0], 1)
    self.assertTrue(np.allclose(power[:, 0], 0))

  def test_resolves_evaluation_directory_and_sample_period(self):
    root = Path(self.create_tempdir().full_path)
    random_tasks = root / "random_tasks"
    random_tasks.mkdir()
    signals = random_tasks / "signals.npz"
    np.savez(signals, active=np.ones((2, 1)))
    (root / "summary.json").write_text(json.dumps({
        "metadata": {"sample_period_seconds": 0.02}
    }))

    self.assertEqual(plot_policy_spectrogram._signals_path(root), signals)
    self.assertEqual(plot_policy_spectrogram._sample_period(signals), 0.02)


if __name__ == "__main__":
  absltest.main()
