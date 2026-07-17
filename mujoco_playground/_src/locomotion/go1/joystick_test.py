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
"""Tests for the Go1 joystick torque high-pass penalty."""

import types

from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np
from scipy import signal as scipy_signal

from mujoco_playground._src.locomotion.go1 import joystick


class JoystickTorqueHighpassTest(absltest.TestCase):

  def test_eighth_order_filter_has_expected_sections_and_cutoff(self):
    sos, steady_state = joystick._butterworth_highpass_sos(  # pylint: disable=protected-access
        cutoff_hz=5.0, order=8, sample_rate_hz=50.0
    )

    self.assertEqual(sos.shape, (4, 6))
    self.assertEqual(steady_state.shape, (4, 2))
    _, response = scipy_signal.sosfreqz(
        np.asarray(sos), worN=np.asarray([5.0]), fs=50.0
    )
    self.assertAlmostEqual(float(np.abs(response[0]) ** 2), 0.5, places=5)

  def test_accepts_higher_filter_and_fractional_difference_orders(self):
    self.assertEqual(
        joystick._validate_torque_highpass_order(8),  # pylint: disable=protected-access
        8,
    )
    self.assertEqual(
        joystick._validate_torque_difference_order(6.5),  # pylint: disable=protected-access
        6.5,
    )

  def test_seven_difference_stages_run_under_jit(self):
    difference_filter = types.SimpleNamespace(
        _torque_difference_upper_order=7,
        _torque_difference_lower_order=6,
        _torque_difference_mix=0.5,
        _torque_difference_scale_base=1.25,
    )
    apply_differences = jax.jit(
        lambda signal, previous: joystick.Joystick._apply_torque_differences(  # pylint: disable=protected-access
            difference_filter, signal, previous, jp.asarray(False)
        )
    )

    cost, next_inputs = apply_differences(
        jp.arange(12, dtype=jp.float32), jp.zeros((7, 12))
    )
    self.assertEqual(next_inputs.shape, (7, 12))
    self.assertTrue(jp.isfinite(cost))

  def test_rejects_orders_above_supported_maximum(self):
    with self.assertRaisesRegex(ValueError, "between 1 and 8"):
      joystick._validate_torque_highpass_order(9)  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, "between 0 and 8"):
      joystick._validate_torque_difference_order(8.5)  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
