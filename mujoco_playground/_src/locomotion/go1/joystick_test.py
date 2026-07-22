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

  def test_flat_terrain_25_config_only_changes_vx_range(self):
    default = joystick.default_config()
    fast = joystick.flat_terrain_25_config()

    np.testing.assert_allclose(fast.command_config.a, [2.5, 0.8, 1.2])
    np.testing.assert_allclose(default.command_config.a, [1.5, 0.8, 1.2])
    self.assertEqual(fast.command_config.b, default.command_config.b)

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
    self.assertEqual(
        joystick._validate_highpass_penalty_signal("action"),  # pylint: disable=protected-access
        "action",
    )

  def test_highpass_memory_observation_contains_all_reward_memory(self):
    observation = joystick._highpass_memory_observation({  # pylint: disable=protected-access
        "torque_highpass_state": jp.arange(8).reshape((2, 2, 2)),
        "torque_difference_inputs": jp.arange(8, 12).reshape((2, 2)),
    })

    np.testing.assert_array_equal(observation, np.arange(12))

  def test_observe_highpass_state_requires_boolean(self):
    self.assertTrue(
        joystick._validate_observe_highpass_state(True)  # pylint: disable=protected-access
    )
    with self.assertRaisesRegex(ValueError, "must be a boolean"):
      joystick._validate_observe_highpass_state(1)  # pylint: disable=protected-access

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

  def test_actuator_capacities_use_largest_absolute_force_limit(self):
    capacities = joystick._actuator_force_capacities(  # pylint: disable=protected-access
        [[-23.7, 23.7], [-30.0, 35.55]]
    )
    np.testing.assert_allclose(capacities, [23.7, 35.55])

  def test_actuator_capacities_reject_invalid_limits(self):
    with self.assertRaisesRegex(ValueError, "finite, positive force limits"):
      joystick._actuator_force_capacities(  # pylint: disable=protected-access
          [[0.0, 0.0]]
      )

  def test_adaptive_weight_decreases_toward_minimum(self):
    weights = joystick._adaptive_highpass_weight(  # pylint: disable=protected-access
        jp.asarray([0.0, 0.25, 100.0]), 0.1, 1.0, 0.25
    )
    np.testing.assert_allclose(weights, [1.0, 0.1 + 0.9 / np.e, 0.1], atol=1e-6)

  def test_rejects_invalid_adaptive_weight_config(self):
    with self.assertRaisesRegex(ValueError, "min_weight"):
      joystick._validate_adaptive_highpass_config(  # pylint: disable=protected-access
          True, 1.0, 0.1, 0.25
      )
    with self.assertRaisesRegex(ValueError, "sigma must be positive"):
      joystick._validate_adaptive_highpass_config(  # pylint: disable=protected-access
          True, 0.1, 1.0, 0.0
      )

  def test_rejects_orders_above_supported_maximum(self):
    with self.assertRaisesRegex(ValueError, "between 1 and 8"):
      joystick._validate_torque_highpass_order(9)  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, "between 0 and 8"):
      joystick._validate_torque_difference_order(8.5)  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, "torque_highpass_signal"):
      joystick._validate_highpass_penalty_signal("motor_target")  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
