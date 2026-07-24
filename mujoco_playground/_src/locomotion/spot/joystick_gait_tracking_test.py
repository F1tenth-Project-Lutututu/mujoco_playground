# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Spot joystick gait-tracking regularization."""

from absl.testing import absltest
import jax.numpy as jp

from mujoco_playground._src.locomotion.spot import joystick_gait_tracking


class JoystickGaitTrackingTest(absltest.TestCase):

  def test_action_rate_is_enabled_by_default(self):
    config = joystick_gait_tracking.default_config()

    self.assertEqual(config.reward_config.scales.action_rate, -0.01)

  def test_action_rate_is_squared_action_difference(self):
    env = object.__new__(joystick_gait_tracking.JoystickGaitTracking)

    cost = env._cost_action_rate(  # pylint: disable=protected-access
        jp.array([1.0, -1.0]), jp.array([0.5, 0.0])
    )

    self.assertAlmostEqual(float(cost), 1.25)


if __name__ == "__main__":
  absltest.main()
