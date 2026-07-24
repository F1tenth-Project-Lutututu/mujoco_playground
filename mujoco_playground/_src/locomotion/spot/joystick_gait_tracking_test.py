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

import types

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

  def test_reward_path_uses_action_for_action_rate(self):
    env = object.__new__(joystick_gait_tracking.JoystickGaitTracking)
    env._config = (  # pylint: disable=protected-access
        joystick_gait_tracking.default_config()
    )
    env._feet_site_id = jp.arange(4)  # pylint: disable=protected-access
    env._hx_idxs = jp.array([0, 3, 6, 9])  # pylint: disable=protected-access
    env._hx_default_pose = jp.zeros(4)  # pylint: disable=protected-access
    env.get_local_linvel = lambda _: jp.zeros(3)
    env.get_global_linvel = lambda _: jp.zeros(3)
    env.get_gyro = lambda _: jp.zeros(3)
    env.get_global_angvel = lambda _: jp.zeros(3)
    data = types.SimpleNamespace(
        qpos=jp.zeros(19),
        site_xpos=jp.zeros((4, 3)),
    )
    info = {
        "command": jp.zeros(3),
        "phase": jp.zeros(4),
        "foot_height": jp.asarray(0.1),
        "gait": jp.asarray(0),
        "last_act": jp.zeros(2),
    }

    _, penalties = env._get_reward(  # pylint: disable=protected-access
        data,
        jp.array([1.0, -1.0]),
        info,
        {},
        jp.asarray(False),
        jp.asarray(0.0),
        jp.asarray(0.0),
    )

    self.assertEqual(float(penalties["action_rate"]), 2.0)


if __name__ == "__main__":
  absltest.main()
