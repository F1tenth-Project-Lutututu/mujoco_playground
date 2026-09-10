"""Tests for action-history policy observations."""

from absl.testing import absltest
import jax.numpy as jp
from ml_collections import config_dict
import numpy as np

from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion import action_history


class ActionHistoryTest(absltest.TestCase):

  def test_all_quadruped_action_rate_configs_support_fixed_observation(self):
    quadruped_prefixes = ("Barkour", "Go1", "SilverBadger", "Spot")
    missing = []
    for env_name in locomotion.ALL_ENVS:
      if not env_name.startswith(quadruped_prefixes):
        continue
      config = locomotion.get_default_config(env_name)
      reward_config = config.get("reward_config", {})
      if "action_rate" not in reward_config.get("scales", {}):
        continue
      if "action_rate_use_fixed_observation" not in reward_config:
        missing.append(env_name)
    self.assertEmpty(missing)

  def setUp(self):
    super().setUp()
    self.info = {
        "last_act": jp.array([2.0, 3.0]),
        "last_last_act": jp.array([0.0, 1.0]),
    }
    self.current_action = jp.array([4.0, 5.0])

  def test_legacy_observation_is_unchanged(self):
    config = config_dict.create(
        action_rate_use_second_difference=False,
        action_rate_use_fixed_observation=False,
    )
    np.testing.assert_array_equal(
        action_history.observation(config, self.info, self.current_action),
        self.info["last_act"],
    )

  def test_fixed_action_rate_observes_previous_action(self):
    config = config_dict.create(
        action_rate_use_second_difference=False,
        action_rate_use_fixed_observation=True,
    )
    np.testing.assert_array_equal(
        action_history.observation(config, self.info, self.current_action),
        self.current_action,
    )

  def test_fixed_action_smoothness_observes_both_previous_actions(self):
    config = config_dict.create(
        action_rate_use_second_difference=True,
        action_rate_use_fixed_observation=True,
    )
    np.testing.assert_array_equal(
        action_history.observation(config, self.info, self.current_action),
        jp.array([4.0, 5.0, 2.0, 3.0]),
    )

  def test_fixed_history_is_zero_initialized_on_reset(self):
    config = config_dict.create(
        action_rate_use_second_difference=True,
        action_rate_use_fixed_observation=True,
    )
    info = {
        "last_act": jp.zeros(2),
        "last_last_act": jp.zeros(2),
    }
    np.testing.assert_array_equal(
        action_history.observation(config, info), jp.zeros(4)
    )

  def test_fixed_smoothness_without_current_action_uses_stored_history(self):
    config = config_dict.create(
        action_rate_use_second_difference=True,
        action_rate_use_fixed_observation=True,
    )
    np.testing.assert_array_equal(
        action_history.observation(config, self.info),
        jp.array([2.0, 3.0, 0.0, 1.0]),
    )


if __name__ == "__main__":
  absltest.main()
