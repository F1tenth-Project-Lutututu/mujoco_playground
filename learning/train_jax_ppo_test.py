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
"""Tests for persisted JAX PPO run configuration."""

import json
import types

from absl.testing import absltest
from etils import epath
import numpy as np

from learning import train_jax_ppo


class RunConfigTest(absltest.TestCase):

  def test_torque_smoothness_uses_complete_active_episodes(self):
    torques = np.array([
        [[0.0], [0.0]],
        [[1.0], [2.0]],
        [[4.0], [0.0]],
        [[9.0], [100.0]],
        [[16.0], [100.0]],
    ])
    active = np.array([
        [True, True],
        [True, True],
        [True, True],
        [True, False],
        [True, False],
    ])

    metrics = train_jax_ppo._torque_smoothness_metrics(
        torques,
        active,
        savgol_window_length=5,
        savgol_polyorder=2,
    )

    self.assertAlmostEqual(
        metrics["mean_squared_delta_l2_per_step"], 12.5
    )
    self.assertAlmostEqual(
        metrics["mssd_mean_squared_second_difference_per_dof"], 10.0
    )
    self.assertAlmostEqual(
        metrics["msgfd_mean_absolute_savgol_filter_deviation_per_dof"], 0.0
    )

  def test_torque_smoothness_metrics_use_separate_wandb_section(self):
    metric = (
        "eval/episode_torque_smoothness/"
        "mssd_mean_squared_second_difference_per_dof"
    )

    self.assertEqual(
        train_jax_ppo._wandb_metric_name(metric),
        "smoothness/torque/"
        "mssd_mean_squared_second_difference_per_dof",
    )

  def test_tracking_mae_uses_only_active_episode_steps(self):
    linear_absolute_errors = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 100.0],
    ])
    yaw_absolute_errors = np.array([
        [2.0, 1.0],
        [4.0, 3.0],
        [6.0, 100.0],
    ])
    active = np.array([
        [True, True],
        [True, True],
        [True, False],
    ])

    metrics = train_jax_ppo._tracking_mae_metrics(
        linear_absolute_errors, yaw_absolute_errors, active
    )

    self.assertAlmostEqual(metrics["linear_velocity_mae"], 3.0)
    self.assertAlmostEqual(metrics["yaw_rate_mae"], 3.0)

  def test_tracking_mae_metrics_use_tracking_wandb_section(self):
    self.assertEqual(
        train_jax_ppo._wandb_metric_name(
            "eval/tracking/linear_velocity_mae"
        ),
        "tracking/eval_linear_velocity_mae",
    )
    self.assertEqual(
        train_jax_ppo._wandb_metric_name("eval/tracking/yaw_rate_mae"),
        "tracking/eval_yaw_rate_mae",
    )

  def test_all_joystick_environments_enable_tracking_mae(self):
    for env_name in (
        "Go1JoystickRoughTerrain",
        "G1JoystickFlatTerrain",
        "T1JoystickRoughTerrain",
        "SpotJoystick",
        "SilverBadgerJoystickRoughTerrain",
    ):
      self.assertTrue(train_jax_ppo._tracks_velocity_mae(env_name))

  def test_non_joystick_environments_do_not_enable_tracking_mae(self):
    for env_name in ("Go1Getup", "Go1Handstand", "CartpoleBalance"):
      self.assertFalse(train_jax_ppo._tracks_velocity_mae(env_name))

  def test_run_logdir_has_environment_parent(self):
    root = self.create_tempdir().full_path

    logdir = train_jax_ppo._run_logdir(
        root, "Go1JoystickFlatTerrain", "260717-experiment-seed0"
    )

    self.assertEqual(
        logdir,
        epath.Path(root).resolve()
        / "Go1JoystickFlatTerrain"
        / "260717-experiment-seed0",
    )

  def test_wandb_project_name_includes_environment(self):
    self.assertEqual(
        train_jax_ppo._wandb_project_name("Go1JoystickFlatTerrain"),
        "spectral_playground_highpass_Go1JoystickFlatTerrain",
    )

  def test_wandb_group_name_includes_date_prefix(self):
    self.assertEqual(
        train_jax_ppo._wandb_group_name("experiment", "260722"),
        "260722-experiment",
    )

  def test_flat_terrain_25_uses_go1_ppo_config(self):
    config = train_jax_ppo.get_rl_config(
        "Go1JoystickFlatTerrain25", vision=False, impl="warp"
    )

    self.assertEqual(config.num_timesteps, 200_000_000)
    self.assertEqual(
        config.network_factory.policy_hidden_layer_sizes, (512, 256, 128)
    )
    self.assertEqual(
        config.network_factory.value_obs_key, "privileged_state"
    )

  def test_flat_terrain_35_uses_go1_ppo_config(self):
    config = train_jax_ppo.get_rl_config(
        "Go1JoystickFlatTerrain35", vision=False, impl="warp"
    )

    self.assertEqual(config.num_timesteps, 200_000_000)
    self.assertEqual(
        config.network_factory.policy_hidden_layer_sizes, (512, 256, 128)
    )
    self.assertEqual(
        config.network_factory.value_obs_key, "privileged_state"
    )

  def test_rough_terrain_25_uses_go1_ppo_config(self):
    config = train_jax_ppo.get_rl_config(
        "Go1JoystickRoughTerrain25", vision=False, impl="warp"
    )

    self.assertEqual(config.num_timesteps, 200_000_000)
    self.assertEqual(
        config.network_factory.policy_hidden_layer_sizes, (512, 256, 128)
    )
    self.assertEqual(
        config.network_factory.value_obs_key, "privileged_state"
    )

  def test_merge_saved_config_fills_new_nested_defaults(self):
    defaults = {
        "impl": "warp",
        "reward_config": {
            "new_field": 3.0,
            "scales": {"action_rate": -0.01, "new_reward": 0.0},
        },
    }
    saved = {
        "impl": "jax",
        "reward_config": {"scales": {"action_rate": -0.02}},
    }

    merged = train_jax_ppo._merge_saved_config(defaults, saved)

    self.assertEqual(merged["impl"], "jax")
    self.assertEqual(merged["reward_config"]["new_field"], 3.0)
    self.assertEqual(
        merged["reward_config"]["scales"],
        {"action_rate": -0.02, "new_reward": 0.0},
    )

  def test_resolve_checkpoint_path_selects_latest_numeric_directory(self):
    root = epath.Path(self.create_tempdir().full_path)
    (root / "000000000010").mkdir()
    (root / "000000000002").mkdir()
    (root / "artifacts").mkdir()

    resolved = train_jax_ppo._resolve_checkpoint_path(str(root))

    self.assertEqual(resolved, root / "000000000010")

  def test_load_run_config_from_specific_checkpoint(self):
    root = epath.Path(self.create_tempdir().full_path)
    checkpoint = root / "000000000010"
    checkpoint.mkdir()
    expected = {
        "schema_version": 1,
        "env_name": "Go1JoystickFlatTerrain",
        "environment_config": {"impl": "jax"},
        "ppo_config": {"num_timesteps": 10},
    }
    (root / "run_config.json").write_text(json.dumps(expected))

    actual = train_jax_ppo._load_run_config(checkpoint)

    self.assertEqual(actual, expected)

  def test_load_run_config_supports_legacy_environment_config(self):
    root = epath.Path(self.create_tempdir().full_path)
    checkpoint = root / "000000000010"
    checkpoint.mkdir()
    environment_config = {"impl": "warp", "episode_length": 1000}
    (root / "config.json").write_text(json.dumps(environment_config))

    actual = train_jax_ppo._load_run_config(checkpoint)

    self.assertEqual(actual["schema_version"], 0)
    self.assertEqual(actual["impl"], "warp")
    self.assertEqual(actual["environment_config"], environment_config)

  def test_load_checkpoint_network_config_supports_null_initializers(self):
    root = epath.Path(self.create_tempdir().full_path)
    config_path = root / "ppo_network_config.json"
    config_path.write_text(json.dumps({
        "action_size": 12,
        "normalize_observations": True,
        "observation_size": {"state": 48},
        "network_factory_kwargs": {
            "activation": "silu",
            "policy_network_kernel_init_fn": "lecun_uniform",
            "mean_kernel_init_fn": None,
        },
    }))

    actual = train_jax_ppo._load_checkpoint_network_config(config_path)

    self.assertTrue(callable(actual.network_factory_kwargs.activation))
    self.assertTrue(
        callable(actual.network_factory_kwargs.policy_network_kernel_init_fn)
    )
    self.assertIsNone(actual.network_factory_kwargs.mean_kernel_init_fn)

  def test_explicit_flag_takes_precedence_over_saved_value(self):
    explicit_flag = types.SimpleNamespace(present=True, value=7)

    value = train_jax_ppo._saved_or_flag(
        {"seed": 3}, "seed", explicit_flag
    )

    self.assertEqual(value, 7)


if __name__ == "__main__":
  absltest.main()
