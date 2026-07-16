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

from learning import train_jax_ppo


class RunConfigTest(absltest.TestCase):

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
