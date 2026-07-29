import unittest

from learning import recover_missing_cluster_seeds as recovery


class RecoverMissingClusterSeedsTest(unittest.TestCase):

  def test_discover_reports_only_absent_expected_seeds(self):
    groups = recovery._discover(
        [
            "260729-highpass-400M-hp1em5-f5o1m10-seed0",
            "260729-highpass-400M-hp1em5-f5o1m10-seed2",
            "260728-baseline-400M-ar1em2-seed0",
        ],
        (0, 1, 2),
        "260729-*",
    )
    self.assertEqual(len(groups), 1)
    self.assertEqual(groups[0].missing, (1,))

  def test_launcher_arguments_use_saved_exact_settings(self):
    config = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps",
            "400000000",
            "--env_name",
            "Go1JoystickRoughTerrain",
            "--playground_config_overrides="
            '{"reward_config.scales.torque_high_freq": -1e-5, '
            '"reward_config.torque_highpass_cutoff_hz": 7.0, '
            '"reward_config.torque_highpass_difference_order": 2.0}',
        ]
    }
    self.assertEqual(
        recovery._launcher_arguments(config),
        [
            "hp",
            "1e-5",
            "Go1JoystickRoughTerrain",
            "7",
            "2",
            "400000000",
        ],
    )

  def test_launcher_arguments_preserve_scientific_scale_tags(self):
    config = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps=400000000",
            "--env_name=Go1JoystickRoughTerrain",
            '--playground_config_overrides={"reward_config.scales.action_rate": -0.02}',
        ]
    }
    self.assertEqual(recovery._launcher_arguments(config)[1], "2e-2")


if __name__ == "__main__":
  unittest.main()
