import unittest
from unittest import mock

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
            "2.0",
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

  def test_launcher_arguments_identify_action_smoothness(self):
    config = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps=1000000000",
            "--env_name=Go1JoystickFlatTerrain",
            "--playground_config_overrides="
            '{"reward_config.scales.action_rate": -0.02, '
            '"reward_config.action_rate_use_second_difference": true}',
        ]
    }

    self.assertEqual(
        recovery._launcher_arguments(config),
        [
            "as",
            "2e-2",
            "Go1JoystickFlatTerrain",
            "5",
            "1.0",
            "1000000000",
        ],
    )

  def test_launcher_arguments_remove_binary_float_artifacts(self):
    config = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps=400000000",
            "--env_name=Go1JoystickRoughTerrain",
            '--playground_config_overrides={"reward_config.scales.torque_rate": -0.0006}',
        ]
    }
    self.assertEqual(recovery._launcher_arguments(config)[1], "6e-4")

  def test_launcher_arguments_identify_torque_smoothness(self):
    config = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps=400000000",
            "--env_name=Go1JoystickRoughTerrain",
            "--playground_config_overrides="
            '{"reward_config.scales.torque_rate": -0.0006, '
            '"reward_config.torque_rate_use_second_difference": true}',
        ]
    }

    self.assertEqual(recovery._launcher_arguments(config)[0], "ts")

  def test_launcher_arguments_preserve_m10_default_convention(self):
    config = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps=400000000",
            "--env_name=Go1JoystickRoughTerrain",
            '--playground_config_overrides={"reward_config.scales.torque_high_freq": -1e-5}',
        ]
    }
    arguments = recovery._launcher_arguments(config)

    self.assertEqual(arguments[4], "1.0")

  def test_launcher_family_round_trips_all_naming_conventions(self):
    cases = (
        (
            ["ar", "2e-2", "Env", "5", "1.0", "400000000"],
            "260729-baseline-400M-ar2em2",
        ),
        (
            ["as", "2e-2", "Env", "5", "1.0", "1000000000"],
            "260817-actionsmoothness-1000M-as2em2",
        ),
        (
            ["tr", "6e-4", "Env", "5", "1.0", "400M"],
            "260729-torquerate-400M-tr6em4",
        ),
        (
            ["ts", "6e-4", "Env", "5", "1.0", "400M"],
            "260818-torquesmoothness-400M-ts6em4",
        ),
        (
            ["hp", "1e-5", "Env", "7", "2.0", "400000000"],
            "260729-highpass-400M-hp1em5-f7o1m20",
        ),
    )
    for arguments, expected in cases:
      with self.subTest(arguments=arguments):
        self.assertEqual(
            recovery._launcher_family(arguments, expected[:6]), expected
        )
        recovery._validate_launcher_family(expected, arguments)

  def test_launcher_arguments_reject_unsupported_highpass_order(self):
    config = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps=400000000",
            "--env_name=Go1JoystickRoughTerrain",
            "--playground_config_overrides="
            '{"reward_config.scales.torque_high_freq": -1e-5, '
            '"reward_config.torque_highpass_order": 2}',
        ]
    }
    with self.assertRaisesRegex(ValueError, "fixes the high-pass order at 1"):
      recovery._launcher_arguments(config)

  @mock.patch.object(recovery, "_ssh")
  @mock.patch.object(recovery.downloader, "_remote_run_names")
  def test_main_skips_template_without_run_config_and_submits_other_families(
      self, remote_run_names, ssh
  ):
    remote_run_names.return_value = [
        "260729-baseline-400M-ar4ep0-seed4",
        "260729-torquerate-400M-tr8em4-seed4",
    ]
    baseline_config = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps=400000000",
            "--env_name=TestEnvironment",
            '--playground_config_overrides={"reward_config.scales.action_rate": -4}',
        ]
    }
    missing_config = recovery.subprocess.CalledProcessError(
        1, ("ssh", "eagle", "cat")
    )
    ssh.side_effect = [
        recovery.json.dumps(baseline_config),
        missing_config,
        "Submitted batch job 123",
    ]

    result = recovery.main(
        [
            "TestEnvironment",
            "--run-pattern=260729-*",
            "--execute",
        ]
    )

    self.assertEqual(result, 0)
    self.assertEqual(ssh.call_count, 3)
    submitted_command = ssh.call_args_list[-1].args[1]
    self.assertIn("sbatch --array=0,1,2,3", submitted_command)
    self.assertIn("slurm.sh ar 4e+0 TestEnvironment", submitted_command)


if __name__ == "__main__":
  unittest.main()
