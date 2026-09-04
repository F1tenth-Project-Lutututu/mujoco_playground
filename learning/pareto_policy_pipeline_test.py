"""Tests for the policy Pareto pipeline."""

import copy
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from learning import pareto_policy_pipeline


class ParetoPolicyPipelineTest(absltest.TestCase):

  def test_decode_scale(self):
    self.assertEqual(pareto_policy_pipeline.decode_scale("1em3"), 1e-3)
    self.assertEqual(pareto_policy_pipeline.decode_scale("2ep2"), 2e2)

  def test_select_runs_keeps_latest_duplicate(self):
    runs = pareto_policy_pipeline.select_runs([
        "260727-actionsmoothness-400M-as1em3-seed3",
        "260722-baseline-400M-ar1em2-seed0",
        "260727-baseline-400M-ar1em2-seed0",
        "260727-torquerate-400M-tr2em4-seed1",
        "260727-highpass-400M-hp4em3-f5o1m10-seed2",
        "260727-highpass-400M-hp4em3-f7o1m10-seed2",
        "unrelated",
    ])

    self.assertEqual(
        [run.run_name for run in runs],
        [
            "260727-actionsmoothness-400M-as1em3-seed3",
            "260727-baseline-400M-ar1em2-seed0",
            "260727-highpass-400M-hp4em3-f5o1m10-seed2",
            "260727-highpass-400M-hp4em3-f7o1m10-seed2",
            "260727-torquerate-400M-tr2em4-seed1",
        ],
    )

  def test_select_runs_recognizes_action_smoothness_sweep(self):
    runs = pareto_policy_pipeline.select_runs([
        "260817-actionsmoothness-1000M-as1em1-seed0",
        "260817-actionsmoothness-1000M-as1ep1-seed4",
    ])

    self.assertEqual([run.method for run in runs], [
        "action_smoothness",
        "action_smoothness",
    ])
    self.assertEqual([run.scale for run in runs], [0.1, 10.0])

  def test_select_runs_recognizes_optional_torque_smoothness_sweep(self):
    runs = pareto_policy_pipeline.select_runs([
        "260817-baseline-400M-ar1em2-seed0",
        "260818-torquesmoothness-400M-ts4em4-seed2",
    ])

    self.assertEqual(
        [(run.method, run.scale, run.seed) for run in runs],
        [
            ("baseline", 0.01, 0),
            ("torque_smoothness", 0.0004, 2),
        ],
    )

  def test_select_runs_separates_high_pass_cutoff_and_m_sweeps(self):
    runs = pareto_policy_pipeline.select_runs([
        "260729-highpass-400M-hp1em4-f5o1m10-seed0",
        "260729-highpass-400M-hp1em4-f10o1m10-seed0",
        "260729-highpass-400M-hp1em4-f5o1m20-seed0",
        "260729-highpass-400M-hp1em4-f2p5o1m0p5-seed0",
    ])

    self.assertEqual(
        {run.method for run in runs},
        {
            "high_pass",
            "high_pass_f10_m1",
            "high_pass_f5_m2",
            "high_pass_f2p5_m0p5",
        },
    )
    configured = next(
        run for run in runs if run.method == "high_pass_f2p5_m0p5"
    )
    self.assertEqual(configured.cutoff_hz, 2.5)
    self.assertEqual(configured.difference_order, 0.5)

  def test_select_runs_separates_high_pass_filter_orders(self):
    runs = pareto_policy_pipeline.select_runs([
        "260729-highpass-400M-hp1em4-f5o1m10-seed0",
        "260731-highpass-400M-hp1em4-f5o2m10-seed0",
    ])

    self.assertEqual(
        {(run.method, run.filter_order) for run in runs},
        {("high_pass", 1), ("high_pass_f5_o2_m1", 2)},
    )

  def test_select_runs_recognizes_policy_observed_high_pass_sweep(self):
    runs = pareto_policy_pipeline.select_runs([
        "260903-highpass-400M-hpp1em4-f5o1m10-seed0",
    ])

    self.assertEqual(runs[0].method, "high_pass_policy")
    self.assertEqual(runs[0].run_name,
                     "260903-highpass-400M-hpp1em4-f5o1m10-seed0")

  def test_select_runs_can_keep_every_matching_run(self):
    names = [
        "260902-highpass-400M-hpp1em4-f5o1m10-seed0",
        "260903-highpass-400M-hpp1em4-f5o1m10-seed0",
    ]

    runs = pareto_policy_pipeline.select_runs(names, keep_all=True)

    self.assertEqual([run.run_name for run in runs], names)

  def test_select_runs_can_restrict_experiment_date(self):
    runs = pareto_policy_pipeline.select_runs(
        [
            "260728-baseline-400M-ar2em2-seed0",
            "260729-baseline-400M-ar1em5-seed0",
            "260729-highpass-400M-hp1em5-f5o1m10-seed0",
        ],
        run_date=260729,
    )

    self.assertEqual(
        [run.run_name for run in runs],
        [
            "260729-baseline-400M-ar1em5-seed0",
            "260729-highpass-400M-hp1em5-f5o1m10-seed0",
        ],
    )

  def test_select_runs_accepts_non_400m_training_duration(self):
    runs = pareto_policy_pipeline.select_runs([
        "260730-baseline-1000M-ar1em1-seed0",
    ])

    self.assertEqual(
        [run.run_name for run in runs],
        ["260730-baseline-1000M-ar1em1-seed0"],
    )

  def test_select_runs_can_restrict_nominal_training_duration(self):
    runs = pareto_policy_pipeline.select_runs(
        [
            "260817-baseline-400M-ar1em1-seed0",
            "260817-baseline-1000M-ar1em1-seed0",
            "260817-actionsmoothness-1000M-as1em1-seed0",
        ],
        training_steps_millions=1000,
    )

    self.assertEqual(
        [run.run_name for run in runs],
        [
            "260817-actionsmoothness-1000M-as1em1-seed0",
            "260817-baseline-1000M-ar1em1-seed0",
        ],
    )

  def test_environment_checkpoint_targets(self):
    self.assertEqual(
        pareto_policy_pipeline.evaluation_target_step("BarkourJoystick"),
        400_000_000,
    )
    self.assertEqual(
        pareto_policy_pipeline.evaluation_target_step(
            "Go1JoystickFlatTerrain25"
        ),
        1_000_000_000,
    )

  def test_select_checkpoint_uses_nearest_and_prefers_later_ties(self):
    self.assertEqual(
        pareto_policy_pipeline.select_checkpoint_name(
            ["not-a-step", "380000000", "410000000"], 400_000_000
        ),
        "410000000",
    )

  def test_mixed_horizon_target_uses_common_400m_terminal_step(self):
    runs = [
        pareto_policy_pipeline.PolicyRun(
            "action_smoothness",
            "1em5",
            1e-5,
            0,
            260817,
            "short-a",
            training_steps_millions=400,
        ),
        pareto_policy_pipeline.PolicyRun(
            "action_smoothness",
            "2em2",
            2e-2,
            0,
            260817,
            "long-a",
            training_steps_millions=1000,
        ),
        pareto_policy_pipeline.PolicyRun(
            "action_smoothness",
            "1em5",
            1e-5,
            1,
            260817,
            "short-b",
            training_steps_millions=400,
        ),
    ]

    target = pareto_policy_pipeline.mixed_horizon_evaluation_target(
        runs,
        {
            "short-a": [392_000_000, 417_792_000],
            "short-b": [390_000_000, 410_000_000],
            "long-a": [400_000_000, 425_000_000, 1_000_000_000],
        },
        400_000_000,
    )

    self.assertEqual(target, 410_000_000)
    self.assertEqual(
        pareto_policy_pipeline.select_checkpoint_name(
            ["390000000", "410000000"], 400_000_000
        ),
        "410000000",
    )

  def test_checkpoint_complete_requires_network_config(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      checkpoint = Path(temporary_directory)
      (checkpoint / "_sharding").touch()
      self.assertFalse(
          pareto_policy_pipeline._checkpoint_complete(checkpoint)
      )
      (checkpoint / "ppo_network_config.json").touch()
      self.assertTrue(
          pareto_policy_pipeline._checkpoint_complete(checkpoint)
      )

  def test_seed_coverage_reports_failed_seed(self):
    selected = [
        pareto_policy_pipeline.PolicyRun(
            "baseline", "1em2", 0.01, seed, 260729, f"run-seed{seed}"
        )
        for seed in (0, 1)
    ]
    completed = [
        pareto_policy_pipeline.PolicyRun(
            "baseline", "1em2", 0.01, 0, 260729, "run-seed0", "400"
        )
    ]
    skipped = [{
        "method": "baseline",
        "scale_tag": "1em2",
        "seed": 1,
        "run_name": "run-seed1",
        "reason": "no numeric checkpoint exists",
    }]

    coverage = pareto_policy_pipeline._seed_coverage(
        selected, completed, skipped
    )

    self.assertFalse(coverage[0]["all_seeds_available"])
    self.assertEqual(coverage[0]["completed_seeds"], [0])
    self.assertEqual(coverage[0]["failed_seeds"][0]["seed"], 1)

  def test_delete_remote_run_is_limited_to_validated_run_name(self):
    run = pareto_policy_pipeline.PolicyRun(
        "baseline",
        "1em3",
        1e-3,
        1,
        260729,
        "260729-baseline-400M-ar1em3-seed1",
    )
    root = pareto_policy_pipeline.PurePosixPath("/remote/logs/environment")

    with mock.patch.object(
        pareto_policy_pipeline.downloader, "_run"
    ) as run_command:
      pareto_policy_pipeline._delete_remote_run_without_checkpoint(
          "eagle", root, run
      )

    run_command.assert_called_once_with((
        "ssh",
        "eagle",
        "rm -rf -- /remote/logs/environment/"
        "260729-baseline-400M-ar1em3-seed1",
    ))

  def test_comparable_config_ignores_only_seed_scale_and_provenance(self):
    config = {
        "created_at": "now",
        "seed": 3,
        "command": ["train"],
        "ppo_config": {"seed": 3, "learning_rate": 3e-4},
        "environment_config": {
            "reward_config": {"scales": {"action_rate": -0.01}}
        },
        "environment_config_overrides": {
            "reward_config.scales.action_rate": -0.01
        },
    }
    other = copy.deepcopy(config)
    other["created_at"] = "later"
    other["seed"] = 4
    other["ppo_config"]["seed"] = 4
    other["environment_config"]["reward_config"]["scales"][
        "action_rate"
    ] = -0.02
    other["environment_config_overrides"][
        "reward_config.scales.action_rate"
    ] = -0.02

    self.assertEqual(
        pareto_policy_pipeline._comparable_run_config(config, "baseline"),
        pareto_policy_pipeline._comparable_run_config(other, "baseline"),
    )

  def test_comparable_config_can_ignore_mixed_training_horizon(self):
    short = {
        "ppo_config": {
            "seed": 0,
            "num_timesteps": 400_000_000,
            "num_evals": 16,
            "learning_rate": 3e-4,
        },
        "environment_config": {
            "reward_config": {"scales": {"action_rate": -1e-5}}
        },
        "environment_config_overrides": {
            "reward_config.scales.action_rate": -1e-5
        },
    }
    long = copy.deepcopy(short)
    long["ppo_config"]["num_timesteps"] = 1_000_000_000
    long["ppo_config"]["num_evals"] = 40
    long["environment_config"]["reward_config"]["scales"][
        "action_rate"
    ] = -0.02
    long["environment_config_overrides"][
        "reward_config.scales.action_rate"
    ] = -0.02

    self.assertEqual(
        pareto_policy_pipeline._comparable_run_config(
            short, "action_smoothness", ignore_training_horizon=True
        ),
        pareto_policy_pipeline._comparable_run_config(
            long, "action_smoothness", ignore_training_horizon=True
        ),
    )

  def test_comparable_config_fills_missing_environment_defaults(self):
    older = {
        "ppo_config": {"seed": 0},
        "environment_config": {
            "reward_config": {"scales": {"torque_high_freq": -1e-4}}
        },
        "environment_config_overrides": {
            "reward_config.scales.torque_high_freq": -1e-4
        },
    }
    newer = copy.deepcopy(older)
    newer["environment_config"]["domain_randomization"] = False
    defaults = {"domain_randomization": False}

    self.assertEqual(
        pareto_policy_pipeline._comparable_run_config(
            older, "high_pass", defaults
        ),
        pareto_policy_pipeline._comparable_run_config(
            newer, "high_pass", defaults
        ),
    )

  def test_action_smoothness_comparison_sweeps_action_rate_scale(self):
    config = {
        "environment_config": {
            "reward_config": {
                "action_rate_use_second_difference": True,
                "scales": {"action_rate": -0.001},
            }
        },
        "environment_config_overrides": {
            "reward_config.action_rate_use_second_difference": True,
            "reward_config.scales.action_rate": -0.001,
        },
    }
    other = copy.deepcopy(config)
    other["environment_config"]["reward_config"]["scales"][
        "action_rate"
    ] = -0.1
    other["environment_config_overrides"][
        "reward_config.scales.action_rate"
    ] = -0.1

    self.assertEqual(
        pareto_policy_pipeline._comparable_run_config(
            config, "action_smoothness"
        ),
        pareto_policy_pipeline._comparable_run_config(
            other, "action_smoothness"
        ),
    )

  def test_torque_smoothness_comparison_sweeps_torque_rate_scale(self):
    config = {
        "environment_config": {
            "reward_config": {
                "torque_rate_use_second_difference": True,
                "scales": {"torque_rate": -0.001},
            }
        },
        "environment_config_overrides": {
            "reward_config.torque_rate_use_second_difference": True,
            "reward_config.scales.torque_rate": -0.001,
        },
    }
    other = copy.deepcopy(config)
    other["environment_config"]["reward_config"]["scales"][
        "torque_rate"
    ] = -0.1
    other["environment_config_overrides"][
        "reward_config.scales.torque_rate"
    ] = -0.1

    self.assertEqual(
        pareto_policy_pipeline._comparable_run_config(
            config, "torque_smoothness"
        ),
        pareto_policy_pipeline._comparable_run_config(
            other, "torque_smoothness"
        ),
    )


if __name__ == "__main__":
  absltest.main()
