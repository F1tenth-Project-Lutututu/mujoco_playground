"""Tests for the policy Pareto pipeline."""

import copy
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import absltest

from learning import pareto_policy_pipeline


class ParetoPolicyPipelineTest(absltest.TestCase):

  def test_decode_scale(self):
    self.assertEqual(pareto_policy_pipeline.decode_scale("1em3"), 1e-3)
    self.assertEqual(pareto_policy_pipeline.decode_scale("2ep2"), 2e2)

  def test_select_runs_keeps_latest_duplicate(self):
    runs = pareto_policy_pipeline.select_runs([
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
            "260727-baseline-400M-ar1em2-seed0",
            "260727-highpass-400M-hp4em3-f5o1m10-seed2",
            "260727-torquerate-400M-tr2em4-seed1",
        ],
    )

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


if __name__ == "__main__":
  absltest.main()
