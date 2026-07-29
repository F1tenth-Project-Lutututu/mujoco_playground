"""Tests for the policy Pareto pipeline."""

import copy

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


if __name__ == "__main__":
  absltest.main()
