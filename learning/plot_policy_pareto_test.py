"""Tests for Pareto-front plotting helpers."""

import json
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np

from learning import plot_policy_pareto


class PlotPolicyParetoTest(absltest.TestCase):

  def test_pareto_mask_maximizes_x_and_minimizes_y(self):
    actual = plot_policy_pareto.pareto_mask(
        np.array([1.0, 2.0, 3.0, 2.0]),
        np.array([1.0, 2.0, 3.0, 3.0]),
    )

    np.testing.assert_array_equal(actual, [True, True, True, False])

  def _evaluation_fixture(self) -> tuple[Path, Path, Path]:
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    manifest = root / "pareto_manifest.json"
    evaluation_root = root / "evaluations"
    report = (
        evaluation_root
        / "raw_torque"
        / "run-seed0"
        / "000400000000"
        / "rollouts.csv"
    )
    report.parent.mkdir(parents=True)
    report.write_text(
        "eval_reward_means/total_without_regularization,metric\n"
        "32,2\n"
        "34,4\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({
            "runs": [{
                "method": "baseline",
                "scale": 0.01,
                "scale_tag": "1em2",
                "run_name": "run-seed0",
                "checkpoint": "000400000000",
            }]
        }),
        encoding="utf-8",
    )
    return manifest, evaluation_root, report

  def test_aggregate_cache_is_reused_while_inputs_are_unchanged(self):
    manifest, evaluation_root, _ = self._evaluation_fixture()
    first = plot_policy_pareto.load_aggregates(manifest, evaluation_root)

    with mock.patch.object(
        plot_policy_pareto,
        "_build_aggregates",
        side_effect=AssertionError("cache should be reused"),
    ):
      second = plot_policy_pareto.load_aggregates(
          manifest, evaluation_root
      )

    self.assertEqual(first, second)
    self.assertEqual(float(second[0]["metric"]), 3.0)

  def test_aggregate_cache_is_invalidated_when_rollout_changes(self):
    manifest, evaluation_root, report = self._evaluation_fixture()
    plot_policy_pareto.load_aggregates(manifest, evaluation_root)
    report.write_text(
        "eval_reward_means/total_without_regularization,metric\n"
        "32,8\n"
        "34,10\n",
        encoding="utf-8",
    )

    rebuilt = plot_policy_pareto.load_aggregates(
        manifest, evaluation_root
    )

    self.assertEqual(float(rebuilt[0]["metric"]), 9.0)


if __name__ == "__main__":
  absltest.main()
