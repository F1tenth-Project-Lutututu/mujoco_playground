"""Tests for Pareto-front plotting helpers."""

import json
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import absltest
import numpy as np

from learning import plot_policy_pareto


class PlotPolicyParetoTest(absltest.TestCase):

  def test_environment_only_sets_default_input_and_output_paths(self):
    with mock.patch.object(
        plot_policy_pareto, "plot", return_value=Path("result.png")
    ) as plot:
      plot_policy_pareto.main(["Go1JoystickRoughTerrain"])

    self.assertEqual(
        plot.call_args.args[:3],
        (
            plot_policy_pareto.pareto_policy_pipeline.DEFAULT_LOCAL_ROOT
            / "Go1JoystickRoughTerrain"
            / plot_policy_pareto.pareto_policy_pipeline.MANIFEST_NAME,
            plot_policy_pareto.pareto_policy_pipeline.DEFAULT_OUTPUT_ROOT
            / "Go1JoystickRoughTerrain",
            plot_policy_pareto.pareto_policy_pipeline.DEFAULT_OUTPUT_ROOT
            / "Go1JoystickRoughTerrain"
            / "policy_pareto.png",
        ),
    )
    self.assertIsNone(plot.call_args.args[5])

  def test_xlim_is_forwarded_to_plot(self):
    with mock.patch.object(
        plot_policy_pareto, "plot", return_value=Path("result.png")
    ) as plot:
      plot_policy_pareto.main([
          "Go1JoystickRoughTerrain",
          "--xlim",
          "25",
          "40",
      ])

    self.assertEqual(plot.call_args.args[5], (25.0, 40.0))

  def test_single_xlim_sets_only_lower_bound(self):
    with mock.patch.object(
        plot_policy_pareto, "plot", return_value=Path("result.png")
    ) as plot:
      plot_policy_pareto.main([
          "Go1JoystickRoughTerrain",
          "--xlim",
          "25",
      ])

    self.assertEqual(plot.call_args.args[5], (25.0, None))

  def test_pareto_mask_maximizes_x_and_minimizes_y(self):
    actual = plot_policy_pareto.pareto_mask(
        np.array([1.0, 2.0, 3.0, 2.0]),
        np.array([1.0, 2.0, 3.0, 3.0]),
    )

    np.testing.assert_array_equal(actual, [True, True, True, False])

  def test_cluster_cli_accepts_only_environment_and_flag(self):
    arguments = plot_policy_pareto._build_parser().parse_args([
        "Go1JoystickFlatTerrain",
        "--cluster",
    ])

    self.assertEqual(arguments.environment, "Go1JoystickFlatTerrain")
    self.assertTrue(arguments.cluster)

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
                "seed": 0,
                "run_name": "run-seed0",
                "checkpoint": "000400000000",
            }],
            "seed_coverage": [{
                "method": "baseline",
                "scale_tag": "1em2",
                "expected_seeds": [0, 1],
                "completed_seeds": [0],
                "all_seeds_available": False,
            }],
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
    self.assertEqual(int(second[0]["seed_count"]), 1)
    self.assertEqual(int(second[0]["expected_seed_count"]), 2)
    self.assertEqual(second[0]["missing_seeds"], "1")

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

  def test_aggregates_union_expected_seeds_from_legacy_manifest(self):
    manifest, evaluation_root, report = self._evaluation_fixture()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value.pop("seed_coverage")
    second_run = {
        **value["runs"][0],
        "seed": 1,
        "run_name": "run-seed1",
    }
    value["runs"].append(second_run)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    second_report = (
        evaluation_root
        / "raw_torque"
        / "run-seed1"
        / "000400000000"
        / "rollouts.csv"
    )
    second_report.parent.mkdir(parents=True)
    second_report.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")

    aggregates = plot_policy_pareto.load_aggregates(
        manifest, evaluation_root
    )

    self.assertEqual(int(aggregates[0]["seed_count"]), 2)
    self.assertEqual(int(aggregates[0]["expected_seed_count"]), 2)
    self.assertEqual(aggregates[0]["missing_seeds"], "")


if __name__ == "__main__":
  absltest.main()
