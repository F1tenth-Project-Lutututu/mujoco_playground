"""Tests for Pareto-front plotting helpers."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from absl.testing import absltest

from learning import plot_policy_pareto


class PlotPolicyParetoTest(absltest.TestCase):

  def test_default_metrics_include_joint_velocity_mssd_and_msgfd(self):
    self.assertIn(
        "smoothness/joint_velocity/"
        "mssd_mean_squared_second_difference_per_dof",
        plot_policy_pareto.DEFAULT_Y_METRICS,
    )
    self.assertIn(
        "smoothness/joint_velocity/"
        "msgfd_mean_absolute_savgol_filter_deviation_per_dof",
        plot_policy_pareto.DEFAULT_Y_METRICS,
    )

  def test_require_metrics_explains_that_evaluation_must_be_regenerated(self):
    with self.assertRaisesRegex(
        plot_policy_pareto.MissingEvaluationMetricsError,
        "Regenerate the policy evaluations",
    ):
      plot_policy_pareto._require_metrics(
          [{"existing": "1.0"}], ("existing", "new_metric")
      )

  def test_environment_only_sets_default_input_and_output_paths(self):
    expected_manifest = Path("local-manifest.json")
    expected_evaluation_root = Path("local-evaluations")
    with mock.patch.object(
        plot_policy_pareto, "plot", return_value=Path("result.png")
    ) as plot, mock.patch.object(
        plot_policy_pareto,
        "_default_source",
        return_value=(expected_manifest, expected_evaluation_root),
    ):
      plot_policy_pareto.main(["Go1JoystickRoughTerrain"])

    self.assertEqual(
        plot.call_args_list[0].args[:3],
        (
            expected_manifest,
            expected_evaluation_root,
            plot_policy_pareto.DEFAULT_RESULTS_ROOT
            / "Go1JoystickRoughTerrain"
            / "policy_pareto.png",
        ),
    )
    self.assertIsNone(plot.call_args_list[0].args[5])
    self.assertEqual(
        [call.args[2].name for call in plot.call_args_list],
        [
            "policy_pareto.png",
            "policy_pareto_size.png",
            "policy_pareto_opacity.png",
            "policy_pareto_arrows.png",
        ],
    )

  def test_xlim_is_forwarded_to_plot(self):
    with mock.patch.object(
        plot_policy_pareto, "plot", return_value=Path("result.png")
    ) as plot, mock.patch.object(
        plot_policy_pareto,
        "_default_source",
        return_value=(Path("manifest"), Path("evaluations")),
    ):
      plot_policy_pareto.main([
          "Go1JoystickRoughTerrain",
          "--xlim",
          "25",
          "40",
      ])

    self.assertEqual(plot.call_args_list[0].args[5], (25.0, 40.0))

  def test_single_xlim_sets_only_lower_bound(self):
    with mock.patch.object(
        plot_policy_pareto, "plot", return_value=Path("result.png")
    ) as plot, mock.patch.object(
        plot_policy_pareto,
        "_default_source",
        return_value=(Path("manifest"), Path("evaluations")),
    ):
      plot_policy_pareto.main([
          "Go1JoystickRoughTerrain",
          "--xlim",
          "25",
      ])

    self.assertEqual(plot.call_args_list[0].args[5], (25.0, None))

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

  def test_default_source_prefers_complete_local_evaluations(self):
    local = (Path("local-manifest"), Path("local-root"))
    cluster = (Path("cluster-manifest"), Path("cluster-root"))
    with mock.patch.object(
        plot_policy_pareto, "_source_is_available", side_effect=[True, True]
    ) as available, mock.patch.object(
        plot_policy_pareto.pareto_policy_pipeline,
        "DEFAULT_LOCAL_ROOT",
        local[0].parent,
    ), mock.patch.object(
        plot_policy_pareto.pareto_policy_pipeline,
        "DEFAULT_OUTPUT_ROOT",
        local[1].parent,
    ), mock.patch.object(
        plot_policy_pareto, "DEFAULT_CLUSTER_ROOT", cluster[1].parent
    ):
      actual = plot_policy_pareto._default_source("environment")

    self.assertEqual(
        actual,
        (
            local[0].parent
            / "environment"
            / plot_policy_pareto.pareto_policy_pipeline.MANIFEST_NAME,
            local[1].parent / "environment",
        ),
    )
    available.assert_called_once()

  def test_default_source_falls_back_to_complete_cluster_evaluations(self):
    with mock.patch.object(
        plot_policy_pareto, "_source_is_available", side_effect=[False, True]
    ):
      manifest, evaluation_root = plot_policy_pareto._default_source(
          "environment"
      )

    self.assertEqual(
        evaluation_root,
        plot_policy_pareto.DEFAULT_CLUSTER_ROOT / "environment",
    )
    self.assertEqual(
        manifest,
        evaluation_root / plot_policy_pareto.pareto_policy_pipeline.MANIFEST_NAME,
    )

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

  def test_report_paths_use_sole_evaluated_checkpoint_as_fallback(self):
    manifest, evaluation_root, report = self._evaluation_fixture()
    expected = report.parent.parent / "000419430400" / "rollouts.csv"
    report.parent.rename(expected.parent)

    reports = plot_policy_pareto._report_paths(manifest, evaluation_root)

    self.assertEqual(reports[0][1], expected)

  def test_report_paths_reject_ambiguous_checkpoint_fallback(self):
    manifest, evaluation_root, report = self._evaluation_fixture()
    second = report.parent.parent / "000419430400" / "rollouts.csv"
    second.parent.mkdir()
    second.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["runs"][0]["checkpoint"] = "000410000000"
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with self.assertRaisesRegex(FileNotFoundError, "ambiguous"):
      plot_policy_pareto._report_paths(manifest, evaluation_root)

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
