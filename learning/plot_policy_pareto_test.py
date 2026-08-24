"""Tests for Pareto-front plotting helpers."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from absl.testing import absltest

from learning import plot_policy_pareto


class PlotPolicyParetoTest(absltest.TestCase):

  def test_default_metrics_cover_torque_variation_and_savgol_timescales(self):
    for cutoff in (1, 2, 5, 10, 15, 20):
      self.assertNotIn(
          f"torque_spectrum/eval/fft_above_{cutoff}hz_energy_per_step",
          plot_policy_pareto.DEFAULT_Y_METRICS,
      )
    for window, polyorder in ((5, 2), (11, 3), (21, 3), (31, 3)):
      self.assertIn(
          f"smoothness/torque/msgfd_w{window}_p{polyorder}_"
          "mean_absolute_savgol_filter_deviation_per_dof",
          plot_policy_pareto.DEFAULT_Y_METRICS,
      )
    self.assertIn(
        "smoothness/torque/total_variation",
        plot_policy_pareto.DEFAULT_Y_METRICS,
    )
    self.assertIn(
        "smoothness/torque/sign_changes_total",
        plot_policy_pareto.DEFAULT_Y_METRICS,
    )
    self.assertFalse(
        any(
            metric.startswith("smoothness/base_")
            for metric in plot_policy_pareto.DEFAULT_Y_METRICS
        )
    )

  def test_metric_labels_are_readable_and_explain_pareto_direction(self):
    self.assertEqual(
        plot_policy_pareto._metric_label(plot_policy_pareto.DEFAULT_X_METRIC),
        "Task reward without regularization\n(higher is better)",
    )
    for metric in plot_policy_pareto.DEFAULT_Y_METRICS:
      label = plot_policy_pareto._metric_label(metric)
      self.assertNotEqual(label, metric)
      self.assertNotIn("_", label)
      self.assertIn("lower is better", label)

  def test_custom_metric_label_has_a_readable_fallback(self):
    self.assertEqual(
        plot_policy_pareto._metric_label("custom/mean_foot_error"),
        "Mean foot error",
    )

  def test_percent_above_minimum_scaling(self):
    np.testing.assert_allclose(
        plot_policy_pareto._percent_above_minimum(
            np.array([2.0, 2.5, 4.0]), 2.0, "smoothness"
        ),
        [0.0, 25.0, 100.0],
    )

  def test_percent_above_minimum_requires_positive_baseline(self):
    with self.assertRaisesRegex(ValueError, "minimum must be positive"):
      plot_policy_pareto._percent_above_minimum(
          np.array([0.0, 1.0]), 0.0, "smoothness"
      )

  def test_exponential_x_axis_transform_is_invertible(self):
    forward, inverse = plot_policy_pareto._exponential_axis_functions(
        20.0, 40.0, 2.0
    )
    values = np.array([20.0, 25.0, 30.0, 35.0, 40.0])

    np.testing.assert_allclose(inverse(forward(values)), values)
    self.assertLess(forward(30.0), 0.5)

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
            / "policy_pareto_x-linear_y-absolute-linear_repr-labels.png",
        ),
    )
    self.assertEqual(plot.call_args_list[0].args[5], (29.7, None))
    self.assertTrue(
        all(call.kwargs["all_methods"] for call in plot.call_args_list)
    )
    self.assertEqual(
        [call.args[2].name for call in plot.call_args_list],
        [
            "policy_pareto_x-linear_y-absolute-linear_repr-labels.png",
            "policy_pareto_x-linear_y-absolute-linear_repr-size.png",
            "policy_pareto_x-linear_y-absolute-linear_repr-opacity.png",
            "policy_pareto_x-linear_y-absolute-linear_repr-arrows.png",
        ],
    )
    self.assertTrue(
        all(call.kwargs["hide_non_pareto"] for call in plot.call_args_list)
    )
    self.assertFalse(
        any(
            call.kwargs["y_percent_above_minimum"]
            for call in plot.call_args_list
        )
    )
    self.assertTrue(
        all(call.kwargs["log_percentage_y"] for call in plot.call_args_list)
    )
    self.assertFalse(
        any(call.kwargs["log_y_axis"] for call in plot.call_args_list)
    )

  def test_output_names_describe_axes_and_representation(self):
    base = Path("custom.png")

    self.assertEqual(
        plot_policy_pareto._output_variant_path(
            base,
            "labels",
            y_percent_above_minimum=False,
            log_y_axis=True,
            log_percentage_y=True,
            shifted_log_percentage_y=False,
            x_exponential_strength=1.5,
        ),
        Path("custom_x-exp-1p5_y-absolute-log_repr-labels.png"),
    )
    self.assertEqual(
        plot_policy_pareto._output_variant_path(
            base,
            "arrows",
            y_percent_above_minimum=True,
            log_y_axis=False,
            log_percentage_y=False,
            shifted_log_percentage_y=True,
            x_exponential_strength=0.0,
        ),
        Path("custom_x-linear_y-percent-shifted-log_repr-arrows.png"),
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

  def test_plotted_methods_are_an_explicit_filter_and_order(self):
    points = [
        plot_policy_pareto.Point(
            method=method,
            scale=1.0,
            scale_tag="1",
            x=1.0,
            y=1.0,
            sample_count=1,
            seed_count=1,
            expected_seed_count=1,
            missing_seeds="",
        )
        for method in ("unknown", "high_pass", "baseline")
    ]

    self.assertEqual(
        [point.method for point in plot_policy_pareto._filter_methods(points)],
        ["high_pass", "baseline"],
    )
    self.assertEqual(
        plot_policy_pareto._method_order(points), ["baseline", "high_pass"]
    )
    self.assertEqual(
        plot_policy_pareto._method_order(points, all_methods=True),
        ["baseline", "high_pass", "unknown"],
    )

  def test_configured_high_pass_variants_have_distinct_colors(self):
    methods = (
        "action_smoothness",
        "baseline",
        "torque_rate",
        "torque_smoothness",
        "high_pass",
        "high_pass_f2_m1",
        "high_pass_f3_m1",
        "high_pass_f4_m1",
        "high_pass_f5_m0p5",
        "high_pass_f5_m1p5",
        "high_pass_f5_m2",
        "high_pass_f6_m1",
    )

    colors = [plot_policy_pareto._method_color(method) for method in methods]
    self.assertLen(set(colors), len(methods))

    f5_rgb = np.asarray(
        plot_policy_pareto.matplotlib_colors.to_rgb(
            plot_policy_pareto._method_color("high_pass")
        )
    )
    f6_rgb = np.asarray(
        plot_policy_pareto.matplotlib_colors.to_rgb(
            plot_policy_pareto._method_color("high_pass_f6_m1")
        )
    )
    self.assertGreater(np.linalg.norm(f5_rgb - f6_rgb), 0.5)

  def test_cluster_cli_accepts_only_environment_and_flag(self):
    arguments = plot_policy_pareto._build_parser().parse_args([
        "Go1JoystickFlatTerrain",
        "--cluster",
    ])

    self.assertEqual(arguments.environment, "Go1JoystickFlatTerrain")
    self.assertTrue(arguments.cluster)

  def test_all_methods_cli_is_opt_in(self):
    parser = plot_policy_pareto._build_parser()

    self.assertFalse(parser.parse_args([]).all_methods)
    self.assertTrue(parser.parse_args(["--all-methods"]).all_methods)

  def test_method_cli_accepts_an_ordered_subset(self):
    arguments = plot_policy_pareto._build_parser().parse_args([
        "--method",
        "baseline",
        "--method",
        "torque_rate",
    ])

    self.assertEqual(arguments.methods, ["baseline", "torque_rate"])

  def test_environment_plot_config_can_select_a_method_subset(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "Environment.toml").write_text(
          """all_methods = false
methods = [
  "baseline",
  # Temporarily omitted methods can remain visible here as comments.
  "high_pass_f6_m1",
]
""",
          encoding="utf-8",
      )

      self.assertEqual(
          plot_policy_pareto._configured_methods("Environment", root),
          (False, ("baseline", "high_pass_f6_m1")),
      )

  def test_environment_plot_config_defaults_to_all_when_missing(self):
    with tempfile.TemporaryDirectory() as directory:
      self.assertEqual(
          plot_policy_pareto._configured_methods("Unknown", Path(directory)),
          (True, ()),
      )

  def test_non_pareto_policies_are_hidden_by_default(self):
    parser = plot_policy_pareto._build_parser()

    self.assertTrue(parser.parse_args([]).hide_non_pareto)
    self.assertFalse(
        parser.parse_args(["--no-hide-non-pareto"]).hide_non_pareto
    )
    self.assertFalse(parser.parse_args([]).y_percent_above_minimum)
    self.assertTrue(
        parser.parse_args(["--y-percent-above-minimum"])
        .y_percent_above_minimum
    )
    self.assertFalse(
        parser.parse_args(["--absolute-y-values"])
        .y_percent_above_minimum
    )
    self.assertFalse(parser.parse_args([]).log_y_axis)
    logarithmic = parser.parse_args(["--log-y-axis"])
    self.assertTrue(logarithmic.log_y_axis)
    self.assertFalse(logarithmic.y_percent_above_minimum)
    self.assertTrue(parser.parse_args([]).log_percentage_y)
    self.assertFalse(
        parser.parse_args(["--linear-percentage-y-axis"]).log_percentage_y
    )
    shifted = parser.parse_args(["--shifted-log-percentage-y-axis"])
    self.assertTrue(shifted.shifted_log_percentage_y)

  def test_log_y_axis_is_forwarded_and_named_for_every_representation(self):
    with mock.patch.object(
        plot_policy_pareto, "plot", return_value=Path("result.png")
    ) as plot, mock.patch.object(
        plot_policy_pareto,
        "_default_source",
        return_value=(Path("manifest"), Path("evaluations")),
    ):
      plot_policy_pareto.main([
          "Go1JoystickRoughTerrain",
          "--log-y-axis",
          "--x-exponential-strength",
          "2",
          "--output",
          "custom.png",
      ])

    self.assertTrue(
        all(call.kwargs["log_y_axis"] for call in plot.call_args_list)
    )
    self.assertEqual(
        [call.args[2].name for call in plot.call_args_list],
        [
            "custom_x-exp-2_y-absolute-log_repr-labels.png",
            "custom_x-exp-2_y-absolute-log_repr-size.png",
            "custom_x-exp-2_y-absolute-log_repr-opacity.png",
            "custom_x-exp-2_y-absolute-log_repr-arrows.png",
        ],
    )

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

  def test_logarithmic_absolute_y_axis_renders(self):
    manifest, evaluation_root, _ = self._evaluation_fixture()
    output = evaluation_root / "logarithmic.png"

    actual = plot_policy_pareto.plot(
        manifest,
        evaluation_root,
        output,
        plot_policy_pareto.DEFAULT_X_METRIC,
        ("metric",),
        log_y_axis=True,
        all_methods=True,
    )

    self.assertEqual(actual, output)
    self.assertTrue(output.is_file())
    self.assertTrue(output.with_suffix(".csv").is_file())

  def test_no_xlim_keeps_all_points(self):
    points = [
        plot_policy_pareto.Point(
            "baseline", 0.01, "1em2", 28.0, 1.0, 100, 5, 5, ""
        )
    ]

    self.assertEqual(plot_policy_pareto._filter_points(points, None), points)
    self.assertEqual(
        plot_policy_pareto._filter_points(points, (31.0, None)), []
    )

  def test_environment_xlim_config_and_unlisted_fallback(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      path = Path(temporary_directory) / "xlim.json"
      path.write_text('{"Known": 12.5}', encoding="utf-8")

      self.assertEqual(
          plot_policy_pareto._configured_xlim("Known", path), (12.5, None)
      )
      self.assertIsNone(
          plot_policy_pareto._configured_xlim("Unlisted", path)
      )

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
