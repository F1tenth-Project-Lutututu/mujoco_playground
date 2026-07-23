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
"""Tests for policy evaluation comparison helpers."""

import csv
import json
import math
from pathlib import Path

from absl.testing import absltest

from learning import compare_policy_evaluations


class ComparePolicyEvaluationsTest(absltest.TestCase):

  def _write_evaluation(self, directory, seed, values):
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "summary.json").write_text(
        json.dumps({
            "metadata": {"seed": seed},
            "scenarios": {"random_tasks": {}},
        })
    )
    with (checkpoint / "rollouts.csv").open("w", newline="") as fp:
      writer = csv.DictWriter(fp, fieldnames=("scenario", "task", "score"))
      writer.writeheader()
      for task, value in enumerate(values):
        writer.writerow({
            "scenario": "random_tasks",
            "task": task,
            "score": value,
        })

  def test_policy_seed_paths_and_rollouts_include_all_sibling_seeds(self):
    parent = Path(self.create_tempdir().full_path)
    seed0 = parent / "policy-seed0"
    seed1 = parent / "policy-seed1"
    # The report seed is the evaluation RNG seed and may be identical across
    # policy seeds, so task namespacing must use the directory name.
    self._write_evaluation(seed0, 0, [1, 3])
    self._write_evaluation(seed1, 0, [5, 7])

    paths = compare_policy_evaluations._policy_seed_paths(seed0)
    summary, rows = compare_policy_evaluations._aggregate_rollouts(
        paths, ["score"]
    )

    self.assertEqual(paths, [seed0, seed1])
    self.assertEqual(summary["scenarios"]["random_tasks"]["score/mean"], 4)
    self.assertAlmostEqual(
        summary["scenarios"]["random_tasks"]["score/std"], math.sqrt(5)
    )
    self.assertEqual(
        [row["task"] for row in rows], ["0:0", "0:1", "1:0", "1:1"]
    )

  def test_load_summary_accepts_evaluation_directory(self):
    directory = Path(self.create_tempdir().full_path)
    report = {"metadata": {}, "scenarios": {"stand": {}}}
    (directory / "summary.json").write_text(json.dumps(report))
    self.assertEqual(
        compare_policy_evaluations._load_summary(directory), report
    )

  def test_load_summary_accepts_run_directory_with_one_checkpoint(self):
    directory = Path(self.create_tempdir().full_path)
    checkpoint_directory = directory / "checkpoint_100"
    checkpoint_directory.mkdir()
    report = {"metadata": {}, "scenarios": {"stand": {}}}
    (checkpoint_directory / "summary.json").write_text(json.dumps(report))
    self.assertEqual(
        compare_policy_evaluations._load_summary(directory), report
    )

  def test_run_directory_with_multiple_checkpoints_requires_selection(self):
    directory = Path(self.create_tempdir().full_path)
    for checkpoint in ("checkpoint_100", "checkpoint_200"):
      checkpoint_directory = directory / checkpoint
      checkpoint_directory.mkdir()
      (checkpoint_directory / "summary.json").write_text(
          json.dumps({"metadata": {}, "scenarios": {}})
      )
    with self.assertRaisesRegex(ValueError, "multiple evaluated checkpoints"):
      compare_policy_evaluations._summary_path(directory)

  def test_comparison_rows_use_aggregate_mean_and_std(self):
    summaries = [{
        "scenarios": {
            "stand": {
                "tracking/error/mean": 1.5,
                "tracking/error/std": 0.25,
            }
        }
    }]
    scenarios, rows = compare_policy_evaluations._comparison_rows(
        summaries, ["policy"], ["tracking/error"]
    )
    self.assertEqual(scenarios, ["stand"])
    self.assertEqual(rows[0]["mean"], 1.5)
    self.assertEqual(rows[0]["std"], 0.25)

  def test_comparison_rows_keep_union_of_scenarios(self):
    summaries = [
        {"scenarios": {"stand": {"score/mean": 1.0}}},
        {"scenarios": {"walk": {"score/mean": 2.0}}},
    ]
    scenarios, rows = compare_policy_evaluations._comparison_rows(
        summaries, ["first", "second"], ["score"]
    )
    self.assertEqual(scenarios, ["stand", "walk"])
    missing = next(
        row
        for row in rows
        if row["evaluation"] == "first" and row["scenario"] == "walk"
    )
    self.assertTrue(math.isnan(missing["mean"]))

  def test_unique_labels_disambiguates_duplicates(self):
    self.assertEqual(
        compare_policy_evaluations._unique_labels(["run", "run", "other"]),
        ["run", "run (2)", "other"],
    )

  def test_labels_include_aggregated_seed_counts(self):
    self.assertEqual(
        compare_policy_evaluations._labels_with_seed_counts(
            ["first", "second"],
            [
                [Path("first-seed0")],
                [Path("second-seed0"), Path("second-seed1")],
            ],
        ),
        ["first (1 seed)", "second (2 seeds)"],
    )

  def test_smoothness_acronyms_have_compact_plot_titles(self):
    self.assertEqual(
        compare_policy_evaluations._display_name(
            "smoothness/action/mssd_mean_squared_second_difference_per_dof"
        ),
        "MSSD",
    )
    self.assertEqual(
        compare_policy_evaluations._display_name(
            "smoothness/action/"
            "msgfd_mean_absolute_savgol_filter_deviation_per_dof"
        ),
        "MSGFD",
    )

  def test_torque_smoothness_titles_include_signal_and_units(self):
    self.assertEqual(
        compare_policy_evaluations._display_name(
            "smoothness/torque/"
            "mssd_mean_squared_second_difference_per_dof"
        ),
        "torque MSSD (N²·m²)",
    )
    self.assertEqual(
        compare_policy_evaluations._display_name(
            "smoothness/torque/"
            "msgfd_mean_absolute_savgol_filter_deviation_per_dof"
        ),
        "torque MSGFD (N·m)",
    )

  def test_panel_titles_show_metric_improvement_direction(self):
    self.assertEqual(
        compare_policy_evaluations._panel_title(
            "eval_reward_means/total_without_regularization"
        ),
        "total without regularization\n↑ better",
    )
    self.assertEqual(
        compare_policy_evaluations._panel_title(
            "tracking/linear_velocity_vector_rmse"
        ),
        "linear velocity vector rmse\n↓ better",
    )

  def test_normalized_panel_titles_always_point_up(self):
    self.assertEqual(
        compare_policy_evaluations._panel_title(
            "tracking/linear_velocity_vector_rmse", normalized=True
        ),
        "linear velocity vector rmse\n↑ better",
    )

  def test_paired_delta_rows_subtract_matched_reference_tasks(self):
    reference = [
        {"scenario": "random_tasks", "task": "0", "score": "10"},
        {"scenario": "random_tasks", "task": "1", "score": "30"},
    ]
    method = [
        {"scenario": "random_tasks", "task": "0", "score": "13"},
        {"scenario": "random_tasks", "task": "1", "score": "28"},
    ]
    rows = compare_policy_evaluations._paired_delta_rows(
        [reference, method], ["baseline", "method"], ["score"], "baseline"
    )
    self.assertEqual([row["delta"] for row in rows], [3.0, -2.0])

  def test_paired_delta_rows_reject_unmatched_tasks(self):
    reference = [{"scenario": "random_tasks", "task": "0", "score": "1"}]
    method = [{"scenario": "random_tasks", "task": "1", "score": "1"}]
    with self.assertRaisesRegex(ValueError, "same tasks"):
      compare_policy_evaluations._paired_delta_rows(
          [reference, method],
          ["baseline", "method"],
          ["score"],
          "baseline",
      )

  def test_paired_validation_falls_back_to_environment_foot_target(self):
    common_metadata = {
        "evaluation_mode": "random_tasks",
        "task_seed": 0,
        "num_random_tasks": 2,
    }
    reference = {
        "metadata": common_metadata,
        "environment_config": {
            "reward_config": {"max_foot_height": 0.1}
        },
    }
    method = {
        "metadata": {
            **common_metadata,
            "feet_height_target_meters": 0.1,
        },
        "environment_config": {
            "reward_config": {"max_foot_height": 0.1}
        },
    }
    compare_policy_evaluations._validate_paired_reports(
        [reference, method], ["baseline", "method"], "baseline"
    )

  def test_paired_delta_rows_skip_metrics_missing_from_old_reports(self):
    reference = [{"scenario": "random_tasks", "task": "0", "score": "1"}]
    method = [{"scenario": "random_tasks", "task": "0", "score": "2"}]
    rows = compare_policy_evaluations._paired_delta_rows(
        [reference, method],
        ["baseline", "method"],
        ["score", "new_metric"],
        "baseline",
    )
    self.assertLen(rows, 1)
    self.assertEqual(rows[0]["metric"], "score")

  def test_paired_percent_rows_orient_improvement_by_metric_direction(self):
    rows = [
        {"metric": "reward", "reference_value": 10.0, "delta": 2.0},
        {"metric": "error", "reference_value": 10.0, "delta": -2.0},
        {"metric": "error", "reference_value": 10.0, "delta": 2.0},
    ]

    result = compare_policy_evaluations._paired_percent_rows(
        rows, {"reward": 1, "error": -1}
    )

    self.assertSequenceAlmostEqual(
        [row["percent_improvement"] for row in result],
        [20.0, 20.0, -20.0],
    )

  def test_paired_percent_rows_use_absolute_reference_denominator(self):
    rows = [{"metric": "reward", "reference_value": -10.0, "delta": 2.0}]

    result = compare_policy_evaluations._paired_percent_rows(
        rows, {"reward": 1}
    )

    self.assertAlmostEqual(result[0]["percent_improvement"], 20.0)

  def test_paired_percent_rows_skip_zero_reference_values(self):
    rows = [{"metric": "error", "reference_value": 0.0, "delta": 1.0}]

    result = compare_policy_evaluations._paired_percent_rows(
        rows, {"error": -1}
    )

    self.assertEmpty(result)

  def test_paired_percent_rows_require_metric_direction(self):
    rows = [{"metric": "score", "reference_value": 1.0, "delta": 1.0}]

    with self.assertRaisesRegex(ValueError, "needs an improvement direction"):
      compare_policy_evaluations._paired_percent_rows(rows, {})

  def test_values_within_iqr_removes_extreme_plot_outliers(self):
    values = compare_policy_evaluations.np.asarray([0, 0, 0, 0, 100])

    filtered = compare_policy_evaluations._values_within_iqr(values, 1.5)

    self.assertSequenceEqual(filtered.tolist(), [0, 0, 0, 0])

  def test_values_within_iqr_can_keep_all_points(self):
    values = compare_policy_evaluations.np.asarray([0, 1, 100])

    filtered = compare_policy_evaluations._values_within_iqr(values, None)

    self.assertSequenceEqual(filtered.tolist(), [0, 1, 100])

  def test_figure_size_uses_configured_aspect_ratio(self):
    width, height = compare_policy_evaluations._figure_size()

    self.assertAlmostEqual(
        width / height, compare_policy_evaluations.FIGURE_ASPECT_RATIO
    )


if __name__ == "__main__":
  absltest.main()
