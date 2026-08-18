"""Tests for cluster evaluation coverage reporting."""

import tempfile
from pathlib import Path, PurePosixPath
from unittest import mock

from absl.testing import absltest

from learning import report_cluster_evaluations as report


class ReportClusterEvaluationsTest(absltest.TestCase):

  def test_discovers_only_matching_quadruped_joystick_environments(self):
    with mock.patch.object(
        report.downloader,
        "_ssh_lines",
        return_value=[
            "Go1JoystickFlatTerrain",
            "SpotJoystickGaitTracking",
            "BerkeleyHumanoidJoystickFlatTerrain",
            "G1JoystickFlatTerrain",
            "T1JoystickFlatTerrain",
            "Go1Stand",
        ],
    ):
      names = report._environment_names(
          "eagle", PurePosixPath("/logs"), "*Flat*"
      )

    self.assertEqual(names, ["Go1JoystickFlatTerrain"])

  def test_collects_unique_reports_matching_saved_runs(self):
    with mock.patch.object(
        report,
        "_environment_names",
        return_value=["Go1JoystickFlatTerrain"],
    ), mock.patch.object(
        report.downloader,
        "_remote_run_names",
        return_value=["run-a", "run-b", "run-c"],
    ), mock.patch.object(
        report,
        "_evaluation_manifest",
        return_value={
            "runs": [
                {"run_name": "run-a"},
                {"run_name": "run-b"},
                {"run_name": "run-c"},
            ],
            "skipped_runs": [],
        },
    ), mock.patch.object(
        report,
        "_evaluation_artifact_run_names",
        return_value=(
            {"run-a", "run-b", "stale-run"},
            {"run-c"},
        ),
    ):
      rows = report.collect_coverage(
          "eagle", PurePosixPath("/logs"), PurePosixPath("/evaluations")
      )

    self.assertEqual(
        rows,
        [
            report.Coverage(
                "Go1JoystickFlatTerrain", 2, 3, 1,
                method_progress="other 2/3"
            )
        ],
    )
    self.assertEqual(rows[0].missing_runs, 1)

  def test_ignores_hidden_administrative_directories(self):
    with mock.patch.object(
        report.downloader,
        "_remote_run_names",
        return_value=["run-a", ".incomplete-runs"],
    ), mock.patch.object(
        report,
        "_evaluation_manifest",
        return_value=None,
    ), mock.patch.object(
        report,
        "_evaluation_artifact_run_names",
        return_value=({"run-a"}, set()),
    ):
      row = report._collect_environment_coverage(
          "eagle",
          PurePosixPath("/logs"),
          PurePosixPath("/evaluations"),
          "Go1JoystickFlatTerrain",
      )

    self.assertEqual(
        row,
        report.Coverage(
            "Go1JoystickFlatTerrain",
            1,
            1,
            manifest_available=False,
            method_progress="other 1/1",
        ),
    )

  def test_comprehensive_plan_includes_old_and_current_pareto_methods(self):
    run_names = [
        "260729-baseline-400M-ar1em2-seed0",
        "260729-torquerate-400M-tr1em4-seed0",
        "260818-torquesmoothness-400M-ts1em4-seed0",
    ]
    with mock.patch.object(
        report.downloader, "_remote_run_names", return_value=run_names
    ), mock.patch.object(
        report,
        "_evaluation_manifest",
        return_value={
            "runs": [{"run_name": run_names[-1]}],
            "skipped_runs": [],
        },
    ), mock.patch.object(
        report,
        "_evaluation_artifact_run_names",
        return_value=({run_names[0], run_names[1]}, {run_names[2]}),
    ):
      row = report._collect_environment_coverage(
          "eagle",
          PurePosixPath("/logs"),
          PurePosixPath("/evaluations"),
          "Go1JoystickFlatTerrain",
      )

    self.assertEqual(row.evaluated_runs, 2)
    self.assertEqual(row.active_runs, 1)
    self.assertEqual(row.planned_runs, 3)
    self.assertEqual(row.missing_runs, 1)
    self.assertEqual(row.method_progress, "ar 1/1, tr 1/1, ts 0/1")

  def test_formats_table_and_status(self):
    table = report.format_table([
        report.Coverage("Go1JoystickFlatTerrain", 3, 3),
        report.Coverage("SpotJoystick", 2, 5, 1, 2),
    ])

    self.assertIn("Environment", table)
    self.assertIn("Go1JoystickFlatTerrain", table)
    self.assertIn("COMPLETE", table)
    self.assertIn("RUNNING", table)

  def test_reuses_cached_coverage(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      cache_file = Path(temporary_directory) / "coverage.json"
      expected = [report.Coverage("Go1JoystickFlatTerrain", 3, 3)]
      with mock.patch.object(
          report, "collect_coverage", return_value=expected
      ) as collect, mock.patch.object(
          report,
          "_environment_names",
          return_value=["Go1JoystickFlatTerrain"],
      ), mock.patch.object(
          report, "_changed_environment_names", return_value=set()
      ):
        first = report.collect_coverage_cached(
            "eagle",
            PurePosixPath("/logs"),
            PurePosixPath("/evaluations"),
            cache_file=cache_file,
        )
        second = report.collect_coverage_cached(
            "eagle",
            PurePosixPath("/logs"),
            PurePosixPath("/evaluations"),
            cache_file=cache_file,
        )

      self.assertEqual(first, expected)
      self.assertEqual(second, expected)
      collect.assert_called_once()

  def test_refreshes_only_environments_changed_since_previous_invocation(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      cache_file = Path(temporary_directory) / "coverage.json"
      initial = [
          report.Coverage("Go1JoystickFlatTerrain", 1, 3),
          report.Coverage("SpotJoystick", 2, 5),
      ]
      updated = report.Coverage("Go1JoystickFlatTerrain", 2, 3)
      with mock.patch.object(
          report, "collect_coverage", return_value=initial
      ), mock.patch.object(
          report,
          "_environment_names",
          return_value=["Go1JoystickFlatTerrain", "SpotJoystick"],
      ), mock.patch.object(
          report,
          "_changed_environment_names",
          return_value={"Go1JoystickFlatTerrain"},
      ), mock.patch.object(
          report,
          "_collect_environment_coverage",
          side_effect=[updated, initial[1]],
      ) as collect_environment:
        report.collect_coverage_cached(
            "eagle",
            PurePosixPath("/logs"),
            PurePosixPath("/evaluations"),
            cache_file=cache_file,
        )
        rows = report.collect_coverage_cached(
            "eagle",
            PurePosixPath("/logs"),
            PurePosixPath("/evaluations"),
            cache_file=cache_file,
        )

      self.assertEqual(rows, [updated, initial[1]])
      self.assertEqual(collect_environment.call_count, 2)

  def test_refresh_cache_queries_cluster_again(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      cache_file = Path(temporary_directory) / "coverage.json"
      with mock.patch.object(
          report,
          "collect_coverage",
          side_effect=[
              [report.Coverage("Go1JoystickFlatTerrain", 1, 3)],
              [report.Coverage("Go1JoystickFlatTerrain", 2, 3)],
          ],
      ) as collect:
        report.collect_coverage_cached(
            "eagle",
            PurePosixPath("/logs"),
            PurePosixPath("/evaluations"),
            cache_file=cache_file,
        )
        refreshed = report.collect_coverage_cached(
            "eagle",
            PurePosixPath("/logs"),
            PurePosixPath("/evaluations"),
            cache_file=cache_file,
            refresh=True,
        )

      self.assertEqual(refreshed[0].evaluated_runs, 2)
      self.assertEqual(collect.call_count, 2)


if __name__ == "__main__":
  absltest.main()
