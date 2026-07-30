"""Tests for cluster evaluation coverage reporting."""

from pathlib import PurePosixPath
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
        "_evaluated_run_names",
        return_value={"run-a", "run-b", "stale-run"},
    ):
      rows = report.collect_coverage(
          "eagle", PurePosixPath("/logs"), PurePosixPath("/evaluations")
      )

    self.assertEqual(
        rows,
        [report.Coverage("Go1JoystickFlatTerrain", 2, 3)],
    )
    self.assertEqual(rows[0].missing_runs, 1)

  def test_formats_table_and_status(self):
    table = report.format_table([
        report.Coverage("Go1JoystickFlatTerrain", 3, 3),
        report.Coverage("SpotJoystick", 2, 5),
    ])

    self.assertIn("Environment", table)
    self.assertIn("Go1JoystickFlatTerrain", table)
    self.assertIn("COMPLETE", table)
    self.assertIn("INCOMPLETE", table)


if __name__ == "__main__":
  absltest.main()
