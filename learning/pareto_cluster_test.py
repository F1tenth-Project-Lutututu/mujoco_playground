"""Tests for the localhost Pareto cluster controller."""

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from learning import evaluate_pareto_on_cluster, pareto_cluster


class ParetoClusterTest(unittest.TestCase):

  def test_cluster_evaluation_does_not_save_full_signals(self):
    self.assertFalse(evaluate_pareto_on_cluster.SAVE_FULL_SIGNALS)

  def test_submit_defaults_to_1024_parallel_environments(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      manifest = Path(temporary_directory) / "manifest.json"
      arguments = pareto_cluster._build_parser().parse_args([
          "submit",
          "--manifest",
          str(manifest),
      ])

    self.assertEqual(arguments.num_random_tasks, 1024)
    self.assertEqual(
        arguments.remote_models_root,
        pareto_cluster.downloader.DEFAULT_REMOTE_LOGS,
    )

  def test_submit_accepts_environment_without_manifest(self):
    arguments = pareto_cluster._build_parser().parse_args([
        "submit",
        "Go1JoystickFlatTerrain",
        "--run-date",
        "260729",
    ])

    self.assertEqual(arguments.environment, "Go1JoystickFlatTerrain")
    self.assertIsNone(arguments.manifest)
    self.assertEqual(arguments.run_date, 260729)

  def test_submit_accepts_parallel_policy_shards(self):
    arguments = pareto_cluster._build_parser().parse_args([
        "submit",
        "BarkourJoystick",
        "--run-date",
        "260902",
        "--shards",
        "8",
    ])

    self.assertEqual(arguments.shards, 8)

  def test_sharded_submit_prepares_then_submits_an_array(self):
    arguments = pareto_cluster._build_parser().parse_args([
        "submit",
        "BarkourJoystick",
        "--run-date",
        "260902",
        "--shards",
        "4",
    ])
    with mock.patch.object(
        pareto_cluster, "_ssh", side_effect=("10", "11")
    ) as ssh:
      pareto_cluster.submit(arguments)

    preparation, array_submission = [call.args[1] for call in ssh.call_args_list]
    self.assertIn("--wait", preparation)
    self.assertIn("prepare-only", preparation)
    self.assertIn("--array=0-3", array_submission)
    self.assertIn("pareto_pending_manifest.json", array_submission)

  def test_fetch_defaults_to_cluster_result_roots(self):
    arguments = pareto_cluster._build_parser().parse_args([
        "fetch",
        "Go1JoystickFlatTerrain",
    ])

    self.assertEqual(
        arguments.remote_output_root,
        pareto_cluster.DEFAULT_REMOTE_OUTPUT_ROOT,
    )
    self.assertEqual(
        arguments.local_output_root,
        pareto_cluster.DEFAULT_LOCAL_OUTPUT_ROOT,
    )
    self.assertFalse(arguments.include_signals)
    self.assertEqual(arguments.archive_cpus, 8)

  def test_fetch_pack_command_uses_parallel_gzip_and_excludes_signals(self):
    command = pareto_cluster._pack_command(
        remote_archive=pareto_cluster.PurePosixPath("/tmp/results.tar.gz"),
        remote_output_root=pareto_cluster.PurePosixPath("/remote/results"),
        environment="Go1JoystickFlatTerrain",
        include_signals=False,
        cpus=8,
    )

    self.assertIn("pigz -1 -p 8", command)
    self.assertIn("gzip -1", command)
    self.assertIn("--exclude=*/signals.npz", command)
    self.assertIn("--exclude=*/rollout.mp4", command)

  def test_fetch_pack_command_can_include_signals(self):
    command = pareto_cluster._pack_command(
        remote_archive=pareto_cluster.PurePosixPath("/tmp/results.tar.gz"),
        remote_output_root=pareto_cluster.PurePosixPath("/remote/results"),
        environment="Go1JoystickFlatTerrain",
        include_signals=True,
        cpus=4,
    )

    self.assertNotIn("signals.npz", command)
    self.assertNotIn("rollout.mp4", command)

  def test_metrics_accepts_repeatable_metric_selection(self):
    metric = (
        "smoothness/joint_velocity/"
        "mssd_mean_squared_second_difference_per_dof"
    )
    arguments = pareto_cluster._build_parser().parse_args([
        "metrics",
        "Go1JoystickFlatTerrain",
        "--metric",
        metric,
    ])

    self.assertEqual(arguments.metric, [metric])


if __name__ == "__main__":
  unittest.main()
