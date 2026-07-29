"""Tests for the localhost Pareto cluster controller."""

from pathlib import Path
import tempfile
import unittest

from learning import pareto_cluster


class ParetoClusterTest(unittest.TestCase):

  def test_submit_defaults_to_large_h100_rollout_batch(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      manifest = Path(temporary_directory) / "manifest.json"
      arguments = pareto_cluster._build_parser().parse_args([
          "submit",
          "--manifest",
          str(manifest),
      ])

    self.assertEqual(arguments.num_random_tasks, 2048)
    self.assertEqual(
        arguments.remote_models_root,
        pareto_cluster.downloader.DEFAULT_REMOTE_LOGS,
    )

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


if __name__ == "__main__":
  unittest.main()
