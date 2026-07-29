"""Tests for learning.download_models_to_evaluate."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from learning import download_models_to_evaluate


class DownloadModelsToEvaluateTest(unittest.TestCase):

  def test_read_policy_families_ignores_comments_and_duplicates(self):
    with tempfile.TemporaryDirectory() as directory:
      manifest = Path(directory) / "to_evaluate.txt"
      manifest.write_text(
          "# policies\nbaseline\nhighpass # note\nbaseline\n",
          encoding="utf-8",
      )
      self.assertEqual(
          download_models_to_evaluate._read_policy_families(manifest),
          ["baseline", "highpass"],
      )

  def test_matching_runs_selects_all_seeds_only(self):
    matches = download_models_to_evaluate._matching_runs(
        ["baseline-seed2", "baseline-seed0", "baseline-extra-seed1"],
        ["baseline"],
    )
    self.assertEqual(
        matches, {"baseline": ["baseline-seed0", "baseline-seed2"]}
    )

  def test_copy_remote_retries_and_atomically_replaces_partial_target(self):
    with tempfile.TemporaryDirectory() as directory:
      local_parent = Path(directory)
      target = local_parent / "checkpoint"
      target.mkdir()
      (target / "partial").write_text("incomplete", encoding="utf-8")
      calls = 0

      def fake_run(arguments, **unused_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
          raise subprocess.CalledProcessError(1, arguments)
        staged_target = Path(arguments[-1]) / "checkpoint"
        staged_target.mkdir()
        (staged_target / "_sharding").write_text("complete", encoding="utf-8")

      with (
          mock.patch.object(
              download_models_to_evaluate, "_run", side_effect=fake_run
          ),
          mock.patch.object(download_models_to_evaluate.time, "sleep") as sleep,
      ):
        download_models_to_evaluate._copy_remote(
            "eagle",
            download_models_to_evaluate.PurePosixPath("/runs/checkpoint"),
            local_parent,
            attempts=2,
            retry_delay_seconds=0.25,
        )

      self.assertEqual(calls, 2)
      sleep.assert_called_once_with(0.25)
      self.assertFalse((target / "partial").exists())
      self.assertEqual(
          (target / "_sharding").read_text(encoding="utf-8"), "complete"
      )


if __name__ == "__main__":
  unittest.main()
