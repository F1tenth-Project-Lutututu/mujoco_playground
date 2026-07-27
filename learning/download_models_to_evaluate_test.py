"""Tests for learning.download_models_to_evaluate."""

from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
  unittest.main()
