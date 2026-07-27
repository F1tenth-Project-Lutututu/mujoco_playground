"""Tests for learning.plot_all_policy_evaluations."""

from pathlib import Path
import tempfile
import unittest

from learning import plot_all_policy_evaluations


class PlotAllPolicyEvaluationsTest(unittest.TestCase):

  def test_first_manifest_family_uses_first_noncomment_record(self):
    with tempfile.TemporaryDirectory() as directory:
      manifest = Path(directory) / "to_evaluate.txt"
      manifest.write_text(
          "\n# reference follows\nbaseline # preferred\nhighpass\n",
          encoding="utf-8",
      )
      self.assertEqual(
          plot_all_policy_evaluations._first_manifest_family(manifest),
          "baseline",
      )

  def test_discover_methods_groups_seeds_and_skips_incomplete_runs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for run in ("baseline-seed0", "baseline-seed1", "highpass-seed2"):
        (root / run / "100").mkdir(parents=True)
        (root / run / "100" / "summary.json").touch()
      (root / "incomplete-seed0").mkdir()

      methods = plot_all_policy_evaluations.discover_methods(root)

      self.assertEqual(list(methods), ["baseline", "highpass"])
      self.assertEqual(methods["baseline"].name, "baseline-seed0")
      self.assertEqual(methods["highpass"].name, "highpass-seed2")


if __name__ == "__main__":
  unittest.main()
