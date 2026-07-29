import tempfile
import unittest
from pathlib import Path
from unittest import mock

from learning import recover_stalled_slurm_runs


class RecoverStalledSlurmRunsTest(unittest.TestCase):

  def test_failure_reason_detects_cpu_fallback(self):
    with tempfile.TemporaryDirectory() as directory:
      stderr = Path(directory) / "task.err"
      stderr.write_text("operation cuInit failed: CUDA_ERROR_NO_DEVICE\n")
      self.assertEqual(
          recover_stalled_slurm_runs._failure_reason(stderr),
          "CUDA_ERROR_NO_DEVICE",
      )

  def test_find_failed_tasks_only_returns_active_array_children(self):
    rows = [
        ("123", "RUNNING", "sbatch slurm.sh tr 4e-5 Go1"),
        ("123_0", "COMPLETED", "sbatch slurm.sh tr 4e-5 Go1"),
        ("123_1", "RUNNING", "sbatch slurm.sh tr 4e-5 Go1"),
        ("123_2", "RUNNING", "sbatch slurm.sh tr 4e-5 Go1"),
    ]
    with tempfile.TemporaryDirectory() as directory:
      bad = Path(directory) / "bad.err"
      good = Path(directory) / "good.err"
      bad.write_text("Falling back to cpu.")
      good.write_text("GPU ready")
      properties = {
          "123_1": {"StdErr": str(bad)},
          "123_2": {"StdErr": str(good)},
      }
      with (
          mock.patch.object(
              recover_stalled_slurm_runs,
              "_accounting_rows",
              return_value=rows,
          ),
          mock.patch.object(
              recover_stalled_slurm_runs,
              "_job_properties",
              side_effect=lambda job_id: properties[job_id],
          ),
      ):
        tasks = recover_stalled_slurm_runs.find_failed_tasks("123")
    self.assertEqual([task.array_index for task in tasks], [1])

  def test_array_override_replaces_original_option(self):
    command = ["sbatch", "--array", "0-4", "--qos=x", "slurm.sh", "tr"]
    self.assertEqual(
        recover_stalled_slurm_runs._without_array_option(command),
        ["sbatch", "--qos=x", "slurm.sh", "tr"],
    )


if __name__ == "__main__":
  unittest.main()
