"""Tests for detailed cluster evaluation diagnosis."""

from absl.testing import absltest

from learning import diagnose_cluster_evaluations as diagnose


class DiagnoseClusterEvaluationsTest(absltest.TestCase):

  def test_classifies_missing_and_partial_runs(self):
    rows = [
        {"run": "no-checkpoint", "latest_checkpoint": None},
        {
            "run": "stale",
            "latest_checkpoint": "20",
            "evaluation_checkpoints": ["10"],
        },
        {
            "run": "partial",
            "latest_checkpoint": "20",
            "evaluation_checkpoints": ["20"],
            "artifacts": {
                "rollouts.csv": {"exists": True, "size": 10, "valid": True},
                "summary.json": {"exists": False, "size": 0},
            },
        },
    ]

    actual = [diagnose._diagnose_row("Go1Joystick", row) for row in rows]

    self.assertEqual(
        [row.status for row in actual],
        ["NO_CHECKPOINT", "STALE_CHECKPOINT", "PARTIAL_OUTPUT"],
    )
    self.assertIn("missing summary.json", actual[2].detail)

  def test_complete_requires_valid_reports(self):
    artifacts = {
        "rollouts.csv": {"exists": True, "size": 10, "valid": True},
        "summary.json": {"exists": True, "size": 10, "valid": True},
    }
    row = {
        "run": "run-a",
        "latest_checkpoint": "20",
        "evaluation_checkpoints": ["20"],
        "artifacts": artifacts,
    }
    self.assertEqual(
        diagnose._diagnose_row("Go1Joystick", row).status, "COMPLETE"
    )

  def test_formatter_hides_complete_rows_by_default(self):
    rows = [
        diagnose.Diagnosis("Go1Joystick", "good", "20", "COMPLETE", "ok"),
        diagnose.Diagnosis(
            "Go1Joystick", "bad", "20", "NOT_EVALUATED", "no output"
        ),
    ]

    output = diagnose.format_diagnoses(rows)

    self.assertNotIn("good", output)
    self.assertIn("bad", output)
    self.assertIn("NOT_EVALUATED", output)


if __name__ == "__main__":
  absltest.main()
