"""Tests for batched metric recomputation from trajectory archives."""

import csv
import json
from pathlib import Path
import tempfile

from absl.testing import absltest
import numpy as np

from learning import recompute_evaluation_metrics


class RecomputeEvaluationMetricsTest(absltest.TestCase):

  def test_mssd_batches_rollouts_and_respects_active_lengths(self):
    qvel = np.zeros((5, 2, 8))
    qvel[:, 0, 6] = [0.0, 1.0, 0.0, 1.0, 0.0]
    qvel[:, 1, 6] = [0.0, 1.0, 0.0, 99.0, 99.0]
    active = np.asarray([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 0],
        [1, 0],
    ], dtype=bool)

    result = recompute_evaluation_metrics.compute_metrics(
        {"qvel": qvel, "active": active},
        (recompute_evaluation_metrics.JOINT_VELOCITY_MSSD,),
        savgol_window_length=5,
        savgol_polyorder=2,
    )

    np.testing.assert_allclose(
        result[recompute_evaluation_metrics.JOINT_VELOCITY_MSSD],
        [2.0, 2.0],
    )

  def test_msgfd_batches_equal_length_rollouts(self):
    qvel = np.zeros((5, 3, 8))
    qvel[:, :, 6] = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    result = recompute_evaluation_metrics.compute_metrics(
        {"qvel": qvel, "active": np.ones((5, 3), dtype=bool)},
        (recompute_evaluation_metrics.JOINT_VELOCITY_MSGFD,),
        savgol_window_length=5,
        savgol_polyorder=2,
    )

    values = result[recompute_evaluation_metrics.JOINT_VELOCITY_MSGFD]
    self.assertGreater(values[0], 0.0)
    np.testing.assert_allclose(values, values[0] * np.asarray([1.0, 2.0, 3.0]))

  def test_update_evaluation_updates_rollouts_and_summaries(self):
    with tempfile.TemporaryDirectory() as temporary:
      evaluation = Path(temporary)
      scenario = evaluation / "random_tasks"
      scenario.mkdir()
      qvel = np.zeros((5, 2, 8))
      qvel[:, :, 6] = np.asarray([
          [0.0, 0.0],
          [1.0, 2.0],
          [0.0, 0.0],
          [1.0, 2.0],
          [0.0, 0.0],
      ])
      np.savez_compressed(
          scenario / "signals.npz",
          qvel=qvel,
          active=np.ones((5, 2), dtype=bool),
      )
      rows = [
          {"scenario": "random_tasks", "rollout": index}
          for index in range(2)
      ]
      for path in (evaluation / "rollouts.csv",):
        with path.open("w", newline="", encoding="utf-8") as file:
          writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
          writer.writeheader()
          writer.writerows(rows)
      summary = {
          "metadata": {
              "savgol_window_length": 5,
              "savgol_polyorder": 2,
          },
          "overall": {},
          "scenarios": {"random_tasks": {}},
      }
      (evaluation / "summary.json").write_text(json.dumps(summary))
      (scenario / "summary.json").write_text("{}")
      (evaluation / "summary.csv").write_text(
          "checkpoint\n000400000000\n", encoding="utf-8"
      )

      recompute_evaluation_metrics.update_evaluation(
          evaluation,
          (recompute_evaluation_metrics.JOINT_VELOCITY_MSSD,),
      )

      with (evaluation / "rollouts.csv").open(
          newline="", encoding="utf-8"
      ) as file:
        updated_rows = list(csv.DictReader(file))
      self.assertAlmostEqual(
          float(updated_rows[0][
              recompute_evaluation_metrics.JOINT_VELOCITY_MSSD
          ]),
          2.0,
      )
      updated_summary = json.loads(
          (evaluation / "summary.json").read_text(encoding="utf-8")
      )
      self.assertIn(
          f"{recompute_evaluation_metrics.JOINT_VELOCITY_MSSD}/mean",
          updated_summary["overall"],
      )


if __name__ == "__main__":
  absltest.main()
