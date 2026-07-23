"""Tests for learning.evaluate_all_models."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from learning import evaluate_all_models


class EvaluateAllModelsTest(unittest.TestCase):

  def test_model_directories_only_include_runs_with_checkpoints(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      root = Path(temporary_directory)
      (root / "model-b" / "checkpoints").mkdir(parents=True)
      (root / "model-a" / "checkpoints").mkdir(parents=True)
      (root / "notes").mkdir()

      actual = evaluate_all_models._model_directories(root)

      self.assertEqual([path.name for path in actual], ["model-a", "model-b"])

  def test_select_checkpoint_uses_latest_numeric_directory(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      model = Path(temporary_directory) / "model"
      checkpoints = model / "checkpoints"
      (checkpoints / "9").mkdir(parents=True)
      (checkpoints / "10").mkdir()
      (checkpoints / "not-a-checkpoint").mkdir()

      with mock.patch.object(evaluate_all_models, "CHECKPOINT_NAME", None):
        actual = evaluate_all_models._select_checkpoint(model)

      self.assertEqual(actual.name, "10")

  def test_select_checkpoint_rejects_incomplete_run(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      model = Path(temporary_directory) / "model"
      checkpoints = model / "checkpoints"
      checkpoints.mkdir(parents=True)
      (checkpoints / "config.json").write_text("{}")

      with mock.patch.object(evaluate_all_models, "CHECKPOINT_NAME", None):
        with self.assertRaisesRegex(ValueError, "No numeric checkpoints"):
          evaluate_all_models._select_checkpoint(model)

  def test_arguments_use_paired_random_task_and_fast_defaults(self):
    arguments = evaluate_all_models._evaluation_arguments(
        Path("checkpoint"), Path("output"), True
    )

    self.assertIn("--num_random_tasks", arguments)
    self.assertEqual(
        arguments[arguments.index("--num_random_tasks") + 1],
        str(evaluate_all_models.NUM_RANDOM_TASKS),
    )
    self.assertIn("--deterministic", arguments)
    self.assertIn("--disable_perturbations", arguments)
    self.assertIn("--no-render_video", arguments)
    self.assertIn("--no-save_signals", arguments)
    self.assertIn("--require_cuda", arguments)
    self.assertIn("--no-use_saved_environment_config", arguments)
    self.assertIn("--torque_highpass_normalize_by_capacity", arguments)

  def test_arguments_can_select_raw_torque_mode(self):
    arguments = evaluate_all_models._evaluation_arguments(
        Path("checkpoint"),
        Path("output"),
        False,
    )

    self.assertIn("--no-torque_highpass_normalize_by_capacity", arguments)

  def test_hash_changes_when_checkpoint_contents_change(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      checkpoint = Path(temporary_directory) / "checkpoint"
      checkpoint.mkdir()
      parameter = checkpoint / "parameter"
      parameter.write_bytes(b"before")
      before = evaluate_all_models._hash_paths([checkpoint])

      parameter.write_bytes(b"after")
      after = evaluate_all_models._hash_paths([checkpoint])

      self.assertNotEqual(before, after)

  def test_cache_manifest_round_trip(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
      manifest = Path(temporary_directory) / "evaluation_cache.json"
      signature = {"cache_format_version": 1, "value": [1, 2, 3]}

      evaluate_all_models._write_cache_manifest(manifest, signature)

      self.assertEqual(
          evaluate_all_models._read_cache_manifest(manifest), signature
      )



if __name__ == "__main__":
  unittest.main()
