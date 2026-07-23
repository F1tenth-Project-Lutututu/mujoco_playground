# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluate models using one ``./eagle`` run's environment configuration.

This script intentionally has no command-line arguments. Configure the
constants below, then run. Each policy is evaluated with both high-pass torque
normalization modes, which are saved separately for plotting:

  python learning/evaluate_all_models.py
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from learning import evaluate_policy
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODELS_DIRECTORY = Path("eagle")
OUTPUT_DIRECTORY = Path("evaluations")
#ENV_NAME = "Go1JoystickFlatTerrain"
ENV_NAME = "Go1JoystickRoughTerrain"

# Evaluate and log both definitions of the high-pass torque penalty. Results
# are written below <ENV_NAME>/capacity_normalized/ and
# <ENV_NAME>/raw_torque/, respectively.
TORQUE_NORMALIZATION_MODES = {
    "capacity_normalized": True,
    "raw_torque": False,
}

# None selects the numerically latest checkpoint in every model directory.
# Set a numeric directory name, for example "000183500800", to compare every
# model at the same training step.
CHECKPOINT_NAME: str | None = None

NUM_RANDOM_TASKS = 512
TASK_SEED = 0
EPISODE_LENGTH = 1000
POLICY_SEED = 0
FFT_CUTOFFS_HZ = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
SAVGOL_WINDOW_LENGTH = 11
SAVGOL_POLYORDER = 3
DETERMINISTIC = True
DISABLE_PERTURBATIONS = True
RENDER_VIDEO = False
SAVE_SIGNALS = False
REQUIRE_CUDA = True
# Use the registry configuration for ENV_NAME for every policy instead of each
# checkpoint's training reward config. This keeps the comparison environment
# identical across runs.
USE_SAVED_ENVIRONMENT_CONFIG = False
USE_WANDB = False
WANDB_PROJECT = "spectral_playground_policy_evaluation"

# Reuse a successful report when its model, evaluation settings, evaluator,
# environment code, and locked dependencies are unchanged. Set to False to
# force every model to be evaluated again.
REUSE_UNCHANGED_RESULTS = True

# These files/directories affect evaluation behavior and are included in the
# cache fingerprint. Add another local dependency here if evaluation starts
# depending on it.
CACHE_DEPENDENCY_PATHS = (
    Path("learning/evaluate_policy.py"),
    Path("learning/train_jax_ppo.py"),
    Path("mujoco_playground/_src/locomotion/go1"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)
CACHE_MANIFEST_NAME = "evaluation_cache.json"
CACHE_FORMAT_VERSION = 1

# If False, the first failed evaluation stops the batch immediately.
CONTINUE_ON_ERROR = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
  return path if path.is_absolute() else PROJECT_ROOT / path


def _model_directories(models_directory: Path) -> list[Path]:
  if not models_directory.is_dir():
    raise FileNotFoundError(f"Models directory does not exist: {models_directory}")
  models = sorted(
      path
      for path in models_directory.iterdir()
      if path.is_dir() and (path / "checkpoints").is_dir()
  )
  if not models:
    raise ValueError(
        f"No model directories containing checkpoints/ found in "
        f"{models_directory}"
    )
  return models


def _select_checkpoint(model_directory: Path) -> Path:
  checkpoints_directory = model_directory / "checkpoints"
  if CHECKPOINT_NAME is not None:
    checkpoint = checkpoints_directory / CHECKPOINT_NAME
    if not checkpoint.is_dir():
      raise FileNotFoundError(
          f"Checkpoint {CHECKPOINT_NAME!r} not found for "
          f"{model_directory.name}: {checkpoint}"
      )
    return checkpoint

  numeric_checkpoints = [
      path
      for path in checkpoints_directory.iterdir()
      if path.is_dir() and path.name.isdigit()
  ]
  if not numeric_checkpoints:
    raise ValueError(f"No numeric checkpoints found in {checkpoints_directory}")
  return max(numeric_checkpoints, key=lambda path: int(path.name))


def _boolean_argument(name: str, enabled: bool) -> str:
  return f"--{name}" if enabled else f"--no-{name}"


def _hash_paths(paths: list[Path]) -> str:
  """Hashes file contents and relative names deterministically."""
  digest = hashlib.sha256()
  for root in sorted(paths, key=lambda path: str(path)):
    if not root.exists():
      raise FileNotFoundError(f"Cache dependency does not exist: {root}")
    files = [root] if root.is_file() else sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in (".pyc", ".pyo")
    )
    for path in files:
      relative = path.name if root.is_file() else str(path.relative_to(root))
      digest.update(str(root).encode())
      digest.update(b"\0")
      digest.update(relative.encode())
      digest.update(b"\0")
      with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
          digest.update(chunk)
      digest.update(b"\0")
  return digest.hexdigest()


def _evaluation_settings(torque_normalization: bool) -> dict[str, Any]:
  return {
      "env_name": ENV_NAME,
      "num_random_tasks": NUM_RANDOM_TASKS,
      "task_seed": TASK_SEED,
      "episode_length": EPISODE_LENGTH,
      "policy_seed": POLICY_SEED,
      "fft_cutoffs_hz": list(FFT_CUTOFFS_HZ),
      "savgol_window_length": SAVGOL_WINDOW_LENGTH,
      "savgol_polyorder": SAVGOL_POLYORDER,
      "deterministic": DETERMINISTIC,
      "disable_perturbations": DISABLE_PERTURBATIONS,
      "render_video": RENDER_VIDEO,
      "save_signals": SAVE_SIGNALS,
      "require_cuda": REQUIRE_CUDA,
      "use_saved_environment_config": USE_SAVED_ENVIRONMENT_CONFIG,
      "torque_highpass_normalize_by_capacity": torque_normalization,
      "use_wandb": USE_WANDB,
      "wandb_project": WANDB_PROJECT if USE_WANDB else None,
  }


def _cache_signature(
    checkpoint: Path,
    torque_normalization: bool,
) -> dict[str, Any]:
  checkpoint_inputs = [checkpoint]
  legacy_config = checkpoint.parent / "config.json"
  if legacy_config.is_file():
    checkpoint_inputs.append(legacy_config)
  dependencies = [_resolve(path) for path in CACHE_DEPENDENCY_PATHS]
  return {
      "cache_format_version": CACHE_FORMAT_VERSION,
      "checkpoint": checkpoint.name,
      "checkpoint_sha256": _hash_paths(checkpoint_inputs),
      "evaluation_settings": _evaluation_settings(torque_normalization),
      "evaluation_code_sha256": _hash_paths(dependencies),
  }


def _read_cache_manifest(path: Path) -> dict[str, Any] | None:
  if not path.is_file():
    return None
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None
  return value if isinstance(value, dict) else None


def _write_cache_manifest(path: Path, signature: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = path.with_suffix(path.suffix + ".tmp")
  temporary_path.write_text(
      json.dumps(signature, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  temporary_path.replace(path)


def _evaluation_arguments(
    checkpoint: Path,
    output_directory: Path,
    torque_normalization: bool,
) -> list[str]:
  arguments = [
      "--checkpoint",
      str(checkpoint),
      "--env_name",
      ENV_NAME,
      "--output_dir",
      str(output_directory),
      "--num_random_tasks",
      str(NUM_RANDOM_TASKS),
      "--task_seed",
      str(TASK_SEED),
      "--episode_length",
      str(EPISODE_LENGTH),
      "--seed",
      str(POLICY_SEED),
      "--fft_cutoffs_hz",
      ",".join(map(str, FFT_CUTOFFS_HZ)),
      "--savgol_window_length",
      str(SAVGOL_WINDOW_LENGTH),
      "--savgol_polyorder",
      str(SAVGOL_POLYORDER),
      _boolean_argument("deterministic", DETERMINISTIC),
      _boolean_argument("disable_perturbations", DISABLE_PERTURBATIONS),
      _boolean_argument("render_video", RENDER_VIDEO),
      _boolean_argument("save_signals", SAVE_SIGNALS),
      _boolean_argument("require_cuda", REQUIRE_CUDA),
      _boolean_argument(
          "use_saved_environment_config", USE_SAVED_ENVIRONMENT_CONFIG
      ),
      _boolean_argument(
          "torque_highpass_normalize_by_capacity", torque_normalization
      ),
  ]
  if USE_WANDB:
    arguments.extend(("--use_wandb", "--wandb_project", WANDB_PROJECT))
  return arguments


def main() -> None:
  models_directory = _resolve(MODELS_DIRECTORY / ENV_NAME)
  output_root = _resolve(OUTPUT_DIRECTORY / ENV_NAME)
  models = _model_directories(models_directory)
  failures: list[tuple[str, str]] = []
  pending = []
  skipped = 0
  unavailable = 0
  rollout_cache: dict[str, Any] = {}

  print(
      f"Found {len(models)} models in {models_directory}; shared environment "
      f"is {ENV_NAME}",
      flush=True,
  )
  for index, model_directory in enumerate(models, start=1):
    prefix = f"[{index}/{len(models)}] {model_directory.name}"
    try:
      checkpoint = _select_checkpoint(model_directory)
    except (FileNotFoundError, ValueError) as error:
      print(f"{prefix}: skipped ({error})", flush=True)
      unavailable += 1
      continue
    for normalization_name, torque_normalization in (
        TORQUE_NORMALIZATION_MODES.items()
    ):
      variant_prefix = f"{prefix} [{normalization_name}]"
      output_directory = (
          output_root
          / normalization_name
          / model_directory.name
          / checkpoint.name
      )
      summary_path = output_directory / "summary.json"
      manifest_path = output_directory / CACHE_MANIFEST_NAME
      signature = _cache_signature(
          checkpoint, torque_normalization
      )

      if (
          REUSE_UNCHANGED_RESULTS
          and summary_path.is_file()
          and _read_cache_manifest(manifest_path) == signature
      ):
        print(
            f"{variant_prefix}: skipped (result is unchanged)", flush=True
        )
        skipped += 1
        continue

      reason = "no result" if not summary_path.is_file() else "result is stale"
      pending.append((
          variant_prefix,
          model_directory,
          checkpoint,
          output_directory,
          manifest_path,
          signature,
          reason,
          torque_normalization,
      ))

  with tqdm(
      pending,
      desc="Evaluating models",
      unit="model",
      dynamic_ncols=True,
      smoothing=0.2,
  ) as progress:
    for (
        prefix,
        model_directory,
        checkpoint,
        output_directory,
        manifest_path,
        signature,
        reason,
        torque_normalization,
    ) in progress:
      progress.set_postfix_str(model_directory.name, refresh=True)
      tqdm.write(
          f"{prefix}: evaluating checkpoint {checkpoint.name} ({reason})"
      )
      arguments = _evaluation_arguments(
          checkpoint,
          output_directory,
          torque_normalization,
      )
      try:
        evaluate_policy.main(arguments, rollout_cache=rollout_cache)
      except Exception as error:  # pylint: disable=broad-exception-caught
        failures.append((model_directory.name, str(error)))
        tqdm.write(f"{prefix}: failed: {error}")
        if not CONTINUE_ON_ERROR:
          raise
      else:
        _write_cache_manifest(manifest_path, signature)
        tqdm.write(f"{prefix}: complete -> {output_directory}")

  if failures:
    details = ", ".join(f"{name} ({error})" for name, error in failures)
    raise RuntimeError(f"{len(failures)} evaluations failed: {details}")
  print(
      f"Evaluation batch complete: {len(pending)} evaluated, {skipped} "
      f"unchanged models skipped, {unavailable} models had no usable "
      f"checkpoint, {len(rollout_cache)} compiled compatibility group(s).",
      flush=True,
  )


if __name__ == "__main__":
  main()
