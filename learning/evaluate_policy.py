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
"""Evaluate a PPO policy on reproducible command scenarios or random tasks.

The evaluator is intentionally independent of the regularizer used for
training.  It records raw trajectories, environment/W&B-compatible reward
metrics, physical command-tracking errors, and order-independent action and
torque smoothness measurements.
"""

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import re
import types
from typing import Any, Mapping, Sequence

from brax.training import checkpoint as brax_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground import registry
import numpy as np
from scipy import signal as scipy_signal

from learning import train_jax_ppo as train_utils


DEFAULT_COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_0p5": (0.5, 0.0, 0.0),
    "forward_1p0": (1.0, 0.0, 0.0),
    "backward_0p5": (-0.5, 0.0, 0.0),
    "lateral_0p5": (0.0, 0.5, 0.0),
    "yaw_0p5": (0.0, 0.0, 0.5),
    "combined": (0.8, 0.3, 0.5),
}
DEFAULT_FFT_CUTOFFS_HZ = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_SAVGOL_WINDOW_LENGTH = 11
DEFAULT_SAVGOL_POLYORDER = 3


def _parse_commands(value: str | None) -> dict[str, tuple[float, float, float]]:
  """Parses a JSON object mapping scenario names to [vx, vy, yaw_rate]."""
  if value is None:
    return dict(DEFAULT_COMMANDS)
  raw = json.loads(value)
  if not isinstance(raw, dict) or not raw:
    raise ValueError("--commands must be a non-empty JSON object.")
  commands = {}
  for name, command in raw.items():
    if not isinstance(name, str) or not name:
      raise ValueError("Every command scenario must have a non-empty name.")
    if not isinstance(command, (list, tuple)) or len(command) != 3:
      raise ValueError(f"Command {name!r} must contain exactly three values.")
    command = tuple(float(x) for x in command)
    if not all(math.isfinite(x) for x in command):
      raise ValueError(f"Command {name!r} must contain only finite values.")
    commands[name] = command
  return commands


def _parse_cutoffs(value: str) -> tuple[float, ...]:
  cutoffs = tuple(float(x) for x in value.split(",") if x.strip())
  if not cutoffs or any(not math.isfinite(x) or x <= 0 for x in cutoffs):
    raise ValueError("--fft_cutoffs_hz must contain positive finite values.")
  return tuple(sorted(set(cutoffs)))


def _safe_name(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "scenario"


def _jsonable(value: Any) -> Any:
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  if isinstance(value, (np.generic,)):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, Path):
    return str(value)
  if callable(value):
    return f"{value.__module__}.{value.__qualname__}"
  return value


def _masked_episode_values(
    values: np.ndarray, active: np.ndarray
) -> list[np.ndarray]:
  """Returns one valid prefix per rollout from time-major arrays."""
  values = np.asarray(values)
  active = np.asarray(active, dtype=bool)
  return [values[active[:, i], i] for i in range(values.shape[1])]


def _fft_energy_metrics(
    signal: np.ndarray,
    sample_period: float,
    cutoffs_hz: Sequence[float],
) -> dict[str, float]:
  """Computes Parseval-normalized spectral energy for one episode."""
  signal = np.asarray(signal, dtype=np.float64)
  if signal.ndim == 1:
    signal = signal[:, None]
  if signal.shape[0] == 0:
    return {"total_energy_per_step": 0.0}

  spectrum = np.fft.rfft(signal, axis=0)
  bin_energy = np.square(np.abs(spectrum)) / signal.shape[0] ** 2
  if signal.shape[0] > 1:
    upper = -1 if signal.shape[0] % 2 == 0 else None
    bin_energy[1:upper] *= 2.0
  frequencies = np.fft.rfftfreq(signal.shape[0], d=sample_period)
  energy_by_frequency = np.sum(bin_energy, axis=-1)
  total_energy = float(np.sum(energy_by_frequency))
  ac_energy = energy_by_frequency.copy()
  ac_energy[0] = 0.0
  ac_total = float(np.sum(ac_energy))

  metrics = {
      "total_energy_per_step": total_energy,
      "ac_energy_per_step": ac_total,
      "spectral_centroid_hz": (
          float(np.sum(frequencies * ac_energy) / ac_total)
          if ac_total > 0
          else 0.0
      ),
  }
  cumulative = np.cumsum(ac_energy)
  if ac_total > 0:
    rolloff_index = min(
        int(np.searchsorted(cumulative, 0.95 * ac_total)),
        len(frequencies) - 1,
    )
    metrics["spectral_rolloff_95_hz"] = float(frequencies[rolloff_index])
  else:
    metrics["spectral_rolloff_95_hz"] = 0.0
  for cutoff in cutoffs_hz:
    metrics[f"fft_above_{cutoff:g}hz_energy_per_step"] = float(
        np.sum(energy_by_frequency[frequencies >= cutoff])
    )
  return metrics


def _smoothness_metrics(
    signal: np.ndarray,
    sample_period: float,
    cutoffs_hz: Sequence[float],
    savgol_window_length: int = DEFAULT_SAVGOL_WINDOW_LENGTH,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> dict[str, float]:
  """Computes scale-explicit time- and frequency-domain smoothness metrics."""
  signal = np.asarray(signal, dtype=np.float64)
  if signal.ndim == 1:
    signal = signal[:, None]
  dofs = signal.shape[-1]
  metrics = {
      "rms_per_dof": float(np.sqrt(np.mean(np.square(signal)))),
      "peak_abs": float(np.max(np.abs(signal))),
  }
  delta = np.diff(signal, axis=0)
  if len(delta):
    metrics.update({
        # This is the unweighted quantity used by the PPO mean-action loss.
        "mean_squared_delta_l2_per_step": float(
            np.mean(np.sum(np.square(delta), axis=-1))
        ),
        "delta_rms_per_dof": float(np.sqrt(np.mean(np.square(delta)))),
        "rate_rms_per_dof_per_second": float(
            np.sqrt(np.mean(np.square(delta / sample_period)))
        ),
        "total_variation_per_second_per_dof": float(
            np.sum(np.abs(delta))
            / (max(signal.shape[0] - 1, 1) * sample_period * dofs)
        ),
    })
  else:
    metrics.update({
        "mean_squared_delta_l2_per_step": 0.0,
        "delta_rms_per_dof": 0.0,
        "rate_rms_per_dof_per_second": 0.0,
        "total_variation_per_second_per_dof": 0.0,
    })
  second_delta = np.diff(signal, n=2, axis=0)
  # MSSD follows the control-smoothness convention: the mean square of the
  # discrete second difference, averaged over time and degrees of freedom.
  metrics["mssd_mean_squared_second_difference_per_dof"] = (
      float(np.mean(np.square(second_delta))) if len(second_delta) else 0.0
  )
  metrics["second_difference_rms_per_dof"] = (
      float(np.sqrt(np.mean(np.square(second_delta))))
      if len(second_delta)
      else 0.0
  )
  metrics["acceleration_rms_per_dof_per_second2"] = (
      float(
          np.sqrt(np.mean(np.square(second_delta / sample_period**2)))
      )
      if len(second_delta)
      else 0.0
  )
  # Use the largest valid odd window for very short, terminated episodes.
  effective_window = min(savgol_window_length, signal.shape[0])
  if effective_window % 2 == 0:
    effective_window -= 1
  if effective_window >= 1:
    effective_polyorder = min(savgol_polyorder, effective_window - 1)
    filtered = scipy_signal.savgol_filter(
        signal,
        window_length=effective_window,
        polyorder=effective_polyorder,
        axis=0,
        mode="interp",
    )
    metrics["msgfd_mean_absolute_savgol_filter_deviation_per_dof"] = float(
        np.mean(np.abs(signal - filtered))
    )
  else:
    metrics["msgfd_mean_absolute_savgol_filter_deviation_per_dof"] = 0.0
  third_delta = np.diff(signal, n=3, axis=0)
  metrics["third_difference_rms_per_dof"] = (
      float(np.sqrt(np.mean(np.square(third_delta))))
      if len(third_delta)
      else 0.0
  )
  metrics["jerk_rms_per_dof_per_second3"] = (
      float(np.sqrt(np.mean(np.square(third_delta / sample_period**3))))
      if len(third_delta)
      else 0.0
  )
  metrics.update(_fft_energy_metrics(signal, sample_period, cutoffs_hz))
  return metrics


def _tracking_metrics(
    command: np.ndarray,
    local_linear_velocity: np.ndarray,
    gyro: np.ndarray,
    upvector: np.ndarray,
    qvel: np.ndarray,
    torque: np.ndarray,
    sample_period: float,
) -> dict[str, float]:
  command = np.asarray(command)
  if command.ndim == 1:
    command = np.broadcast_to(command, (local_linear_velocity.shape[0], 3))
  linear_error = local_linear_velocity[:, :2] - command[:, :2]
  yaw_error = gyro[:, 2] - command[:, 2]
  mechanical_power = qvel[:, 6:] * torque
  orientation_error = np.arccos(np.clip(upvector[:, 2], -1.0, 1.0))
  return {
      "linear_velocity_rmse": float(np.sqrt(np.mean(np.square(linear_error)))),
      "linear_velocity_vector_rmse": float(
          np.sqrt(np.mean(np.sum(np.square(linear_error), axis=-1)))
      ),
      "linear_velocity_mae": float(np.mean(np.abs(linear_error))),
      "yaw_rate_rmse": float(np.sqrt(np.mean(np.square(yaw_error)))),
      "yaw_rate_mae": float(np.mean(np.abs(yaw_error))),
      "vertical_velocity_rms": float(
          np.sqrt(np.mean(np.square(local_linear_velocity[:, 2])))
      ),
      "roll_pitch_rate_rms": float(
          np.sqrt(np.mean(np.square(gyro[:, :2])))
      ),
      "orientation_error_rms_degrees": float(
          np.rad2deg(np.sqrt(np.mean(np.square(orientation_error))))
      ),
      "upright_z_mean": float(np.mean(upvector[:, 2])),
      "upright_z_min": float(np.min(upvector[:, 2])),
      "absolute_mechanical_power_mean": float(
          np.mean(np.sum(np.abs(mechanical_power), axis=-1))
      ),
      "absolute_mechanical_energy": float(
          np.sum(np.sum(np.abs(mechanical_power), axis=-1)) * sample_period
      ),
      "total_absolute_torque_impulse": float(
          np.sum(np.abs(torque)) * sample_period
      ),
  }


def _feet_height_metrics(
    feet_position: np.ndarray,
    feet_contact: np.ndarray,
    command: np.ndarray,
    target_height: float,
) -> dict[str, float]:
  """Measures completed-swing peak-height error at touchdown."""
  feet_position = np.asarray(feet_position, dtype=np.float64)
  feet_contact = np.asarray(feet_contact, dtype=bool)
  command = np.asarray(command, dtype=np.float64)
  if command.ndim == 1:
    command = np.broadcast_to(command, (feet_position.shape[0], 3))

  errors = []
  for foot_index in range(feet_position.shape[1]):
    in_swing = False
    swing_peak = -math.inf
    for step in range(feet_position.shape[0]):
      if not feet_contact[step, foot_index]:
        swing_peak = max(swing_peak, feet_position[step, foot_index, 2])
        in_swing = True
      elif in_swing:
        if np.linalg.norm(command[step]) > 0.01:
          errors.append(swing_peak - target_height)
        in_swing = False
        swing_peak = -math.inf
  errors_mm = 1000.0 * np.asarray(errors, dtype=np.float64)
  return {
      "feet_height_error_mean_mm": (
          float(np.mean(np.abs(errors_mm))) if len(errors_mm) else 0.0
      ),
      "feet_height_error_rmse_mm": (
          float(np.sqrt(np.mean(np.square(errors_mm))))
          if len(errors_mm)
          else 0.0
      ),
      "feet_height_touchdowns": float(len(errors_mm)),
  }


def _aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
  """Aggregates numeric rollout rows without weighting long episodes more."""
  if not rows:
    return {}
  keys = sorted(set.intersection(*(set(row) for row in rows)))
  result = {}
  for key in keys:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
      continue
    result[f"{key}/mean"] = float(np.mean(values))
    result[f"{key}/std"] = float(np.std(values))
    result[f"{key}/median"] = float(np.median(values))
    result[f"{key}/min"] = float(np.min(values))
    result[f"{key}/max"] = float(np.max(values))
  return result


def _wandb_compatible_summary(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
  """Produces the scalar evaluation keys used by the training W&B logger."""
  if not rows:
    return {}
  result = {}
  keys = sorted(set.intersection(*(set(row) for row in rows)))
  for key in keys:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
      continue
    if key.startswith("eval_reward_means/"):
      suffix = key.removeprefix("eval_reward_means/")
      result[key] = float(np.mean(values))
      result[f"eval_reward_stds/{suffix}"] = float(np.std(values))
    elif key.startswith(("torque_spectrum/eval/", "rollouts/eval_")):
      result[key] = float(np.mean(values))
  lengths = np.asarray(
      [row["episode/length_steps"] for row in rows], dtype=np.float64
  )
  result["rollouts/eval_avg_episode_length"] = float(np.mean(lengths))
  result["rollouts/eval_std_episode_length"] = float(np.std(lengths))
  return result


def _episode_rows(
    signals: Mapping[str, np.ndarray],
    command: np.ndarray | None,
    sample_period: float,
    cutoffs_hz: Sequence[float],
    savgol_window_length: int = DEFAULT_SAVGOL_WINDOW_LENGTH,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    feet_height_target: float | None = None,
) -> list[dict[str, float]]:
  """Builds one comparable metric row per rollout."""
  active = np.asarray(signals["active"], dtype=bool)
  episodes = {
      name: _masked_episode_values(value, active)
      for name, value in signals.items()
      if name != "active"
  }
  rows = []
  for rollout_index in range(active.shape[1]):
    length = int(np.sum(active[:, rollout_index]))
    fell = bool(np.any(episodes["done"][rollout_index]))
    row = {
        "episode/length_steps": float(length),
        "episode/duration_seconds": float(length * sample_period),
        "episode/completed_horizon": float(not fell),
        "episode/fell": float(fell),
        "eval_reward_means/total": float(
            np.sum(episodes["reward"][rollout_index])
        ),
    }
    for name in sorted(k for k in episodes if k.startswith("metric/")):
      metric_name = name.removeprefix("metric/")
      value = float(np.sum(episodes[name][rollout_index]))
      if metric_name.startswith("reward/"):
        row[f"eval_reward_means/{metric_name.removeprefix('reward/')}"] = value
      elif metric_name == "reward_without_action_rate":
        row["eval_reward_means/total_without_action_rate"] = value
      elif metric_name == "reward_without_regularization":
        row["eval_reward_means/total_without_regularization"] = value
      elif metric_name.startswith("torque_spectrum/"):
        row[f"torque_spectrum/eval/online_{metric_name.removeprefix('torque_spectrum/')}"] = value
      else:
        row[f"rollouts/eval_{metric_name}"] = value

    row.update({
        f"smoothness/action/{key}": value
        for key, value in _smoothness_metrics(
            episodes["action"][rollout_index],
            sample_period,
            cutoffs_hz,
            savgol_window_length,
            savgol_polyorder,
        ).items()
    })
    row.update({
        f"smoothness/motor_target/{key}": value
        for key, value in _smoothness_metrics(
            episodes["motor_target"][rollout_index],
            sample_period,
            cutoffs_hz,
            savgol_window_length,
            savgol_polyorder,
        ).items()
    })
    torque_metrics = _smoothness_metrics(
        episodes["actuator_force"][rollout_index],
        sample_period,
        cutoffs_hz,
        savgol_window_length,
        savgol_polyorder,
    )
    row.update({
        f"smoothness/torque/{key}": value
        for key, value in torque_metrics.items()
    })
    for key, value in torque_metrics.items():
      if key.startswith("fft_") or key == "total_energy_per_step":
        row[f"torque_spectrum/eval/{key}"] = value
    row.update({
        f"tracking/{key}": value
        for key, value in _tracking_metrics(
            (
                command
                if command is not None
                else episodes["command"][rollout_index]
            ),
            episodes["local_linear_velocity"][rollout_index],
            episodes["gyro"][rollout_index],
            episodes["upvector"][rollout_index],
            episodes["qvel"][rollout_index],
            episodes["actuator_force"][rollout_index],
            sample_period,
        ).items()
    })
    if feet_height_target is not None and "feet_position" in episodes:
      row.update({
          f"tracking/{key}": value
          for key, value in _feet_height_metrics(
              episodes["feet_position"][rollout_index],
              episodes["feet_contact"][rollout_index],
              episodes["command"][rollout_index],
              feet_height_target,
          ).items()
      })
    rows.append(row)
  return rows


def _set_constant_command(env, state, command: jax.Array):
  """Sets the initial command and recomputes observations consistently."""
  info = dict(state.info)
  info["command"] = jp.broadcast_to(command, info["command"].shape)
  info["steps_until_next_cmd"] = jp.full_like(
      info["steps_until_next_cmd"], jp.iinfo(jp.int32).max
  )

  def get_obs(data, single_info):
    return env._get_obs(data, single_info)  # pylint: disable=protected-access

  obs = jax.vmap(get_obs)(state.data, info)
  return state.replace(obs=obs, info=info)


def _where_batch(mask, new, old):
  if not hasattr(new, "shape") or not new.shape:
    return new
  if new.shape[0] != mask.shape[0]:
    return new
  expanded = mask.reshape((mask.shape[0],) + (1,) * (new.ndim - 1))
  return jp.where(expanded, new, old)


def _rollout(
    env,
    policy,
    command: np.ndarray | None,
    reset_keys: jax.Array,
    episode_length: int,
    action_repeat: int,
    policy_seed: int,
    record_full_signals: bool = False,
    record_render_signals: bool = False,
):
  """Runs a batched rollout and returns time-major signals.

  When command is None, each environment keeps the reproducible random command
  schedule sampled by its reset key.  Otherwise, the command is held constant.
  """
  num_rollouts = reset_keys.shape[0]
  reset = jax.vmap(env.reset)
  step_env = jax.vmap(env.step)
  infer = jax.vmap(policy)
  state = reset(reset_keys)
  if command is not None:
    state = _set_constant_command(env, state, jp.asarray(command))
  policy_steps = episode_length // action_repeat
  active = jp.ones((num_rollouts,), dtype=bool)
  policy_keys = jax.random.split(
      jax.random.PRNGKey(policy_seed), num_rollouts
  )

  def scan_step(carry, _):
    current_state, current_active, keys = carry
    split_keys = jax.vmap(jax.random.split)(keys)
    keys, action_keys = split_keys[:, 0], split_keys[:, 1]
    action, _ = infer(current_state.obs, action_keys)

    def repeat_step(repeat_state, _):
      next_state = step_env(repeat_state, action)
      return next_state, next_state.reward

    next_state, repeated_rewards = jax.lax.scan(
        repeat_step, current_state, (), length=action_repeat
    )
    reward = jp.sum(repeated_rewards, axis=0)
    done = next_state.done.astype(bool)
    local_linear_velocity = jax.vmap(env.get_local_linvel)(next_state.data)
    gyro = jax.vmap(env.get_gyro)(next_state.data)
    upvector = jax.vmap(env.get_upvector)(next_state.data)
    feet_position = jax.vmap(env.get_feet_pos)(next_state.data)
    signals = {
        "active": current_active,
        "done": done,
        "reward": reward,
        "action": action,
        # This is the command that produced the action and transition.  The
        # environment may already have sampled the next command in next_state.
        "command": current_state.info["command"],
        "motor_target": next_state.data.ctrl,
        "actuator_force": next_state.data.actuator_force,
        "qvel": next_state.data.qvel,
        "local_linear_velocity": local_linear_velocity,
        "gyro": gyro,
        "upvector": upvector,
        "feet_position": feet_position,
        "feet_contact": next_state.info["last_contact"],
        "feet_air_time": next_state.info["feet_air_time"],
        **{
            f"metric/{name}": value
            for name, value in next_state.metrics.items()
        },
    }
    if record_full_signals:
      signals.update({
          "accelerometer": jax.vmap(env.get_accelerometer)(next_state.data),
          "qpos": next_state.data.qpos,
          "qacc": next_state.data.qacc,
          "ctrl": next_state.data.ctrl,
          "sensordata": next_state.data.sensordata,
          "obs/state": next_state.obs["state"],
          "obs/privileged_state": next_state.obs["privileged_state"],
      })
    if record_render_signals:
      signals.update({
          "render/qpos": next_state.data.qpos,
          "render/qvel": next_state.data.qvel,
          "render/mocap_pos": next_state.data.mocap_pos,
          "render/mocap_quat": next_state.data.mocap_quat,
          "render/xfrc_applied": next_state.data.xfrc_applied,
      })
    next_active = current_active & ~done
    frozen_state = jax.tree.map(
        lambda new, old: _where_batch(current_active, new, old),
        next_state,
        current_state,
    )
    return (frozen_state, next_active, keys), signals

  (_, _, _), signals = jax.lax.scan(
      scan_step, (state, active, policy_keys), (), length=policy_steps
  )
  return signals


def _save_video(
    env,
    signals: Mapping[str, np.ndarray],
    output_path: Path,
    sample_period: float,
    camera: str,
    width: int,
    height: int,
):
  import mediapy as media  # pylint: disable=g-import-not-at-top
  import mujoco  # pylint: disable=g-import-not-at-top

  active = np.asarray(signals["active"][:, 0], dtype=bool)
  frame_stride = max(1, round((1.0 / 30.0) / sample_period))
  frame_indices = np.flatnonzero(active)[::frame_stride]
  trajectory = []
  for index in frame_indices:
    data = types.SimpleNamespace(
        qpos=signals["render/qpos"][index, 0],
        qvel=signals["render/qvel"][index, 0],
        mocap_pos=signals["render/mocap_pos"][index, 0],
        mocap_quat=signals["render/mocap_quat"][index, 0],
        xfrc_applied=signals["render/xfrc_applied"][index, 0],
    )
    trajectory.append(types.SimpleNamespace(data=data))
  if not trajectory:
    return
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  frames = env.render(
      trajectory,
      camera=camera,
      width=width,
      height=height,
      scene_option=scene_option,
  )
  media.write_video(output_path, frames, fps=1.0 / sample_period / frame_stride)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]):
  keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
  with path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def _load_policy_and_environment(args):
  checkpoint = train_utils._resolve_checkpoint_path(args.checkpoint)
  if checkpoint is None:
    raise ValueError("A checkpoint path is required.")
  saved = train_utils._load_run_config(checkpoint)
  env_name = args.env_name or (saved or {}).get("env_name")
  if env_name is None:
    raise ValueError(
        "This legacy checkpoint does not store env_name; pass --env_name."
    )
  default_env_config = registry.get_default_config(env_name)
  if (
      args.use_saved_environment_config
      and saved
      and "environment_config" in saved
  ):
    env_config = config_dict.ConfigDict(train_utils._merge_saved_config(
        default_env_config.to_dict(), saved["environment_config"]
    ))
  else:
    env_config = default_env_config
  if args.disable_perturbations and "pert_config" in env_config:
    env_config.pert_config.enable = False
  env = registry.load(env_name, config=env_config)

  network_config = train_utils._load_checkpoint_network_config(
      checkpoint / "ppo_network_config.json"
  )
  network = brax_checkpoint.get_network(
      network_config, ppo_networks.make_ppo_networks
  )
  params = brax_checkpoint.load(checkpoint)
  ppo_config = (saved or {}).get("ppo_config", {})
  action_repeat = int(ppo_config.get("action_repeat", env_config.action_repeat))
  episode_length = args.episode_length or int(
      ppo_config.get("episode_length", env_config.episode_length)
  )
  return (
      checkpoint,
      saved,
      env_name,
      env_config,
      env,
      network_config,
      network,
      params,
      action_repeat,
      episode_length,
  )


def _default_output_dir(checkpoint: epath.Path) -> Path:
  run_name = checkpoint.parent.parent.name
  return Path("evaluations") / run_name / checkpoint.name


def _rollout_cache_key(
    env_name: str,
    env_config: config_dict.ConfigDict,
    network_config: config_dict.ConfigDict,
    args,
    action_repeat: int,
    episode_length: int,
    num_parallel_rollouts: int,
    random_task_mode: bool,
) -> str:
  """Returns the static structure that determines JAX compilation reuse."""
  value = {
      "env_name": env_name,
      "environment_config": env_config.to_dict(),
      "network_config": network_config.to_dict(),
      "action_repeat": action_repeat,
      "episode_length": episode_length,
      "num_parallel_rollouts": num_parallel_rollouts,
      "random_task_mode": random_task_mode,
      "task_seed": args.task_seed if random_task_mode else args.seed,
      "policy_seed": args.seed + 1,
      "deterministic": args.deterministic,
      "save_signals": args.save_signals,
      "render_video": args.render_video,
  }
  return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def evaluate(args, rollout_cache: dict[str, Any] | None = None) -> Path:
  commands = _parse_commands(args.commands)
  cutoffs_hz = _parse_cutoffs(args.fft_cutoffs_hz)
  (
      checkpoint,
      saved,
      env_name,
      env_config,
      env,
      network_config,
      network,
      params,
      action_repeat,
      episode_length,
  ) = _load_policy_and_environment(args)
  if episode_length % action_repeat:
    raise ValueError("episode_length must be divisible by action_repeat.")
  sample_period = float(env.dt * action_repeat)
  feet_height_target = float(env_config.reward_config.max_foot_height)
  nyquist_hz = 0.5 / sample_period
  if any(cutoff > nyquist_hz for cutoff in cutoffs_hz):
    raise ValueError(
        f"FFT cutoffs cannot exceed the {nyquist_hz:g} Hz Nyquist frequency."
    )

  output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(checkpoint)
  output_dir.mkdir(parents=True, exist_ok=True)
  random_task_mode = args.num_random_tasks > 0
  num_parallel_rollouts = (
      args.num_random_tasks if random_task_mode else args.num_rollouts
  )
  reset_seed = args.task_seed if random_task_mode else args.seed
  reset_keys = jax.random.split(
      jax.random.PRNGKey(reset_seed), num_parallel_rollouts
  )
  cache_key = _rollout_cache_key(
      env_name,
      env_config,
      network_config,
      args,
      action_repeat,
      episode_length,
      num_parallel_rollouts,
      random_task_mode,
  )
  cache = rollout_cache if rollout_cache is not None else {}
  rollout_fn = cache.get(cache_key)
  if rollout_fn is None:
    inference_factory = ppo_networks.make_inference_fn(network)

    def rollout_with_params(policy_params, command):
      policy = inference_factory(
          policy_params, deterministic=args.deterministic
      )
      return _rollout(
          env,
          policy,
          command,
          reset_keys,
          episode_length,
          action_repeat,
          args.seed + 1,
          record_full_signals=args.save_signals,
          record_render_signals=args.render_video,
      )

    rollout_fn = jax.jit(rollout_with_params)
    cache[cache_key] = rollout_fn
    print("Created a new compiled-rollout compatibility group.")
  else:
    print("Reusing the compiled rollout; only policy weights changed.")

  if random_task_mode:
    scenarios = (("random_tasks", None),)
  else:
    scenarios = tuple(commands.items())

  all_rows = []
  all_metric_rows = []
  scenario_summaries = {}
  for scenario_name, command_tuple in scenarios:
    print(f"Running scenario {scenario_name!r}: {command_tuple}")
    command = (
        None
        if command_tuple is None
        else np.asarray(command_tuple, dtype=np.float32)
    )
    signals = jax.device_get(rollout_fn(params, command))
    signals = {name: np.asarray(value) for name, value in signals.items()}
    rows = _episode_rows(
        signals,
        command,
        sample_period,
        cutoffs_hz,
        args.savgol_window_length,
        args.savgol_polyorder,
        feet_height_target,
    )
    scenario_dir = output_dir / _safe_name(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    if args.save_signals:
      np.savez_compressed(scenario_dir / "signals.npz", **signals)
    indexed_rows = [
        {
            "scenario": scenario_name,
            "rollout": i,
            **({"task": i} if random_task_mode else {}),
            **row,
        }
        for i, row in enumerate(rows)
    ]
    _write_rows(scenario_dir / "rollouts.csv", indexed_rows)
    summary = {
        **_aggregate(rows),
        **_wandb_compatible_summary(rows),
    }
    scenario_summaries[scenario_name] = summary
    with (scenario_dir / "summary.json").open("w", encoding="utf-8") as fp:
      json.dump(_jsonable(summary), fp, indent=2, sort_keys=True)
    if args.render_video:
      _save_video(
          env,
          signals,
          scenario_dir / "rollout.mp4",
          sample_period,
          args.camera,
          args.video_width,
          args.video_height,
      )
    all_rows.extend(indexed_rows)
    all_metric_rows.extend(rows)

  overall = {
      **_aggregate(all_metric_rows),
      **_wandb_compatible_summary(all_metric_rows),
  }
  _write_rows(output_dir / "rollouts.csv", all_rows)
  summary = {
      "metadata": {
          "schema_version": 4,
          "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "checkpoint": str(checkpoint),
          "env_name": env_name,
          "seed": args.seed,
          "evaluation_mode": (
              "random_tasks" if random_task_mode else "constant_commands"
          ),
          "num_rollouts_per_scenario": num_parallel_rollouts,
          "num_random_tasks": args.num_random_tasks,
          "task_seed": args.task_seed if random_task_mode else None,
          "episode_length_environment_steps": episode_length,
          "action_repeat": action_repeat,
          "sample_period_seconds": sample_period,
          "commands": None if random_task_mode else commands,
          "fft_cutoffs_hz": cutoffs_hz,
          "savgol_window_length": args.savgol_window_length,
          "savgol_polyorder": args.savgol_polyorder,
          "feet_height_target_meters": feet_height_target,
          "signals_saved": args.save_signals,
          "video_rendered": args.render_video,
          "used_saved_environment_config": args.use_saved_environment_config,
          "deterministic": args.deterministic,
          "perturbations_disabled": args.disable_perturbations,
          "primary_comparison_metrics": [
              "eval_reward_means/total_without_regularization",
              "tracking/linear_velocity_vector_rmse",
              "tracking/yaw_rate_rmse",
              "smoothness/torque/mean_squared_delta_l2_per_step",
              "smoothness/torque/mssd_mean_squared_second_difference_per_dof",
              "smoothness/torque/msgfd_mean_absolute_savgol_filter_deviation_per_dof",
              "torque_spectrum/eval/fft_above_5hz_energy_per_step",
              "tracking/absolute_mechanical_energy",
              "tracking/total_absolute_torque_impulse",
              "tracking/orientation_error_rms_degrees",
              "tracking/roll_pitch_rate_rms",
              "tracking/feet_height_error_mean_mm",
              "episode/fell",
          ],
      },
      "environment_config": env_config.to_dict(),
      "saved_run_config": saved,
      "overall": overall,
      "scenarios": scenario_summaries,
  }
  with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
    json.dump(_jsonable(summary), fp, indent=2, sort_keys=True)
  flat_summary = {
      "checkpoint": str(checkpoint),
      "env_name": env_name,
      **overall,
      **{
          f"scenario/{scenario}/{key}": value
          for scenario, values in scenario_summaries.items()
          for key, value in values.items()
      },
  }
  _write_rows(output_dir / "summary.csv", [flat_summary])

  if args.use_wandb:
    try:
      import wandb  # pylint: disable=g-import-not-at-top
    except ImportError as error:
      raise ImportError("Install wandb to use --use_wandb.") from error
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"evaluation-{checkpoint.parent.parent.name}-{checkpoint.name}",
        job_type="policy-evaluation",
        config=summary["metadata"],
    )
    run.log(flat_summary)
    artifact = wandb.Artifact(
        f"policy-evaluation-{_safe_name(checkpoint.parent.parent.name)}-{checkpoint.name}",
        type="policy-evaluation",
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)
    run.finish()

  print(f"Evaluation report written to: {output_dir.resolve()}")
  return output_dir


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, help="Checkpoint or checkpoints directory.")
  parser.add_argument("--env_name", help="Required only for legacy checkpoints without run_config.json.")
  parser.add_argument("--output_dir")
  parser.add_argument("--commands", help="JSON object mapping names to [vx, vy, yaw_rate].")
  parser.add_argument("--num_rollouts", type=int, default=8)
  parser.add_argument(
      "--num_random_tasks",
      type=int,
      default=0,
      help=(
          "Evaluate this many environment-sampled tasks in one parallel batch. "
          "This preserves random command schedules instead of using --commands; "
          "the task set is fixed by --task_seed."
      ),
  )
  parser.add_argument(
      "--task_seed",
      type=int,
      default=0,
      help="Fixed reset seed used by --num_random_tasks (default: 0).",
  )
  parser.add_argument("--episode_length", type=int)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--fft_cutoffs_hz", default=",".join(map(str, DEFAULT_FFT_CUTOFFS_HZ)))
  parser.add_argument(
      "--savgol_window_length",
      type=int,
      default=DEFAULT_SAVGOL_WINDOW_LENGTH,
      help="Odd Savitzky-Golay window length used by MSGFD (default: 11).",
  )
  parser.add_argument(
      "--savgol_polyorder",
      type=int,
      default=DEFAULT_SAVGOL_POLYORDER,
      help="Savitzky-Golay polynomial order used by MSGFD (default: 3).",
  )
  parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--disable_perturbations", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      "--save_signals",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Save the full signals.npz trajectory archive (default: false).",
  )
  parser.add_argument(
      "--render_video",
      action=argparse.BooleanOptionalAction,
      default=False,
  )
  parser.add_argument(
      "--require_cuda",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Fail instead of silently evaluating on CPU (default: true).",
  )
  parser.add_argument(
      "--use_saved_environment_config",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Restore the checkpoint's training environment configuration. "
          "Disable for a common default evaluation environment."
      ),
  )
  parser.add_argument("--camera", default="track")
  parser.add_argument("--video_width", type=int, default=640)
  parser.add_argument("--video_height", type=int, default=480)
  parser.add_argument("--use_wandb", action="store_true")
  parser.add_argument("--wandb_project", default="spectral_playground_policy_evaluation")
  parser.add_argument("--wandb_run_name")
  return parser


def main(
    argv: Sequence[str] | None = None,
    rollout_cache: dict[str, Any] | None = None,
):
  args = _build_parser().parse_args(argv)
  if args.require_cuda and jax.default_backend() != "gpu":
    raise RuntimeError(
        "CUDA evaluation was requested, but JAX is using "
        f"{jax.default_backend()!r} with devices {jax.devices()}. Install the "
        "CUDA extra (for example `uv sync --extra cuda`) and verify that "
        "jax.devices() contains a GPU. Use --no-require_cuda only for an "
        "intentional CPU evaluation."
    )
  if args.num_rollouts <= 0:
    raise ValueError("--num_rollouts must be positive.")
  if args.num_random_tasks < 0:
    raise ValueError("--num_random_tasks cannot be negative.")
  if args.num_random_tasks and args.commands is not None:
    raise ValueError("--num_random_tasks cannot be combined with --commands.")
  if args.savgol_window_length <= 0 or args.savgol_window_length % 2 == 0:
    raise ValueError("--savgol_window_length must be a positive odd integer.")
  if (
      args.savgol_polyorder < 0
      or args.savgol_polyorder >= args.savgol_window_length
  ):
    raise ValueError(
        "--savgol_polyorder must be non-negative and smaller than "
        "--savgol_window_length."
    )
  evaluate(args, rollout_cache=rollout_cache)


if __name__ == "__main__":
  main()
