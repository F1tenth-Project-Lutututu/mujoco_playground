# Copyright 2025 DeepMind Technologies Limited
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
"""Joystick task for SilverBadger."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from scipy import signal as scipy_signal

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion import torque_penalty
from mujoco_playground._src.locomotion.silver_badger import (
    base as silver_badger_base,
)
from mujoco_playground._src.locomotion.silver_badger import (
    silver_badger_constants as consts,
)


_TORQUE_SPECTRUM_DIAGNOSTIC_ORDER = 1
_MAX_TORQUE_HIGHPASS_ORDER = 8
_MAX_TORQUE_DIFFERENCE_ORDER = 8.0
_HIGHPASS_PENALTY_SIGNALS = ("torque", "action")
_HIGHPASS_FREQUENCY_NORMALIZATIONS = ("legacy", "white_spectrum")
_FREQUENCY_NORMALIZER_GRID_SIZE = 16_384


def _butterworth_highpass_sos(
    cutoff_hz: float, order: int, sample_rate_hz: float
) -> tuple[jax.Array, jax.Array]:
  """Designs a digital Butterworth high-pass filter and its steady state."""
  sos = scipy_signal.butter(
      order,
      cutoff_hz,
      btype="highpass",
      fs=sample_rate_hz,
      output="sos",
  ).astype(np.float32)
  steady_state = scipy_signal.sosfilt_zi(sos).astype(np.float32)
  return jp.asarray(sos), jp.asarray(steady_state)


def _validate_torque_highpass_order(value: Any) -> int:
  if (
      isinstance(value, bool)
      or not isinstance(value, (int, np.integer))
      or not 1 <= value <= _MAX_TORQUE_HIGHPASS_ORDER
  ):
    raise ValueError(
        "reward_config.torque_highpass_order must be an integer between "
        f"1 and {_MAX_TORQUE_HIGHPASS_ORDER}, got {value}."
    )
  return int(value)


def _validate_torque_difference_order(value: Any) -> float:
  if (
      isinstance(value, bool)
      or not isinstance(value, (int, float, np.integer, np.floating))
      or not np.isfinite(value)
      or not 0.0 <= value <= _MAX_TORQUE_DIFFERENCE_ORDER
  ):
    raise ValueError(
        "reward_config.torque_highpass_difference_order must be a number "
        f"between 0 and {_MAX_TORQUE_DIFFERENCE_ORDER:g}, got {value}."
    )
  return float(value)


def _validate_highpass_penalty_signal(value: Any) -> str:
  if value not in _HIGHPASS_PENALTY_SIGNALS:
    raise ValueError(
        "reward_config.torque_highpass_signal must be one of "
        f"{_HIGHPASS_PENALTY_SIGNALS}, got {value!r}."
    )
  return value


def _validate_highpass_frequency_normalization(value: Any) -> str:
  if value not in _HIGHPASS_FREQUENCY_NORMALIZATIONS:
    raise ValueError(
        "reward_config.torque_highpass_frequency_normalization must be one "
        f"of {_HIGHPASS_FREQUENCY_NORMALIZATIONS}, got {value!r}."
    )
  return value


def _white_spectrum_frequency_normalizer(
    sos: Any,
    cutoff_hz: float,
    sample_rate_hz: float,
    difference_order: float,
) -> float:
  """Returns the mean penalty weight for unit-variance white input."""
  frequencies, response = scipy_signal.sosfreqz(
      np.asarray(sos),
      worN=_FREQUENCY_NORMALIZER_GRID_SIZE,
      fs=sample_rate_hz,
  )
  lower_order = int(np.floor(difference_order))
  upper_order = int(np.ceil(difference_order))
  interpolation = difference_order - lower_order
  difference_gain = 2.0 * np.sin(np.pi * frequencies / sample_rate_hz)
  gain_at_cutoff = 2.0 * np.sin(np.pi * cutoff_hz / sample_rate_hz)
  normalized_gain = difference_gain / gain_at_cutoff
  difference_weight = (
      (1.0 - interpolation) * normalized_gain ** (2 * lower_order)
      + interpolation * normalized_gain ** (2 * upper_order)
  )
  frequency_weight = np.abs(response) ** 2 * difference_weight
  normalizer = float(np.mean(frequency_weight))
  if not np.isfinite(normalizer) or normalizer <= 0.0:
    raise ValueError(
        "White-spectrum high-pass normalization must be finite and positive, "
        f"got {normalizer}."
    )
  return normalizer


def _validate_observe_highpass_state(value: Any) -> bool:
  if not isinstance(value, (bool, np.bool_)):
    raise ValueError(
        "reward_config.torque_highpass_observe_state must be a boolean."
    )
  return bool(value)


def _validate_observe_torque_rate_state(value: Any) -> bool:
  if not isinstance(value, (bool, np.bool_)):
    raise ValueError(
        "reward_config.torque_rate_observe_state must be a boolean."
    )
  return bool(value)


def _highpass_memory_observation(info: dict[str, Any]) -> jax.Array:
  """Flattens all causal memory used by the high-pass reward."""
  return jp.concatenate((
      jp.ravel(info["torque_highpass_state"]),
      jp.ravel(info["torque_difference_inputs"]),
  ))


def _actuator_force_capacities(force_ranges: Any) -> jax.Array:
  """Returns the largest absolute torque allowed for every actuator."""
  force_ranges = np.asarray(force_ranges)
  if force_ranges.ndim != 2 or force_ranges.shape[-1] != 2:
    raise ValueError(
        "Actuator force limits must have shape (num_actuators, 2), got "
        f"{force_ranges.shape}."
    )
  capacities = np.max(np.abs(force_ranges), axis=-1)
  if not np.all(np.isfinite(capacities)) or np.any(capacities <= 0.0):
    raise ValueError(
        "All actuators need finite, positive force limits to normalize the "
        f"high-pass torque penalty, got {force_ranges}."
    )
  return jp.asarray(capacities)


def _validate_adaptive_highpass_config(
    enabled: Any, min_weight: Any, max_weight: Any, sigma: Any
) -> tuple[bool, float, float, float]:
  """Validates adaptive high-pass penalty configuration."""
  values = (min_weight, max_weight, sigma)
  if not isinstance(enabled, (bool, np.bool_)):
    raise ValueError("torque_highpass_adaptive_weight must be a boolean.")
  if any(
      isinstance(value, bool)
      or not isinstance(value, (int, float, np.integer, np.floating))
      or not np.isfinite(value)
      for value in values
  ):
    raise ValueError(
        "Adaptive high-pass weights and sigma must be finite numbers."
    )
  if not 0.0 <= min_weight <= max_weight:
    raise ValueError(
        "Adaptive high-pass weights must satisfy 0 <= min_weight <= max_weight."
    )
  if sigma <= 0.0:
    raise ValueError("torque_highpass_adaptive_sigma must be positive.")
  return bool(enabled), float(min_weight), float(max_weight), float(sigma)


def _adaptive_highpass_weight(
    disturbance: jax.Array,
    min_weight: float,
    max_weight: float,
    sigma: float,
) -> jax.Array:
  """Decreases the penalty smoothly as disturbance increases."""
  return min_weight + (max_weight - min_weight) * jp.exp(-disturbance / sigma)


def _heightfield_height(
    xy: jax.Array,
    heights: jax.Array,
    x_size: float,
    y_size: float,
    z_scale: float,
) -> jax.Array:
  """Samples MuJoCo's piecewise-planar height-field surface at local xy."""
  ncol, nrow = heights.shape
  grid_x = (xy[..., 0] + x_size) / (2.0 * x_size) * (ncol - 1)
  grid_y = (xy[..., 1] + y_size) / (2.0 * y_size) * (nrow - 1)
  grid_x = jp.clip(grid_x, 0.0, ncol - 1)
  grid_y = jp.clip(grid_y, 0.0, nrow - 1)

  col = jp.minimum(jp.floor(grid_x).astype(int), ncol - 2)
  row = jp.minimum(jp.floor(grid_y).astype(int), nrow - 2)
  u = grid_x - col
  v = grid_y - row

  z00 = heights[col, row]
  z10 = heights[col + 1, row]
  z01 = heights[col, row + 1]
  z11 = heights[col + 1, row + 1]
  lower_triangle = (1.0 - u) * z00 + v * z11 + (u - v) * z10
  upper_triangle = (1.0 - v) * z00 + u * z11 + (v - u) * z01
  return jp.where(u >= v, lower_triangle, upper_triangle) * z_scale


def _terrain_curriculum_difficulty(
    steps: jax.Array,
    initial_difficulty: float,
    ramp_steps: int,
) -> jax.Array:
  """Linearly increases terrain amplitude from initial to full difficulty."""
  progress = jp.clip(steps / ramp_steps, 0.0, 1.0)
  return initial_difficulty + (1.0 - initial_difficulty) * progress


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=20.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      policy_observes_linear_velocity=True,
      domain_randomization=False,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      terrain_curriculum=config_dict.create(
          enabled=True,
          initial_difficulty=0.1,
          ramp_steps=25_000,
      ),
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base reward.
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              # Other.
              dof_pos_limits=-1.0,
              pose=0.5,
              # Other.
              termination=-1.0,
              stand_still=-1.0,
              # Regularization.
              torques=-0.0002,
              torque_high_freq=0.0,
              torque_rate=0.0,
              action_rate=-0.01,
              energy=-0.001,
              # Feet.
              feet_clearance=-2.0,
              feet_height=-0.2,
              feet_slip=-0.1,
              feet_air_time=0.1,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.1,
          torque_highpass_cutoff_hz=5.0,
          torque_highpass_order=1,
          torque_highpass_difference_order=0.0,
          torque_highpass_frequency_normalization="legacy",
          torque_highpass_signal="torque",
          torque_highpass_normalize_by_capacity=True,
          torque_highpass_observe_state=False,
          torque_rate_observe_state=False,
          torque_highpass_adaptive_weight=False,
          torque_highpass_adaptive_min_weight=0.1,
          torque_highpass_adaptive_max_weight=1.0,
          torque_highpass_adaptive_sigma=0.25,
          torque_spectrum_cutoffs_hz=(1.0, 2.0, 5.0, 10.0, 15.0, 20.0),
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[1.5, 0.8, 1.2],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      impl="warp",
      naconmax=4 * 8192,
      naccdmax=4 * 8192,
      njmax=40,
  )


def velocity_25_config() -> config_dict.ConfigDict:
  """Returns the joystick config with a ±2.5 m/s vx command range."""
  config = default_config()
  config.command_config.a[0] = 2.5
  return config


def velocity_35_config() -> config_dict.ConfigDict:
  """Returns the joystick config with a ±3.5 m/s vx command range."""
  config = default_config()
  config.command_config.a[0] = 3.5
  return config


# Retain the terrain-specific name for compatibility with existing callers.
flat_terrain_25_config = velocity_25_config
flat_terrain_35_config = velocity_35_config


def no_linear_velocity_config() -> config_dict.ConfigDict:
  """Returns a config whose actor receives no planar/base linear velocity."""
  config = default_config()
  config.policy_observes_linear_velocity = False
  return config


def variant_config(
    *,
    no_linear_velocity: bool = False,
    pushes: bool = False,
    domain_randomization: bool = False,
) -> config_dict.ConfigDict:
  """Returns a SilverBadger robustness-task configuration."""
  config = default_config()
  config.policy_observes_linear_velocity = not no_linear_velocity
  config.pert_config.enable = pushes
  config.domain_randomization = domain_randomization
  return config


class Joystick(silver_badger_base.SilverBadgerEnv):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if task.startswith("rough"):
      config.naconmax = 8 * 8192
      config.naccdmax = 8 * 8192
      config.njmax = 13 + 48
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  @property
  def action_size(self) -> int:
    """Number of policy-controlled leg joints (the spine stays locked)."""
    return self.mjx_model.nu - 1

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._floor_pos = jp.asarray(
        self._mj_model.geom_pos[self._floor_geom_id]
    )
    self._floor_mat = math.quat_to_mat(
        jp.asarray(self._mj_model.geom_quat[self._floor_geom_id])
    )
    hfield_id = self._mj_model.geom_dataid[self._floor_geom_id]
    self._terrain_hfield = None
    if hfield_id >= 0:
      nrow = self._mj_model.hfield_nrow[hfield_id]
      ncol = self._mj_model.hfield_ncol[hfield_id]
      adr = self._mj_model.hfield_adr[hfield_id]
      self._terrain_hfield = jp.asarray(
          self._mj_model.hfield_data[adr : adr + nrow * ncol].reshape(
              (ncol, nrow), order="F"
          )
      )
      self._terrain_hfield_size = jp.asarray(
          self._mj_model.hfield_size[hfield_id]
      )
    curriculum = self._config.terrain_curriculum
    if not 0.0 <= curriculum.initial_difficulty <= 1.0:
      raise ValueError(
          "terrain_curriculum.initial_difficulty must be between 0 and 1."
      )
    if curriculum.ramp_steps <= 0:
      raise ValueError("terrain_curriculum.ramp_steps must be positive.")
    self._terrain_curriculum_enabled = bool(
        self._terrain_hfield is not None and curriculum.enabled
    )
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    cutoff_hz = self._config.reward_config.torque_highpass_cutoff_hz
    nyquist_hz = 0.5 / self.dt
    self._torque_penalty = torque_penalty.TorquePenalty(
        self._config.reward_config, self._mj_model, self.dt
    )
    spectrum_cutoffs_hz = tuple(
        self._config.reward_config.torque_spectrum_cutoffs_hz
    )
    if not spectrum_cutoffs_hz:
      raise ValueError(
          "reward_config.torque_spectrum_cutoffs_hz must not be empty."
      )
    if any(not 0.0 < cutoff < nyquist_hz for cutoff in spectrum_cutoffs_hz):
      raise ValueError(
          "All reward_config.torque_spectrum_cutoffs_hz values must be "
          f"between 0 and {nyquist_hz} Hz, got {spectrum_cutoffs_hz}."
      )
    spectrum_filters = [
        _butterworth_highpass_sos(
            cutoff,
            _TORQUE_SPECTRUM_DIAGNOSTIC_ORDER,
            1.0 / self.dt,
        )
        for cutoff in spectrum_cutoffs_hz
    ]
    self._torque_spectrum_sos = jp.stack(
        [filter_sos for filter_sos, _ in spectrum_filters]
    )
    self._torque_spectrum_steady_state = jp.stack(
        [steady_state for _, steady_state in spectrum_filters]
    )
    self._torque_spectrum_metric_names = tuple(
        f"torque_spectrum/highpass_{cutoff:g}hz_per_step"
        for cutoff in spectrum_cutoffs_hz
    )

  def _initial_highpass_state(
      self, signal: jax.Array, steady_state: jax.Array
  ) -> jax.Array:
    """Returns filter state that initially produces zero high-pass output."""
    return steady_state[..., None] * signal[..., None, None, :]

  def _apply_highpass_filter(
      self,
      signal: jax.Array,
      previous_state: jax.Array,
      sos: jax.Array,
      steady_state: jax.Array,
      reset: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Applies a Butterworth SOS filter in direct-form II transposed form."""
    initial_state = self._initial_highpass_state(signal, steady_state)
    previous_state = jp.where(reset, initial_state, previous_state)
    filtered = signal
    next_states = []
    for section in range(sos.shape[-2]):
      coefficients = sos[..., section, :]
      section_state = previous_state[..., section, :, :]
      output = coefficients[..., 0, None] * filtered + section_state[..., 0, :]
      state_0 = (
          coefficients[..., 1, None] * filtered
          - coefficients[..., 4, None] * output
          + section_state[..., 1, :]
      )
      state_1 = (
          coefficients[..., 2, None] * filtered
          - coefficients[..., 5, None] * output
      )
      filtered = output
      next_states.append(jp.stack((state_0, state_1), axis=-2))
    return filtered, jp.stack(next_states, axis=-3)

  def _apply_torque_differences(
      self,
      signal: jax.Array,
      previous_inputs: jax.Array,
      reset: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Compatibility wrapper for environments using the former Go1 helper."""
    return torque_penalty.apply_torque_differences(
        signal,
        previous_inputs,
        reset,
        self._torque_difference_lower_order,
        self._torque_difference_upper_order,
        self._torque_difference_mix,
        self._torque_difference_scale_base,
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    terrain_difficulty = self._terrain_difficulty(jp.asarray(0))
    qpos = qpos.at[2].set(
        self._init_q[2]
        + self._terrain_height_world(qpos[:2], terrain_difficulty)
        - self._floor_pos[2]
    )
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    terrain_model = self._terrain_model(terrain_difficulty)
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(terrain_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * 5.0
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )
    cmd = jax.random.uniform(
        key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )

    info = {
        "rng": rng,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "torque_spectrum_filter_state": self._initial_highpass_state(
            jp.broadcast_to(
                data.actuator_force,
                (len(self._torque_spectrum_metric_names), self.mjx_model.nu),
            ),
            self._torque_spectrum_steady_state,
        ),
        "torque_for_spectrum": data.actuator_force,
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "terrain_curriculum_steps": jp.asarray(0, dtype=jp.int32),
        "terrain_difficulty": terrain_difficulty,
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
    }
    self._torque_penalty.reset(info, data.actuator_force)

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward_without_action_rate"] = jp.zeros(())
    metrics["reward_without_regularization"] = jp.zeros(())
    metrics["torque_highpass/disturbance"] = jp.zeros(())
    metrics["torque_highpass/adaptive_weight"] = jp.asarray(
        self._torque_penalty.adaptive_max_weight
        if self._torque_penalty.adaptive_enabled
        else 1.0
    )
    metrics["torque_highpass/frequency_normalizer"] = jp.asarray(
        self._torque_penalty.frequency_normalizer
    )
    metrics["torque_spectrum/total_energy_per_step"] = jp.zeros(())
    for metric_name in self._torque_spectrum_metric_names:
      metrics[metric_name] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    metrics["terrain/difficulty"] = terrain_difficulty

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  # def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
  #   qpos = state.data.qpos
  #   new_x = jp.where(jp.abs(qpos[0]) > 9.5, 0.0, qpos[0])
  #   new_y = jp.where(jp.abs(qpos[1]) > 9.5, 0.0, qpos[1])
  #   qpos = qpos.at[0:2].set(jp.array([new_x, new_y]))
  #   state = state.replace(data=state.data.replace(qpos=qpos))
  #   return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    # state = self._reset_if_outside_bounds(state)

    terrain_difficulty = self._terrain_difficulty(
        state.info["terrain_curriculum_steps"]
    )
    terrain_model = self._terrain_model(terrain_difficulty)
    episode_reset = state.info.get("episode_done", False)

    def align_reset_height(data: mjx.Data) -> mjx.Data:
      qpos = data.qpos.at[2].set(
          self._init_q[2]
          + self._terrain_height_world(data.qpos[:2], terrain_difficulty)
          - self._floor_pos[2]
      )
      return mjx.forward(terrain_model, data.replace(qpos=qpos))

    data = jax.lax.cond(
        episode_reset, align_reset_height, lambda data: data, state.data
    )

    motor_targets = jp.concatenate((
        self._default_pose[:1],
        self._default_pose[1:] + action * self._config.action_scale,
    ))
    data = mjx_env.step(
        terrain_model, data, motor_targets, self.n_substeps
    )

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    foot_clearance = self._foot_terrain_clearance(data, terrain_difficulty)
    state.info["swing_peak"] = jp.maximum(
        state.info["swing_peak"], foot_clearance
    )

    done = self._get_termination(data)

    torque_spectrum_highpass, torque_spectrum_filter_state = (
        self._apply_highpass_filter(
            jp.broadcast_to(
                data.actuator_force,
                (len(self._torque_spectrum_metric_names), self.mjx_model.nu),
            ),
            state.info["torque_spectrum_filter_state"],
            self._torque_spectrum_sos,
            self._torque_spectrum_steady_state,
            episode_reset,
        )
    )
    torque_spectrum_energy = jp.sum(
        jp.square(torque_spectrum_highpass), axis=-1
    )

    torque_high_freq_cost, torque_rate_cost = self._torque_penalty.compute(
        state.info,
        data.actuator_force,
        action,
        episode_reset,
    )
    tracking_disturbance = jp.sum(
        jp.square(state.info["command"][:2] - self.get_local_linvel(data)[:2])
    ) + jp.square(state.info["command"][2] - self.get_gyro(data)[2])
    orientation_disturbance = jp.sum(jp.square(self.get_upvector(data)[:2]))
    highpass_disturbance = tracking_disturbance + orientation_disturbance
    torque_high_freq_cost, highpass_adaptive_weight = (
        self._torque_penalty.apply_adaptive_weight(
            torque_high_freq_cost, highpass_disturbance
        )
    )
    obs = self._get_obs(data, state.info)

    rewards = self._get_reward(
        data,
        action,
        state.info,
        state.metrics,
        done,
        first_contact,
        contact,
        torque_high_freq_cost,
        torque_rate_cost,
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    reward_without_action_rate = jp.clip(
        sum(v for k, v in rewards.items() if k != "action_rate") * self.dt,
        0.0,
        10000.0,
    )
    reward_without_regularization = jp.clip(
        sum(
            v
            for k, v in rewards.items()
            if k not in ("action_rate", "torque_high_freq", "torque_rate")
        )
        * self.dt,
        0.0,
        10000.0,
    )

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["torque_spectrum_filter_state"] = torque_spectrum_filter_state
    state.info["torque_for_spectrum"] = data.actuator_force
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["command"]),
        state.info["command"],
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    state.info["terrain_curriculum_steps"] += 1
    state.info["terrain_difficulty"] = terrain_difficulty
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["reward_without_action_rate"] = reward_without_action_rate
    state.metrics["reward_without_regularization"] = (
        reward_without_regularization
    )
    state.metrics["torque_highpass/disturbance"] = highpass_disturbance
    state.metrics["torque_highpass/adaptive_weight"] = highpass_adaptive_weight
    state.metrics["torque_highpass/frequency_normalizer"] = jp.asarray(
        self._torque_penalty.frequency_normalizer
    )
    state.metrics["torque_spectrum/total_energy_per_step"] = jp.sum(
        jp.square(data.actuator_force)
    )
    for metric_name, energy in zip(
        self._torque_spectrum_metric_names, torque_spectrum_energy
    ):
      state.metrics[metric_name] = energy
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    state.metrics["terrain/difficulty"] = terrain_difficulty

    done = done.astype(reward.dtype)
    state = state.replace(
        data=data, obs=obs, reward=reward, done=done
    )  # pyrefly: ignore[missing-attribute]
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    return fall_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    policy_linear_velocity = (
        noisy_linvel
        if self._config.policy_observes_linear_velocity
        else jp.zeros((0,))
    )
    state = jp.hstack([
        policy_linear_velocity,  # 3 when enabled, otherwise omitted.
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 13
        noisy_joint_vel,  # 13
        info["last_act"],  # 12 leg actions; the spine is locked.
        info["command"],  # 3
    ])
    state = jp.hstack([
        state,
        self._torque_penalty.observation(info, data.actuator_force),
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        angvel,  # 3
        joint_angles - self._default_pose,  # 13
        joint_vel,  # 13
        data.actuator_force,  # 13
        info["last_contact"],  # 4
        feet_vel,  # 4*3
        info["feet_air_time"],  # 4
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
      torque_high_freq_cost: jax.Array,
      torque_rate_cost: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "torque_high_freq": torque_high_freq_cost,
        "torque_rate": torque_rate_cost,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(
            data, info["terrain_difficulty"]
        ),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
    }

  # Tracking rewards.

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_torque_rate(
      self, torque: jax.Array, last_torque: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.square(torque - last_torque))

  # Other rewards.

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    # Stay close to the default pose.
    # Spine, followed by hip/thigh/calf for each of four legs.
    weight = jp.array([1.0] + [1.0, 1.0, 0.1] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _terrain_difficulty(self, steps: jax.Array) -> jax.Array:
    if not self._terrain_curriculum_enabled:
      return jp.asarray(1.0)
    curriculum = self._config.terrain_curriculum
    return _terrain_curriculum_difficulty(
        steps, curriculum.initial_difficulty, curriculum.ramp_steps
    )

  def _terrain_model(self, difficulty: jax.Array) -> mjx.Model:
    if self._terrain_hfield is None:
      return self.mjx_model
    return self.mjx_model.tree_replace({
        "hfield_data": self.mjx_model.hfield_data * difficulty
    })

  def _terrain_height_world(
      self, world_xy: jax.Array, difficulty: jax.Array
  ) -> jax.Array:
    """Returns terrain surface world-z below a world-frame xy position."""
    if self._terrain_hfield is None:
      return self._floor_pos[2]
    world_pos = jp.array([world_xy[0], world_xy[1], self._floor_pos[2]])
    local_xy = ((world_pos - self._floor_pos) @ self._floor_mat)[:2]
    local_z = _heightfield_height(
        local_xy,
        self._terrain_hfield,
        self._terrain_hfield_size[0],
        self._terrain_hfield_size[1],
        self._terrain_hfield_size[2] * difficulty,
    )
    local_surface = jp.array([local_xy[0], local_xy[1], local_z])
    return (self._floor_pos + local_surface @ self._floor_mat.T)[2]

  def _foot_terrain_clearance(
      self, data: mjx.Data, difficulty: jax.Array
  ) -> jax.Array:
    """Returns foot height above the terrain directly below each foot."""
    foot_pos = data.site_xpos[self._feet_site_id]
    floor_pos = data.geom_xpos[self._floor_geom_id]
    floor_mat = data.geom_xmat[self._floor_geom_id]
    foot_pos_local = (foot_pos - floor_pos) @ floor_mat
    if self._terrain_hfield is None:
      terrain_height = jp.zeros_like(foot_pos_local[..., 2])
    else:
      terrain_height = _heightfield_height(
          foot_pos_local[..., :2],
          self._terrain_hfield,
          self._terrain_hfield_size[0],
          self._terrain_hfield_size[1],
          self._terrain_hfield_size[2] * difficulty,
      )
    return foot_pos_local[..., 2] - terrain_height

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

  def _cost_feet_clearance(
      self, data: mjx.Data, terrain_difficulty: jax.Array
  ) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_clearance = self._foot_terrain_clearance(
        data, terrain_difficulty
    )
    delta = jp.abs(
        foot_clearance - self._config.reward_config.max_foot_height
    )
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
    return rew_air_time

  # Perturbation and command sampling.

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
      force = (
          u_t  # (unitless)
          * self._torso_mass  # kg
          * state.info["pert_mag"]  # m/s
          / state.info["pert_duration_seconds"]  # 1/s
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)  # pyrefly: ignore[missing-attribute]
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)  # pyrefly: ignore[missing-attribute]

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    return x_kp1
