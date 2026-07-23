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
"""Joystick task for Berkeley Humanoid."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.berkeley_humanoid import base as berkeley_humanoid_base
from mujoco_playground._src.locomotion.berkeley_humanoid import berkeley_humanoid_constants as consts
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              hip_pos=0.03,  # rad
              kfe_pos=0.05,
              ffe_pos=0.08,
              faa_pos=0.03,
              joint_vel=1.5,  # rad/s
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,  # angvel.
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              base_height=0.0,
              # Energy related rewards.
              torques=-2.5e-5,
              torque_high_freq=0.0,
              torque_rate=0.0,
              action_rate=-0.01,
              energy=0.0,
              # Feet related rewards.
              feet_clearance=0.0,
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.0,
              # Other rewards.
              stand_still=0.0,
              alive=0.0,
              termination=-1.0,
              # Pose related rewards.
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.25,
              dof_pos_limits=-1.0,
              pose=-1.0,
          ),
          tracking_sigma=0.5,
          max_foot_height=0.1,
          base_height_target=0.5,
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
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-1.0, 1.0],
      ang_vel_yaw=[-1.0, 1.0],
      impl="warp",
      naconmax=8 * 8192,
      njmax=60,
  )


class Joystick(berkeley_humanoid_base.BerkeleyHumanoidEnv):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    hip_indices = []
    hip_joint_names = ["HR", "HAA"]
    for side in ["LL", "LR"]:
      for joint_name in hip_joint_names:
        hip_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}").qposadr - 7
        )
    self._hip_indices = jp.array(hip_indices)

    knee_indices = []
    for side in ["LL", "LR"]:
      knee_indices.append(self._mj_model.joint(f"{side}_KFE").qposadr - 7)
    self._knee_indices = jp.array(knee_indices)

    # fmt: off
    self._weights = jp.array([
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # left leg.
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # right leg.
    ])
    # fmt: on

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
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

    qpos_noise_scale = np.zeros(12)
    hip_ids = [0, 1, 2, 6, 7, 8]
    kfe_ids = [3, 9]
    ffe_ids = [4, 10]
    faa_ids = [5, 11]
    qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
    qpos_noise_scale[kfe_ids] = self._config.noise_config.scales.kfe_pos
    qpos_noise_scale[ffe_ids] = self._config.noise_config.scales.ffe_pos
    qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
    self._qpos_noise_scale = jp.array(qpos_noise_scale)

    cutoff_hz = self._config.reward_config.torque_highpass_cutoff_hz
    nyquist_hz = 0.5 / self.dt
    if not 0.0 < cutoff_hz < nyquist_hz:
      raise ValueError(
          "reward_config.torque_highpass_cutoff_hz must be between 0 and "
          f"the control-rate Nyquist frequency ({nyquist_hz} Hz), got "
          f"{cutoff_hz} Hz."
      )
    self._torque_highpass_order = (
        go1_joystick._validate_torque_highpass_order(  # pylint: disable=protected-access
            self._config.reward_config.torque_highpass_order
        )
    )
    (
        self._torque_highpass_sos,
        self._torque_highpass_steady_state,
    ) = go1_joystick._butterworth_highpass_sos(  # pylint: disable=protected-access
        cutoff_hz, self._torque_highpass_order, 1.0 / self.dt
    )
    self._torque_highpass_difference_order = (
        go1_joystick._validate_torque_difference_order(  # pylint: disable=protected-access
            self._config.reward_config.torque_highpass_difference_order
        )
    )
    difference_order = self._torque_highpass_difference_order
    self._torque_difference_lower_order = int(np.floor(difference_order))
    self._torque_difference_upper_order = int(np.ceil(difference_order))
    self._torque_difference_mix = float(
        difference_order - self._torque_difference_lower_order
    )
    difference_gain_at_cutoff = 2.0 * np.sin(np.pi * cutoff_hz * self.dt)
    self._torque_difference_scale_base = float(1.0 / difference_gain_at_cutoff)
    self._torque_highpass_frequency_normalization = (
        go1_joystick._validate_highpass_frequency_normalization(  # pylint: disable=protected-access
            self._config.reward_config.torque_highpass_frequency_normalization
        )
    )
    self._torque_highpass_frequency_normalizer = 1.0
    if self._torque_highpass_frequency_normalization == "white_spectrum":
      self._torque_highpass_frequency_normalizer = (
          go1_joystick._white_spectrum_frequency_normalizer(  # pylint: disable=protected-access
              self._torque_highpass_sos,
              cutoff_hz,
              1.0 / self.dt,
              difference_order,
          )
      )
    self._torque_highpass_signal = (
        go1_joystick._validate_highpass_penalty_signal(  # pylint: disable=protected-access
            self._config.reward_config.torque_highpass_signal
        )
    )
    self._torque_highpass_observe_state = (
        go1_joystick._validate_observe_highpass_state(  # pylint: disable=protected-access
            self._config.reward_config.torque_highpass_observe_state
        )
    )
    self._torque_rate_observe_state = (
        go1_joystick._validate_observe_torque_rate_state(  # pylint: disable=protected-access
            self._config.reward_config.torque_rate_observe_state
        )
    )
    actuator_joint_ids = self._mj_model.actuator_trnid[:, 0]
    self._torque_capacities = (
        go1_joystick._actuator_force_capacities(  # pylint: disable=protected-access
            self._mj_model.jnt_actfrcrange[actuator_joint_ids]
        )
    )
    (
        self._torque_highpass_adaptive_weight,
        self._torque_highpass_adaptive_min_weight,
        self._torque_highpass_adaptive_max_weight,
        self._torque_highpass_adaptive_sigma,
    ) = go1_joystick._validate_adaptive_highpass_config(  # pylint: disable=protected-access
        self._config.reward_config.torque_highpass_adaptive_weight,
        self._config.reward_config.torque_highpass_adaptive_min_weight,
        self._config.reward_config.torque_highpass_adaptive_max_weight,
        self._config.reward_config.torque_highpass_adaptive_sigma,
    )

    high_freq_scale = self._config.reward_config.scales.torque_high_freq
    torque_rate_scale = self._config.reward_config.scales.torque_rate
    if high_freq_scale > 0.0:
      raise ValueError(
          "reward_config.scales.torque_high_freq must be non-positive."
      )
    if torque_rate_scale > 0.0:
      raise ValueError(
          "reward_config.scales.torque_rate must be non-positive."
      )
    if high_freq_scale < 0.0 or torque_rate_scale < 0.0:
      self._config.reward_config.scales.action_rate = 0.0

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
        go1_joystick._butterworth_highpass_sos(  # pylint: disable=protected-access
            cutoff, 1, 1.0 / self.dt
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

    # Contact sensor IDs.
    self._feet_floor_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in consts.FEET_GEOMS
    ]

  _initial_highpass_state = go1_joystick.Joystick._initial_highpass_state
  _apply_highpass_filter = go1_joystick.Joystick._apply_highpass_filter
  _apply_torque_differences = go1_joystick.Joystick._apply_torque_differences

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (12,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Phase, freq=U(1.0, 1.5)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "last_torque": data.actuator_force,
        "torque_highpass_state": self._initial_highpass_state(
            (
                data.actuator_force / self._torque_capacities
                if self._config.reward_config.torque_highpass_normalize_by_capacity
                else data.actuator_force
            )
            if self._torque_highpass_signal == "torque"
            else jp.zeros(self.mjx_model.nu),
            self._torque_highpass_steady_state,
        ),
        "torque_spectrum_filter_state": self._initial_highpass_state(
            jp.broadcast_to(
                data.actuator_force,
                (len(self._torque_spectrum_metric_names), self.mjx_model.nu),
            ),
            self._torque_spectrum_steady_state,
        ),
        "torque_difference_inputs": jp.zeros(
            (self._torque_difference_upper_order, self.mjx_model.nu)
        ),
        "torque_for_spectrum": data.actuator_force,
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward_without_action_rate"] = jp.zeros(())
    metrics["reward_without_regularization"] = jp.zeros(())
    metrics["torque_highpass/disturbance"] = jp.zeros(())
    metrics["torque_highpass/adaptive_weight"] = jp.asarray(
        self._torque_highpass_adaptive_max_weight
        if self._torque_highpass_adaptive_weight
        else 1.0
    )
    metrics["torque_highpass/frequency_normalizer"] = jp.asarray(
        self._torque_highpass_frequency_normalizer
    )
    metrics["torque_spectrum/total_energy_per_step"] = jp.zeros(())
    for metric_name in self._torque_spectrum_metric_names:
      metrics[metric_name] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])

    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)  # pyrefly: ignore[missing-attribute]

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    done = self._get_termination(data)

    episode_reset = state.info.get("episode_done", False)
    highpass_penalty_signal = (
        (
            data.actuator_force / self._torque_capacities
            if self._config.reward_config.torque_highpass_normalize_by_capacity
            else data.actuator_force
        )
        if self._torque_highpass_signal == "torque"
        else action
    )
    torque_highpass, torque_highpass_state = self._apply_highpass_filter(
        highpass_penalty_signal,
        state.info["torque_highpass_state"],
        self._torque_highpass_sos,
        self._torque_highpass_steady_state,
        episode_reset,
    )
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
    torque_high_freq_cost, torque_difference_inputs = (
        self._apply_torque_differences(
            torque_highpass,
            state.info["torque_difference_inputs"],
            episode_reset,
        )
    )
    torque_high_freq_cost /= self._torque_highpass_frequency_normalizer
    tracking_disturbance = jp.sum(
        jp.square(state.info["command"][:2] - self.get_local_linvel(data)[:2])
    ) + jp.square(state.info["command"][2] - self.get_gyro(data)[2])
    orientation_disturbance = jp.sum(jp.square(self.get_gravity(data)[:2]))
    highpass_disturbance = tracking_disturbance + orientation_disturbance
    highpass_adaptive_weight = jp.asarray(1.0)
    if self._torque_highpass_adaptive_weight:
      highpass_adaptive_weight = (
          go1_joystick._adaptive_highpass_weight(  # pylint: disable=protected-access
              highpass_disturbance,
              self._torque_highpass_adaptive_min_weight,
              self._torque_highpass_adaptive_max_weight,
              self._torque_highpass_adaptive_sigma,
          )
      )
    torque_high_freq_cost *= highpass_adaptive_weight
    torque_rate_cost = self._cost_torque_rate(
        data.actuator_force, state.info["last_torque"]
    )
    state.info["torque_highpass_state"] = torque_highpass_state
    state.info["torque_difference_inputs"] = torque_difference_inputs
    obs = self._get_obs(data, state.info, contact)

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

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_torque"] = data.actuator_force
    state.info["torque_spectrum_filter_state"] = torque_spectrum_filter_state
    state.info["torque_for_spectrum"] = data.actuator_force
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["reward_without_action_rate"] = reward_without_action_rate
    state.metrics["reward_without_regularization"] = (
        reward_without_regularization
    )
    state.metrics["torque_highpass/disturbance"] = highpass_disturbance
    state.metrics["torque_highpass/adaptive_weight"] = highpass_adaptive_weight
    state.metrics["torque_highpass/frequency_normalizer"] = jp.asarray(
        self._torque_highpass_frequency_normalizer
    )
    state.metrics["torque_spectrum/total_energy_per_step"] = jp.sum(
        jp.square(data.actuator_force)
    )
    for metric_name, energy in zip(
        self._torque_spectrum_metric_names, torque_spectrum_energy
    ):
      state.metrics[metric_name] = energy
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
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
        * self._qpos_noise_scale
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        phase,
    ])
    if self._torque_highpass_observe_state:
      state = jp.hstack([
          state,
          go1_joystick._highpass_memory_observation(  # pylint: disable=protected-access
              info
          ),
      ])
    if self._torque_rate_observe_state:
      state = jp.hstack([state, data.actuator_force])

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        data.actuator_force,  # 12
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
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
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "torque_high_freq": torque_high_freq_cost,
        "torque_rate": torque_rate_cost,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
    }

  # Tracking rewards.

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    c1 = jp.sum(jp.square(act - last_act))
    return c1

  def _cost_torque_rate(
      self, torque: jax.Array, last_torque: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.square(torque - last_torque))

  # Other rewards.

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.1)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Pose-related rewards.

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    cost = jp.sum(
        jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
    )
    cost *= jp.abs(cmd[1]) > 0.1
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(
        jp.abs(
            qpos[self._knee_indices] - self._default_pose[self._knee_indices]
        )
    )

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    body_vel = self.get_global_linvel(data)[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    del commands  # Unused.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )
