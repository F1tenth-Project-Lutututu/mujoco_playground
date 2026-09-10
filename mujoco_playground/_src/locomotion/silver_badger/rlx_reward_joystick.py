"""SilverBadger joystick task using RL-X's default locomotion reward."""

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.silver_badger import joystick


def default_config() -> config_dict.ConfigDict:
  """Returns RLXHard dynamics with RL-X default reward coefficients."""
  config = joystick.rlx_hard_no_motor_damping_config()
  config.soft_joint_pos_limit_factor = 0.9
  config.reward_config = config_dict.create(
      scales=config_dict.create(
          tracking_xy_velocity_command=2.0,
          tracking_yaw_velocity_command=1.0,
          alive_clipped=0.05,
          alive_unclipped=0.05,
          z_velocity=-2.0,
          imu_acceleration=-1e-4,
          roll_pitch_velocity=-0.05,
          roll_pitch_position=-10.0,
          actuator_joint_nominal_difference=-20.0,
          joint_position_limit=-40.0,
          actuator_joint_velocity_limit=-5.0,
          joint_velocity=-4e-4,
          joint_acceleration=-5e-6,
          joint_torque=-4e-4,
          torque_high_freq=0.0,
          torque_rate=0.0,
          power_draw=-4e-4,
          action_rate=-10.0,
          collision=-2.0,
          base_height=-30.0,
          foot_clearance=0.0,
          foot_air_time=3.0,
          symmetry_air=-1.0,
          foot_slip=-0.1,
          foot_z_velocity=-0.2,
          foot_velocity=0.0,
          foot_flat_contact=-0.01,
      ),
      tracking_xy_temperature=0.25,
      tracking_yaw_temperature=0.25,
      action_rate_use_second_difference=False,
      action_rate_use_fixed_observation=False,
      soft_actuator_joint_velocity_limit=0.9,
      foot_clearance_max_height_m=0.25,
      foot_air_time_per_robot_size_m=0.4,
      robot_dimensions_mean=0.5,
      actuator_joint_max_velocities=(
          3.15,
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
      ),
      torque_highpass_cutoff_hz=5.0,
      torque_highpass_order=1,
      torque_highpass_difference_order=1.0,
      torque_highpass_frequency_normalization="white_spectrum",
      torque_highpass_signal="torque",
      torque_highpass_normalize_by_capacity=False,
      torque_highpass_observe_state=False,
      torque_highpass_observe_state_in_policy=False,
      torque_rate_observe_state=False,
      torque_rate_use_second_difference=False,
      torque_highpass_adaptive_weight=False,
      torque_highpass_adaptive_min_weight=0.1,
      torque_highpass_adaptive_max_weight=1.0,
      torque_highpass_adaptive_sigma=0.25,
      torque_spectrum_cutoffs_hz=(1.0, 2.0, 5.0, 10.0, 15.0, 20.0),
  )
  return config


class RLXRewardJoystick(joystick.Joystick):
  """Joystick environment porting RL-X ``DefaultReward`` semantics."""

  def _post_init(self) -> None:
    super()._post_init()
    self._rlx_max_joint_velocities = jp.asarray(
        self._config.reward_config.actuator_joint_max_velocities
    )
    reward_spheres = np.flatnonzero(self._mj_model.geom_group == 5)
    self._reward_collision_geom_ids = jp.asarray(reward_spheres)
    self._nominal_imu_height = self._init_q[2] - self._floor_pos[2]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = super().reset(rng)
    state.info["previous_joint_velocities"] = state.data.qvel[6:]
    state.info["previous_imu_linear_velocity"] = self.get_local_linvel(
        state.data
    )
    return state

  def _combine_rewards(self, rewards: dict[str, jax.Array]) -> jax.Array:
    alive_unclipped = rewards["alive_unclipped"] * self.dt
    clipped_terms = sum(
        value for name, value in rewards.items() if name != "alive_unclipped"
    ) * self.dt
    return jp.nan_to_num(
        jp.maximum(clipped_terms, 0.0) + alive_unclipped,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

  def _update_reward_memory(
      self, info: dict[str, Any], data: mjx.Data
  ) -> None:
    info["previous_joint_velocities"] = data.qvel[6:]
    info["previous_imu_linear_velocity"] = self.get_local_linvel(data)

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
    del metrics, done, first_contact
    config = self._config.reward_config
    local_linvel = self.get_local_linvel(data)
    gyro = self.get_gyro(data)
    quat_w, quat_x, quat_y, quat_z = data.qpos[3:7]
    roll = jp.arctan2(
        2 * (quat_w * quat_x + quat_y * quat_z),
        1 - 2 * (jp.square(quat_x) + jp.square(quat_y)),
    )
    pitch = jp.arcsin(
        jp.clip(2 * (quat_w * quat_y - quat_z * quat_x), -1.0, 1.0)
    )
    joint_pos = data.qpos[7:]
    joint_vel = data.qvel[6:]
    standing = jp.all(info["command"] == 0.0)
    keep_nominal = jp.where(standing, jp.ones_like(joint_pos), 0.0)

    lower_penalty = -jp.minimum(joint_pos - self._soft_lowers, 0.0).mean()
    upper_penalty = jp.maximum(joint_pos - self._soft_uppers, 0.0).mean()
    velocity_limit = jp.maximum(
        jp.abs(joint_vel)
        - config.soft_actuator_joint_velocity_limit
        * self._rlx_max_joint_velocities,
        0.0,
    ).mean()
    foot_vel = data.sensordata[self._foot_linvel_sensor_adr]
    foot_clearance = jp.minimum(
        self._foot_terrain_clearance(data), config.foot_clearance_max_height_m
    )
    target_air_time = (
        (~standing)
        * config.foot_air_time_per_robot_size_m
        * config.robot_dimensions_mean
    )
    air_time = jp.mean(
        contact * jp.minimum(info["feet_air_time"] - target_air_time, 0.0)
    )
    symmetry_pairs = jp.array(((0, 1), (2, 3)))
    symmetry_air = jp.mean(
        (~contact[symmetry_pairs[:, 0]])
        & (~contact[symmetry_pairs[:, 1]])
    )
    collision_positions = data.geom_xpos[self._reward_collision_geom_ids]
    collision_sizes = self.mjx_model.geom_size[
        self._reward_collision_geom_ids, 0
    ]
    distances = jp.linalg.norm(
        collision_positions[:, None] - collision_positions[None], axis=-1
    )
    collisions = jp.maximum(
        (
            jp.sum(
                distances
                <= collision_sizes[:, None] + collision_sizes[None]
            )
            - collision_positions.shape[0]
        )
        // 2,
        0,
    )
    base_height = (
        data.qpos[2] - self._terrain_height_world(data.qpos[:2])
        - self._nominal_imu_height
    )
    missing_flat_contact = jp.zeros(4)  # SilverBadger feet are spheres.
    return {
        "tracking_xy_velocity_command": jp.exp(
            -jp.sum(jp.square(info["command"][:2] - local_linvel[:2]))
            / config.tracking_xy_temperature
        ),
        "tracking_yaw_velocity_command": jp.exp(
            -jp.square(info["command"][2] - gyro[2])
            / config.tracking_yaw_temperature
        ),
        "alive_clipped": jp.ones(()),
        "alive_unclipped": jp.ones(()),
        "z_velocity": jp.square(local_linvel[2]),
        "imu_acceleration": jp.mean(
            jp.square(
                (local_linvel - info["previous_imu_linear_velocity"])
                / self.dt
            )
        ),
        "roll_pitch_velocity": jp.sum(jp.square(gyro[:2])),
        "roll_pitch_position": jp.square(roll) + jp.square(pitch),
        "actuator_joint_nominal_difference": jp.mean(
            jp.square((joint_pos - self._default_pose) * keep_nominal)
        ),
        "joint_position_limit": lower_penalty + upper_penalty,
        "actuator_joint_velocity_limit": velocity_limit,
        "joint_velocity": jp.mean(jp.square(joint_vel)),
        "joint_acceleration": jp.mean(
            jp.square(
                (info["previous_joint_velocities"] - joint_vel) / self.dt
            )
        ),
        "joint_torque": jp.mean(jp.square(data.actuator_force)),
        "power_draw": jp.mean(jp.maximum(data.actuator_force * joint_vel, 0.0)),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "torque_high_freq": torque_high_freq_cost,
        "torque_rate": torque_rate_cost,
        "collision": collisions,
        "base_height": jp.square(base_height),
        "foot_clearance": jp.mean(jp.square(foot_clearance) * (~contact)),
        "foot_air_time": air_time,
        "symmetry_air": symmetry_air,
        "foot_slip": jp.mean(
            contact * jp.sum(jp.square(foot_vel[:, :2]), axis=1)
        ),
        "foot_z_velocity": jp.mean(jp.square(foot_vel[:, 2])),
        "foot_velocity": jp.mean(jp.sum(jp.square(foot_vel), axis=1)),
        "foot_flat_contact": jp.mean(contact * missing_flat_contact),
    }
