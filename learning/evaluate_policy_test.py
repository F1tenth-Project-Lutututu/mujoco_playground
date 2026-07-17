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
"""Tests for constant-command policy evaluation metrics."""

from absl.testing import absltest
from flax import struct
import jax
import jax.numpy as jp
from ml_collections import config_dict
import numpy as np

from learning import evaluate_policy


@struct.dataclass
class _FakeData:
  qpos: jax.Array
  qvel: jax.Array
  qacc: jax.Array
  ctrl: jax.Array
  actuator_force: jax.Array
  sensordata: jax.Array
  mocap_pos: jax.Array
  mocap_quat: jax.Array
  xfrc_applied: jax.Array


@struct.dataclass
class _FakeState:
  data: _FakeData
  obs: dict[str, jax.Array]
  reward: jax.Array
  done: jax.Array
  metrics: dict[str, jax.Array]
  info: dict[str, jax.Array]


class _FakeEnv:

  def reset(self, rng):
    del rng
    data = _FakeData(
        qpos=jp.zeros(19),
        qvel=jp.zeros(18),
        qacc=jp.zeros(18),
        ctrl=jp.zeros(12),
        actuator_force=jp.zeros(12),
        sensordata=jp.zeros(9),
        mocap_pos=jp.zeros((0, 3)),
        mocap_quat=jp.zeros((0, 4)),
        xfrc_applied=jp.zeros((1, 6)),
    )
    info = {
        "command": jp.zeros(3),
        "steps_until_next_cmd": jp.asarray(1, dtype=jp.int32),
        "last_contact": jp.zeros(4, dtype=bool),
        "feet_air_time": jp.zeros(4),
    }
    return _FakeState(
        data=data,
        obs=self._get_obs(data, info),
        reward=jp.asarray(0.0),
        done=jp.asarray(0.0),
        metrics={"reward/test": jp.asarray(0.0)},
        info=info,
    )

  def _get_obs(self, data, info):
    del data
    return {
        "state": jp.pad(info["command"], (0, 45)),
        "privileged_state": jp.pad(info["command"], (0, 120)),
    }

  def step(self, state, action):
    qpos = state.data.qpos.at[0].add(1.0)
    data = state.data.replace(
        qpos=qpos,
        ctrl=action,
        actuator_force=2.0 * action,
    )
    done = qpos[0] >= 2.0
    return state.replace(
        data=data,
        obs=self._get_obs(data, state.info),
        reward=jp.asarray(1.0),
        done=done.astype(jp.float32),
        metrics={"reward/test": jp.asarray(0.5)},
    )

  def get_local_linvel(self, data):
    del data
    return jp.zeros(3)

  def get_gyro(self, data):
    del data
    return jp.zeros(3)

  def get_upvector(self, data):
    del data
    return jp.asarray([0.0, 0.0, 1.0])

  def get_accelerometer(self, data):
    del data
    return jp.zeros(3)

  def get_feet_pos(self, data):
    del data
    return jp.zeros((4, 3))


class _RandomTaskFakeEnv(_FakeEnv):

  def reset(self, rng):
    state = super().reset(rng)
    command = jax.random.uniform(rng, (3,), minval=-1.0, maxval=1.0)
    info = dict(state.info)
    info["command"] = command
    return state.replace(info=info, obs=self._get_obs(state.data, info))


class EvaluatePolicyTest(absltest.TestCase):

  def test_expensive_artifacts_are_disabled_and_cuda_required_by_default(self):
    args = evaluate_policy._build_parser().parse_args(["--checkpoint", "test"])

    self.assertFalse(args.render_video)
    self.assertFalse(args.save_signals)
    self.assertTrue(args.require_cuda)
    self.assertTrue(args.use_saved_environment_config)

  def test_rollout_cache_key_depends_on_structure_not_weights(self):
    args = evaluate_policy._build_parser().parse_args(["--checkpoint", "test"])
    env_config = config_dict.ConfigDict({"dt": 0.02, "option": 1})
    network_config = config_dict.ConfigDict({
        "action_size": 12,
        "network_factory_kwargs": {"activation": jax.nn.silu},
    })

    first = evaluate_policy._rollout_cache_key(
        "fake",
        env_config,
        network_config,
        args,
        action_repeat=1,
        episode_length=1000,
        num_parallel_rollouts=512,
        random_task_mode=True,
    )
    second = evaluate_policy._rollout_cache_key(
        "fake",
        env_config,
        network_config,
        args,
        action_repeat=1,
        episode_length=1000,
        num_parallel_rollouts=512,
        random_task_mode=True,
    )

    self.assertEqual(first, second)
    self.assertNotIn("params", first)

  def test_rollout_records_and_masks_fixed_command_episode(self):
    env = _FakeEnv()

    def policy(obs, key):
      del obs, key
      return jp.ones(12), {}

    signals = evaluate_policy._rollout(
        env,
        policy,
        command=np.asarray([0.5, 0.0, 0.2]),
        reset_keys=jax.random.split(jax.random.PRNGKey(0), 2),
        episode_length=4,
        action_repeat=1,
        policy_seed=1,
    )
    np.testing.assert_array_equal(
        np.asarray(signals["active"][:, 0]),
        np.asarray([True, True, False, False]),
    )
    np.testing.assert_allclose(
        np.asarray(signals["command"][:, 0]),
        np.tile([0.5, 0.0, 0.2], (4, 1)),
    )
    self.assertEqual(signals["action"].shape, (4, 2, 12))
    self.assertNotIn("qpos", signals)
    self.assertNotIn("obs/state", signals)
    self.assertNotIn("render/qpos", signals)

  def test_rollout_records_full_and_render_signals_only_when_requested(self):
    env = _FakeEnv()

    def policy(obs, key):
      del obs, key
      return jp.ones(12), {}

    signals = evaluate_policy._rollout(
        env,
        policy,
        command=np.asarray([0.5, 0.0, 0.2]),
        reset_keys=jax.random.split(jax.random.PRNGKey(0), 1),
        episode_length=2,
        action_repeat=1,
        policy_seed=1,
        record_full_signals=True,
        record_render_signals=True,
    )

    self.assertIn("qpos", signals)
    self.assertIn("obs/state", signals)
    self.assertIn("render/qpos", signals)

  def test_random_tasks_are_parallel_and_reproducible(self):
    env = _RandomTaskFakeEnv()

    def policy(obs, key):
      del obs, key
      return jp.ones(12), {}

    keys = jax.random.split(jax.random.PRNGKey(17), 3)
    first = jax.jit(
        lambda: evaluate_policy._rollout(
            env,
            policy,
            None,
            keys,
            episode_length=2,
            action_repeat=1,
            policy_seed=1,
        )
    )()
    second = evaluate_policy._rollout(
        env, policy, None, keys, episode_length=2, action_repeat=1,
        policy_seed=999,
    )
    self.assertEqual(first["command"].shape, (2, 3, 3))
    np.testing.assert_array_equal(first["command"], second["command"])
    self.assertFalse(
        np.array_equal(first["command"][:, 0], first["command"][:, 1])
    )

  def test_episode_rows_use_random_task_command_schedule(self):
    time_steps = 4
    command = np.broadcast_to(
        np.asarray([0.5, -0.25, 0.2]), (time_steps, 1, 3)
    )
    signals = {
        "active": np.ones((time_steps, 1), dtype=bool),
        "done": np.zeros((time_steps, 1), dtype=bool),
        "reward": np.zeros((time_steps, 1)),
        "command": command,
        "action": np.zeros((time_steps, 1, 12)),
        "motor_target": np.zeros((time_steps, 1, 12)),
        "actuator_force": np.zeros((time_steps, 1, 12)),
        "local_linear_velocity": command,
        "gyro": np.broadcast_to([0.0, 0.0, 0.2], (time_steps, 1, 3)),
        "upvector": np.broadcast_to(
            [0.0, 0.0, 1.0], (time_steps, 1, 3)
        ),
        "qvel": np.zeros((time_steps, 1, 18)),
    }
    rows = evaluate_policy._episode_rows(
        signals, command=None, sample_period=0.02, cutoffs_hz=(1.0,)
    )
    self.assertEqual(rows[0]["tracking/linear_velocity_vector_rmse"], 0.0)
    self.assertEqual(rows[0]["tracking/yaw_rate_rmse"], 0.0)

  def test_parse_commands(self):
    commands = evaluate_policy._parse_commands(
        '{"forward": [1, 0, 0], "turn": [0, 0, -0.5]}'
    )
    self.assertEqual(commands["forward"], (1.0, 0.0, 0.0))
    self.assertEqual(commands["turn"], (0.0, 0.0, -0.5))

  def test_parse_commands_rejects_wrong_dimension(self):
    with self.assertRaisesRegex(ValueError, "exactly three"):
      evaluate_policy._parse_commands('{"bad": [1, 2]}')

  def test_fft_energy_obeys_parseval_for_constant_signal(self):
    signal = np.full((100, 3), 2.0)
    metrics = evaluate_policy._fft_energy_metrics(
        signal, sample_period=0.02, cutoffs_hz=(1.0, 5.0)
    )
    self.assertAlmostEqual(metrics["total_energy_per_step"], 12.0)
    self.assertAlmostEqual(metrics["ac_energy_per_step"], 0.0)
    self.assertAlmostEqual(metrics["fft_above_1hz_energy_per_step"], 0.0)

  def test_fft_detects_high_frequency_energy(self):
    times = np.arange(100) * 0.02
    signal = np.sin(2.0 * np.pi * 10.0 * times)[:, None]
    metrics = evaluate_policy._fft_energy_metrics(
        signal, sample_period=0.02, cutoffs_hz=(5.0, 15.0)
    )
    self.assertAlmostEqual(
        metrics["fft_above_5hz_energy_per_step"], 0.5, places=6
    )
    self.assertAlmostEqual(
        metrics["fft_above_15hz_energy_per_step"], 0.0, places=6
    )
    self.assertAlmostEqual(metrics["spectral_centroid_hz"], 10.0, places=6)

  def test_smoothness_matches_mean_action_rate_definition(self):
    signal = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    metrics = evaluate_policy._smoothness_metrics(
        signal, sample_period=0.5, cutoffs_hz=(0.5,)
    )
    self.assertAlmostEqual(
        metrics["mean_squared_delta_l2_per_step"], 5.0
    )
    self.assertAlmostEqual(metrics["delta_rms_per_dof"], np.sqrt(2.5))

  def test_mssd_and_msgfd_for_quadratic_signal(self):
    time = np.arange(9, dtype=np.float64)
    signal = np.stack((time**2, 2.0 * time**2), axis=-1)
    metrics = evaluate_policy._smoothness_metrics(
        signal,
        sample_period=0.02,
        cutoffs_hz=(1.0,),
        savgol_window_length=5,
        savgol_polyorder=2,
    )
    # The second differences are 2 and 4 for the two degrees of freedom.
    self.assertAlmostEqual(
        metrics["mssd_mean_squared_second_difference_per_dof"], 10.0
    )
    self.assertAlmostEqual(
        metrics["msgfd_mean_absolute_savgol_filter_deviation_per_dof"],
        0.0,
        places=12,
    )

  def test_msgfd_detects_deviation_from_local_polynomial(self):
    signal = np.zeros((11, 1))
    signal[5] = 1.0
    metrics = evaluate_policy._smoothness_metrics(
        signal,
        sample_period=0.02,
        cutoffs_hz=(1.0,),
        savgol_window_length=5,
        savgol_polyorder=2,
    )
    self.assertGreater(
        metrics["msgfd_mean_absolute_savgol_filter_deviation_per_dof"], 0.0
    )

  def test_feet_height_error_uses_completed_swing_peak(self):
    feet_position = np.zeros((6, 1, 3))
    feet_position[:, 0, 2] = [0.0, 0.02, 0.12, 0.09, 0.0, 0.0]
    feet_contact = np.asarray([[1], [0], [0], [0], [1], [1]], dtype=bool)
    metrics = evaluate_policy._feet_height_metrics(
        feet_position,
        feet_contact,
        command=np.asarray([0.5, 0.0, 0.0]),
        target_height=0.1,
    )
    self.assertAlmostEqual(metrics["feet_height_error_mean_mm"], 20.0)
    self.assertAlmostEqual(metrics["feet_height_error_rmse_mm"], 20.0)
    self.assertEqual(metrics["feet_height_touchdowns"], 1.0)

  def test_tracking_metrics_include_physical_totals_and_orientation(self):
    metrics = evaluate_policy._tracking_metrics(
        command=np.zeros(3),
        local_linear_velocity=np.zeros((2, 3)),
        gyro=np.asarray([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]),
        upvector=np.broadcast_to(
            [np.sin(np.deg2rad(30.0)), 0.0, np.cos(np.deg2rad(30.0))],
            (2, 3),
        ),
        qvel=np.concatenate((np.zeros((2, 6)), np.ones((2, 2))), axis=1),
        torque=np.full((2, 2), 3.0),
        sample_period=0.5,
    )
    self.assertAlmostEqual(metrics["absolute_mechanical_energy"], 6.0)
    self.assertAlmostEqual(metrics["total_absolute_torque_impulse"], 6.0)
    self.assertAlmostEqual(metrics["orientation_error_rms_degrees"], 30.0)
    self.assertAlmostEqual(metrics["roll_pitch_rate_rms"], np.sqrt(2.5))

  def test_episode_rows_mask_after_fall_and_emit_fair_reward(self):
    time_steps, rollouts = 8, 2
    active = np.ones((time_steps, rollouts), dtype=bool)
    active[4:, 1] = False
    done = np.zeros_like(active)
    done[3, 1] = True
    signals = {
        "active": active,
        "done": done,
        "reward": np.ones((time_steps, rollouts)),
        "action": np.zeros((time_steps, rollouts, 12)),
        "motor_target": np.zeros((time_steps, rollouts, 12)),
        "actuator_force": np.zeros((time_steps, rollouts, 12)),
        "local_linear_velocity": np.zeros((time_steps, rollouts, 3)),
        "gyro": np.zeros((time_steps, rollouts, 3)),
        "upvector": np.broadcast_to([0.0, 0.0, 1.0], (time_steps, rollouts, 3)),
        "qvel": np.zeros((time_steps, rollouts, 18)),
        "metric/reward_without_regularization": np.full(
            (time_steps, rollouts), 0.25
        ),
        "metric/reward/tracking_lin_vel": np.full(
            (time_steps, rollouts), 0.5
        ),
    }
    rows = evaluate_policy._episode_rows(
        signals,
        command=np.zeros(3),
        sample_period=0.02,
        cutoffs_hz=(1.0, 5.0),
    )
    self.assertEqual(rows[0]["episode/length_steps"], 8)
    self.assertEqual(rows[0]["episode/fell"], 0)
    self.assertEqual(rows[1]["episode/length_steps"], 4)
    self.assertEqual(rows[1]["episode/fell"], 1)
    self.assertAlmostEqual(
        rows[1]["eval_reward_means/total_without_regularization"], 1.0
    )

  def test_wandb_summary_includes_reward_mean_and_std(self):
    rows = [
        {"eval_reward_means/total": 1.0, "episode/length_steps": 10.0},
        {"eval_reward_means/total": 3.0, "episode/length_steps": 8.0},
    ]
    summary = evaluate_policy._wandb_compatible_summary(rows)
    self.assertEqual(summary["eval_reward_means/total"], 2.0)
    self.assertEqual(summary["eval_reward_stds/total"], 1.0)
    self.assertEqual(summary["rollouts/eval_avg_episode_length"], 9.0)


if __name__ == "__main__":
  absltest.main()
