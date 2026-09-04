"""Tests for the SilverBadger joystick environment."""

from absl.testing import absltest
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mujoco_playground import registry
from mujoco_playground._src.locomotion.silver_badger import joystick


class JoystickTest(absltest.TestCase):

  def test_heightfield_height_matches_mujoco_triangles(self):
    heights = jp.array([[0.0, 2.0], [1.0, 4.0]])
    points = jp.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.5, -0.5],
        [-0.5, 0.5],
    ])

    actual = joystick._heightfield_height(  # pylint: disable=protected-access
        points, heights, x_size=1.0, y_size=1.0, z_scale=0.5
    )

    np.testing.assert_allclose(actual, [0.0, 0.5, 1.0, 2.0, 0.75, 1.0])

  def test_default_tracking_reward_weights(self):
    config = joystick.default_config()

    self.assertEqual(config.reward_config.scales.tracking_lin_vel, 2.0)
    self.assertEqual(config.reward_config.scales.tracking_ang_vel, 1.0)
    self.assertEqual(config.Kp, 20.0)
    self.assertEqual(config.action_scale, 1.0)
    self.assertNotIn("terrain_curriculum", config)

  def test_reset_and_step_shapes(self):
    config = joystick.default_config()
    config.impl = "jax"
    env = joystick.Joystick(config=config)

    state = env.reset(jax.random.PRNGKey(0))
    next_state = env.step(state, jp.ones(env.action_size))

    self.assertEqual(env.action_size, 12)
    self.assertEqual(state.obs["state"].shape, (50,))
    self.assertEqual(state.obs["privileged_state"].shape, (128,))
    self.assertEqual(next_state.obs["state"].shape, (50,))
    self.assertEqual(next_state.reward.shape, ())
    self.assertEqual(next_state.done.shape, ())
    self.assertEqual(next_state.data.ctrl[0], env._default_pose[0])
    np.testing.assert_allclose(
        next_state.data.ctrl[1:],
        env._default_pose[1:] + config.action_scale,
    )

  def test_no_linear_velocity_actor_observation(self):
    config = joystick.no_linear_velocity_config()
    config.impl = "jax"
    env = joystick.Joystick(config=config)

    state = env.reset(jax.random.PRNGKey(0))

    self.assertFalse(config.policy_observes_linear_velocity)
    self.assertEqual(state.obs["state"].shape, (47,))
    # The critic still receives the ground-truth local linear velocity.
    self.assertEqual(state.obs["privileged_state"].shape, (125,))

  def test_noisy_highpass_memory_is_capacity_relative(self):
    info = {
        "torque_highpass_state": jp.ones((1, 2, 3)),
        "torque_difference_inputs": jp.ones((1, 3)),
    }
    capacity = jp.array([10.0, 20.0, 30.0])

    unchanged = joystick._noisy_highpass_memory_observation(  # pylint: disable=protected-access
        info, capacity, 0.0, 0.0, jax.random.PRNGKey(0)
    )
    noised = joystick._noisy_highpass_memory_observation(  # pylint: disable=protected-access
        info, capacity, 0.01, 0.01, jax.random.PRNGKey(0)
    )

    np.testing.assert_allclose(unchanged, np.ones(9))
    self.assertFalse(np.allclose(noised, unchanged))
    self.assertLessEqual(np.max(np.abs(noised - unchanged)), 0.3)

  def test_noisy_highpass_rlx_hard_variant_is_registered(self):
    name = "SilverBadgerJoystickFlatTerrainRLXHardNoisyHighpassObservation"
    self.assertIn(name, registry.ALL_ENVS)
    config = registry.get_default_config(name)

    self.assertEqual(config.noise_config.scales.highpass_filter_state, 0.01)
    self.assertEqual(
        config.noise_config.scales.highpass_difference_inputs, 0.01
    )

  def test_5_percent_noisy_highpass_rlx_hard_variant_is_registered(self):
    name = (
        "SilverBadgerJoystickFlatTerrainRLXHard"
        "NoisyHighpassObservation5Percent"
    )
    self.assertIn(name, registry.ALL_ENVS)
    config = registry.get_default_config(name)

    self.assertEqual(config.noise_config.scales.highpass_filter_state, 0.05)
    self.assertEqual(
        config.noise_config.scales.highpass_difference_inputs, 0.05
    )

  def test_rough_terrain_uses_height_field(self):
    config = joystick.default_config()
    config.impl = "jax"
    env = joystick.Joystick(task="rough_terrain", config=config)

    floor_id = env.mj_model.geom("floor").id

    self.assertEqual(
        env.mj_model.geom_type[floor_id], mujoco.mjtGeom.mjGEOM_HFIELD
    )

  def test_robustness_task_variants_have_expected_defaults(self):
    bases = (
        "SilverBadgerJoystickFlatTerrain",
        "SilverBadgerJoystickFlatTerrain25",
        "SilverBadgerJoystickFlatTerrain25NoLinearVelocity",
        "SilverBadgerJoystickFlatTerrainNoLinearVelocity",
        "SilverBadgerJoystickRoughTerrain",
        "SilverBadgerJoystickRoughTerrainNoLinearVelocity",
    )
    variants = {
        "Pushes": (True, False),
        "DomainRandomization": (False, True),
        "PushesAndDomainRandomization": (True, True),
    }

    for base in bases:
      for suffix, (pushes, randomization) in variants.items():
        name = f"{base}{suffix}"
        self.assertIn(name, registry.ALL_ENVS)
        config = registry.get_default_config(name)
        self.assertEqual(config.pert_config.enable, pushes)
        self.assertEqual(config.domain_randomization, randomization)


if __name__ == "__main__":
  absltest.main()
