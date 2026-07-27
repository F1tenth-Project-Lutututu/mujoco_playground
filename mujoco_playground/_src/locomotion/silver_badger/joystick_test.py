"""Tests for the SilverBadger joystick environment."""

from absl.testing import absltest
import jax
import jax.numpy as jp

from mujoco_playground._src.locomotion.silver_badger import joystick


class JoystickTest(absltest.TestCase):

  def test_reset_and_step_shapes(self):
    config = joystick.default_config()
    config.impl = "jax"
    env = joystick.Joystick(config=config)

    state = env.reset(jax.random.PRNGKey(0))
    next_state = env.step(state, jp.zeros(env.action_size))

    self.assertEqual(env.action_size, 13)
    self.assertEqual(state.obs["state"].shape, (51,))
    self.assertEqual(state.obs["privileged_state"].shape, (129,))
    self.assertEqual(next_state.obs["state"].shape, (51,))
    self.assertEqual(next_state.reward.shape, ())
    self.assertEqual(next_state.done.shape, ())

  def test_no_linear_velocity_actor_observation(self):
    config = joystick.no_linear_velocity_config()
    config.impl = "jax"
    env = joystick.Joystick(config=config)

    state = env.reset(jax.random.PRNGKey(0))

    self.assertFalse(config.policy_observes_linear_velocity)
    self.assertEqual(state.obs["state"].shape, (48,))
    # The critic still receives the ground-truth local linear velocity.
    self.assertEqual(state.obs["privileged_state"].shape, (126,))


if __name__ == "__main__":
  absltest.main()
