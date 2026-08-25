"""Smoke tests for the band-limited joystick task variants."""

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest, parameterized

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


class BandLimitedJoystickTest(parameterized.TestCase):

  def test_silver_badger_robust_variant_configuration(self):
    env_name = "SilverBadgerBandLimitedPushesAndDomainRandomization"
    config = registry.get_default_config(
        env_name
    )
    self.assertTrue(config.pert_config.enable)
    self.assertTrue(config.domain_randomization)
    self.assertIn("band_limited_command_config", config)
    self.assertEqual(
        locomotion_params.brax_ppo_config(env_name).num_timesteps,
        200_000_000,
    )

  def test_spot_band_limited_variant_has_ppo_config(self):
    config = locomotion_params.brax_ppo_config("SpotJoystickBandLimited")
    self.assertEqual(config.num_timesteps, 100_000_000)

  @parameterized.parameters(
      "Go1JoystickBandLimited",
      "SilverBadgerJoystickBandLimited",
      "SpotJoystickBandLimited",
  )
  def test_registry_reset_step_and_torque_penalties(self, env_name):
    self.assertIn(env_name, registry.ALL_ENVS)
    config = registry.get_default_config(env_name)
    config.impl = "jax"
    env = registry.load(env_name, config=config)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    commands = [np.asarray(state.info["command"])]
    for _ in range(3):
      state = step(state, jp.zeros(env.action_size))
      commands.append(np.asarray(state.info["command"]))
      self.assertTrue(np.all(np.isfinite(np.asarray(state.obs["state"]))))
      self.assertTrue(np.isfinite(np.asarray(state.reward)))
      self.assertTrue(np.isfinite(np.asarray(state.metrics["reward/torque_rate"])))
      self.assertTrue(
          np.isfinite(np.asarray(state.metrics["reward/torque_high_freq"]))
      )
    commands = np.asarray(commands)
    self.assertEqual(commands.shape[1:], (3,))
    self.assertTrue(np.any(np.abs(np.diff(commands, axis=0)) > 0.0))
    self.assertIn("band_limited_command", state.info)
    parameters = state.info["band_limited_command"]
    derivative_bound = 2.0 * np.pi * np.sum(
        np.asarray(parameters.amplitudes)
        * np.asarray(parameters.frequencies_hz),
        axis=1,
    )
    self.assertTrue(
        np.all(np.abs(np.diff(commands, axis=0)) <= derivative_bound * env.dt + 1e-5)
    )


if __name__ == "__main__":
  absltest.main()
