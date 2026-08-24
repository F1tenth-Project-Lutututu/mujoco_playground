"""Tests for the fixed-parameter Fourier command process."""

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest
from ml_collections import config_dict

from mujoco_playground._src.locomotion import band_limited_command


def _config(**overrides):
  config = config_dict.create(
      num_harmonics=3,
      frequency_min_hz=[0.1, 0.1, 0.1],
      frequency_max_hz=[2.0, 2.0, 1.5],
      max_dc_fraction=0.25,
      max_amplitude_fraction=0.75,
  )
  for name, value in overrides.items():
    config[name] = value
  return config


class BandLimitedCommandTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.lower = jp.array([-1.5, -0.8, -1.2])
    self.upper = jp.array([1.5, 0.8, 1.2])

  def test_fixed_seed_is_deterministic_and_jittable(self):
    config = _config()
    first = band_limited_command.sample_parameters(
        jax.random.PRNGKey(7), self.lower, self.upper, config
    )
    second = band_limited_command.sample_parameters(
        jax.random.PRNGKey(7), self.lower, self.upper, config
    )
    for first_field, second_field in zip(first, second):
      np.testing.assert_array_equal(first_field, second_field)
    command = jax.jit(band_limited_command.command_at)(first, jp.array(0.37))
    np.testing.assert_allclose(
        command, band_limited_command.command_at(second, jp.array(0.37)), atol=1e-6
    )

  def test_different_seeds_generate_different_trajectories(self):
    config = _config()
    first = band_limited_command.sample_parameters(
        jax.random.PRNGKey(1), self.lower, self.upper, config
    )
    second = band_limited_command.sample_parameters(
        jax.random.PRNGKey(2), self.lower, self.upper, config
    )
    times = jp.linspace(0.0, 1.0, 20)
    first_trajectory = jax.vmap(lambda t: band_limited_command.command_at(first, t))(times)
    second_trajectory = jax.vmap(lambda t: band_limited_command.command_at(second, t))(times)
    self.assertFalse(np.allclose(first_trajectory, second_trajectory))

  def test_sampled_frequencies_and_commands_respect_limits(self):
    config = _config()
    parameters = band_limited_command.sample_parameters(
        jax.random.PRNGKey(3), self.lower, self.upper, config
    )
    lower_frequency = np.asarray(config.frequency_min_hz)[:, None]
    upper_frequency = np.asarray(config.frequency_max_hz)[:, None]
    self.assertTrue(np.all(np.asarray(parameters.frequencies_hz) >= lower_frequency))
    self.assertTrue(np.all(np.asarray(parameters.frequencies_hz) <= upper_frequency))
    commands = jax.vmap(lambda t: band_limited_command.command_at(parameters, t))(
        jp.linspace(0.0, 10.0, 1000)
    )
    self.assertEqual(commands.shape, (1000, 3))
    self.assertTrue(np.all(np.asarray(commands) >= np.asarray(self.lower)))
    self.assertTrue(np.all(np.asarray(commands) <= np.asarray(self.upper)))

  def test_zero_amplitude_process_is_constant(self):
    config = _config(max_amplitude_fraction=0.0)
    parameters = band_limited_command.sample_parameters(
        jax.random.PRNGKey(4), self.lower, self.upper, config
    )
    np.testing.assert_array_equal(parameters.amplitudes, jp.zeros((3, 3)))
    self.assertEqual(band_limited_command.command_at(parameters, 0.3).shape, (3,))
    np.testing.assert_allclose(
        band_limited_command.command_at(parameters, 0.0),
        band_limited_command.command_at(parameters, 1.0),
    )

  def test_invalid_frequency_ranges_raise_useful_errors(self):
    with self.assertRaisesRegex(ValueError, "min_hz < max_hz"):
      band_limited_command.validate_config(
          _config(frequency_max_hz=[0.1, 2.0, 1.5])
      )
    with self.assertRaisesRegex(ValueError, "shape"):
      band_limited_command.validate_config(_config(frequency_min_hz=[0.1, 0.2]))

  def test_known_sinusoid_matches_analytical_value(self):
    parameters = band_limited_command.Parameters(
        frequencies_hz=jp.array([[1.0], [0.5], [0.25]]),
        amplitudes=jp.array([[2.0], [0.0], [0.0]]),
        phases=jp.zeros((3, 1)),
        offsets=jp.array([0.5, 0.0, 0.0]),
        lower=jp.array([-3.0, -3.0, -3.0]),
        upper=jp.array([3.0, 3.0, 3.0]),
    )
    np.testing.assert_allclose(
        band_limited_command.command_at(parameters, jp.array(0.25)),
        jp.array([2.5, 0.0, 0.0]),
        atol=1e-6,
    )

  def test_evolution_is_continuous_across_control_steps(self):
    parameters = band_limited_command.sample_parameters(
        jax.random.PRNGKey(5), self.lower, self.upper, _config()
    )
    dt = 0.002
    change = jp.abs(
        band_limited_command.command_at(parameters, dt)
        - band_limited_command.command_at(parameters, 0.0)
    )
    derivative_bound = 2.0 * jp.pi * jp.sum(
        parameters.amplitudes * parameters.frequencies_hz, axis=1
    )
    self.assertTrue(np.all(np.asarray(change) <= np.asarray(derivative_bound * dt + 1e-6)))


if __name__ == "__main__":
  absltest.main()
