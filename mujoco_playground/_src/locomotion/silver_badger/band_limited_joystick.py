"""SilverBadger joystick task with an episode-long band-limited command."""

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src.locomotion import band_limited_command
from mujoco_playground._src.locomotion.silver_badger import joystick


def default_config() -> config_dict.ConfigDict:
  config = joystick.default_config()
  config.band_limited_command_config = config_dict.create(
      num_harmonics=3,
      frequency_min_hz=[0.1, 0.1, 0.1],
      frequency_max_hz=[2.0, 2.0, 1.5],
      max_dc_fraction=0.25,
      max_amplitude_fraction=0.75,
  )
  return config


class BandLimitedJoystick(joystick.Joystick):
  """Uses one fixed Fourier command process per episode."""

  def __init__(self, *args, **kwargs):
    if len(args) < 2 and "config" not in kwargs:
      kwargs["config"] = default_config()
    super().__init__(*args, **kwargs)
    band_limited_command.validate_config(
        self._config.band_limited_command_config
    )
    self._band_lower = -self._cmd_a
    self._band_upper = self._cmd_a

  def _initialize_command(self, info):
    rng, command_rng = jax.random.split(info["rng"])
    parameters = band_limited_command.sample_parameters(
        command_rng,
        self._band_lower,
        self._band_upper,
        self._config.band_limited_command_config,
    )
    info["rng"] = rng
    info["band_limited_command"] = parameters
    info["band_limited_command_time"] = jp.zeros(())
    info["command"] = band_limited_command.command_at(parameters, jp.zeros(()))

  def _advance_command(self, info):
    time = info["band_limited_command_time"] + self.dt
    info["band_limited_command_time"] = time
    info["command"] = band_limited_command.command_at(
        info["band_limited_command"], time
    )

  def sample_command(self, rng, x_k):
    del rng
    return x_k

  def _resample_command(self, rng, command):
    del rng
    return command
