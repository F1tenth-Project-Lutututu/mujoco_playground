"""Shared torque-rate and high-pass regularization for locomotion tasks."""

from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from scipy import signal as scipy_signal


def apply_torque_differences(
    signal: jax.Array,
    previous_inputs: jax.Array,
    reset: jax.Array,
    lower_order: int,
    upper_order: int,
    mix: float,
    scale_base: float,
) -> tuple[jax.Array, jax.Array]:
  """Returns interpolated adjacent-order energy and updated causal state."""
  differenced = signal
  normalized = [signal]
  next_inputs = []
  for difference in range(upper_order):
    previous_input = jp.where(
        reset, differenced, previous_inputs[difference]
    )
    next_inputs.append(differenced)
    differenced = differenced - previous_input
    normalized.append(differenced * scale_base ** (difference + 1))
  lower = jp.sum(jp.square(normalized[lower_order]))
  upper = jp.sum(jp.square(normalized[upper_order]))
  cost = (1.0 - mix) * lower + mix * upper
  return cost, jp.stack(next_inputs) if next_inputs else previous_inputs


def add_config(config: Any) -> None:
  """Adds the torque regularizer defaults to a reward config."""
  config.scales.torque_high_freq = 0.0
  config.scales.torque_rate = 0.0
  config.torque_highpass_cutoff_hz = 5.0
  config.torque_highpass_order = 1
  config.torque_highpass_difference_order = 0.0
  config.torque_highpass_frequency_normalization = "legacy"
  config.torque_highpass_signal = "torque"
  config.torque_highpass_normalize_by_capacity = True
  config.torque_highpass_observe_state = False
  config.torque_rate_observe_state = False
  config.torque_highpass_adaptive_weight = False
  config.torque_highpass_adaptive_min_weight = 0.1
  config.torque_highpass_adaptive_max_weight = 1.0
  config.torque_highpass_adaptive_sigma = 0.25
  config.torque_spectrum_cutoffs_hz = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0)


class TorquePenalty:
  """Maintains the causal state used by torque regularization."""

  def __init__(self, reward_config: Any, model: Any, dt: float):
    self.config = reward_config
    cutoff = reward_config.torque_highpass_cutoff_hz
    nyquist = 0.5 / dt
    if not 0.0 < cutoff < nyquist:
      raise ValueError(
          "reward_config.torque_highpass_cutoff_hz must be between 0 and "
          f"{nyquist} Hz, got {cutoff} Hz."
      )
    order = reward_config.torque_highpass_order
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
      raise ValueError("reward_config.torque_highpass_order must be an integer.")
    if not 1 <= order <= 8:
      raise ValueError("reward_config.torque_highpass_order must be in [1, 8].")
    difference_order = reward_config.torque_highpass_difference_order
    if (
        isinstance(difference_order, bool)
        or not isinstance(
            difference_order, (int, float, np.integer, np.floating)
        )
        or not np.isfinite(difference_order)
        or not 0 <= difference_order <= 8
    ):
      raise ValueError(
          "reward_config.torque_highpass_difference_order must be in [0, 8]."
      )
    if reward_config.torque_highpass_signal not in ("torque", "action"):
      raise ValueError(
          "reward_config.torque_highpass_signal must be 'torque' or 'action'."
      )
    frequency_normalization = (
        reward_config.torque_highpass_frequency_normalization
    )
    if frequency_normalization not in ("legacy", "white_spectrum"):
      raise ValueError(
          "reward_config.torque_highpass_frequency_normalization must be "
          "'legacy' or 'white_spectrum'."
      )
    for name in ("torque_high_freq", "torque_rate"):
      if reward_config.scales[name] > 0:
        raise ValueError(f"reward_config.scales.{name} must be non-positive.")
      if reward_config.scales[name] < 0:
        reward_config.scales.action_rate = 0.0
    self.adaptive_enabled = reward_config.torque_highpass_adaptive_weight
    self.adaptive_min_weight = (
        reward_config.torque_highpass_adaptive_min_weight
    )
    self.adaptive_max_weight = (
        reward_config.torque_highpass_adaptive_max_weight
    )
    self.adaptive_sigma = reward_config.torque_highpass_adaptive_sigma
    if not isinstance(self.adaptive_enabled, (bool, np.bool_)):
      raise ValueError("torque_highpass_adaptive_weight must be a boolean.")
    adaptive_values = (
        self.adaptive_min_weight,
        self.adaptive_max_weight,
        self.adaptive_sigma,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        for value in adaptive_values
    ):
      raise ValueError("Adaptive high-pass parameters must be finite numbers.")
    if (
        self.adaptive_min_weight < 0.0
        or self.adaptive_max_weight < self.adaptive_min_weight
        or self.adaptive_sigma <= 0.0
    ):
      raise ValueError("Invalid adaptive high-pass weight bounds or sigma.")

    sos = scipy_signal.butter(
        order, cutoff, btype="highpass", fs=1.0 / dt, output="sos"
    ).astype(np.float32)
    self.sos = jp.asarray(sos)
    self.steady_state = jp.asarray(
        scipy_signal.sosfilt_zi(sos).astype(np.float32)
    )
    force_range = np.asarray(model.actuator_forcerange)
    if np.any(~np.isfinite(force_range)) or np.any(
        np.max(np.abs(force_range), axis=-1) <= 0
    ):
      actuator_joint_ids = np.asarray(model.actuator_trnid)[:, 0]
      force_range = np.asarray(model.jnt_actfrcrange)[actuator_joint_ids]
    self.capacity = jp.asarray(np.max(np.abs(force_range), axis=-1))
    if np.any(~np.isfinite(self.capacity)) or np.any(self.capacity <= 0):
      raise ValueError("Actuators need finite, positive force limits.")
    self.upper_difference_order = int(np.ceil(difference_order))
    self.lower_difference_order = int(np.floor(difference_order))
    self.difference_mix = difference_order - self.lower_difference_order
    self.difference_scale = 1.0 / (2.0 * np.sin(np.pi * cutoff * dt))
    self.frequency_normalizer = 1.0
    if frequency_normalization == "white_spectrum":
      frequencies, response = scipy_signal.sosfreqz(
          sos, worN=16_384, fs=1.0 / dt
      )
      gain = (
          2.0
          * np.sin(np.pi * frequencies * dt)
          * self.difference_scale
      )
      lower_weight = gain ** (2 * self.lower_difference_order)
      upper_weight = gain ** (2 * self.upper_difference_order)
      difference_weight = (
          (1.0 - self.difference_mix) * lower_weight
          + self.difference_mix * upper_weight
      )
      self.frequency_normalizer = float(
          np.mean(np.abs(response) ** 2 * difference_weight)
      )

  def _signal(self, torque: jax.Array, action: jax.Array) -> jax.Array:
    if self.config.torque_highpass_signal == "action":
      return action
    if self.config.torque_highpass_normalize_by_capacity:
      return torque / self.capacity
    return torque

  def initial_filter_state(self, signal: jax.Array) -> jax.Array:
    return self.steady_state[..., None] * signal[None, None, :]

  def reset(self, info: dict[str, Any], torque: jax.Array) -> None:
    action = jp.zeros_like(torque)
    signal = self._signal(torque, action)
    info["last_torque"] = torque
    info["torque_for_spectrum"] = torque
    info["torque_highpass_state"] = self.initial_filter_state(signal)
    info["torque_difference_inputs"] = jp.zeros(
        (self.upper_difference_order, torque.shape[0])
    )

  def compute(
      self,
      info: dict[str, Any],
      torque: jax.Array,
      action: jax.Array,
      reset: jax.Array = jp.asarray(False),
  ) -> tuple[jax.Array, jax.Array]:
    filtered = self._signal(torque, action)
    initial_state = self.initial_filter_state(filtered)
    previous_filter_state = jp.where(
        reset, initial_state, info["torque_highpass_state"]
    )
    next_states = []
    for section in range(self.sos.shape[0]):
      coefficients = self.sos[section]
      section_state = previous_filter_state[section]
      output = coefficients[0] * filtered + section_state[0]
      state_0 = (
          coefficients[1] * filtered
          - coefficients[4] * output
          + section_state[1]
      )
      state_1 = coefficients[2] * filtered - coefficients[5] * output
      filtered = output
      next_states.append(jp.stack((state_0, state_1)))
    info["torque_highpass_state"] = jp.stack(next_states)

    high_freq, info["torque_difference_inputs"] = apply_torque_differences(
        filtered,
        info["torque_difference_inputs"],
        reset,
        self.lower_difference_order,
        self.upper_difference_order,
        self.difference_mix,
        self.difference_scale,
    )
    high_freq /= self.frequency_normalizer
    last_torque = jp.where(reset, torque, info["last_torque"])
    torque_rate = jp.sum(jp.square(torque - last_torque))
    info["last_torque"] = torque
    info["torque_for_spectrum"] = torque
    return high_freq, torque_rate

  def observation(self, info: dict[str, Any], torque: jax.Array) -> jax.Array:
    values = []
    if self.config.torque_highpass_observe_state:
      values.extend((
          jp.ravel(info["torque_highpass_state"]),
          jp.ravel(info["torque_difference_inputs"]),
      ))
    if self.config.torque_rate_observe_state:
      values.append(torque)
    return jp.concatenate(values) if values else jp.zeros((0,))

  def apply_adaptive_weight(
      self, cost: jax.Array, disturbance: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Applies the Go1 disturbance-dependent regularization weight."""
    weight = jp.asarray(1.0)
    if self.adaptive_enabled:
      interpolation = jp.exp(-disturbance / self.adaptive_sigma)
      weight = self.adaptive_min_weight + (
          self.adaptive_max_weight - self.adaptive_min_weight
      ) * interpolation
    return cost * weight, weight
