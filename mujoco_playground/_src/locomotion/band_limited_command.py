"""JAX-compatible finite-Fourier joystick command processes."""

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np


class Parameters(NamedTuple):
    """Fixed parameters of one three-component command trajectory."""

    frequencies_hz: jax.Array
    amplitudes: jax.Array
    phases: jax.Array
    offsets: jax.Array
    lower: jax.Array
    upper: jax.Array


def validate_config(config) -> None:
    """Validates the static configuration before an environment is constructed."""
    harmonics = config.num_harmonics
    if isinstance(harmonics, bool) or not isinstance(harmonics, int) or harmonics < 1:
        raise ValueError("band_limited_command_config.num_harmonics must be positive.")
    minimum = np.asarray(config.frequency_min_hz, dtype=float)
    maximum = np.asarray(config.frequency_max_hz, dtype=float)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("Band-limited frequency limits must each have shape (3,).")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("Band-limited frequency limits must be finite.")
    if np.any(minimum < 0.0) or np.any(maximum <= minimum):
        raise ValueError(
            "Every band-limited frequency range must satisfy 0 <= min_hz < max_hz."
        )
    for name in ("max_dc_fraction", "max_amplitude_fraction"):
        value = float(getattr(config, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"band_limited_command_config.{name} must lie in [0, 1].")


def sample_parameters(
    rng: jax.Array, lower: jax.Array, upper: jax.Array, config
) -> Parameters:
    """Samples bounded Fourier coefficients using only fixed-shape JAX arrays."""
    lower, upper = jp.asarray(lower), jp.asarray(upper)
    frequency_min = jp.asarray(config.frequency_min_hz)
    frequency_max = jp.asarray(config.frequency_max_hz)
    count = int(config.num_harmonics)
    key_frequency, key_phase, key_offset, key_weight, key_total = jax.random.split(
        rng, 5
    )
    frequencies = jax.random.uniform(
        key_frequency,
        (3, count),
        minval=frequency_min[:, None],
        maxval=frequency_max[:, None],
    )
    phases = jax.random.uniform(key_phase, (3, count), minval=0.0, maxval=2.0 * jp.pi)
    dc = jax.random.uniform(
        key_offset,
        (3,),
        minval=-float(config.max_dc_fraction),
        maxval=float(config.max_dc_fraction),
    )
    center = (lower + upper) / 2.0
    half_span = (upper - lower) / 2.0
    weights = jax.random.uniform(key_weight, (3, count))
    weights = weights / jp.maximum(jp.sum(weights, axis=1, keepdims=True), 1e-12)
    total = jax.random.uniform(
        key_total, (3,), minval=0.0, maxval=float(config.max_amplitude_fraction)
    )
    budget = half_span * (1.0 - jp.abs(dc)) * total
    amplitudes = weights * budget[:, None]
    return Parameters(
        frequencies, amplitudes, phases, center + half_span * dc, lower, upper
    )


def command_at(parameters: Parameters, time_seconds: jax.Array) -> jax.Array:
    """Evaluates the continuous command and clips only against roundoff."""
    phase = 2.0 * jp.pi * parameters.frequencies_hz * time_seconds + parameters.phases
    command = parameters.offsets + jp.sum(parameters.amplitudes * jp.sin(phase), axis=1)
    return jp.clip(command, parameters.lower, parameters.upper)
