"""Policy observations for action-difference regularizers."""

from typing import Any

import jax
from jax import numpy as jp


def observation(
    reward_config: Any,
    info: dict[str, Any],
    current_action: jax.Array | None = None,
    *,
    include_legacy: bool = True,
) -> jax.Array:
  """Returns the action memory exposed to the policy.

  Environments conventionally construct the observation produced by
  ``step(a_t)`` before advancing ``last_act``.  The legacy observation is kept
  unchanged for old action-rate/action-smoothness experiments.  Fixed variants
  explicitly use ``a_t`` (and ``a_{t-1}`` for a second-difference penalty), so
  the policy selecting ``a_{t+1}`` observes all reward-relevant action memory.
  """
  use_fixed_observation = reward_config.get(
      "action_rate_use_fixed_observation", False
  )
  if not use_fixed_observation:
    return info["last_act"] if include_legacy else jp.zeros((0,))

  previous_action = (
      info["last_act"] if current_action is None else current_action
  )
  if not reward_config.action_rate_use_second_difference:
    return previous_action
  previous_previous_action = info[
      "last_last_act" if current_action is None else "last_act"
  ]
  return jp.concatenate([previous_action, previous_previous_action])
