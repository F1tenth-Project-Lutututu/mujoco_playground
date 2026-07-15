# Copyright 2025 DeepMind Technologies Limited
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
"""Train a PPO agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
from typing import Optional
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training import logger as brax_logger
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import numpy as np
try:
  import tensorboardX
except ImportError:
  tensorboardX = None

try:
  import wandb
except ImportError:
  wandb = None


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_WANDB_EXPERIMENT_NAME = flags.DEFINE_string(
    "wandb_experiment_name",
    None,
    "W&B group name. The run name is '<group>-seed<seed>'. Defaults to the "
    "environment name.",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_RENDER_VIDEOS = flags.DEFINE_boolean(
    "render_videos",
    False,
    "Render rollout videos after training. Enable only with a working EGL, "
    "OSMesa, or GLFW backend.",
)
_CAMERA = flags.DEFINE_string(
    "camera",
    "track",
    "Camera used for rollout videos.",
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_MEAN_ACTION_RATE_COST = flags.DEFINE_float(
    "mean_action_rate_cost",
    0.0,
    "Coefficient for an auxiliary squared rate loss on deterministic policy "
    "means. A positive value disables the environment action-rate reward.",
)
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.3, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)
_WARP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "warp_kernel_cache_dir", None,
    "Directory for caching compiled Warp kernels.",
)
_LOGDIR = flags.DEFINE_string(
    "logdir", None, "Directory for logging."
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


_LOSS_METRICS = {
    "total_loss": "losses/total",
    "policy_loss": "losses/policy",
    "v_loss": "losses/value",
    "entropy_loss": "losses/entropy_regularization",
    "mean_action_rate_loss": "losses/mean_action_rate",
}
_STABILITY_METRICS = {
    "kl_mean": "stability/kl_divergence",
    "mean_action_rate": "stability/mean_action_rate",
    "policy_dist_mean_std": "stability/action_std_mean",
    "policy_dist_max_std": "stability/action_std_max",
    "policy_dist_min_std": "stability/action_std_min",
    "policy_dist_mean_loc": "stability/action_mean_mean",
    "policy_dist_max_loc": "stability/action_mean_max",
    "policy_dist_min_loc": "stability/action_mean_min",
}
_BRAX_COMPUTE_PPO_LOSS = ppo_losses.compute_ppo_loss


def _mean_action_rate(
    actions: jax.Array, episode_done: Optional[jax.Array] = None
) -> jax.Array:
  """Mean squared change of consecutive actions, excluding reset boundaries."""
  action_delta = actions[:, 1:] - actions[:, :-1]
  squared_rate = jp.sum(jp.square(action_delta), axis=-1)
  if episode_done is None:
    return jp.mean(squared_rate)

  # episode_done[:, t] means that action t ended an episode. The difference
  # from action t to t+1 therefore crosses a reset and is not a control-rate
  # transition belonging to either episode.
  valid_transition = 1.0 - episode_done[:, :-1].astype(squared_rate.dtype)
  return jp.sum(squared_rate * valid_transition) / jp.maximum(
      jp.sum(valid_transition), 1.0
  )


def _compute_ppo_loss_with_mean_action_rate(
    params,
    normalizer_params,
    data,
    rng,
    *,
    ppo_network,
    mean_action_rate_cost: float,
    **kwargs,
):
  """Adds policy-mean smoothness diagnostics and an optional auxiliary loss."""
  total_loss, metrics = _BRAX_COMPUTE_PPO_LOSS(
      params,
      normalizer_params,
      data,
      rng,
      ppo_network=ppo_network,
      **kwargs,
  )

  action_distribution = ppo_network.parametric_action_distribution
  behavior_distribution_params = data.extras["policy_extras"][
      "distribution_params"
  ]
  behavior_mean_actions = action_distribution.mode(
      behavior_distribution_params
  )
  episode_done = data.extras["state_extras"]["episode_done"]
  behavior_mean_action_rate = _mean_action_rate(
      behavior_mean_actions, episode_done
  )

  optimized_mean_action_rate = behavior_mean_action_rate
  if mean_action_rate_cost > 0.0:
    # Re-evaluate the current policy so gradients from this term update the
    # actor. Stored behavior distribution parameters are gradient-free.
    current_distribution_params = ppo_network.policy_network.apply(
        normalizer_params, params.policy, data.observation
    )
    current_mean_actions = action_distribution.mode(
        current_distribution_params
    )
    optimized_mean_action_rate = _mean_action_rate(
        current_mean_actions, episode_done
    )

  mean_action_rate_loss = mean_action_rate_cost * optimized_mean_action_rate
  total_loss = total_loss + mean_action_rate_loss
  metrics = {
      **metrics,
      "total_loss": total_loss,
      "mean_action_rate": behavior_mean_action_rate,
      "mean_action_rate_loss": mean_action_rate_loss,
  }
  return total_loss, metrics


class _EpisodeMetricsLoggerWithStd(brax_logger.EpisodeMetricsLogger):
  """Brax episode logger that reports both means and standard deviations."""

  def log_metrics(self, pad=35):
    self._log_count += 1
    now = time.time()
    steps_per_second = (self._num_steps - self._last_log_steps) / (
        now - self._last_log_time + 1e-8
    )
    self._last_log_time = now
    log_string = (
        f"\n{'Steps':>{pad}} Env: {self._num_steps} Log: {self._log_count}\n"
    )
    aggregated_metrics = {"sps": steps_per_second}
    log_string += f"{'Steps per second:':>{pad}} {steps_per_second:.0f}\n"

    for metric_name, values in self._ep_metrics_buffer.items():
      aggregated_metrics[metric_name] = np.mean(values)
      aggregated_metrics[f"{metric_name}_std"] = np.std(values)
      log_string += (
          f"{f'Episode {metric_name}:':>{pad}}"
          f" {aggregated_metrics[metric_name]:.4f} +-"
          f" {aggregated_metrics[f'{metric_name}_std']:.4f}\n"
      )

    for metric_name, values in self._train_metrics_buffer.items():
      aggregated_metrics[metric_name] = np.mean(values)
      log_string += (
          f"{f'Train {metric_name}:':>{pad}}"
          f" {aggregated_metrics[metric_name]:.6f}\n"
      )

    logging.info(log_string)
    if self._progress_fn is not None:
      self._progress_fn(
          int(self._num_steps),
          {
              f"episode/{name}": value
              for name, value in aggregated_metrics.items()
          },
      )


def _wandb_metric_name(name: str) -> str:
  """Places a Brax metric into a focused W&B section."""
  if name.startswith("training/"):
    metric = name.removeprefix("training/")
    if metric in _LOSS_METRICS:
      return _LOSS_METRICS[metric]
    if metric in _STABILITY_METRICS:
      return _STABILITY_METRICS[metric]
    if metric == "learning_rate":
      return "optimization/learning_rate"
    if metric in ("sps", "walltime"):
      return f"performance/train_{metric}"
    return f"optimization/{metric}"

  if name.startswith("episode/"):
    metric = name.removeprefix("episode/")
    # EpisodeMetricsLogger also reports PPO diagnostics under episode/.
    if metric in _LOSS_METRICS:
      return _LOSS_METRICS[metric]
    if metric in _STABILITY_METRICS:
      return _STABILITY_METRICS[metric]
    if metric == "learning_rate":
      return "optimization/learning_rate"
    if metric == "sum_reward":
      return "train_reward_means/total"
    if metric == "sum_reward_std":
      return "train_reward_stds/total"
    if metric == "reward_without_action_rate":
      return "train_reward_means/total_without_action_rate"
    if metric == "reward_without_action_rate_std":
      return "train_reward_stds/total_without_action_rate"
    if metric.startswith("torque_spectrum/"):
      return f"torque_spectrum/train/{metric.removeprefix('torque_spectrum/')}"
    if metric.startswith("reward/"):
      reward_name = metric.removeprefix("reward/")
      if reward_name.endswith("_std"):
        return f"train_reward_stds/{reward_name.removesuffix('_std')}"
      return f"train_reward_means/{reward_name}"
    if metric == "sps":
      return "performance/rollout_sps"
    return f"rollouts/train_{metric}"

  if name.startswith("eval/episode_torque_spectrum/"):
    metric = name.removeprefix("eval/episode_torque_spectrum/")
    return f"torque_spectrum/eval/{metric}"

  if name.startswith("eval/episode_reward"):
    metric = name.removeprefix("eval/episode_reward")
    if not metric:
      return "eval_reward_means/total"
    if metric == "_std":
      return "eval_reward_stds/total"
    if metric == "_without_action_rate":
      return "eval_reward_means/total_without_action_rate"
    if metric == "_without_action_rate_std":
      return "eval_reward_stds/total_without_action_rate"
    reward_name = metric.removeprefix("/")
    if reward_name.endswith("_std"):
      return f"eval_reward_stds/{reward_name.removesuffix('_std')}"
    return f"eval_reward_means/{reward_name}"

  if name.startswith("eval/"):
    metric = name.removeprefix("eval/")
    performance_names = {
        "sps": "eval_sps",
        "walltime": "eval_walltime",
        "epoch_eval_time": "eval_epoch_time",
    }
    if metric in performance_names:
      return f"performance/{performance_names[metric]}"
    if metric.startswith("episode_"):
      return f"rollouts/eval_{metric.removeprefix('episode_')}"
    return f"rollouts/eval_{metric}"

  return f"misc/{name}"


def _wandb_metrics(metrics, entropy_cost: float):
  """Renames Brax metrics and adds diagnostics derivable without approximation."""
  renamed = {_wandb_metric_name(name): value for name, value in metrics.items()}

  entropy_loss = metrics.get("training/entropy_loss")
  if entropy_loss is None:
    entropy_loss = metrics.get("episode/entropy_loss")
  if entropy_loss is not None and entropy_cost:
    # Brax defines entropy_loss = -entropy_cost * policy_entropy.
    renamed["stability/policy_entropy"] = -entropy_loss / entropy_cost

  return renamed


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  if _WARP_KERNEL_CACHE_DIR.value is not None:
    import warp as wp  # pylint: disable=g-import-not-at-top
    wp.config.kernel_cache_dir = _WARP_KERNEL_CACHE_DIR.value

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)

  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  env_cfg_overrides = {"impl": _IMPL.value}
  if _VISION.value:
    env_cfg_overrides["vision"] = True
    env_cfg_overrides["vision_config.nworld"] = ppo_params.num_envs
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))
  if _MEAN_ACTION_RATE_COST.value < 0.0:
    raise ValueError("--mean_action_rate_cost must be non-negative.")
  if _MEAN_ACTION_RATE_COST.value > 0.0:
    try:
      env_cfg.reward_config.scales.action_rate = 0.0
    except (AttributeError, KeyError) as error:
      raise ValueError(
          f"{_ENV_NAME.value} does not define an action_rate reward term."
      ) from error
    # Apply this after user overrides so the sampled-action penalty cannot be
    # accidentally active together with the auxiliary policy-mean loss.
    env_cfg_overrides["reward_config.scales.action_rate"] = 0.0

  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  if env_cfg_overrides:
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. "
          "Install via: pip install wandb"
      )
    wandb_group = _WANDB_EXPERIMENT_NAME.value or _ENV_NAME.value
    wandb_run_name = f"{wandb_group}-seed{_SEED.value}"
    reward_scales = env_cfg.get("reward_config", {}).get("scales", {})
    environment_action_rate_scale = reward_scales.get("action_rate")
    wandb.init(
        project="mjxrl",
        group=wandb_group,
        name=wandb_run_name,
        config={
            "environment": env_cfg.to_dict(),
            "ppo": ppo_params.to_dict(),
            "env_name": _ENV_NAME.value,
            "impl": _IMPL.value,
            "seed": _SEED.value,
            "domain_randomization": _DOMAIN_RANDOMIZATION.value,
            "mean_action_rate_cost": _MEAN_ACTION_RATE_COST.value,
            "environment_action_rate_disabled": (
                environment_action_rate_scale == 0.0
            ),
            "environment_action_rate_scale": environment_action_rate_scale,
            "wandb_experiment_name": wandb_group,
        },
    )
    wandb.define_metric("environment_steps")
    for section in (
        "train_reward_means",
        "train_reward_stds",
        "eval_reward_means",
        "eval_reward_stds",
        "losses",
        "stability",
        "optimization",
        "torque_spectrum",
        "rollouts",
        "performance",
        "misc",
    ):
      wandb.define_metric(
          f"{section}/*", step_metric="environment_steps"
      )

  # Initialize TensorBoard if required
  writer = None
  if _USE_TB.value and not _PLAY_ONLY.value and tensorboardX is not None:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  num_eval_envs = ppo_params.get("num_eval_envs", 128)

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  # Brax only reports means for completed training episodes by default. Use a
  # compatible logger that also exposes the corresponding standard deviations.
  brax_logger.EpisodeMetricsLogger = _EpisodeMetricsLoggerWithStd
  ppo_losses.compute_ppo_loss = functools.partial(
      _compute_ppo_loss_with_mean_action_rate,
      mean_action_rate_cost=_MEAN_ACTION_RATE_COST.value,
  )

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      vision=_VISION.value,
  )

  times = [time.monotonic()]

  # Progress function for logging
  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log({
          "environment_steps": num_steps,
          **_wandb_metrics(metrics, ppo_params.entropy_cost),
      })

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value and writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  eval_env_overrides = dict(env_cfg_overrides)
  if _VISION.value:
    eval_env_overrides["vision_config.nworld"] = num_eval_envs
  eval_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=eval_env_overrides,
  )

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(
          _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
      )
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      # rscope_handle.dump_rollout(params) # Disabled to prevent rendering slice crash

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  if not _RENDER_VIDEOS.value:
    print("Skipping post-training inference and video rendering.")
    if writer is not None:
      writer.close()
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.finish()
    return

  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  infer_env_overrides = dict(env_cfg_overrides)
  if _VISION.value:
    infer_env_overrides["vision_config.nworld"] = _NUM_VIDEOS.value
  infer_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=infer_env_overrides,
  )

  # Run evaluation rollouts matching how training handles batched environments.
  wrapped_infer_env = wrapper.wrap_for_brax_training(
      infer_env,
      episode_length=ppo_params.episode_length,
      action_repeat=ppo_params.get("action_repeat", 1),
  )

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(wrapped_infer_env.reset)(rng)

  empty_data = reset_states.data.__class__(
      **{k: None for k in reset_states.data.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = reset_states.__class__(
      **{k: None for k in reset_states.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = empty_traj.replace(data=empty_data)

  def step(carry, _):
    state, rng = carry
    rng, act_key = jax.random.split(rng)
    act_keys = jax.random.split(act_key, _NUM_VIDEOS.value)
    act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
    state = wrapped_infer_env.step(state, act)
    traj_data = empty_traj.tree_replace({
        "data.qpos": state.data.qpos,
        "data.qvel": state.data.qvel,
        "data.time": state.data.time,
        "data.ctrl": state.data.ctrl,
        "data.mocap_pos": state.data.mocap_pos,
        "data.mocap_quat": state.data.mocap_quat,
        "data.xfrc_applied": state.data.xfrc_applied,
    })
    return (state, rng), traj_data

  @jax.jit
  def do_rollout(state, rng):
    _, traj = jax.lax.scan(
        step, (state, rng), None, length=ppo_params.episode_length
    )
    return traj

  traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(_SEED.value + 1))
  # traj_stacked has shape (time, nworld, ...), swap to (nworld, time, ...).
  traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(ppo_params.episode_length)
    ]

  # Render and save the rollout.
  render_every = 2
  fps = 1.0 / infer_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  camera = _CAMERA.value
  if camera is None:
    track_camera_id = mujoco.mj_name2id(
        infer_env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track"
    )
    if track_camera_id != -1:
      camera = "track"
  print(f"Camera for rendering: {camera or 'default'}")
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = infer_env.render(
        traj,
        camera=camera,
        height=480,
        width=640,
        scene_option=scene_option,
    )
    media.write_video(logdir / f"rollout{i}.mp4", frames, fps=fps)
    print(f"Rollout video saved as '{logdir}/rollout{i}.mp4'.")

  if writer is not None:
    writer.close()
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.finish()


def run():
  """Entry point for uv/pip script."""
  app.run(main)


if __name__ == "__main__":
  run()
