# Learning RL Agents

In this directory, we demonstrate learning RL agents from MuJoCo Playground environments using [Brax](https://github.com/google/brax) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). We provide two entrypoints from the command line: `python train_jax_ppo.py` and `python train_rsl_rl.py`.

For more detailed tutorials on using MuJoCo Playground for RL, see:

1. Intro. to the Playground with DM Control Suite [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb)
2. Locomotion Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb)
3. Manipulation Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb)
4. Training CartPole from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb)
5. Robotic Manipulation from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb)

## Training with brax PPO

To train with brax PPO, you can use the `train_jax_ppo.py` script. This script uses the brax PPO algorithm to train an agent on a given environment.

```bash
python train_jax_ppo.py --env_name=CartpoleBalance
```

To train a vision-based policy using pixel observations:
```bash
python train_jax_ppo.py --env_name=CartpoleBalance --vision
```

Use `python train_jax_ppo.py --help` to see possible options and usage. Logs and checkpoints are saved in `logs` directory.

For locomotion environments with an `action_rate` reward, an auxiliary loss can
smooth the deterministic policy mean without penalizing sampled exploration:

```bash
python train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --mean_action_rate_cost=0.01
```

When the coefficient is positive, the environment's sampled-action rate reward
is disabled automatically. The unweighted deterministic mean-action rate is
logged for both baseline and auxiliary-loss runs so they can be compared.

W&B runs can be grouped under a user-defined experiment name. Each run name is
prefixed with the date in `YYMMDD` format:

```bash
python train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --use_wandb \
  --wandb_experiment_name=go1-action-smoothing \
  --seed=1
```

This creates project
`spectral_playground_highpass_Go1JoystickFlatTerrain`, group
`go1-action-smoothing`, and a run such as
`260716-go1-action-smoothing-seed1`. Models are saved under the environment
and run names in
`logs/Go1JoystickFlatTerrain/260716-go1-action-smoothing-seed1/checkpoints`,
making the local checkpoints easy to match to their W&B run.

Post-training rollout rendering is disabled by default so training works on
headless cluster nodes without an OpenGL backend. Enable it explicitly on a
machine with working EGL, OSMesa, or GLFW support:

```bash
python train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --render_videos
```

Without `--render_videos`, training, checkpointing, and metric logging still
complete normally.

Each new JAX PPO run saves a versioned `run_config.json` next to its
checkpoints.  It contains the effective environment and PPO/network
configuration together with the environment name, implementation, vision and
randomization settings, seed, auxiliary-loss coefficient, and original
command.  Loading a checkpoint automatically restores these settings; flags
provided explicitly on the new command line take precedence.  For example:

```bash
python learning/train_jax_ppo.py \
  --play_only \
  --render_videos \
  --load_checkpoint_path=\
logs/Go1JoystickFlatTerrain/260716-go1-action-smoothing-seed1/checkpoints
```

Older checkpoint directories containing only `config.json` are also supported:
their environment configuration is restored automatically, while settings not
present in that legacy file continue to use defaults or explicit flags.

### Reproducible policy evaluation

Use the constant-command evaluator to compare policies with matched reset
seeds and scenarios. It records per-rollout CSV files and aggregate JSON/CSV
reports, with optional videos and compressed raw signals (`signals.npz`):

```bash
python learning/evaluate_policy.py \
  --checkpoint=eagle/260716-go1-newhf1em6-f5o2m30-seed0/checkpoints \
  --env_name=Go1JoystickFlatTerrain \
  --num_rollouts=8
```

Evaluation requires a CUDA-backed JAX device by default and fails rather than
silently falling back to CPU. Install the `cuda` extra and verify that
`jax.devices()` lists a GPU. Video rendering and full `signals.npz` trajectory
archives are disabled by default; enable them explicitly with `--render_video`
or `--save_signals` when needed.

`--env_name` is only needed for older checkpoints that predate
`run_config.json`. The default suite tests standing, forward and backward
motion, lateral motion, turning, and a combined command. Supply a custom suite
as JSON when needed:

```bash
python learning/evaluate_policy.py \
  --checkpoint=<checkpoints-or-specific-checkpoint> \
  --commands='{"slow": [0.3, 0, 0], "fast_turn": [0.8, 0, 1.0]}'
```

For fair comparisons, use the same command JSON, rollout count, episode length,
and seed for every policy. The primary task score is
`eval_reward_means/total_without_regularization`; unlike total reward, it does
not favor either an action-rate or high-pass regularizer. Physical tracking
RMSE, fall rate, action differences, torque differences, mechanical energy,
and order-independent FFT energy above fixed cutoffs provide complementary
measurements. Add `--use_wandb` to upload the summary and output directory as a
W&B evaluation artifact.

To evaluate the latest checkpoint of every model directory under `./eagle` on
the same random tasks, configure the constants at the top of the batch script
and run:

```bash
python learning/evaluate_all_models.py
```

Successful results are cached. On later runs, the script skips models whose
checkpoint, evaluation settings, evaluator/environment code, and locked
dependencies are unchanged. Set `REUSE_UNCHANGED_RESULTS = False` to force a
complete reevaluation.

The batch evaluator runs all models in one Python process, avoiding repeated
Python startup and allowing JAX compilation caches to remain available between
models. Its progress bar reports completed models, elapsed time, processing
rate, and estimated remaining time; cached models are excluded from the ETA.
For fair tasks and compilation reuse, the batch uses one common default
environment configuration. Checkpoint weights are dynamic inputs to the jitted
rollout, so compatible models reuse the first compiled executable. Set
`USE_SAVED_ENVIRONMENT_CONFIG = True` only when reproducing each model's
training environment is more important than common-task evaluation and maximal
compilation reuse.

The Go1 joystick environment also provides an optional configurable-order
high-pass torque penalty. Its scale is zero by default; enable it and choose a
cutoff in Hz and an order from 1 through 8 with environment overrides:

```bash
python train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --playground_config_overrides='{"reward_config.scales.torque_high_freq": -1e-5, "reward_config.torque_highpass_cutoff_hz": 5.0, "reward_config.torque_highpass_order": 2, "reward_config.torque_highpass_difference_order": 0}'
```

The filter samples actuator torque at the 50 Hz control rate, so the cutoff
must be greater than 0 and less than the 25 Hz Nyquist frequency. Order 1 is
the default. Orders 1 through 8 use a proper digital Butterworth high-pass
filter represented as numerically stable second-order sections, so the
configured cutoff remains the -3 dB point for every order. The resulting
component is logged as `reward/torque_high_freq`. Enabling this penalty with a
negative scale automatically sets the sampled-action `action_rate` scale to
zero, so the two smoothing penalties are not applied together.

The `torque_highpass_difference_order` parameter, denoted by `m`, controls how
strongly the penalty grows with frequency after the Butterworth high-pass
filter. It accepts any finite number from 0 through 8. For integer `m`, the
repeated difference is normalized at the configured cutoff, giving the squared
frequency weighting

`|H_HP(f)|^2 [sin(pi f / f_s) / sin(pi f_c / f_s)]^(2m)`.

Consequently, every value of `m` has weight 0.5 at the Butterworth cutoff
`f_c`. Increasing `m` changes the steepness without moving that shared
absolute reference point, and weights above the cutoff can grow beyond 1.

For fractional `m = k + alpha`, the environment linearly interpolates the
penalty energies at the adjacent integer orders:

`W_m(f) = (1 - alpha) W_k(f) + alpha W_(k+1)(f)`.

This remains causal and preserves the same cutoff weight. `m=0` is the
high-pass energy penalty, while `m=1` is its first-difference penalty.

For example, use `m=1.5` with:

```bash
python train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --playground_config_overrides='{"reward_config.scales.torque_high_freq": -1e-5, "reward_config.torque_highpass_cutoff_hz": 5.0, "reward_config.torque_highpass_order": 2, "reward_config.torque_highpass_difference_order": 1.5}'
```

Different values of `m` have different numerical scales, so tune
`reward_config.scales.torque_high_freq` independently for each value.

Training and evaluation also log `total_without_regularization`, which excludes
both the `action_rate` and `torque_high_freq` reward terms. This provides a
common task-reward metric for comparing either smoothing method. The existing
`total_without_action_rate` metric remains available separately.

Unscaled high-pass torque energies are always logged at 1, 2, 5, 10, 15, and
20 Hz, together with total torque energy, under the `torque_spectrum` W&B
section. These online diagnostics always use an order-1 measurement filter,
independently of the regularization filter order, so runs remain comparable.

Evaluation additionally logs `fft_above_<cutoff>hz_energy_per_step` metrics
computed from the complete torque rollout with a real FFT. These report the
mean torque energy above each cutoff using a common, order-independent
measurement. Torque-energy standard deviations are intentionally omitted;
reward and other episode metrics continue to include their standard deviations.

Action, motor-target, and torque smoothness reports also include Mean Squared
Second Derivative (MSSD), implemented as the mean squared discrete second
difference per degree of freedom, and Mean Savitzky-Golay Filter Deviation
(MSGFD), implemented as the mean absolute deviation from a Savitzky-Golay
smoothed signal. MSGFD uses an 11-sample, order-3 filter by default; customize
it with `--savgol_window_length` and `--savgol_polyorder`. Both settings are
stored in evaluation metadata. The visual comparison includes action MSSD and
MSGFD panels by default.

The evaluator also reports physical totals and tracking diagnostics used by the
comparison plots: absolute mechanical energy in joules, integrated absolute
actuator torque in N·m·s, RMS body-orientation error in degrees, RMS roll/pitch
angular velocity in rad/s, and mean absolute swing-peak foot-height error in
millimetres. Foot height is measured for completed swings at touchdown against
the environment's configured `max_foot_height`; that target is saved in report
metadata.

Configure `METHODS`, `METRICS`, and the other uppercase constants at the top of
`compare_policy_evaluations.py`, then compare the evaluations with:

```bash
python learning/compare_policy_evaluations.py
```

The figure contains one grouped-bar panel for each primary comparison metric,
with matched command scenarios and rollout standard-deviation error bars. The
plotted data is also saved to `evaluation_comparison.csv`. Set `SHOW = True` to
open an interactive window in addition to saving the image.

For random-task evaluations, the script additionally writes
`evaluation_comparison_paired.png` and `.csv`. This report subtracts the
configured `PAIRED_REFERENCE` result from every other method on each matched
task. Its boxplots therefore show within-task method differences without
between-task difficulty inflating the variability. Individual translucent
points are task differences, boxes show their distribution, and black diamonds
show the paired mean with a 95% confidence interval. Positive deltas mean the
method has a larger metric value than the reference; whether that is desirable
depends on the metric (positive is better for reward, but negative is better
for errors and smoothness costs).

## Training with RSL-RL

To train with RSL-RL, you can use the `train_rsl_rl.py` script. This script uses the RSL-RL algorithm to train an agent on a given environment.

```bash
python train_rsl_rl.py --env_name=LeapCubeReorient
```

To render the behaviour from the resulting policy:
```bash
python learning/train_rsl_rl.py --env_name LeapCubeReorient --play_only --load_run_name <run_name>
```

where `run_name` is the name of the run you want to load (will be printed in the console when the training run is started).

Logs and checkpoints are saved in `logs` directory.
