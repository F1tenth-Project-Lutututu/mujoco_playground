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

W&B runs can be grouped under a user-defined experiment name. Each run is named
from that group and its seed:

```bash
python train_jax_ppo.py \
  --env_name=Go1JoystickFlatTerrain \
  --use_wandb \
  --wandb_experiment_name=go1-action-smoothing \
  --seed=1
```

This creates group `go1-action-smoothing` and run
`go1-action-smoothing-seed1` in the `mjxrl` W&B project.

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
