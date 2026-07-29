#!/bin/bash

#SBATCH --job-name=silver_badger_factors
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=0-05:59:59
#SBATCH --array=0-4
#SBATCH --gres=gpu:1

#set -euo pipefail

# Trains five seeds for one linear/angular velocity reward-weight pair.
#
# Usage:
#   sbatch slurm_silver_badger_reward_factors.sh \
#     <linear_weight> <angular_weight> [termination_weight] [environment] \
#     [timesteps] [wandb_project]
#
# Example:
#   sbatch slurm_silver_badger_reward_factors.sh \
#     9e0 3e0 -5e0 SilverBadgerJoystickRoughTerrain 400000000 \
#     spectral_playground_silver_badger_factor_search

if (( $# < 2 )); then
  echo "Usage: sbatch $0 <linear_weight> <angular_weight> [termination_weight] [environment] [timesteps] [wandb_project]" >&2
  exit 2
fi

LINEAR_WEIGHT=$1
ANGULAR_WEIGHT=$2
TERMINATION_WEIGHT=${3:--1.0}
#ENV_NAME=${4:-SilverBadgerJoystickFlatTerrain}
ENV_NAME=${4:-SilverBadgerJoystickRoughTerrain}
#ENV_NAME=${4:-Go1JoystickRoughTerrain}
NUM_TIMESTEPS=${5:-400000000}
WANDB_PROJECT=${6:-spectral_playground_silver_badger_rough_factor_search}
SEED=${SLURM_ARRAY_TASK_ID:-0}
NUM_EVALS=16

NUMBER_PATTERN='^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
if [[ ! "$LINEAR_WEIGHT" =~ $NUMBER_PATTERN ]]; then
  echo "Invalid linear velocity weight: $LINEAR_WEIGHT" >&2
  exit 2
fi
if [[ ! "$ANGULAR_WEIGHT" =~ $NUMBER_PATTERN ]]; then
  echo "Invalid angular velocity weight: $ANGULAR_WEIGHT" >&2
  exit 2
fi
if [[ ! "$TERMINATION_WEIGHT" =~ $NUMBER_PATTERN ]]; then
  echo "Invalid termination weight: $TERMINATION_WEIGHT" >&2
  exit 2
fi

OVERRIDES=$(printf \
  '{"reward_config.scales.tracking_lin_vel": %s, "reward_config.scales.tracking_ang_vel": %s, "reward_config.scales.termination": %s}' \
  "$LINEAR_WEIGHT" "$ANGULAR_WEIGHT" "$TERMINATION_WEIGHT")
LINEAR_TAG=$(sed 's/[^[:alnum:]]/p/g' <<< "$LINEAR_WEIGHT")
ANGULAR_TAG=$(sed 's/[^[:alnum:]]/p/g' <<< "$ANGULAR_WEIGHT")
TERMINATION_TAG=$(sed 's/[^[:alnum:]]/p/g' <<< "$TERMINATION_WEIGHT")
EXPERIMENT="sb-tracking-curr-lin${LINEAR_TAG}-ang${ANGULAR_TAG}-term${TERMINATION_TAG}"

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate spectral_fixed

echo "Environment: $ENV_NAME"
echo "W&B project: $WANDB_PROJECT"
echo "Linear velocity tracking weight: $LINEAR_WEIGHT"
echo "Angular velocity tracking weight: $ANGULAR_WEIGHT"
echo "Termination weight: $TERMINATION_WEIGHT"
echo "Seed: $SEED"
echo "Overrides: $OVERRIDES"

train-jax-ppo \
  --env_name "$ENV_NAME" \
  --num_timesteps "$NUM_TIMESTEPS" \
  --num_evals "$NUM_EVALS" \
  --playground_config_overrides="$OVERRIDES" \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_experiment_name "$EXPERIMENT" \
  --seed "$SEED"
