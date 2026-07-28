#!/bin/bash

#SBATCH --job-name=silver_badger_tracking
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=0-05:59:59
#SBATCH --array=0-79
#SBATCH --gres=gpu:1

#set -euo pipefail

# Tests the Cartesian product of four linear-velocity and four angular-velocity
# tracking weights, with five seeds per pair. No other setting is changed.
#
# Usage:
#   sbatch slurm_silver_badger_rewards.sh [environment] [timesteps] [wandb_project]
#
# Example:
#   sbatch slurm_silver_badger_rewards.sh \
#     SilverBadgerJoystickFlatTerrain 400000000

ENV_NAME=${1:-SilverBadgerJoystickFlatTerrain}
NUM_TIMESTEPS=${2:-400000000}
WANDB_PROJECT=${3:-spectral_playground_silver_badger_factor_search}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
COMBINATION_ID=$((TASK_ID / 5))
SEED=$((TASK_ID % 5))
LINEAR_WEIGHT_ID=$((COMBINATION_ID / 4))
ANGULAR_WEIGHT_ID=$((COMBINATION_ID % 4))
NUM_EVALS=16

case "$LINEAR_WEIGHT_ID" in
  0) LINEAR_WEIGHT=1.0 ;;
  1) LINEAR_WEIGHT=2.0 ;;
  2) LINEAR_WEIGHT=4.0 ;;
  3) LINEAR_WEIGHT=8.0 ;;
  *)
    echo "Invalid linear-weight index: $LINEAR_WEIGHT_ID" >&2
    exit 2
    ;;
esac

case "$ANGULAR_WEIGHT_ID" in
  0) ANGULAR_WEIGHT=0.5 ;;
  1) ANGULAR_WEIGHT=1.0 ;;
  2) ANGULAR_WEIGHT=2.0 ;;
  3) ANGULAR_WEIGHT=4.0 ;;
  *)
    echo "Invalid angular-weight index: $ANGULAR_WEIGHT_ID" >&2
    exit 2
    ;;
esac

OVERRIDES=$(printf \
  '{"reward_config.scales.tracking_lin_vel": %s, "reward_config.scales.tracking_ang_vel": %s}' \
  "$LINEAR_WEIGHT" "$ANGULAR_WEIGHT")
LINEAR_TAG=${LINEAR_WEIGHT/./p}
ANGULAR_TAG=${ANGULAR_WEIGHT/./p}
EXPERIMENT="silver-badger-tracking-lin${LINEAR_TAG}-ang${ANGULAR_TAG}"

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate spectral_fixed

echo "Environment: $ENV_NAME"
echo "W&B project: $WANDB_PROJECT"
echo "Linear velocity tracking weight: $LINEAR_WEIGHT"
echo "Angular velocity tracking weight: $ANGULAR_WEIGHT"
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
