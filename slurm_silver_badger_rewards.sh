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
#SBATCH --array=0-19
#SBATCH --gres=gpu:1

#set -euo pipefail

# Tests four weights for the existing tracking_lin_vel reward, with five seeds
# per weight. No other environment or reward setting is changed.
#
# Usage:
#   sbatch slurm_silver_badger_rewards.sh [environment] [timesteps]
#
# Example:
#   sbatch slurm_silver_badger_rewards.sh \
#     SilverBadgerJoystickFlatTerrain 400000000

ENV_NAME=${1:-SilverBadgerJoystickFlatTerrain}
NUM_TIMESTEPS=${2:-400000000}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
WEIGHT_ID=$((TASK_ID / 5))
SEED=$((TASK_ID % 5))
NUM_EVALS=16

case "$WEIGHT_ID" in
  0) TRACKING_WEIGHT=1.0 ;;
  1) TRACKING_WEIGHT=2.0 ;;
  2) TRACKING_WEIGHT=4.0 ;;
  3) TRACKING_WEIGHT=8.0 ;;
  *)
    echo "Invalid tracking-weight index: $WEIGHT_ID" >&2
    exit 2
    ;;
esac

OVERRIDES=$(printf \
  '{"reward_config.scales.tracking_lin_vel": %s}' \
  "$TRACKING_WEIGHT")
WEIGHT_TAG=${TRACKING_WEIGHT/./p}
EXPERIMENT="silver-badger-tracking-lin-${WEIGHT_TAG}"

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate spectral_fixed

echo "Environment: $ENV_NAME"
echo "Linear velocity tracking weight: $TRACKING_WEIGHT"
echo "Seed: $SEED"
echo "Overrides: $OVERRIDES"

train-jax-ppo \
  --env_name "$ENV_NAME" \
  --num_timesteps "$NUM_TIMESTEPS" \
  --num_evals "$NUM_EVALS" \
  --playground_config_overrides="$OVERRIDES" \
  --use_wandb \
  --wandb_experiment_name "$EXPERIMENT" \
  --seed "$SEED"
