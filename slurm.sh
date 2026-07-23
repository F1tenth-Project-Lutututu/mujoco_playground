#!/bin/bash

#SBATCH --job-name=playground_go1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
##SBATCH --partition=tesla
#SBATCH --partition=proxima
##SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=6
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=0-02:59:59
##SBATCH --time=0-23:59:59
#SBATCH --array=0-4
##SBATCH --array=0-3
#SBATCH --gres=gpu:1

#set -euo pipefail

# Usage:
#   sbatch slurm.sh <ar|tr|hp> <penalty-strength> [environment] \
#     [cutoff-hz] [difference-order]
#
# Examples:
#   sbatch slurm.sh ar 1e-1 BarkourJoystick
#   sbatch slurm.sh tr 8e-4 BerkeleyHumanoidJoystickFlatTerrain
#   sbatch slurm.sh hp 8e-3 SpotFlatTerrainJoystick
#   sbatch slurm.sh hp 8e-3 SpotFlatTerrainJoystick 10.0 2.0
#
# The corresponding environment variables can also be used:
#   METHOD=hp PENALTY_STRENGTH=1e-3 CUTOFF_HZ=10 \
#     DIFFERENCE_ORDER=2 ENV_NAME=Go1JoystickFlatTerrain sbatch slurm.sh
METHOD=${1:-${METHOD:-ar}}
PENALTY_STRENGTH=${2:-${PENALTY_STRENGTH:-1e-1}}
ENV_NAME=${3:-${ENV_NAME:-BarkourJoystick}}
CUTOFF_HZ=${4:-${CUTOFF_HZ:-5.0}}
DIFFERENCE_ORDER=${5:-${DIFFERENCE_ORDER:-1.0}}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-400000000}
SEED=${SLURM_ARRAY_TASK_ID:-0}
HIGHPASS_ORDER=1

NUMBER_PATTERN='^[+]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$'
for value_name in PENALTY_STRENGTH CUTOFF_HZ DIFFERENCE_ORDER; do
  value=${!value_name}
  if ! [[ $value =~ $NUMBER_PATTERN ]]; then
    echo "$value_name must be a non-negative number, got: $value" >&2
    exit 2
  fi
done

# Produce compact filesystem-safe tags: 1e-1 -> 1em1, 8e-4 -> 8em4.
STRENGTH_TAG=${PENALTY_STRENGTH,,}
STRENGTH_TAG=${STRENGTH_TAG#+}
STRENGTH_TAG=$(sed -E \
  -e 's/e-0*([0-9]+)/em\1/' \
  -e 's/e\+0*([0-9]+)/ep\1/' \
  -e 's/e0*([0-9]+)/ep\1/' \
  -e 's/\./p/g' \
  <<< "$STRENGTH_TAG")

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate spectral_fixed

EXP_NAME_SUFFIX=
case "$METHOD" in
  ar)
    METHOD_NAME=baseline
    PLAYGROUND_OVERRIDES=$(printf \
      '{"reward_config.scales.action_rate": -%s}' \
      "$PENALTY_STRENGTH")
    ;;
  tr)
    METHOD_NAME=torquerate
    PLAYGROUND_OVERRIDES=$(printf \
      '{"reward_config.scales.torque_rate": -%s, "reward_config.torque_rate_observe_state": true}' \
      "$PENALTY_STRENGTH")
    ;;
  hp)
    METHOD_NAME=highpass
    CUTOFF_TAG=$(sed -E \
      -e 's/\.0+$//' \
      -e 's/\.//g' \
      <<< "${CUTOFF_HZ,,}")
    DIFFERENCE_ORDER_TAG=$(sed -E \
      -e 's/\.//g' \
      <<< "${DIFFERENCE_ORDER,,}")
    EXP_NAME_SUFFIX="-f${CUTOFF_TAG}o${HIGHPASS_ORDER}m${DIFFERENCE_ORDER_TAG}"
    PLAYGROUND_OVERRIDES=$(printf \
      '{"reward_config.scales.torque_high_freq": -%s, "reward_config.torque_highpass_cutoff_hz": %s, "reward_config.torque_highpass_order": %s, "reward_config.torque_highpass_difference_order": %s, "reward_config.torque_highpass_normalize_by_capacity": false, "reward_config.torque_highpass_frequency_normalization": "white_spectrum", "reward_config.torque_highpass_observe_state": true}' \
      "$PENALTY_STRENGTH" "$CUTOFF_HZ" "$HIGHPASS_ORDER" \
      "$DIFFERENCE_ORDER")
    ;;
  *)
    echo "Unknown method '$METHOD'. Choose one of: ar, tr, hp." >&2
    exit 2
    ;;
esac

TIMESTEP_TAG=$((NUM_TIMESTEPS / 1000000))M
EXP_NAME=${EXP_NAME:-${METHOD_NAME}-${TIMESTEP_TAG}-${METHOD}${STRENGTH_TAG}${EXP_NAME_SUFFIX}}

export EXP_NAME
export PLAYGROUND_OVERRIDES

echo "Environment: $ENV_NAME"
echo "Method: $METHOD"
echo "Penalty strength: $PENALTY_STRENGTH"
if [[ $METHOD == hp ]]; then
  echo "High-pass cutoff: $CUTOFF_HZ Hz"
  echo "High-pass difference order: $DIFFERENCE_ORDER"
fi
echo "Experiment: $EXP_NAME"
echo "Overrides: $PLAYGROUND_OVERRIDES"

train-jax-ppo \
  --num_timesteps "$NUM_TIMESTEPS" \
  --env_name "$ENV_NAME" \
  --playground_config_overrides="$PLAYGROUND_OVERRIDES" \
  --use_wandb \
  --wandb_experiment_name "$EXP_NAME" \
  --seed "$SEED"
