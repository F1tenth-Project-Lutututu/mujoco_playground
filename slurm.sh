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
#   sbatch slurm.sh <ar|as|tr|ts|hp> <penalty-strength> [environment] \
#     [cutoff-hz] [difference-order] [num-timesteps]
#
# Examples:
#   sbatch slurm.sh ar 1e-1 BarkourJoystick
#   sbatch slurm.sh as 1e-1 BarkourJoystick
#   sbatch slurm.sh tr 8e-4 BerkeleyHumanoidJoystickFlatTerrain
#   sbatch slurm.sh ts 8e-4 Go1JoystickFlatTerrain
#   sbatch slurm.sh hp 8e-3 SpotFlatTerrainJoystick
#   sbatch slurm.sh hp 8e-3 SpotFlatTerrainJoystick 10.0 2.0
#   sbatch slurm.sh hp 8e-3 SpotFlatTerrainJoystick 10.0 2.0 800M
#
# The corresponding environment variables can also be used:
#   METHOD=hp PENALTY_STRENGTH=1e-3 CUTOFF_HZ=10 \
#     DIFFERENCE_ORDER=2 NUM_TIMESTEPS=800M \
#     ENV_NAME=Go1JoystickFlatTerrain sbatch slurm.sh
METHOD=${1:-${METHOD:-ar}}
PENALTY_STRENGTH=${2:-${PENALTY_STRENGTH:-1e-1}}
ENV_NAME=${3:-${ENV_NAME:-BarkourJoystick}}
CUTOFF_HZ=${4:-${CUTOFF_HZ:-5.0}}
DIFFERENCE_ORDER=${5:-${DIFFERENCE_ORDER:-1.0}}
NUM_TIMESTEPS_INPUT=${6:-${NUM_TIMESTEPS:-400M}}
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

if [[ $NUM_TIMESTEPS_INPUT =~ ^([1-9][0-9]*)[Mm]$ ]]; then
  NUM_TIMESTEPS=$((${BASH_REMATCH[1]} * 1000000))
elif [[ $NUM_TIMESTEPS_INPUT =~ ^[1-9][0-9]*$ ]]; then
  NUM_TIMESTEPS=$NUM_TIMESTEPS_INPUT
else
  echo "NUM_TIMESTEPS must be a positive integer or use an M suffix (for example, 400M), got: $NUM_TIMESTEPS_INPUT" >&2
  exit 2
fi

LOG_INTERVAL_TIMESTEPS=25000000
NUM_EVALS=$(((NUM_TIMESTEPS + LOG_INTERVAL_TIMESTEPS - 1) / LOG_INTERVAL_TIMESTEPS))

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

# Fail fast when Slurm allocates a GPU that CUDA/JAX cannot use.  A running job
# cannot update its own ExcNodeList, so submit a replacement one-seed array
# which excludes the failed node.  Carry an explicit counter between
# replacement jobs to prevent an infinite retry loop.
MAX_GPU_REQUEUES=${MAX_GPU_REQUEUES:-3}
GPU_RETRY_COUNT=${GPU_RETRY_COUNT:-0}
GPU_DIAGNOSTICS_DIR=${GPU_DIAGNOSTICS_DIR:-logs/gpu_diagnostics}
if ! [[ $MAX_GPU_REQUEUES =~ ^[0-9]+$ ]]; then
  echo "MAX_GPU_REQUEUES must be a non-negative integer, got: $MAX_GPU_REQUEUES" >&2
  exit 2
fi
if ! [[ $GPU_RETRY_COUNT =~ ^[0-9]+$ ]]; then
  echo "GPU_RETRY_COUNT must be a non-negative integer, got: $GPU_RETRY_COUNT" >&2
  exit 2
fi

log_gpu_diagnostics() {
  local diagnostic_file=$1
  mkdir -p "$GPU_DIAGNOSTICS_DIR"
  {
    echo "=== GPU failure diagnostics ==="
    date --iso-8601=seconds
    echo "hostname=$(hostname --fqdn 2>&1)"
    echo "slurmd_nodename=${SLURMD_NODENAME:-unset}"
    echo "job_id=${SLURM_JOB_ID:-unset}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID:-unset}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID:-unset}"
    echo "gpu_retry_count=$GPU_RETRY_COUNT"
    echo "slurm_restart_count=${SLURM_RESTART_COUNT:-0}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
    echo "slurm_step_gpus=${SLURM_STEP_GPUS:-unset}"
    echo "gpu_device_ordinal=${GPU_DEVICE_ORDINAL:-unset}"
    echo
    echo "=== Slurm job record ==="
    scontrol show job --details "${SLURM_JOB_ID:-}" 2>&1 || true
    echo
    echo "=== NVIDIA device files ==="
    ls -la /dev/nvidia* 2>&1 || true
    echo
    echo "=== NVIDIA driver ==="
    cat /proc/driver/nvidia/version 2>&1 || true
    echo
    echo "=== nvidia-smi -L ==="
    nvidia-smi -L 2>&1 || true
    echo
    echo "=== nvidia-smi ==="
    nvidia-smi 2>&1 || true
    echo
    echo "=== NVIDIA health and ECC ==="
    nvidia-smi -q 2>&1 || true
    echo
    echo "=== NVIDIA topology ==="
    nvidia-smi topo -m 2>&1 || true
    echo
    echo "=== Loaded NVIDIA modules ==="
    lsmod 2>&1 | grep -E '^nvidia' || true
    echo
    echo "=== Recent kernel GPU messages ==="
    dmesg --level=err,warn 2>&1 | tail -200 || true
    echo
    echo "=== JAX probe ==="
    python - <<'PY'
import os
import platform
import traceback

print("python:", platform.python_version())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))
try:
  import jax
  import jaxlib

  print("jax:", jax.__version__)
  print("jaxlib:", jaxlib.__version__)
  print("devices:", jax.devices())
  print("default_backend:", jax.default_backend())
except Exception:
  traceback.print_exc()
PY
  } 2>&1 | tee "$diagnostic_file"
}

if ! python - <<'PY'
import jax

devices = jax.devices()
backend = jax.default_backend()
print(f"JAX preflight: backend={backend}, devices={devices}")
if backend != "gpu" or not any(device.platform == "gpu" for device in devices):
  raise RuntimeError("Slurm allocated a GPU, but JAX cannot access it.")
PY
then
  DIAGNOSTIC_FILE="${GPU_DIAGNOSTICS_DIR}/gpu_failure-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-unknown}}_${SLURM_ARRAY_TASK_ID:-0}-retry${GPU_RETRY_COUNT}-$(date +%Y%m%dT%H%M%S).log"
  log_gpu_diagnostics "$DIAGNOSTIC_FILE"

  if ((GPU_RETRY_COUNT >= MAX_GPU_REQUEUES)); then
    echo "GPU preflight failed after $GPU_RETRY_COUNT retries; not retrying." >&2
    echo "Diagnostics: $DIAGNOSTIC_FILE" >&2
    exit 70
  fi

  FAILED_NODE=${SLURMD_NODENAME:-$(hostname --short)}
  JOB_RECORD=$(scontrol show job --oneliner "$SLURM_JOB_ID")
  CURRENT_EXCLUSIONS=$(sed -n 's/.* ExcNodeList=\([^ ]*\).*/\1/p' <<< "$JOB_RECORD")
  if [[ -z $CURRENT_EXCLUSIONS || $CURRENT_EXCLUSIONS == "(null)" ]]; then
    NEW_EXCLUSIONS=$FAILED_NODE
  elif [[ ,$CURRENT_EXCLUSIONS, == *",$FAILED_NODE,"* ]]; then
    NEW_EXCLUSIONS=$CURRENT_EXCLUSIONS
  else
    NEW_EXCLUSIONS="${CURRENT_EXCLUSIONS},${FAILED_NODE}"
  fi

  SCRIPT_PATH=$(sed -n 's/.* Command=\([^ ]*\).*/\1/p' <<< "$JOB_RECORD")
  if [[ -z $SCRIPT_PATH || ! -f $SCRIPT_PATH ]]; then
    echo "Could not resolve the submitted Slurm script from: $JOB_RECORD" >&2
    exit 71
  fi

  NEXT_GPU_RETRY_COUNT=$((GPU_RETRY_COUNT + 1))
  echo "Excluding $FAILED_NODE and submitting replacement seed ${SLURM_ARRAY_TASK_ID:-0} (attempt $NEXT_GPU_RETRY_COUNT/$MAX_GPU_REQUEUES)." >&2
  if ! (
    cd "${SLURM_SUBMIT_DIR:-$(dirname "$SCRIPT_PATH")}" &&
    sbatch \
      --array="${SLURM_ARRAY_TASK_ID:-0}" \
      --exclude="$NEW_EXCLUSIONS" \
      --export="ALL,GPU_RETRY_COUNT=$NEXT_GPU_RETRY_COUNT" \
      "$SCRIPT_PATH" "$@"
  ); then
    echo "Slurm refused to submit the replacement task." >&2
    exit 72
  fi
  exit 0
fi

EXP_NAME_SUFFIX=
case "$METHOD" in
  ar)
    METHOD_NAME=baseline
    PLAYGROUND_OVERRIDES=$(printf \
      '{"reward_config.scales.action_rate": -%s}' \
      "$PENALTY_STRENGTH")
    ;;
  as)
    METHOD_NAME=actionsmoothness
    PLAYGROUND_OVERRIDES=$(printf \
      '{"reward_config.scales.action_rate": -%s, "reward_config.action_rate_use_second_difference": true}' \
      "$PENALTY_STRENGTH")
    ;;
  tr)
    METHOD_NAME=torquerate
    PLAYGROUND_OVERRIDES=$(printf \
      '{"reward_config.scales.torque_rate": -%s, "reward_config.torque_rate_observe_state": true}' \
      "$PENALTY_STRENGTH")
    ;;
  ts)
    METHOD_NAME=torquesmoothness
    PLAYGROUND_OVERRIDES=$(printf \
      '{"reward_config.scales.torque_rate": -%s, "reward_config.torque_rate_use_second_difference": true, "reward_config.torque_rate_observe_state": true}' \
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
    echo "Unknown method '$METHOD'. Choose one of: ar, as, tr, ts, hp." >&2
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
echo "Number of timesteps: $NUM_TIMESTEPS"
echo "Number of evaluations/logs: $NUM_EVALS (every 25M timesteps)"
if [[ $METHOD == hp ]]; then
  echo "High-pass cutoff: $CUTOFF_HZ Hz"
  echo "High-pass difference order: $DIFFERENCE_ORDER"
fi
echo "Experiment: $EXP_NAME"
echo "Overrides: $PLAYGROUND_OVERRIDES"

train-jax-ppo \
  --num_timesteps "$NUM_TIMESTEPS" \
  --num_evals "$NUM_EVALS" \
  --env_name "$ENV_NAME" \
  --playground_config_overrides="$PLAYGROUND_OVERRIDES" \
  --use_wandb \
  --wandb_experiment_name "$EXP_NAME" \
  --seed "$SEED"
