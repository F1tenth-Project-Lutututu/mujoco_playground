#!/bin/bash

#SBATCH --job-name=pareto_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

#set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: sbatch $0 <manifest> <models-root> <output-root> [num-random-tasks] [task-seed]" >&2
  exit 2
fi

MANIFEST=$1
MODELS_ROOT=$2
OUTPUT_ROOT=$3
NUM_RANDOM_TASKS=${4:-2048}
TASK_SEED=${5:-0}

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate spectral_fixed

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export MUJOCO_GL=egl

echo "Host: $(hostname --fqdn)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import jax

devices = jax.devices()
backend = jax.default_backend()
print(f"JAX preflight: backend={backend}, devices={devices}")
if backend != "gpu" or not any(device.platform == "gpu" for device in devices):
  raise RuntimeError(
      "Slurm job has no JAX-visible GPU; refusing CPU fallback."
  )
PY
echo "Manifest: $MANIFEST"
echo "Models root: $MODELS_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Random tasks per policy: $NUM_RANDOM_TASKS"

python -m learning.evaluate_pareto_on_cluster \
  --manifest "$MANIFEST" \
  --models-root "$MODELS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --num-random-tasks "$NUM_RANDOM_TASKS" \
  --task-seed "$TASK_SEED"
