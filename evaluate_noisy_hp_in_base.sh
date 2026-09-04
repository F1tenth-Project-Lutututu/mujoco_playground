#!/bin/bash
# Evaluate noisy-high-pass-policy checkpoints in the non-noisy RLX-hard task.
#
# Usage:
#   sbatch --array=0-7 evaluate_noisy_hp_in_base.sh \
#     SilverBadgerJoystickFlatTerrainRLXHardNoisyHighpassObservation \
#     noisy_highpass_1pct
#
# Results are intentionally written outside the ordinary Pareto directory:
# their hp run names collide with the standard high-pass sweep until they are
# assigned distinct Pareto method IDs during collection.

#SBATCH --job-name=sb_noisy_hp_base_eval
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00

set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: sbatch --array=0-7 $0 <training-environment> <output-tag>" >&2
  exit 2
fi

TRAIN_ENV=$1
OUTPUT_TAG=$2
ROOT="$HOME/pl0467-01/scratch/pkicki/spectral_playground"
MODELS="$ROOT/logs/$TRAIN_ENV"
OUTPUT_ROOT="$ROOT/evaluations/pareto_cross_eval/SilverBadgerJoystickFlatTerrainRLXHard/$OUTPUT_TAG/raw_torque"
EVAL_ENV="SilverBadgerJoystickFlatTerrainRLXHard"
CHECKPOINT="000417792000"

# Conda's activation hook references optional toolchain variables that may be
# unset on clean Slurm nodes, so do not activate it under ``set -u``.
set +u
eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate spectral_fixed
set -u

mapfile -t RUNS < <(
  find "$MODELS" -mindepth 1 -maxdepth 1 -type d \
    -name '260904-highpass-400M-hp*-f5o1m10-seed*' \
    -printf '%f\n' | sort
)

for ((i=SLURM_ARRAY_TASK_ID; i<${#RUNS[@]}; i+=SLURM_ARRAY_TASK_COUNT)); do
  RUN="${RUNS[i]}"
  CKPT="$MODELS/$RUN/checkpoints/$CHECKPOINT"
  OUT="$OUTPUT_ROOT/$RUN/$CHECKPOINT"

  test -d "$CKPT"
  mkdir -p "$OUT"

  srun --exclusive --ntasks=1 --gpus-per-task=1 \
    python -m learning.evaluate_policy \
      --checkpoint "$CKPT" \
      --env_name "$EVAL_ENV" \
      --no-use_saved_environment_config \
      --environment_impl jax \
      --output_dir "$OUT" \
      --num_random_tasks 1024 \
      --task_seed 0 \
      --episode_length 1000 \
      --no-torque_highpass_normalize_by_capacity \
      --no-save_signals \
      --no-render_video \
      --require_cuda
done
