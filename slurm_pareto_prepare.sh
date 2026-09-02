#!/bin/bash

#SBATCH --job-name=pareto_prepare
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: sbatch $0 <environment> <models-root> <output-root> [run-date]" >&2
  exit 2
fi

ENVIRONMENT=$1
MODELS_ROOT=$2
OUTPUT_ROOT=$3
RUN_DATE=${4:-}

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate spectral_fixed

RUN_DATE_ARGUMENT=()
if [[ -n "$RUN_DATE" ]]; then
  RUN_DATE_ARGUMENT=(--run-date "$RUN_DATE")
fi

python -m learning.evaluate_pareto_on_cluster \
  --environment "$ENVIRONMENT" \
  --models-root "$MODELS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --prepare-only \
  "${RUN_DATE_ARGUMENT[@]}"
