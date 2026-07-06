#!/bin/bash

#SBATCH --job-name=playground_go1
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=0-23:59:59
#SBATCH --array=0-4
##SBATCH --array=0-3

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate playground

SEED=${SLURM_ARRAY_TASK_ID}
ENV_NAME=${ENV_NAME:-Go1JoystickFlatTerrain}
IMPL=${IMPL:-jax}
DOMAIN_RANDOMIZATION=${DOMAIN_RANDOMIZATION:-False}
RESULTS_DIR=${RESULTS_DIR:-./results}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}

echo "Running MuJoCo Playground PPO: ENV_NAME=${ENV_NAME}, IMPL=${IMPL}, DOMAIN_RANDOMIZATION=${DOMAIN_RANDOMIZATION}, SEED=${SEED}"

python train.py \
	--env_name "${ENV_NAME}" \
	--impl "${IMPL}" \
	--domain_randomization "${DOMAIN_RANDOMIZATION}" \
	--use_wandb True \
	--wandb_project "${WANDB_PROJECT}" \
	--wandb_entity "${WANDB_ENTITY}" \
	--wandb_mode "${WANDB_MODE}" \
	--results_dir "${RESULTS_DIR}" \
	--seed $SEED
