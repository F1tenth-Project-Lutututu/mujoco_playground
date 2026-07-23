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

SEED=${SLURM_ARRAY_TASK_ID}

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate spectral_fixed


export EXP_NAME=baseline-400M-ar1em1
export PLAYGROUND_OVERRIDES='{"reward_config.scales.action_rate": -1e-1}'

#export EXP_NAME=torquerate-400M-ar8em4
#export PLAYGROUND_OVERRIDES='{
#    "reward_config.scales.torque_rate": -8e-4,
#    "reward_config.torque_rate_observe_state": true
#}'

      #"reward_config.torque_highpass_signal": "action",
#export EXP_NAME=newhfstateaction1em4-400M-f5o2m25
#export EXP_NAME=newhfstate1em4-400M-f5o1m10

#export EXP_NAME=normhfstate8em3-400M-f5o1m10
#export PLAYGROUND_OVERRIDES='{
#      "reward_config.scales.torque_high_freq": -8e-3, 
#      "reward_config.torque_highpass_cutoff_hz": 5.0,
#      "reward_config.torque_highpass_order": 1,
#      "reward_config.torque_highpass_difference_order": 1.0,
#      "reward_config.torque_highpass_normalize_by_capacity": false,
#      "reward_config.torque_highpass_frequency_normalization": "white_spectrum",
#      "reward_config.torque_highpass_observe_state": true
#    }'

#normalized - for future tests
#"reward_config.scales.torque_high_freq": -0.01,

  #--env_name Go1JoystickFlatTerrain25 \
  #--env_name Go1JoystickRoughTerrain \
  #--env_name Go1JoystickFlatTerrain \
  #--env_name BerkeleyHumanoidJoystickFlatTerrain \
  #--env_name BerkeleyHumanoidJoystickRoughTerrain \
  #--env_name BarkourJoystick \
train-jax-ppo \
  --num_timesteps 400000000 \
  --env_name BarkourJoystick \
  --playground_config_overrides="$PLAYGROUND_OVERRIDES" \
  --use_wandb \
  --wandb_experiment_name $EXP_NAME \
  --seed $SEED
