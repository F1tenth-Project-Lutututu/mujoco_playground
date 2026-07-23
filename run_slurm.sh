#sbatch slurm.sh <method> <strength> [environment] [cutoff-hz] [difference-order]
#sbatch slurm.sh ar 1e-1 BarkourJoystick
#sbatch slurm.sh tr 8e-4 BerkeleyHumanoidJoystickFlatTerrain
#sbatch slurm.sh hp 8e-3 SpotFlatTerrainJoystick


#--env_name Go1JoystickFlatTerrain25 \
#--env_name Go1JoystickRoughTerrain \
#--env_name Go1JoystickFlatTerrain \
#--env_name BerkeleyHumanoidJoystickFlatTerrain \
#--env_name BerkeleyHumanoidJoystickRoughTerrain \
#--env_name BarkourJoystick \

#for P in 1e-1 1e-2 1e-3 1e-4; do
#for P in 1e-2 1e-3 1e-4; do
#for P in 3.0 4.0 6.0 7.0; do
for P in 1.5 2.0 2.5 3.0; do
  echo $P
  #sbatch slurm.sh ar $P BarkourJoystick
  #sbatch slurm.sh tr $P BarkourJoystick
  #sbatch slurm.sh hp $P BarkourJoystick
  #sbatch slurm.sh hp $P Go1JoystickFlatTerrain25 7.0 1.0
  #sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickRoughTerrain $P 1.0
  sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickRoughTerrain 5.0 $P
done

