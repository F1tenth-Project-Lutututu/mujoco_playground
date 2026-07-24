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

#export NUM_TIMESTEPS=1000M
#export NUM_TIMESTEPS=400M

#sbatch slurm.sh hp 3e-3 BerkeleyHumanoidJoystickFlatTerrain 6.0 1.0
#sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickFlatTerrain 7.0 1.0
#sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickFlatTerrain 8.0 1.0
#sbatch slurm.sh hp 3e-3 BarkourJoystick 5.0 1.0

for P in 1e-1 1e-2 1e-3 1e-4; do
#for P in 1e-2 1e-3 1e-4; do
#for P in 1e-1 3e-1 1e0; do
#for P in 2e-4 4e-4 8e-4; do
#for P in 4e-4 5e-4 3e-4; do
#for P in 2e-3 4e-3 8e-3; do
#for P in 2e-2 4e-2 8e-2; do
#for P in 1e-2 2e-2 4e-2 6e-2; do
#for P in 2e-1 4e-1 8e-1 5e-2; do
#for P in 1e-3 5e-4 8e-4 2e-3 4e-3; do
#for P in 3.0 4.0 6.0 7.0; do
#for P in 4.0 5.0 6.0; do
#for P in 8.0 9.0 10.0; do
#for P in 1.5 2.0 2.5 3.0; do
  echo $P
  #sbatch slurm.sh ar $P Go1JoystickFlatTerrain25 
  #sbatch slurm.sh tr $P Go1JoystickFlatTerrain25 
  #sbatch slurm.sh hp $P Go1JoystickFlatTerrain25 7.0 1.0
  #sbatch slurm.sh hp 8e-4 Go1JoystickFlatTerrain25 $P 1.0
  #sbatch slurm.sh hp 8e-4 Go1JoystickFlatTerrain25 7.0 $P

  sbatch slurm.sh ar $P Go1JoystickFlatTerrain35


  #sbatch slurm.sh ar $P SpotFlatTerrainJoystick
  #sbatch slurm.sh tr $P SpotFlatTerrainJoystick
  #sbatch slurm.sh hp $P SpotFlatTerrainJoystick 5.0 1.0

  #sbatch slurm.sh ar $P BarkourJoystick
  #sbatch slurm.sh tr $P BarkourJoystick
  #sbatch slurm.sh hp $P BarkourJoystick 5.0 1.0

  #sbatch slurm.sh ar $P Go1JoystickRoughTerrain
  #sbatch slurm.sh tr $P Go1JoystickRoughTerrain

  #sbatch slurm.sh ar $P BerkeleyHumanoidJoystickRoughTerrain
  #sbatch slurm.sh tr $P BerkeleyHumanoidJoystickRoughTerrain
  #sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickRoughTerrain $P 1.0
  #sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickRoughTerrain 5.0 $P
done

