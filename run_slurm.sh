#sbatch slurm.sh <method> <strength> [environment] [cutoff-hz] [difference-order]
#sbatch slurm.sh ar 1e-1 BarkourJoystick
#sbatch slurm.sh tr 8e-4 BerkeleyHumanoidJoystickFlatTerrain
#sbatch slurm.sh hp 8e-3 SpotFlatTerrainJoystick


#--env_name Go1JoystickFlatTerrain25 \
#--env_name Go1JoystickFlatTerrain35 \
#--env_name Go1JoystickRoughTerrain \
#--env_name Go1JoystickFlatTerrain \
#--env_name BerkeleyHumanoidJoystickFlatTerrain \
#--env_name BerkeleyHumanoidJoystickRoughTerrain \
#--env_name BarkourJoystick \

#export NUM_TIMESTEPS=1000M
export NUM_TIMESTEPS=400M

#sbatch slurm.sh hp 3e-3 BerkeleyHumanoidJoystickFlatTerrain 6.0 1.0
#sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickFlatTerrain 7.0 1.0
#sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickFlatTerrain 8.0 1.0
#sbatch slurm.sh hp 3e-3 BarkourJoystick 5.0 1.0
#sbatch slurm.sh hp 3e-2 Op3Joystick 5.0 1.0
#sbatch slurm.sh tr 2e-4 Go1JoystickFlatTerrain25 

#for P in 1e-1 1e-2 1e-3 1e-4; do
#for P in 1e-2 1e-3 1e-4; do
#for P in 1e0 1e1; do
#for P in 5e-3; do
#for P in 1e-5 8e-6 4e-6; do
#for P in 2e-1 3e-1 1e0; do
#for P in 2e-1 3e-1 5e-2; do
#for P in 1e-5 2e-5 4e-5 8e-5; do
#for P in 4e-4 5e-4 3e-4; do
#for P in 2e-4 4e-4 8e-4; do
for P in 2e-3 4e-3 8e-3; do
#for P in 2e-2 4e-2 8e-2; do
#for P in 2e-2 3e-2 4e-2; do
#for P in 1e-2 2e-2 4e-2 6e-2; do
#for P in 2e-1 4e-1 8e-1; do
#for P in 2e-1 5e-1 1e0; do
#for P in 1e-3 5e-4 8e-4 2e-3 4e-3; do
#for P in 3.0 4.0 6.0 7.0; do
#for P in 4.0 5.0 6.0; do
#for P in 8.0 9.0 10.0; do
#for P in 1.5 2.0 2.5 3.0; do
#for P in 0.0 0.5 1.5 2.0; do
  echo $P
  #export NUM_TIMESTEPS=1000M
  #sbatch slurm.sh ar $P Go1JoystickFlatTerrain25 
  #sbatch slurm.sh tr $P Go1JoystickFlatTerrain25 
  #sbatch slurm.sh hp $P Go1JoystickFlatTerrain25 7.0 1.0
  #sbatch slurm.sh hp 8e-4 Go1JoystickFlatTerrain25 $P 1.0
  #sbatch slurm.sh hp 8e-4 Go1JoystickFlatTerrain25 7.0 $P

  #export NUM_TIMESTEPS=1000M
  #sbatch slurm.sh ar $P Go1JoystickFlatTerrain35
  #sbatch slurm.sh tr $P Go1JoystickFlatTerrain35
  #sbatch slurm.sh hp $P Go1JoystickFlatTerrain35 7.0 1.0

  #sbatch slurm.sh ar $P Go1JoystickFlatTerrain
  #sbatch slurm.sh tr $P Go1JoystickFlatTerrain 
  #sbatch slurm.sh hp $P Go1JoystickFlatTerrain 5.0 1.0

  #sbatch slurm.sh ar $P Go1JoystickRoughTerrain
  #sbatch slurm.sh tr $P Go1JoystickRoughTerrain 
  #sbatch slurm.sh hp $P Go1JoystickRoughTerrain 5.0 1.0

  #sbatch slurm.sh ar $P Go1JoystickRoughTerrainPushesAndDomainRandomization
  sbatch slurm.sh tr $P Go1JoystickRoughTerrainPushesAndDomainRandomization 
  #sbatch slurm.sh hp $P Go1JoystickRoughTerrainPushesAndDomainRandomization 5.0 1.0

  #sbatch slurm.sh ar $P SpotFlatTerrainJoystick
  #sbatch slurm.sh tr $P SpotFlatTerrainJoystick
  #sbatch slurm.sh hp $P SpotFlatTerrainJoystick 5.0 1.0

  #sbatch slurm.sh ar $P SpotFlatTerrainJoystickPushesAndDomainRandomization
  #sbatch slurm.sh tr $P SpotFlatTerrainJoystickPushesAndDomainRandomization
  #sbatch slurm.sh hp $P SpotFlatTerrainJoystickPushesAndDomainRandomization 5.0 1.0

  #sbatch slurm.sh ar $P SpotJoystickGaitTracking
  #sbatch slurm.sh tr $P SpotJoystickGaitTracking
  #sbatch slurm.sh hp $P SpotJoystickGaitTracking 5.0 1.0

  #sbatch slurm.sh ar $P SpotJoystickGaitTrackingDomainRandomization
  #sbatch slurm.sh tr $P SpotJoystickGaitTrackingDomainRandomization
  #sbatch slurm.sh hp $P SpotJoystickGaitTrackingDomainRandomization 5.0 1.0

  #sbatch slurm.sh ar $P BarkourJoystick
  #sbatch slurm.sh tr $P BarkourJoystick
  #sbatch slurm.sh hp $P BarkourJoystick 5.0 1.0
  #sbatch slurm.sh hp 3e-3 BarkourJoystick 5.0 $P 

  #sbatch slurm.sh ar $P SilverBadgerJoystickFlatTerrain
  #sbatch slurm.sh tr $P SilverBadgerJoystickFlatTerrain
  #sbatch slurm.sh hp $P SilverBadgerJoystickFlatTerrain 5.0 1.0

  #sbatch slurm.sh ar $P SilverBadgerJoystickRoughTerrain
  #sbatch slurm.sh tr $P SilverBadgerJoystickRoughTerrain
  #sbatch slurm.sh hp $P SilverBadgerJoystickRoughTerrain 5.0 1.0

  #sbatch slurm.sh ar $P SilverBadgerJoystickFlatTerrainNoLinearVelocity
  #sbatch slurm.sh tr $P SilverBadgerJoystickFlatTerrainNoLinearVelocity
  #sbatch slurm.sh hp $P SilverBadgerJoystickFlatTerrainNoLinearVelocity 5.0 1.0

  #sbatch slurm.sh ar $P SilverBadgerJoystickRoughTerrainNoLinearVelocity
  #sbatch slurm.sh tr $P SilverBadgerJoystickRoughTerrainNoLinearVelocity
  #sbatch slurm.sh hp $P SilverBadgerJoystickRoughTerrainNoLinearVelocity 5.0 1.0

  #sbatch slurm.sh ar $P SilverBadgerJoystickRoughTerrainNoLinearVelocityPushes
  #sbatch slurm.sh tr $P SilverBadgerJoystickRoughTerrainNoLinearVelocityPushes
  #sbatch slurm.sh hp $P SilverBadgerJoystickRoughTerrainNoLinearVelocityPushes 5.0 1.0

  #sbatch slurm.sh ar $P SilverBadgerJoystickRoughTerrainNoLinearVelocityPushesAndDomainRandomization
  #sbatch slurm.sh tr $P SilverBadgerJoystickRoughTerrainNoLinearVelocityPushesAndDomainRandomization
  #sbatch slurm.sh hp $P SilverBadgerJoystickRoughTerrainNoLinearVelocityPushesAndDomainRandomization 5.0 1.0

  #sbatch slurm.sh ar $P BerkeleyHumanoidJoystickRoughTerrain
  #sbatch slurm.sh tr $P BerkeleyHumanoidJoystickRoughTerrain
  #sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickRoughTerrain $P 1.0
  #sbatch slurm.sh hp 2e-3 BerkeleyHumanoidJoystickRoughTerrain 5.0 $P

  #sbatch slurm.sh ar $P ApolloJoystickFlatTerrain
  #sbatch slurm.sh tr $P ApolloJoystickFlatTerrain
  #sbatch slurm.sh hp $P ApolloJoystickFlatTerrain 5.0 1.0

  #sbatch slurm.sh ar $P G1JoystickFlatTerrain
  #sbatch slurm.sh tr $P G1JoystickFlatTerrain
  #sbatch slurm.sh hp $P G1JoystickFlatTerrain 5.0 1.0
  #sbatch slurm.sh hp 4e-4 G1JoystickFlatTerrain $P 1.0
  #sbatch slurm.sh hp 4e-4 G1JoystickFlatTerrain 5.0 $P

  #sbatch slurm.sh ar $P G1JoystickRoughTerrain
  #sbatch slurm.sh tr $P G1JoystickRoughTerrain
  #sbatch slurm.sh hp $P G1JoystickRoughTerrain 5.0 1.0

  #sbatch slurm.sh ar $P T1JoystickFlatTerrain
  #sbatch slurm.sh tr $P T1JoystickFlatTerrain
  #sbatch slurm.sh hp $P T1JoystickFlatTerrain 5.0 1.0
  #sbatch slurm.sh hp 1e-4 T1JoystickFlatTerrain $P 1.0
  #sbatch slurm.sh hp 1e-4 T1JoystickFlatTerrain 5.0 $P

  #sbatch slurm.sh ar $P T1JoystickRoughTerrain
  #sbatch slurm.sh tr $P T1JoystickRoughTerrain
  #sbatch slurm.sh hp $P T1JoystickRoughTerrain 5.0 1.0
  #sbatch slurm.sh hp 1e-4 T1JoystickRoughTerrain $P 1.0
  #sbatch slurm.sh hp 1e-4 T1JoystickRoughTerrain 5.0 $P

  #sbatch slurm.sh ar $P H1JoystickGaitTracking
  #sbatch slurm.sh tr $P H1JoystickGaitTracking
  #sbatch slurm.sh hp $P H1JoystickGaitTracking 5.0 1.0

  #sbatch slurm.sh ar $P Op3Joystick
  #sbatch slurm.sh tr $P Op3Joystick
  #sbatch slurm.sh hp $P Op3Joystick 5.0 1.0
done

