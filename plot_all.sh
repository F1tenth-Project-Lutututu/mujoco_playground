environments=(
    BarkourJoystick
    Go1JoystickFlatTerrain
    Go1JoystickFlatTerrain25
    Go1JoystickFlatTerrain35
    Go1JoystickRoughTerrain
    Go1JoystickRoughTerrainPushesAndDomainRandomization
    SilverBadgerJoystickFlatTerrain
    SilverBadgerJoystickRoughTerrain
    SilverBadgerJoystickRoughTerrainNoLinearVelocityPushesAndDomainRandomization
    SpotFlatTerrainJoystick
    SpotFlatTerrainJoystickPushesAndDomainRandomization
    SpotJoystickGaitTracking
    SpotJoystickGaitTrackingDomainRandomization
  )
for environment in "${environments[@]}"; do
  #python learning/plot_policy_pareto.py "$environment" --shifted-log-percentage-y-axis || break
  #python learning/plot_policy_pareto.py "$environment" --linear-percentage-y-axis --x-exponential-strength 2 || break
  python learning/plot_policy_pareto.py "$environment" --x-exponential-strength 1 || break
done
