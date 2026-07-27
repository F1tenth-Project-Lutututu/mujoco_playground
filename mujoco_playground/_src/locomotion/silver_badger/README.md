# SilverBadger

The robot model and mesh assets are adapted from Nico Bohlinger's
[`one_policy_to_run_them_all`](https://github.com/nico-bohlinger/one_policy_to_run_them_all)
project:

> Copyright (c) 2024 Nico Bohlinger

They are distributed under the MIT License in
[`LICENSE_MODEL`](LICENSE_MODEL). The Playground integration is derived from
MuJoCo Playground's Apache-2.0 Go1 environment; see [`NOTICE`](NOTICE).

`SilverBadgerJoystickFlatTerrain` tracks commanded planar linear velocity and
yaw rate using the supplied SilverBadger MJX model. The action controls all 13
position actuators: the spine followed by the 12 leg joints.

`SilverBadgerJoystickFlatTerrainNoLinearVelocity` defines the same task but
removes local linear velocity from the actor observation. Its policy uses only
IMU angular velocity and projected gravity together with joint state, previous
action, and the velocity command. The privileged critic retains true linear
velocity for asymmetric training.

The actor observation follows the Go1 joystick convention and contains local
linear velocity, angular velocity, projected gravity, joint offsets, joint
velocities, the previous action, and the three-dimensional command. The critic
also receives simulator-only ground-truth state, torques, contacts, foot
velocities, air times, and perturbation state.

Train it with:

```bash
python learning/train_jax_ppo.py \
  --env_name=SilverBadgerJoystickFlatTerrain
```

For the IMU-only velocity variant:

```bash
python learning/train_jax_ppo.py \
  --env_name=SilverBadgerJoystickFlatTerrainNoLinearVelocity
```
