# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for the SilverBadger environment."""

import jax
import jax.numpy as jp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.4, 1.0).
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.4, maxval=1.0)
    )

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(13,), minval=0.9, maxval=1.1
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(13,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Jitter center of mass positiion: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + dpos
    )

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(13,), minval=-0.05, maxval=0.05)
    )

    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    )

  (
      friction,
      body_ipos,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
  })

  model = model.tree_replace({  # pyrefly: ignore[bad-assignment]
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
  })

  return model, in_axes


def domain_randomize_rlx_hard(model: mjx.Model, rng: jax.Array):
  """Reset-time subset of RL-X's hardest Silver Badger model randomization.

  RL-X additionally changes stateful quantities during an episode (delayed
  actions, joint dropout, and curriculum-scaled perturbations).  The Brax
  randomization callback only receives a model at reset, so those parts are
  implemented by the task's existing velocity-kick mechanism where possible
  and are deliberately not simulated here.
  """
  @jax.vmap
  def rand_dynamics(key):
    keys = jax.random.split(key, 12)
    # RL-X randomizes all three contact-friction dimensions by +/- 100%.
    friction = model.geom_friction * (
        1.0 + jax.random.uniform(
            keys[0], model.geom_friction.shape, minval=-1.0, maxval=1.0
        )
    )
    # Contact stiffness/damping, impedance, gravity, and fluid parameters.
    solref = model.geom_solref.at[:, 0].set(
        model.geom_solref[:, 0]
        * jax.random.uniform(keys[1], model.geom_solref[:, 0].shape,
                             minval=0.5, maxval=1.5)
    )
    solref = solref.at[:, 1].set(
        model.geom_solref[:, 1]
        * jax.random.uniform(keys[2], model.geom_solref[:, 1].shape,
                             minval=0.4, maxval=1.6)
    )
    solimp = model.geom_solimp * jax.random.uniform(
        keys[3], model.geom_solimp.shape, minval=0.2, maxval=1.8
    )
    solimp = jp.clip(solimp, 0.0, 6.0)
    gravity = model.opt.gravity.at[:2].set(
        jax.random.uniform(keys[4], (2,), minval=-0.5, maxval=0.5)
    )
    gravity = gravity.at[2].set(
        model.opt.gravity[2] * jax.random.uniform(keys[5], minval=0.9, maxval=1.1)
    )
    body_mass = model.body_mass * jax.random.uniform(
        keys[6], model.body_mass.shape, minval=0.85, maxval=1.15
    )
    body_inertia = model.body_inertia * jax.random.uniform(
        keys[7], model.body_inertia.shape, minval=0.80, maxval=1.20
    )
    body_ipos = model.body_ipos + jax.random.uniform(
        keys[8], model.body_ipos.shape, minval=-0.005, maxval=0.005
    )
    armature = model.dof_armature * jax.random.uniform(
        keys[9], model.dof_armature.shape, minval=0.5, maxval=1.5
    )
    frictionloss = model.dof_frictionloss * jax.random.uniform(
        keys[10], model.dof_frictionloss.shape, minval=0.0, maxval=2.0
    )
    force_range = model.actuator_forcerange * jax.random.uniform(
        keys[11], model.actuator_forcerange.shape, minval=0.85, maxval=1.15
    )
    return (friction, solref, solimp, gravity, body_mass, body_inertia,
            body_ipos, armature, frictionloss, force_range)

  (friction, solref, solimp, gravity, body_mass, body_inertia, body_ipos,
   armature, frictionloss, force_range) = rand_dynamics(rng)
  in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({
      "geom_friction": 0, "geom_solref": 0, "geom_solimp": 0,
      "opt.gravity": 0, "body_mass": 0, "body_inertia": 0,
      "body_ipos": 0, "dof_armature": 0, "dof_frictionloss": 0,
      "actuator_forcerange": 0,
  })
  return model.tree_replace({
      "geom_friction": friction, "geom_solref": solref, "geom_solimp": solimp,
      "opt.gravity": gravity, "body_mass": body_mass,
      "body_inertia": body_inertia, "body_ipos": body_ipos,
      "dof_armature": armature, "dof_frictionloss": frictionloss,
      "actuator_forcerange": force_range,
  }), in_axes
