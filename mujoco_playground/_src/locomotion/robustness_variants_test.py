"""Tests for registered quadruped robustness-task variants."""

from absl.testing import absltest

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


class RobustnessVariantsTest(absltest.TestCase):

  def test_go1_and_spot_variants_have_expected_defaults(self):
    go1_bases = (
        "Go1JoystickFlatTerrain",
        "Go1JoystickFlatTerrain25",
        "Go1JoystickFlatTerrain35",
        "Go1JoystickRoughTerrain",
        "Go1JoystickRoughTerrain25",
    )
    spot_bases = ("SpotFlatTerrainJoystick",)
    variants = {
        "Pushes": (True, False),
        "DomainRandomization": (False, True),
        "PushesAndDomainRandomization": (True, True),
    }
    for base in go1_bases + spot_bases:
      for suffix, (pushes, randomization) in variants.items():
        name = f"{base}{suffix}"
        self.assertIn(name, registry.ALL_ENVS)
        config = registry.get_default_config(name)
        self.assertEqual(config.pert_config.enable, pushes)
        self.assertEqual(config.domain_randomization, randomization)
        registry.get_domain_randomizer(name)
        locomotion_params.brax_ppo_config(name)

  def test_go1_nonjoystick_domain_randomization_variants(self):
    for base in ("Go1Getup", "Go1Handstand", "Go1Footstand"):
      name = f"{base}DomainRandomization"
      self.assertIn(name, registry.ALL_ENVS)
      self.assertTrue(
          registry.get_default_config(name).domain_randomization
      )
      registry.get_domain_randomizer(name)
      locomotion_params.brax_ppo_config(name)

  def test_spot_nonjoystick_domain_randomization_variants(self):
    for base in ("SpotGetup", "SpotJoystickGaitTracking"):
      name = f"{base}DomainRandomization"
      self.assertIn(name, registry.ALL_ENVS)
      self.assertTrue(
          registry.get_default_config(name).domain_randomization
      )
      registry.get_domain_randomizer(name)
      locomotion_params.brax_ppo_config(name)


if __name__ == "__main__":
  absltest.main()
