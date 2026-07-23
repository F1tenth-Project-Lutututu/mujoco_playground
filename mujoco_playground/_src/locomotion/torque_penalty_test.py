"""Tests for shared quadruped torque regularization."""

from types import SimpleNamespace

from absl.testing import absltest
import jax.numpy as jp
from ml_collections import config_dict
import numpy as np

from mujoco_playground._src.locomotion import torque_penalty


class TorquePenaltyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = config_dict.create(
        scales=config_dict.create(action_rate=-0.1)
    )
    torque_penalty.add_config(self.config)
    model = SimpleNamespace(
        actuator_forcerange=np.array([[-10.0, 10.0], [-20.0, 20.0]])
    )
    self.penalty = torque_penalty.TorquePenalty(self.config, model, 0.02)

  def test_constant_initial_torque_has_no_penalty(self):
    info = {}
    torque = jp.array([1.0, 2.0])
    self.penalty.reset(info, torque)

    high_freq, torque_rate = self.penalty.compute(
        info, torque, jp.zeros_like(torque)
    )

    self.assertAlmostEqual(float(high_freq), 0.0, places=6)
    self.assertAlmostEqual(float(torque_rate), 0.0, places=6)

  def test_torque_change_has_rate_and_highpass_penalty(self):
    info = {}
    self.penalty.reset(info, jp.array([1.0, 2.0]))

    high_freq, torque_rate = self.penalty.compute(
        info, jp.array([2.0, 4.0]), jp.zeros(2)
    )

    self.assertGreater(float(high_freq), 0.0)
    self.assertEqual(float(torque_rate), 5.0)

  def test_enabled_regularizer_disables_action_rate(self):
    self.config.scales.torque_rate = -1e-5
    model = SimpleNamespace(
        actuator_forcerange=np.array([[-10.0, 10.0], [-20.0, 20.0]])
    )

    torque_penalty.TorquePenalty(self.config, model, 0.02)

    self.assertEqual(self.config.scales.action_rate, 0.0)


if __name__ == "__main__":
  absltest.main()
