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
    np.testing.assert_array_equal(info["torque_for_spectrum"], torque)

    high_freq, torque_rate = self.penalty.compute(
        info, torque, jp.zeros_like(torque)
    )

    self.assertAlmostEqual(float(high_freq), 0.0, places=6)
    self.assertAlmostEqual(float(torque_rate), 0.0, places=6)
    np.testing.assert_array_equal(info["torque_for_spectrum"], torque)

  def test_torque_change_has_rate_and_highpass_penalty(self):
    info = {}
    self.penalty.reset(info, jp.array([1.0, 2.0]))

    high_freq, torque_rate = self.penalty.compute(
        info, jp.array([2.0, 4.0]), jp.zeros(2)
    )

    self.assertGreater(float(high_freq), 0.0)
    self.assertEqual(float(torque_rate), 5.0)

  def test_torque_smoothness_uses_squared_second_difference(self):
    self.config.torque_rate_use_second_difference = True
    model = SimpleNamespace(
        actuator_forcerange=np.array([[-10.0, 10.0], [-20.0, 20.0]])
    )
    penalty = torque_penalty.TorquePenalty(self.config, model, 0.02)
    info = {}
    penalty.reset(info, jp.array([1.0, 2.0]))
    penalty.compute(info, jp.array([2.0, 4.0]), jp.zeros(2))

    _, smoothness = penalty.compute(
        info, jp.array([4.0, 7.0]), jp.zeros(2)
    )

    # Rate: ||[2, 3]||^2 = 13; second difference: ||[1, 1]||^2 = 2.
    self.assertEqual(float(smoothness), 15.0)

  def test_torque_smoothness_observes_two_torque_samples(self):
    self.config.torque_rate_observe_state = True
    self.config.torque_rate_use_second_difference = True
    model = SimpleNamespace(
        actuator_forcerange=np.array([[-10.0, 10.0], [-20.0, 20.0]])
    )
    penalty = torque_penalty.TorquePenalty(self.config, model, 0.02)
    info = {}
    penalty.reset(info, jp.array([1.0, 2.0]))
    penalty.compute(info, jp.array([2.0, 4.0]), jp.zeros(2))

    observation = penalty.observation(info, jp.array([2.0, 4.0]))

    np.testing.assert_array_equal(observation, [2.0, 4.0, 1.0, 2.0])

  def test_enabled_regularizer_disables_action_rate(self):
    self.config.scales.torque_rate = -1e-5
    model = SimpleNamespace(
        actuator_forcerange=np.array([[-10.0, 10.0], [-20.0, 20.0]])
    )

    torque_penalty.TorquePenalty(self.config, model, 0.02)

    self.assertEqual(self.config.scales.action_rate, 0.0)

  def test_default_tfr_memory_does_not_change_baseline_observation(self):
    info = {}
    torque = jp.array([1.0, 2.0])
    self.penalty.reset(info, torque)

    self.assertEqual(self.penalty.observation(info, torque).shape, (0,))

  def test_default_tfr_memory_is_observed_when_tfr_is_enabled(self):
    self.config.scales.torque_high_freq = -1e-5
    model = SimpleNamespace(
        actuator_forcerange=np.array([[-10.0, 10.0], [-20.0, 20.0]])
    )
    penalty = torque_penalty.TorquePenalty(self.config, model, 0.02)
    info = {}
    torque = jp.array([1.0, 2.0])
    penalty.reset(info, torque)

    self.assertGreater(penalty.observation(info, torque).size, 0)

  def test_uses_joint_force_limits_when_actuator_limits_are_unset(self):
    model = SimpleNamespace(
        actuator_forcerange=np.zeros((2, 2)),
        actuator_trnid=np.array([[1, -1], [0, -1]]),
        jnt_actfrcrange=np.array([[-20.0, 20.0], [-10.0, 10.0]]),
    )

    penalty = torque_penalty.TorquePenalty(self.config, model, 0.02)

    np.testing.assert_array_equal(penalty.capacity, [10.0, 20.0])


if __name__ == "__main__":
  absltest.main()
