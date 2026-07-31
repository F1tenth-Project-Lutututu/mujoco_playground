"""Tests for incomplete cluster training recovery."""

from unittest import mock

from absl.testing import absltest

from learning import recover_incomplete_cluster_runs as recovery


class RecoverIncompleteClusterRunsTest(absltest.TestCase):

  def test_discovers_below_threshold_runs_and_chooses_template(self):
    inventory = {
        "260728-baseline-400M-ar1em1-seed0": (27_852_800,),
        "260728-baseline-400M-ar1em1-seed1": (417_792_000,),
        "260728-highpass-400M-hp1em1-f5o1m10-seed0": (),
        "unrelated": (),
    }

    result = recovery._discover(inventory, 400_000_000, "260728-*")

    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].seed, 0)
    self.assertEqual(result[0].latest_checkpoint, 27_852_800)
    self.assertEqual(
        result[0].template_run,
        "260728-baseline-400M-ar1em1-seed1",
    )
    self.assertEqual(
        result[1].template_run,
        "260728-highpass-400M-hp1em1-f5o1m10-seed0",
    )

  def test_archive_command_is_recoverable_and_seed_specific(self):
    run = recovery.IncompleteRun(
        "260728-baseline-400M-ar1em1",
        "260728-baseline-400M-ar1em1-seed3",
        3,
        27_852_800,
        "260728-baseline-400M-ar1em1-seed0",
    )

    command = recovery._archive_and_submit_command(
        recovery.PurePosixPath("/project"),
        recovery.PurePosixPath("/logs/Env"),
        run,
        ["ar", "1e-1", "Env", "5", "1.0", "400000000"],
    )

    self.assertIn("mv --", command)
    self.assertIn("/logs/Env/.incomplete-runs/", command)
    self.assertIn("sbatch --array=3 slurm.sh", command)

  @mock.patch.object(recovery.seed_recovery, "_saved_run_config")
  @mock.patch.object(recovery, "_checkpoint_inventory")
  def test_main_is_dry_run_by_default(self, inventory, saved_config):
    inventory.return_value = {
        "260728-baseline-400M-ar1em1-seed0": (27_852_800,),
        "260728-baseline-400M-ar1em1-seed1": (417_792_000,),
    }
    saved_config.return_value = {
        "command": [
            "train-jax-ppo",
            "--num_timesteps=400000000",
            "--env_name=TestEnvironment",
            "--playground_config_overrides="
            '{"reward_config.scales.action_rate": -0.1}',
        ]
    }
    with mock.patch.object(recovery.seed_recovery, "_ssh") as ssh:
      result = recovery.main(["TestEnvironment"])

    self.assertEqual(result, 0)
    ssh.assert_not_called()


if __name__ == "__main__":
  absltest.main()
