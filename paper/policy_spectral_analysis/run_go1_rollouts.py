#!/usr/bin/env python3
"""Record torques from a trained locomotion policy on random tracking tasks.

Run this from the repository root on an Eagle GPU node.  The policy network is
restored from its checkpoint, but the rollout environment deliberately uses
MujocoPlayground's current default ``Go1JoystickFlatTerrain`` configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from learning import evaluate_policy

DEFAULT_ENVIRONMENT = "Go1JoystickFlatTerrain"
DEFAULT_CHECKPOINT = Path(
    "/mnt/storage_5/scratch/pl0467-01/pkicki/spectral_playground/logs/"
    "Go1JoystickFlatTerrain/260727-baseline-400M-ar1em1-seed0/"
    "checkpoints/000417792000"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--env-name", default=DEFAULT_ENVIRONMENT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/policy_spectral_analysis/data/go1_default"),
    )
    parser.add_argument("--num-tasks", type=int, default=64)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="Number of 20 ms environment steps per task (default: 1000).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parser().parse_args(argv)
    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be positive.")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be positive.")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    # evaluate_policy handles checkpoint/network restoration, vectorized random
    # environment resets, active masks for early termination, and raw signals.
    # Disabling saved environment configuration is essential here: it evaluates
    # every policy in the current, unmodified MujocoPlayground default setting.
    return evaluate_policy.main(
        [
            "--checkpoint",
            str(args.checkpoint),
            "--env_name",
            args.env_name,
            "--output_dir",
            str(args.output_dir),
            "--num_random_tasks",
            str(args.num_tasks),
            "--task_seed",
            str(args.task_seed),
            "--episode_length",
            str(args.episode_length),
            "--save_signals",
            "--no-use_saved_environment_config",
            "--environment_impl",
            "jax",
            "--deterministic",
            "--disable_perturbations",
            "--require_cuda",
        ]
    )


if __name__ == "__main__":
    main()
