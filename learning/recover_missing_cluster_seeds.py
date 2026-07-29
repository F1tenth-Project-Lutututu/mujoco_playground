#!/usr/bin/env python3
"""Find missing cluster policy seeds and resubmit only those array indices.

Examples:

  python -m learning.recover_missing_cluster_seeds Go1JoystickRoughTerrain
  python -m learning.recover_missing_cluster_seeds Go1JoystickRoughTerrain \
      --run-pattern '260729-*' --execute

The script is a dry run unless ``--execute`` is supplied.  It reconstructs the
launcher arguments from a present sibling seed's saved ``run_config.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess
from typing import Any, Sequence

from learning import download_models_to_evaluate as downloader


SEEDED_RUN = re.compile(r"(?P<family>.+)-seed(?P<seed>\d+)")
DATED_FAMILY = re.compile(r"(?P<date>\d{6})-.+")


@dataclass(frozen=True)
class MissingSeeds:
  family: str
  missing: tuple[int, ...]
  template_run: str


def _ssh(host: str, command: str) -> str:
  return subprocess.run(
      ("ssh", host, command),
      check=True,
      text=True,
      capture_output=True,
  ).stdout


def _discover(
    run_names: Sequence[str],
    expected_seeds: Sequence[int],
    run_pattern: str,
) -> list[MissingSeeds]:
  families: dict[str, dict[int, str]] = {}
  for run_name in run_names:
    match = SEEDED_RUN.fullmatch(run_name)
    if match is None or not fnmatch.fnmatch(match.group("family"), run_pattern):
      continue
    families.setdefault(match.group("family"), {})[
        int(match.group("seed"))
    ] = run_name

  result = []
  for family, present in sorted(families.items()):
    missing = tuple(seed for seed in expected_seeds if seed not in present)
    if missing:
      template_seed = min(present)
      result.append(
          MissingSeeds(family, missing, present[template_seed])
      )
  return result


def _argument(command: Sequence[str], name: str) -> str:
  for index, value in enumerate(command):
    if value == name and index + 1 < len(command):
      return command[index + 1]
    prefix = f"{name}="
    if value.startswith(prefix):
      return value[len(prefix):]
  raise ValueError(f"Saved command does not contain {name}")


def _launcher_arguments(run_config: dict[str, Any]) -> list[str]:
  """Reconstructs slurm.sh arguments from a saved exact training command."""
  command = run_config["command"]
  environment = _argument(command, "--env_name")
  timesteps = _argument(command, "--num_timesteps")
  overrides = json.loads(_argument(command, "--playground_config_overrides"))

  scales = {
      "ar": overrides.get("reward_config.scales.action_rate"),
      "tr": overrides.get("reward_config.scales.torque_rate"),
      "hp": overrides.get("reward_config.scales.torque_high_freq"),
  }
  active = [(method, value) for method, value in scales.items() if value is not None]
  if len(active) != 1:
    raise ValueError(
        "Expected exactly one supported swept penalty in saved overrides, "
        f"found {active}"
    )
  method, signed_scale = active[0]
  if method == "hp":
    highpass_order = float(
        overrides.get("reward_config.torque_highpass_order", 1.0)
    )
    if highpass_order != 1.0:
      raise ValueError(
          "slurm.sh fixes the high-pass order at 1 and cannot reproduce "
          f"the saved order {highpass_order:g}"
      )
  # Twelve scientific digits are ample for configured penalty sweeps while
  # suppressing binary-float artifacts such as 0.0006 becoming
  # 5.999999999999999e-4 in reconstructed run names.
  mantissa, exponent = f"{abs(float(signed_scale)):.12e}".split("e")
  mantissa = mantissa.rstrip("0").rstrip(".")
  scale = f"{mantissa}e{int(exponent):+d}"
  cutoff = format(
      float(overrides.get("reward_config.torque_highpass_cutoff_hz", 5.0)),
      ".15g",
  )
  # slurm.sh removes the decimal point when constructing the historical
  # difference-order tag: 1.0 -> m10, 2.0 -> m20.
  difference_order = format(
      float(
          overrides.get(
              "reward_config.torque_highpass_difference_order", 1.0
          )
      ),
      ".1f",
  )
  return [method, scale, environment, cutoff, difference_order, timesteps]


def _launcher_family(arguments: Sequence[str], date_prefix: str) -> str:
  """Reproduces the run-family naming rules implemented by slurm.sh."""
  method, scale, _, cutoff, difference_order, timesteps = arguments
  method_name = {
      "ar": "baseline",
      "tr": "torquerate",
      "hp": "highpass",
  }.get(method)
  if method_name is None:
    raise ValueError(f"Unsupported launcher method: {method!r}")
  if re.fullmatch(r"[1-9][0-9]*[Mm]", timesteps):
    timestep_count = int(timesteps[:-1]) * 1_000_000
  elif re.fullmatch(r"[1-9][0-9]*", timesteps):
    timestep_count = int(timesteps)
  else:
    raise ValueError(f"Invalid saved timestep count: {timesteps!r}")
  timestep_tag = f"{timestep_count // 1_000_000}M"

  strength_tag = scale.lower().lstrip("+")
  strength_tag = re.sub(r"e-0*([0-9]+)", r"em\1", strength_tag)
  strength_tag = re.sub(r"e\+0*([0-9]+)", r"ep\1", strength_tag)
  strength_tag = re.sub(r"e0*([0-9]+)", r"ep\1", strength_tag)
  strength_tag = strength_tag.replace(".", "p")
  suffix = ""
  if method == "hp":
    cutoff_tag = re.sub(r"\.0+$", "", cutoff.lower()).replace(".", "")
    difference_tag = difference_order.lower().replace(".", "")
    suffix = f"-f{cutoff_tag}o1m{difference_tag}"
  return (
      f"{date_prefix}-{method_name}-{timestep_tag}-{method}"
      f"{strength_tag}{suffix}"
  )


def _validate_launcher_family(
    family: str, launcher_arguments: Sequence[str]
) -> None:
  """Fails when reconstructed arguments would change the run family."""
  match = DATED_FAMILY.fullmatch(family)
  if match is None:
    raise ValueError(f"Run family lacks a YYMMDD prefix: {family!r}")
  reconstructed = _launcher_family(
      launcher_arguments, match.group("date")
  )
  if reconstructed != family:
    raise ValueError(
        "reconstructed launcher changes the run family: "
        f"expected {family!r}, got {reconstructed!r}"
    )


def _saved_run_config(
    host: str,
    environment_root: PurePosixPath,
    run_name: str,
) -> dict[str, Any]:
  path = environment_root / run_name / "checkpoints" / "run_config.json"
  return json.loads(_ssh(host, f"cat -- {shlex.quote(str(path))}"))


def _parse_seed_spec(value: str) -> tuple[int, ...]:
  seeds = sorted({int(item) for item in value.split(",")})
  if not seeds or any(seed < 0 for seed in seeds):
    raise argparse.ArgumentTypeError("seeds must be comma-separated integers")
  return tuple(seeds)


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("environment")
  parser.add_argument("--host", default=downloader.DEFAULT_HOST)
  parser.add_argument(
      "--remote-logs",
      type=PurePosixPath,
      default=downloader.DEFAULT_REMOTE_LOGS,
  )
  parser.add_argument(
      "--expected-seeds",
      type=_parse_seed_spec,
      default=(0, 1, 2, 3, 4),
      help="Comma-separated expected seeds (default: 0,1,2,3,4).",
  )
  parser.add_argument(
      "--run-pattern",
      default="*",
      help="Shell-style family filter, for example '260729-*'.",
  )
  parser.add_argument(
      "--execute",
      action="store_true",
      help="Submit recovery arrays; default behavior is inspection only.",
  )
  args = parser.parse_args(argv)

  environment = downloader._validate_name(
      args.environment, "environment name"
  )
  host = downloader._validate_name(args.host, "SSH host")
  environment_root = args.remote_logs / environment
  run_names = downloader._remote_run_names(host, environment_root)
  missing_groups = _discover(
      run_names, args.expected_seeds, args.run_pattern
  )
  if not missing_groups:
    print(
        f"No missing seeds found for {environment} matching "
        f"{args.run_pattern!r}."
    )
    return 0

  project_root = args.remote_logs.parent
  submissions: list[tuple[MissingSeeds, list[str]]] = []
  for group in missing_groups:
    try:
      config = _saved_run_config(
          host, environment_root, group.template_run
      )
      launcher_arguments = _launcher_arguments(config)
      _validate_launcher_family(group.family, launcher_arguments)
    except (
        KeyError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
      print(f"{group.family}: cannot reconstruct launcher: {error}")
      continue
    indices = ",".join(map(str, group.missing))
    command = [
        "sbatch",
        f"--array={indices}",
        "slurm.sh",
        *launcher_arguments,
    ]
    submissions.append((group, command))
    print(
        f"{group.family}: missing seeds {indices}; template "
        f"{group.template_run}"
    )
    print("  " + shlex.join(command))

  if not args.execute:
    print("\nDry run only. Add --execute to submit these recovery arrays.")
    return 0

  for group, command in submissions:
    remote_command = (
        f"cd {shlex.quote(str(project_root))} && {shlex.join(command)}"
    )
    output = _ssh(host, remote_command).strip()
    print(f"{group.family}: {output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
