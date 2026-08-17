"""Download the latest checkpoint for every requested policy seed.

The policy families to download are read from
``eagle/<environment>/to_evaluate.txt``.  Each nonempty, non-comment line is a
run name without its trailing ``-seedN``.  Matching runs are discovered on the
cluster and copied to the layout consumed by ``evaluate_all_models.py``.

Example:

  python learning/download_models_to_evaluate.py SpotJoystickGaitTracking
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "eagle"
DEFAULT_REMOTE_LOGS = PurePosixPath(
    "/mnt/storage_3/home/pkicki/pl0467-01/scratch/pkicki/"
    "spectral_playground/logs"
)
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_RETRY_DELAY_SECONDS = 2.0


def _validate_name(value: str, description: str) -> str:
  if not SAFE_NAME.fullmatch(value) or value in (".", ".."):
    raise ValueError(
        f"Invalid {description} {value!r}; use only letters, digits, '.', "
        "'_', and '-'."
    )
  return value


def _read_policy_families(path: Path) -> list[str]:
  if not path.is_file():
    raise FileNotFoundError(f"Evaluation manifest not found: {path}")
  families = []
  for line_number, raw_line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    value = raw_line.split("#", 1)[0].strip()
    if not value:
      continue
    try:
      families.append(_validate_name(value, "policy family"))
    except ValueError as error:
      raise ValueError(f"{path}:{line_number}: {error}") from error
  if not families:
    raise ValueError(f"Evaluation manifest is empty: {path}")
  return list(dict.fromkeys(families))


def _run(
    arguments: Sequence[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      arguments,
      check=True,
      text=True,
      capture_output=capture_output,
  )


def _ssh_lines(host: str, command: str) -> list[str]:
  result = _run(("ssh", host, command), capture_output=True)
  return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _remote_run_names(host: str, environment_root: PurePosixPath) -> list[str]:
  root = shlex.quote(str(environment_root))
  command = (
      f"find {root} -mindepth 1 -maxdepth 1 -type d "
      "-printf '%f\\n'"
  )
  return _ssh_lines(host, command)


def _matching_runs(
    run_names: Sequence[str], families: Sequence[str]
) -> dict[str, list[str]]:
  result = {}
  for family in families:
    pattern = re.compile(rf"{re.escape(family)}-seed(\d+)")
    matches = [name for name in run_names if pattern.fullmatch(name)]
    result[family] = sorted(
        matches, key=lambda name: int(pattern.fullmatch(name).group(1))  # type: ignore[union-attr]
    )
  return result


def _checkpoint_inventory(
    host: str, remote_run: PurePosixPath
) -> tuple[list[str], list[PurePosixPath]]:
  checkpoints = remote_run / "checkpoints"
  quoted_checkpoints = shlex.quote(str(checkpoints))
  names = _ssh_lines(
      host,
      f"find {quoted_checkpoints} -mindepth 1 -maxdepth 1 -type d "
      "-printf '%f\\n'",
  )
  numeric = [name for name in names if name.isdigit()]
  if not numeric:
    raise ValueError(f"No numeric checkpoints found at {host}:{checkpoints}")

  quoted_run = shlex.quote(str(remote_run))
  config_files = _ssh_lines(
      host,
      f"find {quoted_run} -maxdepth 2 -type f "
      "\\( -name run_config.json -o -name config.json \\) -print",
  )
  return numeric, [PurePosixPath(path) for path in config_files]


def _latest_checkpoint(
    host: str, remote_run: PurePosixPath
) -> tuple[str, list[PurePosixPath]]:
  numeric, config_files = _checkpoint_inventory(host, remote_run)
  return max(numeric, key=int), config_files


def _copy_remote(
    host: str,
    remote_path: PurePosixPath,
    local_parent: Path,
    *,
    attempts: int = DOWNLOAD_ATTEMPTS,
    retry_delay_seconds: float = DOWNLOAD_RETRY_DELAY_SECONDS,
) -> None:
  """Copies a remote path atomically, retrying transient transport failures."""
  if attempts <= 0:
    raise ValueError("attempts must be positive")
  local_parent.mkdir(parents=True, exist_ok=True)
  remote = f"{host}:{shlex.quote(str(remote_path))}"
  target = local_parent / remote_path.name

  for attempt in range(1, attempts + 1):
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{remote_path.name}.download-", dir=local_parent)
    )
    try:
      _run(("scp", "-r", remote, str(staging_parent)))
      staged_target = staging_parent / remote_path.name
      if not staged_target.exists():
        raise RuntimeError(
            f"scp reported success but did not create {staged_target}"
        )
      if target.is_dir():
        shutil.rmtree(target)
      elif target.exists() or target.is_symlink():
        target.unlink()
      os.replace(staged_target, target)
      return
    except (subprocess.CalledProcessError, RuntimeError) as error:
      if attempt == attempts:
        raise
      delay = min(retry_delay_seconds * 2 ** (attempt - 1), 30.0)
      print(
          f"Download failed for {remote_path} (attempt {attempt}/{attempts}): "
          f"{error}. Retrying in {delay:g}s...",
          flush=True,
      )
      time.sleep(delay)
    finally:
      shutil.rmtree(staging_parent, ignore_errors=True)


def download(
    environment: str,
    *,
    host: str = DEFAULT_HOST,
    remote_logs: PurePosixPath = DEFAULT_REMOTE_LOGS,
    local_root: Path = PROJECT_ROOT / "eagle",
) -> None:
  environment = _validate_name(environment, "environment name")
  host = _validate_name(host, "SSH host")
  environment_root = remote_logs / environment
  local_environment = local_root / environment
  manifest = local_environment / "to_evaluate.txt"
  families = _read_policy_families(manifest)
  matches = _matching_runs(
      _remote_run_names(host, environment_root), families
  )

  missing = [family for family, runs in matches.items() if not runs]
  if missing:
    raise ValueError(
        "No cluster runs found for: " + ", ".join(missing)
    )

  runs = [run for family in families for run in matches[family]]
  print(
      f"Found {len(runs)} seeded runs from {len(families)} policy families."
  )
  for index, run_name in enumerate(runs, start=1):
    remote_run = environment_root / run_name
    checkpoint, configs = _latest_checkpoint(host, remote_run)
    local_run = local_environment / run_name
    print(f"[{index}/{len(runs)}] {run_name}: checkpoint {checkpoint}")
    _copy_remote(
        host, remote_run / "checkpoints" / checkpoint, local_run / "checkpoints"
    )
    for remote_config in configs:
      relative_parent = remote_config.relative_to(remote_run).parent
      _copy_remote(host, remote_config, local_run / Path(relative_parent))

  print(f"Downloaded models to {local_environment.resolve()}")
  print(
      "They are ready for learning/evaluate_all_models.py with "
      f"ENV_NAME = {environment!r}."
  )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("environment", help="Registered environment name.")
  parser.add_argument("--host", default=DEFAULT_HOST, help="SSH host alias.")
  parser.add_argument(
      "--remote-logs",
      type=PurePosixPath,
      default=DEFAULT_REMOTE_LOGS,
      help="Remote logs root containing one directory per environment.",
  )
  parser.add_argument(
      "--local-root",
      type=Path,
      default=PROJECT_ROOT / "eagle",
      help="Local models root (default: %(default)s).",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  download(
      args.environment,
      host=args.host,
      remote_logs=args.remote_logs,
      local_root=args.local_root,
  )


if __name__ == "__main__":
  main()
