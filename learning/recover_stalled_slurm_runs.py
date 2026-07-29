#!/usr/bin/env python3
"""Find array tasks that fell back to CPU and resubmit only their seeds.

The script is dry-run by default.  Run it on the cluster from the repository
checkout, for example:

  python learning/recover_stalled_slurm_runs.py 7829274
  python learning/recover_stalled_slurm_runs.py 7829274 --execute

The original ``sbatch`` command is read from Slurm accounting.  The new array
keeps the failed array indices, which are also the seeds used by our launchers.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


GPU_FAILURE_PATTERNS = (
    re.compile(r"CUDA_ERROR_NO_DEVICE", re.IGNORECASE),
    re.compile(r"falling back to cpu", re.IGNORECASE),
    re.compile(r"jax\.default_backend\(\).*cpu", re.IGNORECASE),
    re.compile(r"no usable gpu", re.IGNORECASE),
)
ACTIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}


@dataclass(frozen=True)
class FailedTask:
  array_index: int
  job_id: str
  state: str
  stderr_path: Path
  reason: str


def _run(command: list[str], *, cwd: Path | None = None) -> str:
  return subprocess.run(
      command,
      cwd=cwd,
      check=True,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  ).stdout


def _accounting_rows(array_job_id: str) -> list[tuple[str, str, str]]:
  output = _run([
      "sacct",
      "-X",
      "-j",
      array_job_id,
      "--noheader",
      "--parsable2",
      "--format=JobID,State,SubmitLine%1000",
  ])
  rows = []
  for line in output.splitlines():
    fields = line.split("|", maxsplit=2)
    if len(fields) == 3:
      rows.append((fields[0].strip(), fields[1].strip(), fields[2].strip()))
  return rows


def _job_properties(job_id: str) -> dict[str, str]:
  # Slurm paths used here do not contain whitespace.  StdErr and WorkDir are
  # therefore safely recoverable from the one-line representation.
  output = _run(["scontrol", "show", "job", "--oneliner", job_id])
  return dict(re.findall(r"(?:^| )(\w+)=(\S+)", output))


def _failure_reason(stderr_path: Path) -> str | None:
  try:
    text = stderr_path.read_text(errors="replace")
  except OSError as error:
    print(f"warning: cannot read {stderr_path}: {error}", file=sys.stderr)
    return None
  for pattern in GPU_FAILURE_PATTERNS:
    match = pattern.search(text)
    if match:
      return match.group(0)
  return None


def find_failed_tasks(array_job_id: str) -> list[FailedTask]:
  """Returns active array tasks whose stderr proves that no GPU was used."""
  failed = []
  task_pattern = re.compile(rf"^{re.escape(array_job_id)}_(\d+)$")
  for job_id, state, _ in _accounting_rows(array_job_id):
    match = task_pattern.fullmatch(job_id)
    if match is None or state.split()[0] not in ACTIVE_STATES:
      continue
    properties = _job_properties(job_id)
    stderr_value = properties.get("StdErr")
    if not stderr_value:
      print(f"warning: Slurm did not report StdErr for {job_id}", file=sys.stderr)
      continue
    stderr_path = Path(stderr_value)
    reason = _failure_reason(stderr_path)
    if reason:
      failed.append(
          FailedTask(
              array_index=int(match.group(1)),
              job_id=job_id,
              state=state,
              stderr_path=stderr_path,
              reason=reason,
          )
      )
  return sorted(failed, key=lambda task: task.array_index)


def _original_submission(array_job_id: str) -> tuple[list[str], Path]:
  rows = _accounting_rows(array_job_id)
  submit_line = next(
      (submit for job_id, _, submit in rows if job_id == array_job_id and submit),
      "",
  )
  if not submit_line:
    # Some Slurm versions repeat the submission only on array child records.
    submit_line = next((submit for _, _, submit in rows if submit), "")
  if not submit_line:
    raise RuntimeError(f"Slurm did not retain SubmitLine for job {array_job_id}")

  command = shlex.split(submit_line)
  if not command or Path(command[0]).name != "sbatch":
    raise RuntimeError(f"Unexpected SubmitLine: {submit_line!r}")

  properties = _job_properties(array_job_id)
  work_dir = Path(properties["WorkDir"])
  return command, work_dir


def _without_array_option(command: list[str]) -> list[str]:
  result = []
  skip_next = False
  for argument in command:
    if skip_next:
      skip_next = False
      continue
    if argument == "--array":
      skip_next = True
    elif not argument.startswith("--array="):
      result.append(argument)
  return result


def main() -> int:
  parser = argparse.ArgumentParser(
      description=(
          "Detect active tasks in a Slurm array that fell back to CPU, then "
          "cancel and resubmit only their array indices/seeds."
      )
  )
  parser.add_argument("job_id", help="Parent Slurm array job ID")
  parser.add_argument(
      "--execute",
      action="store_true",
      help="Actually cancel and resubmit (the default only prints the actions)",
  )
  args = parser.parse_args()

  failed = find_failed_tasks(args.job_id)
  if not failed:
    print(f"No active GPU-fallback tasks found in array {args.job_id}.")
    return 0

  indices = ",".join(str(task.array_index) for task in failed)
  cancel_ids = [task.job_id for task in failed]
  original_command, work_dir = _original_submission(args.job_id)
  submit_command = _without_array_option(original_command)
  submit_command.insert(1, f"--array={indices}")

  print("Detected GPU-fallback tasks:")
  for task in failed:
    print(
        f"  seed {task.array_index}: {task.job_id} ({task.state}), "
        f"{task.reason!r} in {task.stderr_path}"
    )
  print(f"Working directory: {work_dir}")
  print("Cancel:", shlex.join(["scancel", *cancel_ids]))
  print("Resubmit:", shlex.join(submit_command))

  if not args.execute:
    print("\nDry run only. Add --execute to perform these actions.")
    return 0

  _run(["scancel", *cancel_ids])
  submission = _run(submit_command, cwd=work_dir).strip()
  print(submission)
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except subprocess.CalledProcessError as error:
    detail = error.stderr.strip() if error.stderr else str(error)
    print(f"Slurm command failed: {detail}", file=sys.stderr)
    raise SystemExit(1) from error
  except RuntimeError as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(1) from error
