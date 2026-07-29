"""Submit Eagle Pareto evaluation jobs and fetch their results as tar archives.

Run this module on localhost. The ``submit`` command uploads a local manifest
and starts one H100-oriented Slurm job. The ``fetch`` command packages the
remote environment results on a CPU worker, downloads one archive, and safely
extracts it below the local output root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import tempfile
from typing import Sequence
import uuid

from learning import download_models_to_evaluate as downloader


DEFAULT_PROJECT_ROOT = downloader.DEFAULT_REMOTE_LOGS.parent
DEFAULT_REMOTE_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "evaluations" / "pareto_cluster"
DEFAULT_LOCAL_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "evaluations" / "pareto_cluster"


def _ssh(host: str, command: str) -> str:
  return subprocess.run(
      ("ssh", host, command),
      check=True,
      text=True,
      capture_output=True,
  ).stdout.strip()


def submit(args: argparse.Namespace) -> None:
  manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
  environment = downloader._validate_name(
      str(manifest["environment"]), "environment name"
  )
  transfer_id = uuid.uuid4().hex
  remote_job = args.remote_project_root / ".pareto-jobs" / transfer_id
  remote_manifest = remote_job / "pareto_manifest.json"
  _ssh(args.host, f"mkdir -p {shlex.quote(str(remote_job))}")
  subprocess.run(
      (
          "scp",
          str(args.manifest.resolve()),
          f"{args.host}:{shlex.quote(str(remote_manifest))}",
      ),
      check=True,
  )
  output_root = args.remote_output_root
  submission = shlex.join([
      "sbatch",
      "--parsable",
      str(args.remote_project_root / "slurm_pareto_evaluate.sh"),
      str(remote_manifest),
      str(args.remote_models_root),
      str(output_root),
      str(args.num_random_tasks),
      str(args.task_seed),
  ])
  job_id = _ssh(
      args.host,
      f"cd {shlex.quote(str(args.remote_project_root))} && {submission}",
  ).split(";", maxsplit=1)[0]
  print(f"Submitted Eagle Pareto evaluation job {job_id}.")
  print(f"Environment: {environment}")
  print(f"Remote output: {output_root / environment}")
  print(f"Status: ssh {args.host} squeue -j {job_id}")


def fetch(args: argparse.Namespace) -> None:
  environment = downloader._validate_name(args.environment, "environment name")
  remote_environment = args.remote_output_root / environment
  transfer_id = uuid.uuid4().hex
  remote_transfer = args.remote_output_root / f".pareto-results-{transfer_id}"
  remote_archive = remote_transfer / f"{environment}.tar.gz"
  remote_log = remote_transfer / "tar.log"
  tar_command = shlex.join([
      "tar",
      "-czf",
      str(remote_archive),
      "-C",
      str(args.remote_output_root),
      environment,
  ])
  _ssh(args.host, f"mkdir -p {shlex.quote(str(remote_transfer))}")
  try:
    if _ssh(
        args.host,
        f"test -d {shlex.quote(str(remote_environment))} && echo yes",
    ) != "yes":
      raise FileNotFoundError(
          f"Remote evaluation directory does not exist: {remote_environment}"
      )
    submission = shlex.join([
        "sbatch",
        "--wait",
        "--parsable",
        f"--partition={args.archive_partition}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=01:00:00",
        "--job-name=pareto_results",
        f"--output={remote_log}",
        f"--wrap={tar_command}",
    ])
    job_id = _ssh(args.host, submission)
    print(f"Packaged results in Eagle Slurm job {job_id}.")
    args.local_output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pareto-results-", dir=args.local_output_root
    ) as temporary_directory:
      temporary = Path(temporary_directory)
      downloader._copy_remote(args.host, remote_archive, temporary)
      local_archive = temporary / remote_archive.name
      with tarfile.open(local_archive, "r:gz") as archive:
        archive.extractall(args.local_output_root, filter="data")
    print(
        "Results downloaded to: "
        f"{(args.local_output_root / environment).resolve()}"
    )
  finally:
    subprocess.run(
        (
            "ssh",
            args.host,
            f"rm -rf -- {shlex.quote(str(remote_transfer))}",
        ),
        check=False,
    )


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default=downloader.DEFAULT_HOST)
  subparsers = parser.add_subparsers(dest="command", required=True)

  submit_parser = subparsers.add_parser("submit")
  submit_parser.add_argument("--manifest", type=Path, required=True)
  submit_parser.add_argument(
      "--remote-project-root",
      type=PurePosixPath,
      default=DEFAULT_PROJECT_ROOT,
  )
  submit_parser.add_argument(
      "--remote-models-root",
      type=PurePosixPath,
      default=downloader.DEFAULT_REMOTE_LOGS,
  )
  submit_parser.add_argument(
      "--remote-output-root",
      type=PurePosixPath,
      default=DEFAULT_REMOTE_OUTPUT_ROOT,
  )
  submit_parser.add_argument("--num-random-tasks", type=int, default=2048)
  submit_parser.add_argument("--task-seed", type=int, default=0)
  submit_parser.set_defaults(handler=submit)

  fetch_parser = subparsers.add_parser("fetch")
  fetch_parser.add_argument("environment")
  fetch_parser.add_argument(
      "--remote-output-root",
      type=PurePosixPath,
      default=DEFAULT_REMOTE_OUTPUT_ROOT,
  )
  fetch_parser.add_argument(
      "--local-output-root",
      type=Path,
      default=DEFAULT_LOCAL_OUTPUT_ROOT,
  )
  fetch_parser.add_argument("--archive-partition", default="standard")
  fetch_parser.set_defaults(handler=fetch)
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  args.handler(args)


if __name__ == "__main__":
  main()
