"""Submit/evolve Eagle Pareto evaluations and fetch their metric reports.

``submit`` evaluates policies and retains full trajectory archives on Eagle.
``metrics`` recomputes selected metrics from those archives without policy
replay. ``fetch`` downloads compact reports by default, leaving large signal
archives on the cluster unless ``--include-signals`` is explicitly requested.
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
  if bool(args.environment) == bool(args.manifest):
    raise ValueError(
        "Specify either an environment name or --manifest, but not both."
    )
  if args.manifest is not None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    environment = downloader._validate_name(
        str(manifest["environment"]), "environment name"
    )
    transfer_id = uuid.uuid4().hex
    remote_job = args.remote_project_root / ".pareto-jobs" / transfer_id
    source = remote_job / "pareto_manifest.json"
    _ssh(args.host, f"mkdir -p {shlex.quote(str(remote_job))}")
    subprocess.run(
        (
            "scp",
            str(args.manifest.resolve()),
            f"{args.host}:{shlex.quote(str(source))}",
        ),
        check=True,
    )
  else:
    environment = downloader._validate_name(
        args.environment, "environment name"
    )
    source = environment
  output_root = args.remote_output_root
  submission = shlex.join([
      "sbatch",
      "--parsable",
      str(args.remote_project_root / "slurm_pareto_evaluate.sh"),
      str(source),
      str(args.remote_models_root),
      str(output_root),
      str(args.num_random_tasks),
      str(args.task_seed),
      str(args.run_date or ""),
  ])
  job_id = _ssh(
      args.host,
      f"cd {shlex.quote(str(args.remote_project_root))} && {submission}",
  ).split(";", maxsplit=1)[0]
  print(f"Submitted Eagle Pareto evaluation job {job_id}.")
  print(f"Environment: {environment}")
  print(f"Remote output: {output_root / environment}")
  print(f"Status: ssh {args.host} squeue -j {job_id}")


def metrics(args: argparse.Namespace) -> None:
  """Submits CPU postprocessing over saved batched trajectory archives."""
  environment = downloader._validate_name(args.environment, "environment name")
  evaluation_root = args.remote_output_root / environment
  manifest = evaluation_root / "pareto_manifest.json"
  command = shlex.join([
      "python",
      "-m",
      "learning.recompute_evaluation_metrics",
      "--manifest",
      str(manifest),
      "--evaluation-root",
      str(evaluation_root),
      *[
          argument
          for metric in args.metric
          for argument in ("--metric", metric)
      ],
  ])
  wrapped = (
      f"cd {shlex.quote(str(args.remote_project_root))} && {command}"
  )
  submission = shlex.join([
      "sbatch",
      "--parsable",
      f"--partition={args.partition}",
      "--nodes=1",
      "--ntasks=1",
      f"--cpus-per-task={args.cpus}",
      f"--mem={args.memory}",
      f"--time={args.time}",
      "--job-name=pareto_metrics",
      f"--wrap={wrapped}",
  ])
  job_id = _ssh(args.host, submission).split(";", maxsplit=1)[0]
  print(f"Submitted Eagle metric job {job_id}.")
  print(f"Environment: {environment}")
  print(f"Metrics: {', '.join(args.metric)}")
  print(f"Status: ssh {args.host} squeue -j {job_id}")


def fetch(args: argparse.Namespace) -> None:
  environment = downloader._validate_name(args.environment, "environment name")
  remote_environment = args.remote_output_root / environment
  transfer_id = uuid.uuid4().hex
  remote_transfer = args.remote_output_root / f".pareto-results-{transfer_id}"
  remote_archive = remote_transfer / f"{environment}.tar.gz"
  remote_log = remote_transfer / "tar.log"
  tar_arguments = [
      "tar",
      "-czf",
      str(remote_archive),
  ]
  if not args.include_signals:
    tar_arguments.extend((
        "--exclude=*/signals.npz",
        "--exclude=*/rollout.mp4",
    ))
  tar_arguments.extend([
      "-C",
      str(args.remote_output_root),
      environment,
  ])
  tar_command = shlex.join(tar_arguments)
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
        f"{'Full results' if args.include_signals else 'Metric reports'} "
        "downloaded to: "
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
  submit_parser.add_argument("environment", nargs="?")
  submit_parser.add_argument(
      "--manifest",
      type=Path,
      help="Use an existing local manifest instead of cluster discovery.",
  )
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
  submit_parser.add_argument(
      "--run-date",
      type=int,
      help="Restrict cluster discovery to a YYMMDD run-name prefix.",
  )
  submit_parser.set_defaults(handler=submit)

  metrics_parser = subparsers.add_parser(
      "metrics",
      help="Recompute metrics from full trajectory archives on Eagle.",
  )
  metrics_parser.add_argument("environment")
  metrics_parser.add_argument(
      "--metric",
      action="append",
      default=[],
      choices=(
          "joint_velocity_mssd",
          "joint_velocity_msgfd",
          "smoothness/joint_velocity/"
          "mssd_mean_squared_second_difference_per_dof",
          "smoothness/joint_velocity/"
          "msgfd_mean_absolute_savgol_filter_deviation_per_dof",
      ),
      help="Metric to compute; repeat as needed. Defaults to both.",
  )
  metrics_parser.add_argument(
      "--remote-project-root",
      type=PurePosixPath,
      default=DEFAULT_PROJECT_ROOT,
  )
  metrics_parser.add_argument(
      "--remote-output-root",
      type=PurePosixPath,
      default=DEFAULT_REMOTE_OUTPUT_ROOT,
  )
  metrics_parser.add_argument("--partition", default="standard")
  metrics_parser.add_argument("--cpus", type=int, default=8)
  metrics_parser.add_argument("--memory", default="64G")
  metrics_parser.add_argument("--time", default="04:00:00")
  metrics_parser.set_defaults(handler=metrics)

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
  fetch_parser.add_argument(
      "--include-signals",
      action="store_true",
      help="Also download large signals.npz trajectory archives and videos.",
  )
  fetch_parser.set_defaults(handler=fetch)
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  args = _build_parser().parse_args(argv)
  if args.command == "metrics" and not args.metric:
    from learning import recompute_evaluation_metrics
    args.metric = list(recompute_evaluation_metrics.DEFAULT_METRICS)
  args.handler(args)


if __name__ == "__main__":
  main()
