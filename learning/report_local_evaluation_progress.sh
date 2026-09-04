#!/usr/bin/env bash
# Reports rollout-artifact progress for one Pareto evaluation environment.
#
# Run this on Eagle (including an interactive worker shell):
#   bash learning/report_local_evaluation_progress.sh BarkourJoystick
#
# An alternate evaluations root can be supplied as the second argument, which
# makes the script convenient for staging directories and tests.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 ENVIRONMENT [EVALUATIONS_ROOT]" >&2
  exit 2
fi

environment=$1
evaluations_root=${2:-/mnt/storage_3/home/pkicki/pl0467-01/scratch/pkicki/spectral_playground/evaluations/pareto_cluster}
environment_root=$evaluations_root/$environment
raw_torque=$environment_root/raw_torque
manifest=$environment_root/pareto_manifest.json

if [[ ! -d $raw_torque ]]; then
  echo "Evaluation artifact directory not found: $raw_torque" >&2
  exit 1
fi

python3 - "$environment" "$raw_torque" "$manifest" <<'PY'
"""Report the state of the policies named in the evaluation manifest."""

import json
import sys
from pathlib import Path

environment, raw_torque_arg, manifest_arg = sys.argv[1:]
raw_torque = Path(raw_torque_arg)
manifest_path = Path(manifest_arg)

# The artifact layout is raw_torque/<run name>/<checkpoint>/... .  A run is
# complete if any checkpoint evaluation has produced rollouts.csv.
completed = {path.parent.parent.name for path in raw_torque.glob("*/*/rollouts.csv")}
active = {
    path.parent.parent.name
    for path in raw_torque.glob("*/*/.evaluation_in_progress.json")
}

print(f"Environment: {environment}")
if not manifest_path.is_file():
    print(f"Completed:   {len(completed)}")
    print(f"Active:      {len(active - completed)}")
    print("Plan:        unavailable (pareto_manifest.json is absent)")
    sys.exit()

with manifest_path.open(encoding="utf-8") as source:
    manifest = json.load(source)

planned = {
    str(run["run_name"])
    for run in manifest.get("runs", [])
    if isinstance(run, dict) and "run_name" in run
}
skipped = manifest.get("skipped_runs", [])
completed_planned = completed & planned
active_planned = (active & planned) - completed
pending = planned - completed_planned
pending_not_active = pending - active_planned

progress = 100 * len(completed_planned) / len(planned) if planned else 0
print(f"Completed:   {len(completed_planned)}")
print(f"Planned:     {len(planned)}")
print(f"Progress:    {progress:.1f}%")
print(f"Active:      {len(active_planned)}")
print(f"Pending:     {len(pending)}")
print(f"Not started: {len(pending_not_active)}")
print(f"Skipped:     {len(skipped)}")

if pending:
    print("\nPolicies still needing evaluation:")
    for run_name in sorted(pending):
        state = "active" if run_name in active_planned else "not started"
        print(f"  [{state}] {run_name}")

unexpected_completed = completed - planned
if unexpected_completed:
    print(
        f"\nNote: {len(unexpected_completed)} completed artifact(s) are not "
        "in this manifest and are excluded from progress.",
        file=sys.stderr,
    )
PY
