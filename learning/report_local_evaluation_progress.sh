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

count_unique_runs() {
  local filename=$1
  find "$raw_torque" -mindepth 3 -maxdepth 3 -type f -name "$filename" \
    -printf '%P\n' |
    cut -d/ -f1 |
    sort -u |
    sed '/^$/d' |
    wc -l
}

completed=$(count_unique_runs rollouts.csv)
active=$(count_unique_runs .evaluation_in_progress.json)

if [[ -f $manifest ]]; then
  read -r planned skipped < <(
    python3 -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
  manifest = json.load(source)
print(len(manifest.get("runs", [])), len(manifest.get("skipped_runs", [])))
' "$manifest"
  )
else
  planned=0
  skipped=0
fi

if (( planned > 0 )); then
  percent=$(awk -v done="$completed" -v total="$planned" \
    'BEGIN { printf "%.1f", 100 * done / total }')
  remaining=$((planned - completed))
  if (( remaining < 0 )); then
    remaining=0
  fi
else
  percent="n/a"
  remaining="n/a"
fi

printf 'Environment: %s\n' "$environment"
printf 'Completed:   %s\n' "$completed"
printf 'Planned:     %s\n' "$planned"
printf 'Progress:    %s%%\n' "$percent"
printf 'Active:      %s\n' "$active"
printf 'Skipped:     %s\n' "$skipped"
printf 'Remaining:   %s\n' "$remaining"

if (( planned == 0 )); then
  echo "Note: pareto_manifest.json is absent; planned progress is unavailable." >&2
fi
