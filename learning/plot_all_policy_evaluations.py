"""Compare every policy evaluation found for one environment.

Example:

  python learning/plot_all_policy_evaluations.py \
    SpotJoystickGaitTracking --normalization raw_torque
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from learning import compare_policy_evaluations as comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _policy_family(run_name: str) -> str:
  """Removes the seed suffix used to group independently trained policies."""
  return re.sub(r"-seed\d+$", "", run_name)


def _first_manifest_family(path: Path) -> str:
  """Returns the first policy family listed in ``to_evaluate.txt``."""
  if not path.is_file():
    raise FileNotFoundError(f"Evaluation manifest not found: {path}")
  for raw_line in path.read_text(encoding="utf-8").splitlines():
    family = raw_line.split("#", 1)[0].strip()
    if family:
      return family
  raise ValueError(f"Evaluation manifest is empty: {path}")


def discover_methods(evaluation_directory: Path) -> dict[str, Path]:
  """Finds one seed template for every completed policy family."""
  if not evaluation_directory.is_dir():
    raise FileNotFoundError(
        f"Evaluation directory does not exist: {evaluation_directory}"
    )
  families: dict[str, list[Path]] = {}
  for run in sorted(evaluation_directory.iterdir()):
    if not run.is_dir() or re.fullmatch(r".+-seed\d+", run.name) is None:
      continue
    if not any(run.glob("*/summary.json")):
      continue
    families.setdefault(_policy_family(run.name), []).append(run)
  if len(families) < 2:
    raise ValueError(
        f"Need at least two completed policy families in "
        f"{evaluation_directory}; found {len(families)}."
    )
  return {
      family: sorted(paths, key=lambda path: path.name)[0]
      for family, paths in families.items()
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("environment", help="Environment evaluation directory.")
  parser.add_argument(
      "--normalization",
      choices=("capacity_normalized", "raw_torque"),
      default="raw_torque",
  )
  parser.add_argument(
      "--reference",
      help=(
          "Reference family label (default: first entry in to_evaluate.txt)."
      ),
  )
  parser.add_argument(
      "--evaluations-root",
      type=Path,
      default=PROJECT_ROOT / "evaluations",
  )
  parser.add_argument(
      "--models-root",
      type=Path,
      default=PROJECT_ROOT / "eagle",
      help="Root containing <environment>/to_evaluate.txt.",
  )
  parser.add_argument(
      "--output-directory",
      type=Path,
      help="Default: <evaluation directory>/comparison",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  evaluation_directory = (
      args.evaluations_root / args.environment / args.normalization
  )
  methods = discover_methods(evaluation_directory)
  manifest = args.models_root / args.environment / "to_evaluate.txt"
  reference = args.reference or _first_manifest_family(manifest)
  if reference not in methods:
    raise ValueError(
        f"Reference {reference!r} was not found. Available families: "
        + ", ".join(methods)
    )
  output = args.output_directory or evaluation_directory / "comparison"
  output.mkdir(parents=True, exist_ok=True)

  comparison.METHODS = methods
  comparison.PAIRED_REFERENCE = reference
  comparison.TITLE = f"{args.environment} policy comparison"
  comparison.PAIRED_TITLE = f"{args.environment} paired per-task differences"
  comparison.PAIRED_PERCENT_TITLE = (
      f"{args.environment} paired per-task improvement/deterioration"
  )
  comparison.OUTPUT = output / "evaluation_comparison.png"
  comparison.CSV_OUTPUT = None
  comparison.PAIRED_OUTPUT = output / "evaluation_comparison_paired.png"
  comparison.PAIRED_CSV_OUTPUT = None
  comparison.PAIRED_PERCENT_OUTPUT = (
      output / "evaluation_comparison_paired_percent.png"
  )
  comparison.PAIRED_PERCENT_CSV_OUTPUT = None

  print(
      f"Comparing {len(methods)} policy families from "
      f"{evaluation_directory.resolve()}; reference: {reference}"
  )
  comparison.compare()


if __name__ == "__main__":
  main()
