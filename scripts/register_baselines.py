"""Populate tests/fixtures/benchmark_baselines/ from existing training_output runs.

Run this once after a reference training run to commit the baseline fixtures that
tests/test_benchmark_reproducibility.py will compare against.

Usage
-----
    python scripts/register_baselines.py                        # all problems
    python scripts/register_baselines.py function_fitting       # specific problems
    python scripts/register_baselines.py pooling sin_ineq
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "benchmark_baselines"

# For each problem, the subdirectory inside training_output/ where run_* dirs live.
_PROBLEM_DIRS: dict[str, Path] = {
    "function_fitting": ROOT / "training_output" / "function_fitting",
    "extraction_column": ROOT / "training_output" / "extraction_column" / "extraction_column",
    "pooling": ROOT / "training_output" / "pooling" / "pooling",
    "sin_ineq": ROOT / "training_output" / "sin_ineq" / "sin_ineq",
    "nonconvex_linear": ROOT / "training_output" / "50" / "nonconvex_linear",
    "nonconvex_nonlinear": ROOT / "training_output" / "50" / "nonconvex_nonlinear",
}

# Columns excluded from reproducibility comparison.
# "projection iterations" varies naturally with model quality and Newton convergence.
# Note: column name differs between CSVs — "projection iterations" in metrics,
# "projection_iterations" in per-epoch losses.
_SKIP_COLS = frozenset([
    "training_time", "inference time", "seed",
    "projection iterations", "projection_iterations",
])


def _latest_full_run(problem_dir: Path) -> Path | None:
    """Return the most recent run that has both constrained and unconstrained results."""
    runs = sorted(problem_dir.glob("run_*"))
    full = [
        r for r in runs
        if (r / "constrained_metrics_runs.csv").exists()
        and (r / "unconstrained_metrics_runs.csv").exists()
    ]
    if full:
        return full[-1]
    partial = [r for r in runs if (r / "constrained_metrics_runs.csv").exists()]
    return partial[-1] if partial else None


def _read_run(run_dir: Path) -> dict:
    result: dict = {}
    for mode in ("constrained", "unconstrained"):
        metrics_path = run_dir / f"{mode}_metrics_runs.csv"
        losses_path = run_dir / "losses" / f"{mode}_losses_run_1.csv"

        if metrics_path.exists():
            row = pd.read_csv(metrics_path).iloc[0]
            result["seed"] = int(row["seed"]) if "seed" in row.index else None
            result[f"{mode}_metrics"] = {
                k: float(v) for k, v in row.items() if k not in _SKIP_COLS
            }

        if losses_path.exists():
            last = pd.read_csv(losses_path).iloc[-1]
            result[f"{mode}_final_loss"] = {
                k: float(v) for k, v in last.items() if k not in _SKIP_COLS
            }

    return result


def register(problems: list[str]) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for problem in problems:
        problem_dir = _PROBLEM_DIRS.get(problem)
        if problem_dir is None or not problem_dir.exists():
            print(f"SKIP  {problem}: {problem_dir} not found")
            continue

        run_dir = _latest_full_run(problem_dir)
        if run_dir is None:
            print(f"SKIP  {problem}: no run with both constrained/unconstrained metrics in {problem_dir}")
            continue

        data = _read_run(run_dir)
        if not data:
            print(f"SKIP  {problem}: no metric CSVs found in {run_dir}")
            continue

        payload = {"problem": problem, "source_run": run_dir.name, **data}
        fixture_path = FIXTURES_DIR / f"{problem}.json"
        fixture_path.write_text(json.dumps(payload, indent=2))
        seed_info = f"  seed={data.get('seed', '?')}"
        print(f"OK    {problem}: {run_dir.name}{seed_info} -> {fixture_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problems", nargs="*", help="Problems to register (default: all)")
    args = parser.parse_args()
    register(args.problems or list(_PROBLEM_DIRS.keys()))
