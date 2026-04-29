"""Benchmark reproducibility tests.

These tests run full benchmark training (slow — minutes to hours per problem) and
verify that results match the saved baselines in tests/fixtures/benchmark_baselines/.

Workflow
--------
1. Run a reference training for each problem with the default config::

       python scripts/run_benchmark.py   # set PROBLEM in config_benchmarking.py

2. Register those results as fixtures::

       python scripts/register_baselines.py

3. From now on, this suite guards against regressions::

       pytest -m slow tests/test_benchmark_reproducibility.py

A test is skipped (not failed) when no fixture exists for that problem yet, so
adding a new problem doesn't immediately break CI.

Config contract
---------------
Reference runs were produced with ``FIX_SEED=False, N=1`` and the module-level
``_DATA_SEED = 41`` in run_benchmark.py.  The first call to
``random.randint(0, 100000)`` after ``random.seed(41)`` always yields **49941**,
so every re-run produces the same weight initialisation and data shuffling.  The
tests inject this exact config (``FIX_SEED=False, N=1``) to reproduce it.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.utils import DATA_FILE_PATH_KEYS, skip_if_data_missing

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "benchmark_baselines"

# Columns excluded from comparison: timing, per-run metadata, and projection
# iteration count (which varies naturally with model convergence behaviour).
# Note: column name differs between CSVs — "projection iterations" in metrics,
# "projection_iterations" in per-epoch losses.
_SKIP_COLS = frozenset(
    [
        "training_time",
        "inference time",
        "seed",
        "projection iterations",
        "projection_iterations",
    ]
)

# ── Full per-problem configs ──────────────────────────────────────────────────
# These mirror config_benchmarking.py exactly, with four test overrides:
#   N=1, FIX_SEED=False, SAVE=True, PLOT=False
# TRAINING_DIR is set per-test to pytest's tmp_path.

_BASE = dict(
    MODEL="BOTH",
    N=1,
    FIX_SEED=False,
    SAVE=True,
    PLOT=False,
    VERBOSE=False,
    IFT_BACKWARD=False,
    REGULARISE_GRAM=False,
    SOFT_CONSTRAINED=False,
    WEIGHT_LOSS_SOFT=0.0,
    DATA_TRAINING=1000,
    DATA_TEST=100000,
    LEFT_LIMIT=-2.0,
    RIGHT_LIMIT=2.0,
    N_VARIABLES_OPT=100,
    N_CONSTRAINTS_OPT=50,
    PARAMS_PATH="unused",
)

_PROBLEM_CONFIGS: dict[str, dict] = {
    "function_fitting": {
        **_BASE,
        "PROBLEM": "function_fitting",
        "WEIGHT_LOSS_DISPLACEMENT": 0.5,
        "EPOCH_START_HARD_CONSTRAINED": 0,
        "TRAINING_TOLERANCE": 0.0001,
        "INFERENCE_TOLERANCE": 1e-6,
        "MAX_IT": 100,
        "PROJ_WEIGHTING_OPTION": 5,
        "ADA_NP_AUTO_ACTIVATION": True,
        "INPUT_NEURONS": 1,
        "HIDDEN_NEURONS": 64,
        "OUTPUT_NEURONS": 2,
        "HIDDEN_LAYERS": 1,
        "SUPERVISED": True,
        "EPOCHS": 50000,
        "LEARNING_RATE": 0.001,
        "BATCH_SIZE": 1000,
    },
    "extraction_column": {
        **_BASE,
        "PROBLEM": "extraction_column",
        "WEIGHT_LOSS_DISPLACEMENT": 0.5,
        "EPOCH_START_HARD_CONSTRAINED": 0,
        "TRAINING_TOLERANCE": 0.0001,
        "INFERENCE_TOLERANCE": 1e-6,
        "MAX_IT": 100,
        "PROJ_WEIGHTING_OPTION": 5,
        "ADA_NP_AUTO_ACTIVATION": True,
        "INPUT_NEURONS": 3,
        "HIDDEN_NEURONS": 64,
        "OUTPUT_NEURONS": 9,
        "HIDDEN_LAYERS": 1,
        "SUPERVISED": True,
        "EPOCHS": 1200,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 64,
        "INPUT_DATA_PATH_TRAIN": "data/raw/extraction_column/ED_Col_Data_x_train.csv",
        "OUTPUT_DATA_PATH_TRAIN": "data/raw/extraction_column/ED_Col_Data_y_train.csv",
        "INPUT_DATA_PATH_TEST": "data/raw/extraction_column/ED_Col_Data_x_test.csv",
        "OUTPUT_DATA_PATH_TEST": "data/raw/extraction_column/ED_Col_Data_y_test.csv",
    },
    "pooling": {
        **_BASE,
        "PROBLEM": "pooling",
        "WEIGHT_LOSS_DISPLACEMENT": 0.5,
        "EPOCH_START_HARD_CONSTRAINED": 0,
        "TRAINING_TOLERANCE": 1e-4,
        "INFERENCE_TOLERANCE": 1e-4,
        "MAX_IT": 100,
        "PROJ_WEIGHTING_OPTION": 1,
        "ADA_NP_AUTO_ACTIVATION": True,
        "POOLING_EPS_CHOL": 1e-3,
        "REGULARISE_GRAM": True,
        "INPUT_NEURONS": 4,
        "HIDDEN_NEURONS": 64,
        "OUTPUT_NEURONS": 5,
        "HIDDEN_LAYERS": 1,
        "SUPERVISED": True,
        "EPOCHS": 10000,
        "LEARNING_RATE": 0.5e-4,
        "BATCH_SIZE": 64,
        "INPUT_DATA_PATH_TRAIN": "data/raw/pooling/Pooling_dataset_x_train.csv",
        "OUTPUT_DATA_PATH_TRAIN": "data/raw/pooling/Pooling_dataset_y_train.csv",
        "INPUT_DATA_PATH_TEST": "data/raw/pooling/Pooling_dataset_x_test.csv",
        "OUTPUT_DATA_PATH_TEST": "data/raw/pooling/Pooling_dataset_y_test.csv",
    },
    "sin_ineq": {
        **_BASE,
        "PROBLEM": "sin_ineq",
        "WEIGHT_LOSS_DISPLACEMENT": 0.5,
        "EPOCH_START_HARD_CONSTRAINED": 0,
        "TRAINING_TOLERANCE": 1e-4,
        "INFERENCE_TOLERANCE": 1e-4,
        "MAX_IT": 100,
        "PROJ_WEIGHTING_OPTION": 1,
        "ADA_NP_AUTO_ACTIVATION": False,
        "INPUT_NEURONS": 1,
        "HIDDEN_NEURONS": 64,
        "OUTPUT_NEURONS": 1,
        "HIDDEN_LAYERS": 1,
        "SUPERVISED": True,
        "EPOCHS": 500,
        "LEARNING_RATE": 1e-3,
        "BATCH_SIZE": 128,
        "SIN_INEQ_N_TRAIN": 1200,
        "SIN_INEQ_N_TEST": 300,
        "SIN_INEQ_LEFT": 0.0,
        "SIN_INEQ_RIGHT": 9.4248,
        "SIN_INEQ_NOISE_STD": 0.3,
        "SIN_INEQ_NOISE_BIAS": 0.1,
    },
    "nonconvex_linear": {
        **_BASE,
        "PROBLEM": "nonconvex_linear",
        "WEIGHT_LOSS_DISPLACEMENT": 0.5,
        "EPOCH_START_HARD_CONSTRAINED": 0,
        "TRAINING_TOLERANCE": 1e-3,
        "INFERENCE_TOLERANCE": 1e-3,
        "MAX_IT": 100,
        "PROJ_WEIGHTING_OPTION": 1,
        "ADA_NP_AUTO_ACTIVATION": True,
        "INPUT_NEURONS": 50,
        "HIDDEN_NEURONS": 200,
        "OUTPUT_NEURONS": 100,
        "HIDDEN_LAYERS": 2,
        "SUPERVISED": False,
        "EPOCHS": 1000,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 200,
        "INPUT_DATA_PATH": (
            "data/raw/nonconvex_linear/X_data_random_nonconvex_dataset_var100_ineq0_eq50_ex10000_int-5_5.csv"
        ),
        "OUTPUT_DATA_PATH": (
            "data/raw/nonconvex_linear/Y_data_random_nonconvex_dataset_var100_ineq0_eq50_ex10000_int-5_5.csv"
        ),
        "PARAMS_PATH": (
            "data/raw/nonconvex_linear/parameters_random_nonconvex_dataset_var100_ineq0_eq50_ex10000_int-5_5.pkl"
        ),
    },
    "nonconvex_nonlinear": {
        **_BASE,
        "PROBLEM": "nonconvex_nonlinear",
        "WEIGHT_LOSS_DISPLACEMENT": 0.5,
        "EPOCH_START_HARD_CONSTRAINED": 0,
        "TRAINING_TOLERANCE": 1e-3,
        "INFERENCE_TOLERANCE": 1e-3,
        "MAX_IT": 100,
        "PROJ_WEIGHTING_OPTION": 1,
        "ADA_NP_AUTO_ACTIVATION": True,
        "INPUT_NEURONS": 50,
        "HIDDEN_NEURONS": 200,
        "OUTPUT_NEURONS": 100,
        "HIDDEN_LAYERS": 2,
        "SUPERVISED": False,
        "EPOCHS": 1000,
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 200,
        "INPUT_DATA_PATH": (
            "data/raw/nonconvex_nonlinear/X_data_random_nonlinearnonconvex_dataset_dim100_eq50_ex10000_int-5_5.csv"
        ),
        "OUTPUT_DATA_PATH": (
            "data/raw/nonconvex_nonlinear/Y_data_random_nonlinearnonconvex_dataset_dim100_eq50_ex10000_int-5_5.csv"
        ),
        "PARAMS_PATH": (
            "data/raw/nonconvex_nonlinear/parameters_random_nonlinearnonconvex_dataset_dim100_eq50_ex10000_int-5_5.pkl"
        ),
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_run_dir(base: Path, problem: str) -> Path:
    # nonconvex problems write to an extra subdirectory (nonconvex/nonlinear)
    runs = sorted((base / problem).glob("**/run_*"))
    assert runs, f"No run_* directory created under {base / problem}"
    return runs[-1]


def _read_results(run_dir: Path) -> dict[str, dict]:
    """Read metrics and final-epoch losses for each mode from a completed run."""
    results: dict[str, dict] = {}
    for mode in ("constrained", "unconstrained"):
        metrics_path = run_dir / f"{mode}_metrics_runs.csv"
        losses_path = run_dir / "losses" / f"{mode}_losses_run_1.csv"
        if metrics_path.exists():
            row = pd.read_csv(metrics_path).iloc[0]
            results[f"{mode}_metrics"] = {k: float(v) for k, v in row.items() if k not in _SKIP_COLS}
        if losses_path.exists():
            last = pd.read_csv(losses_path).iloc[-1]
            results[f"{mode}_final_loss"] = {k: float(v) for k, v in last.items() if k not in _SKIP_COLS}
    return results


# Columns skipped only for unconstrained (MLP) sections: the MLP does not enforce
# constraints, so its residuals and violation metrics vary freely and are not
# meaningful reproducibility signals.
_SKIP_UNCONSTRAINED = frozenset(["residual_avg", "residual_max"])


def _is_constraint_col(key: str) -> bool:
    return any(p in key for p in ("ineq_", "infeasible", "violation"))


def _assert_close(actual: dict, expected: dict, label: str, unconstrained: bool = False) -> None:
    skip_extra = _SKIP_UNCONSTRAINED if unconstrained else frozenset()

    # Fail if the baseline records a column that the fresh run dropped (regression signal).
    # Do NOT fail if the fresh run has extra columns — new metrics may be added over time.
    missing = (expected.keys() - actual.keys()) - skip_extra
    assert not missing, f"[{label}] column mismatch — missing from fresh run: {missing}"

    print(f"\n  [{label}]")
    failures = []
    for key, exp in expected.items():
        if key in skip_extra:
            print(f"    {'SKIP':<6}  {key}")
            continue
        if unconstrained and _is_constraint_col(key):
            print(f"    {'SKIP':<6}  {key}  (MLP constraint metric, not meaningful)")
            continue
        act = actual[key]
        # MAPE > 1 (>100%) means the output has near-zero true values: the metric
        # explodes with tiny prediction differences and is not a useful regression
        # signal in that range. Skip rather than generate spurious failures.
        if key.startswith("mape_") and abs(exp) > 1.0:
            print(f"    {'SKIP':<6}  {key:<45}  (MAPE>100%, near-zero denominator)")
            continue
        # rel=0.05 catches >5% regressions in quality metrics (R², MSE, MAPE, etc.).
        # abs=0.01 floor covers near-zero constraint residuals and violation metrics
        # that may show sub-tolerance numerical noise between code versions.
        ok = act == pytest.approx(exp, rel=0.05, abs=0.01)
        if exp != 0:
            rel_diff = abs(act - exp) / abs(exp) * 100
            diff_str = f"  rel={rel_diff:.2f}%"
        else:
            diff_str = f"  abs_diff={abs(act - exp):.2e}"
        status = "OK" if ok else "FAIL"
        print(f"    {status:<6}  {key:<45}  got={act:.6g}  exp={exp:.6g}{diff_str}")
        if not ok:
            failures.append(f"  {key}: got {act!r}, expected {exp!r}")
    assert not failures, f"[{label}] value mismatch:\n" + "\n".join(failures)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("problem", list(_PROBLEM_CONFIGS.keys()))
def test_benchmark_reproducibility(monkeypatch, tmp_path, problem):
    """Re-run full training and assert results match the saved baseline fixture."""
    skip_if_data_missing([_PROBLEM_CONFIGS[problem].get(k) for k in DATA_FILE_PATH_KEYS], problem)

    fixture_path = FIXTURES_DIR / f"{problem}.json"
    if not fixture_path.exists():
        pytest.skip(
            f"No baseline fixture for '{problem}'. "
            f"Run a reference training, then: python scripts/register_baselines.py {problem}"
        )

    baseline = json.loads(fixture_path.read_text())

    # Inject config: full real hyperparameters, tmp_path as output directory.
    cfg_dict = {**_PROBLEM_CONFIGS[problem], "TRAINING_DIR": str(tmp_path)}
    cfg = SimpleNamespace(**cfg_dict)
    monkeypatch.setitem(sys.modules, "benchmark_problems.config_benchmarking", cfg)
    monkeypatch.delitem(sys.modules, "scripts.run_benchmark", raising=False)
    monkeypatch.delitem(sys.modules, "run_benchmark", raising=False)

    import scripts.run_benchmark as run_benchmark

    run_benchmark.main()

    run_dir = _find_run_dir(tmp_path, problem)
    results = _read_results(run_dir)

    # Only compare test-set evaluation metrics — not per-epoch training losses.
    # Final-epoch training loss depends heavily on the training trajectory
    # (Ada-NP activation pattern, numerical path) and varies across library
    # versions even when the model generalises identically.
    for section in ("constrained_metrics", "unconstrained_metrics"):
        if section not in baseline:
            continue
        assert section in results, f"Section '{section}' present in baseline but missing from fresh run"
        _assert_close(
            results[section],
            baseline[section],
            f"{problem}/{section}",
            unconstrained="unconstrained" in section,
        )
