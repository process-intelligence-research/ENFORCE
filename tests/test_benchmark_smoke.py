"""Smoke tests — verify that run_benchmark.main() completes for every problem.

Strategy
--------
run_benchmark.py resolves its config at *import time* via
``from benchmark_problems.config_benchmarking import ...``.  Each test therefore:

1. Injects a minimal ``SimpleNamespace`` mock for the
   ``benchmark_problems.config_benchmarking`` module into ``sys.modules``.
2. Evicts any cached ``run_benchmark`` module so that the fresh import
   picks up the mocked config.
3. Imports ``run_benchmark`` and calls ``main()``.

All tests use N=1, EPOCHS=2, HIDDEN_NEURONS=8, SAVE=False, PLOT=False
so the suite stays fast while still exercising the full code path for
each problem.
"""

import sys
from types import SimpleNamespace

import pytest

from tests.utils import DATA_FILE_PATH_KEYS, skip_if_data_missing

# ── Config factory ────────────────────────────────────────────────────────────


def _cfg(**overrides) -> SimpleNamespace:
    """Return a minimal config namespace for run_benchmark.py."""
    base = dict(
        # Run control
        MODEL="BOTH",
        SAVE=False,
        PLOT=False,
        FIX_SEED=True,
        N=1,
        VERBOSE=False,
        # Backprop
        IFT_BACKWARD=False,
        # Soft constraints
        SOFT_CONSTRAINED=False,
        WEIGHT_LOSS_SOFT=0.0,
        WEIGHT_LOSS_DISPLACEMENT=0.5,
        # Projection
        EPOCH_START_HARD_CONSTRAINED=0,
        TRAINING_TOLERANCE=0.1,
        INFERENCE_TOLERANCE=0.1,
        MAX_IT=10,
        ADA_NP_AUTO_ACTIVATION=False,
        REGULARISE_GRAM=False,
        PROJ_WEIGHTING_OPTION=1,
        # Synthetic data generation (function_fitting only)
        LEFT_LIMIT=-2.0,
        RIGHT_LIMIT=2.0,
        DATA_TRAINING=50,
        DATA_TEST=20,
        # Output
        TRAINING_DIR="training_output",
        # Network (overridden per-problem)
        HIDDEN_NEURONS=8,
        HIDDEN_LAYERS=1,
        BATCH_SIZE=32,
        LEARNING_RATE=1e-3,
        EPOCHS=2,
        # Always imported even when unused
        PARAMS_PATH="unused",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── Per-problem minimal configs ───────────────────────────────────────────────

_CONFIGS: dict[str, SimpleNamespace] = {
    "function_fitting": _cfg(
        PROBLEM="function_fitting",
        INPUT_NEURONS=1,
        OUTPUT_NEURONS=2,
        SUPERVISED=True,
    ),
    "extraction_column": _cfg(
        PROBLEM="extraction_column",
        INPUT_NEURONS=3,
        OUTPUT_NEURONS=9,
        SUPERVISED=True,
        INPUT_DATA_PATH_TRAIN="data/raw/extraction_column/ED_Col_Data_x_train.csv",
        OUTPUT_DATA_PATH_TRAIN="data/raw/extraction_column/ED_Col_Data_y_train.csv",
        INPUT_DATA_PATH_TEST="data/raw/extraction_column/ED_Col_Data_x_test.csv",
        OUTPUT_DATA_PATH_TEST="data/raw/extraction_column/ED_Col_Data_y_test.csv",
    ),
    "pooling": _cfg(
        PROBLEM="pooling",
        INPUT_NEURONS=4,
        OUTPUT_NEURONS=5,
        SUPERVISED=True,
        POOLING_EPS_CHOL=1e-3,
        REGULARISE_GRAM=True,
        INPUT_DATA_PATH_TRAIN="data/raw/pooling/Pooling_dataset_x_train.csv",
        OUTPUT_DATA_PATH_TRAIN="data/raw/pooling/Pooling_dataset_y_train.csv",
        INPUT_DATA_PATH_TEST="data/raw/pooling/Pooling_dataset_x_test.csv",
        OUTPUT_DATA_PATH_TEST="data/raw/pooling/Pooling_dataset_y_test.csv",
    ),
    "sin_ineq": _cfg(
        PROBLEM="sin_ineq",
        INPUT_NEURONS=1,
        OUTPUT_NEURONS=1,
        SUPERVISED=True,
        SIN_INEQ_N_TRAIN=80,
        SIN_INEQ_N_TEST=30,
        SIN_INEQ_LEFT=0.0,
        SIN_INEQ_RIGHT=9.4248,
        SIN_INEQ_NOISE_STD=0.3,
        SIN_INEQ_NOISE_BIAS=0.1,
    ),
    "nonconvex_linear": _cfg(
        PROBLEM="nonconvex_linear",
        INPUT_NEURONS=50,
        OUTPUT_NEURONS=100,
        SUPERVISED=False,
        BATCH_SIZE=512,
        INPUT_DATA_PATH=(
            "data/raw/nonconvex_linear/X_data_random_nonconvex_dataset_var100_ineq0_eq50_ex10000_int-5_5.csv"
        ),
        OUTPUT_DATA_PATH=(
            "data/raw/nonconvex_linear/Y_data_random_nonconvex_dataset_var100_ineq0_eq50_ex10000_int-5_5.csv"
        ),
        PARAMS_PATH=(
            "data/raw/nonconvex_linear/parameters_random_nonconvex_dataset_var100_ineq0_eq50_ex10000_int-5_5.pkl"
        ),
    ),
    "nonconvex_nonlinear": _cfg(
        PROBLEM="nonconvex_nonlinear",
        INPUT_NEURONS=50,
        OUTPUT_NEURONS=100,
        SUPERVISED=False,
        BATCH_SIZE=512,
        INPUT_DATA_PATH=(
            "data/raw/nonconvex_nonlinear/X_data_random_nonlinearnonconvex_dataset_dim100_eq50_ex10000_int-5_5.csv"
        ),
        OUTPUT_DATA_PATH=(
            "data/raw/nonconvex_nonlinear/Y_data_random_nonlinearnonconvex_dataset_dim100_eq50_ex10000_int-5_5.csv"
        ),
        PARAMS_PATH=(
            "data/raw/nonconvex_nonlinear/parameters_random_nonlinearnonconvex_dataset_dim100_eq50_ex10000_int-5_5.pkl"
        ),
    ),
}


# ── Smoke tests ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("problem", list(_CONFIGS.keys()))
def test_benchmark_smoke(monkeypatch, problem):
    """run_benchmark.main() must complete without raising for every problem."""
    cfg = _CONFIGS[problem]
    skip_if_data_missing([getattr(cfg, k, None) for k in DATA_FILE_PATH_KEYS], problem)

    # 1. Replace the config module with our minimal mock.
    monkeypatch.setitem(sys.modules, "benchmark_problems.config_benchmarking", cfg)

    # 2. Evict any cached run_benchmark so the fresh import picks up the mock.
    monkeypatch.delitem(sys.modules, "run_benchmark", raising=False)

    # 3. Import and run.  monkeypatch restores sys.modules on teardown.
    import scripts.run_benchmark as run_benchmark

    run_benchmark.main()
