"""
Regenerate the sin_ineq comparison plot combining ENFORCE, MLP, and Soft results.

Sources
-------
  ENFORCE + MLP predictions : training_output/sin_ineq/sin_ineq/run_20260313_162904
  Soft (lambda_C=1)          : training_output/sin_ineq/sin_ineq/run_20260313_165942

The output figure is saved alongside the original in the ENFORCE+MLP run directory.

Usage (from repo root):
    python scripts/plot_sin_ineq_all.py
"""

import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from src.visualization.plot_benchmarking import plot_sin_ineq_results

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 18,
    }
)


# Make sure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Paths ────────────────────────────────────────────────────────────────────
RUN_MAIN = os.path.join("training_output", "sin_ineq", "sin_ineq", "run_20260313_162904")
RUN_SOFT = os.path.join("training_output", "sin_ineq", "sin_ineq", "run_20260313_165942")
OUT_DIR = RUN_MAIN

# ── Data-generation parameters (must match configs/config.py) ────────────────
DATA_SEED = 41  # fixed at module level in main.py
N_TRAIN = 1200
N_TEST = 300
LEFT = 0.0
RIGHT = 9.4248  # ~3*pi
NOISE_STD = 0.3
NOISE_BIAS = 0.1


def _amplitude(x: np.ndarray) -> np.ndarray:
    return 1.0 + x**2 / (3.0 * math.pi**2)


def _true_fn(x: np.ndarray) -> np.ndarray:
    return _amplitude(x) * np.sin(x)


# ── Regenerate training data with same global seed ────────────────────────────
np.random.seed(DATA_SEED)

train_x = np.random.uniform(LEFT, RIGHT, (N_TRAIN, 1)).astype(np.float32)
f_train = _true_fn(train_x)
noise_bias = NOISE_BIAS * np.sign(f_train)
noise = (np.random.normal(0.0, NOISE_STD, (N_TRAIN, 1)) + noise_bias).astype(np.float32)
train_outputs = (f_train + noise).astype(np.float32)
train_inputs = train_x


# ── Load test data and predictions from CSVs ─────────────────────────────────
def _load_preds(run_dir: str, mode: str) -> np.ndarray:
    """Return predictions array shape [1, N_TEST, 1] (1 run)."""
    path = os.path.join(run_dir, "test_predictions", f"{mode}_test_predictions_run_1.csv")
    df = pd.read_csv(path)
    return df[["y1_pred"]].to_numpy(dtype=np.float32)[np.newaxis]  # [1, N, 1]


def _load_before(run_dir: str, mode: str) -> np.ndarray:
    path = os.path.join(run_dir, "test_predictions", f"{mode}_test_predictions_run_1.csv")
    df = pd.read_csv(path)
    return df[["y1_pred_before"]].to_numpy(dtype=np.float32)[np.newaxis]


def _load_test_data(run_dir: str, mode: str):
    path = os.path.join(run_dir, "test_dataset", f"{mode}_test_data.csv")
    df = pd.read_csv(path)
    x = df[["x1"]].to_numpy(dtype=np.float32)
    y = df[["y1_true"]].to_numpy(dtype=np.float32)
    return x, y


test_inputs, test_outputs = _load_test_data(RUN_MAIN, "constrained")

test_predictions_list_dict = {
    "constrained": _load_preds(RUN_MAIN, "constrained"),
    "unconstrained": _load_preds(RUN_MAIN, "unconstrained"),
    "soft": _load_preds(RUN_SOFT, "unconstrained"),  # soft run saved as "unconstrained"
}

test_prediction_before_projection_list_dict = {
    "constrained": _load_before(RUN_MAIN, "constrained"),
}

# ── Call the shared plot function ─────────────────────────────────────────────

plot_sin_ineq_results(
    train_inputs=train_inputs,
    train_outputs=train_outputs,
    test_inputs=test_inputs,
    test_outputs=test_outputs,
    test_predictions_list_dict=test_predictions_list_dict,
    test_prediction_before_projection_list_dict=test_prediction_before_projection_list_dict,
    output_dir=OUT_DIR,
)

out_path = os.path.join(OUT_DIR, "plots", "sin_ineq_results.png")
print(f"Saved: {out_path}")
plt.close("all")
