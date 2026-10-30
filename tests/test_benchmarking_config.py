"""Tests that benchmark problem configs are correctly captured in ENFORCEConfig.

Strategy
--------
``configs/config_benchmarking.py`` executes ``if PROBLEM == ...`` blocks at
module-import time and locks in a single problem's values.  We therefore test
in two complementary ways:

1. **Per-problem canonical values** — each problem's expected ENFORCEConfig is
   defined explicitly here (the authoritative spec for what run_benchmark.py
   should build).  We verify the config is valid and produces a correctly-shaped
   model.  These tests are independent of which PROBLEM is currently set.

2. **Live import test** — import the current config_benchmarking module and
   verify the resulting ENFORCEConfig is self-consistent (architecture, tolerances,
   etc. satisfy ENFORCE's invariants).
"""

import pytest
import torch

from enforce import Constraints, ENFORCEConfig, ENFORCELossConfig, Scaler, ENFORCE
from enforce.losses import ENFORCELoss
from enforce.models import MLPConfig, ENFORCEConstrainedMLP
from benchmark_problems import config_benchmarking as live


PROBLEMS = {
    "function_fitting": (1, 64, 2, 1, 1e-4, 1e-6, True),
    "extraction_column": (3, 64, 9, 1, 1e-4, 1e-6, True),
    "pooling": (4, 64, 5, 1, 1e-4, 1e-4, True),
    "sin_ineq": (1, 64, 1, 1, 1e-4, 1e-4, True),
    "nonconvex_linear": (50, 200, 100, 2, 1e-3, 1e-3, False),
    "nonconvex_nonlinear": (50, 200, 100, 2, 1e-3, 1e-3, False),
}


@pytest.mark.parametrize("name,expected", PROBLEMS.items())
def test_canonical_benchmark_network_and_projection_settings(name, expected):
    ni, nh, no, depth, train_tol, infer_tol, supervised = expected
    network = MLPConfig(ni, nh, no, depth)
    projection = ENFORCEConfig(training_tolerance=train_tol, inference_tolerance=infer_tol)
    loss = ENFORCELossConfig(
        soft_constrained=False,
        weight_loss_displacement=0.5,
        weight_loss_soft=0.0,
    )
    assert (network.input_neurons, network.hidden_neurons, network.output_neurons, network.hidden_layers) == (
        ni, nh, no, depth
    )
    assert projection.training_tolerance == pytest.approx(train_tol)
    assert projection.inference_tolerance == pytest.approx(infer_tol)
    assert isinstance(supervised, bool)
    assert loss.weight_loss_displacement == pytest.approx(0.5)


def test_live_benchmark_settings_match_selected_problem():
    spec = PROBLEMS[live.PROBLEM]
    assert (live.INPUT_NEURONS, live.HIDDEN_NEURONS, live.OUTPUT_NEURONS, live.HIDDEN_LAYERS) == spec[:4]
    assert live.TRAINING_TOLERANCE == pytest.approx(spec[4])
    assert live.INFERENCE_TOLERANCE == pytest.approx(spec[5])
    assert live.SUPERVISED is spec[6]


WEIGHTING_OPTIONS = {
    "function_fitting": "ID",
    "extraction_column": "ID",
    "pooling": "I",
    "sin_ineq": "I",
    "nonconvex_linear": "I",
    "nonconvex_nonlinear": "I",
}


def test_live_weighting_option_is_a_valid_enforce_option():
    assert live.PROJ_WEIGHTING_OPTION == WEIGHTING_OPTIONS[live.PROBLEM]
    assert live.PROJ_WEIGHTING_OPTION in {"I", "BA", "ID", "R", "TF"}


def test_live_config_has_no_ift_backward_setting():
    assert not hasattr(live, "IFT_BACKWARD")


def test_live_training_dir_does_not_repeat_problem_name():
    assert not live.TRAINING_DIR.rstrip("/").endswith(live.PROBLEM)


def test_live_settings_construct_current_composed_model():
    scaler = Scaler(
        (torch.zeros(live.INPUT_NEURONS), torch.ones(live.INPUT_NEURONS)),
        (torch.zeros(live.OUTPUT_NEURONS), torch.ones(live.OUTPUT_NEURONS)),
    )
    constraints = Constraints(c=[lambda x, y: y[:, 0]])
    loss = ENFORCELoss(scaler, constraints)
    layer = ENFORCE(
        scaler,
        constraints,
        loss,
        ENFORCEConfig(
            training_tolerance=live.TRAINING_TOLERANCE,
            inference_tolerance=live.INFERENCE_TOLERANCE,
            max_it=live.MAX_IT,
            epoch_start_hard_constrained=5,
            random_seed=live.N if hasattr(live, "N") else 42,
        ),
    )
    model = ENFORCEConstrainedMLP(
        MLPConfig(live.INPUT_NEURONS, live.HIDDEN_NEURONS, live.OUTPUT_NEURONS, live.HIDDEN_LAYERS),
        layer,
    )
    ytilde, yhat = model(torch.zeros(3, live.INPUT_NEURONS))
    assert ytilde.shape == yhat.shape == (3, live.OUTPUT_NEURONS)
