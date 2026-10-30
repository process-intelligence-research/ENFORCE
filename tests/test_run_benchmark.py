"""Unit tests for the helpers in scripts/run_benchmark.py.

run_benchmark.py reads its config at import time, so each test injects a mock
``benchmark_problems.config_benchmarking`` module and re-imports the script
(same strategy as test_benchmark_smoke.py). No benchmark data is loaded.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from enforce import Constraints, ENFORCELoss
from enforce.engines.train import TrainingConfig
from enforce.losses import ENFORCESSLLoss, MSELoss, SoftConstrainedLoss, SoftConstrainedSSLLoss, SSLLossCore


def _cfg(**overrides) -> SimpleNamespace:
    base = dict(
        MODEL="BOTH",
        SAVE=False,
        PLOT=False,
        FIX_SEED=True,
        N=1,
        VERBOSE=False,
        SOFT_CONSTRAINED=False,
        WEIGHT_LOSS_SOFT=0.0,
        WEIGHT_LOSS_DISPLACEMENT=0.5,
        EPOCH_START_HARD_CONSTRAINED=0,
        TRAINING_TOLERANCE=0.1,
        INFERENCE_TOLERANCE=0.1,
        MAX_IT=10,
        ADA_NP_AUTO_ACTIVATION=False,
        REGULARISE_GRAM=False,
        PROJ_WEIGHTING_OPTION="I",
        LEFT_LIMIT=-2.0,
        RIGHT_LIMIT=2.0,
        DATA_TRAINING=50,
        DATA_TEST=20,
        TRAINING_DIR="training_output",
        INPUT_NEURONS=2,
        HIDDEN_NEURONS=8,
        OUTPUT_NEURONS=3,
        HIDDEN_LAYERS=1,
        BATCH_SIZE=32,
        LEARNING_RATE=1e-3,
        EPOCHS=2,
        PARAMS_PATH="unused",
        PROBLEM="function_fitting",
        SUPERVISED=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _import_run_benchmark(monkeypatch, **overrides):
    monkeypatch.setitem(sys.modules, "benchmark_problems.config_benchmarking", _cfg(**overrides))
    monkeypatch.delitem(sys.modules, "scripts.run_benchmark", raising=False)
    return importlib.import_module("scripts.run_benchmark")


def _problem_data(rb, constraints, ssl_loss=None):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(8, 2)), rng.normal(size=(8, 3))
    return rb.ProblemData(
        train_inputs=x,
        train_outputs=y,
        test_inputs=x,
        test_outputs=y,
        train_outputs_orig=y,
        test_outputs_orig=y,
        constraints=constraints,
        ssl_loss=ssl_loss,
    )


@pytest.mark.parametrize("option", ["I", "BA", "ID", "R", "TF"])
def test_weighting_option_is_passed_through_unchanged(monkeypatch, option):
    rb = _import_run_benchmark(monkeypatch, PROJ_WEIGHTING_OPTION=option)
    assert rb._build_enforce_config(seed=0).weighting_option == option


def test_enforce_config_does_not_enable_ift_backward(monkeypatch):
    rb = _import_run_benchmark(monkeypatch)
    assert rb._build_enforce_config(seed=0).ift_backward is False


def _supervised_models(rb):
    data = _problem_data(rb, Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]]))
    tensors = rb._prepare_tensors(data, device="cpu")
    unconstrained = rb._build_model(data, tensors, constrained=False, seed=0, device="cpu")
    constrained = rb._build_model(data, tensors, constrained=True, seed=0, device="cpu")
    return data, unconstrained, constrained


def test_soft_baseline_uses_soft_constrained_loss(monkeypatch):
    rb = _import_run_benchmark(monkeypatch, SOFT_CONSTRAINED=True, WEIGHT_LOSS_SOFT=0.7)
    _, unconstrained, constrained = _supervised_models(rb)
    assert isinstance(unconstrained.loss, SoftConstrainedLoss)
    assert unconstrained.loss.weight_loss_soft == pytest.approx(0.7)
    assert isinstance(constrained.loss, ENFORCELoss)
    assert constrained.loss.cfg.soft_constrained is True


def test_plain_baseline_uses_mse_without_soft_constrained(monkeypatch):
    rb = _import_run_benchmark(monkeypatch, SOFT_CONSTRAINED=False, WEIGHT_LOSS_SOFT=0.7)
    _, unconstrained, _ = _supervised_models(rb)
    assert type(unconstrained.loss) is MSELoss


def test_training_config_has_no_soft_settings(monkeypatch):
    rb = _import_run_benchmark(monkeypatch, SOFT_CONSTRAINED=True, WEIGHT_LOSS_SOFT=0.7)
    assert rb._build_training_config(seed=0) == TrainingConfig(
        batch_size=32, epochs=2, learning_rate=1e-3, random_seed=0
    )


def test_unconstrained_ssl_baseline_uses_plain_ssl_loss(monkeypatch):
    rb = _import_run_benchmark(monkeypatch, SUPERVISED=False)
    data = _problem_data(
        rb,
        Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]]),
        ssl_loss=lambda x, y: torch.mean(y.square()),
    )
    tensors = rb._prepare_tensors(data, device="cpu")
    model = rb._build_model(data, tensors, constrained=False, seed=0, device="cpu")
    assert isinstance(model.loss, SSLLossCore)
    assert not isinstance(model.loss, ENFORCESSLLoss)
    assert model.loss.weight_loss_soft == 0.0

    constrained = rb._build_model(data, tensors, constrained=True, seed=0, device="cpu")
    assert isinstance(constrained.loss, ENFORCESSLLoss)


def test_unconstrained_soft_ssl_baseline_uses_soft_constrained_ssl_loss(monkeypatch):
    rb = _import_run_benchmark(monkeypatch, SUPERVISED=False, SOFT_CONSTRAINED=True, WEIGHT_LOSS_SOFT=0.4)
    data = _problem_data(
        rb,
        Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]]),
        ssl_loss=lambda x, y: torch.mean(y.square()),
    )
    tensors = rb._prepare_tensors(data, device="cpu")
    model = rb._build_model(data, tensors, constrained=False, seed=0, device="cpu")
    assert isinstance(model.loss, SoftConstrainedSSLLoss)
    assert model.loss.weight_loss_soft == pytest.approx(0.4)


@pytest.mark.parametrize("problem,supervised", [("pooling", True), ("nonconvex_linear", False)])
def test_output_dir_contains_problem_name_once(monkeypatch, tmp_path, problem, supervised):
    rb = _import_run_benchmark(
        monkeypatch, SAVE=True, TRAINING_DIR=str(tmp_path), PROBLEM=problem, SUPERVISED=supervised
    )
    output_dir = rb._setup_output_dir("20260101_000000")
    assert output_dir == Path(tmp_path) / problem / "run_20260101_000000"


def test_build_model_returns_setup_without_attaching_attributes(monkeypatch):
    rb = _import_run_benchmark(monkeypatch, SOFT_CONSTRAINED=True, WEIGHT_LOSS_SOFT=0.7)
    _, unconstrained, constrained = _supervised_models(rb)
    for setup in (unconstrained, constrained):
        assert isinstance(setup, rb.BenchmarkSetup)
        for name in ("scaler", "constraints", "loss_fn"):
            assert not hasattr(setup.model, name)
    # The plain backbone must not carry the loss as a submodule
    assert all(module is not unconstrained.loss for module in unconstrained.model.modules())


@pytest.mark.parametrize("constrained", [False, True])
def test_saved_model_is_a_loadable_state_dict(monkeypatch, tmp_path, constrained):
    rb = _import_run_benchmark(monkeypatch)
    # Lambda constraints cannot be pickled: saving the whole model object would fail
    data = _problem_data(rb, Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]]))
    tensors = rb._prepare_tensors(data, device="cpu")
    setup = rb._build_model(data, tensors, constrained=constrained, seed=0, device="cpu")
    path = tmp_path / "model.pth"
    rb._save_model(setup, path)

    state = torch.load(path, weights_only=True)
    assert state.keys() == setup.model.state_dict().keys()
    fresh = rb._build_model(data, tensors, constrained=constrained, seed=1, device="cpu")
    fresh.model.load_state_dict(state)
    for a, b in zip(setup.model.parameters(), fresh.model.parameters()):
        assert torch.equal(a, b)
