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

from src.enforce.config import ENFORCEConfig
from src.enforce.fb_inequality_constraints import FischerBurmeisterReformulation
from src.enforce.model import ENFORCE

# ── Helpers ───────────────────────────────────────────────────────────────────


def _scaling(n: int):
    return (torch.zeros(n, dtype=torch.float32), torch.ones(n, dtype=torch.float32))


def _simple_constraint(nc: int):
    """Return a constraint c(x, y) -> [BS, nc] using the first nc outputs."""

    def c(x, y):
        return y[:, :nc]

    return c


def _model_from_cfg(cfg: ENFORCEConfig, nc: int = 1, **kwargs) -> ENFORCE:
    si = _scaling(cfg.input_neurons)
    so = _scaling(cfg.output_neurons)
    c = _simple_constraint(nc)
    return ENFORCE(scaling_input=si, scaling_output=so, c=c, config=cfg, **kwargs)


# ── Canonical configs per benchmark problem ───────────────────────────────────
#
# These dicts mirror the if-elif blocks in configs/config_benchmarking.py and
# are the ground truth that run_benchmark.py's ENFORCEConfig construction must
# reproduce.

PROBLEM_CONFIGS = {
    "function_fitting": dict(
        input_neurons=1,
        hidden_neurons=64,
        output_neurons=2,
        hidden_layers=1,
        training_tolerance=1e-4,
        inference_tolerance=1e-6,
        max_it=100,
        epoch_start_hard_constrained=0,
        ada_np_auto_activation=True,
        supervised=True,
        weight_loss_displacement=0.5,
        soft_constrained=False,
        weight_loss_soft=0.0,
    ),
    "extraction_column": dict(
        input_neurons=3,
        hidden_neurons=64,
        output_neurons=9,
        hidden_layers=1,
        training_tolerance=1e-4,
        inference_tolerance=1e-6,
        max_it=100,
        epoch_start_hard_constrained=0,
        ada_np_auto_activation=True,
        supervised=True,
        weight_loss_displacement=0.5,
        soft_constrained=False,
        weight_loss_soft=0.0,
    ),
    "pooling": dict(
        input_neurons=4,
        hidden_neurons=64,
        output_neurons=5,
        hidden_layers=1,
        training_tolerance=1e-4,
        inference_tolerance=1e-4,
        max_it=100,
        epoch_start_hard_constrained=0,
        ada_np_auto_activation=True,
        supervised=True,
        weight_loss_displacement=0.5,
        soft_constrained=False,
        weight_loss_soft=0.0,
    ),
    "sin_ineq": dict(
        input_neurons=1,
        hidden_neurons=64,
        output_neurons=1,
        hidden_layers=1,
        training_tolerance=1e-4,
        inference_tolerance=1e-4,
        max_it=100,
        epoch_start_hard_constrained=0,
        ada_np_auto_activation=False,  # always project regardless of loss
        supervised=True,
        weight_loss_displacement=0.5,
        soft_constrained=False,
        weight_loss_soft=0.0,
    ),
    "nonconvex_linear": dict(
        input_neurons=50,  # N_CONSTRAINTS_OPT default
        hidden_neurons=200,
        output_neurons=100,  # N_VARIABLES_OPT default
        hidden_layers=2,
        training_tolerance=1e-3,
        inference_tolerance=1e-3,
        max_it=100,
        epoch_start_hard_constrained=0,
        ada_np_auto_activation=True,
        supervised=False,
        weight_loss_displacement=0.5,
        soft_constrained=False,
        weight_loss_soft=0.0,
    ),
    "nonconvex_nonlinear": dict(
        input_neurons=50,
        hidden_neurons=200,
        output_neurons=100,
        hidden_layers=2,
        training_tolerance=1e-3,
        inference_tolerance=1e-3,
        max_it=100,
        epoch_start_hard_constrained=0,
        ada_np_auto_activation=True,
        supervised=False,
        weight_loss_displacement=0.5,
        soft_constrained=False,
        weight_loss_soft=0.0,
    ),
}


# ── Per-problem: config fields are correctly set ──────────────────────────────


class TestPerProblemConfigValues:
    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_config_fields_match_spec(self, problem):
        spec = PROBLEM_CONFIGS[problem]
        cfg = ENFORCEConfig(**spec)
        for field, expected in spec.items():
            actual = getattr(cfg, field)
            assert (
                actual == pytest.approx(expected) if isinstance(expected, float) else actual == expected
            ), f"{problem}: {field} expected {expected!r}, got {actual!r}"

    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_tolerances_positive(self, problem):
        cfg = ENFORCEConfig(**PROBLEM_CONFIGS[problem])
        assert cfg.training_tolerance > 0
        assert cfg.inference_tolerance > 0

    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_max_it_positive(self, problem):
        cfg = ENFORCEConfig(**PROBLEM_CONFIGS[problem])
        assert cfg.max_it > 0

    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_neurons_positive(self, problem):
        cfg = ENFORCEConfig(**PROBLEM_CONFIGS[problem])
        assert cfg.input_neurons > 0
        assert cfg.hidden_neurons > 0
        assert cfg.output_neurons > 0
        assert cfg.hidden_layers >= 1


# ── Per-problem: ENFORCE model has correct architecture ───────────────────────


class TestPerProblemModelArchitecture:
    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_input_layer_matches_config(self, problem):
        spec = PROBLEM_CONFIGS[problem]
        cfg = ENFORCEConfig(**spec)
        nc = max(1, cfg.output_neurons - 1)  # nc < no (underdetermined)
        model = _model_from_cfg(cfg, nc=nc)
        assert model.input_layer.in_features == cfg.input_neurons
        assert model.input_layer.out_features == cfg.hidden_neurons

    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_output_layer_matches_config(self, problem):
        spec = PROBLEM_CONFIGS[problem]
        cfg = ENFORCEConfig(**spec)
        nc = max(1, cfg.output_neurons - 1)
        model = _model_from_cfg(cfg, nc=nc)
        assert model.output_layer.out_features == cfg.output_neurons

    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_ni_no_attributes(self, problem):
        spec = PROBLEM_CONFIGS[problem]
        cfg = ENFORCEConfig(**spec)
        nc = max(1, cfg.output_neurons - 1)
        model = _model_from_cfg(cfg, nc=nc)
        assert model.ni == cfg.input_neurons
        assert model.no == cfg.output_neurons

    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_forward_output_shape(self, problem):
        spec = PROBLEM_CONFIGS[problem]
        cfg = ENFORCEConfig(**spec)
        nc = max(1, cfg.output_neurons - 1)
        model = _model_from_cfg(cfg, nc=nc)
        x = torch.randn(4, cfg.input_neurons)
        out = model.forward(x)
        assert out.shape == (4, cfg.output_neurons)

    @pytest.mark.parametrize("problem", list(PROBLEM_CONFIGS))
    def test_cfg_stored_on_model(self, problem):
        spec = PROBLEM_CONFIGS[problem]
        cfg = ENFORCEConfig(**spec)
        nc = max(1, cfg.output_neurons - 1)
        model = _model_from_cfg(cfg, nc=nc)
        assert model.cfg.input_neurons == cfg.input_neurons
        assert model.cfg.output_neurons == cfg.output_neurons
        assert model.cfg.training_tolerance == pytest.approx(cfg.training_tolerance)
        assert model.cfg.inference_tolerance == pytest.approx(cfg.inference_tolerance)
        assert model.cfg.supervised == cfg.supervised
        assert model.cfg.ada_np_auto_activation == cfg.ada_np_auto_activation


# ── Inequality problems: FB-extended config ───────────────────────────────────


class TestFBProblems:
    """sin_ineq and pooling use FB inequalities; the extended output space
    must be consistent with nc < no."""

    def _make_fb_cfg_and_model(self, n_orig, n_ineq, problem_spec):
        def g_positive(x, y):
            return -y[:, 0]  # g <= 0  <=>  y1 >= 0

        inequalities = [g_positive] * n_ineq
        fb = FischerBurmeisterReformulation(
            n_original_outputs=n_orig,
            inequalities=inequalities,
        )
        cfg = ENFORCEConfig(**problem_spec)
        si = _scaling(cfg.input_neurons)
        so = _scaling(n_orig + n_ineq)  # scaling covers full extended space
        model = ENFORCE(scaling_input=si, scaling_output=so, c=fb, fb=fb, config=cfg)
        return model, fb

    def test_sin_ineq_fb_extension(self):
        """sin_ineq: 1 original output + 2 FB multipliers = 3 total."""
        spec = {**PROBLEM_CONFIGS["sin_ineq"]}
        model, fb = self._make_fb_cfg_and_model(n_orig=1, n_ineq=2, problem_spec=spec)
        assert model.output_layer.out_features == 1  # network predicts y only
        assert model.no == 3  # extended space: y + 2*lambda
        x = torch.randn(4, spec["input_neurons"])
        out = model.forward(x)
        assert out.shape == (4, 3)
        assert torch.all(out[:, 1:] == 0.0)  # lambda columns start as zero

    def test_pooling_fb_extension(self):
        """pooling: 5 original outputs + 2 FB multipliers = 7 total."""
        spec = {**PROBLEM_CONFIGS["pooling"]}
        model, fb = self._make_fb_cfg_and_model(n_orig=5, n_ineq=2, problem_spec=spec)
        assert model.output_layer.out_features == 5
        assert model.no == 7
        x = torch.randn(4, spec["input_neurons"])
        out = model.forward(x)
        assert out.shape == (4, 7)
        assert torch.all(out[:, 5:] == 0.0)

    def test_nc_lt_no_invariant_with_fb(self):
        """ENFORCE must accept any FB config without raising (nc < no always holds)."""
        for n_orig, n_ineq in [(1, 2), (5, 2), (9, 3)]:
            fb = FischerBurmeisterReformulation(
                n_original_outputs=n_orig,
                inequalities=[lambda x, y: -y[:, 0]] * n_ineq,
            )
            cfg = ENFORCEConfig(input_neurons=2, hidden_neurons=16, output_neurons=n_orig)
            si = _scaling(2)
            so = _scaling(n_orig + n_ineq)
            model = ENFORCE(scaling_input=si, scaling_output=so, c=fb, fb=fb, config=cfg)
            assert model.no == n_orig + n_ineq  # extended output space


# ── Live import: current config_benchmarking produces a valid ENFORCEConfig ───


class TestLiveConfigImport:
    """Import the currently-configured benchmarking module and verify the
    resulting ENFORCEConfig is self-consistent.  This test will always run
    against whichever PROBLEM is set in config_benchmarking.py."""

    @pytest.fixture(scope="class")
    def live_cfg(self):
        from src.benchmark_problems.config_benchmarking import (
            ADA_NP_AUTO_ACTIVATION,
            EPOCH_START_HARD_CONSTRAINED,
            HIDDEN_LAYERS,
            HIDDEN_NEURONS,
            IFT_BACKWARD,
            INFERENCE_TOLERANCE,
            INPUT_NEURONS,
            MAX_IT,
            OUTPUT_NEURONS,
            SOFT_CONSTRAINED,
            SUPERVISED,
            TRAINING_TOLERANCE,
            VERBOSE,
            WEIGHT_LOSS_DISPLACEMENT,
            WEIGHT_LOSS_SOFT,
        )

        return ENFORCEConfig(
            input_neurons=INPUT_NEURONS,
            hidden_neurons=HIDDEN_NEURONS,
            output_neurons=OUTPUT_NEURONS,
            hidden_layers=HIDDEN_LAYERS,
            training_tolerance=TRAINING_TOLERANCE,
            inference_tolerance=INFERENCE_TOLERANCE,
            max_it=MAX_IT,
            epoch_start_hard_constrained=EPOCH_START_HARD_CONSTRAINED,
            ada_np_auto_activation=ADA_NP_AUTO_ACTIVATION,
            ift_backward=IFT_BACKWARD,
            supervised=SUPERVISED,
            soft_constrained=SOFT_CONSTRAINED,
            weight_loss_displacement=WEIGHT_LOSS_DISPLACEMENT,
            weight_loss_soft=WEIGHT_LOSS_SOFT,
            verbose=VERBOSE,
        )

    def test_architecture_is_positive(self, live_cfg):
        assert live_cfg.input_neurons > 0
        assert live_cfg.hidden_neurons > 0
        assert live_cfg.output_neurons > 0
        assert live_cfg.hidden_layers >= 1

    def test_tolerances_are_positive(self, live_cfg):
        assert live_cfg.training_tolerance > 0
        assert live_cfg.inference_tolerance > 0
        assert live_cfg.max_it > 0

    def test_loss_weights_non_negative(self, live_cfg):
        assert live_cfg.weight_loss_displacement >= 0
        assert live_cfg.weight_loss_soft >= 0

    def test_model_instantiates(self, live_cfg):
        nc = max(1, live_cfg.output_neurons - 1)
        model = _model_from_cfg(live_cfg, nc=nc)
        assert model.cfg.input_neurons == live_cfg.input_neurons
        assert model.cfg.output_neurons == live_cfg.output_neurons

    def test_model_forward_shape(self, live_cfg):
        nc = max(1, live_cfg.output_neurons - 1)
        model = _model_from_cfg(live_cfg, nc=nc)
        x = torch.randn(4, live_cfg.input_neurons)
        out = model.forward(x)
        assert out.shape == (4, live_cfg.output_neurons)

    def test_live_config_matches_a_canonical_problem(self, live_cfg):
        """The live config's architecture must match exactly one of the known
        canonical problem configurations."""
        from src.benchmark_problems.config_benchmarking import PROBLEM

        if PROBLEM in PROBLEM_CONFIGS:
            spec = PROBLEM_CONFIGS[PROBLEM]
            assert live_cfg.input_neurons == spec["input_neurons"]
            assert live_cfg.hidden_neurons == spec["hidden_neurons"]
            assert live_cfg.output_neurons == spec["output_neurons"]
            assert live_cfg.hidden_layers == spec["hidden_layers"]
            assert live_cfg.training_tolerance == pytest.approx(spec["training_tolerance"])
            assert live_cfg.inference_tolerance == pytest.approx(spec["inference_tolerance"])
            assert live_cfg.ada_np_auto_activation == spec["ada_np_auto_activation"]
            assert live_cfg.supervised == spec["supervised"]
