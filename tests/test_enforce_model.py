"""Tests for ENFORCE model instantiation and forward behaviour.

Design notes
------------
- Constraints must satisfy nc < no (underdetermined) or nc == no (determined,
  which triggers a warning but is allowed).
- Do NOT wrap model.predict() calls in torch.no_grad(): the Newton projection
  requires autograd to compute constraint Jacobians.
- Forward-only tests (no projection) use model.forward() directly or
  constrained=False so no Jacobian is needed.
"""

import torch
import torch.nn as nn

from enforce.core.config import ENFORCEConfig
from enforce.core.fb_inequality_constraints import FischerBurmeisterReformulation
from enforce.core.model import ENFORCE

# ── Helpers ──────────────────────────────────────────────────────────────────


def _scaling(n: int, device="cpu"):
    """Return (mean, std) tensors of size n: mean=0, std=1."""
    return (
        torch.zeros(n, dtype=torch.float32, device=device),
        torch.ones(n, dtype=torch.float32, device=device),
    )


def _linear_constraint(nc: int):
    """Return a constraint c(x, y) -> [BS, nc] that is always zero at y=0.

    Uses the first nc outputs so we can easily control how many constraints
    we get for a model with no outputs.
    """

    def c(x, y):
        return y[:, :nc]  # [BS, nc]

    return c


def _make_model(cfg: ENFORCEConfig, nc: int = 1, **kwargs) -> ENFORCE:
    """Instantiate an ENFORCE model with trivial scaling and a linear constraint."""
    si = _scaling(cfg.input_neurons)
    so = _scaling(cfg.output_neurons)
    c = _linear_constraint(nc)
    return ENFORCE(scaling_input=si, scaling_output=so, c=c, config=cfg, **kwargs)


# ── Config forwarding ─────────────────────────────────────────────────────────


class TestConfigStoredOnModel:
    def test_cfg_attribute_is_set(self):
        cfg = ENFORCEConfig(input_neurons=2, hidden_neurons=32, output_neurons=4)
        model = _make_model(cfg, nc=2)
        assert model.cfg is cfg

    def test_default_config_used_when_none_passed(self):
        si = _scaling(1)
        so = _scaling(1)
        c = _linear_constraint(1)
        # No config= kwarg → should fall back to ENFORCEConfig()
        model = ENFORCE(scaling_input=si, scaling_output=so, c=c)
        assert isinstance(model.cfg, ENFORCEConfig)
        assert model.cfg == ENFORCEConfig()

    def test_ift_backward_from_config(self):
        cfg = ENFORCEConfig(ift_backward=True)
        model = _make_model(cfg)
        assert model.ift_backward is True

    def test_random_seed_applied(self):
        cfg_a = ENFORCEConfig(input_neurons=4, hidden_neurons=16, output_neurons=2, random_seed=0)
        cfg_b = ENFORCEConfig(input_neurons=4, hidden_neurons=16, output_neurons=2, random_seed=0)
        m_a = _make_model(cfg_a, nc=1)
        m_b = _make_model(cfg_b, nc=1)
        for (_, pa), (_, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
            assert torch.allclose(pa, pb), "Same seed must produce identical weights"


# ── Architecture ──────────────────────────────────────────────────────────────


class TestArchitecture:
    def test_input_layer_shape(self):
        cfg = ENFORCEConfig(input_neurons=3, hidden_neurons=32, output_neurons=5)
        model = _make_model(cfg, nc=2)
        assert model.input_layer.in_features == 3
        assert model.input_layer.out_features == 32

    def test_hidden_layer_shape(self):
        cfg = ENFORCEConfig(hidden_neurons=128, output_neurons=3)
        model = _make_model(cfg, nc=1)
        assert isinstance(model.hidden_layers, nn.ModuleList)
        assert len(model.hidden_layers) == cfg.hidden_layers
        for layer in model.hidden_layers:
            assert layer.in_features == 128
            assert layer.out_features == 128

    def test_output_layer_shape(self):
        cfg = ENFORCEConfig(hidden_neurons=64, output_neurons=9)
        model = _make_model(cfg, nc=3)
        assert model.output_layer.out_features == 9

    def test_ni_no_attributes(self):
        cfg = ENFORCEConfig(input_neurons=5, output_neurons=7)
        model = _make_model(cfg, nc=3)
        assert model.ni == 5
        assert model.no == 7

    def test_relu_activation(self):
        model = _make_model(ENFORCEConfig())
        assert isinstance(model.hidden_activation, nn.ReLU)

    def test_loss_function_is_mse(self):
        model = _make_model(ENFORCEConfig())
        assert isinstance(model.loss_function, nn.MSELoss)

    def test_weighting_option_7_creates_parameter(self):
        cfg = ENFORCEConfig(output_neurons=4)
        model = _make_model(cfg, nc=2, weighting_option=7)
        assert hasattr(model, "proj_weight_log_diag")
        assert isinstance(model.proj_weight_log_diag, nn.Parameter)
        assert model.proj_weight_log_diag.shape == (4,)

    def test_weighting_option_non7_no_extra_parameter(self):
        for opt in (1, 3, 5, 6):
            cfg = ENFORCEConfig(output_neurons=4)
            model = _make_model(cfg, nc=2, weighting_option=opt)
            assert not hasattr(model, "proj_weight_log_diag")


# ── Forward pass ─────────────────────────────────────────────────────────────


class TestForwardPass:
    def test_output_shape_no_fb(self):
        cfg = ENFORCEConfig(input_neurons=2, hidden_neurons=32, output_neurons=5)
        model = _make_model(cfg, nc=2)
        x = torch.randn(16, 2)
        out = model.forward(x)
        assert out.shape == (16, 5)

    def test_output_shape_varies_with_batch_size(self):
        cfg = ENFORCEConfig(input_neurons=1, hidden_neurons=16, output_neurons=3)
        model = _make_model(cfg, nc=1)
        for bs in (1, 8, 64):
            x = torch.randn(bs, 1)
            assert model.forward(x).shape == (bs, 3)

    def test_hidden_layers_depth(self):
        """A 3-hidden-layer config runs HIDDEN_LAYERS forward iterations."""
        cfg = ENFORCEConfig(input_neurons=2, hidden_neurons=16, output_neurons=4, hidden_layers=3)
        model = _make_model(cfg, nc=2)
        x = torch.randn(8, 2)
        out = model.forward(x)  # should not raise
        assert out.shape == (8, 4)


# ── Fischer-Burmeister extension ──────────────────────────────────────────────


class TestFBExtension:
    def _make_fb_model(self, n_orig, n_ineq, **model_kwargs):
        def g_pos(x, y):  # g = -y[:,0] <= 0  <=>  y1 >= 0
            return -y[:, 0]

        def g_neg(x, y):  # g =  y[:,0] <= 0  <=>  y1 <= 0  (conflicts, but fine for shape tests)
            return y[:, 0]

        inequalities = [g_pos, g_neg][:n_ineq]
        fb = FischerBurmeisterReformulation(n_original_outputs=n_orig, inequalities=inequalities)

        cfg = ENFORCEConfig(
            input_neurons=2,
            hidden_neurons=32,
            output_neurons=n_orig,  # ignored when fb is passed; here for completeness
        )
        si = _scaling(2)
        so = _scaling(n_orig + n_ineq)  # scaling covers the full extended space
        return ENFORCE(scaling_input=si, scaling_output=so, c=fb, fb=fb, config=cfg, **model_kwargs), fb

    def test_output_layer_uses_fb_no(self):
        model, fb = self._make_fb_model(n_orig=3, n_ineq=2)
        assert model.output_layer.out_features == fb.no  # 3, not 5

    def test_self_no_is_extended(self):
        model, fb = self._make_fb_model(n_orig=3, n_ineq=2)
        assert model.no == fb.no + fb.n_ineq  # 5

    def test_forward_appends_zero_lambda_columns(self):
        n_orig, n_ineq = 3, 2
        model, fb = self._make_fb_model(n_orig=n_orig, n_ineq=n_ineq)
        x = torch.randn(8, 2)
        out = model.forward(x)
        assert out.shape == (8, n_orig + n_ineq)
        # Lambda columns appended as zeros by forward()
        assert torch.all(out[:, n_orig:] == 0.0)

    def test_fb_single_inequality(self):
        model, fb = self._make_fb_model(n_orig=2, n_ineq=1)
        assert model.output_layer.out_features == 2
        assert model.no == 3
        x = torch.randn(4, 2)
        assert model.forward(x).shape == (4, 3)


# ── Unconstrained mode (MLP) ──────────────────────────────────────────────────


class TestUnconstrainedMode:
    def test_predict_equals_forward_when_unconstrained(self):
        cfg = ENFORCEConfig(input_neurons=2, hidden_neurons=16, output_neurons=3)
        model = _make_model(cfg, nc=1, constrained=False)
        x = torch.randn(8, 2)
        ytilde, yhat, proj_iter = model.predict(x, training=False)
        expected = model.forward(x)
        assert torch.allclose(ytilde, expected)
        assert torch.allclose(yhat, expected)
        assert proj_iter == 0

    def test_constrained_flag_stored(self):
        model = _make_model(ENFORCEConfig(), constrained=False)
        assert model.constrained is False


# ── Per-run seed override (run_benchmark.py pattern) ─────────────────────────


class TestSeedOverridePattern:
    """ENFORCEConfig(**{**base.__dict__, "random_seed": run_seed}) must produce
    a model with different weights than one using a different seed."""

    def test_different_seeds_give_different_weights(self):
        base_cfg = ENFORCEConfig(input_neurons=4, hidden_neurons=32, output_neurons=2)
        cfg_42 = ENFORCEConfig(**{**base_cfg.__dict__, "random_seed": 42})
        cfg_99 = ENFORCEConfig(**{**base_cfg.__dict__, "random_seed": 99})
        m_42 = _make_model(cfg_42, nc=1)
        m_99 = _make_model(cfg_99, nc=1)
        params_42 = list(m_42.parameters())
        params_99 = list(m_99.parameters())
        any_different = any(not torch.allclose(p1, p2) for p1, p2 in zip(params_42, params_99))
        assert any_different, "Different seeds should produce different initialisations"

    def test_architecture_preserved_across_seed_override(self):
        base_cfg = ENFORCEConfig(input_neurons=3, hidden_neurons=64, output_neurons=5)
        per_run_cfg = ENFORCEConfig(**{**base_cfg.__dict__, "random_seed": 777})
        assert per_run_cfg.input_neurons == 3
        assert per_run_cfg.hidden_neurons == 64
        assert per_run_cfg.output_neurons == 5
        assert per_run_cfg.random_seed == 777


# ── Scaling tensors stored on model ──────────────────────────────────────────


class TestScalingStored:
    def test_scaling_tensors_stored(self):
        cfg = ENFORCEConfig(input_neurons=2, output_neurons=3)
        mean_in = torch.tensor([1.0, 2.0])
        std_in = torch.tensor([0.5, 0.5])
        mean_out = torch.tensor([0.0, 1.0, -1.0])
        std_out = torch.tensor([1.0, 2.0, 3.0])
        c = _linear_constraint(1)
        model = ENFORCE(
            scaling_input=(mean_in, std_in),
            scaling_output=(mean_out, std_out),
            c=c,
            config=cfg,
        )
        assert torch.allclose(model.mean_input, mean_in.float())
        assert torch.allclose(model.std_input, std_in.float())
        assert torch.allclose(model.mean_output, mean_out.float())
        assert torch.allclose(model.std_output, std_out.float())
