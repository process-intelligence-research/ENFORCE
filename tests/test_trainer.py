"""Tests for TrainingConfig and Trainer.

Design notes
------------
- All tests use tiny architectures and very few epochs so the suite stays fast.
- No real benchmark data is loaded; everything is fully synthetic.
- Do NOT wrap Trainer.fit() or model.predict() in torch.no_grad(): the Newton
  projection requires autograd to compute constraint Jacobians.
- The SSL path is exercised via a minimal _MockSSLLoss that returns a scalar.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn

from enforce.core.config import ENFORCEConfig
from enforce.core.model import ENFORCE
from enforce.engines.train import Trainer, TrainingConfig

# ── Helpers ───────────────────────────────────────────────────────────────────


def _scaling(n: int):
    return (
        torch.zeros(n, dtype=torch.float32),
        torch.ones(n, dtype=torch.float32),
    )


def _linear_constraint(nc: int):
    """c(x, y) = y[:, :nc] — zero exactly when the first nc outputs are zero."""

    def c(x, y):
        return y[:, :nc]

    return c


def _make_model(
    ni: int = 2,
    no: int = 4,
    nc: int = 1,
    constrained: bool = True,
    seed: int = 0,
    ssl_loss=None,
) -> ENFORCE:
    cfg = ENFORCEConfig(
        input_neurons=ni,
        hidden_neurons=16,
        output_neurons=no,
        hidden_layers=1,
        random_seed=seed,
        ada_np_auto_activation=False,  # always project so tests are deterministic
    )
    return ENFORCE(
        scaling_input=_scaling(ni),
        scaling_output=_scaling(no),
        c=_linear_constraint(nc),
        config=cfg,
        constrained=constrained,
        ssl_loss=ssl_loss,
    )


def _synthetic_data(n: int = 64, ni: int = 2, no: int = 4):
    torch.manual_seed(7)
    return torch.randn(n, ni), torch.randn(n, no)


class _MockSSLLoss(nn.Module):
    """Minimal SSL loss: mean(y²).  Always positive, differentiable."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(y**2)


# ── TrainingConfig ────────────────────────────────────────────────────────────


class TestTrainingConfigDefaults:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TrainingConfig)

    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 200
        assert cfg.epochs == 1000
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.random_seed == 42
        assert cfg.soft_inequalities is None
        assert cfg.soft_weight == pytest.approx(0.0)
        assert cfg.n_original_outputs is None

    def test_fields_present(self):
        expected = {
            "batch_size",
            "epochs",
            "learning_rate",
            "random_seed",
            "soft_inequalities",
            "soft_weight",
            "n_original_outputs",
        }
        assert {f.name for f in dataclasses.fields(TrainingConfig)} == expected

    def test_custom_values_stored(self):
        cfg = TrainingConfig(
            batch_size=32,
            epochs=500,
            learning_rate=5e-4,
            random_seed=99,
            soft_weight=0.1,
            n_original_outputs=3,
        )
        assert cfg.batch_size == 32
        assert cfg.epochs == 500
        assert cfg.learning_rate == pytest.approx(5e-4)
        assert cfg.random_seed == 99
        assert cfg.soft_weight == pytest.approx(0.1)
        assert cfg.n_original_outputs == 3


# ── Trainer.fit — basic contract ──────────────────────────────────────────────


class TestTrainerFitContract:
    def test_returns_enforce_model(self):
        model = _make_model()
        x, y = _synthetic_data()
        result = Trainer(model, TrainingConfig(epochs=2, batch_size=64)).fit(x, y)
        assert isinstance(result, ENFORCE)

    def test_returns_same_model_object(self):
        model = _make_model()
        x, y = _synthetic_data()
        result = Trainer(model, TrainingConfig(epochs=2, batch_size=64)).fit(x, y)
        assert result is model

    def test_losses_list_length_equals_epochs(self):
        model = _make_model()
        x, y = _synthetic_data()
        Trainer(model, TrainingConfig(epochs=5, batch_size=64)).fit(x, y)
        assert len(model.losses) == 5

    def test_losses_empty_before_fit(self):
        model = _make_model()
        assert model.losses == []

    def test_model_epoch_updated(self):
        model = _make_model()
        x, y = _synthetic_data()
        Trainer(model, TrainingConfig(epochs=3, batch_size=64)).fit(x, y)
        # epoch counter ends at epochs-1 (0-indexed range)
        assert model.epoch == 2


# ── Loss dict keys ────────────────────────────────────────────────────────────


class TestTrainerLossKeys:
    def test_constrained_keys(self):
        model = _make_model(constrained=True)
        x, y = _synthetic_data()
        Trainer(model, TrainingConfig(epochs=2, batch_size=64)).fit(x, y)
        entry = model.losses[0]
        assert "loss_data_after_projection" in entry
        assert "loss_data_before_projection" in entry
        assert "loss_displacement" in entry
        assert "projection_iterations" in entry
        assert "loss_unconstrained" not in entry

    def test_unconstrained_keys(self):
        model = _make_model(constrained=False)
        x, y = _synthetic_data()
        Trainer(model, TrainingConfig(epochs=2, batch_size=64)).fit(x, y)
        entry = model.losses[0]
        assert "loss_unconstrained" in entry
        assert "loss_data_after_projection" not in entry
        assert "projection_iterations" not in entry

    def test_ssl_problem_adds_objective_keys(self):
        ssl = _MockSSLLoss()
        model = _make_model(constrained=True, ssl_loss=ssl)
        x, y = _synthetic_data()
        Trainer(model, TrainingConfig(epochs=2, batch_size=64)).fit(x, y)
        entry = model.losses[0]
        assert "objective_value_optimization" in entry
        assert "objective_value_prediction" in entry

    def test_non_ssl_constrained_has_no_objective_keys(self):
        model = _make_model(constrained=True)
        x, y = _synthetic_data()
        Trainer(model, TrainingConfig(epochs=2, batch_size=64)).fit(x, y)
        entry = model.losses[0]
        assert "objective_value_optimization" not in entry
        assert "objective_value_prediction" not in entry


# ── proj_iter is batch-averaged ───────────────────────────────────────────────


class TestProjectionIterationsAveraged:
    def test_proj_iter_is_numeric(self):
        model = _make_model(constrained=True)
        x, y = _synthetic_data()
        Trainer(model, TrainingConfig(epochs=3, batch_size=64)).fit(x, y)
        for entry in model.losses:
            pi = entry["projection_iterations"]
            assert isinstance(pi, int | float)

    def test_proj_iter_positive_when_projecting(self):
        """With ada_np_auto_activation=False the projection always runs."""
        model = _make_model(constrained=True)
        x, y = _synthetic_data()
        Trainer(model, TrainingConfig(epochs=3, batch_size=64)).fit(x, y)
        for entry in model.losses:
            assert entry["projection_iterations"] >= 1

    def test_proj_iter_is_average_across_batches(self):
        """With multiple batches, proj_iter must be a float (batch average).

        We train on 128 samples with batch_size=32 → 4 batches.
        The stored value must equal the mean of per-batch proj_iter values.
        We verify it is ≥ 1 and consistent across epochs (smoke-test for averaging).
        """
        n = 128
        x = torch.randn(n, 2)
        y = torch.randn(n, 4)
        model = _make_model(constrained=True)
        Trainer(model, TrainingConfig(epochs=2, batch_size=32)).fit(x, y)
        for entry in model.losses:
            pi = entry["projection_iterations"]
            # With 4 batches, the average is a multiple of 0.25 (or integer if all equal)
            assert pi >= 1.0
            # Must be representable as k/4 for some integer k
            assert abs(pi * 4 - round(pi * 4)) < 1e-6


# ── Reproducibility ───────────────────────────────────────────────────────────


class TestTrainerReproducibility:
    def test_same_seed_identical_losses(self):
        x, y = _synthetic_data()
        cfg = TrainingConfig(epochs=3, batch_size=64, random_seed=11)

        m1 = _make_model(seed=5)
        Trainer(m1, cfg).fit(x, y)

        m2 = _make_model(seed=5)
        Trainer(m2, cfg).fit(x, y)

        for e1, e2 in zip(m1.losses, m2.losses):
            for key in e1:
                assert e1[key] == pytest.approx(e2[key], rel=1e-5), f"key={key}: {e1[key]} != {e2[key]}"

    def test_same_seed_identical_weights(self):
        x, y = _synthetic_data()
        cfg = TrainingConfig(epochs=3, batch_size=64, random_seed=11)

        m1 = _make_model(seed=5)
        m2 = _make_model(seed=5)
        Trainer(m1, cfg).fit(x, y)
        Trainer(m2, cfg).fit(x, y)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_different_training_seeds_different_losses(self):
        x, y = _synthetic_data()

        m1 = _make_model(seed=5)
        m2 = _make_model(seed=5)
        Trainer(m1, TrainingConfig(epochs=3, batch_size=32, random_seed=1)).fit(x, y)
        Trainer(m2, TrainingConfig(epochs=3, batch_size=32, random_seed=2)).fit(x, y)

        any_diff = any(
            e1["loss_data_after_projection"] != e2["loss_data_after_projection"] for e1, e2 in zip(m1.losses, m2.losses)
        )
        assert any_diff, "Different data-shuffle seeds should give different training trajectories"


# ── Soft inequality penalty ───────────────────────────────────────────────────


def _g_almost_always_violated(x, y):
    """g(x,y) = 1 - y[:,0] <= 0  (violated when y[:,0] < 1).

    With random-init outputs near zero this constraint is almost always active,
    so the soft penalty gradient is always non-zero and visibly changes training.
    """
    return 1.0 - y[:, 0]


class TestSoftInequalityPenalty:
    def test_soft_penalty_does_not_crash(self):
        model = _make_model(constrained=False)
        x, y = _synthetic_data()
        cfg = TrainingConfig(
            epochs=2,
            batch_size=64,
            soft_inequalities=[_g_almost_always_violated],
            soft_weight=0.1,
        )
        Trainer(model, cfg).fit(x, y)  # must not raise

    def test_soft_penalty_changes_loss(self):
        """With a penalty, final weights differ from the penalty-free run."""
        x, y = _synthetic_data()

        m_no_pen = _make_model(constrained=False, seed=3)
        m_pen = _make_model(constrained=False, seed=3)
        Trainer(m_no_pen, TrainingConfig(epochs=10, batch_size=64)).fit(x, y)
        Trainer(
            m_pen,
            TrainingConfig(
                epochs=10,
                batch_size=64,
                soft_inequalities=[_g_almost_always_violated],
                soft_weight=5.0,
            ),
        ).fit(x, y)

        any_diff = any(not torch.allclose(p1, p2) for p1, p2 in zip(m_no_pen.parameters(), m_pen.parameters()))
        assert any_diff, "Soft penalty should alter the learned weights"

    def test_n_original_outputs_strips_fb_columns(self):
        """n_original_outputs limits which columns are passed to g_i."""
        calls = []

        def _g_spy(x, y):
            calls.append(y.shape[1])
            return 1.0 - y[:, 0]  # always violated so soft_weight actually fires

        # 4 outputs but n_original_outputs=2 → spy should see width 2
        model = _make_model(no=4, constrained=False)
        x, y = _synthetic_data(no=4)
        cfg = TrainingConfig(
            epochs=1,
            batch_size=64,
            soft_inequalities=[_g_spy],
            soft_weight=0.1,
            n_original_outputs=2,
        )
        Trainer(model, cfg).fit(x, y)
        assert all(w == 2 for w in calls), f"Expected width 2, got {calls}"
