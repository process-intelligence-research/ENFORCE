"""Tests for EvaluationConfig, EvalResult, and Evaluator.

Design notes
------------
- Fully synthetic: no benchmark data files required.
- Uses the same tiny-model helpers as test_trainer.py.
- Do NOT wrap model.predict() in torch.no_grad(): the Newton projection
  requires autograd to compute constraint Jacobians.
- Evaluator is tested after a short training run (3 epochs) so the model
  is a realistic object, not a freshly initialised one.
"""

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from enforce.core.config import ENFORCEConfig
from enforce.core.model import ENFORCE
from enforce.engines.evaluate import EvalResult, EvaluationConfig, Evaluator
from enforce.engines.train import Trainer, TrainingConfig

# ── Helpers ───────────────────────────────────────────────────────────────────


def _scaling(n: int):
    return (
        torch.zeros(n, dtype=torch.float32),
        torch.ones(n, dtype=torch.float32),
    )


def _linear_constraint(nc: int):
    def c(x, y):
        return y[:, :nc]

    return c


def _make_scaling_params(ni: int, no: int) -> dict:
    return {
        "input_mean": np.zeros(ni, dtype=np.float32),
        "input_std": np.ones(ni, dtype=np.float32),
        "output_mean": np.zeros(no, dtype=np.float32),
        "output_std": np.ones(no, dtype=np.float32),
    }


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
        ada_np_auto_activation=False,
    )
    return ENFORCE(
        scaling_input=_scaling(ni),
        scaling_output=_scaling(no),
        c=_linear_constraint(nc),
        config=cfg,
        constrained=constrained,
        ssl_loss=ssl_loss,
    )


def _trained_model(**model_kwargs) -> ENFORCE:
    ni = model_kwargs.get("ni", 2)
    no = model_kwargs.get("no", 4)
    model = _make_model(**model_kwargs)
    x = torch.randn(64, ni)
    y = torch.randn(64, no)
    Trainer(model, TrainingConfig(epochs=3, batch_size=64)).fit(x, y)
    return model


def _test_tensors(n: int = 40, ni: int = 2, no: int = 4):
    torch.manual_seed(99)
    return torch.randn(n, ni), torch.randn(n, no)


class _MockSSLLoss(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(y**2)


# ── EvaluationConfig ──────────────────────────────────────────────────────────


class TestEvaluationConfigDefaults:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(EvaluationConfig)

    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.batch_size is None
        assert cfg.n_original_outputs is None
        assert cfg.inequalities is None

    def test_fields_present(self):
        expected = {"batch_size", "n_original_outputs", "inequalities"}
        assert {f.name for f in dataclasses.fields(EvaluationConfig)} == expected

    def test_custom_values(self):
        cfg = EvaluationConfig(batch_size=32, n_original_outputs=3)
        assert cfg.batch_size == 32
        assert cfg.n_original_outputs == 3


# ── EvalResult ────────────────────────────────────────────────────────────────


class TestEvalResult:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(EvalResult)

    def test_fields(self):
        expected = {"metrics", "predictions", "predictions_before_proj"}
        assert {f.name for f in dataclasses.fields(EvalResult)} == expected

    def test_construction(self):
        r = EvalResult(
            metrics={"r2_y1": 0.9},
            predictions=np.zeros((10, 2)),
            predictions_before_proj=np.zeros((10, 2)),
        )
        assert r.metrics["r2_y1"] == 0.9
        assert r.predictions.shape == (10, 2)


# ── Evaluator — basic contract ────────────────────────────────────────────────


class TestEvaluatorContract:
    def test_returns_eval_result(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert isinstance(result, EvalResult)

    def test_predictions_shape(self):
        model = _trained_model(no=4)
        x, y = _test_tensors(n=40, no=4)
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert result.predictions.shape == (40, 4)

    def test_before_proj_shape_matches_predictions(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert result.predictions_before_proj.shape == result.predictions.shape

    def test_predictions_are_numpy(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert isinstance(result.predictions, np.ndarray)
        assert isinstance(result.predictions_before_proj, np.ndarray)

    def test_model_set_to_eval_mode(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert not model.training


# ── Metrics keys ──────────────────────────────────────────────────────────────


class TestMetricsKeys:
    def test_always_present_keys(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        m = result.metrics
        assert "projection iterations" in m
        assert "inference time" in m
        assert "residual_avg" in m
        assert "residual_max" in m

    def test_per_output_keys_all_outputs(self):
        model = _trained_model(no=4)
        x, y = _test_tensors(no=4)
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        m = result.metrics
        for i in range(1, 5):
            assert f"r2_y{i}" in m
            assert f"mse_y{i}" in m
            assert f"nrmse_y{i}" in m
            assert f"mape_y{i}" in m

    def test_n_original_outputs_limits_per_output_keys(self):
        """With n_original_outputs=2, only y1 and y2 metrics appear."""
        model = _trained_model(no=4)
        x, y = _test_tensors(no=4)
        sp = _make_scaling_params(2, 4)
        cfg = EvaluationConfig(n_original_outputs=2)
        result = Evaluator(model, cfg).evaluate(x, y, sp)
        m = result.metrics
        assert "r2_y1" in m and "r2_y2" in m
        assert "r2_y3" not in m and "r2_y4" not in m

    def test_ssl_keys_absent_without_ssl_loss(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert "obj_value_opt" not in result.metrics
        assert "obj_value_pred" not in result.metrics

    def test_ssl_keys_present_with_ssl_loss(self):
        model = _trained_model(ssl_loss=_MockSSLLoss())
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert "obj_value_opt" in result.metrics
        assert "obj_value_pred" in result.metrics

    def test_inequality_keys_absent_without_config(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert not any(k.startswith("ineq_") for k in result.metrics)

    def test_inequality_keys_present_when_configured(self):
        def g1(x, y):
            return y[:, 0] - 1.0

        def g2(x, y):
            return -y[:, 1]

        model = _trained_model(no=4)
        x, y = _test_tensors(no=4)
        sp = _make_scaling_params(2, 4)
        cfg = EvaluationConfig(inequalities=[g1, g2])
        result = Evaluator(model, cfg).evaluate(x, y, sp)
        m = result.metrics
        for i in (1, 2):
            assert f"ineq_g{i}_infeasible_pct" in m
            assert f"ineq_g{i}_mean_violation" in m
            assert f"ineq_g{i}_max_violation" in m


# ── Metric value sanity ───────────────────────────────────────────────────────


class TestMetricValues:
    def test_residual_avg_non_negative(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert result.metrics["residual_avg"] >= 0.0

    def test_residual_max_gte_avg(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert result.metrics["residual_max"] >= result.metrics["residual_avg"]

    def test_mse_non_negative(self):
        model = _trained_model(no=4)
        x, y = _test_tensors(no=4)
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        for i in range(1, 5):
            assert result.metrics[f"mse_y{i}"] >= 0.0

    def test_nrmse_non_negative(self):
        model = _trained_model(no=4)
        x, y = _test_tensors(no=4)
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        for i in range(1, 5):
            assert result.metrics[f"nrmse_y{i}"] >= 0.0

    def test_proj_iter_non_negative(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert result.metrics["projection iterations"] >= 0

    def test_inference_time_positive(self):
        model = _trained_model()
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert result.metrics["inference time"] > 0.0

    def test_infeasible_pct_in_0_100(self):
        def g(x, y):
            return y[:, 0]

        model = _trained_model(no=4)
        x, y = _test_tensors(no=4)
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig(inequalities=[g])).evaluate(x, y, sp)
        pct = result.metrics["ineq_g1_infeasible_pct"]
        assert 0.0 <= pct <= 100.0

    def test_violation_non_negative(self):
        def g(x, y):
            return y[:, 0]

        model = _trained_model(no=4)
        x, y = _test_tensors(no=4)
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig(inequalities=[g])).evaluate(x, y, sp)
        assert result.metrics["ineq_g1_mean_violation"] >= 0.0
        assert result.metrics["ineq_g1_max_violation"] >= 0.0


# ── Batched vs full-batch consistency ─────────────────────────────────────────


class TestBatchedInference:
    def test_batched_predictions_match_full_batch(self):
        """Splitting into batches must not change predictions."""
        model = _trained_model()
        x, y = _test_tensors(n=40)
        sp = _make_scaling_params(2, 4)

        result_full = Evaluator(model, EvaluationConfig(batch_size=40)).evaluate(x, y, sp)
        result_batched = Evaluator(model, EvaluationConfig(batch_size=10)).evaluate(x, y, sp)

        np.testing.assert_allclose(
            result_full.predictions,
            result_batched.predictions,
            rtol=1e-5,
            err_msg="Batched inference predictions differ from full-batch",
        )

    def test_none_batch_size_uses_full_batch(self):
        """batch_size=None should be equivalent to batch_size=N."""
        model = _trained_model()
        x, y = _test_tensors(n=40)
        sp = _make_scaling_params(2, 4)

        result_none = Evaluator(model, EvaluationConfig(batch_size=None)).evaluate(x, y, sp)
        result_n = Evaluator(model, EvaluationConfig(batch_size=40)).evaluate(x, y, sp)

        np.testing.assert_allclose(
            result_none.predictions,
            result_n.predictions,
            rtol=1e-5,
        )


# ── Unconstrained model ───────────────────────────────────────────────────────


class TestUnconstrainedEvaluator:
    def test_predictions_equal_before_proj_when_unconstrained(self):
        """For MLP mode ytilde == yhat so both arrays must be identical."""
        model = _trained_model(constrained=False)
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        np.testing.assert_array_equal(result.predictions, result.predictions_before_proj)

    def test_proj_iter_zero_when_unconstrained(self):
        model = _trained_model(constrained=False)
        x, y = _test_tensors()
        sp = _make_scaling_params(2, 4)
        result = Evaluator(model, EvaluationConfig()).evaluate(x, y, sp)
        assert result.metrics["projection iterations"] == 0
