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
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from enforce import ENFORCE, Constraints, ENFORCEConfig, ENFORCELoss, Scaler
from enforce.engines.evaluate import EvalResult, EvaluationConfig, Evaluator
from enforce.losses import MSELoss, SoftConstrainedLoss
from enforce.models import MLP, ENFORCEConstrainedMLP, MLPConfig


def _system(constrained=False, no=3, inequalities=None):
    scaler = Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(no), torch.ones(no)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]], g=inequalities)
    loss = ENFORCELoss(scaler, constraints) if constrained else MSELoss()
    cfg = MLPConfig(2, 10, no, 1, 4)
    if constrained:
        layer = ENFORCE(scaler, constraints, loss, ENFORCEConfig(epoch_start_hard_constrained=10))
        model = ENFORCEConstrainedMLP(cfg, layer)
    else:
        model = MLP(cfg)
    return scaler, constraints, model, loss


def test_evaluation_config_contains_only_batch_setting():
    assert dataclasses.is_dataclass(EvaluationConfig)
    assert EvaluationConfig().batch_size is None
    assert EvaluationConfig(batch_size=7).batch_size == 7
    assert {field.name for field in dataclasses.fields(EvaluationConfig)} == {"batch_size"}


def test_evaluator_returns_numpy_predictions_and_metrics():
    scaler, constraints, model, loss = _system()
    x, y = torch.randn(13, 2), torch.randn(13, 3)
    result = Evaluator(model, loss, scaler, constraints, EvaluationConfig(batch_size=5)).evaluate(x, y)
    assert isinstance(result, EvalResult)
    assert isinstance(result.predictions, np.ndarray)
    assert result.predictions.shape == result.predictions_before_proj.shape == (13, 3)
    assert "r2_y1" in result.metrics and "mse_y3" in result.metrics
    assert "residual_avg" in result.metrics and "residual_max" in result.metrics


def test_evaluator_handles_single_batch_and_partial_final_batch():
    scaler, constraints, model, loss = _system()
    x, y = torch.randn(11, 2), torch.randn(11, 3)
    full = Evaluator(model, loss, scaler, constraints, EvaluationConfig()).evaluate(x, y)
    batched = Evaluator(model, loss, scaler, constraints, EvaluationConfig(batch_size=4)).evaluate(x, y)
    np.testing.assert_allclose(full.predictions, batched.predictions, atol=1e-6)


def test_evaluator_supports_models_without_constraints():
    scaler = Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(3), torch.ones(3)))
    constraints = Constraints(c=[], g=[])
    model = MLP(MLPConfig(2, 8, 3, 1, 4))
    x, y = torch.randn(5, 2), torch.randn(5, 3)

    result = Evaluator(model, MSELoss(), scaler, constraints, EvaluationConfig()).evaluate(x, y)

    assert result.metrics["residual_avg"] == 0.0
    assert result.metrics["residual_max"] == 0.0
    assert result.predictions.shape == (5, 3)


def test_constrained_evaluator_returns_prediction_after_constraint_module():
    scaler, constraints, model, loss = _system(constrained=True)
    x, y = torch.randn(8, 2), torch.randn(8, 3)
    result = Evaluator(model, loss, scaler, constraints, EvaluationConfig(batch_size=4)).evaluate(x, y)
    assert result.predictions.shape == (8, 3)
    assert result.metrics["projection iterations"] == 0


def test_evaluator_reports_inequality_violation_metrics_and_extracts_original_outputs():
    def g(x, y):
        return y[:, 0] - 0.5

    scaler, constraints, model, loss = _system(inequalities=[g])
    x, y = torch.randn(9, 2), torch.randn(9, 3)
    result = Evaluator(model, loss, scaler, constraints, EvaluationConfig()).evaluate(x, y)
    assert result.predictions.shape == (9, 2)
    assert "ineq_g1_infeasible_pct" in result.metrics
    assert "ineq_g1_mean_violation" in result.metrics
    assert "ineq_g1_max_violation" in result.metrics


def test_evaluator_uses_scaler_to_return_unscaled_predictions():
    scaler = Scaler(
        (torch.tensor([2.0]), torch.tensor([4.0])),
        (torch.tensor([10.0, -3.0]), torch.tensor([2.0, 5.0])),
    )
    constraints = Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]])
    model = MLP(MLPConfig(1, 2, 2))
    with torch.no_grad():
        model.output_layer.weight.zero_()
        model.output_layer.bias.zero_()
    result = Evaluator(model, MSELoss(), scaler, constraints, EvaluationConfig()).evaluate(
        torch.zeros(4, 1), torch.zeros(4, 2)
    )
    np.testing.assert_allclose(result.predictions, np.tile([10.0, -3.0], (4, 1)))


def test_evaluator_accepts_soft_constrained_loss_for_plain_backbone():
    scaler, constraints, model, _ = _system()
    loss = SoftConstrainedLoss(scaler, constraints, weight_loss_soft=1.0)
    x, y = torch.randn(6, 2), torch.randn(6, 3)
    result = Evaluator(model, loss, scaler, constraints, EvaluationConfig()).evaluate(x, y)
    assert result.predictions.shape == (6, 3)


def test_evaluator_rejects_unsupported_loss_for_plain_backbone():
    scaler, constraints, model, _ = _system()
    with pytest.raises(NotImplementedError):
        Evaluator(model, nn.MSELoss(), scaler, constraints, EvaluationConfig())


def test_per_output_metrics_promote_large_float32_residuals_before_r2():
    scaler, constraints, model, loss = _system()
    evaluator = Evaluator(model, loss, scaler, constraints, EvaluationConfig())
    y_true = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    y_pred = np.array([[1e20, 0.0, 0.0], [-1e20, 0.0, 0.0]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = evaluator._per_output_metrics(y_true, y_pred)
    assert np.isfinite(metrics["r2_y1"])
    assert np.isfinite(metrics["mse_y1"])
    assert metrics["mse_y1"] > 1e38


def test_evaluator_supports_generic_constrained_model():
    from enforce.models import ConstrainedMLP

    class Clamp(nn.Module):
        def forward(self, yhat):
            return yhat.clamp(-0.1, 0.1)

    scaler, constraints, _, _ = _system()
    model = ConstrainedMLP(MLPConfig(2, 10, 3, 1, 4), Clamp())
    loss = ENFORCELoss(scaler, constraints)
    x, y = torch.randn(7, 2), torch.randn(7, 3)
    result = Evaluator(model, loss, scaler, constraints, EvaluationConfig(batch_size=3)).evaluate(x, y)
    assert result.metrics["projection iterations"] == 0
    assert np.abs(result.predictions).max() <= 0.1 + 1e-6
