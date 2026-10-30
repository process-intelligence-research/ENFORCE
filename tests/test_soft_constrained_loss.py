"""Tests for the soft-constrained losses used by unconstrained backbones.

The penalty for plain backbones is ``mean|c| + sum_i mean(max(0, g_i)^2)``
evaluated on the unscaled original (non-extended) outputs. This keeps the
squared hinge for inequalities used in the paper's soft-constraint baselines.
"""

import pytest
import torch

from enforce import Constraints, Scaler
from enforce.losses import SoftConstrainedLoss, SoftConstrainedSSLLoss, soft_constraint_penalty


def _eq(x, y):
    return y[:, 0] - y[:, 1]


def _ineq(x, y):
    return 1.0 - y[:, 0]


def _scaler(no=3):
    return Scaler(
        (torch.tensor([1.0, -1.0]), torch.tensor([2.0, 0.5])),
        (torch.arange(no, dtype=torch.float32), torch.full((no,), 3.0)),
    )


def _expected_penalty(x, y):
    c = _eq(x, y)
    g = _ineq(x, y)
    return torch.mean(torch.abs(c)) + torch.mean(torch.clamp(g, min=0.0) ** 2)


def test_penalty_combines_equality_abs_and_inequality_squared_hinge():
    constraints = Constraints(c=[_eq], g=[_ineq])
    x = torch.zeros(4, 2)
    y_ext = torch.tensor([[0.0, 1.0, 9.0], [2.0, 0.0, 9.0], [1.0, 1.0, 9.0], [3.0, 1.0, 9.0]])
    expected = _expected_penalty(x, y_ext[:, :2])
    assert soft_constraint_penalty(constraints, x, y_ext).item() == pytest.approx(expected.item())


def test_penalty_handles_equality_only_and_inequality_only_constraints():
    x = torch.zeros(3, 2)
    y = torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
    eq_only = soft_constraint_penalty(Constraints(c=[_eq]), x, y)
    assert eq_only.item() == pytest.approx(torch.mean(torch.abs(_eq(x, y))).item())
    y_ext = torch.cat([y, torch.zeros(3, 1)], dim=1)
    ineq_only = soft_constraint_penalty(Constraints(g=[_ineq]), x, y_ext)
    assert ineq_only.item() == pytest.approx(torch.mean(torch.clamp(_ineq(x, y), min=0.0) ** 2).item())


def test_penalty_passes_only_original_output_columns():
    widths = []

    def record(x, y):
        widths.append(y.shape[1])
        return y[:, 0]

    constraints = Constraints(c=[record], g=[record, record])
    soft_constraint_penalty(constraints, torch.zeros(2, 2), torch.zeros(2, 5))
    assert set(widths) == {3}


def test_soft_constrained_loss_adds_weighted_penalty_on_unscaled_values():
    scaler = _scaler(no=4)
    constraints = Constraints(c=[_eq], g=[_ineq])
    loss_fn = SoftConstrainedLoss(scaler, constraints, weight_loss_soft=0.3)
    x, yhat, y = torch.randn(6, 2), torch.randn(6, 4), torch.randn(6, 4)
    x_u, yhat_u = scaler.unscale(x, yhat)
    expected = torch.mean((yhat - y) ** 2) + 0.3 * _expected_penalty(x_u, yhat_u[:, :3])
    assert loss_fn(yhat=yhat, y=y, x=x).item() == pytest.approx(expected.item(), rel=1e-5)


def test_soft_constrained_loss_without_inputs_warns_and_returns_data_loss():
    scaler = _scaler()
    loss_fn = SoftConstrainedLoss(scaler, Constraints(c=[_eq]), weight_loss_soft=1.0)
    yhat, y = torch.randn(4, 3), torch.randn(4, 3)
    with pytest.warns(UserWarning):
        value = loss_fn(yhat=yhat, y=y)
    assert value.item() == pytest.approx(torch.mean((yhat - y) ** 2).item())


def test_zero_weight_skips_constraint_evaluation():
    def fail(x, y):
        raise AssertionError("constraints evaluated with zero weight")

    scaler = _scaler()
    loss_fn = SoftConstrainedLoss(scaler, Constraints(c=[fail]), weight_loss_soft=0.0)
    yhat, y = torch.randn(4, 3), torch.randn(4, 3)
    assert loss_fn(yhat=yhat, y=y, x=torch.randn(4, 2)).item() == pytest.approx(torch.mean((yhat - y) ** 2).item())


def test_soft_constrained_ssl_loss_adds_weighted_penalty_on_unscaled_values():
    class Objective(SoftConstrainedSSLLoss):
        def ssl_loss(self, x, yhat):
            return torch.mean(yhat.square())

    scaler = _scaler()
    constraints = Constraints(c=[_eq])
    loss_fn = Objective(scaler, constraints, weight_loss_soft=2.0)
    x, yhat = torch.randn(5, 2), torch.randn(5, 3)
    x_u, yhat_u = scaler.unscale(x, yhat)
    expected = torch.mean(yhat_u.square()) + 2.0 * torch.mean(torch.abs(_eq(x_u, yhat_u)))
    assert loss_fn(x=x, yhat=yhat).item() == pytest.approx(expected.item(), rel=1e-5)
