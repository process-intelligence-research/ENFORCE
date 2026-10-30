"""Focused tests for shared scaling, constraint and loss primitives."""

import numpy as np
import pytest
import torch

from enforce import Constraints, ENFORCELoss, ENFORCELossConfig, Scaler
from enforce.data import scale_data
from enforce.losses.loss_core import SSLLossCore


def test_scaler_round_trips_inputs_and_outputs():
    scaler = Scaler(
        (torch.tensor([1.0, -2.0]), torch.tensor([2.0, 4.0])),
        (torch.tensor([5.0]), torch.tensor([3.0])),
    )
    x = torch.tensor([[3.0, 2.0], [1.0, -6.0]])
    y = torch.tensor([[8.0], [2.0]])
    x_scaled, y_scaled = scaler.scale(x, y)
    x_unscaled, y_unscaled = scaler.unscale(x_scaled, y_scaled)
    torch.testing.assert_close(x_unscaled, x)
    torch.testing.assert_close(y_unscaled, y)


def test_scaler_follows_the_device_of_its_inputs():
    # The statistics may sit on another device (CUDA when available); "meta" reproduces a mismatch on CPU-only CI.
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
    x = torch.zeros(4, 1, device="meta")
    y = torch.zeros(4, 2, device="meta")

    for x_out, y_out in (scaler.scale(x, y), scaler.unscale(x, y)):
        assert x_out.device.type == "meta"
        assert y_out.device.type == "meta"


def test_scaler_zero_output_standard_deviation_uses_centering_only():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.tensor([3.0]), torch.tensor([0.0])))
    _, y_scaled = scaler.scale(torch.zeros(2, 1), torch.tensor([[4.0], [1.0]]))
    torch.testing.assert_close(y_scaled, torch.tensor([[1.0], [-2.0]]))
    _, y = scaler.unscale(torch.zeros(2, 1), y_scaled)
    torch.testing.assert_close(y, torch.tensor([[4.0], [1.0]]))


def test_scaler_handles_near_zero_output_std_per_column_like_scale_data():
    # FB multiplier columns are all zeros in the labels, so their raw std is 0.
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    _, y_scaled_ref, _, _, _ = scale_data(x, y, x, y)
    scaler = Scaler((x.mean(axis=0), x.std(axis=0)), (y.mean(axis=0), y.std(axis=0)))  # raw std, not the safe one

    _, y_scaled = scaler.scale(torch.tensor(x), torch.tensor(y))
    torch.testing.assert_close(y_scaled, torch.tensor(y_scaled_ref))  # first column is still divided by its std
    _, y_unscaled = scaler.unscale(torch.tensor(x), y_scaled)
    torch.testing.assert_close(y_unscaled, torch.tensor(y))


def test_constraints_stack_equality_and_fb_inequality_residuals():
    constraints = Constraints(
        c=[lambda x, y: y[:, 0] + x[:, 0]],
        g=[lambda x, y: y[:, 1] - 1.0],
    )
    x = torch.tensor([[0.0], [1.0]])
    y = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    y_ext = constraints.extend_outputs(y)
    residuals = constraints(x, y_ext)
    assert residuals.shape == (2, 2)
    torch.testing.assert_close(residuals[:, 0], torch.tensor([0.0, 2.0]))
    assert torch.isfinite(residuals).all()
    assert constraints.nc == 1 and constraints.ng == 1


def test_no_constraints_returns_empty_residual_matrix():
    constraints = Constraints()
    residuals = constraints(torch.zeros(3, 2), torch.ones(3, 4))
    assert residuals.shape == (3, 0)


def test_empty_constraint_lists_are_equivalent_to_no_constraints():
    constraints = Constraints(c=[], g=[])
    residuals = constraints(torch.zeros(3, 2), torch.ones(3, 4))
    assert (constraints.nc, constraints.ng) == (0, 0)
    assert residuals.shape == (3, 0)


def test_ssl_loss_core_unscales_inputs_and_predictions_before_objective():
    scaler = Scaler(
        (torch.tensor([2.0]), torch.tensor([4.0])),
        (torch.tensor([3.0]), torch.tensor([5.0])),
    )

    class RecordingLoss(SSLLossCore):
        def ssl_loss(self, x, yhat):
            self.received = (x.detach().clone(), yhat.detach().clone())
            return yhat.mean()

    loss = RecordingLoss(scaler)
    value = loss(x=torch.tensor([[1.0]]), yhat=torch.tensor([[2.0]]))

    torch.testing.assert_close(loss.received[0], torch.tensor([[6.0]]))
    torch.testing.assert_close(loss.received[1], torch.tensor([[13.0]]))
    assert value.item() == pytest.approx(13.0)


def test_enforce_loss_weights_displacement_and_optional_soft_residual():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
    constraints = Constraints(c=[lambda x, y: y[:, 0]])
    loss = ENFORCELoss(
        scaler,
        constraints,
        ENFORCELossConfig(weight_loss_displacement=0.25),
    )
    y = torch.zeros(2, 2)
    yhat = torch.ones(2, 2)
    ytilde = torch.zeros(2, 2)
    total, after, displacement, before = loss.loss(ytilde, yhat, y)
    assert after.item() == pytest.approx(0.0)
    assert displacement.item() == pytest.approx(1.0)
    assert before.item() == pytest.approx(1.0)
    assert total.item() == pytest.approx(0.25)
