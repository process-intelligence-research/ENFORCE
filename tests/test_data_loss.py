"""Tests for the ``data_loss`` contract of the losses.

``data_loss(yhat, y)`` is the data term of a loss (no penalties) and follows the
torch convention ``loss_function(input, target)``: prediction first, target second.
The argument order is checked with an asymmetric stand-in loss that records its calls.
"""

import pytest
import torch
import torch.nn as nn

from enforce import ENFORCE, Constraints, ENFORCEConfig, ENFORCELoss, Scaler
from enforce.engines.train import Trainer, TrainingConfig
from enforce.losses import ENFORCESSLLoss, MSELoss, SoftConstrainedLoss
from enforce.models import MLP, ENFORCEConstrainedMLP, MLPConfig


class _OrderRecorder(nn.Module):
    """Asymmetric stand-in for a torch loss that records (input, target)."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input, target):
        self.calls.append((input, target))
        return torch.mean(input - target)


def _scaler(no=3):
    return Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(no), torch.ones(no)))


def _constraints():
    return Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]])


def _assert_prediction_first(recorder, y):
    assert recorder.calls
    for prediction, target in recorder.calls:
        assert target is y
        assert prediction is not y


def test_mse_loss_data_loss_is_the_loss():
    loss_fn = MSELoss()
    yhat, y = torch.randn(5, 3), torch.randn(5, 3)
    assert loss_fn.data_loss(yhat, y).item() == pytest.approx(torch.mean((yhat - y) ** 2).item())
    assert loss_fn(yhat=yhat, y=y).item() == pytest.approx(loss_fn.data_loss(yhat, y).item())


def test_soft_constrained_data_loss_excludes_penalty():
    loss_fn = SoftConstrainedLoss(_scaler(), _constraints(), weight_loss_soft=5.0)
    x, yhat, y = torch.randn(5, 2), torch.randn(5, 3), torch.randn(5, 3)
    mse = torch.mean((yhat - y) ** 2).item()
    assert loss_fn.data_loss(yhat, y).item() == pytest.approx(mse)
    assert loss_fn(yhat=yhat, y=y, x=x).item() > mse


def test_enforce_loss_data_terms_use_data_loss():
    class ConstantDataLoss(ENFORCELoss):
        def data_loss(self, yhat, y):
            return torch.tensor(7.0)

    loss_fn = ConstantDataLoss(_scaler(), _constraints())
    yhat, ytilde, y = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
    _, after, _, before = loss_fn.loss(ytilde=ytilde, yhat=yhat, y=y)
    assert after.item() == before.item() == 7.0


@pytest.mark.parametrize("make_loss", [
    lambda: MSELoss(),
    lambda: SoftConstrainedLoss(_scaler(), _constraints(), weight_loss_soft=0.0),
])
def test_plain_losses_pass_prediction_first(make_loss):
    loss_fn = make_loss()
    loss_fn.loss_function = recorder = _OrderRecorder()
    y = torch.randn(4, 3)
    loss_fn(yhat=torch.randn(4, 3), y=y, x=torch.randn(4, 2))
    _assert_prediction_first(recorder, y)


def test_enforce_loss_passes_prediction_first():
    loss_fn = ENFORCELoss(_scaler(), _constraints())
    loss_fn.loss_function = recorder = _OrderRecorder()
    y = torch.randn(4, 3)
    loss_fn.loss(ytilde=torch.randn(4, 3), yhat=torch.randn(4, 3), y=y)
    assert len(recorder.calls) == 2
    _assert_prediction_first(recorder, y)


def test_enforce_ssl_loss_passes_prediction_first():
    class Objective(ENFORCESSLLoss):
        def ssl_loss(self, x, yhat):
            return torch.mean(yhat.square())

    loss_fn = Objective(_scaler(), _constraints())
    loss_fn.loss_function = recorder = _OrderRecorder()
    y = torch.randn(4, 3)
    loss_fn.loss(ytilde=torch.randn(4, 3), yhat=torch.randn(4, 3), y=y, x=torch.randn(4, 2))
    assert len(recorder.calls) == 2
    _assert_prediction_first(recorder, y)


def test_enforce_adanp_decision_uses_data_loss_with_prediction_first():
    scaler, constraints = _scaler(), _constraints()
    loss_fn = ENFORCELoss(scaler, constraints)
    loss_fn.loss_function = recorder = _OrderRecorder()
    layer = ENFORCE(scaler, constraints, loss_fn, ENFORCEConfig(ada_np_auto_activation=True))
    model = ENFORCEConstrainedMLP(MLPConfig(2, 8, 3, 1), layer)
    model.train()
    y = torch.randn(6, 3)
    model.predict(x=torch.randn(6, 2), y=y)
    _assert_prediction_first(recorder, y)


def test_trainer_records_data_loss_of_plain_losses():
    scaler, constraints = _scaler(), _constraints()
    loss_fn = MSELoss()
    loss_fn.loss = lambda yhat, y, x=None: torch.mean((yhat - y) ** 2)
    loss_fn.data_loss = lambda yhat, y: torch.tensor(42.0)
    model = MLP(MLPConfig(2, 8, 3, 1))
    Trainer(model, loss_fn, scaler, constraints, TrainingConfig(epochs=1, batch_size=8)).fit(
        torch.randn(8, 2), torch.randn(8, 3)
    )
    assert loss_fn.losses[-1]["loss_unconstrained"] == pytest.approx(42.0)
