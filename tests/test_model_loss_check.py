"""Tests for the model/loss compatibility check shared by Trainer and Evaluator.

Rules (checked with isinstance, so subclasses are accepted):
- ConstrainedModel: ENFORCELoss or ENFORCESSLLoss (the training step needs ytilde and yhat).
- Plain backbone: any LossCore or SSLLossCore, except ENFORCELoss.
"""

import pytest
import torch
import torch.nn as nn

from enforce import ENFORCE, Constraints, ENFORCEConfig, ENFORCELoss, Scaler
from enforce.engines.evaluate import EvaluationConfig, Evaluator
from enforce.engines.train import Trainer, TrainingConfig
from enforce.losses import ENFORCESSLLoss, LossCore, MSELoss, SoftConstrainedLoss, SSLLossCore
from enforce.models import MLP, ENFORCEConstrainedMLP, MLPConfig


class _CustomMSE(MSELoss):
    pass


class _CustomLoss(LossCore):
    def __init__(self):
        super().__init__(loss_function=nn.L1Loss())

    def loss(self, yhat, y, x=None):
        return self.data_loss(yhat, y)


class _CustomENFORCELoss(ENFORCELoss):
    pass


class _Objective(SSLLossCore):
    def ssl_loss(self, x, yhat):
        return torch.mean(yhat.square())


class _ENFORCEObjective(ENFORCESSLLoss):
    def ssl_loss(self, x, yhat):
        return torch.mean(yhat.square())


def _scaler():
    return Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(3), torch.ones(3)))


def _constraints():
    return Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]])


LOSSES = {
    "MSELoss": lambda s, c: MSELoss(),
    "MSELoss subclass": lambda s, c: _CustomMSE(),
    "custom LossCore": lambda s, c: _CustomLoss(),
    "SoftConstrainedLoss": lambda s, c: SoftConstrainedLoss(s, c),
    "SSLLossCore": lambda s, c: _Objective(scaler=s),
    "ENFORCESSLLoss": lambda s, c: _ENFORCEObjective(s, c),
    "ENFORCELoss": lambda s, c: ENFORCELoss(s, c),
    "ENFORCELoss subclass": lambda s, c: _CustomENFORCELoss(s, c),
    "torch nn.MSELoss": lambda s, c: nn.MSELoss(),
}

PLAIN_OK = {"MSELoss", "MSELoss subclass", "custom LossCore", "SoftConstrainedLoss", "SSLLossCore", "ENFORCESSLLoss"}
CONSTRAINED_OK = {"ENFORCELoss", "ENFORCELoss subclass", "ENFORCESSLLoss"}


def _engine(engine, scaler, constraints, model, loss):
    if engine == "trainer":
        return Trainer(model, loss, scaler, constraints, TrainingConfig())
    return Evaluator(model, loss, scaler, constraints, EvaluationConfig())


@pytest.mark.parametrize("engine", ["trainer", "evaluator"])
@pytest.mark.parametrize("name", list(LOSSES))
def test_plain_backbone_loss_check(engine, name):
    scaler, constraints = _scaler(), _constraints()
    model = MLP(MLPConfig(2, 8, 3, 1))
    loss = LOSSES[name](scaler, constraints)
    if name in PLAIN_OK:
        _engine(engine, scaler, constraints, model, loss)
    else:
        with pytest.raises(NotImplementedError):
            _engine(engine, scaler, constraints, model, loss)


@pytest.mark.parametrize("engine", ["trainer", "evaluator"])
@pytest.mark.parametrize("name", list(LOSSES))
def test_constrained_model_loss_check(engine, name):
    scaler, constraints = _scaler(), _constraints()
    loss = LOSSES[name](scaler, constraints)
    layer = ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints), ENFORCEConfig())
    model = ENFORCEConstrainedMLP(MLPConfig(2, 8, 3, 1), layer)
    if name in CONSTRAINED_OK:
        _engine(engine, scaler, constraints, model, loss)
    else:
        with pytest.raises(NotImplementedError):
            _engine(engine, scaler, constraints, model, loss)


def test_constrained_model_trains_with_enforce_loss_subclass():
    scaler, constraints = _scaler(), _constraints()
    loss = _CustomENFORCELoss(scaler, constraints)
    layer = ENFORCE(scaler, constraints, loss, ENFORCEConfig(ada_np_auto_activation=False))
    model = ENFORCEConstrainedMLP(MLPConfig(2, 8, 3, 1), layer)
    Trainer(model, loss, scaler, constraints, TrainingConfig(epochs=1, batch_size=8)).fit(
        torch.randn(8, 2), torch.randn(8, 3)
    )
    assert len(loss.losses) == 1


def test_plain_backbone_trains_with_custom_loss_core():
    scaler, constraints = _scaler(), _constraints()
    loss = _CustomLoss()
    Trainer(MLP(MLPConfig(2, 8, 3, 1)), loss, scaler, constraints, TrainingConfig(epochs=1, batch_size=8)).fit(
        torch.randn(8, 2), torch.randn(8, 3)
    )
    assert len(loss.losses) == 1
