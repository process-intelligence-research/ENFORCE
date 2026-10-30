"""Trainer tests using explicit scaler, constraints, model and loss objects..

Design notes
------------
- All tests use tiny architectures and very few epochs so the suite stays fast.
- No real benchmark data is loaded; everything is fully synthetic.
- Do NOT wrap Trainer.fit() or model.predict() in torch.no_grad(): the Newton
  projection requires autograd to compute constraint Jacobians.
- The SSL path is exercised via a minimal _MockSSLLoss that returns a scalar.
"""

import dataclasses
import warnings

import pytest
import torch

from enforce import ENFORCE, Constraints, ENFORCEConfig, ENFORCELoss, Scaler
from enforce.engines.train import Trainer, TrainingConfig
from enforce.losses import ENFORCESSLLoss, MSELoss, SoftConstrainedLoss
from enforce.models import MLP, ENFORCEConstrainedMLP, MLPConfig


def _make_system(*, constrained=True, seed=0, no=3, inequalities=None, ssl=False, soft_weight=None):
    scaler = Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(no), torch.ones(no)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]], g=inequalities)
    if ssl:
        class ObjectiveLoss(ENFORCESSLLoss):
            def ssl_loss(self, x, yhat):
                return torch.mean(yhat.square())
        loss = ObjectiveLoss(scaler, constraints)
    elif constrained:
        loss = ENFORCELoss(scaler, constraints)
    elif soft_weight is not None:
        loss = SoftConstrainedLoss(scaler, constraints, weight_loss_soft=soft_weight)
    else:
        loss = MSELoss()
    config = MLPConfig(2, 12, no, 1, seed)
    if constrained:
        layer = ENFORCE(scaler, constraints, loss, ENFORCEConfig(random_seed=seed, ada_np_auto_activation=False))
        model = ENFORCEConstrainedMLP(config, layer)
    else:
        model = MLP(config)
    return scaler, constraints, model, loss


def _fit(system, inputs, outputs, config=None):
    scaler, constraints, model, loss = system
    return Trainer(model, loss, scaler, constraints, config or TrainingConfig()).fit(inputs, outputs)


def _data(n=24, no=3):
    generator = torch.Generator().manual_seed(17)
    return torch.randn(n, 2, generator=generator), torch.randn(n, no, generator=generator)


def test_training_config_fields_and_defaults():
    assert dataclasses.is_dataclass(TrainingConfig)
    cfg = TrainingConfig()
    assert (cfg.batch_size, cfg.epochs, cfg.random_seed) == (200, 1000, 42)
    assert cfg.learning_rate == pytest.approx(1e-3)
    assert {field.name for field in dataclasses.fields(TrainingConfig)} == {
        "batch_size", "epochs", "learning_rate", "random_seed"
    }


def test_plain_model_fit_returns_model_and_records_epoch_losses():
    system = _make_system(constrained=False)
    x, y = _data()
    returned = _fit(system, x, y, TrainingConfig(epochs=3, batch_size=8))
    assert returned is system[2]
    assert len(system[3].losses) == 3
    assert "loss_unconstrained" in system[3].losses[-1]


@pytest.mark.parametrize("soft_weight", [None, 5.0])
def test_plain_model_records_data_loss_without_soft_penalty(soft_weight):
    system = _make_system(constrained=False, soft_weight=soft_weight)
    x, y = _data()
    _fit(system, x, y, TrainingConfig(epochs=1, batch_size=24, learning_rate=0.0))
    with torch.no_grad():
        expected = torch.mean((system[2](x) - y) ** 2).item()
    assert system[3].losses[-1]["loss_unconstrained"] == pytest.approx(expected, rel=1e-5)


def test_constrained_model_fit_records_projection_losses_and_updates_epoch():
    system = _make_system(constrained=True)
    x, y = _data()
    _fit(system, x, y, TrainingConfig(epochs=2, batch_size=8))
    loss, model = system[3], system[2]
    assert len(loss.losses) == 2
    assert {
        "loss_data_after_projection",
        "loss_data_before_projection",
        "loss_displacement",
        "projection_iterations",
    } <= set(loss.losses[-1])
    assert model.constraint_module.epoch == 1


def test_ssl_constrained_training_records_objective_values():
    system = _make_system(constrained=True, ssl=True)
    x, y = _data()
    _fit(system, x, y, TrainingConfig(epochs=1, batch_size=8))
    entry = system[3].losses[0]
    assert "objective_value_optimization" in entry
    assert "objective_value_prediction" in entry


def test_same_model_seed_and_training_seed_reproduce_parameters():
    x, y = _data()
    cfg = TrainingConfig(epochs=2, batch_size=8, random_seed=11)
    a, b = _make_system(seed=5), _make_system(seed=5)
    _fit(a, x, y, cfg)
    _fit(b, x, y, cfg)
    for p_a, p_b in zip(a[2].parameters(), b[2].parameters()):
        assert torch.allclose(p_a, p_b)
    for e_a, e_b in zip(a[3].losses, b[3].losses):
        assert e_a.keys() == e_b.keys()
        for key in e_a:
            assert e_a[key] == pytest.approx(e_b[key], rel=1e-5)


def test_different_shuffle_seeds_change_the_training_trajectory():
    x, y = _data()
    a, b = _make_system(seed=5), _make_system(seed=5)
    _fit(a, x, y, TrainingConfig(epochs=2, batch_size=6, random_seed=1))
    _fit(b, x, y, TrainingConfig(epochs=2, batch_size=6, random_seed=2))
    assert any(
        left["loss_data_after_projection"] != right["loss_data_after_projection"]
        for left, right in zip(a[3].losses, b[3].losses)
    )


def test_soft_inequality_penalty_changes_plain_backbone_training():
    def g(x, y):
        return 1.0 - y[:, 0]

    x, y = _data()
    base = _make_system(constrained=False, seed=3, inequalities=[g])
    penalized = _make_system(constrained=False, seed=3, inequalities=[g], soft_weight=5.0)
    _fit(base, x, y, TrainingConfig(epochs=4, batch_size=24))
    _fit(penalized, x, y, TrainingConfig(epochs=4, batch_size=24))
    assert any(not torch.allclose(a, b) for a, b in zip(base[2].parameters(), penalized[2].parameters()))


def test_soft_equality_penalty_changes_plain_backbone_training():
    x, y = _data()
    base = _make_system(constrained=False, seed=3)
    penalized = _make_system(constrained=False, seed=3, soft_weight=5.0)
    _fit(base, x, y, TrainingConfig(epochs=4, batch_size=24))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # fails if the Trainer does not pass x to the loss
        _fit(penalized, x, y, TrainingConfig(epochs=4, batch_size=24))
    assert any(not torch.allclose(a, b) for a, b in zip(base[2].parameters(), penalized[2].parameters()))


def test_soft_penalty_receives_only_original_output_columns():
    widths = []
    def record_width(x, y):
        widths.append(y.shape[1])
        return 1.0 - y[:, 0]

    system = _make_system(constrained=False, no=4, inequalities=[record_width], soft_weight=0.1)
    x, y = _data(no=4)
    _fit(system, x, y, TrainingConfig(epochs=1, batch_size=24))
    assert widths
    assert set(widths) == {3}


def test_trainer_rejects_soft_constrained_loss_for_constrained_model():
    scaler, constraints, model, _ = _make_system(constrained=True)
    with pytest.raises(NotImplementedError):
        Trainer(model, SoftConstrainedLoss(scaler, constraints), scaler, constraints, TrainingConfig())


def test_unconstrained_self_supervised_training_is_supported():
    system = _make_system(constrained=False, ssl=True)
    x, y = _data()
    _fit(system, x, y, TrainingConfig(epochs=1, batch_size=24))
    assert len(system[3].losses) == 1


def test_projection_count_is_available_from_model_predict():
    scaler, constraints, model_template, loss = _make_system(constrained=True)
    layer = ENFORCE(
        scaler, constraints, loss,
        ENFORCEConfig(epoch_start_hard_constrained=5),
    )
    model = ENFORCEConstrainedMLP(model_template.mlp_cfg, layer)
    x = torch.randn(5, 2)
    ytilde, yhat, projection_count = model.predict(x)
    assert ytilde.shape == yhat.shape == (5, 3)
    assert projection_count == 0


def test_trainer_rejects_unsupported_model_loss_pair():
    scaler, constraints, model, _ = _make_system(constrained=False)
    with pytest.raises(NotImplementedError):
        Trainer(model, ENFORCELoss(scaler, constraints), scaler, constraints, TrainingConfig())
