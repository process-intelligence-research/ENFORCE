"""Tests for the split configuration objects."""

import dataclasses

import pytest

from enforce import ENFORCEConfig, ENFORCELossConfig
from enforce.models import MLPConfig
from enforce.engines.train import TrainingConfig
from enforce.engines.evaluate import EvaluationConfig


def test_configs_are_dataclasses():
    for config in (ENFORCEConfig, ENFORCELossConfig, MLPConfig, TrainingConfig, EvaluationConfig):
        assert dataclasses.is_dataclass(config)


def test_enforce_config_projection_defaults():
    cfg = ENFORCEConfig()
    assert cfg.training_tolerance == pytest.approx(1e-4)
    assert cfg.inference_tolerance == pytest.approx(1e-6)
    assert cfg.max_it == 100
    assert cfg.epoch_start_hard_constrained == 0
    assert cfg.ada_np_auto_activation is True
    assert cfg.ift_backward is False
    assert cfg.weighting_option == "ID"
    assert cfg.eps_chol == pytest.approx(1e-8)
    assert cfg.regularise_gram is False
    assert cfg.verbose is False
    assert cfg.random_seed == 42


def test_enforce_config_rejects_max_it_below_one():
    with pytest.raises(ValueError, match="max_it"):
        ENFORCEConfig(max_it=0)


def test_loss_config_has_only_loss_specific_options():
    cfg = ENFORCELossConfig()
    assert cfg.soft_constrained is False
    assert cfg.weight_loss_displacement == pytest.approx(0.5)
    assert cfg.weight_loss_soft == pytest.approx(0.0)
    assert {f.name for f in dataclasses.fields(ENFORCELossConfig)} == {
        "soft_constrained", "weight_loss_displacement", "weight_loss_soft"
    }


def test_configs_keep_custom_values_and_seed_copy_pattern():
    cfg = ENFORCEConfig(
        training_tolerance=1e-3, weighting_option="I",
        regularise_gram=True, random_seed=8,
    )
    assert cfg.training_tolerance == pytest.approx(1e-3)
    assert cfg.weighting_option == "I"
    assert cfg.regularise_gram
    copied = ENFORCEConfig(**{**cfg.__dict__, "random_seed": 99})
    assert copied.random_seed == 99
    assert copied.training_tolerance == cfg.training_tolerance


def test_network_and_engine_configs_own_their_own_parameters():
    net = MLPConfig(input_neurons=3, hidden_neurons=12, output_neurons=5, hidden_layers=2, random_seed=7)
    train = TrainingConfig(batch_size=16, epochs=4, learning_rate=2e-3, random_seed=9)
    evaluate = EvaluationConfig(batch_size=8)
    assert (net.input_neurons, net.hidden_neurons, net.output_neurons, net.hidden_layers) == (3, 12, 5, 2)
    assert (train.batch_size, train.epochs, train.learning_rate) == (16, 4, pytest.approx(2e-3))
    assert evaluate.batch_size == 8


def test_ift_backward_is_rejected_at_configuration():
    with pytest.raises(NotImplementedError, match="IFT"):
        ENFORCEConfig(ift_backward=True)
