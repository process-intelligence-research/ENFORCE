"""Tests for ENFORCEConfig dataclass."""

import dataclasses

import pytest

from enforce.core.config import ENFORCEConfig


class TestENFORCEConfigDefaults:
    """Default values match the documented paper defaults."""

    def test_architecture_defaults(self):
        cfg = ENFORCEConfig()
        assert cfg.input_neurons == 1
        assert cfg.hidden_neurons == 64
        assert cfg.output_neurons == 1
        assert cfg.hidden_layers == 1

    def test_projection_defaults(self):
        cfg = ENFORCEConfig()
        assert cfg.training_tolerance == pytest.approx(1e-4)
        assert cfg.inference_tolerance == pytest.approx(1e-6)
        assert cfg.max_it == 100
        assert cfg.epoch_start_hard_constrained == 0
        assert cfg.ada_np_auto_activation is True
        assert cfg.ift_backward is False

    def test_loss_defaults(self):
        cfg = ENFORCEConfig()
        assert cfg.supervised is True
        assert cfg.soft_constrained is False
        assert cfg.weight_loss_displacement == pytest.approx(0.5)
        assert cfg.weight_loss_soft == pytest.approx(0.0)

    def test_misc_defaults(self):
        cfg = ENFORCEConfig()
        assert cfg.verbose is False
        assert cfg.random_seed == 42


class TestENFORCEConfigCustomValues:
    """Custom values are stored and retrievable."""

    def test_architecture_custom(self):
        cfg = ENFORCEConfig(input_neurons=3, hidden_neurons=128, output_neurons=9, hidden_layers=2)
        assert cfg.input_neurons == 3
        assert cfg.hidden_neurons == 128
        assert cfg.output_neurons == 9
        assert cfg.hidden_layers == 2

    def test_projection_custom(self):
        cfg = ENFORCEConfig(
            training_tolerance=1e-3,
            inference_tolerance=1e-3,
            max_it=50,
            epoch_start_hard_constrained=10,
            ada_np_auto_activation=False,
            ift_backward=True,
        )
        assert cfg.training_tolerance == pytest.approx(1e-3)
        assert cfg.inference_tolerance == pytest.approx(1e-3)
        assert cfg.max_it == 50
        assert cfg.epoch_start_hard_constrained == 10
        assert cfg.ada_np_auto_activation is False
        assert cfg.ift_backward is True

    def test_loss_custom(self):
        cfg = ENFORCEConfig(
            supervised=False,
            soft_constrained=True,
            weight_loss_displacement=0.3,
            weight_loss_soft=1.0,
        )
        assert cfg.supervised is False
        assert cfg.soft_constrained is True
        assert cfg.weight_loss_displacement == pytest.approx(0.3)
        assert cfg.weight_loss_soft == pytest.approx(1.0)

    def test_misc_custom(self):
        cfg = ENFORCEConfig(verbose=True, random_seed=123)
        assert cfg.verbose is True
        assert cfg.random_seed == 123


class TestENFORCEConfigDataclassProtocol:
    """ENFORCEConfig behaves as a standard dataclass."""

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ENFORCEConfig)

    def test_fields_present(self):
        field_names = {f.name for f in dataclasses.fields(ENFORCEConfig)}
        expected = {
            "input_neurons",
            "hidden_neurons",
            "output_neurons",
            "hidden_layers",
            "training_tolerance",
            "inference_tolerance",
            "max_it",
            "epoch_start_hard_constrained",
            "ada_np_auto_activation",
            "ift_backward",
            "regularise_gram",
            "supervised",
            "soft_constrained",
            "weight_loss_displacement",
            "weight_loss_soft",
            "verbose",
            "random_seed",
        }
        assert expected == field_names

    def test_dict_conversion(self):
        cfg = ENFORCEConfig(input_neurons=5, random_seed=7)
        d = cfg.__dict__
        assert d["input_neurons"] == 5
        assert d["random_seed"] == 7

    def test_replace_pattern(self):
        """The per-run seed-override pattern used in run_benchmark.py must work."""
        base = ENFORCEConfig(input_neurons=4, hidden_neurons=64, random_seed=42)
        per_run = ENFORCEConfig(**{**base.__dict__, "random_seed": 99})
        assert per_run.random_seed == 99
        assert per_run.input_neurons == 4
        assert per_run.hidden_neurons == 64

    def test_equality(self):
        assert ENFORCEConfig() == ENFORCEConfig()
        assert ENFORCEConfig(random_seed=1) != ENFORCEConfig(random_seed=2)
