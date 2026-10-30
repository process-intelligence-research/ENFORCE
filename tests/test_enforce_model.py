"""Tests for the current backbone + constraint-module architecture.

Design notes
------------
- Constraints must satisfy nc < no (underdetermined) or nc == no (determined,
  which triggers a warning but is allowed).
- Do NOT wrap model.predict() calls in torch.no_grad(): the Newton projection
  requires autograd to compute constraint Jacobians.
- Forward-only tests (no projection) use model.forward() directly or
  constrained=False so no Jacobian is needed.
"""

import warnings

import pytest
import torch
import torch.nn as nn

from enforce import ENFORCE, Constraint, Constraints, ENFORCEConfig, ENFORCELoss, Scaler
from enforce.models import MLP, ConstrainedModel, ENFORCEConstrainedMLP, MLPConfig


def _system(ni=2, no=4, nc=1, *, seed=0, weighting="ID", config=None):
    scaler = Scaler((torch.zeros(ni), torch.ones(ni)), (torch.zeros(no), torch.ones(no)))
    constraints = Constraints(c=[lambda x, y, i=i: y[:, i] for i in range(nc)])
    loss = ENFORCELoss(scaler, constraints)
    layer = ENFORCE(scaler, constraints, loss, config or ENFORCEConfig(random_seed=seed, weighting_option=weighting))
    mlp_config = MLPConfig(ni, 16, no, 1, seed)
    model = ENFORCEConstrainedMLP(mlp_config, layer)
    return scaler, constraints, loss, layer, model


def test_enforce_stores_config_and_shared_objects():
    cfg = ENFORCEConfig()
    _, constraints, loss, layer, model = _system(config=cfg)
    assert layer.cfg is cfg
    assert layer.constraints is constraints
    assert layer.loss is loss
    assert model.cfg is cfg
    assert layer.ift_backward is False


def test_default_enforce_config_is_created():
    scaler, constraints, _, _, _ = _system()
    layer = ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints))
    assert layer.cfg == ENFORCEConfig()


def test_mlp_architecture_and_forward_dimensions():
    cfg = MLPConfig(input_neurons=3, hidden_neurons=32, output_neurons=5, hidden_layers=3)
    mlp = MLP(cfg)
    assert mlp.input_layer.in_features == 3
    assert mlp.input_layer.out_features == 32
    assert len(mlp.hidden_layers) == 3
    assert all(layer.in_features == layer.out_features == 32 for layer in mlp.hidden_layers)
    assert mlp.output_layer.out_features == 5
    assert isinstance(mlp.hidden_activation, nn.ReLU)
    assert mlp(torch.randn(8, 3)).shape == (8, 5)


def test_model_returns_constrained_and_raw_predictions():
    _, _, _, layer, model = _system(ni=2, no=4, nc=1)
    x = torch.randn(7, 2)
    ytilde, yhat = model(x)
    assert ytilde.shape == yhat.shape == (7, 4)
    assert model.backbone is not None
    assert model.constraint_module is layer


def test_composition_supports_any_constraint_module():
    backbone = nn.Linear(2, 3)
    model = ConstrainedModel(backbone, nn.Identity())
    constrained, raw = model(torch.randn(4, 2))
    assert torch.equal(constrained, raw)


def test_trainable_fixed_weighting_is_parameterized_by_output_width():
    _, _, _, layer, _ = _system(no=4, weighting="TF")
    assert isinstance(layer.proj_weight_log_diag, nn.Parameter)
    assert layer.proj_weight_log_diag.shape == (4,)


def test_other_weighting_modes_do_not_add_trainable_parameter():
    for weighting in ("I", "BA", "ID", "R"):
        *_, layer, _ = _system(weighting=weighting)
        assert not hasattr(layer, "proj_weight_log_diag")


def test_constraints_extend_extract_and_compute_fb_residuals():
    def g(x, y):
        return -y[:, 0]

    constraints = Constraints(g=[g])
    x, y = torch.zeros(3, 1), torch.tensor([[1.], [0.], [-1.]])
    extended = constraints.extend_outputs(y)
    assert extended.shape == (3, 2)
    assert torch.equal(extended[:, 1], torch.zeros(3))
    assert torch.equal(constraints.extract_outputs(extended), y)
    residuals = constraints(x, extended)
    assert residuals.shape == (3, 1)
    assert torch.isfinite(residuals).all()


def test_enforce_constrained_model_predict_reports_projection_count():
    scaler, constraints, loss, _, _ = _system(ni=2, no=3, nc=1)
    layer = ENFORCE(
        scaler,
        constraints,
        loss,
        ENFORCEConfig(epoch_start_hard_constrained=5, random_seed=1),
    )
    model = ENFORCEConstrainedMLP(MLPConfig(2, 8, 3, 1, 1), layer)
    x = torch.randn(6, 2)
    ytilde, yhat, iterations = model.predict(x)
    assert ytilde.shape == yhat.shape == (6, 3)
    assert iterations == 0


def test_inference_projection_enforces_linear_equality_and_preserves_null_space():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]])
    loss = ENFORCELoss(scaler, constraints)
    layer = ENFORCE(
        scaler,
        constraints,
        loss,
        ENFORCEConfig(
            weighting_option="I",
            inference_tolerance=1e-5,
            max_it=10,
            ada_np_auto_activation=False,
        ),
    )
    layer.eval()
    x = torch.zeros(2, 1)
    yhat = torch.tensor([[2.0, 0.0], [3.0, 1.0]])

    projected, original, iterations = layer.project(x, yhat)

    torch.testing.assert_close(projected, torch.tensor([[1.0, 1.0], [2.0, 2.0]]), atol=1e-5, rtol=0)
    torch.testing.assert_close(original, yhat)
    torch.testing.assert_close(constraints(x, projected), torch.zeros(2, 1), atol=1e-5, rtol=0)
    assert iterations > 0


def test_inequality_projection_extends_outputs_and_enforces_complementarity():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
    constraints = Constraints(g=[lambda x, y: y[:, 0] - 1.0])
    loss = ENFORCELoss(scaler, constraints)
    layer = ENFORCE(
        scaler,
        constraints,
        loss,
        ENFORCEConfig(
            weighting_option="I",
            inference_tolerance=1e-3,
            max_it=20,
            ada_np_auto_activation=False,
        ),
    )
    layer.eval()
    x, yhat = torch.zeros(1, 1), torch.tensor([[2.0]])

    projected, original, iterations = layer.project(x, yhat)

    assert projected.shape == (1, 2)
    torch.testing.assert_close(original[:, :1], yhat)
    assert projected[0, 1] >= 0
    assert constraints.g_values(x, projected[:, :1]).item() <= 1e-3
    assert constraints(x, projected).abs().max().item() <= 1e-3
    assert iterations > 0


def test_training_projection_keeps_gradients_finite_and_tangent_to_constraint():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]])
    layer = ENFORCE(
        scaler,
        constraints,
        ENFORCELoss(scaler, constraints),
        ENFORCEConfig(weighting_option="I", ada_np_auto_activation=False, training_tolerance=1e-5),
    )
    x = torch.zeros(2, 1)
    yhat = torch.tensor([[2.0, 0.0], [3.0, 1.0]], requires_grad=True)

    projected = layer(x, yhat, y=torch.zeros_like(yhat))
    projected.square().sum().backward()

    torch.testing.assert_close(constraints(x, projected), torch.zeros(2, 1), atol=1e-5, rtol=0)
    assert yhat.grad is not None and torch.isfinite(yhat.grad).all()
    torch.testing.assert_close(yhat.grad[:, 0], yhat.grad[:, 1], atol=1e-5, rtol=0)


def test_analytic_constraint_jacobian_matches_autograd_jacobian():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(3), torch.ones(3)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] + 2.0 * y[:, 1]])
    loss = ENFORCELoss(scaler, constraints)
    x = torch.zeros(2, 1)
    y = torch.randn(2, 3, requires_grad=True)
    expected = torch.tensor([1.0, 2.0, 0.0]).reshape(1, 1, 3).expand(2, -1, -1)

    autodiff_layer = ENFORCE(scaler, constraints, loss)
    autodiff = autodiff_layer.compute_dc_dy(x, y)
    analytic_constraints = Constraints(
        c=[Constraint(lambda x, y: y[:, 0] + 2.0 * y[:, 1], jac=lambda x, y: expected)]
    )
    analytic_layer = ENFORCE(scaler, analytic_constraints, ENFORCELoss(scaler, analytic_constraints))
    analytic = analytic_layer.compute_dc_dy(x, y)

    torch.testing.assert_close(autodiff, expected)
    torch.testing.assert_close(analytic, expected)


def test_projection_tensor_solver_falls_back_for_rank_deficient_jacobian():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
    constraints = Constraints(c=[lambda x, y: y[:, 0]])
    layer = ENFORCE(
        scaler,
        constraints,
        ENFORCELoss(scaler, constraints),
        ENFORCEConfig(weighting_option="I", regularise_gram=False, eps_chol=1e-5),
    )
    layer.nc = 1
    B = torch.zeros(2, 1, 2)
    v = torch.ones(2, 1)

    B_star, v_star = layer.projection_tensors(B, v)

    assert torch.isfinite(B_star).all() and torch.isfinite(v_star).all()
    torch.testing.assert_close(B_star, torch.eye(2).expand(2, -1, -1))
    torch.testing.assert_close(v_star, torch.zeros(2, 2))


@pytest.mark.parametrize("weighting", ["ID", "BA", "R", "TF"])
def test_projection_weighting_modes_produce_positive_diagonal_matrices(weighting):
    _, _, _, layer, _ = _system(no=3, weighting=weighting)
    layer.dc_dy = torch.tensor([[[2.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]]])

    weights = layer.Wi_f()

    assert weights.shape == (3, 3) if weighting in {"BA", "TF"} else weights.shape == (2, 3, 3)
    diagonal = weights.diagonal(dim1=-2, dim2=-1)
    assert (diagonal > 0).all()
    off_diagonal = weights - torch.diag_embed(diagonal)
    torch.testing.assert_close(off_diagonal, torch.zeros_like(off_diagonal))


def test_identity_weighting_uses_implicit_identity_matrix():
    _, _, _, layer, _ = _system(no=3, weighting="I")
    layer.dc_dy = torch.ones(2, 1, 3)
    assert layer.Wi_f() is None


def test_enforce_rejects_more_constraints_than_outputs():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
    constraints = Constraints(c=[lambda x, y: y[:, i] for i in range(3)])
    with pytest.raises(ValueError, match="Too many constraints"):
        ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints))


def test_enforce_rejects_empty_constraints():
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
    constraints = Constraints()
    with pytest.raises(ValueError, match="at least one constraint"):
        ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints))


def test_compute_residual_rejects_unknown_tolerance_mode():
    _, _, _, layer, _ = _system()
    with pytest.raises(ValueError, match="Invalid tolerance mode"):
        layer.compute_residual(torch.ones(2, 1), tolerance_mode="median")


def test_verbose_self_supervised_projection_without_soft_constraints():
    from enforce import ENFORCELossConfig
    from enforce.losses import ENFORCESSLLoss

    class Objective(ENFORCESSLLoss):
        def ssl_loss(self, x, yhat):
            return torch.mean((yhat[:, 0] - yhat[:, 1]) ** 2)  # minimised on the constraint manifold

    scaler = Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(3), torch.ones(3)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]])
    loss = Objective(scaler, constraints, ENFORCELossConfig(soft_constrained=False))
    layer = ENFORCE(scaler, constraints, loss, ENFORCEConfig(verbose=True))
    layer.train()
    x, yhat, y = torch.randn(5, 2), torch.randn(5, 3), torch.randn(5, 3)
    ytilde, _, _ = layer.project(x, yhat, y)
    assert torch.allclose(ytilde[:, 0], ytilde[:, 1], atol=1e-5)


def test_analytic_jacobian_sets_number_of_constraints_not_inputs():
    scaler = Scaler((torch.zeros(3), torch.ones(3)), (torch.zeros(4), torch.ones(4)))

    def jac(x, y):
        dc_dy = torch.zeros(y.shape[0], 1, 4, dtype=y.dtype)
        dc_dy[:, 0, 0], dc_dy[:, 0, 1] = 1.0, -1.0
        return dc_dy

    constraints = Constraints(c=[Constraint(lambda x, y: y[:, 0] - y[:, 1], jac=jac)])
    cfg = ENFORCEConfig(weighting_option="I", regularise_gram=True, eps_chol=1e-8)
    layer = ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints), cfg)
    layer.eval()
    x, yhat = torch.randn(5, 3), torch.randn(5, 4)
    ytilde, _, _ = layer.project(x, yhat)
    assert layer.nc == 1
    assert torch.allclose(ytilde[:, 0], ytilde[:, 1], atol=1e-5)


def test_ada_np_reports_its_own_iteration_cap(capsys):
    scaler = Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(3), torch.ones(3)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] ** 2 + y[:, 1] - 1.0])
    layer = ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints), ENFORCEConfig(max_it=100))
    y = torch.randn(4, 3, requires_grad=True)
    _, proj_iter = layer.ada_np(torch.randn(4, 2), y, tolerance_mode="mean", tolerance_value=0.0, max_iter=2)
    assert proj_iter == 2
    assert "Max projection iteration reached (2)" in capsys.readouterr().out


def test_enforce_does_not_modify_the_callers_config():
    scaler = Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(3), torch.ones(3)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] - y[:, 1]])
    cfg = ENFORCEConfig(verbose=True)
    with pytest.warns(UserWarning, match="Verbose"):
        layer = ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints), cfg)
    assert cfg.verbose is True
    assert layer.verbose is False


def _fb_system(no, nc, ng):
    # Scaler width includes one multiplier column per inequality (extended outputs)
    scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(no + ng), torch.ones(no + ng)))
    constraints = Constraints(
        c=[lambda x, y, i=i: y[:, i % no] - 1.0 for i in range(nc)] or None,
        g=[lambda x, y, i=i: -y[:, i % no] for i in range(ng)] or None,
    )
    return scaler, constraints


def test_equalities_are_counted_against_original_outputs_only():
    scaler, constraints = _fb_system(no=2, nc=3, ng=2)  # 3 equalities on 2 real outputs
    with pytest.raises(ValueError, match="Too many constraints"):
        ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints))


def test_inequalities_do_not_use_degrees_of_freedom():
    scaler, constraints = _fb_system(no=2, nc=1, ng=3)  # more inequalities than outputs is fine
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints))


def test_determined_warning_uses_original_outputs():
    scaler, constraints = _fb_system(no=2, nc=2, ng=1)
    with pytest.warns(UserWarning, match="determined"):
        ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints))
