"""Constraint blocks: sizes, shapes, residuals and the assembled Jacobian."""

import pytest
import torch

from enforce import Constraint, Constraints


def _y_ext(constraints, bs=5, no=4, seed=0):
    torch.manual_seed(seed)
    x, y = torch.randn(bs, 3), torch.randn(bs, no)
    lambdas = torch.rand(bs, constraints.ng)
    return x, torch.cat([y, lambdas], dim=1)


def _reference_jacobian(constraints, x, y_ext):
    """Per-sample autograd Jacobian of the full residual vector."""
    rows = [torch.autograd.functional.jacobian(lambda v: constraints(x[b:b + 1], v.unsqueeze(0))[0], y_ext[b])
            for b in range(x.shape[0])]
    return torch.stack(rows)


def test_sizes_are_known_without_evaluating_the_constraints():
    def never_called(x, y):
        raise AssertionError("evaluated at construction")

    constraints = Constraints(
        c=[never_called, Constraint(never_called, size=3)],
        g=[Constraint(never_called, size=2), never_called],
    )
    assert (constraints.nc, constraints.ng) == (4, 3)


@pytest.mark.parametrize("fn", [
    lambda x, y: y[:, :2],                      # [BS, size]
    lambda x, y: (y[:, 0], y[:, 1]),            # tuple of [BS]
])
def test_block_accepts_matrix_and_tuple_outputs(fn):
    x, y = torch.randn(3, 1), torch.randn(3, 4)
    torch.testing.assert_close(Constraint(fn, size=2)(x, y), y[:, :2])


def test_block_with_wrong_declared_size_fails_with_its_name():
    def two_equations(x, y):
        return y[:, :2]

    constraints = Constraints(c=[two_equations])  # size defaults to 1
    with pytest.raises(ValueError, match=r"two_equations.*\[BS, 1\]"):
        constraints(torch.randn(3, 1), torch.randn(3, 4))


def test_block_with_wrong_batch_size_fails_with_its_name():
    def first_sample_only(x, y):
        return y[:1, :2]

    constraints = Constraints(c=[Constraint(first_sample_only, size=2)])
    with pytest.raises(ValueError, match=r"first_sample_only.*\[BS, 2\]"):
        constraints(torch.randn(3, 1), torch.randn(3, 4))


def test_size_must_be_positive():
    with pytest.raises(ValueError, match="size"):
        Constraint(lambda x, y: y, size=0)


def test_block_and_scalar_callables_give_the_same_residuals():
    A = torch.randn(3, 4)
    scalars = Constraints(c=[lambda x, y, i=i: y @ A[i] - x[:, i] for i in range(3)])
    block = Constraints(c=[Constraint(lambda x, y: y @ A.T - x, size=3)])
    x, y = torch.randn(5, 3), torch.randn(5, 4)
    torch.testing.assert_close(block(x, y), scalars(x, y))


def test_subclassed_block_with_analytic_jacobian():
    class Linear(Constraint):
        def __init__(self, A):
            super().__init__(size=A.shape[0])
            self.A = A

        def residual(self, x, y):
            return y @ self.A.T - x

        def jacobian(self, x, y):
            return self.A.expand(y.shape[0], -1, -1)

    A = torch.randn(3, 4)
    constraints = Constraints(c=[Linear(A)])
    x, y = torch.randn(5, 3), torch.randn(5, 4)
    torch.testing.assert_close(constraints(x, y), y @ A.T - x)
    torch.testing.assert_close(constraints.jacobian(x, y), A.expand(5, -1, -1))


def test_g_values_are_the_raw_inequalities_and_g_the_fb_residuals():
    constraints = Constraints(g=[Constraint(lambda x, y: y[:, :2] - 1.0, size=2), lambda x, y: -y[:, 2]])
    x, y_ext = _y_ext(constraints)
    y, lambdas = y_ext[:, :4], y_ext[:, 4:]
    g = torch.cat([y[:, :2] - 1.0, -y[:, 2:3]], dim=1)
    torch.testing.assert_close(constraints.g_values(x, y), g)
    expected_fb = torch.sqrt(lambdas ** 2 + g ** 2 + constraints.eps_fb) - lambdas + g
    torch.testing.assert_close(constraints.g(x, y_ext), expected_fb)


def _mixed_constraints(with_jac):
    A = torch.randn(2, 4)

    def eq_block(x, y):
        return y @ A.T + y[:, :2] ** 2 - x[:, :2]

    def eq_jac(x, y):
        return A.expand(y.shape[0], -1, -1) + torch.diag_embed(torch.nn.functional.pad(2 * y[:, :2], (0, 2)))[:, :2]

    def ineq_block(x, y):
        return torch.stack([y[:, 0] * y[:, 1] - x[:, 2], torch.sin(y[:, 3])], dim=1)

    def ineq_jac(x, y):
        jac = torch.zeros(y.shape[0], 2, 4)
        jac[:, 0, 0], jac[:, 0, 1], jac[:, 1, 3] = y[:, 1], y[:, 0], torch.cos(y[:, 3])
        return jac

    return Constraints(
        c=[Constraint(eq_block, size=2, jac=eq_jac if with_jac else None), lambda x, y: y[:, 2] ** 3],
        g=[Constraint(ineq_block, size=2, jac=ineq_jac if with_jac else None), lambda x, y: -y[:, 2]],
    )


@pytest.mark.parametrize("with_jac", [False, True])
def test_jacobian_matches_autograd_of_the_full_residual(with_jac):
    torch.manual_seed(1)
    constraints = _mixed_constraints(with_jac)
    x, y_ext = _y_ext(constraints)
    jac = constraints.jacobian(x, y_ext)
    assert jac.shape == (5, constraints.nc + constraints.ng, 4 + constraints.ng)
    torch.testing.assert_close(jac, _reference_jacobian(constraints, x, y_ext))



def test_fb_jacobian_is_bitwise_equal_to_batched_autograd():
    """The FB rows follow autograd's operations, so float32 results and their gradients are bitwise equal.

    The adaptive projection amplifies any rounding difference, so this keeps training
    trajectories identical to differentiating the full residual by autograd.
    """
    torch.manual_seed(2)
    constraints = _mixed_constraints(with_jac=False)
    x, y_ext = _y_ext(constraints, bs=2000)
    weights = torch.randn(2000, constraints.nc + constraints.ng, y_ext.shape[1])

    def autograd_jacobian(y):
        c = constraints(x, y)
        rows = []
        for i in range(c.shape[1]):
            grad_outputs = torch.zeros_like(c)
            grad_outputs[:, i] = 1
            rows.append(torch.autograd.grad(c, y, grad_outputs, create_graph=True, retain_graph=True)[0])
        return torch.stack(rows, dim=1)

    jacobians, grads = [], []
    for jacobian_fn in (autograd_jacobian, lambda y: constraints.jacobian(x, y)):
        y = y_ext.clone().requires_grad_(True)
        jac = jacobian_fn(y)
        (jac * weights).sum().backward()  # training backpropagates through the Jacobian
        jacobians.append(jac.detach())
        grads.append(y.grad)
    assert torch.equal(jacobians[0], jacobians[1])
    assert torch.equal(grads[0], grads[1])

def test_jacobian_is_differentiable_for_training():
    constraints = _mixed_constraints(with_jac=False)
    x, y_ext = _y_ext(constraints)
    y_ext.requires_grad_(True)
    constraints.jacobian(x, y_ext).square().sum().backward()
    assert y_ext.grad is not None and torch.isfinite(y_ext.grad).all()


def test_jacobian_of_constraints_independent_of_some_outputs():
    constraints = Constraints(c=[lambda x, y: x[:, 0]])
    x, y = torch.randn(2, 3), torch.randn(2, 4, requires_grad=True)
    torch.testing.assert_close(constraints.jacobian(x, y), torch.zeros(2, 1, 4))
