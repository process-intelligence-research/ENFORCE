"""Benchmark problem factories should expose the current Constraints API."""

import pytest
import torch

from benchmark_problems.engineering_problems.extraction_column import (
    constraints_column,
    make_extraction_column_constraints,
)
from benchmark_problems.engineering_problems.pooling import make_pooling_constraints
from benchmark_problems.function_fitting.inequality.sin_ineq import make_sin_ineq_constraints
from benchmark_problems.parametric_optimization.opt_problem import NonconvexProgram, NonlinearProgram
from enforce.core.constraints import Constraint, Constraints


def test_extraction_column_block_matches_existing_residuals():
    x = torch.rand((3, 3))
    y = torch.rand((3, 9))
    constraints = make_extraction_column_constraints()

    assert (constraints.nc, constraints.ng) == (6, 0)
    torch.testing.assert_close(constraints(x, y), torch.stack(constraints_column(x, y), dim=1))


def test_pooling_factory_returns_combined_equality_and_inequality_constraints():
    constraints = make_pooling_constraints()

    assert isinstance(constraints, Constraints)
    assert (constraints.nc, constraints.ng) == (4, 2)

    x = torch.tensor([[1.0, 2.0, 4.0, 5.0]])
    y = torch.tensor([[0.5, 1.0, 2.0, 1.0, 1.0]])
    y_ext = constraints.extend_outputs(y)
    residuals = constraints(x, y_ext)

    expected_equalities = torch.tensor([[0.0, 2.0, 2.0, -3.5]])
    torch.testing.assert_close(residuals[:, :4], expected_equalities)
    torch.testing.assert_close(constraints.g_values(x, y), torch.tensor([[-7.5, -4.5]]))
    torch.testing.assert_close(residuals[:, 4:], constraints.g(x, y_ext))


def test_sinusoidal_inequality_factory_returns_constraints_and_extends_labels():
    constraints = make_sin_ineq_constraints()

    assert isinstance(constraints, Constraints)
    assert (constraints.nc, constraints.ng) == (0, 2)

    labels = torch.tensor([[0.25], [-0.5]])
    extended = constraints.extend_outputs(labels)
    assert extended.shape == (2, 3)
    torch.testing.assert_close(extended[:, :1], labels)
    torch.testing.assert_close(extended[:, 1:], torch.zeros((2, 2)))


@pytest.mark.parametrize("program", ["linear", "nonlinear"])
def test_optimization_constraints_are_one_block_with_matching_analytic_jacobian(program):
    torch.manual_seed(0)
    if program == "linear":
        problem = NonconvexProgram.__new__(NonconvexProgram)
        problem._c = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    else:
        problem = NonlinearProgram.__new__(NonlinearProgram)
        A = torch.randn(2, 3, 3)
        problem._A = A + A.transpose(1, 2)  # the analytic Jacobian assumes symmetric A_i
        problem._c = torch.randn(2, 3)
        problem._d = torch.randn(2)
    constraints = problem.constraints()
    x, y = torch.randn(4, 2), torch.randn(4, 3)

    assert (constraints.nc, constraints.ng) == (2, 0)
    torch.testing.assert_close(constraints(x, y), problem.eq_constraints(x, y))
    autograd = Constraints(c=[Constraint(problem.eq_constraints, size=2)]).jacobian(x, y)
    torch.testing.assert_close(constraints.jacobian(x, y), autograd)
