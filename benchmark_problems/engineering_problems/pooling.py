import torch

from enforce.core.constraints import Constraints


def _g1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """g1 = y1*y2 + 2*y4 - 2.5*x3 <= 0"""
    return y[:, 0] * y[:, 1] + 2 * y[:, 3] - 2.5 * x[:, 2]


def _g2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """g2 = y1*y3 + 2*y5 - 1.5*x4 <= 0"""
    return y[:, 0] * y[:, 2] + 2 * y[:, 4] - 1.5 * x[:, 3]


def _c1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pool balance equality."""
    return y[:, 1] + y[:, 2] - x[:, 0] - x[:, 1]


def _c2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """First output balance equality."""
    return x[:, 2] - y[:, 1] - y[:, 3]


def _c3(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Second output balance equality."""
    return x[:, 3] - y[:, 2] - y[:, 4]


def _c4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bilinear quality balance equality."""
    return y[:, 0] * y[:, 1] + y[:, 0] * y[:, 2] - 3 * x[:, 0] - x[:, 1]


def make_pooling_constraints():
    """Build the equality and inequality constraints for the pooling problem.

    Problem: 4 inputs (x1..x4), 5 outputs (y1..y5)
      - 4 equality constraints (mass + quality balances, bilinear c4)
      - 2 inequality constraints enforced by ENFORCE's FB reformulation

    Returns
    -------
    constraints : Constraints
        Four equality and two inequality functions. The model appends the
        inequality multipliers to the five physical outputs.
    """
    return Constraints(c=[_c1, _c2, _c3, _c4], g=[_g1, _g2])
