import torch

from src.enforce.fb_inequality_constraints import FischerBurmeisterReformulation


def _g1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """g1 = y1*y2 + 2*y4 - 2.5*x3 <= 0"""
    return y[:, 0] * y[:, 1] + 2 * y[:, 3] - 2.5 * x[:, 2]


def _g2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """g2 = y1*y3 + 2*y5 - 1.5*x4 <= 0"""
    return y[:, 0] * y[:, 2] + 2 * y[:, 4] - 1.5 * x[:, 3]


class PoolingConstraints:
    """Picklable combined constraint callable for the pooling problem.

    6 residuals: 4 equality + 2 stabilised-FB inequality reformulations.
    nc=6 < no=7 (y1..y5 + λ1, λ2) ✓
    """

    def __init__(self, fb: FischerBurmeisterReformulation):
        self.fb = fb

    def __call__(self, x: torch.Tensor, y_ext: torch.Tensor) -> torch.Tensor:
        y = y_ext[:, :5]

        c1 = y[:, 1] + y[:, 2] - x[:, 0] - x[:, 1]                    # pool balance
        c2 = x[:, 2] - y[:, 1] - y[:, 3]                               # output-1 balance
        c3 = x[:, 3] - y[:, 2] - y[:, 4]                               # output-2 balance
        c4 = y[:, 0] * y[:, 1] + y[:, 0] * y[:, 2] - 3 * x[:, 0] - x[:, 1]  # quality (bilinear)
        eq = torch.stack([c1, c2, c3, c4], dim=1)                       # [BS, 4]

        fb_res = self.fb(x, y_ext)                                      # [BS, 2]
        return torch.cat([eq, fb_res], dim=1)                           # [BS, 6]


def make_pooling_constraints():
    """Build the FB wrapper and combined constraint callable for the pooling problem.

    Problem: 4 inputs (x1..x4), 5 outputs (y1..y5)
      - 4 equality constraints (mass + quality balances, bilinear c4)
      - 2 inequality constraints enforced via stabilised Fischer-Burmeister

    Returns
    -------
    fb : FischerBurmeisterReformulation
        Wrapper with n_original_outputs=5, n_ineq=2.
    constraints_full : PoolingConstraints
        Combined constraint callable ``c(x, y_ext) -> [BS, 6]``.
        nc=6 < no=7 (y1..y5 + λ1, λ2) ✓
    """
    fb = FischerBurmeisterReformulation(
        n_original_outputs=5,
        inequalities=[_g1, _g2],
    )
    return fb, PoolingConstraints(fb)
