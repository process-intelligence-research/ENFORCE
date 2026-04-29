import math

import torch

from enforce.core.fb_inequality_constraints import FischerBurmeisterReformulation


def _g_upper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Upper bound: g₁(x,y) = y - (1 + x²/(3π²)) ≤ 0  (quadratic amplitude envelope)."""
    return y[:, 0] - (1.0 + x[:, 0] ** 2 / (3.0 * math.pi ** 2))


def _g_lower(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lower bound: g₂(x,y) = -(1 + x²/(3π²)) - y ≤ 0  (quadratic amplitude envelope)."""
    return -(1.0 + x[:, 0] ** 2 / (3.0 * math.pi ** 2)) - y[:, 0]


def make_sin_ineq_constraints():
    """Build the FB wrapper for the sinusoidal inequality demo.

    Problem: 1 input (x), 1 output (y).

    True function: f(x) = (1 + x²/(3π²))·sin(x)  on  x ∈ [0, 3π].
    Amplitude grows quadratically: A(x) = 1 + x²/(3π²)  →  from 1 at x=0 to 4 at x=3π.

    Constraints are the EXACT QUADRATIC AMPLITUDE ENVELOPE:
        g₁(x,y) = y   − (1 + x²/(3π²)) ≤ 0   (upper: y ≤ A(x))
        g₂(x,y) = −(1 + x²/(3π²)) − y  ≤ 0   (lower: y ≥ −A(x))

    The true function always satisfies both constraints (touches them at peaks/troughs).
    Training labels have sign-correlated noise: positive bias when f(x)>0 and negative
    bias when f(x)<0.  The MLP, fitting these asymmetrically noisy labels, learns a
    biased predictor that violates the envelope; ENFORCE projects back.

    The ∂g/∂y Jacobian is identical to the constant-bound case ([1; −1]), so the
    Newton projection works just as efficiently.

    Returns
    -------
    fb : FischerBurmeisterReformulation
        Wrapper with n_original_outputs=1, n_ineq=2.
    c  : FischerBurmeisterReformulation
        Constraint callable (same object — no separate equality constraints).
    """
    fb = FischerBurmeisterReformulation(
        n_original_outputs=1,
        inequalities=[_g_upper, _g_lower],
    )
    return fb, fb
