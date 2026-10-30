import warnings
from abc import ABC

import torch
import torch.nn as nn

from enforce.core.constraints import Constraints
from enforce.core.scaler import Scaler
from enforce.losses.loss_core import LossCore, SSLLossCore


def soft_constraint_penalty(constraints: Constraints, x: torch.Tensor, y_ext: torch.Tensor) -> torch.Tensor:
    """Soft constraint penalty for unconstrained backbones.

    ``mean(|c(x, y)|) + sum_i mean(max(0, g_i(x, y))^2)``, summed over all scalar inequalities

    Both terms are evaluated on the original outputs ``y`` (the FB multiplier
    columns of ``y_ext`` are dropped). Without inequalities, the equality term matches the soft term of
    :class:`ENFORCELoss`; the squared hinge on the inequalities is the penalty used
    for the soft-constraint baselines in the paper.

    Parameters
    ----------
    constraints : Constraints
        :class:`Constraints` object holding the equality and inequality constraints.
    x : torch.Tensor
        Unscaled input tensor, shape ``[BS, NI]``.
    y_ext : torch.Tensor
        Unscaled (extended) output tensor, shape ``[BS, NO+NG]``.

    Returns
    -------
    penalty : torch.Tensor
        Scalar penalty.
    """
    y: torch.Tensor = y_ext[:, : y_ext.shape[1] - constraints.ng]
    penalty: torch.Tensor = torch.zeros((), device=y_ext.device, dtype=y_ext.dtype)
    if constraints.nc > 0:
        penalty = penalty + torch.mean(torch.abs(constraints.c(x, y)))
    if constraints.ng > 0:
        penalty = penalty + torch.sum(torch.mean(torch.clamp(constraints.g_values(x, y), min=0.0) ** 2, dim=0))
    return penalty


class SoftConstrainedLoss(LossCore):
    """
    MSE loss with a soft constraint penalty, for unconstrained backbones.

    ``loss = MSE(yhat, y) + weight_loss_soft * soft_constraint_penalty(x, yhat)``

    The penalty is computed on unscaled values, so the inputs ``x`` must be passed.
    With inequalities, ``yhat`` and ``y`` have the extended width ``NO + NG`` of the scaler: extend the
    labels with :meth:`Constraints.extend_outputs` and let the backbone predict ``NO + NG`` outputs.
    For a :class:`ConstrainedModel` use :class:`ENFORCELoss` with
    ``ENFORCELossConfig.soft_constrained`` instead.

    Parameters
    ----------
    scaler : Scaler
        :class:`Scaler` object for input/output normalisation.
    constraints : Constraints
        :class:`Constraints` object for the penalty.
    weight_loss_soft : float
        Weight of the soft constraint penalty. ``0`` skips the penalty computation.
    """

    def __init__(
        self,
        scaler: Scaler,
        constraints: Constraints,
        weight_loss_soft: float = 1.0,
    ):
        super().__init__(loss_function=nn.MSELoss())
        self.scaler: Scaler = scaler
        self.constraints: Constraints = constraints
        self.weight_loss_soft: float = weight_loss_soft

    def loss(self, yhat: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        loss: torch.Tensor = self.data_loss(yhat, y)
        if self.weight_loss_soft == 0.0:
            return loss
        if x is None:
            warnings.warn("No inputs passed to SoftConstrainedLoss. The soft constraint penalty is not applied.")
            return loss
        x_unscaled, yhat_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=yhat)
        return loss + self.weight_loss_soft * soft_constraint_penalty(self.constraints, x_unscaled, yhat_unscaled)


class SoftConstrainedSSLLoss(SSLLossCore, ABC):
    """
    Self-supervised loss with a soft constraint penalty, for unconstrained backbones.
    Derived classes must implement the ssl_loss method.

    ``loss = ssl_loss(x, yhat) + weight_loss_soft * soft_constraint_penalty(x, yhat)``

    Parameters
    ----------
    scaler : Scaler
        :class:`Scaler` object for input/output normalisation.
    constraints : Constraints
        :class:`Constraints` object for the penalty.
    weight_loss_soft : float
        Weight of the soft constraint penalty. ``0`` skips the penalty computation.
    """

    def __init__(
        self,
        scaler: Scaler,
        constraints: Constraints,
        weight_loss_soft: float = 1.0,
    ):
        super().__init__(scaler=scaler)
        self.constraints: Constraints = constraints
        self.weight_loss_soft: float = weight_loss_soft

    def forward(self, x: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        x_unscaled, yhat_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=yhat)
        ssl_loss: torch.Tensor = self.ssl_loss(x=x_unscaled, yhat=yhat_unscaled)
        if self.weight_loss_soft == 0.0:
            return ssl_loss
        return ssl_loss + self.weight_loss_soft * soft_constraint_penalty(
            self.constraints, x_unscaled, yhat_unscaled
        )
