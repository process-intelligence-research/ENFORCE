import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn import MSELoss

# Local imports
from enforce.losses.loss_core import LossCore, SSLLossCore
from enforce.core.scaler import Scaler
from enforce.core.constraints import Constraints

@dataclass
class ENFORCELossConfig:
    """Configuration for the ENFORCE loss function.

    Parameters
    ----------
    soft_constrained :
        If ``True``, add a soft constraint penalty to the loss.
    weight_loss_displacement :
        Weight for the projection-displacement term in the loss.
    weight_loss_soft :
        Weight for the soft-constraint penalty term.

    """

    # ── Loss ─────────────────────────────────────────────────────────────────
    soft_constrained: bool = False
    weight_loss_displacement: float = 0.5
    weight_loss_soft: float = 0.0


class ENFORCELoss(LossCore):
    """
    Enforce loss function with loss displacement and soft constrained loss.

    Parameters
    ----------
    scaler : Scaler
        :class:`Scaler` object for input/output normalisation.
    constraints : Constraints
        :class:`Constraints` object for constraint residual calculation.
    config : ENFORCELossConfig
        Optional ENFORCELossConfig to configure the ENFORCELoss.
    """

    def __init__(
        self,
        scaler: Scaler,
        constraints: Constraints,
        config: ENFORCELossConfig | None = None,
    ):
        self.cfg: ENFORCELossConfig = config if config is not None else ENFORCELossConfig()
        super().__init__(loss_function=MSELoss())
        self.scaler: Scaler = scaler
        self.constraints: Constraints = constraints

    # noinspection method-overriding
    def loss(self, ytilde: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None = None)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Computes the loss function from projection, prediction, outputs and inputs.
        Returns the loss total, loss after projection, loss displacement, and loss before projection.

        Parameters
        ----------
        ytilde : torch.Tensor
            Projected prediction tensor.
        yhat : torch.Tensor
            Prediction tensor.
        y : torch.Tensor
            Target tensor.
        x : torch.Tensor
            Optional input tensor. Needed for soft constraint penalty.

        Returns
        -------
        losses : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Loss (total, after projection, displacement, before projection).
        """
        loss_data_before_projection: torch.Tensor = self.data_loss(yhat, y)
        loss_data_after_projection: torch.Tensor  = self.data_loss(ytilde, y)
        loss_displacement: torch.Tensor  = torch.mean((yhat - ytilde) ** 2)  # this is zero if the projection is not done

        loss: torch.Tensor  = self.cfg.weight_loss_displacement * loss_displacement

        loss += loss_data_after_projection

        if self.cfg.soft_constrained:
            if x is None:
                warnings.warn("Soft constrained is set to true but no soft constraint penalty be applied "
                              "as the inputs were not passed correctly.")
            else:
                x_unscaled, ytilde_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=ytilde)
                c = self.constraints(x=x_unscaled, y_ext=ytilde_unscaled)
                loss += self.cfg.weight_loss_soft * torch.mean(torch.abs(c))


        return (
            loss,
            loss_data_after_projection,
            loss_displacement,
            loss_data_before_projection,
        )

    # noinspection method-overriding
    def forward(self, ytilde: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor, x=None) -> torch.Tensor:
        """ Computes the loss function from projection, prediction, outputs and inputs.
            Returns only the total loss.

            Parameters
            ----------
            ytilde : torch.Tensor
                Projected prediction tensor.
            yhat : torch.Tensor
                Prediction tensor.
            y : torch.Tensor
                Target tensor.
            x : torch.Tensor
                Optional input tensor. Needed for soft constraint penalty.

            Returns
            -------
            loss : torch.Tensor
                Total loss.
            """
        loss, _, _, _,  = self.loss(ytilde, yhat, y, x)
        return loss


class ENFORCESSLLoss(SSLLossCore, ABC):
    """
    Enforce self-supervised learning loss function with loss displacement, soft constrained, and self-supervised losses.

    Parameters
    ----------
    scaler : Scaler
        :class:`Scaler` object for input/output normalisation.
    constraints : Constraints
        :class:`Constraints` object for constraint residual calculation.
    config : ENFORCELossConfig
        Optional ENFORCELossConfig to configure the :class:`ENFORCESSLLoss`.
    """

    def __init__(
            self,
            scaler: Scaler,
            constraints: Constraints,
            config: ENFORCELossConfig = None,
    ):
        cfg = config if config is not None else ENFORCELossConfig()
        self.cfg = cfg
        super().__init__(scaler=scaler)
        self.constraints = constraints
        self.loss_function = MSELoss()

    def data_loss(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ MSE between prediction and target (logging only), torch convention ``(input, target)``. """
        return self.loss_function(yhat, y)

    def loss(self, ytilde: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor, x: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Computes the self supervised learning loss function from projection, prediction, outputs and inputs.
        Returns the loss total, loss after projection, loss displacement, and loss before projection.

        Parameters
        ----------
        ytilde : torch.Tensor
            Projected prediction tensor.
        yhat : torch.Tensor
            Prediction tensor.
        y : torch.Tensor
            Target tensor.
        x : torch.Tensor
            Input tensor. Needed for self supervised loss and soft constraint penalty.

        Returns
        -------
        losses : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Loss (total, after projection, displacement, before projection).
        """
        loss_data_before_projection: torch.Tensor = self.data_loss(yhat, y)
        loss_data_after_projection: torch.Tensor = self.data_loss(ytilde, y)
        loss_displacement: torch.Tensor = torch.mean((yhat - ytilde) ** 2)  # this is zero if the projection is not done
        loss: torch.Tensor = self.cfg.weight_loss_displacement * loss_displacement

        x_unscaled, ytilde_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=ytilde)
        loss_ssl: torch.Tensor = self.ssl_loss(x=x_unscaled, yhat=ytilde_unscaled)
        loss += loss_ssl

        if self.cfg.soft_constrained:
            c = self.constraints(x=x_unscaled, y_ext=ytilde_unscaled)
            loss += self.cfg.weight_loss_soft * torch.mean(torch.abs(c))

        return (
            loss,
            loss_data_after_projection,
            loss_displacement,
            loss_data_before_projection,
        )

    # noinspection method-overriding
    def forward(self, ytilde: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ Computes the self supervised learning loss function from projection, prediction, outputs and inputs.
        Returns only the total loss.

        Parameters
        ----------
        ytilde : torch.Tensor
            Projected prediction tensor.
        yhat : torch.Tensor
            Prediction tensor.
        y : torch.Tensor
            Target tensor.
        x : torch.Tensor
            Input tensor. Needed for self supervised loss and soft constraint penalty.

        Returns
        -------
        loss : torch.Tensor
            Total loss
        """
        loss, _, _, _,  = self.loss(ytilde, yhat, y, x)
        return loss
