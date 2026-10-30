import warnings
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from enforce.core.scaler import Scaler


class LossCore(nn.Module, ABC):
    """
    Abstract core loss function. Holds the loss function and losses.
    All derived classes must implement the loss method.

    Parameters
    ----------
    loss_function: nn.Module,
        The loss function to be used for the loss.
    """

    def __init__(
        self,
        loss_function: nn.Module,
    ):

        super().__init__()
        self.loss_function: nn.Module = loss_function
        self.losses: List = []

    @abstractmethod
    def loss(self, yhat: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """ Abstract method. Respective loss computations from predictions, targets and optional
            (scaled) inputs must be implemented in the derived classes.
        """
        pass


    def data_loss(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Data term of the loss (without penalties), used for logging and AdaNP decisions.
            Follows the torch convention ``loss_function(input, target)``. Override in derived
            classes whose data term differs from ``loss_function``.
        """
        return self.loss_function(yhat, y)

    def forward(self, yhat: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """ Calls the loss function and returns the loss based on the predictions, targets and optional inputs. """
        return self.loss(yhat=yhat, y=y, x=x)



class SSLLossCore(nn.Module, ABC):
    """
    Abstract core self supervised learning loss function. Holds the self-supervised loss function and losses.
    All derived classes must implement the ssl_loss method.

    Parameters
    ----------
    scaler: Scaler | None = None,
        Optional Scaler object to be used for unscaling the inputs to the losses.
    """

    def __init__(
        self,
        scaler: Scaler | None = None,
    ):
        super().__init__()
        if scaler is None:
            warnings.warn("No scaler provided for self-supervised loss_function. "
                          "Loss will be computed on unscaled inputs.")
        self.scaler: Scaler | None = scaler
        self.losses: List = []

    @abstractmethod
    def ssl_loss(self, x: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        """ Abstract method. Respective loss computations of the self supervised learning loss
            must be implemented in the derived classes.
        """
        pass


    def forward(self, x: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        """ Calls the self supervised learning loss function. If a scaler is provided,
            the inputs and predictions will first be unscaled by this scaler.
        """
        if self.scaler is None:
            ssl_loss: torch.Tensor = self.ssl_loss(x=x, yhat=yhat)
        else:
            x_unscaled, yhat_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=yhat)
            ssl_loss: torch.Tensor = self.ssl_loss(x=x_unscaled, yhat=yhat_unscaled)
        return ssl_loss
