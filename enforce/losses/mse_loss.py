import torch
import torch.nn as nn

from enforce.losses.loss_core import LossCore


class MSELoss(LossCore):
    """ Implements the standard torch MSE loss function based on the abstract loss core.

    Attributes
    ----------
    loss_function: nn.MSELoss,
        The MSE loss function to be used for the loss.
    """
    def __init__(self):
        super().__init__(loss_function=nn.MSELoss())

    def loss(self, yhat: torch.Tensor, y: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        return self.data_loss(yhat, y)
