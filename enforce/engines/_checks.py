from torch.nn import Module

# Local imports
from enforce.losses.enforce_loss import ENFORCELoss, ENFORCESSLLoss
from enforce.losses.loss_core import LossCore, SSLLossCore
from enforce.models import ConstrainedModel


def check_model_loss(model: Module, loss_fn: Module) -> None:
    """Raise if the model and loss function cannot be used together by :class:`Trainer` / :class:`Evaluator`.

    Checked with ``isinstance``, so subclasses are accepted:

    - :class:`ConstrainedModel`: :class:`ENFORCELoss` or :class:`ENFORCESSLLoss`, which
      compute the loss from the projected and the raw prediction.
    - Plain backbone (any other :class:`nn.Module`): any :class:`LossCore` or
      :class:`SSLLossCore`, except :class:`ENFORCELoss`, which requires a projection.

    Raises
    ------
    NotImplementedError
        If the combination is not supported.
    """
    if isinstance(model, ConstrainedModel):
        is_valid = isinstance(loss_fn, (ENFORCELoss, ENFORCESSLLoss))
    else:
        is_valid = isinstance(loss_fn, (LossCore, SSLLossCore)) and not isinstance(loss_fn, ENFORCELoss)

    if not is_valid:
        raise NotImplementedError(f"The combination of {type(model).__name__} model and {type(loss_fn).__name__} "
                                  f"loss function is not supported!")
