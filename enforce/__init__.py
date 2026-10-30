from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("enforce-nn")
except PackageNotFoundError:  # source checkout that is not installed
    __version__ = "unknown"

from .layers import ENFORCE, ENFORCEConfig
from .losses import ENFORCELoss, ENFORCELossConfig
from .models import ENFORCEConstrainedModel, ENFORCEConstrainedMLP
from .core import Scaler, Constraint, Constraints

__all__ = ["ENFORCEConfig", "ENFORCE",
           "ENFORCELoss", "ENFORCELossConfig",
           "ENFORCEConstrainedModel", "ENFORCEConstrainedMLP",
           "Scaler", "Constraint", "Constraints",
           ]