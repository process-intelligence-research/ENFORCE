import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from enforce import ENFORCEConfig
# Local imports
from enforce.layers.enforce import ENFORCE
from enforce.models.mlp import MLPConfig, MLP


class ConstrainedModel(nn.Module):
    """Backbone nn.Module constrained by a constraint module.

    Parameters
    ----------
    backbone : nn.Module
        Neural backbone module.
    constraint_module : nn.Module
        Constraint enforcement module.

    """

    def __init__(
        self,
        backbone: nn.Module,
        constraint_module: nn.Module,
    ):
        super().__init__()
        self.backbone: nn.Module = backbone
        self.constraint_module: nn.Module = constraint_module

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns constrained ytilde and predicted yhat."""
        yhat: torch.Tensor = self.backbone(x)
        return self.constraint_module(yhat), yhat

class ENFORCEConstrainedModel(ConstrainedModel):
    """Backbone nn.Module constrained by ENFORCE.

    Parameters
    ----------
    backbone : nn.Module
        Neural backbone module.
    enforce : ENFORCE
        Instantiated :class:`ENFORCE` constraint enforcement module.

    """

    def __init__(
        self,
        backbone: nn.Module,
        enforce: ENFORCE,
    ):
        super().__init__(backbone=backbone, constraint_module=enforce)
        self.cfg: ENFORCEConfig = enforce.cfg


    def forward(self, x: torch.Tensor, y=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ytilde and yhat."""
        yhat: torch.Tensor = self.backbone(x)
        ytilde: torch.Tensor = self.constraint_module(x=x, yhat=yhat, y=y)
        return ytilde, yhat

    def predict(self, x: torch.Tensor, y=None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Returns extended ytilde and yhat as well as the number of projection iterations."""
        yhat: torch.Tensor = self.backbone(x)
        ytilde, yhat, proj_iter = self.constraint_module.project(x=x, yhat=yhat, y=y)
        return ytilde, yhat, proj_iter


class ConstrainedMLP(ConstrainedModel):
    """Simple MLP backbone constructed from a MLPConfig constrained by a constraint module.

    Parameters
    ----------
    mlp_config : MLPConfig
        MLP Config from which to construct :class:`MLP`.
    constraint_module : nn.Module
        Constraint enforcement module.
    """
    def __init__(
        self,
        mlp_config: MLPConfig,
        constraint_module: nn.Module,
    ):
        mlp: nn.Module = MLP(config=mlp_config)
        super().__init__(backbone=mlp, constraint_module=constraint_module)
        self.mlp_cfg: MLPConfig = mlp_config


class ENFORCEConstrainedMLP(ENFORCEConstrainedModel):
    """Simple MLP backbone constructed from a MLPConfig constrained by ENFORCE.

    Parameters
    ----------
    mlp_config : MLPConfig
        MLP Config from which to construct :class:`MLP`.
    enforce : ENFORCE
        Instantiated :class:`ENFORCE` constraint enforcement module.
    """
    def __init__(
        self,
        mlp_config: MLPConfig,
        enforce: ENFORCE,
    ):
        mlp: nn.Module = MLP(config=mlp_config)
        super().__init__(backbone=mlp, enforce=enforce)
        self.mlp_cfg: MLPConfig = mlp_config





