import torch
import torch.nn as nn
from typing import Union, Any
from src.benchmark_problems.parametric_optimization.opt_problem import NonlinearProgram

class SSLConfig:
    def __init__(self, soft_constrained=True, weight_loss_soft=1.0):
        self.soft_constrained = soft_constrained
        self.weight_loss_soft = weight_loss_soft

class SSLLoss(nn.Module):
    def __init__(self, config: SSLConfig, opt_prob: Union[NonlinearProgram, Any]):
        """
        Initialize the SSLLoss class.
        
        Args:
            config (SSLConfig): Configuration object containing soft constraint 
            settings.
            opt_prob (NonlinearProgram): Nonlinear program object containing 
            constraints and objective.
        """
        super().__init__()
        if not isinstance(config, SSLConfig):
            raise ValueError("config must be an instance of SSLConfig")
        self.config = config
        self.opt_prob = opt_prob

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the SSL loss function.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_x).
            y (torch.Tensor): Output tensor of shape (batch_size, dim_y).
        
        Returns:
            torch.Tensor: Computed SSL loss.

        Notes:
            - Input and output tensors must be unscaled (i.e., in the original
            space).
        """
        if self.config.soft_constrained:
            return torch.mean(self.opt_prob.objective(x, y) +
                              self.config.weight_loss_soft *
                              torch.norm(self.opt_prob.eq_constraints(x, y), p=2, dim=1)**2)
        else:
            return torch.mean(self.opt_prob.objective(x, y))