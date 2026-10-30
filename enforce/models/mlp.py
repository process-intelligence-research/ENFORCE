import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

@dataclass
class MLPConfig:
    """Configuration for the simple multi-layer perceptron backbone model.

    Parameters
    ----------
    input_neurons :
        Number of input neurons.
    hidden_neurons :
        Number of hidden neurons.
    output_neurons :
        Number of output neurons.
    hidden_layers :
        Number of hidden layers.
    """

    # ── Network architecture ─────────────────────────────────────────────────
    input_neurons: int = 1
    hidden_neurons: int = 64
    output_neurons: int = 1
    hidden_layers: int = 1
    # ── Misc ─────────────────────────────────────────────────────────────────
    random_seed: int = 42


class MLP(nn.Module):
    """Multilayer perceptron neural network backbone.

    Architecture configured by MLPConfig.

    Parameters
    ----------
    config : MLPConfig
        MLP Config from which to construct MLP.
    """

    def __init__(
        self,
        config: MLPConfig | None = None,
    ):
        self.cfg: MLPConfig = config if config is not None else MLPConfig()

        torch.manual_seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        random.seed(self.cfg.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(self.cfg.random_seed)
        super().__init__()

        # ── Network architecture ───────────────────────────────────────────────

        self.input_layer: nn.Linear  = nn.Linear(self.cfg.input_neurons, self.cfg.hidden_neurons)
        self.hidden_layers: nn.ModuleList  = nn.ModuleList(
            [nn.Linear(self.cfg.hidden_neurons, self.cfg.hidden_neurons) for _ in range(self.cfg.hidden_layers)]
        )
        self.hidden_activation: nn.ReLU = nn.ReLU()
        self.output_layer: nn.Linear = nn.Linear(self.cfg.hidden_neurons, self.cfg.output_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.hidden_activation(x)
        x = self.output_layer(x)
        return x