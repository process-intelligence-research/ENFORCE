from typing import Tuple

import torch
import torch.autograd.profiler as profiler

class Scaler:
    """
    Scaler that handles input/output scaling.

    Parameters
    ----------
    scaling_input, scaling_output :
        Tuples of (mean, std) tensors for input/output normalization.
    """

    def __init__(
        self,
        scaling_input: Tuple[float, float] | Tuple[torch.Tensor, torch.Tensor],
        scaling_output: Tuple[float, float] | Tuple[torch.Tensor, torch.Tensor],
    ):

        super().__init__()
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.mean_input: torch.Tensor = torch.as_tensor(scaling_input[0], device=self.device, dtype=torch.float32)
        self.std_input: torch.Tensor = torch.as_tensor(scaling_input[1], device=self.device, dtype=torch.float32)
        self.mean_output: torch.Tensor = torch.as_tensor(scaling_output[0], device=self.device, dtype=torch.float32)
        self.std_output: torch.Tensor = torch.as_tensor(scaling_output[1], device=self.device, dtype=torch.float32)
        # Per column, like scale_data: outputs with near-zero std (e.g. FB multiplier columns) are only centred.
        self._std_output_safe: torch.Tensor = torch.where(
            self.std_output >= 1e-5, self.std_output, torch.ones_like(self.std_output)
        )

        self.ni: int = self.mean_input.shape[0]
        self.no: int = self.mean_output.shape[0]

    def scale(self, x_unscaled: torch.Tensor, y_unscaled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Scale the input and output values according to the scaling scheme:

        ``x = (x_unscaled - mean_input) / std_input``

        ``y = (y_unscaled - mean_output) / std_output``

        or ``y = y_unscaled - mean_output`` for output columns with ``std_output ~ 0``

        Parameters
        ----------
        x_unscaled : torch.Tensor
            Unscaled input values.
        y_unscaled : torch.Tensor
            Unscaled output values.

        Returns
        -------
        x, y : Tuple[torch.Tensor, torch.Tensor]
            Scaled input and output values.
        """
        x = (x_unscaled - self.mean_input.to(x_unscaled.device)) / self.std_input.to(x_unscaled.device)
        y = (y_unscaled - self.mean_output.to(y_unscaled.device)) / self._std_output_safe.to(y_unscaled.device)
        return x, y

    def unscale(self, x_scaled: torch.Tensor, y_scaled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Unscale the input and output values according to the scaling scheme:

        ``x = x_scaled * std_input + mean_input``

        ``y = y_scaled * std_output + mean_output``

        or ``y = y_scaled + mean_output`` for output columns with ``std_output ~ 0``

        Parameters
        ----------
        x_scaled : torch.Tensor
            Scaled input values.
        y_scaled : torch.Tensor
            Scaled output values.

        Returns
        -------
        x, y : Tuple[torch.Tensor, torch.Tensor]
            Unscaled input and output values.
        """
        with profiler.record_function("unscale"):
            x = x_scaled * self.std_input.to(x_scaled.device) + self.mean_input.to(x_scaled.device)
            y = y_scaled * self._std_output_safe.to(y_scaled.device) + self.mean_output.to(y_scaled.device)
        return x, y