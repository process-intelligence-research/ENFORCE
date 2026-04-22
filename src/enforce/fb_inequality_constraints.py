"""
Fischer-Burmeister reformulation of inequality constraints for ENFORCE.

Background
----------
For an inequality constraint g_i(x, y) <= 0, the KKT optimality conditions are:

    (1)  g_i(x, y) <= 0          (primal feasibility)
    (2)  lambda_i  >= 0          (dual feasibility)
    (3)  lambda_i * g_i(x, y) = 0   (complementarity slackness)

The Fischer-Burmeister (FB) function merges (2) and (3) into a single equality:

    phi_FB(a, b) = sqrt(a^2 + b^2) - a - b = 0   <=>   a >= 0, b >= 0, a*b = 0

Setting  a = lambda_i  and  b = -g_i(x, y):

    phi_i = sqrt(lambda_i^2 + g_i^2) - lambda_i + g_i = 0

This equality can be enforced directly by ENFORCE's projection without any
modification to the core algorithm.

Extended-output convention
--------------------------
The network predicts an *extended* output vector:

    y_ext = [y_1, ..., y_NO,  lambda_1, ..., lambda_m]

The additional m columns are the FB dual variables (one per inequality).

Training-data labelling
-----------------------
At feasible training points (g_i <= 0 for all i), complementarity forces
lambda_i = 0.  Therefore every training/test label vector is extended with
zeros: ``y_ext = [y, 0, ..., 0]``.  ENFORCE projects predictions onto the
FB equality manifold, automatically recovering the correct lambda at the
constraint boundary.

User API
--------
1. Define inequality functions g_i(x, y) <= 0::

       def g_positive_y1(x, y):
           return -y[:, 0]       # g = -y1 <= 0  <=>  y1 >= 0

2. Create the reformulation wrapper::

       from src.constraints.inequality_constraints import FischerBurmeisterReformulation

       fb = FischerBurmeisterReformulation(
           n_original_outputs=2,
           inequalities=[g_positive_y1],
       )

3. Extend your training/test labels with zero multipliers::

       y_train_ext = fb.extend_outputs(y_train)   # [N, NO + n_ineq]

4. Pass ``fb`` directly to ENFORCE as both the constraint callable and as the
   ``fb`` parameter.  ENFORCE will build a network with ``fb.no`` output neurons
   and append zero lambda columns in ``forward()``::

       from src.models.model import ENFORCE

       model = ENFORCE(c=fb, fb=fb, ...)
       # OUTPUT_NEURONS in config must equal fb.no (= 2 in this example)

5. After prediction, recover the original outputs::

       y_pred = fb.extract_outputs(y_ext_pred)    # [N, NO]

For mixed equality + inequality constraints, pass a combined constraint
callable as ``c`` and ``fb`` for the FB wrapper::

       def constraints_full(x, y_ext):
           y = y_ext[:, :NO]
           eq = torch.stack([c1(x, y), c2(x, y)], dim=1)  # [BS, n_eq]
           return torch.cat([eq, fb(x, y_ext)], dim=1)     # [BS, n_eq + n_ineq]

       model = ENFORCE(c=constraints_full, fb=fb, ...)
"""

import numpy as np
import torch


class FischerBurmeisterReformulation:
    """Wraps inequality constraints g_i(x, y) <= 0 as FB equality constraints.

    Parameters
    ----------
    n_original_outputs:
        Number of original output dimensions (without FB dual variables).
    inequalities:
        List of callables ``g_i(x, y) -> Tensor[BS]``, each encoding one
        inequality constraint ``g_i(x, y) <= 0``.
    eps:
        Small positive regularisation inside the square-root, for numerical
        stability near the degenerate point ``(a, b) = (0, 0)``.
    """

    def __init__(
        self,
        n_original_outputs: int,
        inequalities: list[callable],
        eps: float = 1e-8,
    ) -> None:
        self.no = n_original_outputs
        self.inequalities = inequalities
        self.n_ineq = len(inequalities)
        self.eps = eps

    # ------------------------------------------------------------------
    # Fischer-Burmeister primitive
    # ------------------------------------------------------------------

    def _fb(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Regularised Fischer-Burmeister function.

        phi_FB(a, b) = sqrt(a^2 + b^2 + eps) - a - b

        Equals zero iff a >= 0, b >= 0, a*b = 0  (up to eps regularisation).
        """
        return torch.sqrt(a**2 + b**2 + self.eps) - a - b

    # ------------------------------------------------------------------
    # Constraint function — signature compatible with ENFORCE's c(x, y)
    # ------------------------------------------------------------------

    def __call__(self, x: torch.Tensor, y_ext: torch.Tensor) -> torch.Tensor:
        """Compute the FB equality residuals from the extended output.

        Parameters
        ----------
        x :
            Input tensor, shape ``[BS, NI]``, *unscaled*.
        y_ext :
            Extended output tensor, shape ``[BS, NO + n_ineq]``, *unscaled*.
            Columns ``0 … NO-1``       → original outputs.
            Columns ``NO … NO+m-1``    → FB dual variables (multipliers).

        Returns
        -------
        ``[BS, n_ineq]`` tensor of FB equality residuals.
        ENFORCE drives these to zero through projection.
        """
        y = y_ext[:, : self.no]  # original outputs  [BS, NO]
        lambdas = y_ext[:, self.no :]  # FB multipliers    [BS, n_ineq]

        phi_list: list[torch.Tensor] = []
        for i, g_i in enumerate(self.inequalities):
            g_val = g_i(x, y)  # [BS] or [BS, 1]
            if isinstance(g_val, torch.Tensor) and g_val.ndim > 1:
                g_val = g_val.squeeze(1)

            lam_i = lambdas[:, i]  # [BS]

            # phi_FB(lambda_i, -g_i) = sqrt(lambda_i^2 + g_i^2) - lambda_i + g_i
            # Encodes: lambda_i >= 0,  g_i <= 0,  lambda_i * g_i = 0
            phi_i = self._fb(lam_i, -g_val)
            phi_list.append(phi_i)

        return torch.stack(phi_list, dim=1)  # [BS, n_ineq]

    # ------------------------------------------------------------------
    # Data-preparation helpers
    # ------------------------------------------------------------------

    def extend_outputs(self, y: np.ndarray) -> np.ndarray:
        """Append zero multiplier columns to a numpy output array.

        At feasible points (all ``g_i <= 0``), complementarity forces
        ``lambda_i = 0``, so the correct extended label is ``[y, 0, …, 0]``.

        Parameters
        ----------
        y :
            Shape ``[N, NO]``.

        Returns
        -------
        Shape ``[N, NO + n_ineq]`` with zero columns appended.
        """
        zeros = np.zeros((y.shape[0], self.n_ineq), dtype=np.float32)
        return np.concatenate([y.astype(np.float32), zeros], axis=1)

    def extract_outputs(self, y_ext: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Return only the original outputs, discarding FB multiplier columns.

        Parameters
        ----------
        y_ext :
            Shape ``[N, NO + n_ineq]``, numpy array or torch.Tensor.

        Returns
        -------
        Shape ``[N, NO]``.
        """
        return y_ext[:, : self.no]
