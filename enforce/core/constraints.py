from typing import Callable, Sequence, overload

import numpy as np
import torch

ConstraintFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Constraint:
    """A block of ``size`` constraints evaluated together.

    Whether the block holds equalities ``c(x, y) = 0`` or inequalities
    ``g(x, y) <= 0`` is decided by where it is passed to :class:`Constraints`.
    One block can hold a single scalar constraint or many vectorised ones, e.g.
    ``Constraint(lambda x, y: y @ A.T - b, size=A.shape[0])``.

    Either pass ``fn`` (and optionally ``jac``) or subclass and override
    :meth:`residual` (and optionally :meth:`jacobian`).

    Parameters
    ----------
    fn :
        ``fn(x, y)`` on unscaled inputs ``[BS, NI]`` and original outputs ``[BS, NO]``.
        Returns ``[BS, size]``, ``[BS]`` (``size=1``) or a tuple of ``size`` ``[BS]`` tensors.
        Each row may depend only on the same sample of ``x`` and ``y``, not on other samples in the batch.
    size :
        Number of constraints in the block.
    jac :
        Optional analytical Jacobian ``jac(x, y) -> [BS, size, NO]`` with respect to ``y``.
        Without it, the Jacobian is computed with autograd on this block only.
    name :
        Optional name used in error messages.
    """

    def __init__(self, fn: ConstraintFn | None = None, size: int = 1,
                 jac: ConstraintFn | None = None, name: str | None = None):
        if size < 1:
            raise ValueError(f"Constraint size must be >= 1, got {size}.")
        self.fn: ConstraintFn | None = fn
        self.size: int = size
        self.jac: ConstraintFn | None = jac
        self.name: str = name or getattr(fn, "__name__", type(self).__name__)

    def residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Raw constraint values, any shape accepted by :meth:`__call__`."""
        if self.fn is None:
            raise NotImplementedError("Pass `fn` or override `residual`.")
        return self.fn(x, y)

    def jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
        """Analytical Jacobian ``[BS, size, NO]``, or ``None`` to use autograd."""
        return self.jac(x, y) if self.jac is not None else None

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Constraint values as a ``[BS, size]`` tensor."""
        value = self.residual(x, y)
        if isinstance(value, (tuple, list)):
            value = torch.stack(value, dim=1)
        elif value.ndim == 1:
            value = value.unsqueeze(1)
        if value.ndim != 2 or value.shape[0] != y.shape[0] or value.shape[1] != self.size:
            raise ValueError(
                f"Constraint '{self.name}' returned shape {tuple(value.shape)}, "
                f"expected [BS, {self.size}]. Set `size` to the number of constraints it returns."
            )
        return value


def _as_constraint(item: Constraint | ConstraintFn) -> Constraint:
    """Plain callables are single scalar constraints."""
    return item if isinstance(item, Constraint) else Constraint(item)


class Constraints:
    """
    Class that handles in-/equality constraints, residuals and their Jacobian.

    Parameters
    ----------
    c :
        Equality constraints ``c(x, y) = 0``. Each item is a :class:`Constraint` block
        or a plain callable ``c_i(x, y) -> Tensor[BS]`` (one scalar equation).
    g :
        Inequality constraints ``g(x, y) <= 0``, same format as ``c``.
    eps_fb : float
        Optional regularization parameter for Fischer-Burmeister reformulation equality constraints.

    All callables take unscaled inputs ``[BS, NI]`` and original outputs ``[BS, NO]``.
    Equations with the same structure are fastest as one vectorised block,
    especially with an analytical Jacobian::

        Constraints(c=[Constraint(eq_fn, size=50, jac=eq_jac)], g=[g_upper, g_lower])
    """

    def __init__(
        self,
        c: Sequence[Constraint | ConstraintFn] | None = None,
        g: Sequence[Constraint | ConstraintFn] | None = None,
        eps_fb: float = 1e-8,
    ):

        super().__init__()
        self.eps_fb: float = eps_fb
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.equalities: list[Constraint] = [_as_constraint(item) for item in c or []]
        self.inequalities: list[Constraint] = [_as_constraint(item) for item in g or []]
        self.nc: int = sum(block.size for block in self.equalities)
        self.ng: int = sum(block.size for block in self.inequalities)

    def __call__(self, x: torch.Tensor, y_ext: torch.Tensor) -> torch.Tensor:
        """Compute equality and inequality constraint residuals from outputs in the form of a tensor ``[BS, NC+NG]``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``[BS, NI]``,
        y_ext : torch.Tensor
            Output tensor, shape ``[BS, NO+NG]``,

        Returns
        -------
        residuals : torch.Tensor
            ``[BS, NC+NG]`` tensor of equality and inequality residuals.
            ENFORCE drives these to zero through projection.
                """
        no: int = y_ext.shape[1] - self.ng
        y: torch.Tensor = y_ext[:, :no]
        eq: torch.Tensor = self.c(x, y)
        ineq: torch.Tensor = self.g(x, y_ext)
        residuals: torch.Tensor = self._cat([eq, ineq])
        return residuals

    @staticmethod
    def _cat(tensors: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate along dim 1, without copying when only one tensor is non-empty."""
        non_empty: list[torch.Tensor] = [t for t in tensors if t.shape[1] > 0]
        if len(non_empty) == 1:
            return non_empty[0]
        return torch.cat(tensors, dim=1)

    @staticmethod
    def _evaluate(blocks: list[Constraint], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not blocks:
            return torch.empty((x.shape[0], 0), device=x.device, dtype=y.dtype)
        return Constraints._cat([block(x, y) for block in blocks])

    def c(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute equality constraint residuals from outputs in the form of a tensor ``[BS, NC]``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``[BS, NI]``,
        y : torch.Tensor
            Output tensor, shape ``[BS, NO]``,

        Returns
        -------
        eq_residuals : torch.Tensor
            ``[BS, NC]`` tensor of equality residuals.
            ENFORCE drives these to zero through projection.
        """
        return self._evaluate(self.equalities, x, y)

    def g_values(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the inequality values ``g(x, y)`` (feasible where ``<= 0``) as a tensor ``[BS, NG]``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``[BS, NI]``.
        y : torch.Tensor
            Original output tensor, shape ``[BS, NO]`` (without multipliers).
        """
        return self._evaluate(self.inequalities, x, y)

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
    ENFORCE projects an *extended* output vector (the network predicts only ``y``):

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

    2. Create the constraints wrapper::

           from enforce.core.constraints import Constraints

           constraints = Constraints(
               g=[g_positive_y1],
           )

    3. Extend your training/test labels with zero multipliers::

           y_train_ext = constraints.extend_outputs(y_train)   # [N, NO + NG]

    4. Pass ``constraints`` directly to ENFORCE as the constraint callable.
        ENFORCE will append zero lambda columns in ``forward()``::

           from enforce.layers import ENFORCE

           enforce = ENFORCE(..., constraints=constraints, ...)

    Inequalities use the convention ``g_i(x, y) <= 0``. The extended output
    appends one nonnegative multiplier per inequality; ``g`` converts each
    inequality and multiplier into a regularized equality residual. Use
    :meth:`extend_outputs` to append zero multipliers to labelled outputs and
    :meth:`extract_outputs` to remove them from predictions.

    The original output width is inferred from ``Scaler.no - Constraints.ng``.
    """

    # ------------------------------------------------------------------
    # Fischer-Burmeister primitive
    # ------------------------------------------------------------------

    def _fb(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Regularised Fischer-Burmeister function.

        phi_FB(a, b) = sqrt(a^2 + b^2 + eps_fb) - a - b

        Equals zero iff a >= 0, b >= 0, a*b = 0  (up to eps regularisation).
        """
        return torch.sqrt(a ** 2 + b ** 2 + self.eps_fb) - a - b

    # ------------------------------------------------------------------
    # Constraint function — signature compatible with ENFORCE's c(x, y)
    # ------------------------------------------------------------------


    def g(self, x: torch.Tensor, y_ext: torch.Tensor) -> torch.Tensor:
        """Compute the FB equality residuals from the extended output in the form of a tensor ``[BS, NG]``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``[BS, NI]``, *unscaled*.
        y_ext : torch.Tensor
            Extended output tensor, shape ``[BS, NO+NG]``, *unscaled*.
            Columns ``0 … NO-1``       → original outputs.
            Columns ``NO … NO+m-1``    → FB dual variables (multipliers).

        Returns
        -------
        ineq_residuals : torch.Tensor
            ``[BS, NG]`` tensor of FB equality residuals.
            ENFORCE drives these to zero through projection.
        """
        no: int = y_ext.shape[1] - self.ng
        g_val: torch.Tensor = self.g_values(x, y_ext[:, :no])  # [BS, NG]
        lambdas: torch.Tensor = y_ext[:, no:]  # FB multipliers [BS, NG]

        # phi_FB(lambda_i, -g_i) = sqrt(lambda_i^2 + g_i^2) - lambda_i + g_i
        # Encodes: lambda_i >= 0,  g_i <= 0,  lambda_i * g_i = 0
        return self._fb(lambdas, -g_val)  # [BS, NG]

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------

    @staticmethod
    def _autograd_rows(value: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Jacobian ``[BS, size, N]`` of ``value`` ``[BS, size]`` w.r.t. ``y`` ``[BS, N]``, one autograd call per row."""
        if not value.requires_grad:  # independent of y
            return torch.zeros(y.shape[0], value.shape[1], y.shape[1], device=y.device, dtype=y.dtype)
        rows: list[torch.Tensor] = []
        for i in range(value.shape[1]):
            grad_outputs: torch.Tensor = torch.zeros_like(value)
            grad_outputs[:, i] = 1
            grad: torch.Tensor | None = torch.autograd.grad(
                outputs=value,
                inputs=y,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            rows.append(torch.zeros_like(y) if grad is None else grad)
        return torch.stack(rows, dim=1)

    @staticmethod
    def _block_jacobian(block: Constraint, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Jacobian ``[BS, size, NO]`` of one block, analytical or by autograd on the block's own graph."""
        jac = block.jacobian(x, y)
        if jac is not None:
            return jac
        with torch.enable_grad():
            if not y.requires_grad:
                y = y.detach().requires_grad_(True)
            return Constraints._autograd_rows(block(x, y), y)

    def _fb_block_jacobian(self, block: Constraint, x: torch.Tensor, y_ext: torch.Tensor, start: int) -> torch.Tensor:
        """Jacobian ``[BS, size, NO+NG]`` of the FB residuals of one inequality block.

        Without an analytical ``jac``, autograd differentiates the block's FB residual directly.
        This is the same computation as differentiating the whole residual, so training stays
        bitwise reproducible (AdaNP amplifies any rounding difference). With ``jac``, the chain rule is used.
        """
        no: int = y_ext.shape[1] - self.ng
        cols = slice(no + start, no + start + block.size)  # this block's multiplier columns
        y: torch.Tensor = y_ext[:, :no]
        jac_g: torch.Tensor | None = block.jacobian(x, y)
        if jac_g is None:
            with torch.enable_grad():
                if not y_ext.requires_grad:
                    y_ext = y_ext.detach().requires_grad_(True)
                phi: torch.Tensor = self._fb(y_ext[:, cols], -block(x, y_ext[:, :no]))  # [BS, size]
                return self._autograd_rows(phi, y_ext)

        # phi = r - lambda + g with r = sqrt(lambda^2 + g^2 + eps_fb)
        g_val: torch.Tensor = block(x, y)
        lambdas: torch.Tensor = y_ext[:, cols]
        r: torch.Tensor = torch.sqrt(lambdas ** 2 + g_val ** 2 + self.eps_fb)
        dphi_dg: torch.Tensor = g_val / r + 1.0  # [BS, size]
        dphi_dlambda: torch.Tensor = lambdas / r - 1.0  # [BS, size]
        jac_lambda = torch.nn.functional.pad(torch.diag_embed(dphi_dlambda), (start, self.ng - start - block.size))
        return torch.cat([dphi_dg.unsqueeze(2) * jac_g, jac_lambda], dim=2)

    def jacobian(self, x: torch.Tensor, y_ext: torch.Tensor) -> torch.Tensor:
        """Compute the Jacobian of :meth:`__call__` with respect to ``y_ext``, shape ``[BS, NC+NG, NO+NG]``.

        Each block is differentiated on its own (analytically when it provides ``jac``),
        so the cost grows with the size of the blocks and not with the size of the whole system.
        Fischer-Burmeister rows use the chain rule when the inequality block provides ``jac``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``[BS, NI]``, *unscaled*.
        y_ext : torch.Tensor
            Extended output tensor, shape ``[BS, NO+NG]``, *unscaled*.
        """
        no: int = y_ext.shape[1] - self.ng
        y: torch.Tensor = y_ext[:, :no]
        rows: list[torch.Tensor] = []

        if self.equalities:
            jac_c = self._cat([self._block_jacobian(block, x, y) for block in self.equalities])
            rows.append(torch.nn.functional.pad(jac_c, (0, self.ng)) if self.ng else jac_c)  # [BS, NC, NO+NG]

        start: int = 0
        for block in self.inequalities:
            rows.append(self._fb_block_jacobian(block, x, y_ext, start))  # [BS, size, NO+NG]
            start += block.size

        if not rows:
            return torch.empty((y_ext.shape[0], 0, y_ext.shape[1]), device=y_ext.device, dtype=y_ext.dtype)
        return self._cat(rows)  # [BS, NC+NG, NO+NG]


    # ------------------------------------------------------------------
    # Data-preparation helpers
    # ------------------------------------------------------------------
    @overload
    def extend_outputs(self, y: torch.Tensor) ->  torch.Tensor:
        """Append zero multiplier columns to a torch Tensor.

        At feasible points (all ``g_i <= 0``), complementarity forces
        ``lambda_i = 0``, so the correct extended label is ``[y, 0, …, 0]``.

        Parameters
        ----------
        y : torch.Tensor
            Shape ``[N, NO]``.

        Returns
        -------
        y_ext : torch.Tensor
            Shape ``[N, NO+NG]`` with zero columns appended.
        """

    @overload
    def extend_outputs(self, y: np.ndarray) ->  np.ndarray:
        """Append zero multiplier columns to a numpy output array.

        At feasible points (all ``g_i <= 0``), complementarity forces
        ``lambda_i = 0``, so the correct extended label is ``[y, 0, …, 0]``.

        Parameters
        ----------
        y : np.ndarray
            Shape ``[N, NO]``.

        Returns
        -------
        y_ext : np.ndarray
            Shape ``[N, NO+NG]`` with zero columns appended.
        """

    def extend_outputs(self, y: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Append zero multiplier columns to a torch Tensor or numpy output array.

        At feasible points (all ``g_i <= 0``), complementarity forces
        ``lambda_i = 0``, so the correct extended label is ``[y, 0, …, 0]``.

        Parameters
        ----------
        y : torch.Tensor | np.ndarray
            Shape ``[N, NO]``.

        Returns
        -------
        y_ext : torch.Tensor | np.ndarray
            Shape ``[N, NO+NG]`` with zero columns appended.
        """
        if isinstance(y, torch.Tensor):
            zeros: torch.Tensor = torch.zeros(y.shape[0], self.ng, device=y.device, dtype=y.dtype)
            return torch.cat([y, zeros], dim=1)
        else:
            zeros: np.ndarray = np.zeros((y.shape[0], self.ng), dtype=np.float32)
            return np.concatenate([y.astype(np.float32), zeros], axis=1)


    @overload
    def extract_outputs(self, y_ext: torch.Tensor) -> torch.Tensor:
        """Return only the original outputs of the torch Tensor, discarding FB multiplier columns.

        Parameters
        ----------
        y_ext : torch.Tensor
            Shape ``[N, NO+NG]``, torch.Tensor.

        Returns
        -------
        y: torch.Tensor
            Shape ``[N, NO]``.
        """

    @overload
    def extract_outputs(self, y_ext: np.ndarray) -> np.ndarray:
        """Return only the original outputs of the torch Tensor or numpy output array, discarding FB multiplier columns.

        Parameters
        ----------
        y_ext : np.ndarray
            Shape ``[N, NO+NG]``, numpy array.

        Returns
        -------
        y: np.ndarray
            Shape ``[N, NO]``.
        """

    def extract_outputs(self, y_ext: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Return only the original outputs of the torch Tensor or numpy output array, discarding FB multiplier columns.

        Parameters
        ----------
        y_ext : torch.Tensor | np.ndarray
            Shape ``[N, NO+NG]``, numpy array or torch.Tensor.

        Returns
        -------
        y: torch.Tensor | np.ndarray
            Shape ``[N, NO]``.
        """
        no: int = y_ext.shape[1] - self.ng
        return y_ext[:, : no]
