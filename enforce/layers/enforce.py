import random
import warnings
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn

# Local imports
from enforce.core.scaler import Scaler
from enforce.core.constraints import Constraints
from enforce.losses.loss_core import LossCore, SSLLossCore
from enforce.losses.enforce_loss import ENFORCELossConfig

@dataclass
class ENFORCEConfig:
    """Configuration for the ENFORCE model.

    Parameters
    ----------
    training_tolerance :
        Convergence criterion used during training (mean residual).
    inference_tolerance :
        Convergence criterion used at inference time (max residual).
    max_it :
        Maximum number of neural projection iterations.
    epoch_start_hard_constrained :
        Epoch at which the neural projection is first applied.  Set to 0 to
        project from the very first epoch.
    ada_np_auto_activation :
        If ``True``, skip the projection when it would increase the training
        loss (adaptive neural projection). Supervised losses only.
    ift_backward :
        If ``True``, use the Implicit Function Theorem backward pass instead of
        unrolled autograd. Currently deprecated: ``True`` raises ``NotImplementedError``.
    weighting_option :
        Selects the weighting matrix W used in the weighted Newton step. The options are:
        I = identity (no weighting);
        BA = batch-averaged;
        ID = instance-dependent;
        R = random;
        TF = trainable-fixed diagonal (diagonal initialised to 1, coefficients
        learned end-to-end during training and then fixed).
    eps_chol :
        Regularisation scalar ``ε`` added to the gram matrix as ``ε·I`` before
        inversion.  Larger values help when Jacobians span many orders of magnitude
        (e.g. bilinear constraints).  Default: ``1e-8``.
    regularise_gram :
        If ``True``, add ``eps_chol * I`` to the gram matrix before Cholesky
        inversion.  For weighted projections (``W ≠ I``) this is always done;
        this flag extends it to the unweighted (``W = I``) case, where it is
        otherwise skipped to preserve the exact null-space projector.  Enable
        for ill-conditioned Jacobians despite ``W = I`` (e.g. bilinear
        constraints such as pooling).
    verbose :
        Print detailed per-epoch projection statistics (self-supervised losses only).
    random_seed :
        Random seed for reproducibility.

    """
    # ── Newton projection ────────────────────────────────────────────────────
    training_tolerance: float = 1e-4
    inference_tolerance: float = 1e-6
    max_it: int = 100
    epoch_start_hard_constrained: int = 0
    ada_np_auto_activation: bool = True
    ift_backward: bool = False
    weighting_option: str = "ID"
    eps_chol: float = 1e-8
    regularise_gram: bool = False

    # ── Misc ─────────────────────────────────────────────────────────────────
    verbose: bool = False
    random_seed: int = 42

    def __post_init__(self):
        if self.ift_backward:
            raise NotImplementedError("The IFT backward pass is currently deprecated. Use ift_backward=False.")
        if self.max_it < 1:
            raise ValueError(f"max_it must be at least 1 (one projection step always runs), got {self.max_it}")



class ENFORCE(nn.Module):
    """Differentiable neural network layer that projects predictions onto the constraint manifold.

    Handles both equality constraints and inequality constraints (via the
    Fischer-Burmeister reformulation).

    Parameters
    ----------
    scaler : Scaler
        :class:`Scaler` object for input/output normalisation.
    constraints : Constraints
        :class:`Constraints` object for constraint residual calculation.
    loss: LossCore | SSLLossCore
        The instantiated loss function. Determines the learning setting and is used for automatic activation of AdaNP.
    config : ENFORCEConfig
        Optional ENFORCEConfig to configure :class:`ENFORCE`.

    Analytical Jacobians are given per constraint block, see :class:`Constraint`.
    """

    def __init__(
        self,
        scaler: Scaler,
        constraints: Constraints,
        loss: LossCore | SSLLossCore,
        config: ENFORCEConfig | None = None,
    ):
        super().__init__()
        self.scaler: Scaler = scaler
        self.constraints: Constraints = constraints
        self.loss: LossCore | SSLLossCore = loss
        self.cfg: ENFORCEConfig = config if config is not None else ENFORCEConfig()

        if isinstance(self.loss, LossCore):
            self.supervised: bool = True
            self.ssl_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        else:
            self.supervised: bool = False
            self.ssl_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = self.loss.ssl_loss

        if hasattr(self.loss, 'cfg'):
            self.loss_cfg: ENFORCELossConfig | None = self.loss.cfg
        else:
            self.loss_cfg: ENFORCELossConfig | None = None

        self.verbose: bool = self.cfg.verbose
        if self.verbose and (self.ssl_loss is None or self.loss_cfg is None):
            self.verbose = False
            warnings.warn("Verbose was set to true, but the loss does not have a SSL Loss or ENFORCELossConfig."
                              "Continuing with Verbose=False.")


        torch.manual_seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        random.seed(self.cfg.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(self.cfg.random_seed)

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Check the system: equalities are counted against the original outputs only.
        # Each inequality adds one FB multiplier output and one FB equation, so it uses no degree of freedom.
        n_out_orig: int = self.scaler.no - self.constraints.ng
        # Issue a warning if the system is determined
        if self.constraints.nc == n_out_orig:
            warnings.warn("The system is determined. Maybe you already know your underlying model!")

        # Ensure the number of equality constraints is not greater than the number of original outputs
        if self.constraints.nc > n_out_orig:
            raise ValueError("Too many constraints!")


        self.ni: int = self.scaler.ni
        self.no: int = self.scaler.no

        self.weighting_option: str = self.cfg.weighting_option
        self.eps_chol: float = self.cfg.eps_chol
        self.ift_backward: bool = self.cfg.ift_backward

        # Trainable diagonal projection weighting matrix (option T).
        # Log-parameterised: exp(0) = 1 at init, strictly positive always.
        if self.weighting_option == "TF":
            self.proj_weight_log_diag = nn.Parameter(torch.zeros(self.no))

        self.epoch: int = 1
        self.training_iter: int = 0
        self.start_projection: bool = False
        self._epochprint: int | None = None


    def forward(self, x: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Project the predicted output yhat based on the constraints. Returns the projected output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        yhat : torch.Tensor
            Prediction tensor.
        y : torch.Tensor
            Optional target tensor. Needed for self supervised setting and automatic activation of AdaNP.

        Returns
        -------
        ytilde : torch.Tensor
            Projected prediction tensor.

        """
        ytilde, yhat, proj_iter = self.project(x, yhat, y)
        if self.constraints.ng != 0:
            ytilde: torch.Tensor = self.constraints.extract_outputs(ytilde)
        return ytilde



    def compute_residual(self, c: torch.Tensor, tolerance_mode: str) -> torch.Tensor:
        """ Computes the residual for the given constraints based on the tolerance mode.

        Parameters
        ----------
        c : torch.Tensor
            Constraint residual tensor.
        tolerance_mode : str
            Tolerance mode (either ``mean`` or ``max``).

        """
        if tolerance_mode == "mean":
            tolerance_value = torch.mean(torch.abs(c))
        elif tolerance_mode == "max":
            tolerance_value = torch.max(torch.abs(c))
        else:
            raise ValueError("Invalid tolerance mode. Choose 'mean' or 'max'.")
        return tolerance_value

    def ada_np(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        tolerance_mode: str="mean",
        tolerance_value: float | None = None,
        max_iter: int | None = None,
    ) -> Tuple[torch.Tensor, int]:
        """ Adaptively project the prediction until the desired tolerance is reached. Returns the projected output and
        number of projection iterations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Prediction tensor.
        tolerance_mode : str
            Tolerance mode (either ``mean`` or ``max``).
        tolerance_value : float
            Optional tolerance value.
        max_iter : int
            Optional maximum number of iterations.

        Returns
        -------
        ytilde, proj_iter : Tuple[torch.Tensor, int]
            Projected prediction tensor and number of projection iterations.

        """
        if tolerance_value is None:
            tolerance_value: float = self.cfg.training_tolerance
        if max_iter is None:
            max_iter: int = self.cfg.max_it

        proj_iter: int = 1
        # c = torch.zeros(x.shape[0], self.nc, device=x.device, dtype=x.dtype)
        input_unscaled, output_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=y)
        c: torch.Tensor = self.constraints(x=input_unscaled, y_ext=output_unscaled)  # Shape: [BS, NC]
        c_res: torch.Tensor = self.compute_residual(c=c, tolerance_mode=tolerance_mode)
        while c_res > tolerance_value and proj_iter < max_iter:
            self.compute_dc_dy(x, y)
            y: torch.Tensor = self.projection_step(input=x, output=y)
            proj_iter += 1
            input_unscaled, output_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=y)
            c = self.constraints(x=input_unscaled, y_ext=output_unscaled)  # Shape: [BS, NC]
            c_res = self.compute_residual(c=c, tolerance_mode=tolerance_mode)
        if proj_iter == max_iter:
            print(f"Max projection iteration reached ({max_iter})")
        return y, proj_iter

    def project(self, x: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor | None = None)\
            -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Project the predicted output yhat based on the constraints. Returns the projected output, original prediction,
        and number of projection iterations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        yhat : torch.Tensor
            Prediction tensor.
        y : torch.Tensor
            Optional target tensor. Needed for self supervised setting and automatic activation of AdaNP.

        Returns
        -------
        ytilde, yhat, proj_iter : Tuple[torch.Tensor, torch.Tensor, int]
            Projected prediction tensor, original prediction tensor, and number of projection iterations.

        """
        # tfdc = time.time()
        proj_iter: int = 0

        if self.constraints.ng != 0:
            # Append zero columns for FB dual variables.
            # At feasible interior points the correct KKT value is λ_i = 0;
            # the Newton projection will set the correct λ at active constraints.
            yhat: torch.Tensor = self.constraints.extend_outputs(yhat)

        if self.epoch < self.cfg.epoch_start_hard_constrained:
            if self.training:
                self.training_iter += 1
            return yhat, yhat, proj_iter

        with torch.enable_grad():
            if not yhat.requires_grad:
                yhat = yhat.detach().requires_grad_(True)

            if self.training and self.ift_backward:
                # ── IFT path ─────────────────────────────────────────────────
                # Run the full projection without building a computational graph
                # through the Newton iterations.  The backward pass applies
                # B_star(y*)ᵀ directly (see _ProjectionIFT for derivation).
                #
                # ADA_NP_AUTO_ACTIVATION is supported: we compute a no-grad
                # preview of the first projection step to check whether the
                # projection improves the loss.  If it does not, we return
                # yhat directly (normal backprop through the network only).
                # If it does, we commit to _ProjectionIFT for the full run.
                if self.supervised and self.cfg.ada_np_auto_activation and y is not None:
                    with torch.enable_grad():
                        # Detach from the network graph so no gradients flow to
                        # weights from this preview; requires_grad_(True) so
                        # compute_dc_dy can call torch.autograd.grad w.r.t. y.
                        yhat_preview = yhat.detach().requires_grad_(True)
                        self.compute_dc_dy(x, yhat_preview)
                        ytilde_preview = self.projection_step(x, yhat_preview).detach()
                    loss_before: torch.Tensor = self.loss.data_loss(yhat, y)
                    loss_after: torch.Tensor = self.loss.data_loss(ytilde_preview, y)
                    if loss_before < loss_after:
                        # Projection hurts: skip it, backprop through network only.
                        if self.start_projection:
                            self.start_projection = False
                            print(
                                f"Stop projection: Epoch {self.epoch} "
                                f"Iteration: {self.training_iter} "
                                f"Loss before: {loss_before}"
                            )
                        return yhat, yhat, 0
                    if not self.start_projection:
                        print(
                            f"Start projection: Epoch {self.epoch} "
                            f"Iteration: {self.training_iter} "
                            f"Loss before: {loss_before} Loss after: {loss_after}"
                        )
                ytilde: torch.Tensor = self._ProjectionIFT.apply(yhat, self, x, "mean", self.cfg.training_tolerance, self.cfg.max_it)
                proj_iter: int = getattr(self, "_ift_proj_iter", 1)
                self.start_projection: bool = True
                return ytilde, yhat, proj_iter

            self.compute_dc_dy(x, yhat)
            # print(f"Compute forward and dc_dy time: {time.time()-tfdc:.6f} seconds")
            # t = time.time()
            ytilde: torch.Tensor = self.projection_step(x, yhat)
            # print(f"Projection time: {time.time()-t:.6f} seconds")
            # t_ap = time.time()
            proj_iter += 1
            if not self.training:  # inference
                ytilde, proj_iter = self.ada_np(
                    x,
                    ytilde,
                    tolerance_mode="max",
                    tolerance_value=self.cfg.inference_tolerance,
                    max_iter=self.cfg.max_it,
                )
                print(f"Projection iterations inference: {proj_iter}")
                return ytilde.detach(), yhat.detach(), proj_iter

            if self.supervised:
                if self.cfg.ada_np_auto_activation and y is not None:
                    loss_data_before_projection: torch.Tensor = self.loss.data_loss(yhat, y)
                    loss_data_after_projection: torch.Tensor = self.loss.data_loss(ytilde, y)

                    # Criteria to project (note the first projection is already done)
                    if loss_data_before_projection < loss_data_after_projection:

                        # If the projection does not improve the prediction, then it is not considered
                        if self.start_projection:
                            self.start_projection: bool = False
                            print(
                                f"Stop projection: Epoch {self.epoch} Iteration: {self.training_iter} Loss before: {loss_data_before_projection}"
                            )
                        return yhat, yhat, proj_iter

                    if not self.start_projection:
                        print(
                            f"Start projection: Epoch {self.epoch} Iteration: {self.training_iter} Loss before: {loss_data_before_projection} Loss after: {loss_data_after_projection}"
                        )

                # Project to convergence without checking loss improvement.
                ytilde, proj_iter = self.ada_np(
                    x,
                    ytilde,
                    tolerance_mode="mean",
                )
                self.start_projection: bool = True
                return ytilde, yhat, proj_iter

            if not self.verbose:
                # t_adanp = time.time()
                ytilde, proj_iter = self.ada_np(
                    x,
                    ytilde,
                    tolerance_mode="mean",
                )
                # print(f"AdaNP time: {time.time()-t_adanp:.6f} seconds")
                self.start_projection: bool = True
                return ytilde, yhat, proj_iter

            x_unscaled, yhat_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=yhat)
            _, ytilde_unscaled = self.scaler.unscale(x_scaled=x, y_scaled=ytilde)
            ssl_loss_hat: torch.Tensor = self.ssl_loss(x_unscaled, yhat_unscaled)
            ssl_loss_tilde: torch.Tensor = self.ssl_loss(x_unscaled, ytilde_unscaled)

            if self.epoch % 1 == 0 and self.epoch != self._epochprint:
                avg_pred_obj_value: torch.Tensor = torch.mean(ssl_loss_tilde)
                x_u, y_u = self.scaler.unscale(x_scaled=x, y_scaled=y)
                avg_computed_obj_value: torch.Tensor = torch.mean(self.ssl_loss(x_u, y_u))
                print(
                    f"Epoch {self.epoch} Average computed obj value: {avg_computed_obj_value:.3f} Average predicted obj value: {avg_pred_obj_value:.3f}"
                )
                print(f"Difference objective: {(avg_pred_obj_value - avg_computed_obj_value):.3f}")
                print(
                    f"Relative difference objective: {((avg_pred_obj_value - avg_computed_obj_value) / avg_computed_obj_value * 100):.3f}%"
                )
                self._epochprint = self.epoch
            c_tilde: torch.Tensor = self.constraints(x=x_unscaled, y_ext=ytilde_unscaled)
            if self.loss_cfg.soft_constrained:
                c_hat: torch.Tensor = self.constraints(x=x_unscaled, y_ext=yhat_unscaled)
                if self.epoch % 1 == 0 and self.epoch != self._epochprint:
                    print(
                        f"Epoch {self.epoch} Soft constraint loss before projection: {torch.mean(torch.abs(c_hat)):.3f} after projection: {torch.mean(torch.abs(c_tilde)):.3f}"
                    )
                    self._epochprint = self.epoch
                ssl_loss_hat += self.loss_cfg.weight_loss_soft * torch.mean(torch.abs(c_hat))
                ssl_loss_tilde += self.loss_cfg.weight_loss_soft * torch.mean(torch.abs(c_tilde))
            if ssl_loss_hat < ssl_loss_tilde:
                if self.start_projection:
                    self.start_projection = False
                    print(
                        f"Stop projection: Epoch {self.epoch} Iteration: {self.training_iter} Loss before: {ssl_loss_hat}"
                    )
                return yhat, yhat, proj_iter
            elif (
                ssl_loss_hat > ssl_loss_tilde
                and self.compute_residual(c_tilde, tolerance_mode="mean") > self.cfg.training_tolerance
            ):
                if not self.start_projection:
                    print(
                        f"Start projection: Epoch {self.epoch} Iteration: {self.training_iter} Loss before: {ssl_loss_hat} Loss after: {ssl_loss_tilde}"
                    )
                # t_adanp = time.time()
                ytilde, proj_iter = self.ada_np(
                    x,
                    ytilde,
                    tolerance_mode="mean",
                )
                # print(f"AdaNP time: {time.time()-t_adanp:.6f} seconds")
                self.start_projection: bool = True
            # print(f"Predict (after projection) time: {time.time()-t_ap:.6f} seconds\n\n")
            return ytilde, yhat, proj_iter


    def compute_dc_dy(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """ Compute the jacobian of the constraints with respect to the prediction.

        Parameters
        ----------
        input: torch.Tensor
            Input tensor.
        output: torch.Tensor
            Output tensor.

        Returns
        -------
        dc_dy : torch.Tensor
            Jacobian of the constraints with respect to the prediction.

        """
        with profiler.record_function("compute_dc_dy"):
            input_unscaled, output_unscaled = self.scaler.unscale(x_scaled=input, y_scaled=output)
            self.dc_dy: torch.Tensor = self.constraints.jacobian(input_unscaled, output_unscaled)  # [BS, NC, NO]
            self.bs: int = self.dc_dy.shape[0]
            self.nc: int = self.dc_dy.shape[1]
            return self.dc_dy

    def B_f(self):
        B: torch.Tensor = self.dc_dy  # Shape: [BS, NC, NO]
        return B

    def vi_f(self, input_unscaled, output_unscaled):
        # one fused kernel: for each batch b and constraint i,
        # sum_k  dc_dy[b,i,k] * output_unscaled[b,k]
        vi_lin: torch.Tensor = torch.einsum("bik,bk->bi", self.dc_dy, output_unscaled)  # [BS,NC]
        vi: torch.Tensor = vi_lin - self.constraints(input_unscaled, output_unscaled)  # elementwise subtract
        self.vi = vi
        return vi

    def Wi_f(self):
        # shapes and dtypes
        bs, nc, no = self.dc_dy.shape  # NC == NO
        device = self.dc_dy.device
        dtype = self.dc_dy.dtype

        if self.weighting_option == "I":
            return None
            # Wi = I_{NO} for each batch
            # → allocate zeros and write 1s on the diag only (BS*NO writes)
            # Wi = torch.zeros(bs, no, no, device=device, dtype=dtype)
            # Wi.diagonal(dim1=1, dim2=2).fill_(1.0)
            # return Wi

        elif self.weighting_option == "R":
            # random in [0.1, 1.0):
            w = 0.9 * torch.rand(bs, no, device=device, dtype=dtype) + 0.1  # [BS, NO]
            Wi = torch.zeros(bs, no, no, device=device, dtype=dtype)
            Wi.diagonal(dim1=1, dim2=2).copy_(w)  # copy only BS*NO elements
            return Wi

        elif self.weighting_option == "ID":
            # instance‐dependent
            # 1) mean abs derivative per constraint: [BS, NO]
            d = self.dc_dy.abs().mean(dim=1)

            # 2) replace zeros with 1.0 in-place
            d = d.clamp_min(1.0)

            # 3) invert & normalize by the per‐batch min: [BS, NO]
            inv = 1.0 / d
            inv = inv / inv.min(dim=1, keepdim=True).values

            # 4) fill diag
            Wi = torch.zeros(bs, no, no, device=device, dtype=dtype)
            Wi.diagonal(dim1=1, dim2=2).copy_(inv)
            return Wi

        elif self.weighting_option == "BA":
            # batch‐averaged
            m = self.dc_dy.abs().mean(dim=(0, 1))  # [NO]
            m = m.clamp_min(1.0)  # replace zero
            inv = 1.0 / m  # [NO]
            inv = inv / inv.min()  # normalize
            # build single [NO, NO] diag matrix via broadcast‐mul
            eye_mat = torch.eye(no, device=device, dtype=dtype)
            Wi = eye_mat * inv  # broadcast inv over diag only
            return Wi

        elif self.weighting_option == "TF":
            # Trainable diagonal: w_i = exp(log_diag_i).
            # Initialised to 1 (log = 0); strictly positive throughout training.
            # Returns [NO, NO]; project() broadcasts to [BS, NO, NO].
            w = self.proj_weight_log_diag.to(dtype=dtype).exp()  # [NO]
            eye_mat = torch.eye(no, device=device, dtype=dtype)
            Wi = eye_mat * w  # diag(w), off-diagonals stay zero
            return Wi

        else:
            raise ValueError(f"Unknown weighting_option: {self.weighting_option}")

    def projection_tensors(self, B, v, W_inv=None):
        """Compute the Newton projection tensors B_star and v_star.

        Uses Cholesky decomposition for efficiency. When Cholesky fails (the
        gram matrix is not positive-definite due to float32 ill-conditioning),
        automatically falls back to eigendecomposition with eigenvalue clamping.

        The gram matrix is regularised as ``ε·I + B·W⁻¹·Bᵀ`` (or
        ``ε·I + B·Bᵀ`` when W_inv is None) before inversion when ``W_inv`` is given
        or ``regularise_gram`` is set, so the effective floor on all eigenvalues is ``eps_chol``.

        Parameters
        ----------
        B : ``[BS, NC, NO]``
            Constraint Jacobian.
        v : ``[BS, NC]``
            Right-hand-side vector (linearisation of constraint at current y).
        W_inv : ``[BS, NO, NO]`` or None
            Inverse weighting matrix.  None means W = I (unweighted Newton step).

        Returns
        -------
        B_star : ``[BS, NO, NO]``
            ``I − M·B``
        v_star : ``[BS, NO]``
            ``M·v``
        """
        with profiler.record_function("build_projection_tensors"):
            B_T = B.transpose(1, 2).contiguous()  # [BS, NO, NC]

            # ── 1. Compute gram matrix ─────────────────────────────────────────
            if W_inv is None:
                BWB_T = torch.bmm(B, B_T)  # [BS, NC, NC]
            else:
                BWB_T = torch.bmm(torch.bmm(B, W_inv), B_T)  # [BS, NC, NC]

            # ── 2. Regularise + symmetrise ─────────────────────────────────────
            # Regularisation and symmetrisation are skipped for the unweighted
            # case (W = I) unless cfg.regularise_gram is set.  For linear
            # constraints, B @ B^T is already symmetric and well-conditioned;
            # adding eps corrupts the exact null-space projector, leaking
            # gradients into the constraint-violating direction during SSL
            # training.  Symmetrisation is also skipped because the tiny
            # float32 asymmetry in B @ B^T, when averaged, changes the
            # gradient path and causes measurable metric drift over many steps.
            # Enable regularise_gram for ill-conditioned Jacobians (e.g.
            # bilinear / pooling constraints) where both eps and symmetrisation
            # are needed for numerical stability.
            if W_inv is not None or self.cfg.regularise_gram:
                I_nc = torch.eye(self.nc, device=B.device, dtype=B.dtype)
                BWB_T = BWB_T + self.eps_chol * I_nc
                BWB_T = 0.5 * (BWB_T + BWB_T.transpose(1, 2))

            # ── 3. Invert: Cholesky first; eigh fallback ───────────────────────
            # cholesky_ex returns (L, info) without raising an exception.
            # info[b] > 0 means batch element b is not positive-definite.
            ch, info = torch.linalg.cholesky_ex(BWB_T)
            if info.any():
                # At least one batch element failed Cholesky (float32
                # ill-conditioning). Fall back to eigh which always succeeds on
                # symmetric matrices; clamp negative eigenvalues to eps_chol to
                # recover the intended regularisation.
                vals, vecs = torch.linalg.eigh(BWB_T)  # [BS,NC],[BS,NC,NC]
                vals = vals.clamp_min(self.eps_chol)
                mid_inv = vecs @ torch.diag_embed(1.0 / vals) @ vecs.transpose(1, 2)  # [BS, NC, NC]
            else:
                mid_inv = torch.cholesky_inverse(ch)  # [BS, NC, NC]

            # ── 4. Form M = W⁻¹·Bᵀ·(BWB_T)⁻¹ (or Bᵀ·(BB_T)⁻¹ when unweighted)
            if W_inv is None:
                M = torch.bmm(B_T, mid_inv)  # [BS, NO, NC]
            else:
                M = torch.bmm(torch.bmm(W_inv, B_T), mid_inv)  # [BS, NO, NC]

            # ── 5. B_star = I − M·B ────────────────────────────────────────────
            MB = torch.bmm(M, B)  # [BS, NO, NO]
            B_star = -MB
            B_star.diagonal(dim1=1, dim2=2).add_(1.0)

            # ── 6. v_star = M·v ────────────────────────────────────────────────
            v_star = torch.bmm(M, v.unsqueeze(2)).squeeze(2)  # [BS, NO]

            return B_star, v_star

    def projection_step(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """ Performs one projection step. Returns the once projected output.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.
        output : torch.Tensor
            Output tensor.

        Returns
        -------
        y_proj_scaled : torch.Tensor
            Projected tensor.
        """

        # 1) unscale once
        input_unscaled, output_unscaled = self.scaler.unscale(x_scaled=input, y_scaled=output)

        # 2) build (and invert) Wi on correct device
        self.Wi = self.Wi_f()  # should already be on self.device
        if self.weighting_option == "I":
            W_inv = None
        else:
            Wi_inv = torch.inverse(self.Wi)

            # 3) turn any 2D Wi_inv into a true [BS,NO,NO] tensor
            if Wi_inv.dim() <= 2:
                # unsqueeze+repeat gives a contiguous [BS,NO,NO]
                W_inv = Wi_inv.unsqueeze(0).repeat(self.bs, 1, 1)
            else:
                # if it's already [BS,NO,NO], just make it contiguous once
                W_inv = Wi_inv.contiguous()

        # 4) grab B and v, and force contiguity up-front
        B = self.B_f().contiguous()  # [BS,NC,NO]
        v = self.vi_f(input_unscaled, output_unscaled).contiguous()  # [BS,NC]

        # 5) do the heavy lifting—our optimized routine already returns
        #    contiguous WB_star, Wv_star
        # t = time.time()
        WB_star, Wv_star = self.projection_tensors(B, v, W_inv=W_inv)
        # print(f"Projection tensors time: {time.time()-t:.6f} seconds")

        # 6) apply projection: make output_unscaled[:, :, None] contiguous once
        y_in = output_unscaled.unsqueeze(2).contiguous()  # [BS,NO,1]
        y_proj = torch.bmm(WB_star, y_in).squeeze(2) + Wv_star

        # 7) re-scale and return
        _, y_proj_scaled = self.scaler.scale(x_unscaled=input_unscaled, y_unscaled=y_proj)
        return y_proj_scaled

    class _ProjectionIFT(torch.autograd.Function):
        """Projection with Implicit Function Theorem (IFT) backward pass.

        Forward
        -------
        Runs the full Newton projection to convergence **without** building a
        computational graph through the iterations.  This is identical in result
        to the unrolled-autograd path but uses O(BS × NO²) memory instead of
        O(T × graph_size).

        Backward
        --------
        Applies the Gauss-Newton IFT gradient::

            ∂L/∂ŷ = B_star(y*)ᵀ · ∂L/∂y*

        where ``B_star = I − W⁻¹Bᵀ(BW⁻¹Bᵀ)⁻¹B`` is the oblique projector onto
        ``null(B)`` evaluated at the converged point ``y*``.

        For ``W = I`` (``weighting_option == "I"``) ``B_star`` is symmetric, so the
        transpose has no effect.  For weighted variants the transpose is needed
        because ``B_star`` is only self-adjoint in the W-inner product, not the
        standard one used by the loss.
        """

        @staticmethod
        def forward(
                ctx,
                yhat: torch.Tensor,
                model: "ENFORCE",
                x: torch.Tensor,
                tolerance_mode: str,
                tolerance_value: float,
                max_iter: int,
        ) -> torch.Tensor:
            raise DeprecationWarning("Using the IFT Projection path forward is currently deprecated.")
            with torch.enable_grad():
                # ── 1. Run all Newton steps detached from the network graph ──────
                y = yhat.detach().requires_grad_(True)
                model.compute_dc_dy(x, y)
                y = model.projection_step(x, y)
                y, proj_iter = model.ada_np(
                    x,
                    y,
                    tolerance_mode=tolerance_mode,
                    tolerance_value=tolerance_value,
                    max_iter=max_iter,
                )
                model._ift_proj_iter = proj_iter + 1  # +1 for the initial project step

                # ── 2. Compute B_star at y* for the backward pass ────────────────
                # B_star = I − W⁻¹Bᵀ(BW⁻¹Bᵀ)⁻¹B is independent of v, so we
                # pass a zero RHS to projection_tensors (only B_star is saved).
                y_jac = y.detach().requires_grad_(True)
                model.compute_dc_dy(x, y_jac)
                model.Wi = model.Wi_f()
                B = model.B_f().contiguous()
                v_dummy = torch.zeros(model.bs, model.nc, device=B.device, dtype=B.dtype)
                if model.weighting_option == 1:
                    W_inv = None
                else:
                    Wi_inv = torch.inverse(model.Wi)
                    W_inv = Wi_inv.unsqueeze(0).repeat(model.bs, 1, 1) if Wi_inv.dim() <= 2 else Wi_inv.contiguous()
                B_star, _ = model.projection_tensors(B, v_dummy, W_inv=W_inv)

            ctx.save_for_backward(B_star.detach())
            return y.detach()

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            raise DeprecationWarning("Using the IFT Projection path backward is currently deprecated.")
            (B_star,) = ctx.saved_tensors
            # IFT gradient: ∂L/∂ŷ = B_starᵀ · ∂L/∂y*   shape [BS, NO]
            # Transpose is needed for weighted W ≠ I (B_star is only self-adjoint
            # in the W-inner product, not the standard one used by the loss).
            grad_yhat = torch.bmm(B_star.transpose(1, 2), grad_output.unsqueeze(-1)).squeeze(-1)
            # One return value per forward() positional arg:
            # yhat, model, x, tolerance_mode, tolerance_value, max_iter
            return grad_yhat, None, None, None, None, None

