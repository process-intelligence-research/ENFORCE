import random
import warnings

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn

from src.enforce.config import ENFORCEConfig


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

    For ``W = I`` (``weighting_option == 1``) ``B_star`` is symmetric, so the
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
        with torch.enable_grad():
            # ── 1. Run all Newton steps detached from the network graph ──────
            y = yhat.detach().requires_grad_(True)
            model.compute_dc_dy(x, y)
            y = model.project(x, y)
            y, proj_iter = model.ada_np(
                x, y,
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
            v_dummy = torch.zeros(
                model.bs, model.nc, device=B.device, dtype=B.dtype
            )
            if model.weighting_option == 1:
                W_inv = None
            else:
                Wi_inv = torch.inverse(model.Wi)
                W_inv = (
                    Wi_inv.unsqueeze(0).repeat(model.bs, 1, 1)
                    if Wi_inv.dim() <= 2
                    else Wi_inv.contiguous()
                )
            B_star, _ = model.projection_tensors(B, v_dummy, W_inv=W_inv)

        ctx.save_for_backward(B_star.detach())
        return y.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (B_star,) = ctx.saved_tensors
        # IFT gradient: ∂L/∂ŷ = B_starᵀ · ∂L/∂y*   shape [BS, NO]
        # Transpose is needed for weighted W ≠ I (B_star is only self-adjoint
        # in the W-inner product, not the standard one used by the loss).
        grad_yhat = torch.bmm(
            B_star.transpose(1, 2), grad_output.unsqueeze(-1)
        ).squeeze(-1)
        # One return value per forward() positional arg:
        # yhat, model, x, tolerance_mode, tolerance_value, max_iter
        return grad_yhat, None, None, None, None, None


class ENFORCE(nn.Module):
    """Constrained neural network that projects predictions onto the constraint manifold.

    Handles both equality constraints and inequality constraints (via the
    Fischer-Burmeister reformulation).

    Parameters
    ----------
    scaling_input, scaling_output :
        Tuples of (mean, std) tensors for input/output normalisation.
    c :
        Constraint callable ``c(x, y) -> Tensor[BS, NC]`` (unscaled inputs/outputs).
        For inequality-only problems, pass a
        :class:`FischerBurmeisterReformulation` instance directly.
        For mixed equality + inequality, pass a combined callable
        that concatenates equality residuals and FB residuals.
    fb : FischerBurmeisterReformulation or None
        When provided, the network's output layer is built with ``fb.no`` neurons
        (original outputs only). ``forward()`` appends zero columns for the FB
        dual variables (``fb.n_ineq`` columns) so the Newton projection operates
        in the full extended space without wasting parameters on the trivially-zero
        multipliers.
    constrained :
        If ``False``, run as an unconstrained MLP (no projection).
    weighting_option :
        Selects the weighting matrix W used in the weighted Newton step.
        1 = identity (no weighting); 3 = batch-averaged; 5 = instance-dependent;
        6 = random; 7 = trainable diagonal (diagonal initialised to 1, coefficients
        learned end-to-end during training).
    eps_chol :
        Regularisation scalar ``ε`` added to the gram matrix as ``ε·I`` before
        inversion.  Larger values help when Jacobians span many orders of magnitude
        (e.g. bilinear constraints).  Default: ``1e-8``.
    """

    def __init__(
        self,
        scaling_input,
        scaling_output,
        c,
        config: ENFORCEConfig = None,
        fb=None,
        constrained=True,
        weighting_option=5,
        ssl_loss=None,
        jac=None,
        eps_chol: float = 1e-8,
    ):
        cfg = config if config is not None else ENFORCEConfig()
        self.cfg = cfg

        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(cfg.random_seed)
        super(ENFORCE, self).__init__()

        # ── Network architecture ───────────────────────────────────────────────
        # If fb is provided, the network predicts only the original outputs (fb.no).
        # Zero columns for FB dual variables are appended in forward().
        n_net_outputs = fb.no if fb is not None else cfg.output_neurons

        self.input_layer = nn.Linear(cfg.input_neurons, cfg.hidden_neurons)
        self.hidden_layer = nn.Linear(cfg.hidden_neurons, cfg.hidden_neurons)
        self.hidden_activation = nn.ReLU()
        self.output_layer = nn.Linear(cfg.hidden_neurons, n_net_outputs)
        self.loss_function = nn.MSELoss()
        self.ssl_loss = ssl_loss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.losses = []
        self.jac = jac

        self.ni = cfg.input_neurons
        # self.no is the full extended output dimension (including FB multipliers).
        self.no = (fb.no + fb.n_ineq) if fb is not None else cfg.output_neurons

        self.mean_input = torch.as_tensor(
            scaling_input[0], device=self.device, dtype=torch.float32
        )
        self.std_input = torch.as_tensor(
            scaling_input[1], device=self.device, dtype=torch.float32
        )
        self.mean_output = torch.as_tensor(
            scaling_output[0], device=self.device, dtype=torch.float32
        )
        self.std_output = torch.as_tensor(
            scaling_output[1], device=self.device, dtype=torch.float32
        )
        self._c = c
        self.fb = fb
        self.constrained = constrained
        self.weighting_option = weighting_option
        self.eps_chol = eps_chol
        self.ift_backward = cfg.ift_backward

        # Trainable diagonal projection weighting matrix (option 7).
        # Log-parameterised: exp(0) = 1 at init, strictly positive always.
        if weighting_option == 7:
            self.proj_weight_log_diag = nn.Parameter(torch.zeros(self.no))

        self.epoch = 1
        self.training_iter = 0
        self.start_projection = False
        self._epochprint = None

    def check_system(self):
        # Issue a warning if the system is determined
        if self.nc == self.no:
            warnings.warn(
                "The system is determined. Maybe you already know your underlying model!"
            )

        # Ensure the number of constraints is not greater than the number of outputs
        try:
            assert self.nc <= self.no
        except AssertionError:
            raise ValueError("Too many constraints!")

    def c(self, x, y):
        ci = self._c(x, y)
        if isinstance(ci, tuple):
            self.nc = len(ci)
            self.check_system()
            return torch.stack(ci, dim=1)
        elif isinstance(ci, torch.Tensor) and ci.ndim == 1:
            self.nc = 1  # Since c returns a tensor of shape [BS]
            self.check_system()
            return ci.unsqueeze(1)
        elif (
            isinstance(ci, torch.Tensor) and ci.ndim > 1
        ):  # return a tensor of shape [BS, NC]
            self.nc = ci.shape[1]  # Number of constraints
            self.check_system()
            return ci

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_activation(x)
        for i in range(self.cfg.hidden_layers):
            x = self.hidden_layer(x)
            x = self.hidden_activation(x)
        x = self.output_layer(x)
        if self.fb is not None:
            # Append zero columns for FB dual variables.
            # At feasible interior points the correct KKT value is λ_i = 0;
            # the Newton projection will set the correct λ at active constraints.
            zeros = torch.zeros(
                x.shape[0], self.fb.n_ineq, device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, zeros], dim=1)
        return x

    def loss(self, x, y):
        # t = time.time()
        ytilde, yhat, proj_iter = self.predict(x, y, training=True)
        # print(f"Predict time: {time.time()-t:.6f} seconds")
        loss_data_before_projection = self.loss_function(y, yhat)
        loss_data_after_projection = self.loss_function(y, ytilde)
        loss_displacement = torch.mean(
            (yhat - ytilde) ** 2
        )  # this is zero if the projection is not done
        loss = self.cfg.weight_loss_displacement * loss_displacement

        x_unscaled, ytilde_unscaled = self.unscale(x, ytilde)
        if self.cfg.soft_constrained:
            c = self.c(x=x_unscaled, y=ytilde_unscaled)
            loss += self.cfg.weight_loss_soft * torch.mean(torch.abs(c))

        if self.cfg.supervised:
            loss += loss_data_after_projection

        else:
            loss_ssl = self.ssl_loss(x_unscaled, ytilde_unscaled)
            loss += loss_ssl

        return (
            loss,
            loss_data_after_projection,
            loss_displacement,
            loss_data_before_projection,
            ytilde,
            yhat,
            proj_iter,
        )

    def compute_residual(self, c, tolerance_mode):
        if tolerance_mode == "mean":
            tolerance_value = torch.mean(torch.abs(c))
        elif tolerance_mode == "max":
            tolerance_value = torch.max(torch.abs(c))
        else:
            raise ValueError("Invalid tolerance mode. Choose 'mean' or 'max'.")
        return tolerance_value

    def ada_np(
        self,
        x,
        y,
        tolerance_mode="mean",
        tolerance_value=None,
        max_iter=None,
    ):
        if tolerance_value is None:
            tolerance_value = self.cfg.training_tolerance
        if max_iter is None:
            max_iter = self.cfg.max_it

        proj_iter = 1
        c = torch.zeros(x.shape[0], self.nc, device=x.device, dtype=x.dtype)
        input_unscaled, output_unscaled = self.unscale(x, y)
        c = self.c(x=input_unscaled, y=output_unscaled)  # Shape: [BS, NC]
        c_res = self.compute_residual(c, tolerance_mode)
        while c_res > tolerance_value and proj_iter < max_iter:
            self.compute_dc_dy(x, y)
            y = self.project(x, y)
            proj_iter += 1
            input_unscaled, output_unscaled = self.unscale(x, y)
            c = self.c(x=input_unscaled, y=output_unscaled)  # Shape: [BS, NC]
            c_res = self.compute_residual(c, tolerance_mode)
        if proj_iter == self.cfg.max_it:
            print(f"Max projection iteration reached ({self.cfg.max_it})")
        return y, proj_iter

    def predict(self, x, y=None, training=False):
        # tfdc = time.time()
        proj_iter = 0
        if self.constrained and self.epoch >= self.cfg.epoch_start_hard_constrained:
            yhat = self.forward(x)

            if training and self.ift_backward:
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
                if self.cfg.supervised and self.cfg.ada_np_auto_activation:
                    with torch.enable_grad():
                        # Detach from the network graph so no gradients flow to
                        # weights from this preview; requires_grad_(True) so
                        # compute_dc_dy can call torch.autograd.grad w.r.t. y.
                        yhat_preview = yhat.detach().requires_grad_(True)
                        self.compute_dc_dy(x, yhat_preview)
                        ytilde_preview = self.project(x, yhat_preview).detach()
                    loss_before = self.loss_function(y, yhat)
                    loss_after  = self.loss_function(y, ytilde_preview)
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
                ytilde = _ProjectionIFT.apply(
                    yhat, self, x, "mean", self.cfg.training_tolerance, self.cfg.max_it
                )
                proj_iter = getattr(self, "_ift_proj_iter", 1)
                self.start_projection = True
                return ytilde, yhat, proj_iter

            self.compute_dc_dy(x, yhat)
            # print(f"Compute forward and dc_dy time: {time.time()-tfdc:.6f} seconds")
            # t = time.time()
            ytilde = self.project(x, yhat)
            # print(f"Projection time: {time.time()-t:.6f} seconds")
            # t_ap = time.time()
            proj_iter += 1
            if training:
                if self.cfg.supervised:
                    if self.cfg.ada_np_auto_activation:
                        loss_data_before_projection = self.loss_function(y, yhat)
                        loss_data_after_projection = self.loss_function(y, ytilde)

                        # Criteria to project (note the first projection is already done)
                        if loss_data_before_projection < loss_data_after_projection:
                            # If the projection does not improve the prediction, then it is not considered
                            ytilde = yhat
                            if self.start_projection:
                                self.start_projection = False
                                print(
                                    f"Stop projection: Epoch {self.epoch} Iteration: {self.training_iter} Loss before: {loss_data_before_projection}"
                                )
                        else:
                            if not self.start_projection:
                                print(
                                    f"Start projection: Epoch {self.epoch} Iteration: {self.training_iter} Loss before: {loss_data_before_projection} Loss after: {loss_data_after_projection}"
                                )
                            ytilde, proj_iter = self.ada_np(
                                x,
                                ytilde,
                                tolerance_mode="mean",
                            )
                            self.start_projection = True
                    else:
                        # Always project to convergence without checking loss improvement.
                        ytilde, proj_iter = self.ada_np(
                            x,
                            ytilde,
                            tolerance_mode="mean",
                        )
                        self.start_projection = True
                else:
                    if self.cfg.verbose:
                        x_unscaled, yhat_unscaled = self.unscale(x, yhat)
                        _, ytilde_unscaled = self.unscale(x, ytilde)
                        ssl_loss_hat = self.ssl_loss(x_unscaled, yhat_unscaled)
                        ssl_loss_tilde = self.ssl_loss(x_unscaled, ytilde_unscaled)

                        if self.epoch % 1 == 0 and self.epoch != self._epochprint:
                            avg_pred_obj_value = torch.mean(ssl_loss_tilde)
                            x_u, y_u = self.unscale(x, y)
                            avg_computed_obj_value = torch.mean(self.ssl_loss(x_u, y_u))
                            print(
                                f"Epoch {self.epoch} Average computed obj value: {avg_computed_obj_value:.3f} Average predicted obj value: {avg_pred_obj_value:.3f}"
                            )
                            print(
                                f"Difference objective: {(avg_pred_obj_value - avg_computed_obj_value):.3f}"
                            )
                            print(
                                f"Relative difference objective: {((avg_pred_obj_value - avg_computed_obj_value) / avg_computed_obj_value * 100):.3f}%"
                            )
                            self._epochprint = self.epoch
                        if self.cfg.soft_constrained:
                            c_hat = self.c(x=x_unscaled, y=yhat_unscaled)
                            c_tilde = self.c(x=x_unscaled, y=ytilde_unscaled)
                            if self.epoch % 1 == 0 and self.epoch != self._epochprint:
                                print(
                                    f"Epoch {self.epoch} Soft constraint loss before projection: {torch.mean(torch.abs(c_hat)):.3f} after projection: {torch.mean(torch.abs(c_tilde)):.3f}"
                                )
                                self._epochprint = self.epoch
                            ssl_loss_hat += self.cfg.weight_loss_soft * torch.mean(
                                torch.abs(c_hat)
                            )
                            ssl_loss_tilde += self.cfg.weight_loss_soft * torch.mean(
                                torch.abs(c_tilde)
                            )
                        if ssl_loss_hat < ssl_loss_tilde:
                            ytilde = yhat
                            if self.start_projection:
                                self.start_projection = False
                                print(
                                    f"Stop projection: Epoch {self.epoch} Iteration: {self.training_iter} Loss before: {ssl_loss_hat}"
                                )
                        elif (
                            ssl_loss_hat > ssl_loss_tilde
                            and self.compute_residual(c_tilde, tolerance_mode="mean")
                            > self.cfg.training_tolerance
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
                            self.start_projection = True
                    else:
                        # t_adanp = time.time()
                        ytilde, proj_iter = self.ada_np(
                            x,
                            ytilde,
                            tolerance_mode="mean",
                        )
                        # print(f"AdaNP time: {time.time()-t_adanp:.6f} seconds")
                        self.start_projection = True

            else:  # inference
                ytilde, proj_iter = self.ada_np(
                    x,
                    ytilde,
                    tolerance_mode="max",
                    tolerance_value=self.cfg.inference_tolerance,
                    max_iter=self.cfg.max_it,
                )
                print(f"Projection iterations inference: {proj_iter}")

        else:
            yhat = self.forward(x)
            ytilde = yhat
            if training:
                self.training_iter += 1
        # print(f"Predict (after projection) time: {time.time()-t_ap:.6f} seconds\n\n")
        return ytilde, yhat, proj_iter

    def unscale(self, x_scaled, y_scaled):
        with profiler.record_function("unscale"):
            x = x_scaled * self.std_input + self.mean_input
            if (self.std_output >= 1e-5).all():
                y = y_scaled * self.std_output + self.mean_output
            else:
                # Avoid division by zero if std is zero
                y = y_scaled + self.mean_output
        return x, y

    def scale(self, x_unscaled, y_unscaled):
        x = (x_unscaled - self.mean_input) / self.std_input
        if (self.std_output >= 1e-5).all():
            y = (y_unscaled - self.mean_output) / self.std_output
        else:
            # Avoid division by zero if std is zero
            y = y_unscaled - self.mean_output
        # y = (y_unscaled - self.mean_output) / self.std_output
        return x, y

    def compute_dc_dy(self, input, output):
        with profiler.record_function("compute_dc_dy"):
            if self.jac:
                input_unscaled, output_unscaled = self.unscale(input, output)
                self.bs = input_unscaled.shape[0]
                self.nc = input_unscaled.shape[1]
                self.dc_dy = self.jac(output_unscaled)
                return self.dc_dy
            else:
                input_unscaled, output_unscaled = self.unscale(input, output)
                c = self.c(x=input_unscaled, y=output_unscaled)  # Shape: [BS, NC]

                self.bs = c.size(0)
                self.nc = c.size(1)

                # Initialize the Jacobian tensor: [BS, num_constraints, num_outputs]
                dc_dy = torch.zeros(
                    self.bs,
                    self.nc,
                    self.no,
                    dtype=output_unscaled.dtype,
                    device=output_unscaled.device,
                )

                for i in range(self.nc):  # Loop over constraint components
                    # Create grad_outputs tensor to select c_i
                    grad_outputs = torch.zeros_like(c)  # Shape: [BS, NC]
                    grad_outputs[:, i] = 1  # Set grad_outputs for c_i

                    # Compute gradients: grad_c_i_wrt_y = [BS, NO]

                    grad_c_i_wrt_y = torch.autograd.grad(
                        outputs=c,
                        inputs=output_unscaled,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                    )[0]  # Shape: [BS, NO]

                    # Handle the case where grad might be None
                    if grad_c_i_wrt_y is None:
                        grad_c_i_wrt_y = torch.zeros_like(output_unscaled)

                    # Store gradients in the Jacobian tensor
                    dc_dy[:, i, :] = grad_c_i_wrt_y  # Shape: [BS, NO]

                self.dc_dy = dc_dy  # Shape: [BS, NC, NO]
                return self.dc_dy

    def B_f(self):
        B = self.dc_dy  # Shape: [BS, NC, NO]
        return B

    def vi_f(self, input_unscaled, output_unscaled):
        # one fused kernel: for each batch b and constraint i,
        # sum_k  dc_dy[b,i,k] * output_unscaled[b,k]
        vi_lin = torch.einsum("bik,bk->bi", self.dc_dy, output_unscaled)  # [BS,NC]
        vi = vi_lin - self.c(input_unscaled, output_unscaled)  # elementwise subtract
        self.vi = vi
        return vi

    def Wi_f(self):
        # shapes and dtypes
        bs, nc, no = self.dc_dy.shape  # NC == NO
        device = self.dc_dy.device
        dtype = self.dc_dy.dtype

        if self.weighting_option == 1:
            return None
            # Wi = I_{NO} for each batch
            # → allocate zeros and write 1s on the diag only (BS*NO writes)
            # Wi = torch.zeros(bs, no, no, device=device, dtype=dtype)
            # Wi.diagonal(dim1=1, dim2=2).fill_(1.0)
            # return Wi

        elif self.weighting_option == 6:
            # random in [0.1, 1.0):
            w = 0.9 * torch.rand(bs, no, device=device, dtype=dtype) + 0.1  # [BS, NO]
            Wi = torch.zeros(bs, no, no, device=device, dtype=dtype)
            Wi.diagonal(dim1=1, dim2=2).copy_(w)  # copy only BS*NO elements
            return Wi

        elif self.weighting_option == 5:
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

        elif self.weighting_option == 3:
            # batch‐averaged
            m = self.dc_dy.abs().mean(dim=(0, 1))  # [NO]
            m = m.clamp_min(1.0)  # replace zero
            inv = 1.0 / m  # [NO]
            inv = inv / inv.min()  # normalize
            # build single [NO, NO] diag matrix via broadcast‐mul
            I = torch.eye(no, device=device, dtype=dtype)
            Wi = I * inv  # broadcast inv over diag only
            return Wi

        elif self.weighting_option == 7:
            # Trainable diagonal: w_i = exp(log_diag_i).
            # Initialised to 1 (log = 0); strictly positive throughout training.
            # Returns [NO, NO]; project() broadcasts to [BS, NO, NO].
            w = self.proj_weight_log_diag.to(dtype=dtype).exp()  # [NO]
            I = torch.eye(no, device=device, dtype=dtype)
            Wi = I * w  # diag(w), off-diagonals stay zero
            return Wi

        else:
            raise ValueError(f"Unknown weighting_option: {self.weighting_option}")

    def projection_tensors(self, B, v, W_inv=None):
        """Compute the Newton projection tensors B_star and v_star.

        Uses Cholesky decomposition for efficiency. When Cholesky fails (the
        gram matrix is not positive-definite due to float32 ill-conditioning),
        automatically falls back to eigendecomposition with eigenvalue clamping.

        The gram matrix is always regularised as ``ε·I + B·W⁻¹·Bᵀ`` (or
        ``ε·I + B·Bᵀ`` when W_inv is None) before inversion, so the effective
        floor on all eigenvalues is ``eps_chol``.

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
                BWB_T = torch.bmm(B, B_T)                       # [BS, NC, NC]
            else:
                BWB_T = torch.bmm(torch.bmm(B, W_inv), B_T)    # [BS, NC, NC]

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
                vals, vecs = torch.linalg.eigh(BWB_T)           # [BS,NC],[BS,NC,NC]
                vals = vals.clamp_min(self.eps_chol)
                mid_inv = (
                    vecs @ torch.diag_embed(1.0 / vals) @ vecs.transpose(1, 2)
                )                                                 # [BS, NC, NC]
            else:
                mid_inv = torch.cholesky_inverse(ch)             # [BS, NC, NC]

            # ── 4. Form M = W⁻¹·Bᵀ·(BWB_T)⁻¹ (or Bᵀ·(BB_T)⁻¹ when unweighted)
            if W_inv is None:
                M = torch.bmm(B_T, mid_inv)                      # [BS, NO, NC]
            else:
                M = torch.bmm(torch.bmm(W_inv, B_T), mid_inv)   # [BS, NO, NC]

            # ── 5. B_star = I − M·B ────────────────────────────────────────────
            MB = torch.bmm(M, B)                                  # [BS, NO, NO]
            B_star = -MB
            B_star.diagonal(dim1=1, dim2=2).add_(1.0)

            # ── 6. v_star = M·v ────────────────────────────────────────────────
            v_star = torch.bmm(M, v.unsqueeze(2)).squeeze(2)    # [BS, NO]

            return B_star, v_star

    def project(self, input, output):
        # 1) unscale once
        input_unscaled, output_unscaled = self.unscale(input, output)

        # 2) build (and invert) Wi on correct device
        self.Wi = self.Wi_f()  # should already be on self.device
        if self.weighting_option == 1:
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
        _, y_proj_scaled = self.scale(input_unscaled, y_proj)
        return y_proj_scaled
