from dataclasses import dataclass
from time import time

import numpy as np
import torch
from torch.nn import Module
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Local imports
from enforce.core.scaler import Scaler
from enforce.core.constraints import Constraints
from enforce.losses.loss_core import LossCore, SSLLossCore
from enforce.engines._checks import check_model_loss
from enforce.models import ConstrainedModel
from enforce.models.constrained_model import ENFORCEConstrainedModel


@dataclass
class EvaluationConfig:
    """Options for a single evaluation run.

    Parameters
    ----------
    batch_size :
        Mini-batch size for inference.  ``None`` → single full-batch pass.
    """

    batch_size: int | None = None


@dataclass
class EvalResult:
    """Output of :meth:`Evaluator.evaluate`.

    Attributes
    ----------
    metrics :
        Dictionary of scalar evaluation metrics (R², MSE, NRMSE, MAPE per
        output, constraint residuals, optional SSL objectives, projection
        iterations, inference time).
    predictions :
        Unscaled model predictions after projection, shape ``[N, NO]``.
    predictions_before_proj :
        Unscaled model predictions before projection (``yhat``), shape
        ``[N, NO]``.  Equal to ``predictions`` for unconstrained models.
    """

    metrics: dict
    predictions: np.ndarray
    predictions_before_proj: np.ndarray


class Evaluator:
    """Evaluates a plain backbone or a :class:`ConstrainedModel` on held-out test data.

    Usage::

        cfg = EvaluationConfig(batch_size=64)
        result = Evaluator(model, loss_fn, scaler, constraints, cfg).evaluate(x_test, y_test)

    Parameters
    ----------
    model :
        An instantiated :class:`nn.Module` or :class:`ConstrainedModel`.
    loss_fn :
        An instantiated :class:`LossCore` or :class:`SSLLossCore` loss function.
    scaler : Scaler
        :class:`Scaler` object for input/output normalisation.
    constraints : Constraints
        :class:`Constraints` object for constraint residual calculation.
    config :
        Evaluation options.
    """

    def __init__(self,
                 model: Module | ConstrainedModel,
                 loss_fn: LossCore | SSLLossCore,
                 scaler: Scaler,
                 constraints: Constraints,
                 config: EvaluationConfig):
        check_model_loss(model, loss_fn)

        self.scaler: Scaler = scaler
        self.constraints: Constraints = constraints
        self.n_out_orig: int = self.scaler.no - self.constraints.ng
        self.model: Module | ConstrainedModel = model
        self.loss_fn: LossCore | SSLLossCore = loss_fn
        self.config: EvaluationConfig = config

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_inputs: torch.Tensor,
        test_outputs: torch.Tensor,
    ) -> EvalResult:
        """Run inference on *test_inputs*, compute metrics, return an :class:`EvalResult`."""
        self.model.eval()

        ytilde_s, yhat_s, proj_iter, inference_time = self._run_batches(test_inputs)

        x_u, ytilde_u = self.scaler.unscale(test_inputs, ytilde_s)
        _, yhat_u = self.scaler.unscale(test_inputs, yhat_s)
        _, y_true_u = self.scaler.unscale(test_inputs, test_outputs)

        ytilde_npy = ytilde_u.cpu().detach().numpy()
        yhat_npy = yhat_u.cpu().detach().numpy()
        y_true_npy = y_true_u.cpu().detach().numpy()

        metrics: dict = {
            "projection iterations": proj_iter,
            "inference time": inference_time,
        }
        metrics.update(self._per_output_metrics(y_true_npy, ytilde_npy))
        metrics.update(self._constraint_residuals(x_u, ytilde_u))
        metrics.update(self._inequality_metrics(x_u, ytilde_u))
        metrics.update(self._ssl_metrics(x_u, y_true_u, ytilde_u))
        self._log(metrics)


        return EvalResult(
            metrics=metrics,
            predictions=self.constraints.extract_outputs(ytilde_npy),
            predictions_before_proj=self.constraints.extract_outputs(yhat_npy),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_batches(self, test_inputs: torch.Tensor):
        """Run inference in mini-batches; return (ytilde, yhat, avg_proj_iter, avg_time)."""
        bs = self.config.batch_size or test_inputs.shape[0]
        n = test_inputs.shape[0]

        all_ytilde, all_yhat = [], []
        total_proj_iter = 0.0
        total_time = 0.0
        n_batches = 0

        for start in range(0, n, bs):
            batch = test_inputs[start : start + bs]
            t0 = time()
            if isinstance(self.model, ENFORCEConstrainedModel):
                ytilde, yhat, p_iter = self.model.predict(x=batch)
            elif isinstance(self.model, ConstrainedModel):
                ytilde, yhat = self.model(x=batch)
                p_iter = 0
                if yhat.shape[1] != self.scaler.no:
                    yhat = self.constraints.extend_outputs(y=yhat)
                if ytilde.shape[1] != self.scaler.no:
                    ytilde = self.constraints.extend_outputs(y=ytilde)
            elif isinstance(self.model, Module):
                yhat = self.model(x=batch)
                if yhat.shape[1] != self.scaler.no:
                    # Extend regular backbone prediction to extended dimension
                    yhat = self.constraints.extend_outputs(yhat)
                ytilde = yhat
                p_iter = 0
            else:
                raise NotImplementedError(f"Evaluation not implemented for model type {type(self.model)}")

            total_time += time() - t0
            all_ytilde.append(ytilde)
            all_yhat.append(yhat)
            total_proj_iter += p_iter
            n_batches += 1

        ytilde_s = torch.cat(all_ytilde, dim=0)
        yhat_s = torch.cat(all_yhat, dim=0)
        return ytilde_s, yhat_s, total_proj_iter / n_batches, total_time / n_batches

    def _per_output_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """R², MSE, NRMSE, MAPE for each original output dimension."""
        metrics = {}
        for i in range(self.n_out_orig):
            yt = np.asarray(y_true[:, i], dtype=np.float64)
            yp = np.asarray(y_pred[:, i], dtype=np.float64)
            mse = np.mean((yt - yp) ** 2)
            std = np.std(yt)
            metrics[f"r2_y{i+1}"] = r2_score(yt, yp)
            metrics[f"mse_y{i+1}"] = mse
            metrics[f"nrmse_y{i+1}"] = np.sqrt(mse) / std if std > 1e-8 else np.sqrt(mse)
            metrics[f"mape_y{i+1}"] = mean_absolute_percentage_error(yt, yp)
        return metrics

    def _constraint_residuals(self, x: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Mean and max absolute constraint residual across the test set."""
        if self.constraints.nc + self.constraints.ng == 0:
            return {"residual_avg": 0.0, "residual_max": 0.0}
        residuals = self.constraints(x, y_pred)
        abs_res = torch.abs(residuals)
        return {
            "residual_avg": abs_res.mean().item(),
            "residual_max": abs_res.max().item(),
        }

    def _inequality_metrics(self, x: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Infeasibility percentage and violation statistics per inequality."""
        if self.constraints.ng == 0:
            return {}
        y_orig = y_pred[:, :self.n_out_orig]
        tol = 1e-6
        if isinstance(self.model, ENFORCEConstrainedModel):
            tol = self.model.cfg.inference_tolerance
        metrics = {}

        g_values = self.constraints.g_values(x, y_orig)
        for i in range(self.constraints.ng):
            g_val = g_values[:, i]
            metrics[f"ineq_g{i+1}_infeasible_pct"] = (g_val > tol).float().mean().item() * 100.0
            metrics[f"ineq_g{i+1}_mean_violation"] = g_val.clamp(min=0).mean().item()
            metrics[f"ineq_g{i+1}_max_violation"] = g_val.clamp(min=0).max().item()
        return metrics

    def _ssl_metrics(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> dict:
        """Objective values at true labels and at prediction (SSL problems only)."""
        if isinstance(self.loss_fn, SSLLossCore):
            return {
                "obj_value_opt": self.loss_fn.ssl_loss(x, y_true).item(),
                "obj_value_pred": self.loss_fn.ssl_loss(x, y_pred).item(),
            }
        return {}

    def _log(self, metrics: dict):
        msg = (
            f"Inference time: {metrics['inference time']:.4f}s, "
            f"EqMeanResidual: {metrics['residual_avg']:.2e}, "
            f"EqMaxResidual: {metrics['residual_max']:.2e}"
        )
        if "obj_value_opt" in metrics:
            msg += f", ObjValueOpt: {metrics['obj_value_opt']:.2e}" f", ObjValuePred: {metrics['obj_value_pred']:.2e}"
        print(msg)
