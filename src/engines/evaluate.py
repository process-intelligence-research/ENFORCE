from dataclasses import dataclass, field
from time import time

import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from src.enforce.model import ENFORCE


@dataclass
class EvaluationConfig:
    """Options for a single evaluation run.

    Parameters
    ----------
    batch_size :
        Mini-batch size for inference.  ``None`` → single full-batch pass.
    n_original_outputs :
        For FB problems, the number of *original* outputs (``fb.no``).
        Controls which columns are used for per-output metrics and inequality
        feasibility checks.  ``None`` means use all output columns.
    inequalities :
        List of inequality callables ``g_i(x, y) <= 0``.  When provided,
        feasibility percentage and violation statistics are added to metrics.
    """

    batch_size: int | None = None
    n_original_outputs: int | None = None
    inequalities: list | None = field(default=None, repr=False)


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
    """Evaluates an :class:`ENFORCE` model on held-out test data.

    Usage::

        cfg = EvaluationConfig(batch_size=64, n_original_outputs=fb.no,
                               inequalities=fb.inequalities)
        result = Evaluator(model, cfg).evaluate(x_test, y_test, scaling_params)

    Parameters
    ----------
    model :
        A trained :class:`ENFORCE` model.
    config :
        Evaluation options.
    """

    def __init__(self, model: ENFORCE, config: EvaluationConfig):
        self.model = model
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_inputs: torch.Tensor,
        test_outputs: torch.Tensor,
        scaling_params: dict,
    ) -> EvalResult:
        """Run inference on *test_inputs*, compute metrics, return an :class:`EvalResult`."""
        self.model.eval()

        ytilde_s, yhat_s, proj_iter, inference_time = self._run_batches(test_inputs)

        x_u, ytilde_u = self.model.unscale(test_inputs, ytilde_s)
        _, yhat_u = self.model.unscale(test_inputs, yhat_s)
        _, y_true_u = self.model.unscale(test_inputs, test_outputs)

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
            predictions=ytilde_npy,
            predictions_before_proj=yhat_npy,
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
            ytilde, yhat, p_iter = self.model.predict(batch)
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
        n_out = self.config.n_original_outputs if self.config.n_original_outputs is not None else y_true.shape[1]
        metrics = {}
        for i in range(n_out):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            mse = np.mean((yt - yp) ** 2)
            std = np.std(yt)
            metrics[f"r2_y{i+1}"] = r2_score(yt, yp)
            metrics[f"mse_y{i+1}"] = mse
            metrics[f"nrmse_y{i+1}"] = np.sqrt(mse) / std if std > 1e-8 else np.sqrt(mse)
            metrics[f"mape_y{i+1}"] = mean_absolute_percentage_error(yt, yp)
        return metrics

    def _constraint_residuals(self, x: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Mean and max absolute constraint residual across the test set."""
        residuals = self.model.c(x, y_pred)
        # model.c() always returns a [BS, NC] tensor (it handles the tuple case
        # internally).  Call it directly to get the correctly shaped result.
        abs_res = torch.abs(residuals)
        return {
            "residual_avg": abs_res.mean().item(),
            "residual_max": abs_res.max().item(),
        }

    def _inequality_metrics(self, x: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Infeasibility percentage and violation statistics per inequality."""
        if not self.config.inequalities:
            return {}
        cfg = self.config
        n_out = cfg.n_original_outputs if cfg.n_original_outputs is not None else y_pred.shape[1]
        y_orig = y_pred[:, :n_out]
        tol = self.model.cfg.inference_tolerance
        metrics = {}
        for i, g_i in enumerate(cfg.inequalities):
            g_val = g_i(x, y_orig)
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
        if self.model.ssl_loss is None:
            return {}
        return {
            "obj_value_opt": self.model.ssl_loss(x, y_true).item(),
            "obj_value_pred": self.model.ssl_loss(x, y_pred).item(),
        }

    def _log(self, metrics: dict):
        msg = (
            f"Inference time: {metrics['inference time']:.4f}s, "
            f"EqMeanResidual: {metrics['residual_avg']:.2e}, "
            f"EqMaxResidual: {metrics['residual_max']:.2e}"
        )
        if "obj_value_opt" in metrics:
            msg += f", ObjValueOpt: {metrics['obj_value_opt']:.2e}" f", ObjValuePred: {metrics['obj_value_pred']:.2e}"
        print(msg)
