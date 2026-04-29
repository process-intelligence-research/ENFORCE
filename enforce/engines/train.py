import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import optim

from enforce.core.model import ENFORCE


@dataclass
class TrainingConfig:
    """Hyperparameters for a single training run.

    Parameters
    ----------
    batch_size :
        Mini-batch size.
    epochs :
        Number of full passes over the training set.
    learning_rate :
        Adam learning rate.
    random_seed :
        Seed for data-shuffle reproducibility.
    soft_inequalities :
        List of inequality callables ``g_i(x, y) <= 0``.  When provided
        together with ``soft_weight > 0`` and an unconstrained model, a
        penalty ``soft_weight * mean(max(0, g_i)^2)`` is added to the loss.
    soft_weight :
        Coefficient for the soft inequality penalty.
    n_original_outputs :
        For FB problems, the number of *original* outputs (``fb.no``), used
        to strip the appended lambda columns before evaluating soft penalties.
        ``None`` means no stripping (non-FB problems).
    """

    batch_size: int = 200
    epochs: int = 1000
    learning_rate: float = 1e-3
    random_seed: int = 42
    soft_inequalities: list[callable] | None = field(default=None, repr=False)
    soft_weight: float = 0.0
    n_original_outputs: int | None = None


class Trainer:
    """Trains an :class:`ENFORCE` model given a :class:`TrainingConfig`.

    Usage::

        cfg = TrainingConfig(batch_size=64, epochs=1200, learning_rate=1e-4)
        model = Trainer(model, cfg).fit(train_inputs, train_outputs)

    Parameters
    ----------
    model :
        An instantiated :class:`ENFORCE` model.
    config :
        Training hyperparameters.
    """

    def __init__(self, model: ENFORCE, config: TrainingConfig):
        self.model = model
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
    ) -> ENFORCE:
        """Train the model and return it."""
        self._seed()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()

        for epoch in range(self.config.epochs):
            self.model.epoch = epoch
            inputs_s, outputs_s = self._shuffle(train_inputs, train_outputs, epoch)
            stats = self._run_epoch(inputs_s, outputs_s, optimizer)
            self._record(stats)
            self._log(epoch, stats)

        return self.model

    # ── Private helpers ───────────────────────────────────────────────────────

    def _seed(self):
        seed = self.config.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    def _shuffle(self, inputs: torch.Tensor, outputs: torch.Tensor, epoch: int):
        torch.manual_seed(self.config.random_seed + epoch)
        perm = torch.randperm(inputs.size(0))
        return inputs[perm], outputs[perm]

    def _run_epoch(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> dict:
        """Run one full epoch; return batch-averaged stats."""
        cfg = self.config
        n = inputs.size(0)
        n_batches = (n + cfg.batch_size - 1) // cfg.batch_size

        totals: dict = {
            "loss": 0.0,
            "loss_displacement": 0.0,
            "loss_data_after": 0.0,
            "loss_data_before": 0.0,
            "proj_iter": 0.0,
            "constraint_avg": 0.0,
            "constraint_max": 0.0,
            "obj_opt": 0.0,
            "obj_pred": 0.0,
        }
        is_opt = self.model.ssl_loss is not None

        for i in range(n_batches):
            b_in = inputs[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            b_out = outputs[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            batch = self._run_batch(b_in, b_out, optimizer, is_opt)
            for k in totals:
                totals[k] += batch[k]

        return {k: v / n_batches for k, v in totals.items()}

    def _run_batch(
        self,
        b_in: torch.Tensor,
        b_out: torch.Tensor,
        optimizer: optim.Optimizer,
        is_opt: bool,
    ) -> dict:
        """One mini-batch forward + backward; returns raw (un-averaged) values."""
        cfg = self.config

        optimizer.zero_grad()
        loss, loss_data_after, loss_displacement, loss_data_before, ytilde, yhat, proj_iter = self.model.loss(
            b_in, b_out
        )

        if cfg.soft_inequalities is not None and cfg.soft_weight > 0.0:
            loss = loss + self._soft_penalty(b_in, yhat)

        loss.backward()
        optimizer.step()

        x_u, y_u = self.model.unscale(b_in, b_out)
        _, ytilde_u = self.model.unscale(b_in, ytilde)
        c_res = self.model.c(x_u, ytilde_u)
        c_avg = torch.mean(torch.abs(c_res)).item()
        c_max = torch.max(torch.abs(c_res)).item()

        if is_opt:
            _, yhat_u = self.model.unscale(b_in, yhat)
            obj_opt = self.model.ssl_loss(x_u, y_u).item()
            obj_pred = self.model.ssl_loss(x_u, ytilde_u).item()
            loss_data_after_val = obj_pred
            loss_data_before_val = self.model.ssl_loss(x_u, yhat_u).item()
        else:
            obj_opt = obj_pred = 0.0
            loss_data_after_val = loss_data_after.item()
            loss_data_before_val = loss_data_before.item()

        return {
            "loss": loss.item(),
            "loss_displacement": loss_displacement.item(),
            "loss_data_after": loss_data_after_val,
            "loss_data_before": loss_data_before_val,
            "proj_iter": proj_iter,
            "constraint_avg": c_avg,
            "constraint_max": c_max,
            "obj_opt": obj_opt,
            "obj_pred": obj_pred,
        }

    def _soft_penalty(self, b_in: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        x_u, yhat_u = self.model.unscale(b_in, yhat)
        no = cfg.n_original_outputs if cfg.n_original_outputs is not None else yhat_u.shape[1]
        y_raw = yhat_u[:, :no]
        return cfg.soft_weight * sum(
            torch.mean(torch.clamp(g_i(x_u, y_raw), min=0.0) ** 2) for g_i in cfg.soft_inequalities
        )

    def _record(self, stats: dict):
        """Append one epoch's stats to ``model.losses``."""
        if not self.model.constrained:
            entry = {"loss_unconstrained": stats["loss_data_after"]}
        else:
            entry = {
                "loss_data_after_projection": stats["loss_data_after"],
                "loss_displacement": stats["loss_displacement"],
                "loss_data_before_projection": stats["loss_data_before"],
                "projection_iterations": stats["proj_iter"],
            }
            if self.model.ssl_loss is not None:
                entry.update(
                    {
                        "objective_value_optimization": stats["obj_opt"],
                        "objective_value_prediction": stats["obj_pred"],
                    }
                )
        self.model.losses.append(entry)

    def _log(self, epoch: int, stats: dict):
        is_opt = self.model.ssl_loss is not None
        log_this_epoch = (is_opt and (epoch + 1) % 1 == 0) or (epoch + 1) % 100 == 0
        if not log_this_epoch:
            return
        msg = (
            f"Epoch {epoch + 1}/{self.config.epochs}, "
            f"Loss: {stats['loss']:.2e}, "
            f"LossDispl: {stats['loss_displacement']:.2e}, "
            f"EqMeanResidual: {stats['constraint_avg']:.2e}, "
            f"EqMaxResidual: {stats['constraint_max']:.2e}"
        )
        if is_opt:
            msg += f", ObjValueOpt: {stats['obj_opt']:.2e}, ObjValuePred: {stats['obj_pred']:.2e}"
        print(msg)
