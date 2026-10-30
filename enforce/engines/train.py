import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import optim
from torch.nn import Module

# Local imports
from enforce.core.scaler import Scaler
from enforce.core.constraints import Constraints
from enforce.losses.enforce_loss import ENFORCELoss, ENFORCESSLLoss
from enforce.losses.loss_core import LossCore, SSLLossCore
from enforce.engines._checks import check_model_loss
from enforce.models import ConstrainedModel
from enforce.models.constrained_model import ENFORCEConstrainedModel


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
    """

    batch_size: int = 200
    epochs: int = 1000
    learning_rate: float = 1e-3
    random_seed: int = 42


class Trainer:
    """Trains a plain backbone or a :class:`ConstrainedModel` given a :class:`TrainingConfig`.

    Usage::

        cfg = TrainingConfig(batch_size=64, epochs=1200, learning_rate=1e-4)
        model = Trainer(model, loss_fn, scaler, constraints, cfg).fit(train_inputs, train_outputs)

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
        Training hyperparameters.
    """

    def __init__(self,
                 model: Module | ConstrainedModel,
                 loss_fn: LossCore | SSLLossCore,
                 scaler: Scaler,
                 constraints: Constraints,
                 config: TrainingConfig):
        check_model_loss(model, loss_fn)

        self.scaler: Scaler = scaler
        self.constraints: Constraints = constraints
        self.model: Module | ConstrainedModel = model
        self.loss_fn: LossCore | SSLLossCore = loss_fn
        self.config: TrainingConfig = config

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        train_inputs: torch.Tensor,
        train_outputs: torch.Tensor,
    ) -> Module:
        """Train the model and return it."""
        self._seed()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()

        for epoch in range(self.config.epochs):
            if isinstance(self.model, ConstrainedModel):
                self.model.constraint_module.epoch = epoch
            else:
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

        if isinstance(self.model, ConstrainedModel):
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
        else:
            totals = {
                "loss": 0.0,
                "loss_data_before": 0.0,
            }

        is_opt = isinstance(self.loss_fn, SSLLossCore)

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
        optimizer.zero_grad()
        proj_iter = 0

        # Return the extended ytilde and yhat tensors depending on the model type
        if isinstance(self.model, ENFORCEConstrainedModel):
            ytilde, yhat, proj_iter = self.model.predict(x=b_in, y=b_out)
        elif isinstance(self.model, ConstrainedModel):
            ytilde, yhat = self.model(x=b_in)
            if yhat.shape[1] != self.scaler.no:
                yhat = self.constraints.extend_outputs(y=yhat)
            if ytilde.shape[1] != self.scaler.no:
                ytilde = self.constraints.extend_outputs(y=ytilde)
        elif isinstance(self.model, Module):
            yhat = self.model(x=b_in)
            if yhat.shape[1] != self.scaler.no:
                yhat = self.constraints.extend_outputs(y=yhat)
            ytilde = yhat
        else:
            raise NotImplementedError(f"Training not implemented for model type {type(self.model)}")

        # Compute loss depending on the loss function type
        if isinstance(self.loss_fn, ENFORCELoss) or isinstance(self.loss_fn, ENFORCESSLLoss):
            loss, loss_data_after, loss_displacement, loss_data_before = self.loss_fn.loss(
                ytilde=ytilde, yhat=yhat, y=b_out, x=b_in
            )
        elif isinstance(self.loss_fn, LossCore):
            loss = self.loss_fn(yhat=yhat, y=b_out, x=b_in)
        elif isinstance(self.loss_fn, SSLLossCore):
            loss = self.loss_fn(x=b_in, yhat=yhat)
        else:
            raise NotImplementedError(f"Training not implemented for loss function type {type(self.loss_fn)}")

        loss.backward()
        optimizer.step()

        if isinstance(self.model, ConstrainedModel):
            x_u, y_u = self.scaler.unscale(b_in, b_out)
            _, ytilde_u = self.scaler.unscale(b_in, ytilde)
            c_res = self.constraints(x_u, ytilde_u)
            c_avg = torch.mean(torch.abs(c_res)).item()
            c_max = torch.max(torch.abs(c_res)).item()
            if is_opt:
                _, yhat_u = self.scaler.unscale(b_in, yhat)
                obj_opt = self.loss_fn.ssl_loss(x_u, y_u).item()
                obj_pred = self.loss_fn.ssl_loss(x_u, ytilde_u).item()
                loss_data_after_val = obj_pred
                loss_data_before_val = self.loss_fn.ssl_loss(x_u, yhat_u).item()
            else:
                obj_opt = obj_pred = 0.0
                loss_data_after_val = loss_data_after.item()
                loss_data_before_val = loss_data_before.item()

            batch_return = {
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
            return batch_return

        else:
            if is_opt:
                x_u, yhat_u = self.scaler.unscale(b_in, yhat)
                loss_data = self.loss_fn.ssl_loss(x_u, yhat_u).item()
            else:
                # Data term only (e.g. without the penalty of SoftConstrainedLoss)
                loss_data = self.loss_fn.data_loss(yhat, b_out).item()
            return {
                "loss": loss.item(),
                "loss_data_before": loss_data,
            }

    def _record(self, stats: dict):
        """Append one epoch's stats to ``model.losses``."""
        if isinstance(self.model, ConstrainedModel):
            entry = {
                "loss_data_after_projection": stats["loss_data_after"],
                "loss_displacement": stats["loss_displacement"],
                "loss_data_before_projection": stats["loss_data_before"],
                "projection_iterations": stats["proj_iter"],
            }
            if isinstance(self.loss_fn, SSLLossCore):
                entry.update(
                    {
                        "objective_value_optimization": stats["obj_opt"],
                        "objective_value_prediction": stats["obj_pred"],
                    }
                )
        else:
            entry = {"loss_unconstrained": stats["loss_data_before"]}
        self.loss_fn.losses.append(entry)

    def _log(self, epoch: int, stats: dict):
        is_opt = isinstance(self.loss_fn, SSLLossCore)
        log_this_epoch = (is_opt and (epoch + 1) % 1 == 0) or (epoch + 1) % 100 == 0
        if not log_this_epoch:
            return
        if isinstance(self.model, ConstrainedModel):
            msg = (
                f"Epoch {epoch + 1}/{self.config.epochs}, "
                f"Loss: {stats['loss']:.2e}, "
                f"LossDispl: {stats['loss_displacement']:.2e}, "
                f"ProjectionIter: {stats['proj_iter']:.1f}, "
                f"EqMeanResidual: {stats['constraint_avg']:.2e}, "
                f"EqMaxResidual: {stats['constraint_max']:.2e}"
            )
        else:
            msg = (
                f"Epoch {epoch + 1}/{self.config.epochs}, "
                f"Loss: {stats['loss']:.2e}"
            )
        if is_opt:
            if isinstance(self.model, ConstrainedModel):
                msg += f", ObjValueOpt: {stats['obj_opt']:.2e}, ObjValuePred: {stats['obj_pred']:.2e}"
            else:
                msg += f", Data loss: {stats['loss_data_before']:.2e}"
        print(msg)
