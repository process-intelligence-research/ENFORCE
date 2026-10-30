"""ENFORCE on backbones other than the built-in MLP.

The package claims that any ``nn.Module`` can be the backbone. These tests train and evaluate ENFORCE with
backbones whose inputs are not flat vectors, through the public API only (``ENFORCEConstrainedModel``,
``Trainer``, ``Evaluator``):

- a Transformer encoder on sequences ``x [BS, L, F]``;
- a graph neural network on a fixed graph, node features ``x [BS, N, F]``, with an inequality constraint.

Requirements on the backbone that these tests rely on: inputs are one tensor whose first dimension is the batch
(``Trainer`` and ``Evaluator`` batch by slicing it), the input scaling broadcasts over its last dimension, and the
backbone returns flat outputs ``[BS, NO - NG]``. Constraints receive the unscaled input tensor as given.
"""

import pytest
import torch
import torch.nn as nn

from enforce import ENFORCE, Constraint, Constraints, ENFORCEConfig, ENFORCEConstrainedModel, ENFORCELoss, Scaler
from enforce.engines import EvaluationConfig, Evaluator, Trainer, TrainingConfig

TOL = 1e-5  # inference tolerance is 1e-6; float32 rounding on the residual


def _fit_and_evaluate(backbone, constraints, x, y, epochs=40):
    """Build ENFORCE around ``backbone``, train it and evaluate on held-out data."""
    n_train = int(0.8 * len(x))
    scaler = Scaler((torch.zeros(x.shape[-1]), torch.ones(x.shape[-1])), (y[:n_train].mean(0), y[:n_train].std(0)))
    _, y_s = scaler.scale(x, y)
    loss = ENFORCELoss(scaler, constraints)
    layer = ENFORCE(scaler, constraints, loss, ENFORCEConfig(training_tolerance=1e-4, inference_tolerance=1e-6))
    model = ENFORCEConstrainedModel(backbone, layer)
    initial = [p.detach().clone() for p in backbone.parameters()]

    trainer = Trainer(model, loss, scaler, constraints, TrainingConfig(batch_size=32, epochs=epochs))
    trainer.fit(x[:n_train], y_s[:n_train])
    result = Evaluator(model, loss, scaler, constraints, EvaluationConfig()).evaluate(x[n_train:], y_s[n_train:])

    changed = [not torch.equal(p0, p) for p0, p in zip(initial, backbone.parameters(), strict=True)]
    return model, result, changed


class SequenceTransformer(nn.Module):
    """Transformer encoder over a sequence, mean-pooled to ``n_out`` outputs."""

    def __init__(self, n_features, n_out, d_model=16):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead=2, dim_feedforward=32, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x):  # [BS, L, F] -> [BS, n_out]
        return self.head(self.encoder(self.embed(x)).mean(dim=1))


class GraphNetwork(nn.Module):
    """Message passing on a fixed graph (GraphSAGE-style: own state plus mean over neighbours), one output per
    node."""

    def __init__(self, adjacency, n_features, hidden=16):
        super().__init__()
        self.register_buffer("a_mean", adjacency / adjacency.sum(1, keepdim=True))
        sizes = [n_features, hidden, hidden]
        self.self_layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])])
        self.neighbour_layers = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in zip(sizes, sizes[1:])])
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):  # [BS, N, F] -> [BS, N]
        h = x
        for own, neighbours in zip(self.self_layers, self.neighbour_layers, strict=True):
            h = torch.relu(own(h) + neighbours(self.a_mean @ h))
        return self.out(h).squeeze(-1)


def test_transformer_backbone_on_sequences():
    """Sequences ``x [BS, 6, 2]``; outputs: the two feature means and a nonlinear combination of them."""
    torch.manual_seed(0)
    x = torch.rand(400, 6, 2)
    m = x.mean(dim=1)
    y = torch.stack([m[:, 0], m[:, 1], m[:, 0] ** 2 + m[:, 1]], dim=1)

    def mean_constraint(x, y):  # linear, depends on the sequence input
        return y[:, 0] - x[:, :, 0].mean(dim=1)

    def nonlinear_constraint(x, y):
        return y[:, 2] - y[:, 0] ** 2 - y[:, 1]

    constraints = Constraints(c=[mean_constraint, nonlinear_constraint])
    model, result, changed = _fit_and_evaluate(SequenceTransformer(2, 3), constraints, x, y)

    assert all(changed), "every backbone parameter is trained through the projection"
    assert result.predictions.shape == (80, 3)
    assert result.metrics["residual_max"] < TOL
    assert result.metrics["r2_y1"] > 0.9  # learned, not just projected


def test_graph_network_backbone_with_inequality():
    """Ring of 5 nodes; node output ``x_i + mean of neighbours``, total conserved, node 0 nonnegative."""
    torch.manual_seed(0)
    n = 5
    adjacency = torch.zeros(n, n)
    for i in range(n):
        adjacency[i, (i + 1) % n] = adjacency[(i + 1) % n, i] = 1.0
    x = torch.rand(400, n, 2)
    y = x[..., 0] + 0.5 * x[..., 0] @ adjacency  # adjacency is symmetric with degree 2
    y = torch.cat([y, torch.zeros(len(y), 1)], dim=1)  # zero FB multiplier column for the inequality

    conservation = Constraint(lambda x, y: y.sum(dim=1) - 2.0 * x[..., 0].sum(dim=1), size=1, name="conservation")
    constraints = Constraints(c=[conservation], g=[lambda x, y: -y[:, 0]])
    model, result, changed = _fit_and_evaluate(GraphNetwork(adjacency, 2), constraints, x, y)

    assert all(changed)
    assert result.predictions.shape == (80, n)  # multiplier column removed
    assert result.metrics["residual_max"] < TOL
    assert result.metrics["ineq_g1_max_violation"] < TOL
    assert all(result.metrics[f"r2_y{i + 1}"] > 0.9 for i in range(n))


def test_custom_backbone_predicts_under_no_grad():
    """Inference of a trained model inside ``torch.no_grad()``, as for any PyTorch model."""
    torch.manual_seed(0)
    x = torch.rand(64, 4, 2)
    constraints = Constraints(c=[lambda x, y: y.sum(dim=1) - x.sum(dim=(1, 2))])
    scaler = Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(3), torch.ones(3)))
    loss = ENFORCELoss(scaler, constraints)
    model = ENFORCEConstrainedModel(SequenceTransformer(2, 3), ENFORCE(scaler, constraints, loss))
    model.eval()
    with torch.no_grad():
        ytilde, yhat, _ = model.predict(x)
    assert ytilde.shape == yhat.shape == (64, 3)
    assert constraints(x, ytilde).abs().max() < TOL


@pytest.mark.parametrize("backbone", [SequenceTransformer(2, 3), nn.Sequential(nn.Flatten(), nn.Linear(8, 3))])
def test_any_module_with_flat_outputs_is_accepted(backbone):
    scaler = Scaler((torch.zeros(2), torch.ones(2)), (torch.zeros(3), torch.ones(3)))
    constraints = Constraints(c=[lambda x, y: y[:, 0] + y[:, 1] - 1.0])
    loss = ENFORCELoss(scaler, constraints)
    model = ENFORCEConstrainedModel(backbone, ENFORCE(scaler, constraints, loss))
    ytilde, _ = model(torch.rand(10, 4, 2))
    assert ytilde.shape == (10, 3)
    assert (ytilde[:, 0] + ytilde[:, 1] - 1.0).abs().max() < TOL
