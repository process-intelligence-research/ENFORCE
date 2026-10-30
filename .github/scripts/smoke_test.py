"""Smoke test of the installed package: import and one constrained prediction. Run from outside the checkout."""

import torch

import enforce
from enforce import ENFORCE, Constraints, ENFORCELoss, Scaler
from enforce.models import ENFORCEConstrainedMLP, MLPConfig

assert "site-packages" in enforce.__file__, enforce.__file__  # the installed package, not the checkout
scaler = Scaler((torch.zeros(1), torch.ones(1)), (torch.zeros(2), torch.ones(2)))
constraints = Constraints(c=[lambda x, y: y[:, 0] ** 2 + y[:, 1] - x[:, 0]])
layer = ENFORCE(scaler, constraints, ENFORCELoss(scaler, constraints))
model = ENFORCEConstrainedMLP(MLPConfig(input_neurons=1, hidden_neurons=16, output_neurons=2), layer).eval()
x = torch.rand(8, 1)
with torch.no_grad():
    ytilde, _, _ = model.predict(x)
residual = constraints(x, ytilde).abs().max().item()
assert ytilde.shape == (8, 2) and torch.isfinite(ytilde).all() and residual < 1e-5, residual
print(f"enforce {enforce.__version__}, torch {torch.__version__}: max residual {residual:.1e}")
