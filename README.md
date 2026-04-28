# ENFORCE

![ENFORCE graphical abstract](docs/ENFORCE_graphical_abstract.png)

[![arXiv](https://img.shields.io/badge/arXiv-2502.06774-b31b1b.svg)](https://arxiv.org/abs/2502.06774)

**Nonlinear Constrained Learning with Adaptive-depth Neural Projection.**

ENFORCE combines a neural network backbone with an **AdaNP** (Adaptive-depth Neural Projection) module to drive predictions toward feasibility with respect to nonlinear equality and inequality constraints. At each forward pass, AdaNP iteratively applies a linearize-and-project correction - an SQP-inspired Gauss-Newton step - until the constraint residual falls below a prescribed tolerance ε.

For constraints that are **affine in the output** `y`, a single NP step achieves exact feasibility. For general nonlinear constraints, ε-feasibility is obtained locally: under standard regularity conditions (LICQ, C² smoothness) and when the backbone prediction is sufficiently close to the constraint manifold, AdaNP reduces the residual `‖c(x,ỹ)‖` below ε with a linear convergence rate. The model is trained with standard unconstrained optimization (Adam), not constrained solvers.

Inequality constraints `g(x,y) ≤ 0` are handled via a Fischer-Burmeister (FB) reformulation that converts them into equalities in an extended output space `[y, λ]`, so the same AdaNP projection applies without modification.

## Reference

If you use ENFORCE in your work, please cite the [paper](https://arxiv.org/abs/2502.06774):

```bibtex
@Article{Lastrucci2025_ENFORCENonlinearConstrained,
  author    = {Lastrucci, Giacomo and Schweidtmann, Artur M.},
  journal   = {arXiv preprint arXiv:2502.06774},
  title     = {ENFORCE: Nonlinear Constrained Learning with Adaptive-depth Neural Projection},
  year      = {2025},
  copyright = {arXiv.org perpetual, non-exclusive license},
  doi       = {https://doi.org/10.48550/arXiv.2502.06774},
  keywords  = {Machine Learning (cs.LG), FOS: Computer and information sciences},
  publisher = {arXiv},
}
```

## Installation

### pip

```
pip install enforce-nn
```

PyTorch must be installed separately for your hardware (CPU or CUDA):

```
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA - see https://pytorch.org/get-started/locally/ for the right command
```

### uv

```
uv add enforce-nn
```

By default this resolves PyTorch from the CPU index (configured in `pyproject.toml`). For CUDA, install PyTorch manually following the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and override the source entry.

### From source

```
git clone https://github.com/giacomolastrucci/ENFORCE
cd ENFORCE
pip install -e .
# or
uv sync
```

## Quick start

### Supervised - nonlinear equality constraint

Fit `x → (y₁, y₂)` subject to the nonlinear constraint `(0.5 y₁)² + x² + y₂ = 0`:

```python
import torch
import numpy as np
from enforce import ENFORCEConfig, ENFORCE
from enforce.engines.train import Trainer, TrainingConfig
from enforce.engines.evaluate import Evaluator, EvaluationConfig
from enforce.data.data_utils import generate_data, scale_data

# 1. Data
x_train, y_train = generate_data([...], n=500)
x_test,  y_test  = generate_data([...], n=200)
x_tr_s, y_tr_s, x_te_s, y_te_s, sp = scale_data(x_train, y_train, x_test, y_test)

# 2. Constraint  c(x, y) -> [BS, NC]  (operates on unscaled x and y)
def my_constraint(x, y):
    return ((0.5 * y[:, 0])**2 + x[:, 0]**2 + y[:, 1]).unsqueeze(1)

# 3. Build
scaling_input  = (torch.tensor(sp["input_mean"]),  torch.tensor(sp["input_std"]))
scaling_output = (torch.tensor(sp["output_mean"]), torch.tensor(sp["output_std"]))
cfg   = ENFORCEConfig(input_neurons=1, output_neurons=2, hidden_neurons=64,
                      hidden_layers=1, training_tolerance=1e-4,
                      inference_tolerance=1e-6, max_it=100,
                      supervised=True, weight_loss_displacement=0.5)
model = ENFORCE(scaling_input=scaling_input, scaling_output=scaling_output,
                c=my_constraint, config=cfg, constrained=True, weighting_option=1)

# 4. Train / evaluate  (do NOT wrap in torch.no_grad() - AdaNP needs autograd)
x_tr_t = torch.tensor(x_tr_s, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr_s, dtype=torch.float32)
x_te_t = torch.tensor(x_te_s, dtype=torch.float32)
y_te_t = torch.tensor(y_te_s, dtype=torch.float32)

model  = Trainer(model, TrainingConfig(epochs=2000)).fit(x_tr_t, y_tr_t)
result = Evaluator(model, EvaluationConfig()).evaluate(x_te_t, y_te_t, sp)

preds = result.predictions  # shape [N, 2], already unscaled
```

### Self-supervised - parametric optimization with inequality

For each `x ∈ [2, 4]`, minimize `‖y‖²` subject to `y₁² + y₂ = x` (equality) and `y₁ ≥ 0` (inequality via FB):

```python
import torch, torch.nn as nn, numpy as np
from enforce import ENFORCEConfig, ENFORCE
from enforce.fb_inequality_constraints import FischerBurmeisterReformulation
from enforce.engines.train import Trainer, TrainingConfig
from enforce.engines.evaluate import Evaluator, EvaluationConfig
from enforce.data.data_utils import scale_data

# 1. Constraints
def parabola(x, y):     return (y[:, 0]**2 + y[:, 1] - x[:, 0]).unsqueeze(1)
def g_nonneg(x, y):     return -y[:, 0]   # y1 >= 0  =>  g = -y1 <= 0

fb = FischerBurmeisterReformulation(n_original_outputs=2, inequalities=[g_nonneg])

def c_full(x, y_ext):   # NC=2 <= NO=3 ✓
    y = y_ext[:, :2]
    return torch.cat([parabola(x, y), fb(x, y_ext)], dim=1)

# 2. SSL objective - minimize ||y||²
class MinNorm(nn.Module):
    def forward(self, x, y_ext):
        return torch.mean(torch.sum(y_ext[:, :2]**2, dim=1))

# 3. Dummy labels (no targets needed in self-supervised mode)
N = 2000
x_train = np.random.uniform(2.0, 4.0, (N, 1)).astype(np.float32)
y_dummy = fb.extend_outputs(np.zeros((N, 2), dtype=np.float32))
x_tr_s, y_tr_s, _, _, sp = scale_data(x_train, y_dummy, x_train, y_dummy)

scaling_input  = (torch.tensor(sp["input_mean"]),  torch.tensor(sp["input_std"]))
scaling_output = (torch.tensor(sp["output_mean"]), torch.tensor(sp["output_std"]))

# 4. Build - output_neurons=fb.no (network predicts y only; λ appended in forward())
cfg   = ENFORCEConfig(input_neurons=1, output_neurons=fb.no, hidden_neurons=64,
                      hidden_layers=2, training_tolerance=1e-4,
                      inference_tolerance=1e-6, max_it=100,
                      supervised=False, weight_loss_displacement=0.5)
model = ENFORCE(scaling_input=scaling_input, scaling_output=scaling_output,
                c=c_full, fb=fb, ssl_loss=MinNorm(), config=cfg,
                constrained=True, weighting_option=1)

# 5. Train / evaluate
x_tr_t = torch.tensor(x_tr_s, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr_s, dtype=torch.float32)
model  = Trainer(model, TrainingConfig(epochs=2000, n_original_outputs=fb.no)).fit(x_tr_t, y_tr_t)

result = Evaluator(model, EvaluationConfig(
    n_original_outputs=fb.no, inequalities=fb.inequalities
)).evaluate(x_tr_t, y_tr_t, sp)
preds  = fb.extract_outputs(result.predictions)  # [N, 2] - y1, y2 only
```

## Tutorials

Step-by-step notebooks in the `notebooks/tutorials/` folder:

| Notebook | Topic |
|---|---|
| `01_equality_constraints.ipynb` | Supervised fitting with a nonlinear equality constraint (unit circle) |
| `02_inequality_constraints.ipynb` | Supervised fitting with inequality bounds via Fischer-Burmeister |
| `03_parametric_optimization.ipynb` | Self-supervised parametric optimization - mixed equality + inequality, MLP comparison |

## How it works

ENFORCE appends an **AdaNP** module to any backbone network. Each NP layer solves a linearized QP: given the backbone output `ŷ` and the constraint Jacobian `B = ∂c/∂y|_{x,ŷ}`, the closed-form correction step is

```
ỹ = (I − Bᵀ(BBᵀ)⁻¹B) ŷ + Bᵀ(BBᵀ)⁻¹v,   with  v = Bŷ − c(x, ŷ)
```

This is the solution to the locally linearized projection problem and corresponds to a Gauss-Newton SQP step (without Hessian of constraints). AdaNP stacks NP layers adaptively until `‖c(x, ỹ)‖ < ε` or `max_it` is reached, relinearizing at each iterate. The Jacobian `B` is computed via automatic differentiation through the constraint function `c` only (not the backbone), so the per-step cost is `O(N_C³)`.

**Scope of the ε-feasibility claim.** For constraints affine in `y`, a single NP step achieves exact feasibility. For nonlinear constraints, convergence is local: it requires the backbone prediction to lie in a neighborhood of the constraint manifold where LICQ holds and the linearization is accurate. If the backbone is far from feasibility or the Jacobian is ill-conditioned, residuals may remain above ε within the depth cap.

**Inequality constraints** `g(x,y) ≤ 0` are reformulated using the Fischer-Burmeister function

```
φ_FB(λᵢ, −gᵢ) = √(λᵢ² + gᵢ² + ε_FB) − λᵢ + gᵢ = 0
```

which encodes the KKT complementarity conditions as an equality. AdaNP then operates in the extended space `[y, λ]` without any modification to the core algorithm.

**Training.** ENFORCE is trained with standard Adam on a loss `ℓ = ℓ_task + λ_D ‖ŷ − ỹ‖² + λ_C ‖c(x,ỹ)‖`, where the displacement penalty `λ_D ‖ŷ − ỹ‖²` encourages the backbone to produce predictions already close to the constraints manifold, reducing the depth needed at inference. An adaptive activation heuristic (inspired by trust-region methods) enables AdaNP only when the projection improves a task-specific loss measure, providing an automatic warm-up phase before constrained learning begins.

## Reproducing paper results

The original benchmark datasets can be downloaded from [here](https://surfdrive.surf.nl/s/wxH67jTWfAbqTH5) and placed in `data/raw`. The original benchmark training results can be downloaded from [here](https://surfdrive.surf.nl/s/KN5W6fBrWL8Ftry) (7.6 GB).

All benchmarks are run through `scripts/run_benchmark.py`. Select the problem by setting `PROBLEM` in `src/benchmark_problems/config_benchmarking.py`, then run from the repo root:

```
python scripts/run_benchmark.py
```

### Function fitting (equality constraint)

```python
# src/benchmark_problems/config_benchmarking.py
PROBLEM = "function_fitting"
MODEL   = "BOTH"   # trains ENFORCE and MLP baseline
```

To reproduce the hyperparameter study (sweep over `λ_D` and `ε_T`):

```
bash scripts/run_hyperparameter_study.sh
```

### Function fitting with inequality constraints

```python
PROBLEM = "sin_ineq"
MODEL   = "BOTH"
```

### Engineering case studies

```python
PROBLEM = "extraction_column"   # extractive distillation surrogate
# or
PROBLEM = "pooling"             # pooling problem (equality + inequality)
MODEL   = "BOTH"
```

Both require the datasets from [Iftakher et al.](https://github.com/SOULS-TAMU/kkt-hardnet). Download it from [here](https://surfdrive.surf.nl/s/wxH67jTWfAbqTH5). Place the CSV files under `data/raw/<problem>/` as expected by the data paths in `config_benchmarking.py`.

### Parametric optimization benchmarks

```python
PROBLEM = "nonconvex_linear"     # nonconvex objective, linear equality constraints
# or
PROBLEM = "nonconvex_nonlinear"  # nonconvex objective, nonlinear equality constraints
MODEL   = "ENFORCE"
```

These problems require pre-generated data files. [Download](https://surfdrive.surf.nl/s/wxH67jTWfAbqTH5) them and place them under `data/raw/<problem>/` following the filenames in `config_benchmarking.py`. Data can be generated from the DC3 [repository](https://github.com/locuslab/DC3).

### General settings

Key flags in `src/benchmark_problems/config_benchmarking.py`:

| Flag | Default | Effect |
|---|---|---|
| `MODEL` | `"BOTH"` | `"ENFORCE"`, `"MLP"`, or `"BOTH"` |
| `N` | `5` | number of independent runs |
| `PLOT` | `True` | save result figures |
| `SAVE` | `True` | save model weights and metrics |
| `FIX_SEED` | `False` | fix random seed across runs |

## Acknowledgements

This research is supported by Shell Global Solutions International B.V., for which we express sincere gratitude.

## License

MIT - see `LICENSE`.
