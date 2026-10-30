# AGENTS.md

Guidance for coding and research agents using or changing ENFORCE (`enforce-nn`, version 2.0). It describes the
current behaviour; check the README and the docstrings for details.

## What ENFORCE does

ENFORCE appends a projection layer (AdaNP) to a neural network so that its predictions satisfy equality constraints
`c(x, y) = 0` and inequality constraints `g(x, y) <= 0` up to a tolerance. Each projection step is a Gauss-Newton
step on the linearised constraints; the steps repeat until the residual is below a tolerance or until `max_it`
steps.

Limitations to keep in mind:

- Constraints affine in `y` are satisfied in one step. Nonlinear constraints converge only locally: the network's
  prediction must be close enough to the feasible set, and the constraint Jacobian must have full row rank.
- **Feasibility is not guaranteed; check it** (see "Checking feasibility"). When `max_it` is reached, the layer
  prints `Max projection iteration reached` and returns the last iterate, which may be infeasible.
- There must be fewer equality constraints than outputs (`ValueError` otherwise; a warning when equal).

## Setup

Published package (install the PyTorch build for your hardware first, see the README):

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install enforce-nn            # add [examples] for matplotlib and pandas
```

Source checkout for development (CPU PyTorch from the index set in `pyproject.toml`):

```
git clone https://github.com/process-intelligence-research/ENFORCE
cd ENFORCE
uv sync
```

## Repository map

| Path | Contents |
|---|---|
| `enforce/core/` | `Scaler` (input/output normalisation), `Constraint` and `Constraints` (residuals, Jacobians, FB reformulation) |
| `enforce/layers/enforce.py` | `ENFORCE` projection layer and `ENFORCEConfig` (tolerances, `max_it`, weighting, seeds) |
| `enforce/losses/` | `ENFORCELoss` (supervised), `ENFORCESSLLoss` (self-supervised), `MSELoss`, soft-constrained baselines |
| `enforce/models/` | `MLP`, `ConstrainedModel`, `ENFORCEConstrainedModel` (any `nn.Module` backbone), `ENFORCEConstrainedMLP` |
| `enforce/engines/` | `Trainer` / `TrainingConfig`, `Evaluator` / `EvaluationConfig` / `EvalResult` |
| `enforce/data/` | `scale_data` |
| `notebooks/tutorials/` | Runnable tutorials: equality, inequality, self-supervised parametric optimization |
| `benchmark_problems/`, `scripts/run_benchmark.py` | Paper benchmarks (need external datasets) |
| `tests/` | Test suite; `test_readme.py` runs the README quick start |

## Using the API

Start from the README quick start (tested in CI) or the tutorials rather than writing from scratch. The contract:

- **Data** are scaled before training: `scale_data(...)` returns scaled arrays and the statistics for
  `Scaler((input_mean, input_std), (output_mean, output_std))`. Models take and return *scaled* tensors.
- **Constraints** receive *unscaled* `x [BS, ...]` and `y [BS, NO]`. A plain callable returns one constraint `[BS]`;
  `Constraint(fn, size, jac=None)` returns `[BS, size]`, optionally with an analytical Jacobian `[BS, size, NO]`.
  Items in `Constraints(c=[...])` are equalities, in `g=[...]` inequalities `g <= 0`.
- **Inequalities** extend the outputs with one Fischer-Burmeister multiplier per inequality: labels are extended with
  `constraints.extend_outputs(y)`, the scaler covers `NO + NG` columns, and the backbone predicts `scaler.no -
  constraints.ng` outputs. `Evaluator` results and `constraints.extract_outputs` drop the multipliers.
- **Backbone**: any `nn.Module` whose input is one batch-first tensor and whose output is `[BS, NO - NG]`
  (e.g. Transformers, GNNs on a fixed graph; see `tests/test_custom_backbones.py`).
- **Training mode vs inference**: in `model.train()` the projection iterates until the *mean* `|c|` over the batch
  is below `training_tolerance`; with `ada_np_auto_activation` (supervised losses) it is skipped while it does not
  improve the data loss. In `model.eval()` it iterates until the *maximum* `|c|` is below `inference_tolerance`, so
  call `model.eval()` before predicting. `model.predict(x)` returns `(ytilde, yhat, iterations)` in scaled units; `Evaluator.evaluate` calls `eval()`,
  runs in batches and returns unscaled predictions with metrics. Prediction works under `torch.no_grad()` (gradients
  for the Jacobian are enabled internally).

## Checking feasibility

Do not assume a prediction is feasible. With `Evaluator`, read `result.metrics`:

- `residual_max`, `residual_avg`: equality residuals `|c|` (unscaled);
- `ineq_g<i>_max_violation`, `ineq_g<i>_infeasible_pct`: violations of the original inequalities `g <= 0`;
- `projection iterations`: if it equals `max_it`, the projection stopped at the cap.

Directly:

```python
model.eval()
ytilde, _, iterations = model.predict(x_scaled)
x_u, y_u = scaler.unscale(x_scaled, ytilde)
residual = constraints(x_u, y_u).abs().max()  # all constraints, incl. FB residuals of inequalities
```

The termination test uses the same quantity: the maximum absolute residual over the batch and all constraints,
against `inference_tolerance`. It is a batch-level test: one hard sample keeps the whole batch iterating.

## Reproducibility, devices and saving

- Seeds: `TrainingConfig.random_seed` (training, shuffling) and `MLPConfig.random_seed` (initial weights). Seed the
  data generation yourself.
- Devices: `Scaler`, `Constraints` and `ENFORCE` use CUDA if available, chosen at construction. Their tensors are not
  PyTorch buffers and do not follow `model.to(...)`; keep the model and data on that same device.
- Saving: there is no save/load API yet. Save `model.state_dict()` together with the scaling statistics and the
  configs, and rebuild the objects before `load_state_dict` (as `scripts/run_benchmark.py` does with
  `scaling.json`). `torch.save(model)` fails if a constraint is a lambda.

## Validating changes

```
uv run pytest                       # fast tests, about a minute; what CI runs
uv run ruff check <changed files>   # the existing code is not clean everywhere yet
```

Paper reproduction is separate and slow (minutes to hours per problem). It needs the datasets linked in the README
under `data/raw/` and is not run in CI:

```
uv run pytest -m slow tests/test_benchmark_reproducibility.py
```

Changes to the projection, losses, constraints or training loop can change numerical results; run the relevant
benchmarks and report differences. See `CONTRIBUTING.md` for the pull request workflow.
