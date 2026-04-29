"""Benchmark runner for ENFORCE and MLP baselines.

Configure via configs/config_benchmarking.py, then run:
    python run_benchmark.py
"""

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.autograd.profiler as profiler

from benchmark_problems.config_benchmarking import (
    ADA_NP_AUTO_ACTIVATION,
    BATCH_SIZE,
    DATA_TEST,
    DATA_TRAINING,
    EPOCH_START_HARD_CONSTRAINED,
    EPOCHS,
    FIX_SEED,
    HIDDEN_LAYERS,
    HIDDEN_NEURONS,
    IFT_BACKWARD,
    INFERENCE_TOLERANCE,
    INPUT_NEURONS,
    LEARNING_RATE,
    LEFT_LIMIT,
    MAX_IT,
    MODEL,
    OUTPUT_NEURONS,
    PARAMS_PATH,
    PLOT,
    PROBLEM,
    PROJ_WEIGHTING_OPTION,
    REGULARISE_GRAM,
    RIGHT_LIMIT,
    SAVE,
    SOFT_CONSTRAINED,
    SUPERVISED,
    TRAINING_DIR,
    TRAINING_TOLERANCE,
    VERBOSE,
    WEIGHT_LOSS_DISPLACEMENT,
    WEIGHT_LOSS_SOFT,
    N,
)
from benchmark_problems.engineering_problems.extraction_column import constraints_column
from benchmark_problems.engineering_problems.pooling import make_pooling_constraints
from benchmark_problems.function_fitting.equality.constraints import get_constraints
from benchmark_problems.function_fitting.equality.functions import get_functions
from benchmark_problems.function_fitting.inequality.sin_ineq import make_sin_ineq_constraints
from benchmark_problems.parametric_optimization.opt_problem import NonconvexProgram, NonlinearProgram
from benchmark_problems.parametric_optimization.ssl_loss import SSLConfig, SSLLoss
from enforce.data.data_utils import generate_data, scale_data
from enforce.data.dataloaders import Dataloader
from enforce.core.config import ENFORCEConfig
from enforce.core.model import ENFORCE
from enforce.engines.evaluate import EvaluationConfig, Evaluator
from enforce.engines.train import Trainer, TrainingConfig
from benchmark_problems.visualization.plot_benchmarking import plot_all_results, plot_sin_ineq_results

# ── Module-level seeding (must execute at import time for reproducibility) ─────

_DATA_SEED = 41
np.random.seed(_DATA_SEED)
torch.manual_seed(_DATA_SEED)
random.seed(_DATA_SEED)

seeds: list[int] = [random.randint(0, 100000) for _ in range(N)] if not FIX_SEED else [_DATA_SEED] * N

PROFILER = False  # set True to print per-batch profiling tables


# ── Data containers ───────────────────────────────────────────────────────────


@dataclass
class ProblemData:
    """All data and problem-specific objects for one benchmark."""

    train_inputs: np.ndarray
    train_outputs: np.ndarray  # FB-extended when fb is set
    test_inputs: np.ndarray
    test_outputs: np.ndarray  # FB-extended when fb is set
    train_outputs_orig: np.ndarray  # before FB extension (== train_outputs if no fb)
    test_outputs_orig: np.ndarray  # before FB extension (== test_outputs if no fb)
    c: object  # constraint callable c(x, y) → residuals
    fb: object | None  # FischerBurmeisterReformulation, or None
    jac: object | None  # analytical Jacobian
    ssl_loss: object | None  # SSLLoss module
    eps_chol: float = 1e-8


class PreparedTensors(NamedTuple):
    """Scaled data as torch tensors, ready for training."""

    train_in: torch.Tensor
    train_out: torch.Tensor
    test_in: torch.Tensor
    test_out: torch.Tensor
    scaling_input: tuple  # (mean_tensor, std_tensor)
    scaling_output: tuple  # (mean_tensor, std_tensor)
    scaling_params: dict  # raw numpy scaling stats for Evaluator


# ── Per-problem data loaders ──────────────────────────────────────────────────


def _load_function_fitting_data() -> ProblemData:
    functions = get_functions()
    train_in, train_out = generate_data(functions, DATA_TRAINING, LEFT_LIMIT, RIGHT_LIMIT)
    test_in, test_out = generate_data(functions, DATA_TEST, LEFT_LIMIT, RIGHT_LIMIT)
    return ProblemData(
        train_inputs=train_in,
        train_outputs=train_out,
        test_inputs=test_in,
        test_outputs=test_out,
        train_outputs_orig=train_out,
        test_outputs_orig=test_out,
        c=get_constraints(),
        fb=None,
        jac=None,
        ssl_loss=None,
    )


def _load_extraction_column_data() -> ProblemData:
    from benchmark_problems.config_benchmarking import (
        INPUT_DATA_PATH_TEST,
        INPUT_DATA_PATH_TRAIN,
        OUTPUT_DATA_PATH_TEST,
        OUTPUT_DATA_PATH_TRAIN,
    )

    train_in = pd.read_csv(INPUT_DATA_PATH_TRAIN, header=0).to_numpy()
    train_out = pd.read_csv(OUTPUT_DATA_PATH_TRAIN, header=0).to_numpy()
    test_in = pd.read_csv(INPUT_DATA_PATH_TEST, header=0).to_numpy()
    test_out = pd.read_csv(OUTPUT_DATA_PATH_TEST, header=0).to_numpy()
    return ProblemData(
        train_inputs=train_in,
        train_outputs=train_out,
        test_inputs=test_in,
        test_outputs=test_out,
        train_outputs_orig=train_out,
        test_outputs_orig=test_out,
        c=constraints_column,
        fb=None,
        jac=None,
        ssl_loss=None,
    )


def _load_pooling_data() -> ProblemData:
    from benchmark_problems.config_benchmarking import (
        INPUT_DATA_PATH_TEST,
        INPUT_DATA_PATH_TRAIN,
        OUTPUT_DATA_PATH_TEST,
        OUTPUT_DATA_PATH_TRAIN,
        POOLING_EPS_CHOL,
    )

    train_in = pd.read_csv(INPUT_DATA_PATH_TRAIN, header=0).to_numpy()
    train_out = pd.read_csv(OUTPUT_DATA_PATH_TRAIN, header=0).to_numpy()
    test_in = pd.read_csv(INPUT_DATA_PATH_TEST, header=0).to_numpy()
    test_out = pd.read_csv(OUTPUT_DATA_PATH_TEST, header=0).to_numpy()
    fb, c = make_pooling_constraints()
    return ProblemData(
        train_inputs=train_in,
        train_outputs=fb.extend_outputs(train_out),
        test_inputs=test_in,
        test_outputs=fb.extend_outputs(test_out),
        train_outputs_orig=train_out,
        test_outputs_orig=test_out,
        c=c,
        fb=fb,
        jac=None,
        ssl_loss=None,
        eps_chol=POOLING_EPS_CHOL,
    )


def _load_sin_ineq_data() -> ProblemData:
    from benchmark_problems.config_benchmarking import (
        SIN_INEQ_LEFT,
        SIN_INEQ_N_TEST,
        SIN_INEQ_N_TRAIN,
        SIN_INEQ_NOISE_BIAS,
        SIN_INEQ_NOISE_STD,
        SIN_INEQ_RIGHT,
    )

    # True function f(x) = (1 + x²/(3π²))·sin(x).
    # Asymmetric noise (bias matches sign of f) pushes training labels past the
    # inequality envelope, giving the MLP a reason to violate it.
    # Test outputs are noiseless (ground-truth function values).
    train_in = np.random.uniform(SIN_INEQ_LEFT, SIN_INEQ_RIGHT, (SIN_INEQ_N_TRAIN, 1)).astype(np.float32)
    f_train = (1 + train_in**2 / (3 * np.pi**2)) * np.sin(train_in)
    noise = (
        np.random.normal(0.0, SIN_INEQ_NOISE_STD, (SIN_INEQ_N_TRAIN, 1)) + SIN_INEQ_NOISE_BIAS * np.sign(f_train)
    ).astype(np.float32)
    train_out = (f_train + noise).astype(np.float32)
    test_in = np.random.uniform(SIN_INEQ_LEFT, SIN_INEQ_RIGHT, (SIN_INEQ_N_TEST, 1)).astype(np.float32)
    test_out = ((1 + test_in**2 / (3 * np.pi**2)) * np.sin(test_in)).astype(np.float32)
    fb, c = make_sin_ineq_constraints()
    return ProblemData(
        train_inputs=train_in,
        train_outputs=fb.extend_outputs(train_out),
        test_inputs=test_in,
        test_outputs=fb.extend_outputs(test_out),
        train_outputs_orig=train_out,
        test_outputs_orig=test_out,
        c=c,
        fb=fb,
        jac=None,
        ssl_loss=None,
    )


def _load_nonconvex_data(device: str) -> ProblemData:
    from benchmark_problems.config_benchmarking import INPUT_DATA_PATH, OUTPUT_DATA_PATH

    raw = Dataloader(
        input_path=INPUT_DATA_PATH,
        output_path=OUTPUT_DATA_PATH,
        t_v_t_ratio=(7.0, 2.0, 1.0),
    ).get_data()
    opt_prob = (
        NonconvexProgram(params_path=PARAMS_PATH)
        if PROBLEM == "nonconvex_linear"
        else NonlinearProgram(params_path=PARAMS_PATH)
    )
    ssl_loss = SSLLoss(
        config=SSLConfig(soft_constrained=SOFT_CONSTRAINED, weight_loss_soft=WEIGHT_LOSS_SOFT),
        opt_prob=opt_prob,
    ).to(device=device)
    train_out = raw["train_outputs"]
    test_out = raw["test_outputs"]
    return ProblemData(
        train_inputs=raw["train_inputs"],
        train_outputs=train_out,
        test_inputs=raw["test_inputs"],
        test_outputs=test_out,
        train_outputs_orig=train_out,
        test_outputs_orig=test_out,
        c=opt_prob.eq_constraints,
        fb=None,
        jac=opt_prob.jacobian,
        ssl_loss=ssl_loss,
    )


def _load_problem_data(device: str) -> ProblemData:
    loaders = {
        "function_fitting": _load_function_fitting_data,
        "extraction_column": _load_extraction_column_data,
        "pooling": _load_pooling_data,
        "sin_ineq": _load_sin_ineq_data,
        "nonconvex_linear": lambda: _load_nonconvex_data(device),
        "nonconvex_nonlinear": lambda: _load_nonconvex_data(device),
    }
    if PROBLEM not in loaders:
        raise ValueError(f"Unknown PROBLEM: {PROBLEM!r}")
    return loaders[PROBLEM]()


# ── Tensor preparation ────────────────────────────────────────────────────────


def _prepare_tensors(data: ProblemData, device: str) -> PreparedTensors:
    """Scale data and convert to float32 tensors on *device*."""
    train_in_s, train_out_s, test_in_s, test_out_s, sp = scale_data(
        data.train_inputs,
        data.train_outputs,
        data.test_inputs,
        data.test_outputs,
    )

    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    scaling_input = (to_t(sp["input_mean"]), to_t(sp["input_std"]))
    scaling_output = (to_t(sp["output_mean"]), to_t(sp["output_std"]))
    return PreparedTensors(
        train_in=to_t(train_in_s),
        train_out=to_t(train_out_s),
        test_in=to_t(test_in_s),
        test_out=to_t(test_out_s),
        scaling_input=scaling_input,
        scaling_output=scaling_output,
        scaling_params=sp,
    )


# ── Model construction ────────────────────────────────────────────────────────


def _build_enforce_config(seed: int) -> ENFORCEConfig:
    return ENFORCEConfig(
        input_neurons=INPUT_NEURONS,
        hidden_neurons=HIDDEN_NEURONS,
        output_neurons=OUTPUT_NEURONS,
        hidden_layers=HIDDEN_LAYERS,
        training_tolerance=TRAINING_TOLERANCE,
        inference_tolerance=INFERENCE_TOLERANCE,
        max_it=MAX_IT,
        epoch_start_hard_constrained=EPOCH_START_HARD_CONSTRAINED,
        ada_np_auto_activation=ADA_NP_AUTO_ACTIVATION,
        ift_backward=IFT_BACKWARD,
        regularise_gram=REGULARISE_GRAM,
        supervised=SUPERVISED,
        soft_constrained=SOFT_CONSTRAINED,
        weight_loss_displacement=WEIGHT_LOSS_DISPLACEMENT,
        weight_loss_soft=WEIGHT_LOSS_SOFT,
        verbose=VERBOSE,
        random_seed=seed,
    )


def _build_model(
    data: ProblemData,
    tensors: PreparedTensors,
    constrained: bool,
    seed: int,
    device: str,
) -> ENFORCE:
    return ENFORCE(
        scaling_input=tensors.scaling_input,
        scaling_output=tensors.scaling_output,
        c=data.c,
        config=_build_enforce_config(seed),
        fb=data.fb,
        constrained=constrained,
        weighting_option=PROJ_WEIGHTING_OPTION,
        ssl_loss=data.ssl_loss if not SUPERVISED else None,
        jac=data.jac,
        eps_chol=data.eps_chol,
    ).to(device=device)


def _build_training_config(data: ProblemData, constrained: bool, seed: int) -> TrainingConfig:
    # Unconstrained mode for supervised FB problems uses a soft inequality penalty
    # instead of hard projection.
    use_soft_penalty = data.fb is not None and SOFT_CONSTRAINED and not constrained
    return TrainingConfig(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        random_seed=seed,
        soft_inequalities=data.fb.inequalities if use_soft_penalty else None,
        soft_weight=WEIGHT_LOSS_SOFT if use_soft_penalty else 0.0,
        n_original_outputs=data.fb.no if data.fb is not None else None,
    )


# ── Training ──────────────────────────────────────────────────────────────────


def _train_model(
    model: ENFORCE,
    training_cfg: TrainingConfig,
    train_in: torch.Tensor,
    train_out: torch.Tensor,
) -> tuple[ENFORCE, float]:
    """Train *model* and return (trained model, wall-clock seconds)."""
    t0 = time.time()
    if PROFILER:
        with profiler.profile(with_stack=True, profile_memory=True) as prof:
            model = Trainer(model, training_cfg).fit(train_in, train_out)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=20))
    else:
        model = Trainer(model, training_cfg).fit(train_in, train_out)
    return model, time.time() - t0


# ── Evaluation ────────────────────────────────────────────────────────────────


def _evaluate_model(
    model: ENFORCE,
    data: ProblemData,
    tensors: PreparedTensors,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate *model*; strip FB columns from predictions if needed.

    Returns (metrics, predictions, predictions_before_projection).
    """
    eval_cfg = EvaluationConfig(
        batch_size=BATCH_SIZE,
        n_original_outputs=data.fb.no if data.fb is not None else None,
        inequalities=data.fb.inequalities if data.fb is not None else None,
    )
    result = Evaluator(model, eval_cfg).evaluate(tensors.test_in, tensors.test_out, tensors.scaling_params)
    predictions = result.predictions
    predictions_before = result.predictions_before_proj
    if data.fb is not None:
        predictions = data.fb.extract_outputs(predictions)
        predictions_before = data.fb.extract_outputs(predictions_before)
    return result.metrics, predictions, predictions_before


# ── Output directory and saving ───────────────────────────────────────────────


def _setup_output_dir(timestamp: str) -> Path | None:
    """Create the run output directory tree and write a readme. Returns None if SAVE=False."""
    if not SAVE:
        return None
    training_dir = Path(TRAINING_DIR) / PROBLEM
    if not SUPERVISED:
        suffix = {"nonconvex_linear": "nonconvex", "nonconvex_nonlinear": "nonlinear"}.get(PROBLEM, "")
        if suffix:
            training_dir = training_dir / suffix
    output_dir = training_dir / f"run_{timestamp}"
    for sub in ("losses", "test_dataset", "plots", "test_predictions"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    config_text = Path("src/benchmark_problems/config_benchmarking.py").read_text(encoding="utf-8")
    readme = (
        "Training Output Data\n\n"
        "Files:\n"
        "  <mode>_losses_run_X.csv          Per-epoch losses for run X.\n"
        "  <mode>_mean_losses.csv           Mean/std losses across runs.\n"
        "  <mode>_metrics_runs.csv          Per-run evaluation metrics.\n"
        "  <mode>_mean_metrics.csv          Mean/std metrics across runs.\n"
        "  <mode>_test_data.csv             Test inputs and true outputs.\n"
        "  <mode>_test_predictions_run_X.csv Predictions (and pre-projection) for run X.\n\n"
        "The same seeds were used for corresponding runs in both modes for fair comparison.\n\n" + config_text
    )
    (output_dir / "readme.txt").write_text(readme, encoding="utf-8")
    print(f"Output directory: {output_dir}")
    return output_dir


def _save_scaling_params(output_dir: Path, scaling_params: dict) -> None:
    converted = {
        k: float(v) if isinstance(v, np.float64) else v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in scaling_params.items()
    }
    (output_dir / "scaling.json").write_text(json.dumps(converted, indent=4))


def _save_generated_data(output_dir: Path, data: ProblemData) -> None:
    """Save generated train/test arrays to CSV (function_fitting only)."""
    for split, inputs, outputs in (
        ("train", data.train_inputs, data.train_outputs_orig),
        ("test", data.test_inputs, data.test_outputs_orig),
    ):
        df = pd.concat(
            [
                pd.DataFrame(inputs, columns=[f"x{i + 1}" for i in range(inputs.shape[1])]),
                pd.DataFrame(outputs, columns=[f"y{i + 1}" for i in range(outputs.shape[1])]),
            ],
            axis=1,
        )
        df.to_csv(output_dir / f"{split}_data.csv", index=False)


def _aggregate_losses(losses_runs: list[list[dict]], constrained: bool) -> pd.DataFrame:
    """Compute epoch-wise mean and std of losses across N runs."""

    def _stats(key: str) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array([[ep[key] for ep in run] for run in losses_runs])
        return arr.mean(axis=0), arr.std(axis=0)

    epoch_col = {"epoch": np.arange(1, EPOCHS + 1)}

    if not constrained:
        mean, std = _stats("loss_unconstrained")
        return pd.DataFrame({**epoch_col, "mean_loss_unconstrained": mean, "std_loss_unconstrained": std})

    data: dict = {**epoch_col}
    for key in (
        "loss_data_after_projection",
        "loss_data_before_projection",
        "loss_displacement",
        "projection_iterations",
    ):
        mean, std = _stats(key)
        data[f"mean_{key}"] = mean
        data[f"std_{key}"] = std
    if not SUPERVISED:
        for key in ("objective_value_optimization", "objective_value_prediction"):
            mean, std = _stats(key)
            data[f"mean_{key}"] = mean
            data[f"std_{key}"] = std
    return pd.DataFrame(data)


def _save_mode_results(
    output_dir: Path,
    mode: str,
    constrained: bool,
    losses_runs: list[list[dict]],
    metrics_runs: list[dict],
    test_inputs: np.ndarray,
    test_outputs: np.ndarray,
    predictions_list: list[np.ndarray],
    predictions_before_list: list[np.ndarray],
) -> None:
    # Per-run metrics
    df_metrics = pd.DataFrame(metrics_runs)
    df_metrics["seed"] = seeds
    df_metrics.to_csv(output_dir / f"{mode}_metrics_runs.csv", index=False)

    # Mean/std metrics
    pd.DataFrame(
        {
            "metric": df_metrics.columns,
            "mean": df_metrics.mean().values,
            "std": df_metrics.std().values,
        }
    ).to_csv(output_dir / f"{mode}_mean_metrics.csv", index=False)

    # Per-run losses
    losses_dir = output_dir / "losses"
    for i, run_losses in enumerate(losses_runs):
        df = pd.DataFrame(run_losses)
        df["epoch"] = np.arange(1, EPOCHS + 1)
        df.to_csv(losses_dir / f"{mode}_losses_run_{i + 1}.csv", index=False)

    # Aggregated losses
    _aggregate_losses(losses_runs, constrained).to_csv(output_dir / f"{mode}_mean_losses.csv", index=False)

    # Test data (identical across runs — saved once per mode)
    ni, no = test_inputs.shape[1], test_outputs.shape[1]
    df_test = pd.concat(
        [
            pd.DataFrame(test_inputs, columns=[f"x{i + 1}" for i in range(ni)]),
            pd.DataFrame(test_outputs, columns=[f"y{i + 1}_true" for i in range(no)]),
        ],
        axis=1,
    )
    df_test.to_csv(output_dir / "test_dataset" / f"{mode}_test_data.csv", index=False)

    # Predictions per run
    df_x = pd.DataFrame(test_inputs, columns=[f"x{i + 1}" for i in range(ni)])
    for i, preds in enumerate(predictions_list):
        df_pred = pd.DataFrame(preds, columns=[f"y{j + 1}_pred" for j in range(preds.shape[1])])
        parts = [df_x, df_pred]
        if constrained:
            pb = predictions_before_list[i]
            parts.append(pd.DataFrame(pb, columns=[f"y{j + 1}_pred_before" for j in range(pb.shape[1])]))
        pd.concat(parts, axis=1).to_csv(
            output_dir / "test_predictions" / f"{mode}_test_predictions_run_{i + 1}.csv",
            index=False,
        )


# ── Plotting ──────────────────────────────────────────────────────────────────


def _plot_results(
    data: ProblemData,
    losses_dict: dict,
    metrics_dict: dict,
    inputs_dict: dict,
    outputs_dict: dict,
    preds_dict: dict,
    preds_before_dict: dict,
    output_dir: Path | None,
) -> None:
    plot_fns = get_functions() if PROBLEM == "function_fitting" else range(min(OUTPUT_NEURONS, 6))
    plot_all_results(
        losses_dict,
        metrics_dict,
        inputs_dict,
        outputs_dict,
        preds_dict,
        preds_before_dict,
        output_dir,
        plot_fns,
        data.train_inputs,
        data.train_outputs_orig,
        data.test_inputs,
        data.test_outputs_orig,
        save=SAVE,
        left_limit=LEFT_LIMIT,
        right_limit=RIGHT_LIMIT,
        supervised=SUPERVISED,
    )
    if PROBLEM == "sin_ineq":
        from benchmark_problems.config_benchmarking import SIN_INEQ_LEFT, SIN_INEQ_RIGHT

        plot_sin_ineq_results(
            data.train_inputs,
            data.train_outputs_orig,
            data.test_inputs,
            data.test_outputs_orig,
            preds_dict,
            preds_before_dict,
            output_dir,
            save=SAVE,
            sin_ineq_left=SIN_INEQ_LEFT,
            sin_ineq_right=SIN_INEQ_RIGHT,
        )
    plt.show()


# ── Mode resolution ───────────────────────────────────────────────────────────


def _resolve_modes(model: str) -> list[str]:
    match model:
        case "BOTH":
            return ["constrained", "unconstrained"]
        case "ENFORCE":
            return ["constrained"]
        case "MLP":
            return ["unconstrained"]
        case _:
            raise ValueError(f"MODEL must be 'ENFORCE', 'MLP', or 'BOTH'; got {model!r}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    modes = _resolve_modes(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data = _load_problem_data(device)
    print(f"  train: {data.train_inputs.shape[0]} samples | test: {data.test_inputs.shape[0]} samples")

    tensors = _prepare_tensors(data, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _setup_output_dir(timestamp)
    if output_dir is not None:
        _save_scaling_params(output_dir, tensors.scaling_params)
        if PROBLEM == "function_fitting":
            _save_generated_data(output_dir, data)

    losses_dict: dict[str, list] = {}
    metrics_dict: dict[str, list] = {}
    inputs_dict: dict[str, list] = {}
    outputs_dict: dict[str, list] = {}
    preds_dict: dict[str, list] = {}
    preds_before_dict: dict[str, list] = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode.upper()} ---\n")
        constrained = mode == "constrained"

        run_losses, run_metrics, run_preds, run_preds_before = [], [], [], []

        for run_idx, seed in enumerate(seeds):
            print(f"Run {run_idx + 1}/{N}, Seed={seed}")

            model = _build_model(data, tensors, constrained, seed, device)
            training_cfg = _build_training_config(data, constrained, seed)
            model, training_time = _train_model(model, training_cfg, tensors.train_in, tensors.train_out)

            if output_dir is not None:
                torch.save(model, output_dir / f"model_{timestamp}_{mode}_run{run_idx}.pth")

            metrics, preds, preds_before = _evaluate_model(model, data, tensors)
            metrics["training_time"] = training_time

            run_losses.append(model.losses)
            run_metrics.append(metrics)
            run_preds.append(preds)
            if constrained:
                run_preds_before.append(preds_before)

        losses_dict[mode] = run_losses
        metrics_dict[mode] = run_metrics
        inputs_dict[mode] = [data.test_inputs] * N
        outputs_dict[mode] = [data.test_outputs_orig] * N
        preds_dict[mode] = run_preds
        if constrained:
            preds_before_dict[mode] = run_preds_before

        if output_dir is not None:
            _save_mode_results(
                output_dir,
                mode,
                constrained,
                run_losses,
                run_metrics,
                data.test_inputs,
                data.test_outputs_orig,
                run_preds,
                run_preds_before,
            )

    if PLOT:
        _plot_results(
            data, losses_dict, metrics_dict, inputs_dict, outputs_dict, preds_dict, preds_before_dict, output_dir
        )


if __name__ == "__main__":
    main()
