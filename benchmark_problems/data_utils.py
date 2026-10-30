"""Data helpers for the benchmark problems: toy-function sampling and CSV loading."""

from collections.abc import Callable

import numpy as np
import pandas as pd


def generate_data(
    functions: list[Callable],
    num_samples: int,
    left_limit: float | list[float],
    right_limit: float | list[float],
    function_args: None | list[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates data by applying an arbitrary number of functions to random samples of x values.

    Parameters:
    functions (List[Callable]): List of functions to apply to x values.
                                Each function should accept a vector of inputs (x values) and
                                optionally additional keyword arguments.
    num_samples (int): Number of samples to generate.
    left_limit (Union[float, List[float]]): Lower bounds of the input ranges. Can be a single float
                                            for monovariate functions or a list for multivariate.
    right_limit (Union[float, List[float]]): Upper bounds of the input ranges. Can be a single float
                                             for monovariate functions or a list for multivariate.
    function_args (List[Dict], optional): List of dictionaries with arguments for each function.
                                          Each dictionary corresponds to a function in `functions`.
                                          Defaults to None, meaning no extra arguments.

    Returns:
    Tuple[np.ndarray, np.ndarray]:
        Tuple containing:
        - inputs (np.ndarray): Array of sampled x values, shape (num_samples, num_dimensions).
        - outputs (np.ndarray): 2D array of function outputs, shape (num_samples, num_functions).
    """
    # Ensure left_limit and right_limit are lists for multivariate handling
    if isinstance(left_limit, float):
        left_limit = [left_limit]
    if isinstance(right_limit, float):
        right_limit = [right_limit]

    # Check that left_limit and right_limit have the same length
    assert len(left_limit) == len(right_limit), "left_limit and right_limit must have the same length"

    # Generate random x values for each dimension
    num_dimensions = len(left_limit)
    inputs = np.column_stack(
        [np.random.uniform(left_limit[i], right_limit[i], num_samples) for i in range(num_dimensions)]
    )

    # Initialize an empty list to store the outputs of each function
    outputs = []

    # Apply each function to the inputs with respective arguments
    for i, func in enumerate(functions):
        args = function_args[i] if function_args and len(function_args) > i else {}
        result = func(inputs.T, **args).squeeze()  # Transpose inputs for compatibility with functions
        outputs.append(result)

    # Stack outputs vertically to get a shape of (num_samples, num_functions)
    outputs = np.column_stack(outputs)

    return inputs, outputs


def generate_data_feasible(
    functions,
    num_samples: int,
    left_limit,
    right_limit,
    feasibility_check: Callable,
    max_oversample: int = 20,
    function_args=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data by rejection-sampling only feasible points.

    Repeatedly draws candidate samples in bulk and keeps only those where
    ``feasibility_check(inputs, outputs)`` returns True, until
    ``num_samples`` feasible points have been collected.

    Parameters
    ----------
    functions :
        Same as ``generate_data``.
    num_samples :
        Number of *feasible* samples required.
    left_limit, right_limit :
        Domain bounds (float or list of floats).
    feasibility_check :
        Callable ``(inputs: ndarray [N, NI], outputs: ndarray [N, NO]) -> bool[N]``
        that returns True for every feasible point.
    max_oversample :
        Factor by which to oversample each batch to reduce the number of
        rejection rounds.  A value of 20 means we draw 20*num_samples
        candidates per round and keep the feasible ones.
    function_args :
        Same as ``generate_data``.

    Returns
    -------
    inputs  : ``[num_samples, NI]``
    outputs : ``[num_samples, NO]``
    """
    collected_inputs: list[np.ndarray] = []
    collected_outputs: list[np.ndarray] = []
    total_collected = 0

    while total_collected < num_samples:
        # Draw a larger-than-needed batch
        batch_size = max_oversample * (num_samples - total_collected)
        inp, out = generate_data(functions, batch_size, left_limit, right_limit, function_args)
        mask = feasibility_check(inp, out)
        inp_ok = inp[mask]
        out_ok = out[mask]
        if inp_ok.shape[0] == 0:
            continue  # unlucky draw, try again
        collected_inputs.append(inp_ok)
        collected_outputs.append(out_ok)
        total_collected += inp_ok.shape[0]

    all_inputs = np.concatenate(collected_inputs, axis=0)[:num_samples]
    all_outputs = np.concatenate(collected_outputs, axis=0)[:num_samples]
    return all_inputs, all_outputs


class Dataloader:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        t_v_t_ratio: tuple[float] = (10.0, 1.0, 1.0),
    ):
        self.input_path = input_path
        self.output_path = output_path

        # Load CSV files
        try:
            self.input_data = pd.read_csv(self.input_path, header=0)
            self.output_data = pd.read_csv(self.output_path, header=0)
            print(f"Successfully loaded input from {self.input_path} and output from {self.output_path}")
            # Convert to numpy arrays
            self.input_data = self.input_data.to_numpy()
            self.output_data = self.output_data.to_numpy()
            # Split data into training, validation, and test sets
            total_samples = self.input_data.shape[0]
            train_samples = int(total_samples * t_v_t_ratio[0] / sum(t_v_t_ratio))
            val_samples = int(total_samples * t_v_t_ratio[1] / sum(t_v_t_ratio))
            test_samples = total_samples - train_samples - val_samples
            self.train_inputs = self.input_data[:train_samples]
            self.train_outputs = self.output_data[:train_samples]
            self.val_inputs = self.input_data[train_samples : train_samples + val_samples]
            self.val_outputs = self.output_data[train_samples : train_samples + val_samples]
            self.test_inputs = self.input_data[train_samples + val_samples :]
            self.test_outputs = self.output_data[train_samples + val_samples :]
            # Print the number of samples in each dataset
            print(f"Number of training samples: {train_samples}")
            print(f"Number of validation samples: {val_samples}")
            print(f"Number of test samples: {test_samples}")
            # Print the total number of samples
            print(f"Total number of samples: {total_samples}")
            # Print the ratio of training, validation, and test samples
            print(f"Training, Validation, Test ratio: {t_v_t_ratio[0]}:{t_v_t_ratio[1]}:{t_v_t_ratio[2]}")

        except Exception as e:
            print(f"Error loading CSV files: {e}")
            self.input_data = None
            self.output_data = None

    def get_data(self):
        return {
            "train_inputs": self.train_inputs,
            "train_outputs": self.train_outputs,
            "val_inputs": self.val_inputs,
            "val_outputs": self.val_outputs,
            "test_inputs": self.test_inputs,
            "test_outputs": self.test_outputs,
        }
