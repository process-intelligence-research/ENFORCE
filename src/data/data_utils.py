import numpy as np
from typing import Callable, List, Tuple, Dict, Union


def generate_data(
    functions: List[Callable], 
    num_samples: int, 
    left_limit: Union[float, List[float]], 
    right_limit: Union[float, List[float]],
    function_args: Union[None, List[Dict]] = None
) -> Tuple[np.ndarray, np.ndarray]:
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
    inputs = np.column_stack([
        np.random.uniform(left_limit[i], right_limit[i], num_samples)
        for i in range(num_dimensions)
    ])

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
) -> Tuple[np.ndarray, np.ndarray]:
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
    collected_inputs: List[np.ndarray] = []
    collected_outputs: List[np.ndarray] = []
    total_collected = 0

    while total_collected < num_samples:
        # Draw a larger-than-needed batch
        batch_size = max_oversample * (num_samples - total_collected)
        inp, out = generate_data(
            functions, batch_size, left_limit, right_limit, function_args
        )
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


def scale_data(train_inputs, train_outputs, test_inputs, test_outputs):
    """
    Scales the inputs and outputs of the training and test sets.

    Parameters:
    train_inputs (np.ndarray): Training set inputs.
    train_outputs (np.ndarray): Training set outputs.
    test_inputs (np.ndarray): Test set inputs.
    test_outputs (np.ndarray): Test set outputs.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]: 
        Tuple containing:
        - train_inputs_scaled (np.ndarray): Scaled training set inputs.
        - train_outputs_scaled (np.ndarray): Scaled training set outputs.
        - test_inputs_scaled (np.ndarray): Scaled test set inputs.
        - test_outputs_scaled (np.ndarray): Scaled test set outputs.
        - scaling_params (Dict): Dictionary containing the scaling parameters.
"""
    input_mean = train_inputs.mean(axis=0)
    input_std = train_inputs.std(axis=0)
    output_mean = train_outputs.mean(axis=0)
    output_std = train_outputs.std(axis=0)

    # Per-column normalisation: columns whose std is near zero (e.g. FB
    # multiplier columns that are always 0 in training data) are left at
    # their mean-centred value instead of being divided by a near-zero std.
    output_std_safe = np.where(output_std >= 1e-5, output_std, 1.0)

    train_inputs_scaled = (train_inputs - input_mean) / input_std
    train_outputs_scaled = (train_outputs - output_mean) / output_std_safe

    test_inputs_scaled = (test_inputs - input_mean) / input_std
    test_outputs_scaled = (test_outputs - output_mean) / output_std_safe

    scaling_params = {
        'input_mean': input_mean,  # mean of the training inputs. datatype: np.ndarray
        'input_std': input_std,
        'output_mean': output_mean,
        # Store output_std_safe so model.py's all-or-nothing std check always
        # passes: columns with zero training std (e.g. FB multiplier columns)
        # get a safe std of 1.0, leaving them effectively un-normalised.
        'output_std': output_std_safe,
    }

    return (
        train_inputs_scaled, 
        train_outputs_scaled, 
        test_inputs_scaled, 
        test_outputs_scaled, 
        scaling_params)
