import numpy as np


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
        "input_mean": input_mean,  # mean of the training inputs. datatype: np.ndarray
        "input_std": input_std,
        "output_mean": output_mean,
        # Columns with zero training std (e.g. FB multiplier columns) get a
        # safe std of 1.0, leaving them only centred (same rule as Scaler).
        "output_std": output_std_safe,
    }

    return (train_inputs_scaled, train_outputs_scaled, test_inputs_scaled, test_outputs_scaled, scaling_params)
