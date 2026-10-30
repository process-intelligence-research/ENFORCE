"""Tests for synthetic benchmark data generation and scaling helpers."""

import numpy as np
import pandas as pd
import pytest

from benchmark_problems.data_utils import Dataloader, generate_data, generate_data_feasible
from enforce.data import scale_data


def test_generate_data_supports_multiple_dimensions_and_function_arguments(monkeypatch):
    monkeypatch.setattr(np.random, "uniform", lambda low, high, size: np.full(size, (low + high) / 2))

    inputs, outputs = generate_data(
        functions=[lambda x: x[0] + x[1], lambda x, offset: x[0] * offset],
        num_samples=4,
        left_limit=[-2.0, 2.0],
        right_limit=[2.0, 6.0],
        function_args=[{}, {"offset": 3.0}],
    )

    np.testing.assert_allclose(inputs, np.tile([0.0, 4.0], (4, 1)))
    np.testing.assert_allclose(outputs, np.tile([4.0, 0.0], (4, 1)))


def test_generate_data_checks_matching_dimension_bounds():
    with pytest.raises(AssertionError, match="same length"):
        generate_data([lambda x: x[0]], 3, [-1.0, -2.0], [1.0])


def test_generate_data_feasible_returns_requested_feasible_rows():
    np.random.seed(13)
    inputs, outputs = generate_data_feasible(
        functions=[lambda x: x[0]],
        num_samples=12,
        left_limit=-1.0,
        right_limit=1.0,
        feasibility_check=lambda x, y: x[:, 0] >= 0.0,
        max_oversample=4,
    )

    assert inputs.shape == outputs.shape == (12, 1)
    assert (inputs[:, 0] >= 0.0).all()
    np.testing.assert_allclose(inputs, outputs)


def test_scale_data_uses_training_statistics_and_handles_constant_output_columns():
    train_x = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    train_y = np.array([[1.0, 7.0], [3.0, 7.0], [5.0, 7.0]])
    test_x = np.array([[6.0, 7.0]])
    test_y = np.array([[7.0, 7.0]])

    x_scaled, y_scaled, test_x_scaled, test_y_scaled, stats = scale_data(train_x, train_y, test_x, test_y)

    np.testing.assert_allclose(x_scaled.mean(axis=0), 0.0)
    np.testing.assert_allclose(y_scaled.mean(axis=0), 0.0)
    np.testing.assert_allclose(test_x_scaled, [[4.0 / np.sqrt(8.0 / 3.0)] * 2])
    np.testing.assert_allclose(test_y_scaled, [[4.0 / np.sqrt(8.0 / 3.0), 0.0]])
    np.testing.assert_allclose(stats["output_std"], [np.sqrt(8.0 / 3.0), 1.0])


def test_dataloader_splits_rows_in_order_and_keeps_input_output_alignment(tmp_path):
    input_path, output_path = tmp_path / "inputs.csv", tmp_path / "outputs.csv"
    inputs = pd.DataFrame({"x": np.arange(11), "z": np.arange(11) * 10})
    outputs = pd.DataFrame({"y": np.arange(11) * -1})
    inputs.to_csv(input_path, index=False)
    outputs.to_csv(output_path, index=False)

    data = Dataloader(str(input_path), str(output_path), t_v_t_ratio=(7.0, 2.0, 1.0)).get_data()

    assert data["train_inputs"].shape == (7, 2)
    assert data["val_inputs"].shape == (2, 2)
    assert data["test_inputs"].shape == (2, 2)
    np.testing.assert_array_equal(data["train_inputs"][:, 0], np.arange(7))
    np.testing.assert_array_equal(data["val_outputs"][:, 0], -np.arange(7, 9))
    np.testing.assert_array_equal(data["test_inputs"][:, 0], np.arange(9, 11))
