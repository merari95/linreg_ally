import pytest
import pandas as pd
import numpy as np
from altair import Chart
from linreg_ally.plotting import qq_and_residuals_plot
from altair import HConcatChart

# Unit Tests

def test_qq_and_residuals_plot_valid_inputs():
    """
    Test the function with valid inputs and check if it returns the correct Altair charts.
    """
    np.random.seed(42)
    y_actual = np.random.normal(10, 2, 100)
    y_predicted = y_actual + np.random.normal(0, 1, 100)

    # Test with concatenate=True
    result = qq_and_residuals_plot(y_actual, y_predicted, concatenate=True)
    assert isinstance(result, (Chart, HConcatChart)), "Expected an Altair Chart or HConcatChart when concatenate=True"

    # Test with concatenate=False
    qq_plot, residuals_plot = qq_and_residuals_plot(y_actual, y_predicted, concatenate=False)
    assert isinstance(qq_plot, Chart), "Expected an Altair Chart for Q-Q Plot"
    assert isinstance(residuals_plot, Chart), "Expected an Altair Chart for Residuals vs. Fitted Values Plot"

def test_qq_and_residuals_plot_empty_inputs():
    """
    Test the function with empty inputs to ensure it raises an appropriate error.
    """
    y_actual = []
    y_predicted = []

    with pytest.raises(ValueError, match="Inputs must not be empty"):
        qq_and_residuals_plot(y_actual, y_predicted)

def test_qq_and_residuals_plot_mismatched_lengths():
    """
    Test the function with mismatched input lengths to ensure it raises an appropriate error.
    """
    y_actual = [1, 2, 3, 4]
    y_predicted = [1, 2]

    with pytest.raises(ValueError, match="Inputs must have the same length"):
        qq_and_residuals_plot(y_actual, y_predicted)

def test_qq_and_residuals_plot_non_numeric_inputs():
    """
    Test the function with non-numeric inputs to ensure it raises an appropriate error.
    """
    y_actual = ["a", "b", "c"]
    y_predicted = ["d", "e", "f"]

    with pytest.raises(TypeError, match="Inputs must be numeric"):
        qq_and_residuals_plot(y_actual, y_predicted)

def test_qq_and_residuals_plot_single_point():
    """
    Test the function with a single data point to ensure it handles edge cases correctly.
    """
    y_actual = [10]
    y_predicted = [10]

    with pytest.raises(ValueError, match="Insufficient data points for plotting"):
        qq_and_residuals_plot(y_actual, y_predicted)

# Running the tests
if __name__ == "__main__":
    pytest.main()
