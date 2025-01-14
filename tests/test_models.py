from linreg_ally.models import run_linear_regression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "feature_1": [1, 2, 3, 4, 5],
    "feature_2": [0.5, 0.1, 0.4, 0.9, 0.6],
    "category": ["a", "b", "a", "b", "c"],
    "target": [1.0, 2.5, 3.4, 4.3, 5.1]
})

target_column = 'target'
numeric_feats = ['feature_1', 'feature_2']
categorical_feats = ['category']
drop_feats = []

# Function to create sample DataFrame for testing
def create_sample_dataframe():
    return pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_2": [0.5, 0.1, 0.4, 0.9, 0.6, 0.7, 0.3, 0.2, 0.8, 1.0],
        "category": ["a", "b", "a", "b", "c", "a", "b", "c", "a", "b"],
        "target": [1.0, 2.5, 3.4, 4.3, 5.1, 6.2, 7.1, 8.3, 9.5, 10.0]
    })

df = create_sample_dataframe()
target_column = 'target'
numeric_feats = ['feature_1', 'feature_2']
categorical_feats = ['category']
drop_feats = []

# Test 1: Check if the function result has correct return type and values
def test_return_type_and_length():
    result = run_linear_regression(df, target_column, numeric_feats, categorical_feats, drop_feats)
    assert isinstance(result, tuple), "The return type should be a tuple."
    assert len(result) == 6, "The tuple should have 6 elements."
    assert isinstance(result[0], Pipeline), "The first element should be a Pipeline."
    assert isinstance(result[1], pd.DataFrame), "The second element should be a DataFrame (X_train)."
    assert isinstance(result[2], pd.DataFrame), "The third element should be a DataFrame (X_test)."
    assert isinstance(result[3], pd.Series), "The fourth element should be a Series (y_train)."
    assert isinstance(result[4], pd.Series), "The fifth element should be a Series (y_test)."
    assert isinstance(result[5], dict), "The sixth element should be a dictionary (scores)."
    assert "r2" in result[5], "r2 score should be in the scores dictionary."
    assert "neg_mean_squared_error" in result[5], "neg_mean_squared_error score should be in the scores dictionary."

# Test 2: Check if TypeError is raised when a non-DataFrame is provided
def test_invalid_dataframe():
    try:
        run_linear_regression("not_a_dataframe", target_column, numeric_feats, categorical_feats, drop_feats)
    except TypeError as e:
        assert str(e) == "dataframe must be a pandas DataFrame."
    else:
        assert False, "TypeError not raised for non-DataFrame input."

# Test 3: Check if ValueError is raised when the target column is not in the DataFrame
def test_target_column_present():
    try:
        run_linear_regression(df.drop(columns=[target_column]), target_column, numeric_feats, categorical_feats, drop_feats)
    except ValueError as e:
        assert f"target_column '{target_column}' is not in the dataframe." in str(e)
    else:
        assert False, "ValueError not raised for missing target column."

# Test 4: Check if an invalid test_size raises a ValueError
def test_invalid_test_size():
    try:
        run_linear_regression(df, target_column, numeric_feats, categorical_feats, drop_feats, test_size=1.5)
    except ValueError as e:
        assert str(e) == "test_size must be between 0.0 and 1.0."
    else:
        assert False, "ValueError not raised for invalid test_size."

# Test 5: Check if an invalid random_state raises a ValueError
def test_invalid_random_state():
    try:
        run_linear_regression(df, target_column, numeric_feats, categorical_feats, drop_feats, random_state="not_an_int")
    except TypeError as e:
        assert str(e) == "random_state must be an integer."
    else:
        assert False, "TypeError not raised for non-integer random_state."

# Test 6: Check if invalid scoring_metrics raises a ValueError
def test_invalid_scoring_metric():
    try:
        run_linear_regression(df, target_column, numeric_feats, categorical_feats, drop_feats, scoring_metrics=['invalid_metric'])
    except ValueError as e:
        assert "are not valid" in str(e)
    else:
        assert False, "ValueError not raised for invalid scoring_metric."
