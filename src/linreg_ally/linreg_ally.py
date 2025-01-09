def run_linear_regression(dataframe):
    """
    Performs linear regression with preprocessing using sklearn and outputs evaluation scoring metrics.
    
    Parameters
    ----------
    dataframe: `pandas.DataFrame`
        full dataset including features and target.
    target_column: `string`
        name of the target variable column.
    numeric_feats: `list`
        columns to apply StandardScaler.
    categorical_feats: `list`
        columns to apply OneHotEncoder.
    drop_feats: `list`, optional
        columns to drop (default None).
    test_size: `float`, optional
        proportion of the dataset to include in the test split (default 0.2).
    random_state: `int`, optional
        controls the shuffling applied to the data before the split (default None).
    scoring_metrics: `list`, optional
        scoring metrics to evaluate the model (default 'r2', 'mean_squared_error').
    
    Returns
    -------
    tuple
        the fitted model
        DataFrames for the training and test features
        Series for the training and test labels
        dictionary of metric scores with metric names as keys
    
    Examples
    ---------
    >>> import pandas as pd
    >>> from linreg_ally.linreg_ally import run_linear_regression
    >>> df = pd.DataFrame({
    ...     "feature_1": [1, 2, 3, 4],
    ...     "feature_2": [0.5, 0.1, 0.4, 0.9],
    ...     "category": ["a", "b", "a", "b"],
    ...     "target": [1.0, 2.5, 3.4, 4.3]
    ... })
    >>> target_column = 'target'
    >>> numeric_feats = ['feature_1', 'feature_2']
    >>> categorical_feats = ['category']
    >>> drop_feats = []
    >>> best_model, X_train, X_test, y_train, y_test, scores = run_linear_regression(
    ...     df, target_column, numeric_feats, categorical_feats, drop_feats, metrics=['r2', 'mean_squared_error']
    ... )
    >>> scores
    {'r2': 0.52, 'mean_squared_error': 1.23}
    """
    pass
