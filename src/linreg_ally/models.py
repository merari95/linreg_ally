import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer

def run_linear_regression(dataframe, target_column, numeric_feats, categorical_feats, drop_feats=None, test_size=0.2, random_state=None, scoring_metrics=['r2', 'mean_squared_error']):
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
    >>> best_model, X_train, X_test, y_train, y_test, scoring_metrics = run_linear_regression(
    ...     df, target_column, numeric_feats, categorical_feats, drop_feats, metrics=['r2', 'mean_squared_error']
    ... )
    >>> scores
    {'r2': 0.52, 'mean_squared_error': 1.23}
    """
    
    drop_feats = drop_feats if drop_feats is not None else []

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_feats),
        (OneHotEncoder(), categorical_feats),
        ('drop', drop_feats)
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipe.fit(X_train, y_train)

    best_model = pipe

    predictions = best_model.predict(X_test)

    scores = {}
    for metric in scoring_metrics:
        scorer = get_scorer(metric)
        scores[metric] = scorer._score_func(y_test, predictions)

    print("Model Summary")
    print("------------------------")
    for metric, score in scores.items():
        print(f"Test {metric}: {score:.3f}")

    return best_model, X_train, X_test, y_train, y_test, scores
