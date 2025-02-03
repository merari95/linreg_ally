import pytest
from linreg_ally.multicollinearity import check_multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import altair as alt
import pandas as pd 

data = {
    "Feature_1": [1, 2, 3, 4, 5],
    "Feature_2": [2, 4, 6, 8, 10],  # Perfectly correlated with Feature_1
    "Feature_3": [5, 3, 6, 2, 7],
    "Feature_4": [8, 4, 7, 5, 6]   # Moderately correlated with Feature_3
}

train_df = pd.DataFrame(data)

def test_function_output_with_perfect_correlation():
    """
    Test use case when train data has perfect and moderate correlation.
    """
    vif_df, corr_chart = check_multicollinearity(train_df) 

    vif_df_len = 4 
    vif_column_type = 'float64'
    
    # tests for vif_df output 
    assert isinstance(vif_df, pd.DataFrame)
    assert vif_df.iloc[:,1].dtype == vif_column_type  # check if the VIF scores are floats
    assert len(vif_df) == vif_df_len # check number of row of vif_df 

    # tests for chart output 
    assert type(corr_chart) == alt.vegalite.v5.api.Chart
    assert corr_chart.encoding.x.shorthand == 'level_0'
    assert corr_chart.encoding.y.shorthand == 'level_1'
    assert corr_chart.encoding.size.shorthand == 'corr'

def test_vif_value(): 
    """
    Test VIF values in vif_df created by the function. 
    """
    vif_df = check_multicollinearity(train_df, vif_only=True) 

    vif_0 = variance_inflation_factor(train_df, 0)
    vif_1 = variance_inflation_factor(train_df, 1)
    vif_2 = variance_inflation_factor(train_df, 2)
    vif_3 = variance_inflation_factor(train_df, 3)

    assert vif_df.iloc[0,1] == vif_0
    assert vif_df.iloc[1,1] == vif_1 
    assert vif_df.iloc[2,1] == vif_2 
    assert vif_df.iloc[3,1] == vif_3   

def test_threshold():
    """
    Test if VIF values in vif_df are greater than or equal to the specified threshold.
    """
    vif_df = check_multicollinearity(train_df, threshold=15, vif_only=True)
    threshold = 15

    assert all(vif_df.iloc[:,1] >= threshold)

    




    