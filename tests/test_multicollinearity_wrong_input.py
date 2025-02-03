import pytest
from linreg_ally.multicollinearity import check_multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import altair as alt
import pandas as pd 


data_with_str = {
    "Feature_1": [1, 2, 3, 4, 5],
    "Feature_2": [1, 2, 3, 4, 6],  
    "Feature_3": [5, 3, 6, 2, 7],
    "Feature_4": ['a', 'b', 'c', 'd', 'e']   # string column
}

data_numeric_only = {
    "Feature_1": [1, 2, 3, 4, 5],
    "Feature_2": [1, 2, 3, 4, 6],  
    "Feature_3": [5, 3, 6, 2, 7],
    #"Feature_4": ['a', 'b', 'c', 'd', 'e']   # string column
}

list_data = [1, 2, 3, 4, 5]

train_df_with_str = pd.DataFrame(data_with_str)
train_df_numeric_only = pd.DataFrame(data_numeric_only)

def test_list_as_function_input():
    """
    Test when input is not a dataframe. 
    """
    with pytest.raises(TypeError):
        check_multicollinearity(list_data) 


def test_dataframe_with_str_column():
    """
    Test when input dataframe has a column with str datatype. 
    """

    vif_df = check_multicollinearity(train_df_with_str, vif_only=True) 

    vif_df_len = 3 
    vif_column_type = 'float64'
    vif_0 = variance_inflation_factor(train_df_numeric_only, 0)
    vif_1 = variance_inflation_factor(train_df_numeric_only, 1)
    vif_2 = variance_inflation_factor(train_df_numeric_only, 2)

    assert vif_df.iloc[:,1].dtype == vif_column_type  # check if the VIF scores are floats
    assert len(vif_df) == vif_df_len # check number of row of vif_df 
    assert vif_df.iloc[0,1] == vif_0
    assert vif_df.iloc[1,1] == vif_1 
    assert vif_df.iloc[2,1] == vif_2 




