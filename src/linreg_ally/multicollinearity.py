# Author: Alex Wong
# multicollinearity.py
# 01/10/2025

import altair_ally as aly
import altair as alt
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor 

def check_multicollinearity(train_df: pd.DataFrame, threshold = None, vif_only = False):
    """
    Detects multicollinearity in the training dataset by computing the variance inflation factor (‘VIF’) and pairwise Pearson Correlation for each numeric feature. 

    Parameters
    ----------
    train_df : pd.DataFrame
         Training dataset

    threshold : int 
        Minimum threshold of VIF for a feature to be included in the returned dataframe. 
        Default is None.
    
    vif_only : Boolean
        If true, only a dataframe containing the VIF scores will be returned. Otherwise, the correlation chart is also returned.

    Returns
    -------
    pd.DataFrame
    A dataframe containing the VIF of each feature in train_df. 
    
    alt.ConcatChart
        A chart that shows the pairwise Pearson Correlations of all numeric columns in train_df. 

    Examples
    --------
    >>> from linreg_ally import check_multicollinearity
    >>> vif_df, corr_chart = check_multicollinearity(train_df)
    >>> vif_df = check_multicollinearity(train_df, threshold = 5, vif_only = True)  
    """
    pass