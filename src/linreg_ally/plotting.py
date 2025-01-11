# Author: Merari Santana
# qq_and_residuals_plot.py
# 01/10/2025

import matplotlib.pyplot as plt
import scipy.stats as stats

def qq_and_residuals_plot(model_residuals, fitted_values):
    """
    Generate a Q-Q plot (standardized residuals vs theoretical quantiles) 
    and a Residuals vs. Fitted Values plot for regression diagnostics.

    Parameters
    ----------
    model_residuals : array-like
        Residuals from the regression model. Raw residuals will be used 
        directly for the Residuals vs. Fitted Values plot. For the Q-Q plot, 
        these residuals will be transformed into standardized residuals 
        (i.e., residuals divided by their standard deviation).
    
    fitted_values : array-like
        Fitted (predicted) values from the regression model.

    Returns
    -------
    None
        Displays the Q-Q plot and Residuals vs. Fitted Values plot.

    Examples
    --------
    >>> qq_and_residuals_plot(model.resid, model.fittedvalues)
    """
    pass