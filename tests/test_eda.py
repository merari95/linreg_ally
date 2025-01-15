# Author: Paramveer Singh
# test_eda.py
# 15 January 2025

import pytest
import altair as alt
from vega_datasets import data
from linreg_ally.eda import eda_summary

@pytest.fixture
def iris_data():
    return data.iris()

def test_good_data_no_color(iris_data):
    # Store result
    plot = eda_summary(iris_data)

    # Assert output type
    assert isinstance(plot, alt.ConcatChart), 'Function does not return correct type!'
    
    # Assert x's are correct
    feature_names = iris_data.columns.tolist()
    feature_names.remove('species')
    
    feats_used = []
    for i in range(len(plot.concat)):
        feats_used.append(plot.concat[i].layer[0].encoding.x.shorthand)

    assert set(feature_names) == set(feats_used), 'All x variables are not used!'

def test_good_data_color(iris_data):
    # Store result
    plot = eda_summary(iris_data, color='species')

    # Assert output type
    assert isinstance(plot, alt.ConcatChart), 'Function does not return correct type!'

    # Assert x's are correct
    feature_names = iris_data.columns.tolist()
    feature_names.remove('species')

    feats_used = []
    for i in range(len(plot.concat)):
        feats_used.append(plot.concat[i].layer[0].encoding.x.shorthand)

    assert set(feature_names) == set(
        feats_used), 'All x variables are not used!'

    # Assert color is `species`
    assert plot.concat[0].layer[0].encoding.color.shorthand == 'species', 'The wrong column is used for coloring!'
    