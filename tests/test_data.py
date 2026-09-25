import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep import clean_data, impute_missing_values


def test_clean_data():
    raw_dict = {
        'age': [63, 67, 67, 37],
        'sex': [1, 1, 1, 0],
        'trestbps': ['145', '160', '160', '?'],
        'chol': [233, 286, 286, 250],
        'target': [0, 1, 1, 0]
    }
    df_mock = pd.DataFrame(raw_dict)
    
    df_clean = clean_data(df_mock)
    
    # Assert duplicates removed (row count should decrease from 4 to 3)
    assert len(df_clean) == 3
    # Assert '?' converted to NaN
    assert df_clean['trestbps'].isnull().sum() == 1
    # Assert numeric conversion
    assert pd.api.types.is_numeric_dtype(df_clean['trestbps'])


def test_impute_missing_values():
    df_nan = pd.DataFrame({
        'age': [60, 55, 45, 65, 50],
        'sex': [1, 0, 1, 1, 0],
        'trestbps': [140, np.nan, 120, 130, 150],
        'chol': [240, 260, np.nan, 210, 230],
        'ca': [1.0, np.nan, 0.0, 2.0, np.nan],
        'target': [1, 0, 0, 1, 0]
    })
    
    # Test MICE Imputation
    df_mice = impute_missing_values(df_nan, imputer_type='mice')
    assert df_mice.isnull().sum().sum() == 0
    assert len(df_mice) == 5
    
    # Test KNN Imputation
    df_knn = impute_missing_values(df_nan, imputer_type='knn')
    assert df_knn.isnull().sum().sum() == 0
