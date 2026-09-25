import pytest
import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import engineer_features
from src.cost_analysis import calculate_financial_impact


def test_engineer_features():
    df_mock = pd.DataFrame({
        'age': [60, 50, 70],
        'sex': [1, 0, 1],
        'trestbps': [140, 120, 150],
        'chol': [240, 200, 300],
        'fbs': [1, 0, 1],
        'thalach': [150, 160, 120],
        'exang': [0, 1, 1],
        'oldpeak': [1.5, 0.5, 2.5]
    })
    
    df_eng = engineer_features(df_mock)
    new_cols = set(df_eng.columns) - set(df_mock.columns)
    
    # Assert at least 5 new features created
    assert len(new_cols) >= 5
    assert 'map_bp' in new_cols
    assert 'st_hr_ratio' in new_cols
    assert 'chol_age_ratio' in new_cols
    assert 'cardiac_risk_score' in new_cols
    assert 'risk_category' in new_cols


def test_calculate_financial_impact():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 1, 0, 0] # TP: 1, FP: 1, FN: 1, TN: 1
    
    custom_cost_matrix = {
        'FP': 1000.0,
        'FN': 10000.0,
        'TP': 2000.0,
        'TN': 100.0
    }
    
    res = calculate_financial_impact(y_true, y_pred, custom_cost_matrix)
    
    # Expected model cost: 1*100 + 1*1000 + 1*10000 + 1*2000 = 13100.0
    assert res['model_total_cost'] == 13100.0
    assert res['confusion_matrix']['TN'] == 1
    assert res['confusion_matrix']['FP'] == 1
    assert res['confusion_matrix']['FN'] == 1
    assert res['confusion_matrix']['TP'] == 1


def test_champion_f1_score():
    metrics_path = 'metrics/test_metrics.json'
    assert os.path.exists(metrics_path), "metrics/test_metrics.json does not exist. Run src/model.py first."
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    assert 'f1_score' in metrics
    f1_score_val = metrics['f1_score']
    assert f1_score_val >= 0.80, f"Champion model F1-score ({f1_score_val}) is below 0.80 target threshold!"
