import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def default_cost_matrix():
    """
    Returns standard healthcare cost matrix ($):
    - FP (False Positive): Unnecessary diagnostic workup & preventive treatment ($1,500)
    - FN (False Negative): Missed diagnosis leading to acute cardiac event / readmission ($15,000)
    - TP (True Positive): Timely, appropriate intervention & treatment ($3,000)
    - TN (True Negative): Standard routine preventive care ($200)
    """
    return {
        'FP': 1500.0,
        'FN': 15000.0,
        'TP': 3000.0,
        'TN': 200.0
    }


def calculate_financial_impact(y_true, y_pred, cost_matrix=None) -> dict:
    """
    Calculates simulated financial impact of model predictions vs operational baselines.
    
    Parameters:
    - y_true: Ground truth binary target array (0 or 1)
    - y_pred: Predicted binary outcome array (0 or 1)
    - cost_matrix: Optional dict specifying 'FP', 'FN', 'TP', 'TN' costs
    
    Returns dict with financial metrics:
    - model_total_cost
    - treat_all_cost
    - treat_none_cost
    - net_savings_vs_treat_all
    - net_savings_vs_treat_none
    - confusion_matrix: dict of TN, FP, FN, TP counts
    """
    if cost_matrix is None:
        cost_matrix = default_cost_matrix()
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate model confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    model_cost = (
        tn * cost_matrix.get('TN', 200.0) +
        fp * cost_matrix.get('FP', 1500.0) +
        fn * cost_matrix.get('FN', 15000.0) +
        tp * cost_matrix.get('TP', 3000.0)
    )
    
    # Baseline 1: Treat Everyone (y_pred_all = 1 for all)
    y_pred_all = np.ones_like(y_true)
    cm_all = confusion_matrix(y_true, y_pred_all, labels=[0, 1])
    tn_all, fp_all, fn_all, tp_all = cm_all.ravel()
    treat_all_cost = (
        tn_all * cost_matrix.get('TN', 200.0) +
        fp_all * cost_matrix.get('FP', 1500.0) +
        fn_all * cost_matrix.get('FN', 15000.0) +
        tp_all * cost_matrix.get('TP', 3000.0)
    )
    
    # Baseline 2: Treat No One (y_pred_none = 0 for all)
    y_pred_none = np.zeros_like(y_true)
    cm_none = confusion_matrix(y_true, y_pred_none, labels=[0, 1])
    tn_none, fp_none, fn_none, tp_none = cm_none.ravel()
    treat_none_cost = (
        tn_none * cost_matrix.get('TN', 200.0) +
        fp_none * cost_matrix.get('FP', 1500.0) +
        fn_none * cost_matrix.get('FN', 15000.0) +
        tp_none * cost_matrix.get('TP', 3000.0)
    )
    
    savings_vs_treat_all = treat_all_cost - model_cost
    savings_vs_treat_none = treat_none_cost - model_cost
    
    return {
        'model_total_cost': float(model_cost),
        'treat_all_cost': float(treat_all_cost),
        'treat_none_cost': float(treat_none_cost),
        'net_savings_vs_treat_all': float(savings_vs_treat_all),
        'net_savings_vs_treat_none': float(savings_vs_treat_none),
        'confusion_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        },
        'per_patient_model_cost': float(model_cost / len(y_true)),
        'per_patient_treat_all_cost': float(treat_all_cost / len(y_true)),
        'per_patient_treat_none_cost': float(treat_none_cost / len(y_true))
    }


if __name__ == '__main__':
    y_true_mock = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred_mock = [0, 1, 0, 0, 1, 0, 1, 1]
    res = calculate_financial_impact(y_true_mock, y_pred_mock)
    print("Cost Effectiveness Analysis Test:")
    print(res)
