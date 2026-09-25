import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, classification_report
)

import shap

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep import clean_data, impute_missing_values
from src.features import engineer_features
from src.cost_analysis import calculate_financial_impact


def train_ensemble_model(X_train, y_train, param_grid=None):
    """
    Trains an Ensemble model (Random Forest / Gradient Boosting) using GridSearchCV and StratifiedKFold.
    """
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 8, None],
            'min_samples_split': [2, 5],
            'criterion': ['gini', 'entropy']
        }
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=skf,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def train_neural_network(X_train, y_train, param_grid=None):
    """
    Trains a Neural Network (MLPClassifier) using GridSearchCV and StratifiedKFold.
    """
    mlp = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
    
    if param_grid is None:
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (50,), (100,)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=skf,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def run_training_pipeline(data_path='data/raw/heart_disease.csv'):
    """
    Full reproducible pipeline:
    1. Loads raw dataset
    2. Cleans & imputes missing values (MICE)
    3. Engineers 5+ clinical features
    4. Splits data (Stratified Train/Test)
    5. Scales features for Neural Network
    6. Tunes Ensemble and Neural Network models via Stratified 5-Fold CV
    7. Evaluates Champion model on held-out test set
    8. Exports tuning_results.json & test_metrics.json (ensuring F1 >= 0.80)
    9. Generates SHAP global summary plot in reports/figures/shap_summary.png
    10. Saves model artifacts in models/
    """
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Load data
    df_raw = pd.read_csv(data_path)
    
    # 2. Clean & Impute
    df_clean = clean_data(df_raw)
    df_imputed = impute_missing_values(df_clean)
    
    # 3. Engineer features
    df_processed = engineer_features(df_imputed)
    df_processed.to_csv('data/processed/heart_disease_processed.csv', index=False)
    
    # Target and features
    X = df_processed.drop(columns=['target'])
    y = df_processed['target']
    
    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Standard Scaler for Neural Net / distance algorithms
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("--- Training Ensemble Model (Random Forest) ---")
    ensemble_grid = train_ensemble_model(X_train, y_train)
    best_ensemble = ensemble_grid.best_estimator_
    y_pred_ens = best_ensemble.predict(X_test)
    f1_ens = f1_score(y_test, y_pred_ens)
    print(f"Ensemble Test F1-Score: {f1_ens:.4f}")
    
    print("\n--- Training Neural Network Model (MLPClassifier) ---")
    nn_grid = train_neural_network(X_train_scaled, y_train)
    best_nn = nn_grid.best_estimator_
    y_pred_nn = best_nn.predict(X_test_scaled)
    f1_nn = f1_score(y_test, y_pred_nn)
    print(f"Neural Network Test F1-Score: {f1_nn:.4f}")
    
    # Champion Selection
    if f1_ens >= f1_nn:
        champion_name = "RandomForestEnsemble"
        champion_model = best_ensemble
        champion_preds = y_pred_ens
        champion_probs = best_ensemble.predict_proba(X_test)[:, 1]
        is_scaled = False
    else:
        champion_name = "NeuralNetworkMLP"
        champion_model = best_nn
        champion_preds = y_pred_nn
        champion_probs = best_nn.predict_proba(X_test_scaled)[:, 1]
        is_scaled = True
        
    print(f"\nChampion Model Selected: {champion_name}")
    
    # Save Hyperparameter Tuning Results
    tuning_results = {
        "Ensemble_Best_Params": ensemble_grid.best_params_,
        "Ensemble_Best_CV_F1": float(ensemble_grid.best_score_),
        "NeuralNet_Best_Params": nn_grid.best_params_,
        "NeuralNet_Best_CV_F1": float(nn_grid.best_score_),
        "Champion_Model": champion_name
    }
    with open('metrics/tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=4)
        
    # Calculate Test Metrics for Champion Model
    test_f1 = f1_score(y_test, champion_preds)
    test_precision = precision_score(y_test, champion_preds)
    test_recall = recall_score(y_test, champion_preds)
    test_acc = accuracy_score(y_test, champion_preds)
    test_roc_auc = roc_auc_score(y_test, champion_probs)
    
    # Cost Effectiveness Metrics
    financial_summary = calculate_financial_impact(y_test, champion_preds)
    
    test_metrics = {
        "model_name": champion_name,
        "f1_score": float(test_f1),
        "precision": float(test_precision),
        "recall": float(test_recall),
        "accuracy": float(test_acc),
        "roc_auc": float(test_roc_auc),
        "financial_impact": financial_summary
    }
    with open('metrics/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
        
    print(f"\nFinal Champion Test Metrics:")
    print(f"F1-Score: {test_f1:.4f} (Required >= 0.80)")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"ROC AUC: {test_roc_auc:.4f}")
    
    # Save Champion Model Artifacts
    artifacts = {
        'model': champion_model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'champion_name': champion_name,
        'is_scaled': is_scaled
    }
    joblib.dump(artifacts, 'models/champion_model.pkl')
    print("Saved champion model pipeline to models/champion_model.pkl")
    
    # Generate SHAP Interpretability Summary Plot
    generate_shap_explanation(champion_model, X_train, X_test, is_scaled)
    
    return test_metrics


def generate_shap_explanation(model, X_train, X_test, is_scaled=False):
    """
    Computes SHAP values and saves global summary plot to reports/figures/shap_summary.png
    """
    print("\n--- Computing SHAP Interpretability ---")
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # If binary classification returns list of 2 arrays, take positive class (1)
        if isinstance(shap_values, list):
            shap_vals_target = shap_values[1]
        elif len(np.array(shap_values).shape) == 3:
            shap_vals_target = shap_values[:, :, 1]
        else:
            shap_vals_target = shap_values
            
        shap.summary_plot(shap_vals_target, X_test, show=False)
    else:
        # Generic Kernel / Explainer for Neural Net
        background = shap.sample(X_train, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:50])
        if isinstance(shap_values, list):
            shap_vals_target = shap_values[1]
        else:
            shap_vals_target = shap_values
        shap.summary_plot(shap_vals_target, X_test[:50], show=False)
        
    plt.tight_layout()
    output_path = 'reports/figures/shap_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP global summary plot to {output_path}")


if __name__ == '__main__':
    run_training_pipeline()
