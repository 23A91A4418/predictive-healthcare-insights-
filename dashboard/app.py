import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
# Safely handle PyTorch DLL load failures on Windows when SHAP imports
try:
    import torch
except Exception:
    sys.modules['torch'] = None

import shap


# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep import clean_data, impute_missing_values
from src.features import engineer_features
from src.cost_analysis import calculate_financial_impact, default_cost_matrix
from dashboard.components.widgets import render_header, render_metric_card

st.set_page_config(
    page_title="Predictive Healthcare & Cost Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load cached artifacts
@st.cache_resource
def load_champion_artifacts():
    model_path = 'models/champion_model.pkl'
    if not os.path.exists(model_path):
        from src.model import run_training_pipeline
        run_training_pipeline()
    return joblib.load(model_path)

@st.cache_data
def load_processed_dataset():
    data_path = 'data/processed/heart_disease_processed.csv'
    if not os.path.exists(data_path):
        from src.model import run_training_pipeline
        run_training_pipeline()
    return pd.read_csv(data_path)

@st.cache_data
def load_test_metrics():
    metrics_path = 'metrics/test_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}

artifacts = load_champion_artifacts()
df_processed = load_processed_dataset()
test_metrics = load_test_metrics()

model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']
champion_name = artifacts['champion_name']
is_scaled = artifacts['is_scaled']

render_header()

# Sidebar: Patient Input Controls
st.sidebar.header("📋 Patient Clinical Profile")

age = st.sidebar.slider("Age (years)", 20, 85, 58)
sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
cp = st.sidebar.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                          format_func=lambda x: {1: "1: Typical Angina", 2: "2: Atypical Angina", 3: "3: Non-Anginal Pain", 4: "4: Asymptomatic"}[x])
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 90, 200, 135)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 120, 560, 245)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True (1)" if x == 1 else "False (0)")
restecg = st.sidebar.selectbox("Resting ECG Result", options=[0, 1, 2], 
                               format_func=lambda x: {0: "0: Normal", 1: "1: ST-T Abnormality", 2: "2: LV Hypertrophy"}[x])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 142)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.4, step=0.1)
slope = st.sidebar.selectbox("Slope of ST Segment", options=[1, 2, 3], 
                            format_func=lambda x: {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}[x])
ca = st.sidebar.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3], index=1)
thal = st.sidebar.selectbox("Thalassemia", options=[3, 6, 7], 
                            format_func=lambda x: {3: "3: Normal", 6: "6: Fixed Defect", 7: "7: Reversible Defect"}[x])

# Build Patient DataFrame & Engineer Features
input_raw = pd.DataFrame([{
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
    'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
    'slope': slope, 'ca': ca, 'thal': thal
}])
input_engineered = engineer_features(input_raw)
input_features = input_engineered[feature_names]

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "🩺 1. Patient Risk Predictor & Micro SHAP", 
    "📊 2. Cohort Analytics & Macro SHAP", 
    "💰 3. Financial Cost-Effectiveness Calculator"
])

# ----------------------------------------------------
# TAB 1: Patient Outcome Predictor & Local SHAP
# ----------------------------------------------------
with tab1:
    st.subheader("Individual Patient Clinical Assessment")
    
    if is_scaled:
        input_scaled = scaler.transform(input_features)
        prob = model.predict_proba(input_scaled)[0, 1]
    else:
        prob = model.predict_proba(input_features)[0, 1]
        
    pred_label = "HIGH RISK (Heart Disease Likely)" if prob >= 0.50 else "LOW RISK (Healthy)"
    card_color = "#E74C3C" if prob >= 0.50 else "#27AE60"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Predicted Risk Class", pred_label, color=card_color)
    with col2:
        render_metric_card("Disease Probability", f"{prob*100:.1f}%", color=card_color)
    with col3:
        render_metric_card("Champion Model Architecture", champion_name, f"Test F1: {test_metrics.get('f1_score', 0):.4f}", color="#2E86C1")
        
    st.markdown("---")
    st.subheader("🔬 Local Patient Interpretability (SHAP Value Breakdown)")
    st.markdown("Below is the game-theoretic SHAP explanation decomposing how each patient factor shifts the risk prediction from baseline.")
    
    # Compute local SHAP value
    try:
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_features)
            if isinstance(shap_values, list):
                shap_val_single = shap_values[1][0]
                expected_val = explainer.expected_value[1]
            else:
                shap_val_single = shap_values[0]
                expected_val = explainer.expected_value
        else:
            background = scaler.transform(df_processed.drop(columns=['target']).sample(30, random_state=42)) if is_scaled else df_processed.drop(columns=['target']).sample(30, random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            inp_eval = scaler.transform(input_features) if is_scaled else input_features
            shap_values = explainer.shap_values(inp_eval)
            shap_val_single = shap_values[0][:, 1] if isinstance(shap_values, list) else shap_values[0]
            expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

        fig, ax = plt.subplots(figsize=(10, 5))
        top_indices = np.argsort(np.abs(shap_val_single))[-8:]
        top_features = [feature_names[i] for i in top_indices]
        top_shap_vals = [shap_val_single[i] for i in top_indices]
        colors = ['#E74C3C' if v > 0 else '#27AE60' for v in top_shap_vals]
        
        ax.barh(top_features, top_shap_vals, color=colors)
        ax.set_xlabel("SHAP Value (Impact on Model Risk Score)")
        ax.set_title(f"Top Clinical Risk Drivers for Current Patient")
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not compute real-time SHAP plot: {e}")

# ----------------------------------------------------
# TAB 2: Cohort Analytics & Global SHAP
# ----------------------------------------------------
with tab2:
    st.subheader("Global Clinical Population Insights & Feature Importance")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("### Global SHAP Summary Plot")
        shap_img_path = 'reports/figures/shap_summary.png'
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption="Global SHAP Feature Importance across Population", use_column_width=True)
        else:
            st.info("Run src/model.py to generate global SHAP plot.")
            
    with col_b:
        st.markdown("### Population Feature Distributions")
        selected_feat = st.selectbox("Select Feature to Visualize", options=feature_names, index=feature_names.index('thalach'))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df_processed, x=selected_feat, hue='target', kde=True, ax=ax, palette='coolwarm')
        ax.set_title(f"Distribution of {selected_feat} by Target Class")
        st.pyplot(fig)
        plt.close()

# ----------------------------------------------------
# TAB 3: Financial Impact & Cost Calculator
# ----------------------------------------------------
with tab3:
    st.subheader("Hospital Financial Impact & Cost-Effectiveness Simulator")
    st.markdown("Simulate the financial outcomes of implementing the predictive model against operational baselines ('Treat Everyone' vs 'Treat No One').")
    
    col_c1, col_c2 = st.columns([1, 2])
    
    with col_c1:
        st.markdown("### Adjust Cost Parameters ($)")
        fp_cost = st.slider("Cost of False Positive (Preventive Workup)", 500, 5000, 1500, step=100)
        fn_cost = st.slider("Cost of False Negative (Emergency Readmission)", 5000, 30000, 15000, step=500)
        tp_cost = st.slider("Cost of True Positive (Timely Treatment)", 1000, 10000, 3000, step=250)
        tn_cost = st.slider("Cost of True Negative (Routine Care)", 50, 1000, 200, step=50)
        
        cost_mat = {
            'FP': float(fp_cost),
            'FN': float(fn_cost),
            'TP': float(tp_cost),
            'TN': float(tn_cost)
        }
        
    with col_c2:
        st.markdown("### Financial Simulation Results")
        
        # Calculate impact based on test metrics confusion matrix
        cm = test_metrics.get('financial_impact', {}).get('confusion_matrix', {'TN': 53, 'FP': 1, 'FN': 6, 'TP': 40})
        y_true_sim = [0]*cm['TN'] + [0]*cm['FP'] + [1]*cm['FN'] + [1]*cm['TP']
        y_pred_sim = [0]*cm['TN'] + [1]*cm['FP'] + [0]*cm['FN'] + [1]*cm['TP']
        
        fin_results = calculate_financial_impact(y_true_sim, y_pred_sim, cost_mat)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            render_metric_card("Model Strategy Cost", f"${fin_results['model_total_cost']:,.0f}", f"${fin_results['per_patient_model_cost']:,.0f} / patient", color="#2980B9")
        with col_m2:
            render_metric_card("Treat All Baseline Cost", f"${fin_results['treat_all_cost']:,.0f}", f"${fin_results['per_patient_treat_all_cost']:,.0f} / patient", color="#7F8C8D")
        with col_m3:
            render_metric_card("Treat None Baseline Cost", f"${fin_results['treat_none_cost']:,.0f}", f"${fin_results['per_patient_treat_none_cost']:,.0f} / patient", color="#C0392B")
            
        st.markdown("---")
        
        # Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(8, 4))
        strategies = ['Predictive Model', 'Treat Everyone', 'Treat No One']
        costs = [fin_results['model_total_cost'], fin_results['treat_all_cost'], fin_results['treat_none_cost']]
        colors = ['#27AE60', '#F39C12', '#C0392B']
        
        bars = ax.bar(strategies, costs, color=colors)
        ax.set_ylabel("Total Simulated Expenditure ($)")
        ax.set_title("Total Hospital Cost Comparison across Operational Policies")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5000, f"${yval:,.0f}", ha='center', va='bottom', fontweight='bold')
            
        st.pyplot(fig)
        plt.close()
        
        savings_none = fin_results['net_savings_vs_treat_none']
        st.success(f"🎉 **Financial Benefit Summary**: Deploying the predictive model saves **${savings_none:,.0f}** compared to no intervention strategy!")
