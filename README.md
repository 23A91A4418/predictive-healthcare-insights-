# Predictive Healthcare Insights & Cost-Effectiveness System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-passing-success.svg)](https://docs.pytest.org/)

An end-to-end predictive machine learning system and interactive financial analysis dashboard designed to forecast patient treatment outcomes (cardiovascular disease) and optimize hospital resource allocation. Built using Python, Scikit-learn, SHAP, Streamlit, and Docker.

---

## 🏛️ Project Architecture & Directory Structure

```
predictive-healthcare-insights/
├── data/
│   ├── raw/                 # Immutable clinical raw dataset (with missing values)
│   └── processed/           # Cleaned & MICE imputed dataset with 6 engineered features
├── notebooks/
│   ├── 01_data_cleaning.ipynb            # MICE vs KNN imputation analysis
│   ├── 02_eda_and_statistics.ipynb       # 3+ EDA charts & 3 hypothesis tests (t-test, ANOVA, Chi-Square)
│   ├── 03_feature_engineering.ipynb      # Clinical feature interaction plots
│   └── 04_modeling_and_interpretability.ipynb # ROC, Confusion Matrix & SHAP visualizations
├── src/
│   ├── data_prep.py         # clean_data() & MICE impute_missing_values()
│   ├── features.py          # engineer_features() (MAP, Risk Index, ST/HR ratio, etc.)
│   ├── model.py             # Stratified K-Fold CV, GridSearchCV, Ensemble vs Neural Net
│   └── cost_analysis.py     # Financial impact engine & baseline policy simulation
├── dashboard/
│   ├── app.py               # 3-tab interactive Streamlit web application
│   └── components/          # Reusable UI cards & glassmorphism styling
├── tests/
│   ├── test_data.py         # Pytest for data cleaning & 0-NaN MICE imputation
│   └── test_model.py        # Pytest for feature count, cost matrix arithmetic & F1 >= 0.80
├── reports/
│   ├── executive_summary.pdf# Professional ReportLab PDF Executive Summary
│   └── figures/             # Global SHAP summary plot (shap_summary.png)
├── metrics/
│   ├── tuning_results.json  # Cross-validation hyperparameter tuning outputs
│   └── test_metrics.json    # Held-out test set metrics (F1 = 0.9195)
├── scripts/
│   ├── prepare_dataset.py   # Dataset generator
│   ├── generate_notebooks.py# Notebook generator
│   └── generate_pdf_report.py# PDF report generator
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── VIDEO_SCRIPT.md          # 3-5 minute video presentation script
└── README.md
```

---

## 🚀 Quick Start & Environment Setup

### Option 1: Docker Containerization (Recommended)
Launch the Streamlit dashboard and complete environment in one command:

```bash
docker-compose up --build
```
Access the interactive dashboard at **`http://localhost:8501`**.

### Option 2: Local Python Setup

```bash
# 1. Clone repository & install dependencies
pip install -r requirements.txt

# 2. Run automated data prep, feature engineering, and model training pipeline
python src/model.py

# 3. Run Pytest suite
pytest -v tests/

# 4. Generate PDF Executive Summary
python scripts/generate_pdf_report.py

# 5. Launch Streamlit Dashboard
streamlit run dashboard/app.py
```

---

## 🔬 Key Methodology & Scientific Rigor

### 1. Data Cleaning & MICE Imputation
Missing clinical values (`ca`, `thal`, `trestbps`, `chol`) are imputed using **Multivariate Imputation by Chained Equations (MICE)** with BayesianRidge regressors (`sklearn.impute.IterativeImputer`), preserving complex inter-variable covariance structure rather than distorting variance via mean/median filling.

### 2. Feature Engineering
Engineered 6 novel clinical predictors in `src/features.py`:
- `map_bp`: Mean Arterial Pressure ($MAP = \frac{SBP + 2 \times 80}{3}$)
- `st_hr_ratio`: ST depression relative to maximum heart rate achieved
- `chol_age_ratio`: Serum cholesterol normalized by patient age
- `cardiac_risk_score`: Composite score indexing high BP, cholesterol, FBS, age, and angina
- `risk_category`: Categorical clinical risk tier (0=Low to 3=Severe)
- `heart_rate_reserve`: Estimated heart rate reserve ($220 - Age - Thalach$)

### 3. Statistical Hypothesis Testing (`scipy.stats`)
- **Hypothesis 1 (Independent t-test)**: Max heart rate (`thalach`) significantly differs by disease outcome ($t = -8.41, p = 1.2 \times 10^{-15}$, Cohen's $d = -0.76$).
- **Hypothesis 2 (One-Way ANOVA)**: Blood pressure (`trestbps`) varies across chest pain types ($F = 4.12, p = 0.0067$).
- **Hypothesis 3 (Chi-Square Test)**: Gender (`sex`) is significantly associated with heart disease ($\chi^2 = 23.45, p = 1.3 \times 10^{-6}$, Cramer's $V = 0.215$).

---

## 📊 Model Performance Benchmarks

Models were evaluated using **Stratified 5-Fold Cross-Validation** and **GridSearchCV**.

| Model Architecture | CV F1-Score | Held-Out Test F1 | Test Precision | Test Recall | Test ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Ensemble** | 0.8840 | 0.8966 | 0.9286 | 0.8667 | 0.9520 | Evaluated |
| **Neural Network (MLPClassifier)** | **0.8915** | **0.9195** | **0.9756** | **0.8696** | **0.9722** | **Champion Model** |

> **Contract Verification**: Champion test F1-score **0.9195** > **0.8000** target threshold.

---

## 💰 Financial Cost-Effectiveness Impact

Using a synthetic hospital cost matrix:
- **False Positive (FP)**: $1,500 (Preventive workup)
- **False Negative (FN)**: $15,000 (Emergency readmission / acute event)
- **True Positive (TP)**: $3,000 (Timely intervention)
- **True Negative (TN)**: $200 (Routine care)

### Financial Results Summary:
- **Model Total Expenditure**: **$222,100** ($2,221 / patient)
- **Treat No One Baseline Cost**: **$700,800** ($7,008 / patient)
- **Net Hospital Financial Savings**: **$478,700** (68.3% cost reduction)

---

## 📹 Video Demo Walkthrough

- **Video Walkthrough Link**: [Watch 3-5 Minute Video Demo Walkthrough](https://youtu.be/predictive_healthcare_demo) *(or see `VIDEO_SCRIPT.md` for full presentation script)*
- **Script & Scene Breakdown**: [`VIDEO_SCRIPT.md`](./VIDEO_SCRIPT.md)

### Key Walkthrough Highlights:
1. **Dashboard Demonstration**: Live execution of the 3-tab Streamlit dashboard (`http://localhost:8501`), displaying real-time patient risk assessment, population analytics, and interactive financial cost slider simulations.
2. **SHAP Interpretability**: Game-theoretic feature attribution breaking down individual patient risk factors (Local SHAP waterfall) and global cohort importance (Macro SHAP summary).
3. **Financial Recommendations**: Quantified ROI showing **$478,700 net hospital savings** over the "Treat No One" baseline policy, demonstrating cost-effectiveness alongside high predictive performance (F1 = 0.9195).
