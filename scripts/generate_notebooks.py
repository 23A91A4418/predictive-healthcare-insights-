import os
import json

def create_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    filepath = os.path.join("notebooks", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")

def code_cell(source_code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.splitlines(keepends=True)
    }

def markdown_cell(markdown_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": markdown_text.splitlines(keepends=True)
    }

# ----------------------------------------------------
# Notebook 01: Data Cleaning & Imputation
# ----------------------------------------------------
cells_01 = [
    markdown_cell("# Phase 1: Data Cleaning and Advanced MICE Imputation\n\nThis notebook demonstrates data cleaning, duplicate removal, and multivariate missing value imputation using MICE (IterativeImputer)."),
    code_cell("""import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))
from src.data_prep import clean_data, impute_missing_values

# Load raw dataset
df_raw = pd.read_csv('../data/raw/heart_disease.csv')
print("Raw Dataset Shape:", df_raw.shape)
print("Missing values per column:\\n", df_raw.isnull().sum())
"""),
    markdown_cell("## Step 1: Clean Data & Drop Duplicates"),
    code_cell("""df_cleaned = clean_data(df_raw)
print("Cleaned Dataset Shape:", df_cleaned.shape)
"""),
    markdown_cell("## Step 2: Advanced Imputation via MICE (IterativeImputer) vs KNN"),
    code_cell("""# Apply MICE Imputation
df_imputed_mice = impute_missing_values(df_cleaned, imputer_type='mice')
print("MICE Imputed Shape:", df_imputed_mice.shape)
print("Remaining NaNs after MICE:", df_imputed_mice.isnull().sum().sum())

# Apply KNN Imputation for comparison
df_imputed_knn = impute_missing_values(df_cleaned, imputer_type='knn')
print("KNN Imputed Shape:", df_imputed_knn.shape)
print("Remaining NaNs after KNN:", df_imputed_knn.isnull().sum().sum())
""")
]

# ----------------------------------------------------
# Notebook 02: EDA & Statistical Hypothesis Testing
# ----------------------------------------------------
cells_02 = [
    markdown_cell("# Phase 2: Exploratory Data Analysis & Statistical Hypothesis Testing\n\nConducting comprehensive univariate, bivariate, and correlation EDA alongside 3 formal statistical hypothesis tests."),
    code_cell("""import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.abspath('..'))
from src.data_prep import clean_data, impute_missing_values

# Load cleaned and imputed dataset
df_raw = pd.read_csv('../data/raw/heart_disease.csv')
df = impute_missing_values(clean_data(df_raw))
print("Dataset shape:", df.shape)
"""),
    markdown_cell("## Section 1: Exploratory Data Analysis (3 Visualization Types)\n\n### 1. Univariate Feature Distributions & Target Class Balance"),
    code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Target distribution
sns.countplot(x='target', data=df, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title("Target Class Distribution (0=Healthy, 1=Disease)")

# Plot 2: Age distribution by Target
sns.histplot(data=df, x='age', hue='target', kde=True, ax=axes[0, 1], palette='crest')
axes[0, 1].set_title("Age Distribution by Patient Outcome")

# Plot 3: Max Heart Rate (thalach) distribution
sns.kdeplot(data=df, x='thalach', hue='target', fill=True, ax=axes[1, 0], palette='flare')
axes[1, 0].set_title("Max Heart Rate (Thalach) Distribution")

# Plot 4: ST Depression (oldpeak) distribution
sns.boxplot(x='target', y='oldpeak', data=df, ax=axes[1, 1], palette='magma')
axes[1, 1].set_title("ST Depression (Oldpeak) by Patient Outcome")

plt.tight_layout()
plt.show()
"""),
    markdown_cell("### 2. Bivariate Feature Relationships & Boxplots"),
    code_cell("""plt.figure(figsize=(12, 6))
sns.boxplot(x='cp', y='thalach', hue='target', data=df, palette='Set2')
plt.title("Max Heart Rate across Chest Pain Types & Disease Status")
plt.xlabel("Chest Pain Type (1: Typical, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic)")
plt.ylabel("Max Heart Rate (thalach)")
plt.show()
"""),
    markdown_cell("### 3. Correlation Heatmap"),
    code_cell("""plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Clinical Feature Correlation Matrix")
plt.show()
"""),
    markdown_cell("## Section 2: Statistical Hypothesis Testing (3 Hypotheses)\n\n### Hypothesis 1: Maximum Heart Rate (`thalach`) difference between diseased and healthy patients\n- **H0**: There is no difference in average max heart rate between patients with heart disease vs without.\n- **H1**: Patients with heart disease have significantly lower max heart rate."),
    code_cell("""group_healthy = df[df['target'] == 0]['thalach']
group_diseased = df[df['target'] == 1]['thalach']

# 1. Assumption Check: Shapiro-Wilk test for normality
sw_healthy = stats.shapiro(group_healthy)
sw_diseased = stats.shapiro(group_diseased)
print(f"Shapiro-Wilk Test (Healthy): p-value = {sw_healthy.pvalue:.4f}")
print(f"Shapiro-Wilk Test (Diseased): p-value = {sw_diseased.pvalue:.4f}")

# 2. Assumption Check: Levene's test for equal variance
lev_res = stats.levene(group_healthy, group_diseased)
print(f"Levene Variance Test: p-value = {lev_res.pvalue:.4f}")

# 3. Execution: Two-sample independent t-test (and Mann-Whitney U test fallback)
ttest_res = stats.ttest_ind(group_healthy, group_diseased)
mwu_res = stats.mannwhitneyu(group_healthy, group_diseased)

# Cohen's d Effect Size calculation
mean_diff = np.mean(group_healthy) - np.mean(group_diseased)
pooled_std = np.sqrt((np.std(group_healthy)**2 + np.std(group_diseased)**2) / 2)
cohens_d = mean_diff / pooled_std

print("\\n--- HYPOTHESIS 1 RESULTS ---")
print(f"t-statistic: {ttest_res.statistic:.4f}, p-value: {ttest_res.pvalue:.4e}")
print(f"Mann-Whitney U statistic: {mwu_res.statistic:.4f}, p-value: {mwu_res.pvalue:.4e}")
print(f"Effect Size (Cohen's d): {cohens_d:.4f}")
"""),
    markdown_cell("### Hypothesis 2: Resting Blood Pressure (`trestbps`) variation across Chest Pain Types (`cp`)\n- **H0**: Average blood pressure is equal across all 4 chest pain categories.\n- **H1**: Blood pressure varies significantly by chest pain severity."),
    code_cell("""cp_groups = [group['trestbps'].values for name, group in df.groupby('cp')]
anova_res = stats.f_oneway(*cp_groups)
kw_res = stats.kruskal(*cp_groups)

print("--- HYPOTHESIS 2 RESULTS ---")
print(f"One-Way ANOVA F-statistic: {anova_res.statistic:.4f}, p-value: {anova_res.pvalue:.4f}")
print(f"Kruskal-Wallis H-statistic: {kw_res.statistic:.4f}, p-value: {kw_res.pvalue:.4f}")
"""),
    markdown_cell("### Hypothesis 3: Association between Patient Sex (`sex`) and Disease Status (`target`)\n- **H0**: Gender and heart disease occurrence are independent.\n- **H1**: Gender is significantly associated with heart disease presence."),
    code_cell("""contingency_table = pd.crosstab(df['sex'], df['target'])
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

# Cramer's V effect size for categorical association
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2_stat / (n * min_dim))

print("--- HYPOTHESIS 3 RESULTS ---")
print(f"Contingency Table:\\n{contingency_table}")
print(f"Chi-Squared Statistic: {chi2_stat:.4f}, p-value: {p_val:.4e}, Degrees of Freedom: {dof}")
print(f"Effect Size (Cramer's V): {cramers_v:.4f}")
""")
]

# ----------------------------------------------------
# Notebook 03: Feature Engineering
# ----------------------------------------------------
cells_03 = [
    markdown_cell("# Phase 3: Clinical Feature Engineering\n\nMathematically deriving 5+ novel features to boost predictive signal."),
    code_cell("""import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('..'))
from src.data_prep import clean_data, impute_missing_values
from src.features import engineer_features

# Load & clean
df_raw = pd.read_csv('../data/raw/heart_disease.csv')
df_clean = impute_missing_values(clean_data(df_raw))

# Engineer features
df_eng = engineer_features(df_clean)
print("Original Columns:", df_clean.shape[1])
print("Engineered Columns:", df_eng.shape[1])

new_features = list(set(df_eng.columns) - set(df_clean.columns))
print("Newly Created Features:", new_features)
"""),
    markdown_cell("## Relationship of Engineered Features with Heart Disease Target"),
    code_cell("""fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(new_features):
    if df_eng[col].nunique() <= 5:
        sns.countplot(x=col, hue='target', data=df_eng, ax=axes[idx], palette='Set1')
    else:
        sns.boxplot(x='target', y=col, data=df_eng, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f"{col} vs Target")

plt.tight_layout()
plt.show()
""")
]

# ----------------------------------------------------
# Notebook 04: Modeling & SHAP Interpretability
# ----------------------------------------------------
cells_04 = [
    markdown_cell("# Phase 4: Predictive Modeling & SHAP Interpretability\n\nTraining Ensemble & Neural Network models, selecting Champion model, and evaluating SHAP explanations."),
    code_cell("""import sys
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('..'))
from src.model import run_training_pipeline

# Run complete training pipeline
metrics = run_training_pipeline('../data/raw/heart_disease.csv')
print("Model Pipeline Execution Summary:", metrics)
"""),
    markdown_cell("## View Global SHAP Importance Plot"),
    code_cell("""from IPython.display import Image
Image('../reports/figures/shap_summary.png')
""")
]

if __name__ == '__main__':
    os.makedirs('notebooks', exist_ok=True)
    create_notebook(cells_01, "01_data_cleaning.ipynb")
    create_notebook(cells_02, "02_eda_and_statistics.ipynb")
    create_notebook(cells_03, "03_feature_engineering.ipynb")
    create_notebook(cells_04, "04_modeling_and_interpretability.ipynb")
