import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ingests and cleans raw clinical dataset.
    - Replaces '?' or invalid strings with np.nan
    - Removes duplicate rows
    - Ensures proper numeric types for clinical variables
    """
    df_clean = df.copy()
    
    # Replace string missing markers like '?' with NaN
    df_clean = df_clean.replace('?', np.nan)
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    # Convert all columns to numeric, coercing invalid values to NaN
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    return df_clean


def impute_missing_values(df: pd.DataFrame, imputer_type: str = 'mice') -> pd.DataFrame:
    """
    Programmatically imputes missing clinical values using MICE (IterativeImputer)
    or KNNImputer. Simple mean/median filling across the entire dataset is avoided.
    """
    df_imp = df.copy()
    feature_cols = [c for c in df_imp.columns if c != 'target']
    
    if imputer_type.lower() == 'knn':
        imputer = KNNImputer(n_neighbors=5, weights='distance')
    else: # MICE by default
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=15,
            random_state=42,
            initial_strategy='mean'
        )
        
    # Fit and transform
    imputed_array = imputer.fit_transform(df_imp)
    df_imputed = pd.DataFrame(imputed_array, columns=df_imp.columns, index=df_imp.index)
    
    # Ensure discrete/categorical columns maintain integer logic where appropriate
    int_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in int_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].round().astype(int)
            
    if 'target' in df_imputed.columns:
        df_imputed['target'] = df_imputed['target'].round().astype(int)
        
    return df_imputed


def impute_clinical_data(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
    """
    Wrapper function matching clinical imputation contract.
    """
    return impute_missing_values(df, imputer_type='mice')


if __name__ == '__main__':
    raw_path = 'data/raw/heart_disease.csv'
    if pd.io.common.file_exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        df_cleaned = clean_data(df_raw)
        df_imputed = impute_missing_values(df_cleaned)
        print(f"Data Cleaning & Imputation Complete!")
        print(f"Cleaned shape: {df_cleaned.shape}, Imputed shape: {df_imputed.shape}")
        print(f"Remaining NaNs: {df_imputed.isnull().sum().sum()}")
