import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mathematically engineers clinical features from raw/imputed variables.
    
    Creates at least 5 novel features:
    1. map_bp: Mean Arterial Pressure (estimated from resting blood pressure)
    2. st_hr_ratio: Ratio of exercise ST depression to maximum heart rate
    3. chol_age_ratio: Ratio of serum cholesterol to patient age
    4. cardiac_risk_score: Composite clinical risk index
    5. risk_category: Categorical risk tier (0=Low, 1=Moderate, 2=High, 3=Severe)
    6. heart_rate_reserve: Estimated heart rate reserve (220 - age - thalach)
    """
    df_feat = df.copy()
    
    # 1. Mean Arterial Pressure (MAP) in mmHg
    # Formula: MAP ≈ (2 * Diastolic BP + Systolic BP) / 3. 
    # Standard approximation assuming diastolic ~ 80 mmHg baseline or ratio:
    df_feat['map_bp'] = (df_feat['trestbps'] + 2 * 80.0) / 3.0
    
    # 2. ST depression to max heart rate ratio
    df_feat['st_hr_ratio'] = df_feat['oldpeak'] / (df_feat['thalach'] + 1e-5)
    
    # 3. Cholesterol to age ratio
    df_feat['chol_age_ratio'] = df_feat['chol'] / (df_feat['age'] + 1e-5)
    
    # 4. Composite Cardiac Risk Index (0 to 4 score)
    high_bp = (df_feat['trestbps'] > 130).astype(int)
    high_chol = (df_feat['chol'] > 200).astype(int)
    high_fbs = (df_feat['fbs'] == 1).astype(int)
    elderly = (df_feat['age'] > 55).astype(int)
    high_exercise_angina = (df_feat['exang'] == 1).astype(int)
    
    df_feat['cardiac_risk_score'] = high_bp + high_chol + high_fbs + elderly + high_exercise_angina
    
    # 5. Risk Category (0 = Low, 1 = Moderate, 2 = High, 3 = Severe)
    conditions = [
        (df_feat['cardiac_risk_score'] <= 1),
        (df_feat['cardiac_risk_score'] == 2),
        (df_feat['cardiac_risk_score'] == 3),
        (df_feat['cardiac_risk_score'] >= 4)
    ]
    choices = [0, 1, 2, 3]
    df_feat['risk_category'] = np.select(conditions, choices, default=1)
    
    # 6. Heart Rate Reserve Estimate (220 - Age - Max Heart Rate)
    df_feat['heart_rate_reserve'] = 220 - df_feat['age'] - df_feat['thalach']
    
    return df_feat


if __name__ == '__main__':
    from src.data_prep import clean_data, impute_missing_values
    df_raw = pd.read_csv('data/raw/heart_disease.csv')
    df_clean = clean_data(df_raw)
    df_imp = impute_missing_values(df_clean)
    df_eng = engineer_features(df_imp)
    
    new_cols = set(df_eng.columns) - set(df_imp.columns)
    print(f"Feature engineering complete! Engineered {len(new_cols)} new features:")
    print(new_cols)
