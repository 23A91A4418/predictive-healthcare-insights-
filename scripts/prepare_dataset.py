import os
import numpy as np
import pandas as pd

def generate_uci_heart_disease_dataset(n_samples=500, random_state=42):
    np.random.seed(random_state)
    
    # Target distribution: ~45% positive (heart disease present), 55% negative
    target = np.random.binomial(1, 0.45, size=n_samples)
    
    # Feature distributions conditioned on target for realistic clinical signals
    age = np.where(target == 1, 
                   np.random.normal(59, 7, size=n_samples), 
                   np.random.normal(52, 9, size=n_samples)).astype(int)
    age = np.clip(age, 29, 77)
    
    sex = np.random.binomial(1, 0.68, size=n_samples) # 1 = male, 0 = female
    
    # Chest pain type: 1 = typical angina, 2 = atypical, 3 = non-anginal, 4 = asymptomatic
    cp_probs_pos = [0.08, 0.12, 0.20, 0.60]
    cp_probs_neg = [0.25, 0.35, 0.25, 0.15]
    cp = np.array([np.random.choice([1, 2, 3, 4], p=cp_probs_pos if t == 1 else cp_probs_neg) for t in target])
    
    # Resting blood pressure (trestbps in mm Hg)
    trestbps = np.where(target == 1,
                        np.random.normal(136, 18, size=n_samples),
                        np.random.normal(128, 15, size=n_samples)).astype(int)
    trestbps = np.clip(trestbps, 94, 200)
    
    # Serum cholestoral in mg/dl
    chol = np.where(target == 1,
                    np.random.normal(258, 52, size=n_samples),
                    np.random.normal(235, 45, size=n_samples)).astype(int)
    chol = np.clip(chol, 126, 564)
    
    # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    fbs = np.random.binomial(1, np.where(target == 1, 0.22, 0.10))
    
    # Resting ECG results (0 = normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy)
    restecg = np.array([np.random.choice([0, 1, 2], p=[0.40, 0.15, 0.45] if t == 1 else [0.65, 0.10, 0.25]) for t in target])
    
    # Thalach: maximum heart rate achieved
    thalach = np.where(target == 1,
                       np.random.normal(138, 22, size=n_samples),
                       np.random.normal(160, 18, size=n_samples)).astype(int)
    thalach = np.clip(thalach, 71, 202)
    
    # Exercise induced angina (exang: 1 = yes, 0 = no)
    exang = np.random.binomial(1, np.where(target == 1, 0.55, 0.14))
    
    # Oldpeak: ST depression induced by exercise relative to rest
    oldpeak = np.where(target == 1,
                       np.random.exponential(1.4, size=n_samples),
                       np.random.exponential(0.4, size=n_samples))
    oldpeak = np.round(np.clip(oldpeak, 0.0, 6.2), 1)
    
    # Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)
    slope = np.array([np.random.choice([1, 2, 3], p=[0.20, 0.65, 0.15] if t == 1 else [0.60, 0.35, 0.05]) for t in target])
    
    # Ca: number of major vessels (0-3) colored by fluoroscopy
    ca = np.array([np.random.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.22, 0.13] if t == 1 else [0.75, 0.18, 0.05, 0.02]) for t in target])
    
    # Thal: 3 = normal, 6 = fixed defect, 7 = reversible defect
    thal = np.array([np.random.choice([3, 6, 7], p=[0.25, 0.15, 0.60] if t == 1 else [0.78, 0.07, 0.15]) for t in target])

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    })
    
    # Introduce realistic missing values (NaN) in ca, thal, trestbps, chol (~5-10% missing)
    nan_mask_ca = np.random.rand(n_samples) < 0.08
    nan_mask_thal = np.random.rand(n_samples) < 0.06
    nan_mask_chol = np.random.rand(n_samples) < 0.05
    nan_mask_bp = np.random.rand(n_samples) < 0.04
    
    df.loc[nan_mask_ca, 'ca'] = np.nan
    df.loc[nan_mask_thal, 'thal'] = np.nan
    df.loc[nan_mask_chol, 'chol'] = np.nan
    df.loc[nan_mask_bp, 'trestbps'] = np.nan
    
    # Introduce some duplicate rows to test clean_data duplicate removal
    duplicates = df.sample(n=5, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

if __name__ == '__main__':
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('dashboard/components', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('tests', exist_ok=True)
    
    df_raw = generate_uci_heart_disease_dataset(n_samples=500, random_state=42)
    output_path = 'data/raw/heart_disease.csv'
    df_raw.to_csv(output_path, index=False)
    print(f"Successfully generated raw clinical dataset at {output_path} with shape {df_raw.shape}")
    print(f"Missing counts per column:\n{df_raw.isnull().sum()}")
