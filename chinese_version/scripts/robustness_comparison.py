"""
Robustness Check: Listwise Deletion vs MICE
===========================================
Compares the DML results (Total Effect specification) under:
1. MICE Imputation (N=960)
2. Listwise Deletion (N < 960)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import statsmodels.api as sm
import os

# Configuration
DATA_DIR = 'data'
RAW_FILE = os.path.join(DATA_DIR, 'wdi_expanded_raw.csv')
IMPUTED_FILE = os.path.join(DATA_DIR, 'clean_data_v3_imputed.csv')

def dml_estimation(df, y_col, t_col, w_cols, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = df[y_col].values
    t = df[t_col].values
    W = df[w_cols].values
    y_res = np.zeros_like(y)
    t_res = np.zeros_like(t)
    
    for train_idx, test_idx in kf.split(W):
        W_train, W_test = W[train_idx], W[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t_train, t_test = t[train_idx], t[test_idx]
        
        model_y = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42)
        model_y.fit(W_train, y_train)
        y_pred = model_y.predict(W_test)
        y_res[test_idx] = y_test - y_pred
        
        model_t = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42)
        model_t.fit(W_train, t_train)
        t_pred = model_t.predict(W_test)
        t_res[test_idx] = t_test - t_pred
        
    model_final = sm.OLS(y_res, t_res).fit()
    return model_final.params[0], model_final.pvalues[0], len(y)

def run_robustness():
    print("Loading data...")
    df_raw = pd.read_csv(RAW_FILE)
    df_mice = pd.read_csv(IMPUTED_FILE)
    
    df_raw['CO2_per_capita'] = df_raw['CO2_per_capita'] / 100.0
    df_mice['CO2_per_capita'] = df_mice['CO2_per_capita'] / 100.0
    
    target = 'CO2_per_capita'
    treatment = 'ICT_exports'
    
    df_mice = df_mice.dropna(subset=[target, treatment])
    df_raw = df_raw.dropna(subset=[target, treatment])
    
    exclude = ['country', 'year', target, 'OECD']
    
    all_cols = [c for c in df_mice.columns if c not in exclude + [treatment] and 'Energy' not in c]
    
    print("Running DML on MICE Data...")
    coef_mice, p_mice, n_mice = dml_estimation(df_mice, target, treatment, all_cols)
    
    cols_needed = [target, treatment, 'country', 'year'] + all_cols
    df_listwise = df_raw[cols_needed].dropna()
    
    print("Running DML on Listwise Deletion Data...")
    coef_lw, p_lw, n_lw = dml_estimation(df_listwise, target, treatment, all_cols)
    
    print("\n=== Robustness Check Results ===")
    print(f"{'Method':<20} {'Coef':<10} {'P-value':<10} {'N':<5}")
    print("-" * 50)
    print(f"{'MICE Imputation':<20} {coef_mice:<10.4f} {p_mice:<10.4f} {n_mice:<5}")
    print(f"{'Listwise Deletion':<20} {coef_lw:<10.4f} {p_lw:<10.4f} {n_lw:<5}")

if __name__ == "__main__":
    run_robustness()
