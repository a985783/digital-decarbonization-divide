"""
Double Machine Learning (DML) Analysis (Phase 2)
================================================
Estimates the causal effect of ICT exports on CO2 emissions.
Uses Lasso-selected controls and High-Dimensional Full Set.
Implements Leakage-Safe Cross-Fitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import os
import json

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

# Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
INPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v4_imputed.csv')
SELECTED_VARS_FILE = os.path.join(DATA_DIR, 'lasso_selected_features.json')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'dml_results_v3.csv')

def dml_estimation(y, t, w, groups, n_splits=5):
    """
    Performs Double Machine Learning with Cross-Fitting.
    Model Y: Y ~ W (XGBoost)
    Model T: T ~ W (XGBoost)
    Final: Y_res ~ T_res (OLS)
    """
    y_res = np.zeros_like(y)
    t_res = np.zeros_like(t)
    
    print(f"  DML with {w.shape[1]} controls...")
    
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(w, groups=groups):
        W_train, W_test = w[train_idx], w[test_idx]
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
    
    return {
        'coef': model_final.params[0],
        'std_err': model_final.bse[0],
        'p_value': model_final.pvalues[0],
        'conf_lower': model_final.conf_int()[0][0],
        'conf_upper': model_final.conf_int()[0][1],
        'n_obs': len(y)
    }

def run_analysis():
    print("Loading data...")
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    
    target = cfg["outcome"]
    treatment = cfg["treatment_main"]
    
    n_before = len(df)
    df = df.dropna(subset=[target])
    y, t, x, w, df = prepare_analysis_data(df, cfg, return_df=True)
    groups = df[cfg["groups"]].values
    mask = ~np.isnan(y) & ~np.isnan(t)
    y, t, w, groups = y[mask], t[mask], w[mask], groups[mask]
    print(f"✓ Applied Target/Treatment Leakage Safety: Dropped {n_before - len(y)} rows with missing Y/T.")
    print(f"  Analysis Sample N={len(y)}")
    
    with open(SELECTED_VARS_FILE, 'r') as f:
        lasso_vars = json.load(f)
    
    all_vars = cfg["controls_W"]
    
    results = []
    
    print("\n--- Spec 1: Lasso Selected (Baseline) ---")
    res1 = dml_estimation(y, t, df[lasso_vars].values, groups)
    res1['spec'] = 'Lasso Selected'
    results.append(res1)
    
    lasso_no_energy = [v for v in lasso_vars if 'Energy' not in v] 
    print("\n--- Spec 2: Lasso (No Energy) [Total Effect?] ---")
    if lasso_no_energy:
        res2 = dml_estimation(y, t, df[lasso_no_energy].values, groups)
        res2['spec'] = 'Lasso (No Energy)'
        results.append(res2)
    else:
        print("Skipping Spec 2 (No variables left)")

    print("\n--- Spec 3: Full High-Dim (All Controls) [Direct Effect] ---")
    res3 = dml_estimation(y, t, df[all_vars].values, groups)
    res3['spec'] = 'Full High-Dimensional'
    results.append(res3)

    all_no_energy = [v for v in all_vars if 'Energy' not in v]
    print("\n--- Spec 4: Full High-Dim (No Energy) [Total Effect] ---")
    res4 = dml_estimation(y, t, df[all_no_energy].values, groups)
    res4['spec'] = 'Full High-Dimensional (No Energy)'
    results.append(res4)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n=== DML Results (Corrected Units) ===")
    print(results_df[['spec', 'coef', 'std_err', 'p_value']])

if __name__ == "__main__":
    run_analysis()
