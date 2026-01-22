"""
Mechanism Consistency Check (Phase 3.5)
=======================================
Tests the plausibility of the Energy Efficiency channel.
Runs two auxiliary DML models:
1. ICT -> Energy Use (Is ICT associated with lower energy intensity?)
2. Energy Use -> CO2 (Is Energy Use a strong driver of emissions?)

Method: Leakage-Safe Panel DML (Same as main analysis)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import statsmodels.api as sm
import os
import json

# Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
INPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v3_imputed.csv')
SELECTED_VARS_FILE = os.path.join(DATA_DIR, 'lasso_selected_features.json')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'mechanism_results.csv')

def dml_estimation(df, y_col, t_col, w_cols, n_splits=5):
    # Strict Complete Case for Y and T
    df_clean = df.dropna(subset=[y_col, t_col]).copy()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = df_clean[y_col].values
    t = df_clean[t_col].values
    W = df_clean[w_cols].values
    
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
    return {
        'coef': model_final.params[0],
        'std_err': model_final.bse[0],
        'p_value': model_final.pvalues[0],
        'n_obs': len(y)
    }

def run_mechanism_check():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # --- P0 FIX: Unit Correction ---
    df['CO2_per_capita'] = df['CO2_per_capita'] / 100.0
    print("✓ Applied Unit Correction: CO2 / 100")
    
    # Define Variables
    y_main = 'CO2_per_capita'
    t_main = 'ICT_exports'
    mediator = 'Energy_use_per_capita'
    
    # Load Controls (Full High-Dim Set, excluding metadata)
    exclude = ['country', 'year', 'OECD', y_main, t_main, mediator]
    # We must exclude Y, T, and Mediator from W to avoid leakage/endogeneity in the check
    controls = [c for c in df.columns if c not in exclude]
    
    results = []
    
    print("\n--- Path A: ICT -> Energy Use ---")
    # Does ICT reduce Energy Intensity?
    # Y = Energy, T = ICT, W = Controls
    res_a = dml_estimation(df, y_col=mediator, t_col=t_main, w_cols=controls)
    res_a['path'] = 'ICT -> Energy Use'
    results.append(res_a)
    print(f"Coef: {res_a['coef']:.4f}, P-val: {res_a['p_value']:.4f}")
    
    print("\n--- Path B: Energy Use -> CO2 ---")
    # Does Energy drive Emissions? (Sanity Check)
    # Y = CO2, T = Energy, W = Controls + ICT (controlling for ICT to see partial effect of Energy)
    # Note: Adding ICT to controls here
    controls_b = controls + [t_main]
    res_b = dml_estimation(df, y_col=y_main, t_col=mediator, w_cols=controls_b)
    res_b['path'] = 'Energy Use -> CO2'
    results.append(res_b)
    print(f"Coef: {res_b['coef']:.6f}, P-val: {res_b['p_value']:.4f}")
    
    # Save
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved mechanism results to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_mechanism_check()
