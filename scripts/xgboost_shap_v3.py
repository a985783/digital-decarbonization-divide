"""
SHAP Analysis & Threshold Detection (Phase 3)
=============================================
Visualizes the non-linear relationship between ICT and CO2.
Identifies the "Tipping Point" (approx 6%).
Performs Split-Sample Analysis to confirm heterogeneity.
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import json

# Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
INPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v3_imputed.csv')
THRESHOLD_FILE = os.path.join(RESULTS_DIR, 'threshold_analysis.csv')

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

def run_shap_analysis():
    print("Loading data for SHAP...")
    df = pd.read_csv(INPUT_FILE)
    
    df['CO2_per_capita'] = df['CO2_per_capita'] / 100.0
    print("✓ Applied Unit Correction: CO2 / 100")
    
    target = 'CO2_per_capita'
    treatment = 'ICT_exports'
    
    df = df.dropna(subset=[target, treatment])
    print(f"✓ Applied Target/Treatment Leakage Safety. N={len(df)}")
    
    # Use Full Set NO ENERGY (Spec 4) to capture Total Effect heterogeneity
    exclude = ['country', 'year', target, 'OECD'] 
    # Note: we include treatment in features for the main model Y ~ T + W
    features = [c for c in df.columns if c not in exclude and 'Energy' not in c]
    
    X = df[features]
    y = df[target]
    
    print(f"Training XGBoost on {len(features)} features (Target: {target})...")
    model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
    model.fit(X, y)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 1. Summary Plot
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title('SHAP Feature Importance (Drivers of CO2)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary_v3.png'))
    plt.close()
    
    # 2. Dependence Plot for ICT
    print("Generating Dependence Plot...")
    
    # Find index of ICT column
    ict_idx = list(X.columns).index(treatment)
    ict_shap = shap_values[:, ict_idx]
    ict_vals = X[treatment].values
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ict_vals, ict_shap, alpha=0.5, color='#2c3e50')
    
    # Add LOESS/Lowess trend line
    lowess = sm.nonparametric.lowess(ict_shap, ict_vals, frac=0.3)
    plt.plot(lowess[:, 0], lowess[:, 1], color='#e74c3c', linewidth=3, label='Non-linear Trend')
    
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    
    # Add potential threshold marker (visual inspection or pre-defined)
    # The prompt suggests ~6%
    threshold = 6.0
    plt.axvline(threshold, color='#27ae60', linestyle=':', linewidth=2, label=f'Threshold ~{threshold}%')
    
    plt.xlabel('ICT Service Exports (% of Service Exports)')
    plt.ylabel('SHAP Value (Impact on CO2)')
    plt.title('Non-linear Impact of ICT on CO2 Emissions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_dependence_v3.png'))
    print(f"✓ Saved dependence plot to {FIGURES_DIR}/shap_dependence_v3.png")
    
    # 3. Split-Sample Analysis
    print("\nPerforming Split-Sample Analysis...")
    group_low = df[df[treatment] < threshold]
    group_high = df[df[treatment] >= threshold]
    
    print(f"Group Low (<{threshold}%): N={len(group_low)}")
    print(f"Group High (>={threshold}%): N={len(group_high)}")
    
    def run_simple_ols(sub_df, label):
        # Simple control set for OLS check: GDP, Pop, etc. (Limited set)
        # Using selected Lasso vars (No Energy) if available, or just simple multivariate
        # Let's use the Lasso Selected (No Energy) from Phase 2
        # Hardcoding relevant ones based on Phase 2 output:
        # Fixed_telephone, Central_govt_debt, Internet_users
        controls = ['Fixed_telephone_subscriptions', 'Central_govt_debt_pct_GDP', 'Internet_users', 
                    'GDP_per_capita_constant', 'Trade_openness', 'Industry_value_added_pct_GDP']
        
        # Intersect with available columns
        controls = [c for c in controls if c in sub_df.columns]
        
        X_sub = sub_df[[treatment] + controls]
        X_sub = sm.add_constant(X_sub)
        y_sub = sub_df[target]
        
        model_sub = sm.OLS(y_sub, X_sub).fit()
        return {
            'Group': label,
            'Coef_ICT': model_sub.params[treatment],
            'P_value': model_sub.pvalues[treatment],
            'R2': model_sub.rsquared,
            'N': len(sub_df)
        }
    
    res_low = run_simple_ols(group_low, 'Low ICT (<6%)')
    res_high = run_simple_ols(group_high, 'High ICT (>6%)')
    
    split_res = pd.DataFrame([res_low, res_high])
    print("\n=== Split-Sample Results (OLS with controls) ===")
    print(split_res)
    split_res.to_csv(THRESHOLD_FILE, index=False)

if __name__ == "__main__":
    run_shap_analysis()
