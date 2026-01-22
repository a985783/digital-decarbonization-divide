"""
Lasso Variable Selection (Phase 2)
==================================
Selects the most predictive control variables from the high-dimensional set.
Uses Lasso with Time-Series Cross-Validation.
Generates a Coefficient Path Plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import json

# Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
INPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v3_imputed.csv')
SELECTED_VARS_FILE = os.path.join(DATA_DIR, 'lasso_selected_features.json')

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

def run_lasso_selection():
    print("Loading data for Lasso...")
    df = pd.read_csv(INPUT_FILE)
    
    # --- P0 FIX: Unit Correction ---
    if 'CO2_per_capita' in df.columns:
        df['CO2_per_capita'] = df['CO2_per_capita'] / 100.0
        print("✓ Applied Unit Correction: CO2 / 100")
    # -------------------------------
    
    target = 'CO2_per_capita'
    treatment = 'ICT_exports'
    
    exclude = ['country', 'year', target, treatment, 'OECD']
    features = [c for c in df.columns if c not in exclude]
    
    print(f"Candidate Features: {len(features)}")
    
    X = df[features]
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Running LassoCV...")
    lasso = LassoCV(cv=tscv, random_state=42, max_iter=10000, n_jobs=-1)
    lasso.fit(X_scaled, y)
    
    print(f"Best Alpha: {lasso.alpha_:.6f}")
    
    coefs = pd.Series(lasso.coef_, index=features)
    selected = coefs[coefs != 0].sort_values(ascending=False)
    
    print(f"\nSelected {len(selected)} variables (Non-zero coefficients):")
    print(selected)
    
    selected_vars = selected.index.tolist()
    with open(SELECTED_VARS_FILE, 'w') as f:
        json.dump(selected_vars, f)
    
    print("\nGenerating Lasso Path Plot...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X_scaled, y, eps=0.001, n_alphas=100)
    
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", len(features))
    
    for i, feature in enumerate(features):
        plt.plot(np.log10(alphas_lasso), coefs_lasso[i], label=feature if feature in selected_vars else "", 
                 color='grey' if feature not in selected_vars else None,
                 alpha=0.3 if feature not in selected_vars else 1.0,
                 linewidth=1 if feature not in selected_vars else 2)
        
    plt.axvline(np.log10(lasso.alpha_), linestyle='--', color='k', label='Selected Alpha')
    
    plt.xlabel('Log10(Alpha)')
    plt.ylabel('Coefficient')
    plt.title('Lasso Coefficient Path (Variable Selection)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()
    
    save_fig = os.path.join(FIGURES_DIR, 'lasso_path_v3.png')
    plt.savefig(save_fig, dpi=300)
    print(f"✓ Saved plot to {save_fig}")

if __name__ == "__main__":
    run_lasso_selection()
