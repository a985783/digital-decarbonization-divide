"""
Phase 1: MVP Check - Interaction Term Verification
===================================================
验证ICT对CO2排放的异质性效应是否存在。
通过加入 ICT × Control_of_Corruption 交互项进行检验。
如果交互项系数显著 → 说明异质性存在，继续Phase 2。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import statsmodels.api as sm
import os

# Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
INPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v3_imputed.csv')

def dml_with_interaction(df, y_col, t_col, mod_col, w_cols, n_splits=5):
    """
    DML with Interaction Term.
    Model: Y = β₁T + β₂(T×M) + ε
    where T is treatment (ICT), M is moderator (Institution)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y = df[y_col].values
    t = df[t_col].values
    m = df[mod_col].values
    
    # Center variables for interpretable interaction
    t_centered = t - np.mean(t)
    m_centered = m - np.mean(m)
    
    # Create interaction term
    interaction = t_centered * m_centered
    
    W = df[w_cols].values
    
    y_res = np.zeros_like(y, dtype=float)
    t_res = np.zeros_like(t, dtype=float)
    int_res = np.zeros_like(interaction, dtype=float)
    
    print(f"  DML with {len(w_cols)} controls + interaction term...")
    
    for train_idx, test_idx in kf.split(W):
        W_train, W_test = W[train_idx], W[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t_train, t_test = t_centered[train_idx], t_centered[test_idx]
        int_train, int_test = interaction[train_idx], interaction[test_idx]
        
        # Model Y
        model_y = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                               n_jobs=-1, random_state=42, verbosity=0)
        model_y.fit(W_train, y_train)
        y_pred = model_y.predict(W_test)
        y_res[test_idx] = y_test - y_pred
        
        # Model T (centered)
        model_t = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                               n_jobs=-1, random_state=42, verbosity=0)
        model_t.fit(W_train, t_train)
        t_pred = model_t.predict(W_test)
        t_res[test_idx] = t_test - t_pred
        
        # Model Interaction
        model_int = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                                 n_jobs=-1, random_state=42, verbosity=0)
        model_int.fit(W_train, int_train)
        int_pred = model_int.predict(W_test)
        int_res[test_idx] = int_test - int_pred
    
    # Final stage: regress Y_res on [T_res, Interaction_res]
    X_final = np.column_stack([t_res, int_res])
    X_final = sm.add_constant(X_final)
    model_final = sm.OLS(y_res, X_final).fit()
    
    return {
        'coef_treatment': model_final.params[1],
        'coef_interaction': model_final.params[2],
        'se_treatment': model_final.bse[1],
        'se_interaction': model_final.bse[2],
        'p_treatment': model_final.pvalues[1],
        'p_interaction': model_final.pvalues[2],
        'n_obs': len(y)
    }


def run_mvp_check():
    print("=" * 60)
    print("Phase 1: MVP Check - Heterogeneity Verification")
    print("=" * 60)
    
    print("\n📂 Loading data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df)} observations, {len(df.columns)} variables")
    
    # Define variables
    target = 'CO2_per_capita'
    treatment = 'ICT_exports'
    moderator = 'Control_of_Corruption'
    
    # Check for missing values
    n_before = len(df)
    df = df.dropna(subset=[target, treatment, moderator])
    n_after = len(df)
    print(f"   Dropped {n_before - n_after} rows with missing core variables")
    print(f"   Analysis sample: N = {n_after}")
    
    # ⚠️ CRITICAL: Unit correction for CO2
    df[target] = df[target] / 100.0
    print(f"   ✓ Applied CO2 unit correction: divided by 100")
    print(f"   CO2 mean: {df[target].mean():.2f} metric tons/capita")
    
    # Define control variables (exclude core vars and identifiers)
    exclude = ['country', 'year', target, treatment, moderator, 'OECD']
    w_cols = [c for c in df.columns if c not in exclude and df[c].notna().mean() > 0.8]
    
    # Fill remaining NA in controls
    for col in w_cols:
        df[col] = df[col].fillna(df[col].median())
    
    print(f"   Using {len(w_cols)} control variables")
    
    # Run DML with interaction
    print("\n🔬 Running DML with Interaction Term...")
    results = dml_with_interaction(df, target, treatment, moderator, w_cols)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 MVP CHECK RESULTS")
    print("=" * 60)
    
    print(f"\n1. Main Effect (ICT → CO2):")
    print(f"   Coefficient: {results['coef_treatment']:.6f}")
    print(f"   Std Error:   {results['se_treatment']:.6f}")
    print(f"   P-value:     {results['p_treatment']:.4f}")
    
    print(f"\n2. Interaction Effect (ICT × Institution → CO2):")
    print(f"   Coefficient: {results['coef_interaction']:.6f}")
    print(f"   Std Error:   {results['se_interaction']:.6f}")
    print(f"   P-value:     {results['p_interaction']:.4f}")
    
    # Verdict
    print("\n" + "=" * 60)
    sig_threshold = 0.10
    if results['p_interaction'] < sig_threshold:
        print("🟢 GREEN LIGHT: Interaction is SIGNIFICANT!")
        print(f"   (p = {results['p_interaction']:.4f} < {sig_threshold})")
        print("   ➜ Heterogeneity exists. Proceed to Phase 2!")
        verdict = "PASS"
    else:
        print("🟡 YELLOW LIGHT: Interaction is NOT significant")
        print(f"   (p = {results['p_interaction']:.4f} >= {sig_threshold})")
        print("   ➜ May still try Phase 2, but manage expectations.")
        verdict = "MARGINAL"
    
    print("=" * 60)
    
    # Save results
    output_file = os.path.join(RESULTS_DIR, 'phase1_mvp_results.csv')
    pd.DataFrame([results]).to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    return verdict, results


if __name__ == "__main__":
    run_mvp_check()
