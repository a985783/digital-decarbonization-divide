"""
Phase 1: MVP Check - Interaction Term Verification
===================================================
验证DCI对CO2排放的异质性效应是否存在。
通过加入 DCI × Control_of_Corruption 交互项进行检验。
如果交互项系数显著 → 说明异质性存在，继续Phase 2。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
import statsmodels.api as sm
import os

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

# Configuration
DATA_DIR = "data"
RESULTS_DIR = "results"
INPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")

def dml_with_interaction(y, t, m, w, groups, n_splits=5):
    """
    DML with Interaction Term.
    Model: Y = β₁T + β₂(T×M) + ε
    where T is treatment (ICT), M is moderator (Institution)
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    # Center variables for interpretable interaction
    t_centered = t - np.mean(t)
    m_centered = m - np.mean(m)
    
    # Create interaction term
    interaction = t_centered * m_centered
    
    y_res = np.zeros_like(y, dtype=float)
    t_res = np.zeros_like(t, dtype=float)
    int_res = np.zeros_like(interaction, dtype=float)
    
    print(f"  DML with {w.shape[1]} controls + interaction term...")
    
    for train_idx, test_idx in gkf.split(w, groups=groups):
        w_train, w_test = w[train_idx], w[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t_train, t_test = t_centered[train_idx], t_centered[test_idx]
        int_train, int_test = interaction[train_idx], interaction[test_idx]
        
        # Model Y
        model_y = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                               n_jobs=-1, random_state=42, verbosity=0)
        model_y.fit(w_train, y_train)
        y_pred = model_y.predict(w_test)
        y_res[test_idx] = y_test - y_pred
        
        # Model T (centered)
        model_t = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                               n_jobs=-1, random_state=42, verbosity=0)
        model_t.fit(w_train, t_train)
        t_pred = model_t.predict(w_test)
        t_res[test_idx] = t_test - t_pred
        
        # Model Interaction
        model_int = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                                 n_jobs=-1, random_state=42, verbosity=0)
        model_int.fit(w_train, int_train)
        int_pred = model_int.predict(w_test)
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
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df)} observations, {len(df.columns)} variables")
    
    target = cfg["outcome"]
    treatment = cfg["treatment_main"]
    moderator = "Control_of_Corruption"

    df = df.dropna(subset=[target, moderator])
    y, t, x, w, df = prepare_analysis_data(df, cfg, return_df=True)
    m = df[moderator].values
    groups = df[cfg["groups"]].values
    mask = ~np.isnan(y) & ~np.isnan(t) & ~np.isnan(m)
    y, t, m, w, groups = y[mask], t[mask], m[mask], w[mask], groups[mask]
    print(f"   Analysis sample: N = {len(y)}")
    
    # Run DML with interaction
    print("\n🔬 Running DML with Interaction Term...")
    results = dml_with_interaction(
        y,
        t,
        m,
        w,
        groups,
        n_splits=cfg["cv"]["n_splits"],
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 MVP CHECK RESULTS")
    print("=" * 60)
    
    print(f"\n1. Main Effect (DCI → CO2):")
    print(f"   Coefficient: {results['coef_treatment']:.6f}")
    print(f"   Std Error:   {results['se_treatment']:.6f}")
    print(f"   P-value:     {results['p_treatment']:.4f}")
    
    print(f"\n2. Interaction Effect (DCI × Institution → CO2):")
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
