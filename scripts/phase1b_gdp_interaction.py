"""
Phase 1b: MVP Check - GDP Interaction Test
===========================================
测试 DCI × GDP 交互项（比制度质量更强的调节变量）
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

DATA_DIR = "data"
RESULTS_DIR = "results"
INPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")

def run_gdp_interaction_test():
    print("=" * 60)
    print("Phase 1b: GDP Interaction Test")
    print("=" * 60)
    
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    
    target = cfg["outcome"]
    treatment = cfg["treatment_main"]
    moderator = "GDP_per_capita_constant"
    
    df = df.dropna(subset=[target, moderator])
    _, _, _, _, df = prepare_analysis_data(df, cfg, return_df=True)
    
    print(f"   CO2 mean: {df[target].mean():.2f} metric tons/capita")
    
    # Log GDP for better scaling
    df['log_GDP'] = np.log(df[moderator] + 1)
    
    # Center variables
    t_centered = df[treatment] - df[treatment].mean()
    m_centered = df['log_GDP'] - df['log_GDP'].mean()
    interaction = t_centered * m_centered
    
    # Simple regression with interaction
    X = pd.DataFrame({
        'DCI': t_centered,
        'log_GDP': m_centered, 
        'Interaction': interaction
    })
    X = sm.add_constant(X)
    y = df[target]
    
    model = sm.OLS(y, X).fit()
    
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"\n1. Main Effect (DCI → CO2):")
    print(f"   Coefficient: {model.params['DCI']:.6f}")
    print(f"   Std Error:   {model.bse['DCI']:.6f}")
    print(f"   P-value:     {model.pvalues['DCI']:.6f}")
    
    print(f"\n2. Interaction Effect (DCI × log(GDP) → CO2):")
    print(f"   Coefficient: {model.params['Interaction']:.6f}")
    print(f"   Std Error:   {model.bse['Interaction']:.6f}")
    print(f"   P-value:     {model.pvalues['Interaction']:.6f}")
    
    print("\n" + "=" * 60)
    if model.pvalues['Interaction'] < 0.01:
        print("🟢 GREEN LIGHT: GDP Interaction is HIGHLY SIGNIFICANT!")
        print(f"   (p = {model.pvalues['Interaction']:.6f} < 0.01)")
        print("   ➜ Strong evidence of heterogeneity!")
    elif model.pvalues['Interaction'] < 0.05:
        print("🟢 GREEN LIGHT: GDP Interaction is SIGNIFICANT!")
        print(f"   (p = {model.pvalues['Interaction']:.6f} < 0.05)")
    else:
        print("🟡 Interaction not significant")
    print("=" * 60)
    
    # Save results
    results = {
        'moderator': 'log_GDP',
        'coef_main': model.params['DCI'],
        'se_main': model.bse['DCI'],
        'p_main': model.pvalues['DCI'],
        'coef_interaction': model.params['Interaction'],
        'se_interaction': model.bse['Interaction'],
        'p_interaction': model.pvalues['Interaction'],
        'n_obs': len(df)
    }
    pd.DataFrame([results]).to_csv(os.path.join(RESULTS_DIR, 'phase1b_gdp_results.csv'), index=False)
    print(f"\n💾 Results saved to: {RESULTS_DIR}/phase1b_gdp_results.csv")

if __name__ == "__main__":
    run_gdp_interaction_test()
