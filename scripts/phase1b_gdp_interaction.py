"""
Phase 1b: MVP Check - GDP Interaction Test
===========================================
测试 ICT × GDP 交互项（比制度质量更强的调节变量）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import statsmodels.api as sm
import os

DATA_DIR = 'data'
RESULTS_DIR = 'results'
INPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v3_imputed.csv')

def run_gdp_interaction_test():
    print("=" * 60)
    print("Phase 1b: GDP Interaction Test")
    print("=" * 60)
    
    df = pd.read_csv(INPUT_FILE)
    
    target = 'CO2_per_capita'
    treatment = 'ICT_exports'
    moderator = 'GDP_per_capita_constant'
    
    df = df.dropna(subset=[target, treatment, moderator])
    
    # Unit correction
    df[target] = df[target] / 100.0
    print(f"   CO2 mean: {df[target].mean():.2f} metric tons/capita")
    
    # Log GDP for better scaling
    df['log_GDP'] = np.log(df[moderator] + 1)
    
    # Center variables
    t_centered = df[treatment] - df[treatment].mean()
    m_centered = df['log_GDP'] - df['log_GDP'].mean()
    interaction = t_centered * m_centered
    
    # Simple regression with interaction
    X = pd.DataFrame({
        'ICT': t_centered,
        'log_GDP': m_centered, 
        'Interaction': interaction
    })
    X = sm.add_constant(X)
    y = df[target]
    
    model = sm.OLS(y, X).fit()
    
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"\n1. Main Effect (ICT → CO2):")
    print(f"   Coefficient: {model.params['ICT']:.6f}")
    print(f"   Std Error:   {model.bse['ICT']:.6f}")
    print(f"   P-value:     {model.pvalues['ICT']:.6f}")
    
    print(f"\n2. Interaction Effect (ICT × log(GDP) → CO2):")
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
        'coef_main': model.params['ICT'],
        'p_main': model.pvalues['ICT'],
        'coef_interaction': model.params['Interaction'],
        'p_interaction': model.pvalues['Interaction'],
        'n_obs': len(df)
    }
    pd.DataFrame([results]).to_csv(os.path.join(RESULTS_DIR, 'phase1b_gdp_results.csv'), index=False)
    print(f"\n💾 Results saved to: {RESULTS_DIR}/phase1b_gdp_results.csv")

if __name__ == "__main__":
    run_gdp_interaction_test()
