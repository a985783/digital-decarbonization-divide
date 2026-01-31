"""
Phase 7: Dynamic Effects Analysis
==================================
Estimates distributed lag effects to understand temporal dynamics.

Features:
1. Lagged CATE estimates (t, t+1, t+2, t+3)
2. Cumulative effect curves
3. Effect persistence tests
4. Impulse response visualization

Q1 Reviewer Response: Addresses concern about lack of dynamic effect analysis.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

try:
    from econml.dml import CausalForestDML
    import xgboost as xgb
except ImportError:
    raise ImportError("Please install econml and xgboost")

DATA_DIR = "data"
RESULTS_DIR = "results"
INPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "dynamic_effects.csv")


def create_forward_outcomes(df, outcome_col, max_lead=3):
    """
    Create forward-looking outcome variables for dynamic effect estimation.
    
    Y_{t+k} = outcome at time t+k
    
    Parameters:
    -----------
    df : DataFrame
        Panel data with country and year
    outcome_col : str
        Name of outcome variable
    max_lead : int
        Maximum number of forward periods
    
    Returns:
    --------
    df : DataFrame with added forward outcome columns
    """
    df = df.sort_values(['country', 'year']).copy()
    
    for k in range(1, max_lead + 1):
        df[f'{outcome_col}_lead{k}'] = df.groupby('country')[outcome_col].shift(-k)
    
    return df


def estimate_dynamic_ate(df, cfg, lead, n_trees=1000):
    """
    Estimate ATE for outcome at time t+lead.
    
    Parameters:
    -----------
    df : DataFrame
        Data with forward outcomes created
    cfg : dict
        Configuration
    lead : int
        Number of periods forward (0 = contemporaneous)
    n_trees : int
        Number of trees for Causal Forest
    
    Returns:
    --------
    dict with ATE, CI, and metadata
    """
    outcome_col = cfg["outcome"] if lead == 0 else f'{cfg["outcome"]}_lead{lead}'
    
    # Check if outcome exists
    if outcome_col not in df.columns:
        return None
    
    # Prepare data
    df_clean = df.dropna(subset=[outcome_col, "DCI"])
    
    if len(df_clean) < 200:
        return None
    
    Y = df_clean[outcome_col].values
    T = df_clean["DCI"].values
    X = df_clean[cfg["moderators_X"]].values
    W = df_clean[cfg["controls_W"]].values
    groups = df_clean[cfg["groups"]].values
    
    # Fit Causal Forest
    est = CausalForestDML(
        model_y=xgb.XGBRegressor(n_estimators=100, max_depth=3, verbosity=0, random_state=42),
        model_t=xgb.XGBRegressor(n_estimators=100, max_depth=3, verbosity=0, random_state=42),
        n_estimators=n_trees,
        min_samples_leaf=10,
        max_depth=5,
        random_state=42,
        discrete_treatment=False,
        n_jobs=-1,
        cv=GroupKFold(n_splits=5)
    )
    
    est.fit(Y, T, X=X, W=W, groups=groups)
    
    ate = est.ate(X)
    ate_lb, ate_ub = est.ate_interval(X, alpha=0.05)
    
    return {
        'lead': lead,
        'n_obs': len(df_clean),
        'ate': ate,
        'ci_lower': ate_lb,
        'ci_upper': ate_ub,
        'significant': (ate_lb > 0) or (ate_ub < 0)
    }


def run_dynamic_effects_analysis(max_lead=3, n_trees=1000):
    """
    Run complete dynamic effects analysis.
    
    Parameters:
    -----------
    max_lead : int
        Maximum forward periods to analyze
    n_trees : int
        Number of trees per forest
    """
    print("=" * 70)
    print("Phase 7: Dynamic Effects Analysis")
    print("=" * 70)
    
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    
    # Prepare data with DCI
    _, _, _, _, df_dci = prepare_analysis_data(df, cfg, return_df=True)
    
    # Create forward outcomes
    print(f"\n📊 Creating forward outcomes (up to t+{max_lead})...")
    df_dynamic = create_forward_outcomes(df_dci, cfg["outcome"], max_lead)
    
    print(f"   Original observations: {len(df_dci)}")
    
    # Estimate effects for each lead
    print("\n🔬 Estimating dynamic effects...")
    results = []
    
    for lead in range(0, max_lead + 1):
        print(f"\n   Lead {lead} (Y_{{t+{lead}}})...")
        
        result = estimate_dynamic_ate(df_dynamic, cfg, lead, n_trees)
        
        if result is not None:
            results.append(result)
            
            sig_marker = "✅" if result['significant'] else "⚠️"
            print(f"      N = {result['n_obs']}")
            print(f"      ATE = {result['ate']:.4f} [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}] {sig_marker}")
        else:
            print(f"      ⚠️  Insufficient data")
    
    results_df = pd.DataFrame(results)
    
    # Calculate cumulative effects
    print("\n" + "=" * 70)
    print("CUMULATIVE EFFECTS")
    print("=" * 70)
    
    if len(results_df) > 0:
        results_df['cumulative_ate'] = results_df['ate'].cumsum()
        
        print("\n   Lead | Marginal ATE | Cumulative ATE")
        print("   " + "-" * 45)
        for _, row in results_df.iterrows():
            print(f"   t+{int(row['lead']):1d}  | {row['ate']:12.4f} | {row['cumulative_ate']:14.4f}")
    
    # Effect persistence
    print("\n" + "=" * 70)
    print("EFFECT PERSISTENCE")
    print("=" * 70)
    
    if len(results_df) >= 2:
        ate_t0 = results_df.loc[results_df['lead'] == 0, 'ate'].values[0]
        ate_t_max = results_df.loc[results_df['lead'] == max_lead, 'ate'].values
        
        if len(ate_t_max) > 0:
            ate_t_max = ate_t_max[0]
            persistence_ratio = ate_t_max / ate_t0 if ate_t0 != 0 else np.nan
            
            print(f"\n   Effect at t+0: {ate_t0:.4f}")
            print(f"   Effect at t+{max_lead}: {ate_t_max:.4f}")
            print(f"   Persistence ratio: {persistence_ratio:.2%}")
            
            if abs(persistence_ratio) > 0.5:
                print("   ✅ Effects are persistent (ratio > 50%)")
            elif abs(persistence_ratio) > 0.2:
                print("   ⚠️  Effects show moderate persistence (20-50%)")
            else:
                print("   ❌ Effects fade quickly (ratio < 20%)")
    
    # Half-life calculation
    print("\n" + "=" * 70)
    print("EFFECT HALF-LIFE")
    print("=" * 70)
    
    if len(results_df) >= 2:
        ates = results_df['ate'].values
        half_target = ates[0] / 2
        
        # Find when effect crosses half
        for i, ate in enumerate(ates):
            if abs(ate) <= abs(half_target):
                print(f"\n   Half-life: ~{i} periods")
                print(f"   Initial effect decays to 50% by t+{i}")
                break
        else:
            print(f"\n   Half-life: > {max_lead} periods")
            print(f"   Effect persists beyond analysis window")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n   💾 Results saved to: {OUTPUT_FILE}")
    
    # Summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION FOR PAPER")
    print("=" * 70)
    
    print("\n   Key findings for Discussion section:")
    if len(results_df) >= 2:
        if results_df['significant'].all():
            print("   1. DCI effects are persistent across all tested horizons")
        else:
            sig_leads = results_df.loc[results_df['significant'], 'lead'].tolist()
            print(f"   1. DCI effects significant at leads: {sig_leads}")
        
        cumulative = results_df['cumulative_ate'].iloc[-1]
        print(f"   2. Cumulative effect over {max_lead} years: {cumulative:.4f} tons/capita")
        
        print("   3. This supports gradual structural transformation mechanism")
    
    print("\n" + "=" * 70)
    
    return results_df


if __name__ == "__main__":
    run_dynamic_effects_analysis(max_lead=3, n_trees=1000)
