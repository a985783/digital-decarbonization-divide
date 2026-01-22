"""
Phase 2: Causal Forest Analysis
================================
使用 CausalForestDML 估计每个样本的条件平均处理效应 (CATE)。
探索 ICT 对 CO2 排放的异质性效应。
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import econml, provide installation instructions if missing
try:
    from econml.dml import CausalForestDML
    print("✓ econml loaded successfully")
except ImportError:
    print("❌ econml not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'econml'])
    from econml.dml import CausalForestDML

try:
    import xgboost as xgb
    print("✓ xgboost loaded successfully")
except ImportError:
    print("❌ xgboost not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb

# Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
INPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v3_imputed.csv')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'causal_forest_cate.csv')

def prepare_data(df):
    """Prepare data for Causal Forest analysis."""
    target = 'CO2_per_capita'
    treatment = 'ICT_exports'
    
    # Key moderating variables (used for heterogeneity detection)
    moderators = [
        'Control_of_Corruption', 'Rule_of_Law', 'Government_Effectiveness',
        'Regulatory_Quality', 'GDP_per_capita_constant', 'Renewable_energy_consumption_pct',
        'Energy_use_per_capita', 'Urban_population_pct', 'Internet_users'
    ]
    
    # Exclude identifiers and outcome/treatment
    exclude = ['country', 'year', target, treatment, 'OECD']
    w_cols = [c for c in df.columns if c not in exclude]
    
    # Drop rows with missing core variables
    df = df.dropna(subset=[target, treatment])
    
    # ⚠️ CRITICAL: Unit correction for CO2
    # Raw data appears to be in 0.01 metric tons, need to convert to metric tons
    df[target] = df[target] / 100.0
    print(f"   ✓ Applied CO2 unit correction: divided by 100")
    print(f"   CO2 mean after correction: {df[target].mean():.2f} metric tons/capita")
    
    # Fill remaining NAs in controls with median
    for col in w_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    Y = df[target].values
    T = df[treatment].values
    W = df[w_cols].values
    
    # Create DataFrame for W with column names (for interpretability)
    W_df = df[w_cols].copy()
    
    # Store country/year for later
    meta_df = df[['country', 'year']].copy()
    
    return Y, T, W, W_df, meta_df, w_cols


def run_causal_forest():
    print("=" * 70)
    print("Phase 2: Causal Forest Analysis - Heterogeneous Treatment Effects")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df)} observations, {len(df.columns)} variables")
    
    # Prepare data
    Y, T, W, W_df, meta_df, w_cols = prepare_data(df)
    print(f"   Analysis sample: N = {len(Y)}")
    print(f"   Features for heterogeneity: {len(w_cols)} variables")
    
    # Configure Causal Forest
    print("\n🌲 Configuring Causal Forest DML...")
    print("   - n_estimators: 2000")
    print("   - min_samples_leaf: 10")
    print("   - max_depth: 6")
    print("   - Using XGBoost for first-stage models\n")
    
    est = CausalForestDML(
        model_y=xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1, 
            n_jobs=-1, 
            random_state=42,
            verbosity=0
        ),
        model_t=xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1, 
            n_jobs=-1, 
            random_state=42,
            verbosity=0
        ),
        n_estimators=2000,
        min_samples_leaf=10,
        max_depth=6,
        random_state=42,
        discrete_treatment=False,
        n_jobs=-1
    )
    
    # Fit the model
    print("🔬 Training Causal Forest (this may take several minutes)...")
    import time
    start_time = time.time()
    
    est.fit(Y, T, X=W, W=W)
    
    elapsed = time.time() - start_time
    print(f"   ✓ Training completed in {elapsed:.1f} seconds")
    
    # Extract CATE predictions
    print("\n📊 Extracting CATE predictions...")
    cate_pred = est.effect(W)
    cate_lb, cate_ub = est.effect_interval(W, alpha=0.05)
    
    # Build results DataFrame
    results_df = W_df.copy()
    results_df['country'] = meta_df['country'].values
    results_df['year'] = meta_df['year'].values
    results_df['CATE'] = cate_pred.flatten()
    results_df['CATE_LB'] = cate_lb.flatten()
    results_df['CATE_UB'] = cate_ub.flatten()
    
    # Significance: CI does not cross zero
    results_df['Significant'] = (results_df['CATE_LB'] > 0) | (results_df['CATE_UB'] < 0)
    
    # Tag direction
    results_df['Effect_Direction'] = np.where(
        results_df['CATE'] < 0, 
        'Reducing CO2', 
        'Increasing CO2'
    )
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📈 CAUSAL FOREST RESULTS SUMMARY")
    print("=" * 70)
    
    sig_pct = results_df['Significant'].mean() * 100
    print(f"\n1. Significant heterogeneity detected: {sig_pct:.1f}% of observations")
    
    print(f"\n2. CATE Distribution:")
    print(f"   Mean:   {results_df['CATE'].mean():.6f}")
    print(f"   Median: {results_df['CATE'].median():.6f}")
    print(f"   Std:    {results_df['CATE'].std():.6f}")
    print(f"   Min:    {results_df['CATE'].min():.6f}")
    print(f"   Max:    {results_df['CATE'].max():.6f}")
    
    # Direction breakdown
    reducing = (results_df['CATE'] < 0).sum()
    increasing = (results_df['CATE'] > 0).sum()
    print(f"\n3. Effect Direction:")
    print(f"   ICT reduces CO2:    {reducing} observations ({reducing/len(results_df)*100:.1f}%)")
    print(f"   ICT increases CO2:  {increasing} observations ({increasing/len(results_df)*100:.1f}%)")
    
    # Correlation with moderators
    print(f"\n4. Correlation between CATE and Key Moderators:")
    key_mods = ['Control_of_Corruption', 'GDP_per_capita_constant', 
                'Renewable_energy_consumption_pct', 'Energy_use_per_capita']
    for mod in key_mods:
        if mod in results_df.columns:
            corr = results_df['CATE'].corr(results_df[mod])
            print(f"   CATE × {mod}: r = {corr:.4f}")
    
    # Save results
    print(f"\n💾 Saving results to: {OUTPUT_FILE}")
    
    # Reorder columns for output
    output_cols = ['country', 'year', 'CATE', 'CATE_LB', 'CATE_UB', 'Significant', 
                   'Effect_Direction'] + w_cols
    results_df[output_cols].to_csv(OUTPUT_FILE, index=False)
    
    print("=" * 70)
    
    # Verdict
    if sig_pct >= 20:
        print("🟢 STRONG HETEROGENEITY DETECTED!")
        print(f"   {sig_pct:.1f}% of observations show significant effects")
        print("   ➜ Proceed to Phase 3 for visualization!")
    elif sig_pct >= 10:
        print("🟡 MODERATE HETEROGENEITY DETECTED")
        print(f"   {sig_pct:.1f}% of observations show significant effects")
        print("   ➜ Consider proceeding to Phase 3")
    else:
        print("🔴 WEAK OR NO HETEROGENEITY")
        print(f"   Only {sig_pct:.1f}% of observations show significant effects")
        print("   ➜ Consider revisiting the model or accepting null result")
    
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    run_causal_forest()
