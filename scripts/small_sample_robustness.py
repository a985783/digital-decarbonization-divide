"""
Small Sample Robustness Analysis
=================================
Addresses审稿人关切：
1. Bootstrap收敛诊断
2. 小样本下Causal Forest的稳定性
3. 样本量敏感性分析
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold

try:
    from econml.dml import CausalForestDML
    import xgboost as xgb
except ImportError:
    raise ImportError("Please install econml and xgboost")

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
INPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")
ROBUSTNESS_RESULTS_FILE = os.path.join(RESULTS_DIR, "small_sample_robustness.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)

def bootstrap_convergence_diagnostic(df, cfg, n_bootstrap_list=[100, 200, 500, 1000]):
    """
    Test bootstrap convergence by varying number of bootstrap iterations
    """
    print("\n🔬 Bootstrap Convergence Diagnostic")
    print("=" * 70)
    
    # Prepare data
    df_clean = df.dropna(subset=[cfg["outcome"]])
    y, t, x, w, df_clean = prepare_analysis_data(df_clean, cfg, return_df=True)
    groups = df_clean[cfg["groups"]].values
    
    results = []
    country_codes = df_clean[cfg["groups"]].to_numpy()
    unique_countries = np.unique(country_codes)
    
    for B in n_bootstrap_list:
        print(f"\n   Testing B = {B} bootstrap iterations...")
        
        # Fit Causal Forest
        est = CausalForestDML(
            model_y=xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42),
            model_t=xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42),
            n_estimators=500,  # Smaller for speed
            min_samples_leaf=10,
            max_depth=6,
            random_state=42,
            discrete_treatment=False,
            n_jobs=-1,
            cv=GroupKFold(n_splits=cfg["cv"]["n_splits"]),
        )
        
        est.fit(y, t, X=x, W=w, groups=groups)
        
        # Get CATE predictions
        cate_pred = est.effect(x)
        
        # Country-cluster bootstrap for ATE
        rng = np.random.default_rng(42 + B)
        bootstrap_ates = []
        country_to_idx = {
            c: np.where(country_codes == c)[0]
            for c in unique_countries
        }

        for _ in range(B):
            sampled_countries = rng.choice(unique_countries, size=len(unique_countries), replace=True)
            sampled_idx = np.concatenate([country_to_idx[c] for c in sampled_countries])
            bootstrap_cate = cate_pred[sampled_idx]
            bootstrap_ates.append(np.mean(bootstrap_cate))
        
        # Calculate bootstrap statistics
        ate_mean = np.mean(bootstrap_ates)
        ate_se = np.std(bootstrap_ates)
        ate_ci_lower = np.percentile(bootstrap_ates, 2.5)
        ate_ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        results.append({
            'bootstrap_iterations': B,
            'ate_mean': ate_mean,
            'ate_se': ate_se,
            'ate_ci_lower': ate_ci_lower,
            'ate_ci_upper': ate_ci_upper,
            'ci_width': ate_ci_upper - ate_ci_lower
        })
        
        print(f"      ATE: {ate_mean:.4f} (SE: {ate_se:.4f}, CI: [{ate_ci_lower:.4f}, {ate_ci_upper:.4f}])")
    
    # Check convergence
    results_df = pd.DataFrame(results)
    ci_width_change = results_df['ci_width'].iloc[-1] / results_df['ci_width'].iloc[0]
    
    print("\n📊 Convergence Analysis:")
    print(f"   CI width reduction (B=100 to B=1000): {1-ci_width_change:.1%}")
    
    if ci_width_change < 0.8:
        print("   ✅ Good convergence: CI width reduced by >20%")
    else:
        print("   ⚠️  Slow convergence: Consider larger B")
    
    return results_df

def sample_size_sensitivity(df, cfg, sample_sizes=[0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Test how results change with different sample sizes
    """
    print("\n🔬 Sample Size Sensitivity Analysis")
    print("=" * 70)
    
    df_clean = df.dropna(subset=[cfg["outcome"]])
    rng = np.random.default_rng(42)
    
    results = []
    
    for frac in sample_sizes:
        if frac < 1.0:
            # Subsample countries
            countries = df_clean['country'].unique()
            n_countries = int(len(countries) * frac)
            selected_countries = rng.choice(countries, n_countries, replace=False)
            df_sub = df_clean[df_clean['country'].isin(selected_countries)]
            sample_desc = f"{n_countries} countries ({frac:.0%})"
        else:
            df_sub = df_clean.copy()
            sample_desc = "Full sample (40 countries)"
        
        print(f"\n   Testing with {sample_desc}...")
        
        # Prepare data
        y, t, x, w, df_sub = prepare_analysis_data(df_sub, cfg, return_df=True)
        groups = df_sub[cfg["groups"]].values
        
        # Fit Causal Forest
        est = CausalForestDML(
            model_y=xgb.XGBRegressor(n_estimators=200, max_depth=3, random_state=42),
            model_t=xgb.XGBRegressor(n_estimators=200, max_depth=3, random_state=42),
            n_estimators=500,
            min_samples_leaf=10,
            max_depth=6,
            random_state=42,
            discrete_treatment=False,
            n_jobs=-1,
            cv=GroupKFold(n_splits=cfg["cv"]["n_splits"]),
        )
        
        est.fit(y, t, X=x, W=w, groups=groups)
        
        # Get results
        cate_pred = est.effect(x)
        cate_lb, cate_ub = est.effect_interval(x, alpha=0.05)
        
        significant_pct = np.mean((cate_lb > 0) | (cate_ub < 0))
        
        results.append({
            'sample_fraction': frac,
            'n_countries': len(df_sub['country'].unique()),
            'n_observations': len(df_sub),
            'ate_mean': np.mean(cate_pred),
            'ate_std': np.std(cate_pred),
            'significant_pct': significant_pct * 100
        })
        
        print(f"      ATE: {np.mean(cate_pred):.4f} (SD: {np.std(cate_pred):.4f})")
        print(f"      Significant: {significant_pct:.1%}")
    
    # Check stability
    results_df = pd.DataFrame(results)
    ate_range = results_df['ate_mean'].max() - results_df['ate_mean'].min()
    ate_max = abs(results_df['ate_mean'].max())
    relative_variation = ate_range / ate_max if ate_max > 0 else np.inf
    
    print("\n📊 Stability Analysis:")
    print(f"   ATE range across sample sizes: {ate_range:.4f}")
    print(f"   Relative variation: {relative_variation:.1%}")

    if relative_variation < 0.3:
        print("   ✅ Results stable across sample sizes")
    else:
        print("   ⚠️  Results sensitive to sample size")
    
    return results_df

def run_small_sample_robustness():
    print("=" * 70)
    print("Small Sample Robustness Analysis")
    print("(Addresses Reviewer Concerns about n=40 countries)")
    print("=" * 70)
    
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    
    # Run diagnostics
    bootstrap_results = bootstrap_convergence_diagnostic(df, cfg)
    sample_size_results = sample_size_sensitivity(df, cfg)
    
    # Combine results
    summary = {
        'bootstrap_converged': bootstrap_results['ci_width'].iloc[-1] < bootstrap_results['ci_width'].iloc[0] * 0.8,
        'sample_size_stable': (
            (sample_size_results['ate_mean'].max() - sample_size_results['ate_mean'].min())
            / abs(sample_size_results['ate_mean'].max())
            if abs(sample_size_results['ate_mean'].max()) > 0
            else np.inf
        ) < 0.3,
        'min_significant_pct': sample_size_results['significant_pct'].min(),
        'final_ate': sample_size_results.loc[sample_size_results['sample_fraction'] == 1.0, 'ate_mean'].iloc[0]
    }
    
    # Save results
    bootstrap_results.to_csv(os.path.join(RESULTS_DIR, 'bootstrap_convergence.csv'), index=False)
    sample_size_results.to_csv(os.path.join(RESULTS_DIR, 'sample_size_sensitivity.csv'), index=False)
    
    pd.DataFrame([summary]).to_csv(ROBUSTNESS_RESULTS_FILE, index=False)
    
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    
    if summary['bootstrap_converged'] and summary['sample_size_stable']:
        print("\n✅ ROBUST: Results are stable and bootstrap converges well")
        print("   The n=40 sample size appears adequate for Causal Forest")
    else:
        print("\n⚠️  CONCERNS: Results show sensitivity")
        print("   Recommendations:")
        print("   - Increase bootstrap iterations (B > 1000)")
        print("   - Consider alternative estimators (e.g., DR-Learner)")
        print("   - Acknowledge limitations in discussion")
    
    print("\n📊 Key Statistics:")
    print(f"   Minimum significant observations: {summary['min_significant_pct']:.1f}%")
    print(f"   Final ATE estimate: {summary['final_ate']:.4f}")
    
    print("\n💾 Detailed results saved to:")
    print(f"   - {os.path.join(RESULTS_DIR, 'bootstrap_convergence.csv')}")
    print(f"   - {os.path.join(RESULTS_DIR, 'sample_size_sensitivity.csv')}")
    print(f"   - {ROBUSTNESS_RESULTS_FILE}")

if __name__ == "__main__":
    run_small_sample_robustness()
