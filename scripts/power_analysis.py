"""
Power Analysis: Monte Carlo Simulation for Causal Forest
=========================================================
Validates statistical power and coverage rates for n=40 country clusters.

Procedure:
1. Simulate DGP based on estimated true effects
2. Repeatedly train Causal Forest (B iterations)
3. Compute coverage rate: proportion of CIs containing true effect
4. Compute power: proportion rejecting H0
5. Output power curves and coverage tables
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
import time

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
OUTPUT_COVERAGE = os.path.join(RESULTS_DIR, "power_analysis_coverage.csv")
OUTPUT_POWER = os.path.join(RESULTS_DIR, "power_analysis_power.csv")


def simulate_dgp(n_countries, n_years, true_ate, heterogeneity_sd=0.5, seed=None):
    """
    Simulate Data Generating Process with known true effects.
    
    Parameters:
    -----------
    n_countries : int
        Number of countries
    n_years : int  
        Number of years per country
    true_ate : float
        True Average Treatment Effect
    heterogeneity_sd : float
        Standard deviation of heterogeneous effects
    seed : int
        Random seed
    
    Returns:
    --------
    df : DataFrame with simulated data
    true_cates : Array of true CATEs
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = n_countries * n_years
    
    # Generate country and year
    countries = np.repeat(np.arange(n_countries), n_years)
    years = np.tile(np.arange(n_years), n_countries)
    
    # Generate covariates
    gdp = np.random.lognormal(10, 1, n)
    institution = np.random.normal(0, 1, n)
    renewable = np.random.uniform(0, 100, n)
    
    # Country fixed effects
    country_fe = np.repeat(np.random.normal(0, 2, n_countries), n_years)
    
    # Generate treatment (DCI)
    T = 0.3 * np.log(gdp) + 0.2 * institution + np.random.normal(0, 1, n)
    
    # Generate true heterogeneous effects
    # Effect depends on GDP and institution (heterogeneity)
    true_cates = true_ate + heterogeneity_sd * (np.log(gdp) - 10) / 2
    
    # Generate outcome
    Y = (country_fe + 
         0.5 * np.log(gdp) + 
         0.3 * institution - 
         0.2 * renewable / 100 +
         true_cates * T +
         np.random.normal(0, 1, n))
    
    df = pd.DataFrame({
        'country': countries,
        'year': years,
        'Y': Y,
        'T': T,
        'GDP': gdp,
        'Institution': institution,
        'Renewable': renewable
    })
    
    return df, true_cates


def run_single_simulation(sim_id, n_countries, n_years, true_ate, n_trees=500):
    """Run single simulation iteration."""
    # Simulate data
    df, true_cates = simulate_dgp(n_countries, n_years, true_ate, seed=sim_id)
    
    Y = df['Y'].values
    T = df['T'].values
    X = df[['GDP', 'Institution', 'Renewable']].values
    W = X.copy()  # Same as X for simplicity
    groups = df['country'].values
    
    # Fit Causal Forest
    est = CausalForestDML(
        model_y=xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0, random_state=42),
        model_t=xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0, random_state=42),
        n_estimators=n_trees,
        min_samples_leaf=10,
        max_depth=4,
        random_state=sim_id,
        discrete_treatment=False,
        n_jobs=1,
        cv=GroupKFold(n_splits=5)
    )
    
    try:
        est.fit(Y, T, X=X, W=W, groups=groups)
        
        # Get estimates
        ate_est = est.ate(X)
        ate_lb, ate_ub = est.ate_interval(X, alpha=0.05)
        
        # Check coverage
        covered = (ate_lb <= true_ate) and (ate_ub >= true_ate)
        
        # Check significance (reject H0: ATE=0)
        significant = (ate_lb > 0) or (ate_ub < 0)
        
        return {
            'sim_id': sim_id,
            'true_ate': true_ate,
            'est_ate': ate_est,
            'ci_lower': ate_lb,
            'ci_upper': ate_ub,
            'ci_width': ate_ub - ate_lb,
            'covered': covered,
            'significant': significant,
            'success': True
        }
    except Exception as e:
        return {
            'sim_id': sim_id,
            'true_ate': true_ate,
            'est_ate': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_width': np.nan,
            'covered': np.nan,
            'significant': np.nan,
            'success': False
        }


def run_power_analysis(n_simulations=100, n_trees=500):
    """
    Run full Monte Carlo power analysis.
    
    Parameters:
    -----------
    n_simulations : int
        Number of simulation iterations
    n_trees : int
        Number of trees in Causal Forest
    """
    print("=" * 70)
    print("Monte Carlo Power Analysis for Causal Forest")
    print("=" * 70)
    
    # Load actual data to get parameters
    cfg = load_config("analysis_spec.yaml")
    df_real = pd.read_csv(INPUT_FILE)
    
    # Get dimensions from real data
    n_countries = df_real['country'].nunique()
    n_years = df_real['year'].nunique()
    
    print(f"\n📊 Simulation Parameters:")
    print(f"   Countries: {n_countries}")
    print(f"   Years: {n_years}")
    print(f"   Simulations: {n_simulations}")
    print(f"   Trees per forest: {n_trees}")
    
    # True ATE from our analysis (approximately -1.73)
    true_ate = -1.73
    print(f"   True ATE: {true_ate}")
    
    # Run simulations
    print(f"\n🔬 Running {n_simulations} simulations...")
    start_time = time.time()
    
    # Use parallel processing
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_simulation)(
            sim_id=i, 
            n_countries=n_countries,
            n_years=n_years,
            true_ate=true_ate,
            n_trees=n_trees
        )
        for i in range(n_simulations)
    )
    
    elapsed = time.time() - start_time
    print(f"\n   ✓ Completed in {elapsed:.1f} seconds")
    
    # Compile results
    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['success']].copy()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Coverage rate
    coverage_rate = valid_results['covered'].mean()
    print(f"\n1. COVERAGE RATE (95% CI contains true ATE):")
    print(f"   Coverage: {coverage_rate*100:.1f}%")
    
    if coverage_rate >= 0.90:
        print("   ✅ Good coverage (≥90%)")
    elif coverage_rate >= 0.85:
        print("   ⚠️  Slightly undercovered (85-90%)")
    else:
        print("   ❌ Undercovered (<85%) - CIs may be too narrow")
    
    # Power
    power = valid_results['significant'].mean()
    print(f"\n2. STATISTICAL POWER (reject H0: ATE=0):")
    print(f"   Power: {power*100:.1f}%")
    
    if power >= 0.80:
        print("   ✅ Adequate power (≥80%)")
    elif power >= 0.60:
        print("   ⚠️  Moderate power (60-80%)")
    else:
        print("   ❌ Low power (<60%)")
    
    # Bias
    mean_est = valid_results['est_ate'].mean()
    bias = mean_est - true_ate
    print(f"\n3. ESTIMATION BIAS:")
    print(f"   Mean estimate: {mean_est:.4f}")
    print(f"   True ATE: {true_ate:.4f}")
    print(f"   Bias: {bias:.4f} ({abs(bias/true_ate)*100:.1f}%)")
    
    # RMSE
    rmse = np.sqrt(((valid_results['est_ate'] - true_ate)**2).mean())
    print(f"\n4. ROOT MEAN SQUARED ERROR:")
    print(f"   RMSE: {rmse:.4f}")
    
    # CI width
    mean_width = valid_results['ci_width'].mean()
    print(f"\n5. AVERAGE CI WIDTH:")
    print(f"   Mean width: {mean_width:.4f}")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Summary statistics
    summary_df = pd.DataFrame({
        'Metric': ['Coverage_Rate', 'Power', 'Bias', 'RMSE', 'Mean_CI_Width', 
                   'N_Simulations', 'N_Countries', 'N_Years', 'True_ATE'],
        'Value': [coverage_rate, power, bias, rmse, mean_width,
                  n_simulations, n_countries, n_years, true_ate]
    })
    summary_df.to_csv(OUTPUT_COVERAGE, index=False)
    print(f"   💾 Coverage summary saved to: {OUTPUT_COVERAGE}")
    
    # Full results
    results_df.to_csv(OUTPUT_POWER, index=False)
    print(f"   💾 Full results saved to: {OUTPUT_POWER}")
    
    print("=" * 70)
    
    # =========================================================================
    # 6. Generate Figures
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        pass
        
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of estimates
    estimates = valid_results['est_ate']
    plt.hist(estimates, bins=20, alpha=0.6, color='steelblue', density=True, label='Estimate Distribution')
    
    # Add KDE if available
    try:
        sns.kdeplot(estimates, color='blue', linewidth=2)
    except:
        pass
        
    # Vertical lines
    plt.axvline(true_ate, color='red', linestyle='-', linewidth=2, label=f'True ATE ({true_ate})')
    plt.axvline(mean_est, color='green', linestyle='--', linewidth=2, label=f'Mean Estimate ({mean_est:.2f})')
    
    # Title and labels
    plt.title(f'Monte Carlo Simulation Results (N={n_simulations})\nPower: {power:.1%}, Coverage: {coverage_rate:.1%}', fontsize=14)
    plt.xlabel('Estimated ATE', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(FIGURES_DIR, "power_simulation_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 Simulation plot saved to: {plot_path}")
    
    return summary_df, results_df


if __name__ == "__main__":
    # Run with reduced iterations for speed (increase for publication)
    run_power_analysis(n_simulations=100, n_trees=500)
