"""
Phase 4: IV Analysis (Enhanced)
===============================
Instrumental Variable Strategy using Lagged DCI to address endogeneity.

Identification Strategy:
1. Instrument: Lagged DCI (t-1)
2. Exclusion Restriction: Lagged DCI affects CO2 only through current DCI
3. Relevance: Digital infrastructure persistence ensures correlation

Enhanced Features (Q1 Reviewer Response):
- Placebo IV Tests using DCI(t-2), DCI(t-3) as pseudo-instruments
- Anderson-Rubin weak-IV robust confidence intervals
- Comprehensive diagnostics table

Weak IV Test: First-stage F-statistic > 10 (Staiger & Stock, 1997)
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

try:
    from econml.iv.dml import OrthoIV
    from econml.dml import LinearDML
    import xgboost as xgb
except ImportError:
    raise ImportError("Please install econml and xgboost")

DATA_DIR = "data"
RESULTS_DIR = "results"
INPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")
IV_RESULTS_FILE = os.path.join(RESULTS_DIR, "iv_analysis_results.csv")
PLACEBO_IV_FILE = os.path.join(RESULTS_DIR, "placebo_iv_results.csv")


def first_stage_f_statistic(T, Z, W, groups):
    """
    Calculate first-stage F-statistic for weak IV test
    H0: Instrument is weak (first-stage coefficient = 0)
    Critical value: F > 10 (Staiger & Stock, 1997)
    """
    from sklearn.model_selection import cross_val_predict
    
    # Center variables
    T_centered = T - np.mean(T)
    Z_centered = Z - np.mean(Z)
    
    # First stage regression: T ~ Z + W
    X_first_stage = np.column_stack([Z_centered, W])
    
    model = LinearRegression()
    t_pred = cross_val_predict(model, X_first_stage, T_centered, cv=5)
    
    # Calculate R²
    r2 = r2_score(T_centered, t_pred)
    
    # F-statistic approximation
    n = len(T)
    k = X_first_stage.shape[1]
    f_stat = (r2 / (1 - r2)) * ((n - k - 1) / k)
    
    return f_stat, r2


def anderson_rubin_ci(Y, T, Z, W, alpha=0.05):
    """
    Compute Anderson-Rubin (1949) weak-IV robust confidence interval.
    
    The AR statistic is robust to weak instruments and provides valid
    inference even when first-stage F is low.
    
    Returns:
    --------
    ar_ci_lower, ar_ci_upper : float
        Anderson-Rubin confidence interval bounds
    ar_stat : float
        AR test statistic
    ar_pval : float
        p-value for AR test of beta=0
    """
    n = len(Y)
    
    # Grid search for AR CI
    beta_grid = np.linspace(-5, 2, 500)
    ar_stats = []
    
    for beta in beta_grid:
        # Reduced form residual under H0: beta = beta_0
        resid = Y - beta * T
        
        # Regress residual on Z and W
        X_ar = np.column_stack([Z, W])
        model = LinearRegression()
        model.fit(X_ar, resid)
        resid_fitted = model.predict(X_ar)
        
        # AR statistic (Wald-type test on Z coefficient)
        ssr_restricted = np.sum((resid - np.mean(resid))**2)
        ssr_unrestricted = np.sum((resid - resid_fitted)**2)
        
        k = 1  # number of instruments
        ar_stat = ((ssr_restricted - ssr_unrestricted) / k) / (ssr_unrestricted / (n - X_ar.shape[1]))
        ar_stats.append(ar_stat)
    
    ar_stats = np.array(ar_stats)
    
    # Critical value (chi-squared with 1 df, divided by 1)
    critical_value = stats.chi2.ppf(1 - alpha, df=1)
    
    # Find CI bounds (values where AR stat < critical value)
    in_ci = ar_stats < critical_value
    
    if in_ci.any():
        ci_indices = np.where(in_ci)[0]
        ar_ci_lower = beta_grid[ci_indices[0]]
        ar_ci_upper = beta_grid[ci_indices[-1]]
    else:
        # If no values in CI, return NaN
        ar_ci_lower = np.nan
        ar_ci_upper = np.nan
    
    # AR test at beta=0
    resid_0 = Y
    X_ar = np.column_stack([Z, W])
    model = LinearRegression()
    model.fit(X_ar, resid_0)
    resid_fitted_0 = model.predict(X_ar)
    
    ssr_restricted_0 = np.sum((resid_0 - np.mean(resid_0))**2)
    ssr_unrestricted_0 = np.sum((resid_0 - resid_fitted_0)**2)
    
    ar_stat_0 = ((ssr_restricted_0 - ssr_unrestricted_0) / 1) / (ssr_unrestricted_0 / (n - X_ar.shape[1]))
    ar_pval = 1 - stats.chi2.cdf(ar_stat_0, df=1)
    
    return ar_ci_lower, ar_ci_upper, ar_stat_0, ar_pval


def run_placebo_iv_test(df_dci, cfg, lags=[2, 3]):
    """
    Run placebo IV tests using longer lags as pseudo-instruments.
    
    If exclusion restriction holds, longer lags should be weaker instruments
    (lower F-statistic) because they are more "distant" from current DCI.
    
    Parameters:
    -----------
    df_dci : DataFrame
        Data with DCI computed
    cfg : dict
        Configuration dictionary
    lags : list
        List of lag orders to test as placebo IVs
    
    Returns:
    --------
    placebo_results : DataFrame
        Results for each placebo lag
    """
    print("\n" + "=" * 70)
    print("PLACEBO IV TESTS (Exclusion Restriction Validation)")
    print("=" * 70)
    
    print("\n   Rationale: If exclusion restriction holds, longer lags should")
    print("   have weaker first-stage relationships (lower F-statistics).")
    print("   If F remains high for t-2, t-3, the exclusion may be violated.")
    
    placebo_results = []
    
    for lag in lags:
        print(f"\n   Testing DCI(t-{lag}) as instrument...")
        
        # Create lagged variable
        df_test = df_dci.copy()
        df_test[f"DCI_lag{lag}"] = df_test.groupby("country")["DCI"].shift(lag)
        df_test = df_test.dropna(subset=[f"DCI_lag{lag}", cfg["outcome"]])
        
        if len(df_test) < 100:
            print(f"      ⚠️  Insufficient observations (n={len(df_test)})")
            continue
        
        Y = df_test[cfg["outcome"]].values
        T = df_test["DCI"].values
        Z = df_test[f"DCI_lag{lag}"].values
        W = df_test[cfg["controls_W"]].values
        groups = df_test[cfg["groups"]].values
        
        # First-stage F
        f_stat, r2 = first_stage_f_statistic(T, Z, W, groups)
        
        # Correlation
        corr = np.corrcoef(T, Z)[0, 1]
        
        print(f"      N = {len(df_test)}")
        print(f"      First-stage F: {f_stat:.2f}")
        print(f"      First-stage R²: {r2:.4f}")
        print(f"      Corr(DCI, DCI_lag{lag}): {corr:.4f}")
        
        placebo_results.append({
            'Lag': lag,
            'N': len(df_test),
            'F_Statistic': f_stat,
            'R2': r2,
            'Correlation': corr,
            'Strong_IV': f_stat > 10
        })
    
    placebo_df = pd.DataFrame(placebo_results)
    
    # Interpretation
    print("\n   INTERPRETATION:")
    if len(placebo_df) >= 2:
        f_decay = placebo_df['F_Statistic'].iloc[0] > placebo_df['F_Statistic'].iloc[-1]
        if f_decay:
            print("   ✅ F-statistic decays with longer lags (supports exclusion)")
        else:
            print("   ⚠️  F-statistic does not decay (potential violation)")
    
    return placebo_df


def run_iv_analysis():
    print("=" * 70)
    print("Phase 4: Instrumental Variable Analysis (Enhanced)")
    print("=" * 70)
    
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    
    if "DCI" not in df.columns:
        pass
        
    _, _, _, _, df_dci = prepare_analysis_data(df, cfg, return_df=True)
    
    print("Creating Lagged DCI (Instrument)...")
    df_dci = df_dci.sort_values(["country", "year"])
    df_dci["DCI_lag1"] = df_dci.groupby("country")["DCI"].shift(1)
    
    df_iv = df_dci.dropna(subset=["DCI_lag1", cfg["outcome"]])
    
    print(f"Sample size after lags: {len(df_iv)}")
    
    Y = df_iv[cfg["outcome"]].values
    T = df_iv["DCI"].values
    Z = df_iv["DCI_lag1"].values
    X = df_iv[cfg["moderators_X"]].values
    W = df_iv[cfg["controls_W"]].values
    groups = df_iv[cfg["groups"]].values
    
    print("\n" + "=" * 70)
    print("IV VALIDITY DIAGNOSTICS")
    print("=" * 70)
    
    # 1. Weak IV Test (First-stage F-statistic)
    print("\n1. Testing for Weak Instruments...")
    f_stat, r2_first = first_stage_f_statistic(T, Z, W, groups)
    print(f"   First-stage F-statistic: {f_stat:.2f}")
    print(f"   First-stage R²: {r2_first:.4f}")
    
    if f_stat > 10:
        print("   ✅ PASS: F > 10 (Instrument is strong)")
    else:
        print("   ⚠️  CAUTION: F ≤ 10 (Potential weak instrument)")
    
    # 2. Instrument Relevance (Correlation)
    corr_instrument = np.corrcoef(T, Z)[0, 1]
    print(f"\n2. Instrument-Treatment Correlation:")
    print(f"   Corr(DCI, DCI_lag1): {corr_instrument:.4f}")
    
    if abs(corr_instrument) > 0.3:
        print("   ✅ PASS: Correlation > 0.3")
    else:
        print("   ⚠️  CAUTION: Low correlation")
    
    # 3. Anderson-Rubin Weak-IV Robust CI
    print("\n3. Anderson-Rubin Weak-IV Robust CI:")
    ar_lb, ar_ub, ar_stat, ar_pval = anderson_rubin_ci(Y, T, Z, W)
    print(f"   AR 95% CI: [{ar_lb:.4f}, {ar_ub:.4f}]")
    print(f"   AR Test (H0: β=0): stat={ar_stat:.2f}, p={ar_pval:.4f}")
    if ar_pval < 0.05:
        print("   ✅ AR test rejects H0: β=0")
    else:
        print("   ⚠️  AR test fails to reject H0: β=0")
    
    # 4. Exclusion Restriction Discussion
    print("\n4. Exclusion Restriction Assessment:")
    print("   Argument: Lagged DCI affects CO2 only through current DCI")
    print("   - Digital infrastructure persists (high correlation)")
    print("   - Historical ICT capacity unlikely to directly impact current emissions")
    print("   - Controlled for: GDP, Energy, Institutions, etc.")
    print("   ⚠️  Note: Cannot be tested statistically; requires theoretical justification")
    
    print("\n" + "=" * 70)
    print("IV ESTIMATION RESULTS")
    print("=" * 70)
    
    print("\n🔬 Training OrthoIV (IV-DML)...")
    
    est = OrthoIV(
        model_y_xw=xgb.XGBRegressor(n_estimators=100, max_depth=3, verbosity=0),
        model_t_xw=xgb.XGBRegressor(n_estimators=100, max_depth=3, verbosity=0),
        model_z_xw=xgb.XGBRegressor(n_estimators=100, max_depth=3, verbosity=0), 
        discrete_treatment=False,
        random_state=42,
        cv=5
    )
    
    est.fit(Y, T, Z=Z, X=X, W=W, groups=groups)
    
    print("\n📊 IV Results (ATE):")
    ate_iv = est.ate(X)
    ate_iv_lb, ate_iv_ub = est.ate_interval(X, alpha=0.05)
    
    print(f"   IV ATE: {ate_iv:.4f}")
    print(f"   Standard 95% CI: [{ate_iv_lb:.4f}, {ate_iv_ub:.4f}]")
    print(f"   AR Robust 95% CI: [{ar_lb:.4f}, {ar_ub:.4f}]")
    
    print("\n🔬 Training Naive LinearDML (No IV)...")
    est_naive = LinearDML(
        model_y=xgb.XGBRegressor(n_estimators=100, max_depth=3, verbosity=0),
        model_t=xgb.XGBRegressor(n_estimators=100, max_depth=3, verbosity=0),
        discrete_treatment=False,
        random_state=42,
        cv=5
    )
    est_naive.fit(Y, T, X=X, W=W, groups=groups)
    ate_naive = est_naive.ate(X)
    
    print(f"   Naive ATE: {ate_naive:.4f}")
    
    # Calculate bias reduction
    bias_reduction = ((ate_naive - ate_iv) / ate_naive) * 100 if ate_naive != 0 else 0
    
    # Run Placebo IV Tests
    placebo_df = run_placebo_iv_test(df_dci, cfg, lags=[2, 3])
    
    print("\n" + "=" * 70)
    print("SUMMARY AND INTERPRETATION")
    print("=" * 70)
    print(f"\n📈 Effect Size Comparison:")
    print(f"   Naive ATE: {ate_naive:.4f}")
    print(f"   IV ATE:    {ate_iv:.4f}")
    print(f"   Difference: {ate_iv - ate_naive:.4f} ({bias_reduction:.1f}% change)")
    
    if abs(bias_reduction) > 10:
        print("   ✅ Large bias correction suggests endogeneity concerns were valid")
    else:
        print("   ℹ️  Small bias correction suggests limited endogeneity bias")
    
    # Save main results
    results_df = pd.DataFrame({
        "Model": ["IV (OrthoIV)", "Naive (LinearDML)", "AR Robust CI"],
        "ATE": [ate_iv, ate_naive, ate_iv],
        "CI_Lower": [ate_iv_lb, np.nan, ar_lb],
        "CI_Upper": [ate_iv_ub, np.nan, ar_ub],
        "F_Statistic": [f_stat, np.nan, np.nan],
        "First_stage_R2": [r2_first, np.nan, np.nan],
        "AR_Statistic": [ar_stat, np.nan, ar_stat],
        "AR_PValue": [ar_pval, np.nan, ar_pval]
    })
    
    results_df.to_csv(IV_RESULTS_FILE, index=False)
    print(f"\n💾 Results saved to {IV_RESULTS_FILE}")
    
    # Save placebo results
    if len(placebo_df) > 0:
        placebo_df.to_csv(PLACEBO_IV_FILE, index=False)
        print(f"💾 Placebo IV results saved to {PLACEBO_IV_FILE}")
    
    if (ate_iv < 0) and (ate_iv_ub < 0):
        print("\n✅ IV confirms significant negative effect.")
    else:
        print("\n⚠️  IV result is not significant or direction changed.")
    
    print("=" * 70)
    
    return results_df, placebo_df


if __name__ == "__main__":
    run_iv_analysis()
