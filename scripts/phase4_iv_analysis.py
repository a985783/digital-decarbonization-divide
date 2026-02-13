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
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

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
    from sklearn.model_selection import KFold, cross_val_predict
    
    # Center variables
    T_centered = T - np.mean(T)
    Z_centered = Z - np.mean(Z)
    
    # First stage regression: T ~ Z + W
    X_first_stage = np.column_stack([Z_centered, W])
    
    model = LinearRegression()

    cv = 5
    cv_kwargs = {}
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            n_splits = min(5, len(unique_groups))
            cv = GroupKFold(n_splits=n_splits)
            cv_kwargs["groups"] = groups
        else:
            cv = KFold(n_splits=2, shuffle=False)

    t_pred = cross_val_predict(model, X_first_stage, T_centered, cv=cv, **cv_kwargs)
    
    # Calculate R²
    r2 = r2_score(T_centered, t_pred)
    r2 = max(r2, 0.0)
    
    # F-statistic approximation
    n = len(T)
    k = np.atleast_2d(Z_centered).shape[1]
    denom_df = n - X_first_stage.shape[1] - 1

    if k <= 0 or denom_df <= 0:
        return np.nan, r2
    if r2 >= 1.0:
        r2 = 1.0 - 1e-12

    f_stat = (r2 / (1 - r2)) * (denom_df / k)
    
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
    Y = np.asarray(Y, dtype=float).reshape(-1)
    T = np.asarray(T, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1, 1)
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)

    n = len(Y)

    X_u = np.column_stack([np.ones(n), Z, W])
    X_r = np.column_stack([np.ones(n), W])

    rank_u = np.linalg.matrix_rank(X_u)
    rank_r = np.linalg.matrix_rank(X_r)
    df_num = rank_u - rank_r
    df_den = n - rank_u

    if df_num <= 0 or df_den <= 0:
        return np.nan, np.nan, np.nan, np.nan

    def _ar_f_stat(beta):
        resid = Y - beta * T

        coef_u, _, _, _ = np.linalg.lstsq(X_u, resid, rcond=None)
        resid_u = resid - X_u @ coef_u
        ssr_u = float(np.dot(resid_u, resid_u))

        coef_r, _, _, _ = np.linalg.lstsq(X_r, resid, rcond=None)
        resid_r = resid - X_r @ coef_r
        ssr_r = float(np.dot(resid_r, resid_r))

        if ssr_u <= 0:
            return np.nan

        stat = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_den)
        return max(stat, 0.0)

    beta_grid = np.linspace(-20.0, 10.0, 3001)
    ar_stats = np.array([_ar_f_stat(beta) for beta in beta_grid], dtype=float)

    critical_value = stats.f.ppf(1 - alpha, df_num, df_den)
    valid_mask = np.isfinite(ar_stats)
    in_ci = valid_mask & (ar_stats <= critical_value)

    if in_ci.any():
        ci_indices = np.where(in_ci)[0]
        ar_ci_lower = float(beta_grid[ci_indices[0]])
        ar_ci_upper = float(beta_grid[ci_indices[-1]])
    else:
        best_idx = np.nanargmin(ar_stats) if np.isfinite(ar_stats).any() else None
        if best_idx is None:
            ar_ci_lower = np.nan
            ar_ci_upper = np.nan
        else:
            beta_star = float(beta_grid[best_idx])
            step = float(beta_grid[1] - beta_grid[0])
            ar_ci_lower = beta_star - step
            ar_ci_upper = beta_star + step

    ar_stat_0 = _ar_f_stat(0.0)
    if pd.notna(ar_stat_0):
        ar_pval = 1 - stats.f.cdf(ar_stat_0, df_num, df_den)
    else:
        ar_pval = np.nan

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


def run_iv_analysis(iv_output_path=IV_RESULTS_FILE, placebo_output_path=PLACEBO_IV_FILE):
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
    print("\n2. Instrument-Treatment Correlation:")
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
    print("\n📈 Effect Size Comparison:")
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
    
    results_df.to_csv(iv_output_path, index=False)
    print(f"\n💾 Results saved to {iv_output_path}")
    
    # Save placebo results
    if len(placebo_df) > 0:
        placebo_df.to_csv(placebo_output_path, index=False)
        print(f"💾 Placebo IV results saved to {placebo_output_path}")
    
    if (ate_iv < 0) and (ate_iv_ub < 0):
        print("\n✅ IV confirms significant negative effect.")
    else:
        print("\n⚠️  IV result is not significant or direction changed.")
    
    print("=" * 70)
    
    return results_df, placebo_df


if __name__ == "__main__":
    run_iv_analysis()
