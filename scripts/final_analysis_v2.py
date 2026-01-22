"""
Final Analysis Pipeline v2 (Two-Way FE + Unit-Corrected)
=========================================================
PRIMARY SPECIFICATION:
- Data: clean_A_v2.csv (CO2 in metric tons per capita)
- FE: Country FE (within-transform) + Year FE (year dummies in W)
- Cross-fitting: GroupKFold by country (5 folds)
- SE: Cluster-robust by country with G/(G-1) correction

Generates: results/results_summary_v2.csv (Single Source of Truth)
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
import os

# Configuration
DATA_PATH = 'data/clean_A_v2.csv'
RESULTS_DIR = 'results'

TARGET = 'CO2_per_capita'
TREATMENT = 'ICT_exports'

CONTROLS = [
    'GDP_per_capita', 'GDP_growth', 'Trade_openness', 'Energy_use', 
    'Renewable_energy', 'Industry_VA', 'Manufacturing', 'Services',
    'Urban_pop', 'Internet_users', 'Mobile_subs', 'Inflation', 
    'Gross_investment'
]

np.random.seed(2026)

def load_data():
    df = pd.read_csv(DATA_PATH)
    valid_controls = [c for c in CONTROLS if c in df.columns]
    
    # Drop NAs
    cols = [TARGET, TREATMENT, 'country', 'year'] + valid_controls
    df_model = df[cols].dropna()
    
    print(f"Sample: {len(df_model)} obs, {df_model['country'].nunique()} countries, {df_model['year'].nunique()} years")
    print(f"Y (CO2): Mean={df_model[TARGET].mean():.2f}, Max={df_model[TARGET].max():.2f} (metric tons/cap)")
    print(f"T (ICT): Mean={df_model[TREATMENT].mean():.2f}, Max={df_model[TREATMENT].max():.2f} (%)")
    
    return df_model, valid_controls

def create_year_dummies(df):
    """Create year dummies for Two-Way FE."""
    years = df['year'].unique()
    base_year = years.min()  # Drop first year as reference
    
    dummies = pd.get_dummies(df['year'], prefix='yr', drop_first=True)
    return dummies

def stage1_lasso(df, controls):
    """Stage 1: Variable Selection (on levels, just for screening)."""
    print("\n[Stage 1] Lasso Selection...")
    
    X = df[controls]
    y = df[TARGET]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
    
    coefs = pd.Series(lasso.coef_, index=controls)
    selected = coefs[coefs != 0].index.tolist()
    
    print(f"  Selected {len(selected)}/{len(controls)} controls.")
    
    coefs.to_csv(os.path.join(RESULTS_DIR, 'lasso_coefs_v2.csv'))
    return selected

def stage2_threshold(df, selected_controls):
    """Stage 2: Exploratory threshold (SHAP from XGBoost)."""
    print("\n[Stage 2] XGBoost + SHAP (Exploratory)...")
    
    X = df[[TREATMENT] + selected_controls]
    y = df[TARGET]
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    t_vals = df[TREATMENT].values
    s_vals = shap_values[:, 0]
    
    from statsmodels.nonparametric.smoothers_lowess import lowess
    z = lowess(s_vals, t_vals, frac=0.3)
    
    threshold = None
    for i in range(len(z)-1):
        if (z[i, 1] < 0 and z[i+1, 1] > 0) or (z[i, 1] > 0 and z[i+1, 1] < 0):
            threshold = z[i, 0]
            break
    
    if threshold is None:
        threshold = df[TREATMENT].median()
        
    print(f"  Exploratory Turning Region: ~{threshold:.1f}%")
    
    # --- PLOTTING ---
    import matplotlib.pyplot as plt
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')
        
    # 1. SHAP Dependence Plot (Turning Region)
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(TREATMENT, shap_values, X, interaction_index=None, show=False)
    plt.title(f"SHAP Dependence: {TREATMENT} vs {TARGET}\nTurning Region ~ {threshold:.2f}%")
    plt.tight_layout()
    plt.savefig('results/figures/shap_dependence.png', dpi=300)
    plt.close()
    print("  ✓ Saved shap_dependence.png")
    
    # 2. SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig('results/figures/shap_summary.png', dpi=300)
    plt.close()
    print("  ✓ Saved shap_summary.png")
    
    return threshold

def stage3_panel_dml_twfe(df, selected_controls):
    """Stage 3: Panel DML with Two-Way FE (Country + Year)."""
    print("\n[Stage 3] Panel DML (Country FE + Year FE)...")
    
    df_fe = df.copy()
    
    # Step 1: Country FE via within-transform
    feature_cols = [TARGET, TREATMENT] + selected_controls
    for col in feature_cols:
        df_fe[f'{col}_cfe'] = df_fe[col] - df_fe.groupby('country')[col].transform('mean')
    
    # Step 2: Year FE via dummies (added to W)
    year_dummies = create_year_dummies(df_fe)
    print(f"  Year dummies: {year_dummies.shape[1]} (reference: {df['year'].min()})")
    
    Y = df_fe[f'{TARGET}_cfe'].values
    T = df_fe[f'{TREATMENT}_cfe'].values
    
    # W = selected controls (within-transformed) + year dummies
    W_controls = df_fe[[f'{c}_cfe' for c in selected_controls]].values
    W_years = year_dummies.values
    W = np.hstack([W_controls, W_years])
    
    groups = df['country'].values
    
    # DML with GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    res_Y = np.zeros_like(Y)
    res_T = np.zeros_like(T)
    
    model_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_t = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    
    for train_idx, test_idx in gkf.split(W, Y, groups):
        model_y.fit(W[train_idx], Y[train_idx])
        model_t.fit(W[train_idx], T[train_idx])
        
        res_Y[test_idx] = Y[test_idx] - model_y.predict(W[test_idx])
        res_T[test_idx] = T[test_idx] - model_t.predict(W[test_idx])
    
    # Coefficient
    theta = np.dot(res_T, res_Y) / np.dot(res_T, res_T)
    
    # Cluster-robust SE
    epsilon = res_Y - theta * res_T
    scores = epsilon * res_T
    
    unique_groups = np.unique(groups)
    cluster_scores = np.array([scores[groups == g].sum() for g in unique_groups])
    
    n = len(Y)
    n_clusters = len(unique_groups)
    
    bread = np.dot(res_T, res_T)
    meat = np.sum(cluster_scores**2)
    
    correction = n_clusters / (n_clusters - 1)  # Small sample correction
    var_theta = correction * (1/bread)**2 * meat
    se = np.sqrt(var_theta)
    
    t_stat = theta / se
    df_t = n_clusters - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_t))
    t_crit = stats.t.ppf(0.975, df=df_t)
    
    return {
        'spec': 'MAIN_TWFE',
        'theta': theta,
        'se': se,
        'p_val': p_val,
        'ci_low': theta - t_crit * se,
        'ci_high': theta + t_crit * se,
        'n': n,
        'clusters': n_clusters
    }

def run_subsample(df, selected_controls, name, condition):
    sub_df = df[condition].copy()
    if len(sub_df) < 100:
        return None
    res = stage3_panel_dml_twfe(sub_df, selected_controls)
    res['spec'] = name
    return res

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Load
    df, valid_controls = load_data()
    
    # Stage 1
    selected_controls = stage1_lasso(df, valid_controls)
    
    # Stage 2
    threshold = stage2_threshold(df, selected_controls)
    
    # Stage 3: Main
    res_main = stage3_panel_dml_twfe(df, selected_controls)
    
    # Subsamples
    oecd_list = ['USA', 'DEU', 'JPN', 'GBR', 'FRA', 'CAN', 'AUS', 'KOR', 'ITA', 'ESP',
                 'NLD', 'CHE', 'SWE', 'NOR', 'DNK', 'FIN', 'AUT', 'BEL', 'IRL', 'NZL']
    oecd_mask = df['country'].isin(oecd_list)
    
    res_oecd = run_subsample(df, selected_controls, 'OECD_TWFE', oecd_mask)
    res_nonoecd = run_subsample(df, selected_controls, 'NONOECD_TWFE', ~oecd_mask)
    
    # Combine
    results = [res_main]
    if res_oecd: results.append(res_oecd)
    if res_nonoecd: results.append(res_nonoecd)
    
    res_df = pd.DataFrame(results)
    res_df['threshold_est'] = threshold
    res_df['fe_type'] = 'Country + Year (TWFE)'
    res_df['crossfit'] = 'GroupKFold by Country (K=5)'
    res_df['se_type'] = 'Cluster by Country (G/(G-1) correction)'
    
    # Save
    save_path = os.path.join(RESULTS_DIR, 'results_summary_v2.csv')
    res_df.to_csv(save_path, index=False)
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS (TWFE, Unit-Corrected)")
    print("=" * 50)
    print(res_df[['spec', 'theta', 'se', 'p_val', 'ci_low', 'ci_high', 'n', 'clusters']].to_string(index=False))
    print(f"\nExploratory Turning Region: ~{threshold:.1f}%")
    print(f"✓ Saved to {save_path}")

if __name__ == "__main__":
    main()
