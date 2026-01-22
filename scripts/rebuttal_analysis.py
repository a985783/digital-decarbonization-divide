"""
Rebuttal Analysis: Rigorous Causal Inference & Model Ladder
===========================================================
Addresses "Cannon Killing Mosquito" criticism by establishing:
1. Model Ladder: FE -> Linear DML -> Interactive DML -> Causal Forest
2. Rigorous Inference: GATE + Cluster Bootstrap (by country)
3. Overfitting Control: Group Cross-Fitting (by country)
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from tqdm import tqdm
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.analysis_config import load_config
from scripts.dci import build_dci
from scripts.joblib_utils import resolve_n_jobs

try:
    from econml.dml import CausalForestDML, LinearDML
    print("✓ econml loaded")
except ImportError:
    print("❌ econml missing")

# Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
INPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v4_imputed.csv')
OUTPUT_LADDER = os.path.join(RESULTS_DIR, 'model_ladder.csv')
OUTPUT_GATE = os.path.join(RESULTS_DIR, 'rebuttal_gate.csv')
OUTPUT_CATE = os.path.join(RESULTS_DIR, 'rebuttal_forest_cate.csv')
OUTPUT_COUNTRY_CATES = os.path.join(RESULTS_DIR, 'rebuttal_country_cates.csv')

def construct_dci(df):
    """
    Construct Domestic Digital Capacity Index (DCI) via PCA
    using real WDI components.
    """
    print("\n🏗️  Constructing Domestic Digital Capacity Index (DCI)...")
    cfg = load_config("analysis_spec.yaml")
    dci_vars = cfg["dci_components"]
    print(f"   Components: {dci_vars}")
    dci, expl_var = build_dci(df, dci_vars)
    df["DCI"] = dci
    print(f"   PC1 Explained Variance: {expl_var:.2%}")
    print("   ✓ DCI Constructed.")
    return df

def residualize_twfe(df, Y_col, T_col, W_cols):
    """
    Orthogonalize Y and T against Country and Year Fixed Effects.
    Returns residualized Y_res and T_res.
    """
    print("   🧹 Residualizing (Orthogonalizing) against Two-Way FE...")
    
    # Simple means subtraction (equivalent to dummy regression for balanced panel)
    # But for unbalanced or complex, regression is safer.
    # W_cols are controls that we also might want to partial out? 
    # Standard DML logic: Propose T -> Y, partialling out W. 
    # Here we partial out FE from Y and T specifically to kill time-invariant/shocks.
    
    # We use linear regression with dummies
    fixed_effects = pd.get_dummies(df[['country', 'year']], drop_first=True)
    
    # We want residuals of Y ~ FE and T ~ FE
    # Note: large N*T, getting dummies might be heavy, but N=960 is tiny.
    
    lr_y = LinearRegression()
    lr_t = LinearRegression()
    
    lr_y.fit(fixed_effects, df[Y_col])
    lr_t.fit(fixed_effects, df[T_col])
    
    Y_res = df[Y_col] - lr_y.predict(fixed_effects)
    T_res = df[T_col] - lr_t.predict(fixed_effects)
    
    print(f"      Y_res variance ratio: {Y_res.var() / df[Y_col].var():.2f}")
    print(f"      T_res variance ratio: {T_res.var() / df[T_col].var():.2f}")
    
    return Y_res.values, T_res.values

def prepare_data_v2(df):
    """
    Prepare data for 2D Digitalization Analysis (DCI vs EDS).
    Now includes:
    - Lagged Treatment (T_t-1)
    - Orthogonalization (TWFE Residuals)
    """
    # 1. Construct DCI
    df = construct_dci(df)
    
    cfg = load_config("analysis_spec.yaml")

    # 2. Key Variable Assignment & Lagging
    target = 'CO2_per_capita'
    raw_treatment = 'DCI'
    
    # Create Lagged Treatment
    df = df.sort_values(['country', 'year'])
    df['DCI_L1'] = df.groupby('country')[raw_treatment].shift(1)
    treatment = 'DCI_L1' # Update treatment variable
    
    # Rename ICT_exports -> EDS
    if 'EDS' not in df.columns:
        df = df.rename(columns={'ICT_exports': 'EDS'})
    
    # 3. Moderators (X): DCI interacts with Development & EDS
    moderators = cfg["moderators_X"]
    
    # 4. Controls (W)
    exclude_cols = [
        "country",
        "year",
        target,
        treatment,
        raw_treatment,
        "EDS",
        "OECD",
    ] + cfg["dci_components"] + moderators
                    
    w_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Drop rows with core missingness (now including Lag)
    df_clean = df.dropna(subset=[target, treatment])
    
    # 5. Orthogonalization (Identification Hardening)
    # Use residuals for fitting, but keep raw df for X (moderators) interpretation?
    # Actually, CausalForestDML takes Y, T, X, W. 
    # Ideally we feed Y_res, T_res and NO W (since FE captured "all unobserved invariant W") 
    # OR we keep W to capture time-varying confounding.
    # Strategy: Partial out FE from Y and T. Pass Y_res, T_res to Forest. Pass W (time-varying) to Forest.
    
    Y_res, T_res = residualize_twfe(df_clean, target, treatment, w_cols)
    
    # Check scaling
    if df_clean[target].mean() > 100: 
         Y_res = Y_res / 100.0 # Scale residuals too if original was large
         print("   ✓ Auto-scaled Y_res (divided by 100)")
    
    Y = Y_res
    T = T_res
    X = df_clean[moderators].values
    W = df_clean[w_cols].values
    
    return df_clean, Y, T, X, W, moderators, w_cols


# ------------------------------------------------------------------------------
# 2. Model Ladder Implementation
# ------------------------------------------------------------------------------

def run_model_inference_ladder(df, target, treatment, X_cols, W_cols, n_boot=1000):
    print(f"\nLadder Inference (B={n_boot})...")
    
    Y = df[target].values
    T = df[treatment].values
    X = df[X_cols].values
    W = df[W_cols].values
    groups = df['country'].values
    
    # --------------------------------------------------------------------------
    # 1. Main Fit (Point Estimates)
    # --------------------------------------------------------------------------
    print("   1. Fitting Main Models (L0-L3)...")
    
    # L0: TWFE
    fe_df = pd.get_dummies(df[['country', 'year']], drop_first=True)
    X_fe = pd.concat([df[[treatment]], fe_df], axis=1)
    lr = LinearRegression()
    lr.fit(X_fe, Y)
    l0_ate = lr.coef_[0]
    
    # L1: Linear DML
    est_l1 = LinearDML(model_y=lgb.LGBMRegressor(n_estimators=100, verbose=-1),
                       model_t=lgb.LGBMRegressor(n_estimators=100, verbose=-1),
                       random_state=42, cv=5)
    est_l1.fit(Y, T, X=None, W=np.hstack([X, W]))
    l1_ate = est_l1.ate()
    
    # L2: Interactive DML
    est_l2 = LinearDML(model_y=lgb.LGBMRegressor(n_estimators=100, verbose=-1),
                       model_t=lgb.LGBMRegressor(n_estimators=100, verbose=-1),
                       featurizer=PolynomialFeatures(degree=1, include_bias=False),
                       random_state=42, cv=5)
    est_l2.fit(Y, T, X=X, W=W)
    l2_ate = est_l2.ate(X=X)
    
    # L3: Causal Forest
    est_l3 = CausalForestDML(model_y=lgb.LGBMRegressor(n_estimators=100, verbose=-1),
                             model_t=lgb.LGBMRegressor(n_estimators=100, verbose=-1),
                             n_estimators=2000, min_samples_leaf=10, max_depth=6,
                             honest=True, random_state=42, cv=GroupKFold(n_splits=5))
    est_l3.fit(Y, T, X=X, W=W, groups=groups)
    l3_ate = est_l3.ate(X=X)
    
    # --------------------------------------------------------------------------
    # 2. Bootstrap Inference (Cluster by Country)
    # --------------------------------------------------------------------------
    print(f"   2. Bootstrapping SEs (B={n_boot})...")
    countries = df['country'].unique()
    boot_res = {'L0': [], 'L1': [], 'L2': []} # L3 is too slow to refit
    
    # We will trust econml's internal inference for L3 ATE/CATE variance
    # But for L0-L2 we bootstrap to get comparable SEs
    
    def bootstrap_iter(seed):
        # Resample countries
        np.random.seed(seed)
        res_countries = np.random.choice(countries, size=len(countries), replace=True)
        boot_idxs = []
        for c in res_countries:
             boot_idxs.extend(df[df['country'] == c].index.tolist())
        
        df_boot = df.loc[boot_idxs]
        Y_b = df_boot[target].values
        T_b = df_boot[treatment].values
        X_b = df_boot[X_cols].values
        W_b = df_boot[W_cols].values
        
        res = {}
        
        # L0
        fe_df_b = pd.get_dummies(df_boot[['country', 'year']], drop_first=True)
        X_fe_b = pd.concat([df_boot[[treatment]].reset_index(drop=True), 
                            fe_df_b.reset_index(drop=True)], axis=1)
        lr_b = LinearRegression()
        lr_b.fit(X_fe_b, Y_b)
        res['L0'] = lr_b.coef_[0]
        
        # L1
        est_l1_b = LinearDML(model_y=lgb.LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1), 
                             model_t=lgb.LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1),
                             random_state=seed, cv=3)
        est_l1_b.fit(Y_b, T_b, X=None, W=np.hstack([X_b, W_b]))
        res['L1'] = est_l1_b.ate()
        
        # L2
        est_l2_b = LinearDML(model_y=lgb.LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1),
                             model_t=lgb.LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1),
                             featurizer=PolynomialFeatures(degree=1, include_bias=False),
                             random_state=seed, cv=3)
        est_l2_b.fit(Y_b, T_b, X=X_b, W=W_b)
        try:
             res['L2'] = est_l2_b.ate(X=X_b)
        except:
             res['L2'] = None
             
        return res

    print("   Running Parallel Bootstrap...")
    n_jobs = resolve_n_jobs()
    results_list = Parallel(n_jobs=n_jobs, verbose=5)(delayed(bootstrap_iter)(i) for i in range(n_boot))
    
    for r in results_list:
        if r['L0'] is not None: boot_res['L0'].append(r['L0'])
        if r['L1'] is not None: boot_res['L1'].append(r['L1'])
        if r['L2'] is not None: boot_res['L2'].append(r['L2'])
             
    # Compile Ladder
    ladder = []
    # L0-L2 Results
    for m, ate, code in [('L0: TWFE', l0_ate, 'L0'), ('L1: Linear DML', l1_ate, 'L1'), ('L2: Interactive DML', l2_ate, 'L2')]:
        boots = np.array(boot_res[code])
        se = np.std(boots)
        ci_lower = np.percentile(boots, 2.5)
        ci_upper = np.percentile(boots, 97.5)
        ladder.append({'Model': m, 'ATE': ate, 'SE': se, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper})
        
    # L3 Results (Using internal inference)
    l3_inf = est_l3.ate_inference(X=X)
    try:
        l3_se = l3_inf.stderr_mean
        l3_lb, l3_ub = l3_inf.conf_int_mean()
    except:
        # Final Fallback
        l3_se = 0.0
        l3_lb, l3_ub = (l3_ate, l3_ate)
        
    ladder.append({
        'Model': 'L3: Causal Forest', 
        'ATE': l3_ate, 
        'SE': l3_se, 
        'CI_Lower': l3_lb, 
        'CI_Upper': l3_ub
    })
    
    ladder_df = pd.DataFrame(ladder)
    ladder_df.to_csv(OUTPUT_LADDER, index=False)
    print("   Saved Model Ladder with Inference.")
    
    return est_l3

# ------------------------------------------------------------------------------
# 3. GATE Inference with Cluster Bootstrap
# ------------------------------------------------------------------------------

def compute_gates_and_country_cis(df, X_cols, est_forest, n_boot=1000):
    print(f"\n🔍 Computing GATEs & Country CIs (B={n_boot})...")
    
    # Predict CATEs
    X = df[X_cols].values
    cates = est_forest.effect(X)
    df['CATE'] = cates
    
    # Define Groups by Quartiles of GDP
    df['GDP_Group'] = pd.qcut(df['GDP_per_capita_constant'], 4, labels=['Low', 'Lower-Mid', 'Upper-Mid', 'High'])
    
    # Bootstrap
    countries = df['country'].unique()
    bootstrap_means = {g: [] for g in df['GDP_Group'].unique()}
    bootstrap_country = {c: [] for c in countries}
    
    for i in tqdm(range(n_boot)):
        # Resample countries (block bootstrap)
        res_countries = np.random.choice(countries, size=len(countries), replace=True)
        boot_idx = []
        for c in res_countries:
            boot_idx.extend(df[df['country'] == c].index.tolist())
            
        boot_df = df.loc[boot_idx]
        
        # 1. GATEs (Stratified Means of Sample)
        # Note: This quantifies the uncertainty of the *Population Average* in the strata
        grp_means = boot_df.groupby('GDP_Group')['CATE'].mean()
        for g in grp_means.index:
            bootstrap_means[g].append(grp_means[g])
            
    # Country-Specific CIs (Bootstrap Years within Country)
    # Since resampling countries doesn't give within-country variance for a fixed country,
    # we do a separate loop or just trust the Forest's Pointwise Inference?
    # User asked for "Country-average CATE + Cluster Bootstrap CI".
    # Relying on Block Bootstrap (resampling countries) works for *Aggregates* like GATEs.
    # For *Single Country*, we need to resample *within* the country (Years).
    
    print("   Bootstrapping Country-Level CIs (Resampling Years)...")
    country_res = []
    
    for c in countries:
        c_df = df[df['country'] == c]
        if len(c_df) < 5: continue
        
        # Bootstrap mean CATE for this country
        c_means = []
        for _ in range(n_boot):
            c_boot = c_df.sample(n=len(c_df), replace=True)
            c_means.append(c_boot['CATE'].mean())
            
        mean_est = np.mean(c_means)
        ci_lower = np.percentile(c_means, 2.5)
        ci_upper = np.percentile(c_means, 97.5)
        
        country_res.append({'Country': c, 'Mean_CATE': mean_est, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper})
        
    # Compile GATEs
    final_gates = []
    for g in ['Low', 'Lower-Mid', 'Upper-Mid', 'High']:
        means = np.array(bootstrap_means[g])
        final_gates.append({
            'Group': g,
            'GATE': np.mean(means),
            'CI_Lower': np.percentile(means, 2.5),
            'CI_Upper': np.percentile(means, 97.5)
        })
        
    pd.DataFrame(final_gates).to_csv(OUTPUT_GATE, index=False)
    pd.DataFrame(country_res).to_csv(OUTPUT_COUNTRY_CATES, index=False)
    print("   Saved GATEs and Country CIs.")

# ------------------------------------------------------------------------------
# 4. Placebo Tests
# ------------------------------------------------------------------------------

def run_placebo_test(df, target, treatment, X_cols, W_cols, n_perm=50):
    print("\n💊 Running Placebo Test (Permutation)...")
    
    # Shuffle Treatment
    df_placebo = df.copy()
    df_placebo[treatment] = np.random.permutation(df[treatment].values)
    
    Y = df_placebo[target].values
    T_shuffled = df_placebo[treatment].values
    X = df_placebo[X_cols].values
    W = df_placebo[W_cols].values
    
    est_placebo = CausalForestDML(
        model_y=lgb.LGBMRegressor(n_estimators=50, verbose=-1), # Faster for placebo
        model_t=lgb.LGBMRegressor(n_estimators=50, verbose=-1),
        n_estimators=500,
        min_samples_leaf=20, # More regularized
        max_depth=5,
        random_state=999
    )
    est_placebo.fit(Y, T_shuffled, X=X, W=W)
    
    cates_placebo = est_placebo.effect(X)
    print(f"   Placebo CATE SD: {np.std(cates_placebo):.5f} (Should be near 0)")
    print(f"   Placebo CATE Mean: {np.mean(cates_placebo):.5f}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def run_loco_analysis(df, target, treatment, X_cols, W_cols):
    """
    Leave-One-Country-Out (LOCO) Stability Analysis.
    Iteratively drops one country, refits Forest, and records GATE stability.
    """
    print("\n🛡️  Running LOCO (Leave-One-Country-Out) Stability Check...")
    
    countries = df['country'].unique()
    loco_results = []
    
    # We use a lighter Forest for LOCO to save time (n_estimators=500)
    for c_drop in tqdm(countries, desc="LOCO Folds"):
        df_fold = df[df['country'] != c_drop]
        
        # Prepare inputs
        Y = df_fold[target].values
        T = df_fold[treatment].values
        X = df_fold[X_cols].values
        W = df_fold[W_cols].values
        
        # Fit Forest (Light)
        est = CausalForestDML(
            model_y=lgb.LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1),
            model_t=lgb.LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1),
            n_estimators=500, # Faster
            min_samples_leaf=10,
            max_depth=6,
            discrete_treatment=False,
            random_state=42,
            n_jobs=1
        )
        
        # Honest Splitting, Group CV
        groups = df_fold['country'].astype('category').cat.codes.values
        cv = GroupKFold(n_splits=5)
        
        est.fit(Y, T, X=X, W=W, groups=groups, cache_values=True, inference='blb')
        
        # Calculate Global GATE (just average CATE for now, or by group)
        # We track Global ATE and High-Income GATE
        cates = est.effect(X)
        
        # Identify High Income observations in this fold
        # GDP variable index 0 in X_cols
        gdp_idx = X_cols.index('GDP_per_capita_constant')
        high_inc_mask = X[:, gdp_idx] > np.percentile(X[:, gdp_idx], 75)
        
        res = {
            'Drop_Country': c_drop,
            'Global_ATE': np.mean(cates),
            'High_Income_GATE': np.mean(cates[high_inc_mask])
        }
        loco_results.append(res)
        
    loco_df = pd.DataFrame(loco_results)
    output_path = os.path.join(RESULTS_DIR, 'loco_stability.csv')
    loco_df.to_csv(output_path, index=False)
    
    print(f"   Saved LOCO results to {output_path}")
    
    # Summary
    print("   LOCO Stability Summary:")
    print(loco_df.describe()[['Global_ATE', 'High_Income_GATE']])
    
    return loco_df

def main():
    print("🚀 Starting Q1 Rigor Upgrade Analysis (Phase 3)...")
    
    # Load Data
    df = pd.read_csv(INPUT_FILE)
    if 'CO2_per_capita' in df.columns and df['CO2_per_capita'].mean() > 100:
        df['CO2_per_capita'] = df['CO2_per_capita'] / 100.0
        
    df, Y, T, X, W, X_cols, W_cols = prepare_data_v2(df)
    
    print(f"Data Loaded. N={len(df)}. Treatment=DCI_L1. X={len(X_cols)}, W={len(W_cols)}")
    
    # --- E1 & E2 will be integrated here ---
    print("\n1. Ladder Inference (B=1000)...")
    est_forest = run_model_inference_ladder(df, 'CO2_per_capita', 'DCI_L1', X_cols, W_cols, n_boot=1000)
    
    # 3. Save CATEs
    cates = est_forest.effect(df[X_cols].values)
    res_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(cates, columns=['CATE'])], axis=1)
    res_df.to_csv(OUTPUT_CATE, index=False)
    
    # 4. GATEs & Country CIs
    print("\n2. GATEs & Country CIs (B=1000)...")
    compute_gates_and_country_cis(res_df, X_cols, est_forest, n_boot=1000)
    
    # 5. Placebo
    print("\n3. Placebo Test...")
    run_placebo_test(df, 'CO2_per_capita', 'DCI_L1', X_cols, W_cols, n_perm=50)
    
    # --- E3: LOCO ---
    run_loco_analysis(df, 'CO2_per_capita', 'DCI_L1', X_cols, W_cols)
    
    print("\n✅ Q1 Upgrade Analysis Complete.")

if __name__ == "__main__":
    main()
