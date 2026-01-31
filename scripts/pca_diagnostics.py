"""
PCA Diagnostics for DCI Construction
=====================================
Validates the Domestic Digital Capacity (DCI) index construction.

Reports:
1. Eigenvalues and explained variance ratio
2. Factor loadings table
3. Convergent validity with other digitalization indices
4. Alternative DCI construction comparison (equal weights vs FA)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats

from scripts.analysis_config import load_config

DATA_DIR = "data"
RESULTS_DIR = "results"
INPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "pca_diagnostics.csv")
LOADINGS_FILE = os.path.join(RESULTS_DIR, "pca_loadings.csv")


def run_pca_diagnostics():
    """Run comprehensive PCA diagnostics for DCI construction."""
    print("=" * 70)
    print("PCA Diagnostics for DCI Construction")
    print("=" * 70)
    
    # Load config and data
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    
    dci_components = cfg["dci_components"]
    print(f"\n📊 DCI Components: {dci_components}")
    
    # Extract DCI component data
    X = df[dci_components].dropna()
    print(f"   Valid observations: {len(X)}")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # =========================================================================
    # 1. Full PCA (all components)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. EIGENVALUES AND EXPLAINED VARIANCE")
    print("=" * 70)
    
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    
    eigenvalues = pca_full.explained_variance_
    variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    
    print("\n   Component | Eigenvalue | Variance % | Cumulative %")
    print("   " + "-" * 55)
    for i, (ev, vr, cv) in enumerate(zip(eigenvalues, variance_ratio, cumulative_variance)):
        print(f"   PC{i+1:8d} | {ev:10.4f} | {vr*100:9.2f}% | {cv*100:10.2f}%")
    
    # Kaiser criterion check
    n_kaiser = sum(eigenvalues > 1)
    print(f"\n   Kaiser Criterion (eigenvalue > 1): {n_kaiser} component(s) retained")
    
    # =========================================================================
    # 2. Factor Loadings
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. FACTOR LOADINGS (PC1)")
    print("=" * 70)
    
    pca_1 = PCA(n_components=1, random_state=42)
    pca_1.fit(X_scaled)
    
    loadings = pca_1.components_[0]
    
    print("\n   Variable                          | Loading  | Loading²")
    print("   " + "-" * 55)
    for var, load in zip(dci_components, loadings):
        print(f"   {var:35s} | {load:8.4f} | {load**2:8.4f}")
    
    # Communality
    communality = sum(loadings**2)
    print(f"\n   Total communality (PC1): {communality:.4f}")
    print(f"   Explained variance ratio: {pca_1.explained_variance_ratio_[0]*100:.2f}%")
    
    # =========================================================================
    # 3. Convergent Validity
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. CONVERGENT VALIDITY")
    print("=" * 70)
    
    # Compute DCI
    dci_scores = pca_1.fit_transform(X_scaled).flatten()
    dci_standardized = (dci_scores - dci_scores.mean()) / dci_scores.std()
    
    # Create temporary dataframe with DCI
    df_valid = df.loc[X.index].copy()
    df_valid['DCI_computed'] = dci_standardized
    
    # Check correlation with related variables
    validity_vars = [
        'Mobile_cellular_subscriptions',
        'GDP_per_capita_constant',
        'Tertiary_enrollment',
        'Research_and_development_expenditure_pct_GDP'
    ]
    
    print("\n   Correlation with related constructs:")
    print("   " + "-" * 50)
    validity_results = []
    for var in validity_vars:
        if var in df_valid.columns:
            valid_data = df_valid[['DCI_computed', var]].dropna()
            if len(valid_data) > 30:
                corr, pval = stats.pearsonr(valid_data['DCI_computed'], valid_data[var])
                print(f"   DCI × {var[:30]:30s} | r = {corr:6.3f} (p={pval:.4f})")
                validity_results.append({'Variable': var, 'Correlation': corr, 'P_value': pval})
    
    # =========================================================================
    # 4. Alternative Construction Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. ALTERNATIVE DCI CONSTRUCTIONS")
    print("=" * 70)
    
    # Method 1: PCA (current)
    dci_pca = dci_standardized
    
    # Method 2: Equal weights
    X_equal = X_scaled.mean(axis=1)
    dci_equal = (X_equal - X_equal.mean()) / X_equal.std()
    
    # Method 3: Factor Analysis
    try:
        fa = FactorAnalysis(n_components=1, random_state=42)
        dci_fa = fa.fit_transform(X_scaled).flatten()
        dci_fa = (dci_fa - dci_fa.mean()) / dci_fa.std()
        fa_success = True
    except:
        dci_fa = dci_pca  # fallback
        fa_success = False
    
    # Correlations between methods
    corr_pca_equal = np.corrcoef(dci_pca, dci_equal)[0, 1]
    corr_pca_fa = np.corrcoef(dci_pca, dci_fa)[0, 1] if fa_success else np.nan
    
    print("\n   Method Comparison:")
    print(f"   PCA vs Equal Weights:    r = {corr_pca_equal:.4f}")
    if fa_success:
        print(f"   PCA vs Factor Analysis:  r = {corr_pca_fa:.4f}")
    
    # =========================================================================
    # 5. Internal Consistency (Cronbach's Alpha)
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. INTERNAL CONSISTENCY")
    print("=" * 70)
    
    # Cronbach's alpha
    n_items = X_scaled.shape[1]
    item_variances = X_scaled.var(axis=0, ddof=1).sum()
    total_variance = X_scaled.sum(axis=1).var(ddof=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_variances / total_variance)
    
    print(f"\n   Cronbach's Alpha: {alpha:.4f}")
    if alpha >= 0.7:
        print("   ✅ Good internal consistency (α ≥ 0.7)")
    elif alpha >= 0.6:
        print("   ⚠️  Acceptable internal consistency (0.6 ≤ α < 0.7)")
    else:
        print("   ❌ Poor internal consistency (α < 0.6)")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Main diagnostics
    diagnostics_df = pd.DataFrame({
        'Metric': [
            'Explained_Variance_PC1',
            'Eigenvalue_PC1',
            'Kaiser_Components',
            'Cronbachs_Alpha',
            'Corr_PCA_EqualWeights',
            'Corr_PCA_FA',
            'N_Observations'
        ],
        'Value': [
            pca_1.explained_variance_ratio_[0],
            eigenvalues[0],
            n_kaiser,
            alpha,
            corr_pca_equal,
            corr_pca_fa if fa_success else np.nan,
            len(X)
        ]
    })
    diagnostics_df.to_csv(OUTPUT_FILE, index=False)
    print(f"   💾 Diagnostics saved to: {OUTPUT_FILE}")
    
    # Factor loadings
    loadings_df = pd.DataFrame({
        'Component': dci_components,
        'Loading': loadings,
        'Loading_Squared': loadings**2
    })
    loadings_df.to_csv(LOADINGS_FILE, index=False)
    print(f"   💾 Loadings saved to: {LOADINGS_FILE}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    ev_ratio = pca_1.explained_variance_ratio_[0]
    if ev_ratio >= 0.6:
        print(f"\n   ✅ PC1 explains {ev_ratio*100:.1f}% of variance (good)")
    elif ev_ratio >= 0.5:
        print(f"\n   ⚠️  PC1 explains {ev_ratio*100:.1f}% of variance (acceptable)")
    else:
        print(f"\n   ❌ PC1 explains {ev_ratio*100:.1f}% of variance (weak)")
    
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
    
    # 6.1 Scree Plot
    plt.figure(figsize=(10, 6))
    x_comps = range(1, len(eigenvalues) + 1)
    plt.bar(x_comps, eigenvalues, alpha=0.6, color='steelblue', label='Eigenvalue')
    plt.plot(x_comps, eigenvalues, 'ro-', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (Eigenvalue=1)')
    plt.title('Scree Plot: DCI Components', fontsize=14)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.xticks(x_comps)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    scree_path = os.path.join(FIGURES_DIR, "pca_scree_plot.png")
    plt.savefig(scree_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 Scree Plot saved to: {scree_path}")
    
    # 6.2 Loadings Plot
    plt.figure(figsize=(10, 6))
    loadings_sorted = sorted(zip(dci_components, loadings), key=lambda x: abs(x[1]), reverse=True)
    vars_sorted, loads_sorted = zip(*loadings_sorted)
    
    colors = ['steelblue' if l > 0 else 'firebrick' for l in loads_sorted]
    plt.barh(vars_sorted, loads_sorted, color=colors, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.title('Factor Loadings (PC1)', fontsize=14)
    plt.xlabel('Loading Coefficient', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    loadings_path = os.path.join(FIGURES_DIR, "pca_loadings_plot.png")
    plt.savefig(loadings_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   💾 Loadings Plot saved to: {loadings_path}")

    print("=" * 70)
    
    return diagnostics_df, loadings_df


if __name__ == "__main__":
    run_pca_diagnostics()
