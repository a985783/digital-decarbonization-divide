"""
Phase 4: Placebo Tests (Randomization Inference)
================================================
Checks if the estimated effects are driven by spurious correlations.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

try:
    from econml.dml import CausalForestDML
    import xgboost as xgb
except ImportError:
    # Fail gracefully if dependencies missing (though they should be present)
    raise ImportError("Please install econml and xgboost")

# Constants
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
INPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")
PLACEBO_RESULTS_FILE = os.path.join(RESULTS_DIR, "phase4_placebo_results.csv")
PLACEBO_FIGURE_FILE = os.path.join(FIGURES_DIR, "placebo_distribution.png")

os.makedirs(FIGURES_DIR, exist_ok=True)

def fit_causal_forest(Y, T, X, W, groups, n_estimators=500, random_state=42):
    """
    Fits a Causal Forest and returns the Average Treatment Effect (ATE).
    Using fewer estimators for placebo runs to save time.
    """
    # Configure Causal Forest
    est = CausalForestDML(
        model_y=xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1, verbosity=0),
        model_t=xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1, verbosity=0),
        n_estimators=n_estimators,
        min_samples_leaf=10,
        max_depth=6,
        random_state=random_state,
        discrete_treatment=False,
        n_jobs=-1,
        cv=GroupKFold(n_splits=5), 
    )
    
    est.fit(Y, T, X=X, W=W, groups=groups)
    
    # Calculate ATE on the sample
    # effect(X) returns CATE for each observation. Mean of CATE is ATE.
    cates = est.effect(X)
    ate = np.mean(cates)
    
    return ate

def run_placebo_analysis(
    n_iterations=100,
    n_estimators=200,
    output_csv_path=PLACEBO_RESULTS_FILE,
    output_figure_path=PLACEBO_FIGURE_FILE,
    random_state=42,
):
    print("=" * 70)
    print("Phase 4: Placebo Tests (Randomization Inference)")
    print(f"Running {n_iterations} iterations with {n_estimators} trees each.")
    print("=" * 70)
    rng = np.random.default_rng(random_state)

    # 1. Load Data
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    
    # Prepare analysis vectors
    # We need a clean way to get Y, T, X, W. 
    # Note: prepare_analysis_data filters NaNs.
    df = df.dropna(subset=[cfg["outcome"]])
    y, t, x, w, df_clean = prepare_analysis_data(df, cfg, return_df=True)
    groups = df_clean[cfg["groups"]].values
    
    # Ensure no NaNs (prepare_analysis_data might return arrays with same length as df_clean)
    mask = ~np.isnan(y) & ~np.isnan(t)
    y = y[mask]
    t = t[mask]
    x = x[mask]
    w = w[mask]
    groups = groups[mask]
    
    print(f"Sample size: {len(y)}")

    # 2. True Estimate
    print("\n🌲 Computing True Estimate...")
    true_ate = fit_causal_forest(y, t, x, w, groups, n_estimators=2000) # Use full power for true
    print(f"   True ATE: {true_ate:.6f}")

    # 3. Placebo Loop
    print(f"\n🎲 Running {n_iterations} Placebo Iterations...")
    placebo_ates = []
    
    for i in range(n_iterations):
        if (i+1) % 10 == 0:
            print(f"   Iteration {i+1}/{n_iterations}...")
            
        # Permute Treatment T
        # We shuffle T randomly, breaking the link with Y, X, W
        t_shuffled = rng.permutation(t)
        
        # Fit model on shuffled data
        ate_null = fit_causal_forest(y, t_shuffled, x, w, groups, 
                                    n_estimators=n_estimators, 
                                    random_state=42+i)
        placebo_ates.append(ate_null)

    # 4. Results & Visualization
    results = {
        "true_ate": true_ate,
        "placebo_ates": placebo_ates
    }
    
    # Save results
    df_res = pd.DataFrame({
        "iteration": range(1, n_iterations + 1),
        "placebo_ate": placebo_ates
    })
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_res.to_csv(output_csv_path, index=False)
    print(f"\n💾 Results saved to {output_csv_path}")
    
    # Calculate p-value (two-sided)
    # Proportion of placebos with absolute effect >= absolute true effect
    abs_true = abs(true_ate)
    abs_nulls = np.abs(placebo_ates)
    p_val = np.mean(abs_nulls >= abs_true)
    
    print(f"\n📊 Pseudo p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("🟢 RESULT ROBUST: True effect is unlikely to be spurious.")
    else:
        print("🔴 CAUTION: Result might be indistinguishable from noise.")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(placebo_ates, kde=True, color='gray', label='Placebo Distribution')
    plt.axvline(true_ate, color='red', linestyle='--', linewidth=2, label=f'True ATE ({true_ate:.3f})')
    
    # Add confidence interval of placebos
    lower_ci = np.percentile(placebo_ates, 2.5)
    upper_ci = np.percentile(placebo_ates, 97.5)
    plt.axvline(lower_ci, color='blue', linestyle=':', label='95% Null CI')
    plt.axvline(upper_ci, color='blue', linestyle=':')
    
    plt.title(f"Placebo Test: Randomization Inference (N={n_iterations})\np-value: {p_val:.3f}")
    plt.xlabel("Average Treatment Effect (ATE)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_figure_path), exist_ok=True)
    plt.savefig(output_figure_path, dpi=300)
    print(f"🖼️ Figure saved to {output_figure_path}")
    
    return results

if __name__ == "__main__":
    # For actual run, maybe use 100 iterations and 200 trees to be reasonably fast but valid
    run_placebo_analysis(n_iterations=100, n_estimators=200)
