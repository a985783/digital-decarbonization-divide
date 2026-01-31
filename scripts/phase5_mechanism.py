"""
Phase 5: Mechanism Analysis
===========================
Investigating the "Renewable Paradox": why high renewable energy countries
show weaker DCI effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
INPUT_FILE = os.path.join(RESULTS_DIR, "causal_forest_cate.csv")
OUTPUT_FIGURE = os.path.join(FIGURES_DIR, "mechanism_renewable_curve.png")

def analyze_renewable_mechanism():
    print("=" * 70)
    print("Phase 5: Mechanism Analysis - Renewable Energy Paradox")
    print("=" * 70)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("   Please run Phase 2 (Causal Forest) first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    moderator = "Renewable_energy_consumption_pct"
    if moderator not in df.columns:
        print(f"❌ Moderator {moderator} not found in results.")
        return
        
    print(f"Loaded {len(df)} observations.")
    
    # 1. Curve Fitting
    # We use a simple quadratic fit for visualization
    # CATE ~ beta0 + beta1 * Ren + beta2 * Ren^2
    
    X = df[moderator]
    y = df["CATE"]
    
    # Sort for plotting
    sort_idx = np.argsort(X)
    X_sorted = X.iloc[sort_idx]
    y_sorted = y.iloc[sort_idx]
    
    # Lowess smoothing
    lowess = sm.nonparametric.lowess
    z = lowess(y_sorted, X_sorted, frac=0.3)
    
    # 2. Plotting
    plt.figure(figsize=(10, 6))
    
    # Scatter points colored by Significant
    sns.scatterplot(
        data=df, 
        x=moderator, 
        y="CATE", 
        hue="Significant", 
        palette={True: "red", False: "gray"}, 
        alpha=0.6,
        s=40
    )
    
    # Add Lowess curve
    plt.plot(z[:, 0], z[:, 1], "b-", linewidth=3, label="LOWESS Trend")
    
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    
    plt.title("Mechanism: Diminishing Returns of Digitalization in Clean Grids")
    plt.xlabel("Renewable Energy Consumption (% of total)")
    plt.ylabel("DCI Effect on CO2 (CATE)")
    plt.legend(title="Effect Significant?")
    plt.grid(True, alpha=0.3)
    
    # Annotations
    plt.text(
        x=X.min(), 
        y=y.min(), 
        s="Hypothesis: In clean grids,\ndigital efficiency saves\nless carbon.", 
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300)
    print(f"🖼️ Figure saved to {OUTPUT_FIGURE}")
    
    # 3. Statistical Test (Interaction)
    # Check if interaction is significant controlling for GDP
    # Model: CATE ~ Renewable + GDP
    
    print("\n📊 Statistical Verification:")
    X_reg = df[[moderator, "GDP_per_capita_constant"]]
    X_reg = sm.add_constant(X_reg)
    model = sm.OLS(y, X_reg).fit()
    
    print(model.summary().tables[1])
    
    coef_ren = model.params[moderator]
    p_ren = model.pvalues[moderator]
    
    if p_ren < 0.05 and coef_ren > 0:
        print(f"\n✅ Result: Significant POSITIVE relationship (coeff={coef_ren:.4f}, p={p_ren:.4f})")
        print("   This confirms that as Renewable Share increases, the reduction effect (negative CATE) gets weaker (closer to zero).")
    elif p_ren < 0.05 and coef_ren < 0:
        print(f"\n❓ Result: Significant NEGATIVE relationship (coeff={coef_ren:.4f}, p={p_ren:.4f})")
        print("   This contradicts the hypothesis.")
    else:
        print(f"\n❌ Result: No significant relationship detected (p={p_ren:.4f}).")

if __name__ == "__main__":
    analyze_renewable_mechanism()
