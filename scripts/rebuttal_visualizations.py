"""
Rebuttal Visualizations: Evidence of Necessity
==============================================
Generates visual evidence that "Cannon" (Forest) was needed:
1. Linear vs Forest Overlay (showing non-linearity)
2. Off-Diagonal Outliers (showing policy exceptions)
3. Model Ladder Table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = 'results'
DATA_DIR = 'data' # Added DATA_DIR
FINPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v4_imputed.csv')
INPUT_CATE = os.path.join(RESULTS_DIR, 'rebuttal_forest_cate.csv')
INPUT_GATE = os.path.join(RESULTS_DIR, 'rebuttal_gate.csv')
INPUT_COUNTRY = os.path.join(RESULTS_DIR, 'rebuttal_country_cates.csv')
LADDER_FILE = os.path.join(RESULTS_DIR, 'model_ladder.csv')

FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures') # Moved FIGURES_DIR after RESULTS_DIR definition

os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

def load_data():
    df = pd.read_csv(FINPUT_FILE)
    cates = pd.read_csv(INPUT_CATE)
    gates = pd.read_csv(INPUT_GATE)
    country_cis = pd.read_csv(INPUT_COUNTRY) # Load Country CIs
    return df, cates, gates, country_cis

def load_country_cis():
    country_file = os.path.join(RESULTS_DIR, 'rebuttal_country_cates.csv')
    if os.path.exists(country_file):
        return pd.read_csv(country_file)
    return None

# ------------------------------------------------------------------------------
# 1. Linear vs Forest Prediction Plot
# ------------------------------------------------------------------------------

def plot_linear_vs_forest(df):
    """
    Overlays OLS Interaction prediction vs Forest CATE to show non-linearity.
    """
    print("Generating Figure: Linear vs Forest...")
    
    # 1. Fit OLS Interaction for visualization
    # CATE_linear = a + b * GDP
    # We approximate this by regressing Forest CATE on GDP to see the "Best Linear Fit" of the CATE
    # vs the Actual Forest CATE.
    
    # Log GDP for better x-axis
    df['log_GDP'] = np.log10(df['GDP_per_capita_constant'])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter of Forest Estimates
    sns.scatterplot(
        data=df, x='log_GDP', y='CATE', 
        alpha=0.3, color='gray', s=30, label='Forest Estimates (Points)', ax=ax
    )
    
    # Forest Smooth Curve (Lowess or Bin Mean)
    # Using binning for clarity
    df['GDP_bin'] = pd.qcut(df['log_GDP'], 20)
    bin_means = df.groupby('GDP_bin')[['log_GDP', 'CATE']].mean()
    ax.plot(bin_means['log_GDP'], bin_means['CATE'], 'r-', linewidth=3, label='Forest CATE (Non-Linear)')
    
    # Linear Interaction Fit
    z = np.polyfit(df['log_GDP'], df['CATE'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['log_GDP'].min(), df['log_GDP'].max(), 100)
    ax.plot(x_range, p(x_range), 'b--', linewidth=2, label='Linear Interaction Model')
    
    # Highlight Divergence
    ax.set_xlabel('GDP per Capita (log10)', fontweight='bold')
    ax.set_ylabel('Treatment Effect (DCI -> CO2)', fontweight='bold') # Changed label
    ax.set_title('Why Linear Models Fail: The "Threshold Effect"', fontweight='bold', pad=15)
    ax.legend()
    
    # Annotate Divergence
    ax.text(0.1, 0.1, "Linear model\nmisses non-linearity", transform=ax.transAxes, color='blue')
    ax.text(0.8, 0.1, "Forest detects\nnon-linearities", transform=ax.transAxes, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'linear_vs_forest.png'), dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# 2. Off-Diagonal (Residual) Analysis
# ------------------------------------------------------------------------------

def analyze_off_diagonal(df, cates, country_cis):
    print("\nScatter: Linear vs Forest (Off-Diagonal Analysis)...")
    
    # Merge CATEs with original data
    # Note: df has DCI now? No, input file is raw data.
    # We need to reconstruct DCI or just use Country/Year matching.
    # The cates file has 'CATE' column.
    
    # Actually, simpler: Use Country CIs table directly for the "Exceptions" table
    # And for the plot, we want DCI (x-axis) vs CATE (y-axis) or similar?
    # Original plot: Linear Pred vs Forest Pred.
    # Let's trust the "Verdict" logic from country_cis.
    
    # We will plot: Rank Order of Country CATEs (Forest) with CIs
    # Highlighting USA, CHN, DEU
    
    plt.figure(figsize=(12, 8))
    
    # Sort by Mean CATE
    pdf = country_cis.sort_values('Mean_CATE')
    
    # Top 10 and Bottom 10
    top_bot = pd.concat([pdf.head(15), pdf.tail(15)])
    
    plt.errorbar(x=top_bot['Mean_CATE'], y=top_bot['Country'], 
                 xerr=[top_bot['Mean_CATE'] - top_bot['CI_Lower'], top_bot['CI_Upper'] - top_bot['Mean_CATE']],
                 fmt='o', color='black', ecolor='gray', capsize=3)
                 
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Country-Level CATEs (Domestic Digital Capacity Effect)\n(Sorted by Magnitude)', fontsize=14)
    plt.xlabel('Estimated CATE (Tons CO2 / SD of DCI)', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    # Highlight specific countries
    for i, row in top_bot.iterrows():
        if row['Country'] in ['FIN', 'SWE', 'CHE', 'CAN']:
            plt.text(row['Mean_CATE'], i, f"  {row['Country']}", va='center', fontweight='bold', color='blue')

    output_file = os.path.join(FIGURES_DIR, 'off_diagonal_cis.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"   Saved {output_file}")
    
    # Save Table for Paper (Exceptions)
    # Filter interesting ones
    # Rebound: CI_Lower > 0
    # Strong Reduction: CI_Upper < -1.0
    
    rebound = country_cis[country_cis['CI_Lower'] > 0].copy()
    rebound['Type'] = 'Rebound'
    
    strong_red = country_cis[country_cis['CI_Upper'] < -1.5].copy()
    strong_red['Type'] = 'Strong Reduction'
    
    exceptions = pd.concat([rebound, strong_red]).sort_values('Mean_CATE', ascending=False)
    exceptions.to_csv(os.path.join(FIGURES_DIR, 'off_diagonal_exceptions_table.csv'), index=False)
    print("   Saved Exceptions Table")

# ------------------------------------------------------------------------------
# 3. GATE Plot
# ------------------------------------------------------------------------------

def plot_gate(df_gate): # Renamed function to plot_gate and added df_gate as argument
    if df_gate.empty: # Changed condition to check if df_gate is empty
        print("Skipping GATE plot (data not found or empty)")
        return
        
    print("Generating Figure: GATEs...")
    
    # Order: Low -> High
    order = ['Low', 'Lower-Mid', 'Upper-Mid', 'High']
    df_gate['Group'] = pd.Categorical(df_gate['Group'], categories=order, ordered=True)
    df_gate = df_gate.sort_values('Group')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Error Bar Plot
    x = range(len(df_gate))
    y = df_gate['GATE']
    yerr = [df_gate['GATE'] - df_gate['CI_Lower'], df_gate['CI_Upper'] - df_gate['GATE']]
    
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, linewidth=2, color='darkblue')
    ax.axhline(0, color='gray', linestyle='--')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_gate['Group'])
    ax.set_xlabel('GDP Quartile', fontweight='bold')
    ax.set_ylabel('Average Treatment Effect (DCI -> CO2)', fontweight='bold') # Changed label
    ax.set_title('Group Average Treatment Effects\n(with Country-Cluster Bootstrap CI)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'gate_plot.png'), dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    df, cates, gates, country_cis = load_data()
    
    plot_gate(gates)
    plot_linear_vs_forest(cates) # We can stick with this or remove
    analyze_off_diagonal(df, cates, country_cis) # Updated to use Country CIs
    
    print("\n✅ Visualizations Complete.")

if __name__ == "__main__":
    main()
