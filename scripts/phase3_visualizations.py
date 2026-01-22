"""
Phase 3: Publication-Quality Visualizations
=============================================
创建顶刊级别的可视化图表：
1. The Divide Plot - CATE vs Control of Corruption
2. CATE Distribution Histogram
3. World Map of Average CATE by Country (optional)
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
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
CATE_FILE = os.path.join(RESULTS_DIR, 'causal_forest_cate.csv')

# Create figures directory if needed
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})


def plot_divide_scatter(df, save_path):
    """
    Figure 1: The Digital Decarbonization Divide
    X-axis: Institutional Quality (Control of Corruption)
    Y-axis: Marginal Effect of ICT on CO2 (CATE)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create color palette based on significance and direction
    def get_color(row):
        if row['Significant'] and row['CATE'] < 0:
            return 'green'  # Significant negative effect
        elif row['Significant'] and row['CATE'] > 0:
            return 'red'    # Significant positive effect
        else:
            return 'gray'   # Not significant
    
    df['color'] = df.apply(get_color, axis=1)
    
    # Plot points
    for color, label in [('green', 'ICT Reduces CO₂ (Significant)'), 
                         ('red', 'ICT Increases CO₂ (Significant)'),
                         ('gray', 'Not Significant')]:
        subset = df[df['color'] == color]
        if len(subset) > 0:
            ax.scatter(subset['Control_of_Corruption'], subset['CATE'],
                      c=color, alpha=0.6, s=40, label=label, edgecolors='white', linewidth=0.5)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add trend line
    z = np.polyfit(df['Control_of_Corruption'], df['CATE'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Control_of_Corruption'].min(), df['Control_of_Corruption'].max(), 100)
    ax.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.8, label='Linear Trend')
    
    # Add shaded region for "divide"
    ax.fill_between(x_line, p(x_line) - 1.5, p(x_line) + 1.5, alpha=0.15, color='blue')
    
    # Labels and title
    ax.set_xlabel('Institutional Quality\n(Control of Corruption Index)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Marginal Effect of ICT on CO₂ Emissions\n(CATE)', fontsize=13, fontweight='bold')
    ax.set_title('The Digital Decarbonization Divide:\nInstitutional Quality Moderates ICT\'s Climate Impact', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add annotation
    corr = df['Control_of_Corruption'].corr(df['CATE'])
    ax.text(0.02, 0.02, f'Correlation: r = {corr:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   ✓ Saved: {save_path}")


def plot_divide_scatter_gdp(df, save_path):
    """
    Figure 1b: The Divide by GDP
    X-axis: GDP per capita (log scale)
    Y-axis: CATE
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Log-transform GDP for better visualization
    df['log_GDP'] = np.log10(df['GDP_per_capita_constant'] + 1)
    
    # Color by significance
    colors = ['green' if (row['Significant'] and row['CATE'] < 0) 
              else 'red' if (row['Significant'] and row['CATE'] > 0)
              else 'gray' for _, row in df.iterrows()]
    
    ax.scatter(df['log_GDP'], df['CATE'], c=colors, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    
    # Zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Trend line
    z = np.polyfit(df['log_GDP'], df['CATE'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['log_GDP'].min(), df['log_GDP'].max(), 100)
    ax.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.8)
    ax.fill_between(x_line, p(x_line) - 1.5, p(x_line) + 1.5, alpha=0.15, color='blue')
    
    ax.set_xlabel('GDP per Capita (log₁₀ scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Marginal Effect of ICT on CO₂ Emissions\n(CATE)', fontsize=13, fontweight='bold')
    ax.set_title('The Digital Decarbonization Divide:\nEconomic Development Moderates ICT\'s Climate Impact', 
                 fontsize=15, fontweight='bold', pad=20)
    
    corr = df['log_GDP'].corr(df['CATE'])
    ax.text(0.02, 0.02, f'Correlation: r = {corr:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   ✓ Saved: {save_path}")


def plot_cate_distribution(df, save_path):
    """
    Figure 2: CATE Distribution Histogram
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(df['CATE'], bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    
    # Color negative vs positive
    for i in range(len(patches)):
        if bins[i] < 0:
            patches[i].set_facecolor('green')
            patches[i].set_alpha(0.7)
        else:
            patches[i].set_facecolor('red')
            patches[i].set_alpha(0.7)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    # Add mean line
    mean_cate = df['CATE'].mean()
    ax.axvline(x=mean_cate, color='blue', linestyle='-', linewidth=2, label=f'Mean = {mean_cate:.3f}')
    
    ax.set_xlabel('CATE (Conditional Average Treatment Effect)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Heterogeneous Treatment Effects\nICT → CO₂ Emissions', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Add statistics box
    stats_text = f"N = {len(df)}\nMean = {df['CATE'].mean():.3f}\nMedian = {df['CATE'].median():.3f}\nStd = {df['CATE'].std():.3f}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   ✓ Saved: {save_path}")


def plot_country_average(df, save_path):
    """
    Figure 3: Country-level Average CATE (Bar Chart)
    Top 15 and Bottom 15 countries
    """
    # Calculate country averages
    country_avg = df.groupby('country').agg({
        'CATE': 'mean',
        'CATE_LB': 'mean',
        'CATE_UB': 'mean',
        'Significant': 'mean'
    }).reset_index()
    country_avg.columns = ['Country', 'Mean_CATE', 'Mean_LB', 'Mean_UB', 'Sig_Pct']
    country_avg = country_avg.sort_values('Mean_CATE')
    
    # Select top 10 (most negative) and bottom 10 (most positive)
    top_10 = country_avg.head(10)
    bottom_10 = country_avg.tail(10)
    selected = pd.concat([top_10, bottom_10])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['green' if x < 0 else 'red' for x in selected['Mean_CATE']]
    
    bars = ax.barh(range(len(selected)), selected['Mean_CATE'], color=colors, alpha=0.8)
    
    # Add error bars
    xerr = [(selected['Mean_CATE'] - selected['Mean_LB']).values, 
            (selected['Mean_UB'] - selected['Mean_CATE']).values]
    ax.errorbar(selected['Mean_CATE'], range(len(selected)), xerr=xerr, fmt='none', color='black', capsize=3)
    
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected['Country'])
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('Average CATE (ICT → CO₂ Effect)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Country', fontsize=13, fontweight='bold')
    ax.set_title('Country-Level Heterogeneity in ICT\'s Climate Impact\n(Top 10 Reducers vs Top 10 Increasers)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   ✓ Saved: {save_path}")


def plot_moderator_effects(df, save_path):
    """
    Figure 4: Multi-panel showing CATE vs different moderators
    """
    moderators = [
        ('Control_of_Corruption', 'Institutional Quality'),
        ('GDP_per_capita_constant', 'GDP per Capita'),
        ('Renewable_energy_consumption_pct', 'Renewable Energy (%)'),
        ('Energy_use_per_capita', 'Energy Use per Capita')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (col, label) in enumerate(moderators):
        ax = axes[idx]
        
        # Handle log transform for GDP
        if 'GDP' in col:
            x_data = np.log10(df[col] + 1)
            xlabel = f'{label} (log₁₀)'
        else:
            x_data = df[col]
            xlabel = label
        
        # Scatter
        colors = ['green' if (row['Significant'] and row['CATE'] < 0) 
                  else 'red' if (row['Significant'] and row['CATE'] > 0)
                  else 'gray' for _, row in df.iterrows()]
        ax.scatter(x_data, df['CATE'], c=colors, alpha=0.5, s=30, edgecolors='white', linewidth=0.3)
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        
        # Trend line
        z = np.polyfit(x_data, df['CATE'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.8)
        
        # Correlation
        corr = x_data.corr(df['CATE'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('CATE (ICT → CO₂)', fontsize=11)
        ax.set_title(f'Moderation by {label}', fontsize=13, fontweight='bold')
    
    plt.suptitle('Sources of Heterogeneity in Digital Decarbonization', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   ✓ Saved: {save_path}")


def run_visualizations():
    print("=" * 70)
    print("Phase 3: Publication-Quality Visualizations")
    print("=" * 70)
    
    # Load CATE results
    print("\n📂 Loading CATE results...")
    df = pd.read_csv(CATE_FILE)
    print(f"   Loaded {len(df)} observations with CATE predictions")
    
    # Generate figures
    print("\n🎨 Generating publication-quality figures...\n")
    
    # Figure 1: Main Divide Plot (Control of Corruption)
    plot_divide_scatter(df, os.path.join(FIGURES_DIR, 'divide_plot_institution.png'))
    
    # Figure 1b: Divide by GDP
    plot_divide_scatter_gdp(df, os.path.join(FIGURES_DIR, 'divide_plot_gdp.png'))
    
    # Figure 2: CATE Distribution
    plot_cate_distribution(df, os.path.join(FIGURES_DIR, 'cate_distribution.png'))
    
    # Figure 3: Country-level bar chart
    plot_country_average(df, os.path.join(FIGURES_DIR, 'country_average_cate.png'))
    
    # Figure 4: Multi-moderator panel
    plot_moderator_effects(df, os.path.join(FIGURES_DIR, 'moderator_effects_panel.png'))
    
    print("\n" + "=" * 70)
    print("✅ All visualizations generated successfully!")
    print(f"   Output directory: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_visualizations()
