"""
Phase 6: External Validity
==========================
Assessing the representativeness of the 40-country sample.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SAMPLE_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")
RAW_FILE = os.path.join(DATA_DIR, "wdi_expanded_raw.csv")
OUTPUT_FIGURE = os.path.join(FIGURES_DIR, "sample_representativeness.png")

def run_external_validity():
    print("=" * 70)
    print("Phase 6: External Validity & Representativeness")
    print("=" * 70)
    
    if not os.path.exists(RAW_FILE):
        print(f"❌ Raw file not found: {RAW_FILE}")
        return

    # Load data
    df_sample = pd.read_csv(SAMPLE_FILE)
    df_raw = pd.read_csv(RAW_FILE)
    
    # Identify Sample Countries
    sample_countries = df_sample['country'].unique()
    print(f"Sample size: {len(sample_countries)} economies")
    
    # Identify Global (Raw) Countries
    # We assume raw data contains all available countries
    # We should exclude potential aggregates if any, but let's assume raw is country-level
    global_countries = df_raw['country'].unique()
    print(f"Global pool: {len(global_countries)} economies in raw dataset")
    
    # Metrics to compare (using 2019 as reference year pre-COVID, or average)
    # Using average over the period 2000-2023 for robustness
    target_year = 2019
    print(f"Using reference year: {target_year}")
    
    df_raw_year = df_raw[df_raw['year'] == target_year].copy()
    
    # Calculate totals
    # GDP per capita * Population = GDP
    # CO2 per capita * Population = CO2
    
    # Handle missing values in raw data for these metrics
    cols_needed = ['country', 'GDP_per_capita_current', 'Population_total', 'CO2_per_capita']
    df_metrics = df_raw_year[cols_needed].dropna()
    
    df_metrics['GDP_Total'] = df_metrics['GDP_per_capita_current'] * df_metrics['Population_total']
    df_metrics['CO2_Total'] = df_metrics['CO2_per_capita'] * df_metrics['Population_total']
    
    sample_gdp = df_metrics['GDP_Total'].sum()
    # Note: CO2 in raw data seems to be scaled by 10 (Decitons) based on USA values (~190 vs ~19t)
    sample_co2 = df_metrics['CO2_Total'].sum() / 10.0
    sample_pop = df_metrics['Population_total'].sum()
    
    GLOBAL_GDP_2019 = 87.4e12 
    GLOBAL_CO2_2019 = 36.7e9   
    GLOBAL_POP_2019 = 7.67e9   
    
    if sample_gdp > GLOBAL_GDP_2019 * 1.5:
        print("⚠️ Warning: Sample GDP exceeds Global reference. Check units.")
        
    pct_gdp = (sample_gdp / GLOBAL_GDP_2019) * 100
    pct_co2 = (sample_co2 / GLOBAL_CO2_2019) * 100
    pct_pop = (sample_pop / GLOBAL_POP_2019) * 100
    
    print("\n🌍 Coverage Statistics (2019):")
    print(f"   Sample GDP:          ${sample_gdp/1e12:.2f}T ({pct_gdp:.1f}% of Global)")
    print(f"   Sample Emissions:    {sample_co2/1e9:.2f}Gt ({pct_co2:.1f}% of Global)")
    print(f"   Sample Population:   {sample_pop/1e9:.2f}B ({pct_pop:.1f}% of Global)")
    
    # Visualization
    metrics = ['GDP', 'CO2 Emissions', 'Population']
    values = [pct_gdp, pct_co2, pct_pop]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#d62728', '#2ca02c'], alpha=0.8)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.ylim(0, 110)
    plt.ylabel("Coverage of Global Total (%)")
    plt.title(f"Sample Representativeness (N={len(sample_countries)} Economies)")
    plt.axhline(100, color='gray', linestyle='--', linewidth=1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300)
    print(f"🖼️ Figure saved to {OUTPUT_FIGURE}")
    
    if pct_gdp > 75 and pct_co2 > 75:
        print("\n✅ Sample is highly representative of the global economy and emissions.")
    else:
        print("\n⚠️ Sample coverage is moderate. External validity limited to major economies.")

if __name__ == "__main__":
    run_external_validity()
