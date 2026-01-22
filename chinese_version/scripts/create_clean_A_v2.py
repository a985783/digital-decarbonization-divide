"""
Clean-A v2: Master Cleaning Script
==================================
Pipeline:
1. Load Raw Data (wdi_real_robust.csv)
2. Filter for 40 Major Economies (Strict Inclusion)
3. Apply Logic-Based Clipping (Clean-A Strategy)
4. Apply Unit Corrections (CO2 metric tons)
5. Save Final Dataset (clean_A_v2.csv)
"""

import pandas as pd
import numpy as np

INPUT_FILE = 'data/wdi_real_robust.csv'
OUTPUT_FILE = 'data/clean_A_v2.csv'

# Logic Ranges (Clean-A Strict)
RANGES = {
    'ICT_exports': (0, 100),
    'Internet_users': (0, 100),
    'Trade_openness': (0, 400), # Allow trade hubs > 100%
    'Inflation': (-50, 200),
    'Mobile_subs': (0, 300), # >100 is common
    'Urban_pop': (0, 100),
    'Renewable_energy': (0, 100),
    'Industry_VA': (0, 100),
    'Manufacturing': (0, 100),
    'Services': (0, 100)
}

def main():
    print("=" * 60)
    print("Generating Clean-A v2 (Full Pipeline)")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run solve_wdi_v3_zip.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded Raw Data: {len(df)} rows, {df['country'].nunique()} countries")
    
    # 1. Logic Clipping
    print("\n[Step 1] Applying Logic Clipping...")
    for col, (min_val, max_val) in RANGES.items():
        if col in df.columns:
            mask = (df[col] >= min_val) & (df[col] <= max_val)
            n_dropped = len(df) - mask.sum()
            # We don't drop rows, we invalid values to NaN (Clean-A strategy) -> wait, actually Clean-A was strict.
            # Let's set to NaN so dropna() catches them later in analysis, or keep sample for describing.
            # Strategy: Set outliers to NaN.
            outliers = ~mask & df[col].notna()
            n_outliers = outliers.sum()
            if n_outliers > 0:
                print(f"  {col}: masked {n_outliers} outliers outside [{min_val}, {max_val}]")
                df.loc[outliers, col] = np.nan
    
    # 2. Unit Correction
    print("\n[Step 2] CO2 Unit Correction...")
    # Detect scaling issue
    if df['CO2_per_capita'].mean() > 100:
        print(f"  Detected scaled CO2 (Mean={df['CO2_per_capita'].mean():.2f}). Dividing by 100.")
        df['CO2_per_capita'] = df['CO2_per_capita'] / 100.0
    else:
        print(f"  CO2 appears correct (Mean={df['CO2_per_capita'].mean():.2f}). No change.")
        
    # 3. Inclusion Criteria (15 years valid data)
    print("\n[Step 3] Applying Inclusion Criteria (15+ years valid Y and T)...")
    valid_mask = df['CO2_per_capita'].notna() & df['ICT_exports'].notna()
    counts = df[valid_mask].groupby('country').size()
    valid_countries = counts[counts >= 15].index.tolist()
    
    print(f"  Retained {len(valid_countries)}/{df['country'].nunique()} countries.")
    print(f"  Dropped: {set(df['country'].unique()) - set(valid_countries)}")
    
    df_final = df[df['country'].isin(valid_countries)].copy()
    
    # Save
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print(f"  Final Stats: {len(df_final)} rows, {df_final['country'].nunique()} countries")
    
    # Audit for Paper
    print("\n[Paper Audit Info]")
    print(f"  N_desc: {len(df_final)}")
    # Complete cases for regression
    cols = ['CO2_per_capita', 'ICT_exports', 'GDP_per_capita', 'Energy_use'] # proxy
    n_reg = len(df_final.dropna(subset=cols)) # Approximate
    print(f"  N_reg_approx: {n_reg}")

import os
if __name__ == "__main__":
    main()
