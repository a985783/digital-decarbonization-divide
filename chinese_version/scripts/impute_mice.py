"""
MICE Imputation for High-Dimensional Panel Data (Phase 1)
=========================================================
Performs Multiple Imputation by Chained Equations (MICE) using LightGBM.
Preserves non-linear relationships and interactions.
"""

import pandas as pd
import numpy as np
import miceforest as mf
import os

# Configuration
DATA_DIR = 'data'
INPUT_FILE = os.path.join(DATA_DIR, 'wdi_expanded_raw.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'clean_data_v3_imputed.csv')

def perform_imputation():
    print("Loading raw data...")
    df = pd.read_csv(INPUT_FILE)
    
    df['country'] = df['country'].astype('category')
    
    print("\nMissing values before imputation (Top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    print(f"\nTotal missing cells: {total_missing} ({total_missing/total_cells:.1%})")

    # 2. Setup MICE Kernel
    # We exclude 'country' and 'year' from being imputed (they are keys), but use them as predictors
    # In miceforest, we can specify variable types.
    
    print("\nInitializing MICE kernel...")
    kds = mf.ImputationKernel(
        df,
        random_state=42
    )

    print("Running MICE (3 iterations)...")
    kds.mice(iterations=3, verbose=True)
    
    df_imputed = kds.complete_data()
    
    df_imputed['country'] = df_imputed['country'].astype(str)
    
    print("\nMissing values after imputation:")
    print(df_imputed.isnull().sum().sum())
    
    df_imputed.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved imputed data to {OUTPUT_FILE}")
    print(f"  Shape: {df_imputed.shape}")
    print(f"  Countries: {df_imputed['country'].nunique()}")

if __name__ == "__main__":
    perform_imputation()
