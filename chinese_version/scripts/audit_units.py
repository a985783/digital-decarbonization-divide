"""
Audit Data Units (Fact Card Generator)
======================================
Checks the distribution of Y and T to identify scaling errors.
"""
import pandas as pd
import numpy as np

DF_PATH = 'data/clean_data_v3_imputed.csv'

def audit():
    df = pd.read_csv(DF_PATH)
    
    y = 'CO2_per_capita'
    t = 'ICT_exports'
    
    print("=== DATA FACT CARD ===")
    print(f"Dataset Shape: {df.shape}")
    
    print(f"\n[Variable: {y}]")
    desc_y = df[y].describe()
    print(desc_y[['mean', 'min', 'max', 'std']])
    # Check plausible range for Metric Tons (e.g., US is ~14, World ~4-5)
    # If mean is ~400, it's NOT metric tons.
    
    print(f"\n[Variable: {t}]")
    desc_t = df[t].describe()
    print(desc_t[['mean', 'min', 'max', 'std']])
    # Check if 0-1 or 0-100
    
    print("\n[Sample Countries]")
    # Check specific countries to verify
    sample_countries = ['USA', 'CHN', 'QAT'] # Qatar usually high
    for c in sample_countries:
        if c in df['country'].values:
            val = df[df['country'] == c][y].mean()
            print(f"  {c} Mean Y: {val:.2f}")

if __name__ == "__main__":
    audit()
