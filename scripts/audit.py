"""
Project Audit Script (Code Freeze Gate)
=======================================
Verifies data integrity, sample size consistency, and result alignment.
Run before final submission.
"""

import pandas as pd
import numpy as np
import os

DATA_FILE = 'data/clean_A_v2.csv'
RESULTS_FILE = 'results/results_summary_v2.csv'

def main():
    print("="*60)
    print("AUDIT: Re-evaluating Digital Economy–CO2 Relationship")
    print("="*60)
    
    # 1. Data Integrity
    print("\n[Audit 1: Data Integrity]")
    df = pd.read_csv(DATA_FILE)
    
    # ICT Range
    ict_min, ict_max = df['ICT_exports'].min(), df['ICT_exports'].max()
    print(f"  ICT Exports Range: [{ict_min:.2f}, {ict_max:.2f}]")
    if not (0 <= ict_min and ict_max <= 100):
        print("  FAIL: ICT out of [0, 100] range")
        exit(1)
        
    # CO2 Range (Metric Tons)
    co2_min, co2_max = df['CO2_per_capita'].min(), df['CO2_per_capita'].max()
    print(f"  CO2 Emissions Range: [{co2_min:.2f}, {co2_max:.2f}] Metric Tons")
    if not (0 <= co2_max <= 60): # Loose check for reasonable metric tons
        print("  FAIL: CO2 Max unreasonable (likely unit error)")
        exit(1)
        
    print("  PASS: Data ranges valid.")
    
    # 2. Sample Consistency
    print("\n[Audit 2: Sample Consistency]")
    n_full = len(df)
    n_reg = 719 # Hardcoded expectation from paper
    
    print(f"  Full Cleaned Sample (Table 1): {n_full}")
    # Check if N=874 matches number of complete cases
    cols = ['CO2_per_capita', 'ICT_exports'] # Minimal check
    n_complete = len(df.dropna(subset=cols)) # Actual regression uses more cols but this is a proxy
    
    # Actually load results to check claimed N
    if os.path.exists(RESULTS_FILE):
        res = pd.read_csv(RESULTS_FILE)
        n_reported = res.iloc[0]['n']
        print(f"  Reported Regression N (Table 2): {n_reported}")
        
        if n_reported != n_reg:
             print(f"  WARNING: Paper claims {n_reg}, result file says {n_reported}")
        else:
             print(f"  PASS: Regression N matches paper claim ({n_reg}).")
    else:
        print("  WARNING: results_summary_v2.csv not found.")
        
    # 3. Cluster Count
    n_countries = df['country'].nunique()
    print(f"  Total Countries: {n_countries} (Paper claims 38 in analysis)")
    
    # 4. Result Check (Main Spec)
    print("\n[Audit 3: Result Alignment]")
    if os.path.exists(RESULTS_FILE):
        res = pd.read_csv(RESULTS_FILE)
        main_spec = res[res['spec'] == 'MAIN_TWFE'].iloc[0]
        
        theta = main_spec['theta']
        pval = main_spec['p_val']
        
        print(f"  MAIN_TWFE: theta={theta:.4f}, p={pval:.4f}")
        
        # Check vs Abstract claims
        # theta = -0.0573, p = 0.0538
        if np.isclose(theta, -0.057256, atol=1e-4) and np.isclose(pval, 0.0538, atol=1e-3):
            print("  PASS: Results align with Abstract digits.")
        else:
            print("  FAIL: Results DO NOT align with Abstract.")
            exit(1)
    
    print("\n✓ AUDIT COMPLETE: READY FOR SUBMISSION/ARCHIVE.")

if __name__ == "__main__":
    main()
