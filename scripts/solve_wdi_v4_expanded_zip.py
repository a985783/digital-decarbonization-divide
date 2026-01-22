"""
Direct WDI Bulk Downloader (Expanded - ZIP/CSV)
===============================================
Downloads full CSV datasets (ZIP) from World Bank to bypass JSON API failures.
Updated for expanded high-dimensional variable set.
"""

import pandas as pd
import io
import zipfile
import urllib.request
import os
import shutil

from scripts.analysis_config import load_config
from scripts.wdi_indicators import load_indicators

CFG = load_config("analysis_spec.yaml")

# --- Variable Definitions ---
INDICATORS = load_indicators(CFG)
COUNTRIES = CFG["countries"]

DATA_DIR = 'data'
TEMP_DIR = os.path.join(DATA_DIR, 'temp_downloads')

def download_and_process(indicator_code, indicator_name):
    """Download ZIP, extract CSV, parse, and filter."""
    print(f"  Fetching {indicator_code} ({indicator_name})...", end=" ", flush=True)
    
    url = f"http://api.worldbank.org/v2/en/indicator/{indicator_code}?downloadformat=csv"
    
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            zip_content = response.read()
            
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            csv_filename = [f for f in z.namelist() if f.startswith('API_') and f.endswith('.csv')]
            if not csv_filename:
                print("✗ No data CSV found")
                return None
            
            csv_filename = csv_filename[0]
            
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, skiprows=4)
                
        if 'Country Code' not in df.columns:
            print("✗ CSV missing 'Country Code'")
            return None
            
        df_filtered = df[df['Country Code'].isin(COUNTRIES)].copy()
        
        year_cols = [c for c in df.columns if c.isdigit()]
        year_cols = [c for c in year_cols if 2000 <= int(c) <= 2023]
        
        if not year_cols:
            print("✗ No relevant years found")
            return None
            
        melted = df_filtered.melt(
            id_vars=['Country Code'], 
            value_vars=year_cols,
            var_name='year',
            value_name=indicator_name
        )
        
        melted = melted.rename(columns={'Country Code': 'country'})
        melted['year'] = melted['year'].astype(int)
        
        print(f"✓ {len(melted)} records")
        return melted
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    print("=" * 60)
    print("Direct WDI Bulk Downloader (Expanded - ZIP/CSV)")
    print("=" * 60)
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    all_dfs = []
    
    for code, name in INDICATORS.items():
        df = download_and_process(code, name)
        if df is not None:
            all_dfs.append(df)
            
    if not all_dfs:
        print("Fatal: No data downloaded.")
        return
        
    print("\nMerging datasets...")
    from functools import reduce
    final_df = reduce(lambda left, right: pd.merge(left, right, on=['country', 'year'], how='outer'), all_dfs)
    
    oecd_set = set(COUNTRIES[:20])
    final_df['OECD'] = final_df['country'].apply(lambda x: 1 if x in oecd_set else 0)
    
    final_df = final_df.sort_values(['country', 'year'])
    
    if 'CO2_per_capita' in final_df.columns:
        print(f"CO2 Range: [{final_df['CO2_per_capita'].min()}, {final_df['CO2_per_capita'].max()}]")
        
    save_path = os.path.join(DATA_DIR, 'wdi_expanded_raw.csv')
    final_df.to_csv(save_path, index=False)
    print(f"\n✓ Saved to {save_path}")
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
