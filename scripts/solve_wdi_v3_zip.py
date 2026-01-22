"""
Direct WDI Bulk Downloader (V3 - ZIP/CSV)
==========================================
Downloads full CSV datasets (ZIP) from World Bank to bypass JSON API failures.
This is the most robust method for bulk data retrieval.
"""

import pandas as pd
import io
import zipfile
import urllib.request
import os
import shutil

# Indicators
INDICATORS = {
    'EN.ATM.CO2E.PC': 'CO2_per_capita',
    'BX.GSR.CCIS.ZS': 'ICT_exports',
    'IT.NET.USER.ZS': 'Internet_users',
    'IT.CEL.SETS.P2': 'Mobile_subs',
    'NY.GDP.PCAP.CD': 'GDP_per_capita',
    'NY.GDP.MKTP.KD.ZG': 'GDP_growth',
    'NE.TRD.GNFS.ZS': 'Trade_openness',
    'NE.GDI.TOTL.ZS': 'Gross_investment',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'EG.USE.PCAP.KG.OE': 'Energy_use',
    'EG.FEC.RNEW.ZS': 'Renewable_energy',
    'EG.ELC.ACCS.ZS': 'Electricity_access',
    'NV.IND.TOTL.ZS': 'Industry_VA',
    'NV.IND.MANF.ZS': 'Manufacturing',
    'NV.SRV.TOTL.ZS': 'Services',
    'NV.AGR.TOTL.ZS': 'Agriculture',
    'SP.POP.TOTL': 'Population',
    'SP.URB.TOTL.IN.ZS': 'Urban_pop',
    'SE.TER.ENRR': 'Tertiary_enroll',
    'AG.LND.FRST.ZS': 'Forest_area',
}

COUNTRIES = [
    'USA', 'DEU', 'JPN', 'GBR', 'FRA', 'CAN', 'AUS', 'KOR', 'ITA', 'ESP',
    'NLD', 'CHE', 'SWE', 'NOR', 'DNK', 'FIN', 'AUT', 'BEL', 'IRL', 'NZL',
    'CHN', 'IND', 'BRA', 'RUS', 'ZAF', 'MEX', 'IDN', 'TUR', 'SAU', 'ARG',
    'THA', 'MYS', 'PHL', 'VNM', 'EGY', 'NGA', 'PAK', 'BGD', 'COL', 'PER'
]

DATA_DIR = '/Users/cuiqingsong/Documents/论文/data'
TEMP_DIR = os.path.join(DATA_DIR, 'temp_downloads')

def download_and_process(indicator_code, indicator_name):
    """Download ZIP, extract CSV, parse, and filter."""
    print(f"  Fetching {indicator_code} ({indicator_name})...", end=" ", flush=True)
    
    url = f"http://api.worldbank.org/v2/en/indicator/{indicator_code}?downloadformat=csv"
    
    try:
        # Download
        with urllib.request.urlopen(url, timeout=60) as response:
            zip_content = response.read()
            
        # Unzip in memory
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            # Find the main data file (usually starts with API_...)
            csv_filename = [f for f in z.namelist() if f.startswith('API_') and f.endswith('.csv')]
            if not csv_filename:
                print("✗ No data CSV found in zip")
                return None
            
            csv_filename = csv_filename[0]
            
            # Read CSV - skip first 4 rows (metadata)
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, skiprows=4)
                
        # Filter for our countries
        # Column 'Country Code' is strictly ISO3
        if 'Country Code' not in df.columns:
            print("✗ CSV missing 'Country Code'")
            return None
            
        df_filtered = df[df['Country Code'].isin(COUNTRIES)].copy()
        
        # Melt (Year columns are '1960', '1961', ...)
        # Find year columns (numbers)
        year_cols = [c for c in df.columns if c.isdigit()]
        year_cols = [c for c in year_cols if 2000 <= int(c) <= 2022]
        
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
    print("Direct WDI Bulk Downloader (V3 - ZIP/CSV)")
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
    
    # OECD flag
    oecd_set = set(COUNTRIES[:20])
    final_df['OECD'] = final_df['country'].apply(lambda x: 1 if x in oecd_set else 0)
    
    # Clean & Interpolate -> NO INTERPOLATION (Strict Mode)
    print("Formatting...")
    # numeric_cols = final_df.select_dtypes(include=[float, int]).columns
    # numeric_cols = [c for c in numeric_cols if c not in ['year', 'OECD']]
    
    final_df = final_df.sort_values(['country', 'year'])
    # Interpolation removed to enforce strict listwise deletion downstream
    # for col in numeric_cols:
    #     final_df[col] = final_df.groupby('country')[col].transform(
    #         lambda x: x.interpolate(method='linear', limit_direction='both')
    #     )
        
    # Validation
    if 'CO2_per_capita' in final_df.columns:
        print(f"CO2 Range: [{final_df['CO2_per_capita'].min()}, {final_df['CO2_per_capita'].max()}]")
        
    # Save
    save_path = '/Users/cuiqingsong/Documents/论文/data/wdi_real_robust.csv'
    final_df.to_csv(save_path, index=False)
    print(f"\n✓ Saved to {save_path}")
    
    # Cleanup
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
