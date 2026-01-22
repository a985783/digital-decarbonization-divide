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

# --- Variable Definitions ---
INDICATORS = {
    'EN.ATM.CO2E.PC': 'CO2_per_capita',
    'BX.GSR.CCIS.ZS': 'ICT_exports',

    'CC.EST': 'Control_of_Corruption',
    'RL.EST': 'Rule_of_Law',
    'PV.EST': 'Political_Stability',
    'GE.EST': 'Government_Effectiveness',
    'RQ.EST': 'Regulatory_Quality',
    'VA.EST': 'Voice_and_Accountability',

    'SP.POP.DPND': 'Age_dependency_ratio',
    'SE.TER.ENRR': 'Tertiary_enrollment',
    'SL.TLF.CACT.FE.ZS': 'Female_labor_participation',
    'SP.POP.TOTL': 'Population_total',
    'SP.URB.TOTL.IN.ZS': 'Urban_population_pct',
    'SP.POP.GROW': 'Population_growth',
    'SP.DYN.LE00.IN': 'Life_expectancy',
    'SP.DYN.AMRT.MA': 'Mortality_rate_adult_male',
    
    'IS.RRS.TOTL.KM': 'Railways_route_km',
    'EG.ELC.LOSS.ZS': 'Electric_power_losses',
    'IS.SHP.GOOD.TU': 'Container_port_traffic',
    'IS.AIR.GOOD.MT.K1': 'Air_freight_million_ton_km',
    'IT.CEL.SETS.P2': 'Mobile_cellular_subscriptions',
    'IT.NET.USER.ZS': 'Internet_users',
    'IT.MLT.MAIN.P2': 'Fixed_telephone_subscriptions',

    'FS.AST.PRVT.GD.ZS': 'Domestic_credit_to_private_sector',
    'CM.MKT.LCAP.GD.ZS': 'Market_capitalization',
    'FM.LBL.BMNY.GD.ZS': 'Broad_money_pct_GDP',
    'FR.INR.LEND': 'Lending_interest_rate',
    'FP.CPI.TOTL.ZG': 'Inflation_consumer_prices',
    'NY.GDP.DEFL.KD.ZG': 'Inflation_GDP_deflator',

    'NY.GDP.PCAP.CD': 'GDP_per_capita_current',
    'NY.GDP.PCAP.KD': 'GDP_per_capita_constant',
    'NY.GDP.MKTP.KD.ZG': 'GDP_growth',
    'NE.TRD.GNFS.ZS': 'Trade_openness',
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_net_inflows_pct_GDP',
    'BN.CAB.XOKA.GD.ZS': 'Current_account_balance_pct_GDP',
    'NE.GDI.TOTL.ZS': 'Gross_capital_formation_pct_GDP',
    'NY.GNS.ICTR.ZS': 'Gross_savings_pct_GDP',
    'NE.EXP.GNFS.ZS': 'Exports_pct_GDP',
    'NE.IMP.GNFS.ZS': 'Imports_pct_GDP',
    'NV.IND.TOTL.ZS': 'Industry_value_added_pct_GDP',
    'NV.IND.MANF.ZS': 'Manufacturing_value_added_pct_GDP',
    'NV.SRV.TOTL.ZS': 'Services_value_added_pct_GDP',
    'NV.AGR.TOTL.ZS': 'Agriculture_value_added_pct_GDP',
    'GC.DOD.TOTL.GD.ZS': 'Central_govt_debt_pct_GDP',

    'EG.USE.PCAP.KG.OE': 'Energy_use_per_capita',
    'EG.FEC.RNEW.ZS': 'Renewable_energy_consumption_pct',
    'EG.ELC.ACCS.ZS': 'Access_to_electricity_pct',
    'EN.ATM.METH.KT.CE': 'Methane_emissions_kt_CO2_eq',
    'EN.ATM.NOXE.KT.CE': 'Nitrous_oxide_emissions_kt_CO2_eq',
    'AG.LND.FRST.ZS': 'Forest_area_pct',
    'AG.LND.AGRI.ZS': 'Agricultural_land_pct',
    'AG.LND.ARBL.ZS': 'Arable_land_pct',

    'GB.XPD.RSDV.GD.ZS': 'Research_and_development_expenditure_pct_GDP',
    'TX.VAL.TECH.MF.ZS': 'High_tech_exports_pct_mfg_exports',
    'IP.JRN.ARTC.SC': 'Scientific_journal_articles',
    'IP.PAT.RESD': 'Patent_applications_residents',
    'IP.PAT.NRES': 'Patent_applications_nonresidents',
    
    'SE.XPD.TOTL.GD.ZS': 'Govt_expenditure_education_pct_GDP',
    'SH.XPD.CHEX.GD.ZS': 'Current_health_expenditure_pct_GDP',
}

COUNTRIES = [
    'USA', 'DEU', 'JPN', 'GBR', 'FRA', 'CAN', 'AUS', 'KOR', 'ITA', 'ESP',
    'NLD', 'CHE', 'SWE', 'NOR', 'DNK', 'FIN', 'AUT', 'BEL', 'IRL', 'NZL',
    'CHN', 'IND', 'BRA', 'RUS', 'ZAF', 'MEX', 'IDN', 'TUR', 'SAU', 'ARG',
    'THA', 'MYS', 'PHL', 'VNM', 'EGY', 'NGA', 'PAK', 'BGD', 'COL', 'PER'
]

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
