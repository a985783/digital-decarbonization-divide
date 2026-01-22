"""
WDI/WGI Data Fetcher - Expanded Variable Set (Phase 1)
======================================================
Downloads 50-80 variables for High-Dimensional Lasso/DML analysis.
Uses wbgapi for easy retrieval.

Variables cover:
1.  Outcome: CO2 emissions
2.  Treatment: ICT service exports
3.  Controls (High-Dimensional):
    - Institutional Quality (WGI)
    - Demographics
    - Infrastructure
    - Financial Depth
    - Macroeconomic Stability
    - Social Development
    - Innovation & Technology
    - Environment & Energy
"""

import wbgapi as wb
import pandas as pd
import os

# --- Configuration ---
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

SAVE_PATH = os.path.join(DATA_DIR, 'wdi_expanded_raw.csv')

COUNTRIES = [
    'USA', 'DEU', 'JPN', 'GBR', 'FRA', 'CAN', 'AUS', 'KOR', 'ITA', 'ESP',
    'NLD', 'CHE', 'SWE', 'NOR', 'DNK', 'FIN', 'AUT', 'BEL', 'IRL', 'NZL',
    'CHN', 'IND', 'BRA', 'RUS', 'ZAF', 'MEX', 'IDN', 'TUR', 'SAU', 'ARG',
    'THA', 'MYS', 'PHL', 'VNM', 'EGY', 'NGA', 'PAK', 'BGD', 'COL', 'PER'
]

# Time Range
YEARS = range(2000, 2024)

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

def fetch_data():
    print(f"Fetching {len(INDICATORS)} variables for {len(COUNTRIES)} economies...")
    
    indicator_codes = list(INDICATORS.keys())
    chunk_size = 5 
    dfs = []
    
    for i in range(0, len(indicator_codes), chunk_size):
        chunk = indicator_codes[i:i + chunk_size]
        print(f"Fetching chunk {i//chunk_size + 1}/{(len(indicator_codes)-1)//chunk_size + 1}: {chunk}")
        
        try:
            df_chunk = wb.data.DataFrame(
                series=chunk, 
                economy=COUNTRIES, 
                time=YEARS, 
                numericTimeKeys=True,
                labels=False
            )
            
            if not df_chunk.empty:
                df_chunk.reset_index(inplace=True)
                df_chunk.rename(columns={'economy': 'country', 'time': 'year'}, inplace=True)
                dfs.append(df_chunk)
            else:
                print(f"Warning: Empty result for chunk {chunk}")
                
        except Exception as e:
            print(f"Error fetching chunk {chunk}: {e}")

    if not dfs:
        print("Fatal: No data downloaded.")
        return

    print("Merging data chunks...")
    from functools import reduce
    final_df = reduce(lambda left, right: pd.merge(left, right, on=['country', 'year'], how='outer'), dfs)
    
    final_df.rename(columns=INDICATORS, inplace=True)
    
    print(f"Successfully fetched data. Shape: {final_df.shape}")
    print("\nMissing values summary:")
    print(final_df.isnull().sum())
    
    final_df.to_csv(SAVE_PATH, index=False)
    print(f"\nSaved raw expanded data to {SAVE_PATH}")

if __name__ == "__main__":
    fetch_data()
