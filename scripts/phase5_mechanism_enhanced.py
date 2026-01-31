"""
Phase 5: Enhanced Mechanism Analysis
=====================================
Deepens the mechanism analysis with mediation effects and triple interactions.
Tests: 
1. Mediation: DCI → Energy Efficiency → CO2
2. Triple interaction: DCI × Institution × Renewable
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import statsmodels.api as sm

warnings.filterwarnings('ignore')

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data

DATA_DIR = "data"
RESULTS_DIR = "results"
INPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")
MECHANISM_RESULTS_FILE = os.path.join(RESULTS_DIR, "mechanism_enhanced_results.csv")

def mediation_analysis(df, treatment, mediator, outcome, controls, groups):
    """
    Baron & Kenny (1986) mediation analysis steps:
    1. Y ~ T + W (total effect)
    2. M ~ T + W (mediator effect)
    3. Y ~ T + M + W (direct effect)
    4. Indirect effect = Total - Direct
    """
    results = {}
    
    # Step 1: Total effect
    X1 = df[[treatment] + controls].copy()
    X1 = sm.add_constant(X1)
    model1 = sm.OLS(df[outcome], X1).fit(cov_type='HC3')
    results['total_effect'] = model1.params[treatment]
    results['total_effect_se'] = model1.bse[treatment]
    results['total_effect_p'] = model1.pvalues[treatment]
    
    # Step 2: Mediator effect
    X2 = df[[treatment] + controls].copy()
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(df[mediator], X2).fit(cov_type='HC3')
    results['mediator_effect'] = model2.params[treatment]
    results['mediator_effect_p'] = model2.pvalues[treatment]
    
    # Step 3: Direct effect (controlling for mediator)
    X3 = df[[treatment, mediator] + controls].copy()
    X3 = sm.add_constant(X3)
    model3 = sm.OLS(df[outcome], X3).fit(cov_type='HC3')
    results['direct_effect'] = model3.params[treatment]
    results['direct_effect_se'] = model3.bse[treatment]
    results['direct_effect_p'] = model3.pvalues[treatment]
    
    # Step 4: Indirect effect (mediation)
    results['indirect_effect'] = results['total_effect'] - results['direct_effect']
    
    # Sobel test for indirect effect significance
    a = results['mediator_effect']
    b = model3.params[mediator]
    sa = model2.bse[treatment]
    sb = model3.bse[mediator]
    sobel_z = (a * b) / np.sqrt((a**2 * sb**2) + (b**2 * sa**2) + (sa**2 * sb**2))
    from scipy.stats import norm
    results['sobel_p'] = 2 * (1 - norm.cdf(abs(sobel_z)))
    
    # Proportion mediated
    if results['total_effect'] != 0:
        results['proportion_mediated'] = results['indirect_effect'] / results['total_effect']
    else:
        results['proportion_mediated'] = np.nan
    
    return results

def triple_interaction_test(df, treatment, moderator1, moderator2, outcome, controls):
    """
    Test triple interaction: DCI × Institution × Renewable
    """
    # Center variables
    df_centered = df.copy()
    df_centered[treatment] = df_centered[treatment] - df_centered[treatment].mean()
    df_centered[moderator1] = df_centered[moderator1] - df_centered[moderator1].mean()
    df_centered[moderator2] = df_centered[moderator2] - df_centered[moderator2].mean()
    
    # Create interaction terms
    df_centered['two_way_1'] = df_centered[treatment] * df_centered[moderator1]
    df_centered['two_way_2'] = df_centered[treatment] * df_centered[moderator2]
    df_centered['two_way_3'] = df_centered[moderator1] * df_centered[moderator2]
    df_centered['three_way'] = (df_centered[treatment] * df_centered[moderator1] * 
                                df_centered[moderator2])
    
    # Regression with triple interaction
    X = df_centered[[treatment, moderator1, moderator2, 
                     'two_way_1', 'two_way_2', 'two_way_3', 'three_way'] + controls]
    X = sm.add_constant(X)
    model = sm.OLS(df_centered[outcome], X).fit(cov_type='HC3')
    
    results = {
        'triple_interaction_coef': model.params['three_way'],
        'triple_interaction_se': model.bse['three_way'],
        'triple_interaction_p': model.pvalues['three_way'],
        'two_way_treatment_institution_p': model.pvalues['two_way_1'],
        'two_way_treatment_renewable_p': model.pvalues['two_way_2'],
        'two_way_institution_renewable_p': model.pvalues['two_way_3']
    }
    
    return results

def run_enhanced_mechanism_analysis():
    print("=" * 70)
    print("Phase 5: Enhanced Mechanism Analysis")
    print("=" * 70)
    
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)
    
    # Prepare data using the same function as other phases
    df_clean = df.dropna(subset=[cfg["outcome"]]).copy()
    y, t, x, w, df_clean = prepare_analysis_data(df_clean, cfg, return_df=True)
    
    # Define mediators
    mediators = {
        'Energy_Efficiency': 'energy_efficiency',
        'Structural_Change': 'Services_value_added_pct_GDP',
        'Innovation': 'Research_and_development_expenditure_pct_GDP'
    }
    
    # Create constructed mediators if needed
    if 'energy_efficiency' not in df_clean.columns:
        df_clean['energy_efficiency'] = (df_clean['GDP_per_capita_constant'] / 
                                         df_clean['Energy_use_per_capita'])
    
    controls = cfg["controls_W"][:10]  # Use subset for parsimony
    
    print("\n🔬 Test 1: Multidimensional Mediation Analysis")
    print("=" * 70)
    
    mediation_summary = []
    
    for name, var in mediators.items():
        if var not in df_clean.columns:
            print(f"   ⚠️  Skipping {name} (Variable {var} not found)")
            continue
            
        print(f"\n   Testing Mediator: {name} ({var})...")
        
        # Drop missing for specific mediator
        df_med = df_clean.dropna(subset=[var]).copy()
        
        try:
            res = mediation_analysis(
                df_med,
                treatment=cfg["treatment_main"],
                mediator=var,
                outcome=cfg["outcome"],
                controls=controls,
                groups=cfg["groups"]
            )
            
            print(f"      Total Effect: {res['total_effect']:.4f}")
            print(f"      Indirect Effect: {res['indirect_effect']:.4f}")
            print(f"      Proportion Mediated: {res['proportion_mediated']:.1%}")
            print(f"      Sobel p-value: {res['sobel_p']:.4f}")
            
            res['Mediator'] = name
            mediation_summary.append(res)
            
            if res['sobel_p'] < 0.05:
                print("      ✅ Significant mediation")
            else:
                print("      ⚠️  Insignificant mediation")
                
        except Exception as e:
            print(f"      ❌ Estimation failed: {str(e)}")
            
    # Save combined mediation results
    if mediation_summary:
        pd.DataFrame(mediation_summary).to_csv(
            os.path.join(RESULTS_DIR, "mediation_summary.csv"), index=False
        )
    
    print("\n🔬 Test 2: Triple Interaction (DCI × Institution × Renewable)")
    print("=" * 70)
    
    triple_results = triple_interaction_test(
        df_clean,
        treatment=cfg["treatment_main"],
        moderator1='Control_of_Corruption',
        moderator2='Renewable_energy_consumption_pct',
        outcome=cfg["outcome"],
        controls=controls
    )
    
    print(f"   Triple Interaction Coeff: {triple_results['triple_interaction_coef']:.6f}")
    print(f"   Triple Interaction p-value: {triple_results['triple_interaction_p']:.4f}")
    
    if triple_results['triple_interaction_p'] < 0.05:
        print("   ✅ Significant triple interaction detected")
        print("   Interpretation: Renewable energy moderates the institutional moderation")
    else:
        print("   ⚠️ No significant triple interaction")
    
    # Save results
    # Save Triple Interaction results separately or effectively
    # We will grab the first mediation result just for summary structure consistency if needed, 
    # but primarily rely on mediation_summary.csv for mediation details.
    
    # Try to find Energy Efficiency mediation for the summary
    ee_res = next((res for res in mediation_summary if res.get('Mediator') == 'Energy_Efficiency'), {})
    
    results_summary = {
        'mediation_total_effect': ee_res.get('total_effect', np.nan),
        'mediation_indirect_effect': ee_res.get('indirect_effect', np.nan),
        'mediation_sobel_p': ee_res.get('sobel_p', np.nan),
        'triple_interaction_coef': triple_results['triple_interaction_coef'],
        'triple_interaction_p': triple_results['triple_interaction_p'],
        'triple_two_way_ti_p': triple_results['two_way_treatment_institution_p'],
        'triple_two_way_tr_p': triple_results['two_way_treatment_renewable_p'],
        'sample_size': len(df_clean)
    }
    
    pd.DataFrame([results_summary]).to_csv(MECHANISM_RESULTS_FILE, index=False)
    print(f"\n💾 Enhanced mechanism results saved to {MECHANISM_RESULTS_FILE}")
    
    print("\n" + "=" * 70)
    print("THEORETICAL IMPLICATIONS")
    print("=" * 70)
    print("""
    If mediation is significant: DCI improves energy efficiency, which reduces CO2.
    This supports the 'enabling technology' mechanism.
    
    If triple interaction is significant: The institutional moderation is itself 
    moderated by renewable energy share. This suggests policy complementarities.
    """)

if __name__ == "__main__":
    run_enhanced_mechanism_analysis()
