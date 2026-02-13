"""
Feature Engineering Script for Digital Carbon Divide Analysis

This script implements advanced feature engineering:
1. Interaction terms between DCI and key moderators
2. Feature transformations (DCI squared, log GDP, institution categories)
3. Data expansion analysis
4. Comparison with baseline Causal Forest results

Author: Data Science Expert
Date: 2026-02-13
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from pathlib import Path
import json

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dci import build_dci
from analysis_data import prepare_analysis_data
from analysis_config import load_config

warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "data/clean_data_v4_imputed.csv"
OUTPUT_FILE = "data/clean_data_v5_enhanced.csv"
COMPARISON_FILE = "results/feature_comparison.csv"
WDI_RAW_FILE = "data/wdi_expanded_raw.csv"


def load_data(filepath):
    """Load and validate input data."""
    df = pd.read_csv(filepath)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    return df


def ensure_dci(df, cfg):
    """Ensure DCI is computed if not present."""
    if "DCI" not in df.columns:
        print("Computing DCI from components...")
        dci, evr = build_dci(df, cfg["dci_components"])
        df["DCI"] = dci
        print(f"DCI computed. Explained variance ratio: {evr:.4f}")
    else:
        print("DCI already present in dataset")
    return df


def add_interaction_terms(df):
    """
    Add interaction terms:
    - DCI × Trade_Openness
    - DCI × Financial_Development
    - DCI × Education_Level
    - log(GDP) × DCI_squared
    """
    print("\n=== Adding Interaction Terms ===")

    # DCI × Trade_Openness
    if "DCI" in df.columns and "Trade_openness" in df.columns:
        df["DCI_x_Trade_Openness"] = df["DCI"] * df["Trade_openness"]
        print("Added: DCI_x_Trade_Openness")

    # DCI × Financial_Development (using Domestic_credit_to_private_sector as proxy)
    if "DCI" in df.columns and "Domestic_credit_to_private_sector" in df.columns:
        df["DCI_x_Financial_Development"] = df["DCI"] * df["Domestic_credit_to_private_sector"]
        print("Added: DCI_x_Financial_Development")

    # DCI × Education_Level (using Tertiary_enrollment as proxy)
    if "DCI" in df.columns and "Tertiary_enrollment" in df.columns:
        df["DCI_x_Education_Level"] = df["DCI"] * df["Tertiary_enrollment"]
        print("Added: DCI_x_Education_Level")

    # log(GDP) × DCI_squared
    if "DCI" in df.columns and "GDP_per_capita_constant" in df.columns:
        df["log_GDP"] = np.log(df["GDP_per_capita_constant"].clip(lower=1))
        df["DCI_squared"] = df["DCI"] ** 2
        df["log_GDP_x_DCI_squared"] = df["log_GDP"] * df["DCI_squared"]
        print("Added: log_GDP_x_DCI_squared")

    return df


def add_feature_transformations(df):
    """
    Add feature transformations:
    - DCI_squared (diminishing returns)
    - log(GDP_per_capita)
    - Institution_quality_category (high/medium/low)
    """
    print("\n=== Adding Feature Transformations ===")

    # DCI_squared (already added in interactions, but ensure it exists)
    if "DCI" in df.columns and "DCI_squared" not in df.columns:
        df["DCI_squared"] = df["DCI"] ** 2
        print("Added: DCI_squared")

    # log(GDP_per_capita) - natural log
    if "GDP_per_capita_constant" in df.columns:
        df["log_GDP_per_capita"] = np.log(df["GDP_per_capita_constant"].clip(lower=1))
        print("Added: log_GDP_per_capita")

    # Institution_quality_category
    # Using average of governance indicators
    governance_cols = [
        "Control_of_Corruption",
        "Rule_of_Law",
        "Political_Stability",
        "Government_Effectiveness",
        "Regulatory_Quality",
        "Voice_and_Accountability"
    ]

    available_gov_cols = [col for col in governance_cols if col in df.columns]
    if available_gov_cols:
        df["Institution_quality_index"] = df[available_gov_cols].mean(axis=1)

        # Create categories based on quantiles
        quantiles = df["Institution_quality_index"].quantile([0.33, 0.67])

        def categorize_institution(val):
            if pd.isna(val):
                return "unknown"
            elif val <= quantiles[0.33]:
                return "low"
            elif val <= quantiles[0.67]:
                return "medium"
            else:
                return "high"

        df["Institution_quality_category"] = df["Institution_quality_index"].apply(categorize_institution)

        # Create dummy variables
        df["Institution_quality_high"] = (df["Institution_quality_category"] == "high").astype(int)
        df["Institution_quality_medium"] = (df["Institution_quality_category"] == "medium").astype(int)
        df["Institution_quality_low"] = (df["Institution_quality_category"] == "low").astype(int)

        print(f"Added: Institution_quality_category (quantiles: {quantiles[0.33]:.3f}, {quantiles[0.67]:.3f})")

    return df


def analyze_data_expansion(df):
    """
    Analyze potential for data expansion:
    - Check if more countries can be added
    - Check if time range can extend to 1995
    """
    print("\n=== Data Expansion Analysis ===")

    expansion_report = {}

    # Current state
    current_countries = set(df['country'].unique())
    current_years = set(df['year'].unique())

    expansion_report['current'] = {
        'countries': len(current_countries),
        'country_list': sorted(list(current_countries)),
        'year_range': f"{min(current_years)}-{max(current_years)}",
        'total_observations': len(df)
    }

    print(f"Current dataset: {len(current_countries)} countries, years {min(current_years)}-{max(current_years)}")

    # Check WDI raw data for expansion potential
    if os.path.exists(WDI_RAW_FILE):
        wdi_df = pd.read_csv(WDI_RAW_FILE)

        wdi_countries = set(wdi_df['country'].unique())
        wdi_years = set(wdi_df['year'].unique())

        expansion_report['wdi_raw'] = {
            'countries': len(wdi_countries),
            'year_range': f"{min(wdi_years)}-{max(wdi_years)}"
        }

        # Countries available in WDI but not in current dataset
        additional_countries = wdi_countries - current_countries

        print(f"\nWDI raw data: {len(wdi_countries)} countries, years {min(wdi_years)}-{max(wdi_years)}")
        print(f"Additional countries in WDI: {len(additional_countries)}")

        if additional_countries:
            print(f"Potential additional countries: {sorted(list(additional_countries))[:10]}...")

        expansion_report['potential_additions'] = {
            'additional_countries': len(additional_countries),
            'country_list': sorted(list(additional_countries)) if additional_countries else []
        }

        # Check for earlier years (1995-1999)
        early_years = [y for y in wdi_years if y < min(current_years)]
        if early_years:
            print(f"\nEarlier years available: {sorted(early_years)}")
            expansion_report['potential_additions']['earlier_years'] = sorted(early_years)
        else:
            print("\nNo earlier years available in WDI data")
            expansion_report['potential_additions']['earlier_years'] = []

        # Check data availability for key variables in earlier years
        if early_years:
            print("\nChecking data availability for key variables in earlier years...")
            key_vars = ['Internet_users', 'Fixed_broadband_subscriptions', 'CO2_per_capita']
            available_vars = [v for v in key_vars if v in wdi_df.columns]

            early_data = wdi_df[wdi_df['year'].isin(early_years)]
            for var in available_vars:
                non_null = early_data[var].notna().sum()
                print(f"  {var}: {non_null} non-null observations in early years")

    else:
        print(f"WDI raw file not found: {WDI_RAW_FILE}")
        expansion_report['wdi_raw'] = {'error': 'File not found'}

    return expansion_report


def run_causal_forest_comparison(df_original, df_enhanced, cfg):
    """
    Run Causal Forest with baseline and enhanced features, compare results.
    """
    print("\n=== Running Causal Forest Comparison ===")

    try:
        from econml.dml import CausalForestDML
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("econml not installed. Skipping Causal Forest comparison.")
        return None

    comparison_results = {}

    # Prepare baseline data
    print("\nPreparing baseline data...")
    y_base, t_base, x_base, w_base, df_base = prepare_analysis_data(df_original, cfg, return_df=True)

    # Prepare enhanced data
    print("Preparing enhanced data...")

    # Add enhanced features to the dataframe for analysis
    enhanced_features = [
        "DCI_x_Trade_Openness",
        "DCI_x_Financial_Development",
        "DCI_x_Education_Level",
        "log_GDP_x_DCI_squared",
        "DCI_squared",
        "log_GDP_per_capita",
        "Institution_quality_high",
        "Institution_quality_medium",
        "Institution_quality_low"
    ]

    # Get available enhanced features
    available_enhanced = [f for f in enhanced_features if f in df_enhanced.columns]
    print(f"Enhanced features available: {available_enhanced}")

    # Prepare enhanced moderator matrix
    y_enh, t_enh, x_base_enh, w_enh, df_enh_full = prepare_analysis_data(df_enhanced, cfg, return_df=True)

    # Add enhanced features to moderator matrix
    x_enhanced = np.column_stack([x_base_enh] + [df_enhanced[f].values for f in available_enhanced if f in df_enhanced.columns])

    print(f"Baseline moderators shape: {x_base.shape}")
    print(f"Enhanced moderators shape: {x_enhanced.shape}")

    # Check for NaN values and create valid indices
    base_valid = ~(np.isnan(y_base) | np.isnan(t_base) | np.isnan(x_base).any(axis=1) | np.isnan(w_base).any(axis=1))
    enh_valid = ~(np.isnan(y_enh) | np.isnan(t_enh) | np.isnan(x_enhanced).any(axis=1) | np.isnan(w_enh).any(axis=1))

    print(f"Valid observations - Baseline: {base_valid.sum()}, Enhanced: {enh_valid.sum()}")

    if base_valid.sum() < 100 or enh_valid.sum() < 100:
        print("Insufficient valid observations for Causal Forest. Skipping comparison.")
        return {
            'baseline': {'num_features': x_base.shape[1], 'valid_obs': int(base_valid.sum())},
            'enhanced': {'num_features': x_enhanced.shape[1], 'valid_obs': int(enh_valid.sum())},
            'enhanced_features_added': available_enhanced,
            'note': 'Insufficient valid observations for model training'
        }

    # Filter data to valid observations
    y_base_f = y_base[base_valid]
    t_base_f = t_base[base_valid]
    x_base_f = x_base[base_valid]
    w_base_f = w_base[base_valid]

    y_enh_f = y_enh[enh_valid]
    t_enh_f = t_enh[enh_valid]
    x_enh_f = x_enhanced[enh_valid]
    w_enh_f = w_enh[enh_valid]

    # Run Causal Forest - Baseline
    print("\nTraining baseline Causal Forest...")
    cf_base = CausalForestDML(
        n_estimators=1000,
        min_samples_leaf=10,
        max_depth=None,
        random_state=42,
        discrete_treatment=False
    )

    # Fit baseline model - CausalForestDML uses Y, T, X, W parameters
    cf_base.fit(Y=y_base_f, T=t_base_f, X=x_base_f, W=w_base_f)
    cate_base = cf_base.effect(x_base_f)

    baseline_metrics = {
        'cate_mean': float(np.mean(cate_base)),
        'cate_std': float(np.std(cate_base)),
        'cate_min': float(np.min(cate_base)),
        'cate_max': float(np.max(cate_base)),
        'num_features': x_base.shape[1],
        'valid_obs': int(base_valid.sum())
    }

    print(f"Baseline CATE mean: {baseline_metrics['cate_mean']:.4f}")
    print(f"Baseline CATE std: {baseline_metrics['cate_std']:.4f}")

    # Run Causal Forest - Enhanced
    print("\nTraining enhanced Causal Forest...")
    cf_enh = CausalForestDML(
        n_estimators=1000,
        min_samples_leaf=10,
        max_depth=None,
        random_state=42,
        discrete_treatment=False
    )

    # Fit enhanced model
    cf_enh.fit(Y=y_enh_f, T=t_enh_f, X=x_enh_f, W=w_enh_f)
    cate_enh = cf_enh.effect(x_enh_f)

    enhanced_metrics = {
        'cate_mean': float(np.mean(cate_enh)),
        'cate_std': float(np.std(cate_enh)),
        'cate_min': float(np.min(cate_enh)),
        'cate_max': float(np.max(cate_enh)),
        'num_features': x_enhanced.shape[1],
        'valid_obs': int(enh_valid.sum())
    }

    print(f"Enhanced CATE mean: {enhanced_metrics['cate_mean']:.4f}")
    print(f"Enhanced CATE std: {enhanced_metrics['cate_std']:.4f}")

    # Compare results
    comparison_results = {
        'baseline': baseline_metrics,
        'enhanced': enhanced_metrics,
        'difference': {
            'cate_mean_diff': enhanced_metrics['cate_mean'] - baseline_metrics['cate_mean'],
            'cate_std_diff': enhanced_metrics['cate_std'] - baseline_metrics['cate_std'],
            'feature_increase': enhanced_metrics['num_features'] - baseline_metrics['num_features']
        },
        'enhanced_features_added': available_enhanced
    }

    # Feature importance for enhanced model
    try:
        feature_importance = cf_enh.feature_importances_

        # Get feature names
        base_features = cfg['moderators_X']
        all_features = base_features + available_enhanced

        importance_df = pd.DataFrame({
            'feature': all_features[:len(feature_importance)],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature Importances (Enhanced Model):")
        print(importance_df.head(10).to_string(index=False))

        comparison_results['feature_importance'] = importance_df.head(15).to_dict('records')
    except Exception as e:
        print(f"Could not compute feature importance: {e}")

    return comparison_results


def save_comparison_report(comparison_results, expansion_report):
    """Save comparison results to CSV."""
    os.makedirs(os.path.dirname(COMPARISON_FILE), exist_ok=True)

    # Create comparison dataframe
    rows = []

    if comparison_results:
        # Baseline metrics
        for key, value in comparison_results['baseline'].items():
            rows.append({
                'metric': f'baseline_{key}',
                'value': value
            })

        # Enhanced metrics
        for key, value in comparison_results['enhanced'].items():
            rows.append({
                'metric': f'enhanced_{key}',
                'value': value
            })

        # Differences
        for key, value in comparison_results['difference'].items():
            rows.append({
                'metric': f'diff_{key}',
                'value': value
            })

        # Enhanced features
        rows.append({
            'metric': 'enhanced_features_list',
            'value': ', '.join(comparison_results['enhanced_features_added'])
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(COMPARISON_FILE, index=False)
    print(f"\nComparison report saved to: {COMPARISON_FILE}")

    # Save expansion report as JSON
    expansion_file = COMPARISON_FILE.replace('.csv', '_expansion.json')
    with open(expansion_file, 'w') as f:
        json.dump(expansion_report, f, indent=2)
    print(f"Expansion report saved to: {expansion_file}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Feature Engineering for Digital Carbon Divide Analysis")
    print("=" * 60)

    # Load configuration
    cfg = load_config("analysis_spec.yaml")

    # Load data
    print(f"\nLoading data from: {INPUT_FILE}")
    df = load_data(INPUT_FILE)

    # Keep original for comparison
    df_original = df.copy()

    # Ensure DCI is computed
    df = ensure_dci(df, cfg)

    # Add interaction terms
    df = add_interaction_terms(df)

    # Add feature transformations
    df = add_feature_transformations(df)

    # Analyze data expansion possibilities
    expansion_report = analyze_data_expansion(df)

    # Save enhanced dataset
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nEnhanced dataset saved to: {OUTPUT_FILE}")
    print(f"Final dataset shape: {df.shape}")

    # List new columns added
    new_cols = set(df.columns) - set(df_original.columns)
    print(f"\nNew columns added ({len(new_cols)}):")
    for col in sorted(new_cols):
        print(f"  - {col}")

    # Run Causal Forest comparison
    comparison_results = run_causal_forest_comparison(df_original, df, cfg)

    # Save comparison report
    save_comparison_report(comparison_results, expansion_report)

    print("\n" + "=" * 60)
    print("Feature Engineering Complete!")
    print("=" * 60)

    return df, comparison_results, expansion_report


if __name__ == "__main__":
    main()
