"""
MICE Imputation for High-Dimensional Panel Data (Phase 1)
=========================================================
Fold-safe MICE: imputes W/X only within training folds.
Y/T remain un-imputed.
"""

import pandas as pd
import os

from scripts.analysis_config import load_config
from scripts.imputation import impute_folded

# Configuration
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "wdi_expanded_raw.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "clean_data_v4_imputed.csv")


def perform_imputation():
    print("Loading raw data...")
    cfg = load_config("analysis_spec.yaml")
    df = pd.read_csv(INPUT_FILE)

    outcome = cfg["outcome"]
    treatment = cfg["treatment_main"]
    controls = cfg["controls_W"]
    moderators = cfg["moderators_X"]
    dci_components = cfg["dci_components"]

    y_missing_before = df[outcome].isna().sum() if outcome in df.columns else None
    t_missing_before = df[treatment].isna().sum() if treatment in df.columns else None

    print("\nMissing values before imputation (Top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    print(f"\nTotal missing cells: {total_missing} ({total_missing/total_cells:.1%})")

    impute_cols = list(set(controls + moderators + dci_components))
    impute_cols = [col for col in impute_cols if col in df.columns]

    print(f"\nImputing {len(impute_cols)} columns with fold-safe MICE...")
    df_imputed = impute_folded(
        df,
        y_col=outcome,
        t_col=treatment,
        w_cols=impute_cols,
        x_cols=None,
        group_col=cfg["groups"],
        n_splits=cfg["cv"]["n_splits"],
        iterations=cfg["imputation"]["iterations"],
        random_state=cfg["imputation"]["random_state"],
    )

    if outcome in df_imputed.columns and y_missing_before is not None:
        assert df_imputed[outcome].isna().sum() == y_missing_before
    if treatment in df_imputed.columns and t_missing_before is not None:
        assert df_imputed[treatment].isna().sum() == t_missing_before

    print("\nMissing values after imputation:")
    print(df_imputed.isnull().sum().sum())

    df_imputed.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved imputed data to {OUTPUT_FILE}")
    print(f"  Shape: {df_imputed.shape}")
    print(f"  Countries: {df_imputed['country'].nunique()}")


if __name__ == "__main__":
    perform_imputation()
