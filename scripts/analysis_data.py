from scripts.dci import build_dci


def _ensure_eds(df):
    if "EDS" not in df.columns and "ICT_exports" in df.columns:
        df["EDS"] = df["ICT_exports"]
    return df


def _correct_co2_units(series):
    if series.mean() > 100:
        return series / 100.0
    return series


def prepare_analysis_data(df, cfg, return_df=False):
    df = df.copy()
    df = _ensure_eds(df)

    if cfg["treatment_main"] == "DCI" and "DCI" not in df.columns:
        dci, _ = build_dci(df, cfg["dci_components"])
        df["DCI"] = dci

    outcome = cfg["outcome"]
    treatment = cfg["treatment_main"]

    df[outcome] = _correct_co2_units(df[outcome])

    y = df[outcome].values
    t = df[treatment].values
    x = df[cfg["moderators_X"]].values
    w = df[cfg["controls_W"]].values

    if return_df:
        return y, t, x, w, df
    return y, t, x, w
