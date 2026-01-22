import pandas as pd

from scripts.analysis_config import load_config
from scripts.analysis_data import prepare_analysis_data


def test_treatment_is_dci():
    cfg = load_config("analysis_spec.yaml")
    base = {
        "country": ["A"],
        "year": [2000],
        "DCI": [0.0],
        "ICT_exports": [5.0],
        "CO2_per_capita": [1.0],
    }
    for col in cfg["moderators_X"] + cfg["controls_W"]:
        base.setdefault(col, [0.0])
    df = pd.DataFrame(base)
    y, t, x, w = prepare_analysis_data(df, cfg)
    assert t[0] == 0.0
