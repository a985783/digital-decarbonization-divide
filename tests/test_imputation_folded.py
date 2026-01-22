import pandas as pd

from scripts.imputation import impute_folded


def test_impute_does_not_touch_y_t():
    df = pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "Y": [1.0, None, 2.0, None],
            "T": [0.1, None, 0.2, None],
            "W1": [1.0, None, 3.0, None],
        }
    )
    out = impute_folded(
        df,
        y_col="Y",
        t_col="T",
        w_cols=["W1"],
        group_col="country",
    )
    assert out["Y"].isna().sum() == 2
    assert out["T"].isna().sum() == 2
    assert out["W1"].isna().sum() == 0
