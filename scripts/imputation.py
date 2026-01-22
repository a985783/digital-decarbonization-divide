import numpy as np
import miceforest as mf
from sklearn.model_selection import GroupKFold


def impute_folded(
    df,
    y_col,
    t_col,
    w_cols,
    group_col,
    x_cols=None,
    n_splits=5,
    iterations=3,
    random_state=42,
):
    impute_cols = list(w_cols)
    if x_cols:
        impute_cols.extend([col for col in x_cols if col not in impute_cols])

    if y_col in impute_cols or t_col in impute_cols:
        raise ValueError("Y/T columns must not be imputed.")

    out = df.copy()
    groups = df[group_col].values
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, test_idx in gkf.split(df, groups=groups):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        kernel_df = train[impute_cols].copy()
        if kernel_df.isna().sum().sum() == 0:
            continue
        try:
            kds = mf.ImputationKernel(kernel_df, random_state=random_state)
            kds.mice(iterations=iterations, verbose=False)
            test_kernel = test[impute_cols].copy()
            imputed = kds.impute_new_data(
                test_kernel, iterations=iterations, verbose=False
            ).complete_data()
            out.loc[test.index, impute_cols] = imputed[impute_cols].values
        except Exception:
            for col in impute_cols:
                fill_val = train[col].median()
                if np.isnan(fill_val):
                    fill_val = 0.0
                out.loc[test.index, col] = out.loc[test.index, col].fillna(fill_val)

    return out
