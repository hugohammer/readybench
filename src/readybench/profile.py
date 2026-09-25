"""Describe a dataset along the classical quality dimensions (Phase 1).

These are descriptive statistics of the pool, not the readiness measures
under study; the measures themselves come in Phase 3.
"""

import hashlib

import numpy as np
import pandas as pd

MAX_ROWS_FOR_CORR = 20_000
HIGH_CARDINALITY = 30


def quality_profile(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> dict:
    n, p = X.shape
    numeric = X.select_dtypes(include="number")
    categorical = X.drop(columns=numeric.columns)
    data = X.assign(__y__=y.to_numpy())

    dup_rows = data.duplicated(keep=False)
    # rows whose features also occur with the other label
    x_dups = X.duplicated(keep=False)
    conflicting = 0
    if x_dups.any():
        groups = data[x_dups].groupby(list(X.columns), dropna=False, observed=True)["__y__"].nunique()
        conflicting = int(groups[groups > 1].count())

    counts = y.value_counts()
    prof = {
        "n_rows": n,
        "n_features": p,
        "frac_categorical": round(categorical.shape[1] / p, 4) if p else 0.0,
        "missing_cell_frac": round(float(X.isna().to_numpy().mean()), 5),
        "missing_row_frac": round(float(X.isna().any(axis=1).mean()), 5),
        "cols_with_missing_frac": round(float(X.isna().any().mean()), 4),
        "duplicate_row_frac": round(float(dup_rows.mean()), 5),
        "conflicting_duplicate_groups": conflicting,
        "minority_count": int(counts.min()),
        "imbalance_ratio": round(float(counts.max() / counts.min()), 3),
        "constant_features": int((X.nunique(dropna=True) <= 1).sum()),
        "high_cardinality_categoricals": int((categorical.nunique() > HIGH_CARDINALITY).sum()),
        "features_per_row": round(p / n, 5),
    }

    if numeric.shape[1] > 0:
        num = numeric.sample(n=min(n, MAX_ROWS_FOR_CORR), random_state=seed) if n > MAX_ROWS_FOR_CORR else numeric
        num = num.loc[:, num.nunique() > 1].astype("float64")
        skew = num.skew().abs()
        q1, q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3 - q1
        outside = (num.lt(q1 - 1.5 * iqr) | num.gt(q3 + 1.5 * iqr)) & num.notna()
        prof.update({
            "mean_abs_skew": round(float(skew.mean()), 3) if len(skew) else 0.0,
            "frac_high_skew": round(float((skew > 1).mean()), 4) if len(skew) else 0.0,
            "outlier_cell_frac": round(float(outside.to_numpy().mean()), 5) if num.size else 0.0,
        })
        if num.shape[1] > 1:
            corr = num.corr().abs().to_numpy()
            upper = corr[np.triu_indices_from(corr, k=1)]
            upper = upper[~np.isnan(upper)]
            prof.update({
                "frac_pairs_corr_gt_0.8": round(float((upper > 0.8).mean()), 5) if len(upper) else 0.0,
                "max_abs_feature_corr": round(float(upper.max()), 4) if len(upper) else 0.0,
            })
    return prof


def column_signatures(X: pd.DataFrame) -> list[str]:
    """Order- and name-independent fingerprint per column, for finding duplicate datasets."""
    sigs = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            v = s.astype("float64")
            key = f"num|{v.nunique()}|{v.mean():.4g}|{v.std():.4g}|{v.isna().sum()}"
        else:
            freq = s.astype("string").value_counts(normalize=True).head(3).round(4).tolist()
            key = f"cat|{s.nunique()}|{freq}|{s.isna().sum()}"
        sigs.append(hashlib.sha1(key.encode()).hexdigest()[:12])
    return sigs
