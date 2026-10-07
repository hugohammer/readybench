"""Controlled degradations for Study B (analysis plan section 11, B2).

Every function takes the training data (X, y with minority class = 1) and a
numpy Generator and returns degraded copies. Level 0 is the clean data.
All randomness comes from the Generator, seeded by (replication, type, level),
so degraded data can be regenerated exactly (e.g. for the local TabPFN run).
"""

import numpy as np
import pandas as pd

LEVELS = {
    "mcar": [0, 0.05, 0.10, 0.20, 0.35, 0.50],
    "mar": [0, 0.05, 0.10, 0.20, 0.35, 0.50],
    "mnar": [0, 0.05, 0.10, 0.20, 0.35, 0.50],
    "label_noise": [0, 0.05, 0.10, 0.20, 0.30, 0.40],
    "label_noise_minority": [0, 0.10, 0.20, 0.40, 0.60, 0.80],
    "minority_removal": [1, 2, 4, 8, 16, 32],
    "fewer_rows": [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
    "duplicates": [0, 0.10, 0.25, 0.50, 1.00, 2.00],
    "redundant": [0, 0.25, 0.50, 1.00, 2.00, 4.00],
    "irrelevant": [0, 0.25, 0.50, 1.00, 2.00, 4.00],
    "outliers": [0, 0.01, 0.02, 0.05, 0.10, 0.20],
}
TYPES = list(LEVELS)
MIN_MINORITY = 10
MIN_ROWS = 50


def rng_for(rep: int, dtype: str, level: int) -> np.random.Generator:
    return np.random.default_rng([20260924, rep, TYPES.index(dtype), level])


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]


def _set_missing(X: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    X = X.copy()
    for j, c in enumerate(X.columns):
        if mask[:, j].any():
            col = X[c].copy()
            col[mask[:, j]] = np.nan if pd.api.types.is_numeric_dtype(col) else pd.NA
            X[c] = col
    return X


def _rank01(v: np.ndarray, rng) -> np.ndarray:
    """Ranks scaled to (0, 1]; ties broken at random; missing values get the mean rank."""
    v = v.astype(float)
    noise = rng.random(len(v)) * 1e-9
    filled = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(v).any() else 0.0, v)
    order = np.argsort(np.argsort(filled + noise))
    return (order + 1) / len(v)


def missing(X, y, frac, rng, kind):
    n, p = X.shape
    if frac == 0:
        return X, y
    if kind == "mcar":
        mask = rng.random((n, p)) < frac
    elif kind == "mar":
        nums = _numeric_cols(X)
        if not nums:
            return missing(X, y, frac, rng, "mcar")
        anchor = nums[rng.integers(len(nums))]
        prob = np.clip(2 * frac * _rank01(X[anchor].to_numpy(), rng), 0, 1)
        mask = rng.random((n, p)) < prob[:, None]
        mask[:, list(X.columns).index(anchor)] = False
    else:  # mnar: probability grows with the value itself (categoricals: uniform)
        mask = np.zeros((n, p), bool)
        for j, c in enumerate(X.columns):
            if c in _numeric_cols(X):
                prob = np.clip(2 * frac * _rank01(X[c].to_numpy(), rng), 0, 1)
            else:
                prob = np.full(n, frac)
            mask[:, j] = rng.random(n) < prob
    return _set_missing(X, mask), y


def label_noise(X, y, frac, rng):
    y = y.copy()
    flip = rng.random(len(y)) < frac
    y[flip] = 1 - y[flip]
    return X, y


def label_noise_minority(X, y, frac, rng):
    y = y.copy()
    minority = np.flatnonzero(y == 1)
    keep_one = max(0, len(minority) - MIN_MINORITY)
    flip = rng.choice(minority, min(int(round(frac * len(minority))), keep_one), replace=False)
    y[flip] = 0
    return X, y


def minority_removal(X, y, factor, rng):
    if factor == 1:
        return X, y
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    target = max(MIN_MINORITY, int(round(n0 / (n0 / n1 * factor))))
    minority = np.flatnonzero(y == 1)
    keep = np.sort(np.r_[np.flatnonzero(y == 0), rng.choice(minority, min(target, n1), replace=False)])
    return X.iloc[keep].reset_index(drop=True), y[keep]


def fewer_rows(X, y, frac, rng):
    if frac == 1:
        return X, y
    keep = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        k = max(int(round(frac * len(idx))), MIN_MINORITY if cls == 1 else MIN_ROWS - MIN_MINORITY)
        keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X.iloc[keep].reset_index(drop=True), y[keep]


def duplicates(X, y, frac, rng):
    if frac == 0:
        return X, y
    extra = rng.integers(0, len(y), int(round(frac * len(y))))
    idx = np.r_[np.arange(len(y)), extra]
    return X.iloc[idx].reset_index(drop=True), y[idx]


def _standardised_numeric(X: pd.DataFrame) -> pd.DataFrame:
    nums = _numeric_cols(X)
    Z = X[nums].astype(float)
    Z = Z.fillna(Z.median()).fillna(0.0)
    sd = Z.std().replace(0, 1.0)
    return (Z - Z.mean()) / sd


def redundant(X, y, frac, rng):
    k = int(round(frac * X.shape[1]))
    if k == 0:
        return X, y
    Z = _standardised_numeric(X)
    if Z.shape[1] == 0:
        return X.copy(), y
    new = {}
    for i in range(k):
        a, b = rng.integers(Z.shape[1], size=2)
        w = rng.normal(size=2)
        s = w[0] * Z.iloc[:, a].to_numpy() + w[1] * Z.iloc[:, b].to_numpy()
        s = s if s.std() > 0 else Z.iloc[:, a].to_numpy()
        new[f"__redundant_{i}"] = s + rng.normal(scale=0.33 * (s.std() or 1.0), size=len(s))
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(new)], axis=1), y


def irrelevant(X, y, frac, rng):
    k = int(round(frac * X.shape[1]))
    if k == 0:
        return X.copy(), y
    noise = pd.DataFrame(rng.normal(size=(len(y), k)), columns=[f"__noise_{i}" for i in range(k)])
    return pd.concat([X.reset_index(drop=True), noise], axis=1), y


def outliers(X, y, frac, rng):
    if frac == 0:
        return X, y
    X = X.copy()
    for c in _numeric_cols(X):
        v = X[c].astype(float).to_numpy().copy()
        q1, q3 = np.nanpercentile(v, [25, 75]) if np.isfinite(v).any() else (0.0, 0.0)
        iqr = q3 - q1 if q3 > q1 else (np.nanstd(v) or 1.0)
        hit = (rng.random(len(v)) < frac) & ~np.isnan(v)
        side = rng.random(len(v)) < 0.5
        v[hit & side] = q3 + 10 * iqr
        v[hit & ~side] = q1 - 10 * iqr
        X[c] = v
    return X, y


def apply(X: pd.DataFrame, y: np.ndarray, dtype: str, level: int, rep: int):
    """Degraded copy of (X, y) for degradation `dtype` at `level` (0-5) in replication `rep`."""
    if level == 0:
        return X.copy(), y.copy()
    value = LEVELS[dtype][level]
    rng = rng_for(rep, dtype, level)
    if dtype in ("mcar", "mar", "mnar"):
        return missing(X, y, value, rng, dtype)
    fn = {"label_noise": label_noise, "label_noise_minority": label_noise_minority,
          "minority_removal": minority_removal, "fewer_rows": fewer_rows, "duplicates": duplicates,
          "redundant": redundant, "irrelevant": irrelevant, "outliers": outliers}[dtype]
    return fn(X, y, value, rng)
