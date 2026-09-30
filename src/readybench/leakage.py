"""Shared helpers for target-aware column scoring (leakage screen, relevance measures)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def oof_target_encoding(codes: np.ndarray, y: np.ndarray, seed: int = 0, smoothing: float = 10) -> np.ndarray:
    """5-fold out-of-fold target-mean encoding with smoothing towards the prior."""
    enc = np.zeros(len(y))
    prior = y.mean()
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(codes, y):
        stats = pd.DataFrame({"c": codes[tr], "y": y[tr]}).groupby("c").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        enc[te] = pd.Series(codes[te]).map(smoothed).fillna(prior).to_numpy()
    return enc
