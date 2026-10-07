"""Secondary gold metrics (plan amendment A1). Positive class = minority class."""

import math

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def secondary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """AUC, average precision (AP), normalized AP (lift over prevalence) and
    information gain IG = 1 - log-loss / H(prevalence)."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pi = y.mean()
    ap = average_precision_score(y, p)
    h = -(pi * math.log(pi) + (1 - pi) * math.log(1 - pi))
    return {"auc": roc_auc_score(y, p), "ap": ap, "nap": (ap - pi) / (1 - pi),
            "ig": 1 - log_loss(y, p, labels=[0, 1]) / h, "prevalence": pi, "n_test": len(y)}
