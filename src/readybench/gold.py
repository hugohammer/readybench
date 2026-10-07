"""Gold-standard recipe for datasets outside TabRepo (analysis plan section 4b).

Per OpenML outer fold (0-2):
  * LightGBM and CatBoost, each with its default config plus N random configs;
  * every config is trained as a cross-validation bag on the outer training
    fold: k inner folds (k=5, or a single 80/20 holdout above HOLDOUT_ROWS rows),
    early stopping on the inner validation fold, and test predictions averaged
    over the k models. This mirrors TabRepo's *_BAG_L1 models;
  * selection uses only the inner out-of-fold (OOF) AUC. Candidates are the
    best LightGBM, the best CatBoost and their average. The test fold is only
    used to score the selected candidate.
"""

import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# LightGBM warns on every fit that eval_set API details will change; harmless here.
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

HOLDOUT_ROWS = 50_000
INNER_FOLDS = 5
EARLY_STOPPING = 100


# ---------------------------------------------------------------- search spaces
def sample_configs(family: str, n: int, seed: int) -> list[dict]:
    """Config 0 is the library default; configs 1..n are random draws."""
    rng = np.random.default_rng(seed + (0 if family == "lightgbm" else 1))
    logu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    configs = [{}]
    for _ in range(n):
        if family == "lightgbm":
            configs.append({
                "learning_rate": logu(0.01, 0.2),
                "num_leaves": int(round(logu(8, 256))),
                "min_child_samples": int(round(logu(5, 100))),
                "colsample_bytree": float(rng.uniform(0.4, 1.0)),
                "subsample": float(rng.uniform(0.5, 1.0)),
                "subsample_freq": 1,
                "reg_lambda": logu(1e-3, 10.0),
            })
        else:
            configs.append({
                "learning_rate": logu(0.02, 0.2),
                "depth": int(rng.integers(4, 9)),
                "l2_leaf_reg": logu(1.0, 10.0),
            })
    return configs


# ---------------------------------------------------------------- data handling
def prepare(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Numeric columns stay numeric; everything else becomes a pandas category.
    Column names are reduced to [A-Za-z0-9_] (LightGBM rejects JSON special
    characters, e.g. MagicTelescope's 'fLength:'); renaming does not affect training."""
    X = X.copy()
    safe, seen = [], set()
    for col in X.columns:
        name = re.sub(r"[^0-9A-Za-z_]", "_", str(col)) or "col"
        while name in seen:
            name += "_"
        seen.add(name)
        safe.append(name)
    X.columns = safe
    cats = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_bool_dtype(s):
            X[col] = s.astype("float32")
        elif pd.api.types.is_datetime64_any_dtype(s):
            X[col] = s.astype("int64").astype("float64")
        elif not pd.api.types.is_numeric_dtype(s):
            X[col] = s.astype("string").astype("category")
            cats.append(col)
    return X, cats


def for_catboost(X: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    X = X.copy()
    for col in cats:
        X[col] = X[col].astype("string").fillna("__nan__").astype(str)
    return X


# ---------------------------------------------------------------- one model fit
def fit_predict(family: str, params: dict, threads: int, X_tr, y_tr, X_va, y_va, X_te, cats, seed):
    if family == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping

        model = LGBMClassifier(n_estimators=5000, n_jobs=threads, random_state=seed, verbose=-1, **params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                  callbacks=[early_stopping(EARLY_STOPPING, verbose=False)])
        best = model.best_iteration_ or model.n_estimators
        return model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1], best

    from catboost import CatBoostClassifier

    model = CatBoostClassifier(iterations=3000, thread_count=threads, random_seed=seed, verbose=False,
                               eval_metric="AUC", od_type="Iter", od_wait=EARLY_STOPPING,
                               allow_writing_files=False, cat_features=cats, **params)
    Xtr, Xva, Xte = (for_catboost(d, cats) for d in (X_tr, X_va, X_te))
    model.fit(Xtr, y_tr, eval_set=(Xva, y_va), use_best_model=True)
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1], model.get_best_iteration()


def inner_splits(y_train: np.ndarray, seed: int, holdout: bool = False):
    idx = np.arange(len(y_train))
    if holdout or len(y_train) > HOLDOUT_ROWS:
        tr, va = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=seed)
        return [(tr, va)]
    return list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(idx, y_train))


def run_unit(X: pd.DataFrame, y: np.ndarray, cats: list[str], train_idx, test_idx,
             family: str, params: dict, threads: int, seed: int, holdout: bool = False) -> dict:
    """Train one config as a CV bag on the outer training fold."""
    start = time.perf_counter()
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test = X.iloc[test_idx]
    oof = np.full(len(train_idx), np.nan)
    test_pred = np.zeros(len(test_idx))
    splits = inner_splits(y_train, seed, holdout)
    iters = []
    for tr, va in splits:
        p_va, p_te, best = fit_predict(family, params, threads, X_train.iloc[tr], y_train[tr],
                                       X_train.iloc[va], y_train[va], X_test, cats, seed)
        oof[va] = p_va
        test_pred += p_te / len(splits)
        iters.append(best)
    covered = ~np.isnan(oof)
    return {
        "oof_auc": float(roc_auc_score(y_train[covered], oof[covered])),
        "test_auc": float(roc_auc_score(y[test_idx], test_pred)),
        "best_iteration_median": float(np.median(iters)),
        "seconds": round(time.perf_counter() - start, 1),
        "oof": oof.astype("float32"),
        "test_pred": test_pred.astype("float32"),
    }
