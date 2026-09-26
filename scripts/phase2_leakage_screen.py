"""Phase 2b: screen every pool dataset for target leakage.

Rule (fixed before looking at results): a dataset is flagged when a single
column, or the missingness indicator of a column, reaches AUC >= 0.99 on its
own. Numeric columns are scored on their raw values (rows without missing
values); categorical columns with a 5-fold out-of-fold target-mean encoding, so
that unique ID columns do not score trivially. Flags are reviewed by hand and
resolved in configs/leakage_fixes.csv.

Writes outputs/phase2/leakage_screen.csv (top columns per dataset).
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench import data  # noqa: E402
from readybench.paths import OUTPUTS, ROOT  # noqa: E402

FLAG = 0.99
MAX_ROWS = 200_000
TOP = 3


def auc(y, score) -> float:
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def oof_target_encoding(codes: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    enc = np.zeros(len(y))
    prior = y.mean()
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(codes, y):
        stats = pd.DataFrame({"c": codes[tr], "y": y[tr]}).groupby("c").y.agg(["sum", "count"])
        smoothed = (stats["sum"] + 10 * prior) / (stats["count"] + 10)
        enc[te] = pd.Series(codes[te]).map(smoothed).fillna(prior).to_numpy()
    return enc


def screen(did: int, tid: int) -> list[dict]:
    data.fetch(did, tid)
    X, y = data.load(did)
    if len(X) > MAX_ROWS:
        keep = np.random.default_rng(0).choice(len(X), MAX_ROWS, replace=False)
        X, y = X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
    y = y.to_numpy()
    rows = []
    for col in X.columns:
        s = X[col]
        miss = s.isna().to_numpy()
        if 0 < miss.mean() < 1:
            rows.append({"did": did, "column": col, "kind": "missingness", "auc": auc(y, miss)})
        if miss.mean() > 0.5:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = s.to_numpy(dtype="float64")
            ok = ~np.isnan(v)
            if len(np.unique(v[ok])) > 1 and len(np.unique(y[ok])) == 2:
                rows.append({"did": did, "column": col, "kind": "numeric", "auc": auc(y[ok], v[ok])})
        else:
            codes = s.astype("string").fillna("__nan__").to_numpy()
            if len(set(codes)) > 1:
                rows.append({"did": did, "column": col, "kind": "categorical",
                             "auc": auc(y, oof_target_encoding(codes, y))})
    return sorted(rows, key=lambda r: -r["auc"])[:TOP]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    pool = pool[pool.used]
    for d in pool.itertuples():  # sequential downloads (see phase2_gold)
        data.fetch(int(d.did), int(d.tid))
    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(screen, int(d.did), int(d.tid)): d.did for d in pool.itertuples()}
        for fut in as_completed(futs):
            results.extend(fut.result())
    df = pd.DataFrame(results).merge(pool[["did", "name", "source", "category"]], on="did")
    df["flag"] = df.auc >= FLAG
    df["auc"] = df.auc.round(4)
    out = OUTPUTS / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(["flag", "auc"], ascending=False).to_csv(out / "leakage_screen.csv", index=False)
    flagged = df[df.flag]
    print(f"{flagged.did.nunique()} of {pool.did.nunique()} datasets flagged")
    print(flagged[["did", "name", "source", "category", "column", "kind", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
