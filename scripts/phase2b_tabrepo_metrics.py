"""Amendment A1, TabRepo part: imbalance-sensitive metrics of the secondary-gold model.

For every used TabRepo dataset and fold 0-2, take the LightGBM/CatBoost config
with the lowest validation error, download only that model's test predictions
(HTTP range request into pred-test.dat, which is models x rows float32), and
compute AUC (checked against TabRepo's stored value), AP, nAP and IG with the
minority class as positive. Writes results/raw/gold_metrics_tabrepo.csv.
"""

import io
import json
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readybench.metrics import secondary_metrics  # noqa: E402
from readybench.paths import TABREPO_CACHE  # noqa: E402

BASE = "https://huggingface.co/datasets/Tabrepo/tabrepo/resolve/main/2023_11_14/model_predictions"
FOLDS = [0, 1, 2]
# Not in TabRepo's public prediction release (HTTP 404); handled by scripts/phase2c_rerun_selected.py
NO_STORED_PREDICTIONS = {"Higgs", "KDDCup09_appetency", "MagicTelescope", "airlines", "albert", "christine",
                         "porto-seguro"}


def get(url: str, byte_range: tuple[int, int] | None = None, attempts: int = 4) -> bytes:
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for i in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120).read()
        except Exception:
            if i == attempts - 1:
                raise
    raise RuntimeError("unreachable")


def fold_predictions(dataset: str, fold: int, model: str) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(get(f"{BASE}/{dataset}/{fold}/metadata.json"))
    zipped = get(f"{BASE}/{dataset}/{fold}/label-test.csv.zip")
    labels = pd.read_csv(io.BytesIO(zipfile.ZipFile(io.BytesIO(zipped)).read("label-test.csv")), index_col=0)
    i = meta["models"].index(model)
    n = meta["pred_test_shape"][1]
    raw = get(f"{BASE}/{dataset}/{fold}/pred-test.dat", (i * n * 4, (i + 1) * n * 4 - 1))
    return labels.iloc[:, 0].to_numpy(), np.frombuffer(raw, dtype="float32").astype("float64")


def main() -> None:
    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    pool = pool[pool.used & (pool.source == "tabrepo")]
    c = pd.read_parquet(TABREPO_CACHE / "configs.parquet")
    c = c[c.framework.str.startswith(("LightGBM", "CatBoost")) & c.fold.isin(FOLDS) & c.tid.isin(pool.tid)]
    best = c.loc[c.groupby(["tid", "fold"]).metric_error_val.idxmin()]
    best = best[~best.dataset.isin(NO_STORED_PREDICTIONS)]

    def work(r):
        y, p = fold_predictions(r.dataset, int(r.fold), r.framework)
        return {"tid": int(r.tid), "dataset": r.dataset, "fold": int(r.fold), "model": r.framework,
                "tabrepo_auc": 1 - r.metric_error, "y": y, "p": p}

    with ThreadPoolExecutor(8) as ex:
        folds = list(ex.map(work, best.itertuples()))

    # positive class = minority over the three test folds of the dataset (AUC is unaffected)
    rows = []
    for tid, group in pd.DataFrame(folds).groupby("tid"):
        share_one = np.concatenate(group.y.tolist()).mean()
        positive = 1 if share_one <= 0.5 else 0
        for r in group.itertuples():
            y = (r.y == positive).astype(int)
            p = r.p if positive == 1 else 1 - r.p
            m = secondary_metrics(y, p)
            rows.append({"tid": tid, "dataset": r.dataset, "fold": r.fold, "model": r.model,
                         "auc_check_diff": m["auc"] - r.tabrepo_auc, **m})
    out = pd.DataFrame(rows).merge(pool[["did", "tid"]], on="tid")
    # A fold whose AUC does not reproduce TabRepo's value has broken stored predictions
    # (mammography fold 1: all zeros for every model); it is marked unusable, not repaired.
    out["usable"] = out.auc_check_diff.abs() <= 0.002
    for r in out[~out.usable].itertuples():
        print(f"unusable stored predictions: {r.dataset} fold {r.fold} (AUC diff {r.auc_check_diff:+.3f})")
    target = ROOT / "results" / "raw" / "gold_metrics_tabrepo.csv"
    out.to_csv(target, index=False)
    ok = out[out.usable]
    print(f"{out.did.nunique()} datasets, {len(ok)}/{len(out)} usable folds; "
          f"max |AUC - TabRepo| on usable folds = {ok.auc_check_diff.abs().max():.2e}")
    print(f"-> {target}")


if __name__ == "__main__":
    main()
