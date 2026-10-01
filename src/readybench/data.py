"""Download OpenML tasks once and keep a compact local copy.

Each dataset is stored as data_cache/datasets/<did>.parquet (features plus a
0/1 `__target__` column, minority class = 1, rows in OpenML order) and its task
splits as data_cache/splits/<tid>.parquet (repeat 0, all folds). The raw
OpenML cache is only a download area and is never carried between CI jobs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CACHE, ROOT, setup_openml_cache

DATASETS = CACHE / "datasets"
SPLITS = CACHE / "splits"
TARGET = "__target__"


def dataset_path(did: int) -> Path:
    return DATASETS / f"{did}.parquet"


def split_path(tid: int) -> Path:
    return SPLITS / f"{tid}.parquet"


def fetch(did: int, tid: int, attempts: int = 3) -> dict:
    """Make sure the dataset and its splits are cached; return basic facts.

    Downloads are retried; OpenML's copy of the dataset is removed between
    attempts, because a corrupted download fails its checksum on every reuse."""
    if dataset_path(did).exists() and split_path(tid).exists():
        return {"did": did, "tid": tid, "cached": True}
    import shutil
    import time

    from .paths import OPENML_CACHE

    for attempt in range(1, attempts + 1):
        try:
            return _download(did, tid)
        except Exception:
            if attempt == attempts:
                raise
            shutil.rmtree(OPENML_CACHE / "org" / "openml" / "www" / "datasets" / str(did), ignore_errors=True)
            time.sleep(20 * attempt)


def _download(did: int, tid: int) -> dict:
    import openml

    setup_openml_cache()
    task = openml.tasks.get_task(tid, download_splits=True, download_data=False,
                                 download_qualities=False, download_features_meta_data=False)
    if task.dataset_id != did:
        raise ValueError(f"task {tid} belongs to dataset {task.dataset_id}, not {did}")
    dataset = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                          download_features_meta_data=False)
    X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")

    labels = pd.Series(y).astype("string")
    counts = labels.value_counts()
    if len(counts) != 2:
        raise ValueError(f"target has {len(counts)} classes")
    minority = counts.idxmin()
    X = X.copy()
    X.columns = [str(c) for c in X.columns]
    X[TARGET] = (labels == minority).astype("int8").to_numpy()

    DATASETS.mkdir(parents=True, exist_ok=True)
    X.to_parquet(dataset_path(did), index=False)

    n_folds = task.get_split_dimensions()[1]
    rows = []
    for fold in range(n_folds):
        train, test = task.get_train_test_split_indices(fold=fold, repeat=0)
        rows.append(pd.DataFrame({"fold": fold, "row": np.concatenate([train, test]),
                                  "is_test": np.r_[np.zeros(len(train), bool), np.ones(len(test), bool)]}))
    SPLITS.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_parquet(split_path(tid), index=False)
    return {"did": did, "tid": tid, "cached": False, "positive_label": str(minority), "n_folds": n_folds}


def leakage_drops() -> dict[int, list[str]]:
    """Columns removed from every dataset load (configs/leakage_fixes.csv)."""
    fixes = pd.read_csv(ROOT / "configs" / "leakage_fixes.csv")
    return {int(r.did): r.columns.split(";") for r in fixes.itertuples()}


def load(did: int, apply_leakage_fixes: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(dataset_path(did))
    y = df.pop(TARGET)
    if apply_leakage_fixes:
        drop = leakage_drops().get(did, [])
        missing = set(drop) - set(df.columns)
        if missing:
            raise KeyError(f"leakage fix for {did} names unknown columns {missing}")
        df = df.drop(columns=drop)
    return df, y
