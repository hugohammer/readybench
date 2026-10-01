"""Phase 3, measure group R1b: DREAMER (Ahangaran et al., BMC Med Inform Decis Mak 2024).

Runs the vendored DREAMER code (third_party/dreamer, unchanged) in its own
environment with the dependency versions it was written for (Python 3.10,
pandas 1.5, numpy 1.23, scikit-learn 1.2; see ci/requirements-dreamer.txt).

Per dataset: stratified subsample of <= MAX_ROWS rows (DREAMER's agglomerative
clustering is quadratic in rows), categorical columns as ordinal codes (DREAMER
requires numeric input), SUBTABLES random subtables (the lower end of the
authors' recommended 1,000-1,000,000), one subprocess per dataset in its own
folder, wall-clock timeout per dataset. The readiness score is the master
table's "Total Quality"; its components are recorded too.

Note on DREAMER's "subtables" setting: the code starts one run per entry of
"cores" and every run draws "subtables" subtables, so the total is
cores x subtables (the README describes it as the total). We fix the TOTAL
(--total-subtables, default 1,000) and set subtables = ceil(total / runs).
A first attempt with cores=56 and subtables=1000 (56,000 per dataset) took
about an hour per dataset (pipeline 9283, canceled).

Writes outputs/phase3/dreamer.csv (one row per dataset).
  python scripts/phase3_dreamer.py --total-subtables 1000 --runs 40 [--dids 31,37] [--timeout-min 45]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readybench import data  # noqa: E402

MAX_ROWS = 5_000
DREAMER_SRC = ROOT / "third_party" / "dreamer" / "DREAMER"
WORK = ROOT / "scratch" / "dreamer"
OUT = ROOT / "outputs" / "phase3"
MASTER_FIELDS = ["1-PC", "1-SpearmanCorrelation", "1-NullRatio", "1-Outliers", "1-ClassOverlap",
                 "Classify Accuracy", "Clustering Accuracy", "Average Accuracy", "Total Quality"]


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        k = max(1, int(round(n * len(idx) / len(y))))
        keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def prepare_csv(did: int, folder: Path, seed: int) -> tuple[int, int]:
    X, y = data.load(did)
    keep = stratified_subsample(y.to_numpy(), MAX_ROWS, seed)
    X, y = X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
            codes = X[col].astype("category").cat.codes.astype("float64")
            X[col] = codes.where(codes >= 0)  # missing stays missing
    X = X.astype("float64")
    X.columns = [f"f{i}" for i in range(X.shape[1])]  # DREAMER splits on commas; avoid odd names
    X.assign(target=y.to_numpy()).to_csv(folder / "data.csv", index=False)
    return X.shape


def run_one(did: int, total_subtables: int, runs: int, timeout_min: float, seed: int) -> dict:
    folder = WORK / str(did)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    n, p = prepare_csv(did, folder, seed)
    per_run = math.ceil(total_subtables / runs)
    (folder / "DREAMER_Config.json").write_text(json.dumps({
        "search_space": {"subtables": per_run, "cores": runs},
        "data_settings": {"file_name": "data.csv", "target_column": "target", "output_folder": "out"},
        "free_params": {"rows_exclusion_ratio": 0.7, "columns_exclusion_ratio": 0.5,
                        "clustering_weight": 0.5, "classification_weight": 0.5, "precision": 4}}))
    env = {**os.environ, "PYTHONPATH": str(DREAMER_SRC), "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": str(seed)}
    start = time.perf_counter()
    row = {"did": did, "n_rows_used": n, "n_features_used": p, "runs": runs, "subtables_per_run": per_run,
           "subtables_total": per_run * runs}
    try:
        proc = subprocess.run([sys.executable, "-c", "from DREAMER import DREAMER; DREAMER().run()"],
                              cwd=folder, env=env, capture_output=True, text=True, timeout=timeout_min * 60)
        row["seconds"] = round(time.perf_counter() - start, 1)
        master = next(folder.glob("out/**/*Master*"), None)
        if proc.returncode != 0 or master is None:
            row.update(ok=False, error=(proc.stderr or proc.stdout)[-400:].replace("\n", " | "))
            return row
        table = pd.read_csv(master)
        row.update(ok=True, **{f"dreamer_{k}": float(table[k].iloc[0]) for k in MASTER_FIELDS if k in table})
    except subprocess.TimeoutExpired:
        row.update(ok=False, seconds=round(time.perf_counter() - start, 1), error=f"timeout after {timeout_min} min")
    finally:
        shutil.rmtree(folder / "out", ignore_errors=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-subtables", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=40, help="DREAMER runs = parallel worker tasks ('cores')")
    parser.add_argument("--timeout-min", type=float, default=45)
    parser.add_argument("--dids", default="")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    pool = pool[pool.used]
    if args.dids:
        pool = pool[pool.did.isin([int(d) for d in args.dids.split(",")])]

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "dreamer.csv"
    done = set(pd.read_csv(target).did) if target.exists() else set()
    for i, d in enumerate(pool.sort_values("NumberOfFeatures").itertuples(), 1):
        if d.did in done:
            continue
        try:
            data.fetch(int(d.did), int(d.tid))
            row = run_one(int(d.did), args.total_subtables, args.runs, args.timeout_min, args.seed)
        except Exception as exc:  # one bad dataset must not stop the run
            row = {"did": int(d.did), "ok": False, "seconds": 0.0, "n_rows_used": 0, "n_features_used": 0,
                   "error": f"{type(exc).__name__}: {exc}"[:400]}
        pd.DataFrame([row]).to_csv(target, mode="a", header=not target.exists(), index=False)
        status = f"TQ={row.get('dreamer_Total Quality')}" if row["ok"] else f"FAILED {row['error'][:120]}"
        print(f"[{i}/{len(pool)}] {d.name} ({row['n_rows_used']}x{row['n_features_used']}): "
              f"{status} in {row['seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
