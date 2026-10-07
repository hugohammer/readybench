"""Phase 5 (Study B): controlled degradation, plan section 11 (git tag prereg-v2).

Work unit = (dataset, replication r = OpenML fold 0-4, degradation type, level 1-5),
plus one clean unit (type "clean", level 0) per dataset and replication.
Each unit degrades the training sample, then
  * performance: light recipe (LightGBM + CatBoost, default + 4 configs each, 80/20
    inner holdout), selected candidate scored on the clean test sample (AUC, AP, nAP, IG);
  * measures: all CPU measure groups + empcv_* on the degraded training data.
Outputs (appended, fixed columns; a rerun skips finished units):
  outputs/phase5/units.csv, perf.csv, measures.csv

  python scripts/phase5_studyB.py --workers 50
  python scripts/phase5_studyB.py --dids 37 --reps 0 --types mcar,duplicates --levels 0,5 --workers 4  # smoke test
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readybench import data, degrade, gold, measures  # noqa: E402
from readybench.metrics import secondary_metrics  # noqa: E402

SEED = 20260924
N_BASE = 24
TRAIN_CAP, TEST_CAP = 8_000, 4_000
N_CONFIGS = 4
REPS = [0, 1, 2, 3, 4]
GROUPS = ["icirca", "thesis", "relevance", "landmarkers", "aidrin", "complexity", "empirical_cv"]
OUT = ROOT / "outputs" / "phase5"
UNIT_COLS = ["did", "rep", "dtype", "level", "ok", "seconds", "n_train", "n_features", "error"]
PERF_COLS = ["did", "rep", "dtype", "level", "auc", "ap", "nap", "ig", "prevalence", "choice", "holdout_auc"]


def select_base() -> pd.DataFrame:
    """B1: eligible primary datasets, 24 evenly spaced over row count."""
    g = pd.read_csv(ROOT / "results" / "gold_v1.csv")
    e = g[(g.category == "primary") & g.flag.isna() & (g.gold_auc >= 0.85) & (g.NumberOfInstances >= 1000)
          & (g.NumberOfFeatures - 1 <= 100)].sort_values(["NumberOfInstances", "did"]).reset_index(drop=True)
    idx = np.unique(np.round(np.linspace(0, len(e) - 1, min(N_BASE, len(e)))).astype(int))
    return e.iloc[idx][["did", "tid", "name", "NumberOfInstances", "NumberOfFeatures", "gold_auc"]]


def _stratified(n_cap: int, y: np.ndarray, seed: int) -> np.ndarray:
    if len(y) <= n_cap:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = [rng.choice(np.flatnonzero(y == c), max(2, int(round(n_cap * (y == c).mean()))), replace=False) for c in (0, 1)]
    return np.sort(np.concatenate(keep))


@functools.lru_cache(maxsize=4)
def base_split(did: int, tid: int, rep: int):
    data.fetch(did, tid)
    X, y = data.load(did)
    y = y.to_numpy()
    s = pd.read_parquet(data.split_path(tid))
    s = s[s.fold == rep]
    tr, te = s.row[~s.is_test].to_numpy(), s.row[s.is_test].to_numpy()
    tr = tr[_stratified(TRAIN_CAP, y[tr], SEED + rep)]
    te = te[_stratified(TEST_CAP, y[te], SEED + 100 + rep)]
    return (X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True), y[te])


def performance(Xd, yd, Xte, yte) -> dict:
    X_all, cats = gold.prepare(pd.concat([Xd, Xte], ignore_index=True))
    y_all = np.r_[yd, yte]
    tr, te = np.arange(len(yd)), np.arange(len(yd), len(y_all))
    best = {}
    for family in ("lightgbm", "catboost"):
        for params in gold.sample_configs(family, N_CONFIGS, SEED):
            r = gold.run_unit(X_all, y_all, cats, tr, te, family, params, threads=1, seed=SEED, holdout=True)
            if family not in best or r["oof_auc"] > best[family]["oof_auc"]:
                best[family] = r
    ho = ~np.isnan(best["lightgbm"]["oof"])
    avg_oof = (best["lightgbm"]["oof"][ho] + best["catboost"]["oof"][ho]) / 2
    cands = {"lightgbm": (best["lightgbm"]["oof_auc"], best["lightgbm"]["test_pred"]),
             "catboost": (best["catboost"]["oof_auc"], best["catboost"]["test_pred"]),
             "average": (roc_auc_score(yd[ho], avg_oof),
                         (best["lightgbm"]["test_pred"] + best["catboost"]["test_pred"]) / 2)}
    choice = max(cands, key=lambda k: cands[k][0])
    m = secondary_metrics(yte, cands[choice][1])
    return {"auc": m["auc"], "ap": m["ap"], "nap": m["nap"], "ig": m["ig"], "prevalence": m["prevalence"],
            "choice": choice, "holdout_auc": cands[choice][0]}


class UnitTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise UnitTimeout()


def work(did: int, tid: int, rep: int, dtype: str, level: int, timeout_s: int):
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    key = {"did": did, "rep": rep, "dtype": dtype, "level": level}
    start = time.perf_counter()
    errors, perf, rows = [], None, []
    try:
        Xtr, ytr, Xte, yte = base_split(did, tid, rep)
        Xd, yd = (Xtr, ytr) if dtype == "clean" else degrade.apply(Xtr, ytr, dtype, level, rep)
        try:
            perf = {**key, **performance(Xd, yd, Xte, yte)}
        except UnitTimeout:
            raise
        except Exception as exc:
            errors.append(f"performance: {type(exc).__name__}: {exc}"[:200])
        ys = pd.Series(yd)
        for g in GROUPS:
            try:
                fn = measures.EXTRA_GROUPS.get(g) or measures.GROUPS[g]
                rows += [{**key, "measure": k, "value": v} for k, v in fn(Xd, ys).items()]
            except UnitTimeout:
                raise
            except Exception as exc:
                errors.append(f"{g}: {type(exc).__name__}: {exc}"[:200])
        unit = {**key, "ok": perf is not None, "n_train": len(yd), "n_features": Xd.shape[1]}
    except UnitTimeout:
        unit = {**key, "ok": False}
        errors.append(f"timeout after {timeout_s} s")
    except Exception as exc:
        unit = {**key, "ok": False}
        errors.append(f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=2)[-200:]}")
    finally:
        signal.alarm(0)
    unit.update(seconds=round(time.perf_counter() - start, 1), error=" || ".join(errors).replace("\n", " "))
    return unit, perf, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--dids", default="")
    ap.add_argument("--reps", default=",".join(map(str, REPS)))
    ap.add_argument("--types", default=",".join(degrade.TYPES))
    ap.add_argument("--levels", default="0,1,2,3,4,5")
    ap.add_argument("--timeout-min", type=float, default=40)
    args = ap.parse_args()

    base = select_base()
    OUT.mkdir(parents=True, exist_ok=True)
    base.to_csv(OUT / "base_datasets.csv", index=False)
    if args.dids:
        base = base[base.did.isin([int(d) for d in args.dids.split(",")])]
    reps = [int(r) for r in args.reps.split(",")]
    levels = [int(lv) for lv in args.levels.split(",")]
    types = args.types.split(",")
    for d in base.itertuples():
        data.fetch(int(d.did), int(d.tid))

    ufile, pfile, mfile = OUT / "units.csv", OUT / "perf.csv", OUT / "measures.csv"
    done = set()
    if ufile.exists():
        u = pd.read_csv(ufile)
        done = set(zip(u.loc[u.ok, "did"], u.loc[u.ok, "rep"], u.loc[u.ok, "dtype"], u.loc[u.ok, "level"]))
    jobs = []
    for rep in reps:
        for d in base.itertuples():
            if 0 in levels and (d.did, rep, "clean", 0) not in done:
                jobs.append((int(d.did), int(d.tid), rep, "clean", 0))
            for t in types:
                for lv in levels:
                    if lv > 0 and (d.did, rep, t, lv) not in done:
                        jobs.append((int(d.did), int(d.tid), rep, t, lv))
    print(f"{len(base)} base datasets; {len(jobs)} units to run ({len(done)} done) on {args.workers} workers", flush=True)

    start, last = time.time(), 0.0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, *j, int(args.timeout_min * 60)) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            unit, perf, rows = fut.result()
            pd.DataFrame([unit]).reindex(columns=UNIT_COLS).to_csv(ufile, mode="a", header=not ufile.exists(), index=False)
            if perf:
                pd.DataFrame([perf]).reindex(columns=PERF_COLS).to_csv(pfile, mode="a", header=not pfile.exists(),
                                                                       index=False)
            if rows:
                pd.DataFrame(rows).reindex(columns=["did", "rep", "dtype", "level", "measure", "value"]).to_csv(
                    mfile, mode="a", header=not mfile.exists(), index=False)
            if unit["error"]:
                print(f"issue did {unit['did']} r{unit['rep']} {unit['dtype']} L{unit['level']}: {unit['error'][:200]}",
                      flush=True)
            el = time.time() - start
            if el - last > 300 or i == len(jobs):
                last = el
                print(f"[{i}/{len(jobs)}] {el / 60:.0f} min elapsed, ~{el / i * (len(jobs) - i) / 60:.0f} min left",
                      flush=True)


if __name__ == "__main__":
    main()
