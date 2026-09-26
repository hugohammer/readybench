"""Phase 2: gold standard for pool datasets that TabRepo does not cover.

Datasets: every used extension dataset, plus CALIBRATION_N primary TabRepo
datasets (<= CALIBRATION_MAX_ROWS rows, evenly spaced over TabRepo's gold AUC)
on which the recipe is compared with TabRepo.

Work unit = (dataset, outer fold 0-2, family, config). Units run in parallel,
largest datasets first; every finished unit is appended to
outputs/phase2/units.csv, and predictions go to scratch/phase2_preds/.
At the end outputs/phase2/gold.csv (per dataset) and gold_folds.csv are written.

  python scripts/phase2_gold.py --configs 16 --workers 56
  python scripts/phase2_gold.py --limit-datasets 2 --configs 1 --workers 4   # smoke test
  python scripts/phase2_gold.py --aggregate-only
"""

import argparse
import functools
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench import data, gold  # noqa: E402
from readybench.paths import OUTPUTS, ROOT, TABREPO_CACHE  # noqa: E402

FOLDS = [0, 1, 2]
FAMILIES = ["lightgbm", "catboost"]
CALIBRATION_N = 12
CALIBRATION_MAX_ROWS = 200_000
OUT = OUTPUTS / "phase2"
PREDS = ROOT / "scratch" / "phase2_preds"
UNIT_KEYS = ["did", "fold", "family", "config"]
UNIT_COLUMNS = UNIT_KEYS + ["ok", "threads", "oof_auc", "test_auc", "best_iteration_median", "seconds", "error"]


def tabrepo_gold() -> pd.DataFrame:
    b = pd.read_parquet(TABREPO_CACHE / "baselines.parquet")
    ag = b[b.framework == "AutoGluon_bq_4h8c_2023_11_14"]
    return ag.groupby("tid").metric_error.mean().rsub(1).rename("tabrepo_ag_auc").reset_index()


def tabrepo_boosting_by_val() -> pd.DataFrame:
    """Per fold: the LightGBM/CatBoost config with the best validation error, scored on test."""
    c = pd.read_parquet(TABREPO_CACHE / "configs.parquet")
    c = c[c.framework.str.startswith(("LightGBM", "CatBoost")) & c.fold.isin(FOLDS)]
    best = c.loc[c.groupby(["tid", "fold"]).metric_error_val.idxmin()]
    return best.groupby("tid").metric_error.mean().rsub(1).rename("tabrepo_boost_byval_auc").reset_index()


def select_datasets() -> pd.DataFrame:
    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    ext = pool[pool.used & (pool.source == "extension")].assign(role="extension")
    tr = pool[pool.used & (pool.source == "tabrepo") & (pool.category == "primary")
              & (pool.NumberOfInstances <= CALIBRATION_MAX_ROWS)]
    tr = tr.merge(tabrepo_gold(), on="tid").sort_values("tabrepo_ag_auc").reset_index(drop=True)
    picks = np.unique(np.round(np.linspace(0, len(tr) - 1, CALIBRATION_N)).astype(int))
    cal = tr.iloc[picks].assign(role="calibration")
    cols = ["did", "tid", "name", "category", "NumberOfInstances", "NumberOfFeatures", "role"]
    return pd.concat([ext[cols], cal[cols]], ignore_index=True)


@functools.lru_cache(maxsize=2)
def load_prepared(did: int, tid: int):
    data.fetch(did, tid)  # no-op when cached
    X, y = data.load(did)
    X, cats = gold.prepare(X)
    splits = pd.read_parquet(data.split_path(tid))
    return X, y.to_numpy(), cats, splits


def work(did: int, tid: int, fold: int, family: str, config: int, params: dict, threads: int, seed: int) -> dict:
    key = {"did": did, "fold": fold, "family": family, "config": config}
    try:
        X, y, cats, splits = load_prepared(did, tid)
        s = splits[splits.fold == fold]
        train_idx, test_idx = s.row[~s.is_test].to_numpy(), s.row[s.is_test].to_numpy()
        res = gold.run_unit(X, y, cats, train_idx, test_idx, family, params, threads, seed)
        PREDS.mkdir(parents=True, exist_ok=True)
        np.savez(PREDS / f"{did}_{fold}_{family}_{config}.npz", oof=res.pop("oof"), test=res.pop("test_pred"))
        return {**key, "ok": True, "threads": threads, **res}
    except Exception as exc:
        return {**key, "ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}


def run(datasets: pd.DataFrame, n_configs: int, workers: int, seed: int, threads: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    units_file = OUT / "units.csv"
    done = set()
    if units_file.exists():
        prev = pd.read_csv(units_file)
        done = set(map(tuple, prev.loc[prev.ok, UNIT_KEYS].itertuples(index=False)))

    # download sequentially first: parallel downloads of one dataset collide in the OpenML cache
    for d in datasets.itertuples():
        data.fetch(int(d.did), int(d.tid))
    print(f"{len(datasets)} datasets, {threads} thread(s) per unit", flush=True)

    configs = {f: gold.sample_configs(f, n_configs, seed) for f in FAMILIES}
    # Round-robin over configs: every dataset gets its default config first, so small
    # datasets finish early and partial results are useful. Big datasets first within a config.
    order = datasets.assign(cells=datasets.NumberOfInstances * datasets.NumberOfFeatures)
    order = order.sort_values("cells", ascending=False)
    jobs = []
    for cid in range(n_configs + 1):
        for d in order.itertuples():
            for fold in FOLDS:
                for family in FAMILIES:
                    if (d.did, fold, family, cid) not in done:
                        jobs.append((int(d.did), int(d.tid), fold, family, cid, configs[family][cid], threads, seed))
    print(f"{len(jobs)} units to run ({len(done)} already done) on {workers} workers", flush=True)

    start, last_beat = time.time(), 0.0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            pd.DataFrame([row]).reindex(columns=UNIT_COLUMNS).to_csv(
                units_file, mode="a", header=not units_file.exists(), index=False)
            if not row["ok"]:
                print(f"FAILED {row['did']} f{row['fold']} {row['family']} c{row['config']}: {row['error']}", flush=True)
            elif row["seconds"] > 300:
                print(f"slow unit: did {row['did']} f{row['fold']} {row['family']} c{row['config']} "
                      f"took {row['seconds'] / 60:.0f} min", flush=True)
            el = time.time() - start
            if el - last_beat > 300 or i == len(jobs):
                last_beat = el
                print(f"[{i}/{len(jobs)}] {el / 60:.0f} min elapsed, ~{el / i * (len(jobs) - i) / 60:.0f} min left "
                      f"(rough: early units are the biggest)", flush=True)


def aggregate(datasets: pd.DataFrame) -> None:
    units = pd.read_csv(OUT / "units.csv")
    units = units[units.ok].drop_duplicates(UNIT_KEYS, keep="last")
    rows = []
    for (did, fold), g in units.groupby(["did", "fold"]):
        tid = int(datasets.loc[datasets.did == did, "tid"].iloc[0])
        data.fetch(int(did), tid)
        s = pd.read_parquet(data.split_path(tid))
        s = s[s.fold == fold]
        _, y = data.load(int(did))
        y = y.to_numpy()
        y_train, y_test = y[s.row[~s.is_test].to_numpy()], y[s.row[s.is_test].to_numpy()]

        row = {"did": did, "fold": fold}
        best = {}
        for family in FAMILIES:
            gf = g[g.family == family]
            if gf.empty:
                continue
            b = gf.loc[gf.oof_auc.idxmax()]
            best[family] = np.load(PREDS / f"{did}_{fold}_{family}_{int(b.config)}.npz")
            row[f"{family}_tuned_auc"] = b.test_auc
            row[f"{family}_tuned_oof_auc"] = b.oof_auc
            default = gf[gf.config == 0]
            row[f"{family}_default_auc"] = default.test_auc.iloc[0] if len(default) else np.nan
        cands = {f: (row[f"{f}_tuned_oof_auc"], row[f"{f}_tuned_auc"]) for f in best}
        if len(best) == 2:
            oof = (best["lightgbm"]["oof"] + best["catboost"]["oof"]) / 2
            test = (best["lightgbm"]["test"] + best["catboost"]["test"]) / 2
            ok = ~np.isnan(oof)
            cands["average"] = (roc_auc_score(y_train[ok], oof[ok]), roc_auc_score(y_test, test))
        chosen = max(cands, key=lambda k: cands[k][0])
        row.update({"gold_choice": chosen, "gold_auc": cands[chosen][1], "n_units": len(g)})
        rows.append(row)

    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "gold_folds.csv", index=False)
    per = folds.groupby("did").agg(gold_auc=("gold_auc", "mean"), gold_auc_sd=("gold_auc", "std"),
                                    n_folds=("fold", "nunique"),
                                    **{c: (c, "mean") for c in folds.columns if c.endswith("_auc")
                                       and c not in ("gold_auc",) and "oof" not in c}).reset_index()
    per = datasets.merge(per, on="did", how="left")
    per = per.merge(tabrepo_gold(), on="tid", how="left").merge(tabrepo_boosting_by_val(), on="tid", how="left")
    per.to_csv(OUT / "gold.csv", index=False)

    cal = per[per.role == "calibration"].dropna(subset=["gold_auc"])
    if len(cal) > 1:
        from scipy.stats import spearmanr
        diff = cal.gold_auc - cal.tabrepo_ag_auc
        print(f"calibration on {len(cal)} TabRepo datasets: Spearman vs AutoGluon "
              f"{spearmanr(cal.gold_auc, cal.tabrepo_ag_auc)[0]:.3f}, mean diff {diff.mean():+.4f}, "
              f"max |diff| {diff.abs().max():.4f}")
    print(per[["name", "role", "gold_auc", "tabrepo_ag_auc", "tabrepo_boost_byval_auc"]].round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=int, default=16, help="random configs per family (plus the default)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit-datasets", type=int, default=0, help="smoke test: smallest N datasets only")
    parser.add_argument("--min-rows", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=10**9)
    parser.add_argument("--threads", type=int, default=1, help="threads per unit")
    parser.add_argument("--dids", default="", help="comma-separated dataset ids to (re)run")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    seed = yaml.safe_load((ROOT / "configs" / "study_a.yaml").read_text())["seed"]
    datasets = select_datasets()
    datasets = datasets[datasets.NumberOfInstances.between(args.min_rows, args.max_rows)]
    if args.dids:
        datasets = datasets[datasets.did.isin([int(d) for d in args.dids.split(",")])]
    if args.limit_datasets:
        datasets = datasets.sort_values("NumberOfInstances").head(args.limit_datasets)
    print(datasets.groupby("role").size().to_string(), flush=True)
    if not args.aggregate_only:
        run(datasets, args.configs, args.workers, seed, args.threads)
    aggregate(datasets)


if __name__ == "__main__":
    main()
