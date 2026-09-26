"""Phase 2, final step: one gold-standard table for the whole pool.

  primary gold   = TabRepo AutoGluon (best quality, 4 h) for TabRepo datasets,
                   the ReadyBench recipe (tuned, CV-bagged LightGBM/CatBoost)
                   for extension datasets;
  secondary gold = tuned boosting selected on validation data for every
                   dataset (TabRepo's LightGBM/CatBoost configs by validation
                   error, or the ReadyBench recipe), i.e. one uniform method.

Later runs override earlier ones for the same dataset (e.g. reruns after a
leakage fix). Writes results/gold_v1.csv.

  python scripts/phase2_merge.py outputs/runs/9199 outputs/runs/9200 outputs/runs/9218
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench.paths import ROOT, TABREPO_CACHE  # noqa: E402

FOLDS = [0, 1, 2]


def tabrepo_tables() -> pd.DataFrame:
    b = pd.read_parquet(TABREPO_CACHE / "baselines.parquet")
    ag = b[(b.framework == "AutoGluon_bq_4h8c_2023_11_14") & b.fold.isin(FOLDS)].assign(auc=lambda d: 1 - d.metric_error)
    ag = ag.groupby("tid").auc.agg(["mean", "std"]).rename(columns={"mean": "gold_auc", "std": "gold_auc_sd"})
    c = pd.read_parquet(TABREPO_CACHE / "configs.parquet")
    c = c[c.framework.str.startswith(("LightGBM", "CatBoost")) & c.fold.isin(FOLDS)]
    best = c.loc[c.groupby(["tid", "fold"]).metric_error_val.idxmin()]
    boost = best.groupby("tid").metric_error.mean().rsub(1).rename("secondary_gold_auc")
    return ag.join(boost).reset_index()


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]]
    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    pool = pool[pool.used][["did", "tid", "name", "source", "category", "flag", "NumberOfInstances", "NumberOfFeatures"]]

    own = []
    for order, run in enumerate(runs):
        g = pd.read_csv(run / "run" / "outputs" / "phase2" / "gold.csv").assign(run=run.name, order=order)
        own.append(g)
    own = pd.concat(own).sort_values("order").drop_duplicates("did", keep="last")
    own = own[["did", "role", "run", "gold_auc", "gold_auc_sd", "lightgbm_tuned_auc", "catboost_tuned_auc"]]

    tr = pool[pool.source == "tabrepo"].merge(tabrepo_tables(), on="tid", how="left").assign(
        gold_source="tabrepo_autogluon_bq_4h", secondary_source="tabrepo_boosting_by_val")
    ext = pool[pool.source == "extension"].merge(own[own.role == "extension"], on="did", how="left")
    ext = ext.assign(secondary_gold_auc=ext.gold_auc, gold_source="readybench_recipe",
                     secondary_source="readybench_recipe")
    cal = own[own.role == "calibration"][["did", "gold_auc"]].rename(columns={"gold_auc": "recipe_auc_calibration"})
    tr = tr.merge(cal, on="did", how="left")

    out = pd.concat([tr, ext], ignore_index=True)
    missing = out[out.gold_auc.isna()]
    if len(missing):
        sys.exit(f"no gold standard for: {missing.name.tolist()}")
    target = ROOT / "results" / "gold_v1.csv"
    target.parent.mkdir(exist_ok=True)
    cols = ["did", "tid", "name", "source", "category", "flag", "NumberOfInstances", "NumberOfFeatures",
            "gold_auc", "gold_auc_sd", "gold_source", "secondary_gold_auc", "secondary_source",
            "recipe_auc_calibration", "lightgbm_tuned_auc", "catboost_tuned_auc", "run"]
    out[cols].round(5).to_csv(target, index=False)

    prim = out[out.category == "primary"]
    print(f"{len(out)} datasets with a gold standard; primary {len(prim)}")
    for t in (0.75, 0.80, 0.85):
        print(f"  primary with gold AUC < {t}: {(prim.gold_auc < t).sum()}")
    print(f"-> {target}")


if __name__ == "__main__":
    main()
