"""Amendment A1: one table of secondary gold metrics (AP, nAP, IG) per dataset.

Sources: TabRepo stored predictions (results/raw/gold_metrics_tabrepo.csv,
usable folds only) and ReadyBench recipe reruns (gold.csv of the given
pipelines; later pipelines override earlier ones for the same dataset).
Also checks that the reruns reproduce the Phase 2 gold AUC (results/gold_v1.csv).

  python scripts/phase2d_merge_secondary.py outputs/runs/9414 outputs/runs/9420 outputs/runs/9419
Writes results/gold_secondary_v1.csv.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tr = pd.read_csv(ROOT / "results" / "raw" / "gold_metrics_tabrepo.csv")
    tr = tr[tr.usable]
    tr = tr.groupby("did").agg(ap=("ap", "mean"), nap=("nap", "mean"), ig=("ig", "mean"),
                               n_folds=("fold", "nunique")).reset_index().assign(source="tabrepo_stored_predictions")

    reruns = []
    for order, run in enumerate(Path(p) for p in sys.argv[1:]):
        g = pd.read_csv(run / "run" / "outputs" / "phase2" / "gold.csv").assign(order=order, run=run.name)
        reruns.append(g)
    rr = pd.concat(reruns).sort_values("order").drop_duplicates("did", keep="last")
    rr = rr.rename(columns={"gold_ap": "ap", "gold_nap": "nap", "gold_ig": "ig", "gold_auc": "rerun_auc"})

    gold = pd.read_csv(ROOT / "results" / "gold_v1.csv")[["did", "name", "source", "gold_auc"]]
    check = rr.merge(gold, on="did")
    ext = check[check.source == "extension"]
    diff = (ext.rerun_auc - ext.gold_auc).abs()
    print(f"reproducibility: {len(ext)} extension datasets rerun; max |rerun - Phase 2 gold AUC| = {diff.max():.4f}; "
          f"{(diff > 0.002).sum()} above 0.002")
    if (diff > 0.002).any():
        print(ext.loc[diff > 0.002, ["name", "gold_auc", "rerun_auc", "run"]].to_string(index=False))

    rr = rr[["did", "ap", "nap", "ig", "n_folds", "run"]].assign(source="readybench_recipe")
    out = pd.concat([tr, rr[~rr.did.isin(tr.did)]], ignore_index=True)
    used = gold.did
    missing = sorted(set(used) - set(out.did))
    out.to_csv(ROOT / "results" / "gold_secondary_v1.csv", index=False)
    print(f"{len(out)} datasets with secondary metrics; missing: {missing}")


if __name__ == "__main__":
    main()
