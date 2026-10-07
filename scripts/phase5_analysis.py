"""Study B analysis: plan section 11, B5 (git tag prereg-v2).

  python scripts/phase5_analysis.py outputs/runs/9422 outputs/runs/9423 ...   # one dir per replication job
Outputs in outputs/phase5_analysis/: delta_perf_by_type.csv, h4a_sensitivity.csv, h4b_proportionality.csv,
h4c_false_alarms.csv, tracking.csv, hypotheses_h4.json, merged long tables.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from readybench.measures import DIRECTION  # noqa: E402
from phase4_analysis import Q, T  # noqa: E402

OUT = ROOT / "outputs" / "phase5_analysis"
COMPLETENESS = ["aidrin_completeness", "aidrin_row_completeness", "aidrin_feature_coverage", "thesis_c1_completeness"]
TARGETED = ([(m, t) for m in COMPLETENESS for t in ("mcar", "mar", "mnar")]
            + [("aidrin_duplicity", "duplicates"), ("aidrin_outliers", "outliers")]
            + [(m, t) for m in ("aidrin_imbalance_degree", "thesis_c2_balance")
               for t in ("minority_removal", "label_noise_minority")]
            + [("thesis_c3_sample_size", "fewer_rows")]
            + [(m, "redundant") for m in ("aidrin_max_pairwise_corr", "thesis_c4_feature_quality")])
LEVELS = [0, 1, 2, 3, 4, 5]


def rho(a, b) -> float:
    """Spearman; a constant series counts as 0 (no reaction)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3 or np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
        return 0.0
    return float(spearmanr(a[ok], b[ok])[0])


def grouped_rho(df: pd.DataFrame, by, x: str, y: str) -> pd.Series:
    """rho(x, y) within each group, as a float Series (empty input -> empty Series)."""
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series({k: rho(g[x], g[y]) for k, g in df.groupby(by)}, dtype=float)


def load(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf, meas = [], []
    for d in run_dirs:
        base = d / "run" / "outputs" / "phase5"
        perf.append(pd.read_csv(base / "perf.csv"))
        meas.append(pd.read_csv(base / "measures.csv"))
    keys = ["did", "rep", "dtype", "level"]
    perf = pd.concat(perf).drop_duplicates(keys, keep="last")
    meas = pd.concat(meas).drop_duplicates(keys + ["measure"], keep="last")
    return perf, meas


def with_clean_level0(df: pd.DataFrame, value_cols: list[str], extra_keys: list[str]) -> pd.DataFrame:
    """Expand the shared clean unit into level 0 of every degradation type."""
    clean = df[df.dtype == "clean"]
    types = sorted(set(df.dtype) - {"clean"})
    lvl0 = pd.concat([clean.assign(dtype=t, level=0) for t in types])
    return pd.concat([df[df.dtype != "clean"], lvl0], ignore_index=True)[["did", "rep", "dtype", "level"] + extra_keys
                                                                          + value_cols]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    perf, meas = load([Path(p) for p in sys.argv[1:]])

    # performance change relative to the clean unit of the same dataset and replication
    clean_p = perf[perf.dtype == "clean"].set_index(["did", "rep"])[["auc", "ap", "ig"]]
    p = with_clean_level0(perf, ["auc", "ap", "ig"], [])
    p = p.join(clean_p, on=["did", "rep"], rsuffix="_clean")
    for m in ("auc", "ap", "ig"):
        p[f"d_{m}"] = p[m] - p[f"{m}_clean"]
    p.to_csv(OUT / "perf_long.csv", index=False)
    summary = p.groupby(["dtype", "level"]).d_auc.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]]
    summary.to_csv(OUT / "delta_perf_by_type.csv")

    # measure change, oriented and standardised by the Study A SD (80 primary datasets)
    a = pd.read_csv(ROOT / "results" / "measures_v1.csv")
    g = pd.read_csv(ROOT / "results" / "gold_v1.csv")
    sd = a[a.did.isin(g[g.category == "primary"].did)].groupby("measure").value.std()
    clean_m = meas[meas.dtype == "clean"].set_index(["did", "rep", "measure"]).value
    m = with_clean_level0(meas, ["value"], ["measure"])
    m = m.join(clean_m.rename("value_clean"), on=["did", "rep", "measure"])
    m["dir"] = m.measure.map(DIRECTION)
    m["sd_a"] = m.measure.map(sd)
    m = m[m.sd_a > 0]
    m["dm"] = m.dir * (m.value - m.value_clean) / m.sd_a
    m.to_csv(OUT / "measures_long.csv", index=False)
    pm = m.merge(p[["did", "rep", "dtype", "level", "d_auc", "d_ap", "d_ig"]], on=["did", "rep", "dtype", "level"])

    result = {}
    # H4a sensitivity
    rows = []
    for meas_name, t in TARGETED:
        sub = pm[(pm.measure == meas_name) & (pm.dtype == t)]
        per = grouped_rho(sub, ["did", "rep"], "level", "dm")
        per_ds = per.groupby(level=0).mean() if len(per) else per
        rows.append({"measure": meas_name, "dtype": t, "n_datasets": len(per_ds), "median_rho_level_dm": per_ds.median(),
                     "reacts": bool(len(per_ds) and per_ds.median() <= -0.5)})
    h4a = pd.DataFrame(rows)
    h4a.to_csv(OUT / "h4a_sensitivity.csv", index=False)
    result["H4a"] = {"share_reacting": float(h4a.reacts.mean()), "n_pairs": len(h4a),
                     "supported": bool(h4a.reacts.mean() >= 0.8)}

    # H4b proportionality
    rows = []
    for meas_name, t in TARGETED:
        sub = pm[(pm.measure == meas_name) & (pm.dtype == t)]
        per_ds = grouped_rho(sub, "did", "dm", "d_auc")
        rows.append({"measure": meas_name, "dtype": t, "n_datasets": len(per_ds), "median_rho_dm_dauc": per_ds.median(),
                     "not_proportional": bool(len(per_ds) and per_ds.median() < 0.5)})
    h4b = pd.DataFrame(rows)
    h4b.to_csv(OUT / "h4b_proportionality.csv", index=False)
    result["H4b"] = {"share_not_proportional": float(h4b.not_proportional.mean()),
                     "supported": bool(h4b.not_proportional.mean() >= 0.5)}

    # H4c false alarms
    lvl5 = p[p.level == 5].groupby("dtype").d_auc.median()
    harmless = sorted(lvl5[lvl5 > -0.01].index)
    rows = []
    for t in ("redundant", "irrelevant"):
        for meas_name in Q:
            sub = pm[(pm.measure == meas_name) & (pm.dtype == t) & (pm.level == 5)]
            if sub.empty:
                continue
            med = float(sub.dm.median())
            rows.append({"dtype": t, "harmless": t in harmless, "median_dauc_level5": float(lvl5.get(t, np.nan)),
                         "measure": meas_name, "median_std_dm_level5": med,
                         "false_alarm": bool(t in harmless and med < -0.5)})
    h4c = pd.DataFrame(rows)
    h4c.to_csv(OUT / "h4c_false_alarms.csv", index=False)
    testable = any(t in harmless for t in ("redundant", "irrelevant"))
    result["H4c"] = {"harmless_types": harmless, "testable": testable,
                     "false_alarms": h4c[h4c.false_alarm][["dtype", "measure"]].to_dict("records"),
                     "supported": bool(testable and h4c.false_alarm.any()) if testable else None}

    # exploratory tracking over all degradations
    rows = []
    for meas_name, sub in pm.groupby("measure"):
        for perf_col in ("d_auc", "d_ap", "d_ig"):
            per_ds = grouped_rho(sub, "did", "dm", perf_col)
            rows.append({"measure": meas_name, "perf": perf_col, "set": "Q" if meas_name in Q else "T" if meas_name in T
                         else "", "n_datasets": len(per_ds), "median_tracking_rho": per_ds.median()})
    track = pd.DataFrame(rows).sort_values(["perf", "median_tracking_rho"], ascending=[True, False])
    track.to_csv(OUT / "tracking.csv", index=False)
    t_auc = track[track.perf == "d_auc"]
    result["tracking_auc"] = {"median_Q": float(t_auc[t_auc.set == "Q"].median_tracking_rho.median()),
                              "median_T": float(t_auc[t_auc.set == "T"].median_tracking_rho.median())}

    (OUT / "hypotheses_h4.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
