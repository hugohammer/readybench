"""Phase 4 (Study A): agreement of readiness measures with the gold standard.

Implements docs/analysis_plan.md (frozen as git tag prereg-v1), sections 2a and 6.
Inputs: results/gold_v1.csv, results/measures_v1.csv, results/measure_runtimes_v1.csv.
Outputs in outputs/phase4/: rho_primary.csv, hypotheses.json, feasibility.csv,
sensitivity.csv, cost.csv, report.md.

  python scripts/phase4_analysis.py              # real results
  python scripts/phase4_analysis.py --synthetic  # pipeline test on fake measures (writes outputs/phase4_synthetic/)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readybench.measures import DIRECTION  # noqa: E402

SEED = 20260924
B_PRIMARY = 10_000
B_SENSITIVITY = 2_000
THRESHOLD = 0.80

Q = ["aidrin_completeness", "aidrin_duplicity", "aidrin_outliers", "aidrin_constant_frac", "aidrin_max_pairwise_corr",
     "aidrin_max_abs_skew", "aidrin_max_abs_kurtosis", "aidrin_row_completeness", "aidrin_feature_coverage",
     "aidrin_imbalance_degree", "icirca_dqi", "thesis_manual_score", "thesis_c1_completeness", "thesis_c2_balance",
     "thesis_c3_sample_size", "thesis_c4_feature_quality"]
COMPOSITES = ["thesis_manual_score", "icirca_dqi"]
T = ["rel_max_abs_r", "rel_max_single_auc", "rel_mean_top5_auc", "rel_max_norm_mi", "lm_stump_auc", "lm_nn1_auc",
     "lm_naive_bayes_auc", "lm_logreg_auc", "tabpfn_auc"]
EMPIRICAL = "emp_mean_auc"
EMPIRICAL_F1 = "emp_mean_macro_f1"


# ---------------------------------------------------------------- statistics
def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation per row of two (B, n) arrays."""
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(axis=1) / den


def oriented_rho(x: np.ndarray, gold: np.ndarray, direction: int, n_boot: int, rng) -> dict:
    ok = ~np.isnan(x)
    x, g = x[ok] * direction, gold[ok]
    n = len(x)
    if n < 10:
        return {"n": n, "rho": np.nan, "lo": np.nan, "hi": np.nan}
    rho = float(spearman_rows(x[None, :], g[None, :])[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = spearman_rows(x[idx], g[idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"n": n, "rho": rho, "lo": float(lo), "hi": float(hi)}


def feasibility_auroc(x: np.ndarray, gold: np.ndarray, direction: int, threshold: float, n_boot: int, rng) -> dict:
    ok = ~np.isnan(x)
    x, feasible = x[ok] * direction, (gold[ok] >= threshold).astype(int)
    if feasible.min() == feasible.max():
        return {"n": len(x), "n_infeasible": int((1 - feasible).sum()), "auroc": np.nan, "lo": np.nan, "hi": np.nan}
    auc = roc_auc_score(feasible, x)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if feasible[i].min() != feasible[i].max():
            boots.append(roc_auc_score(feasible[i], x[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(x), "n_infeasible": int((1 - feasible).sum()), "auroc": float(auc), "lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------- data
def load(synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gold = pd.read_csv(ROOT / "results" / "gold_v1.csv")
    if not synthetic:
        long = pd.read_csv(ROOT / "results" / "measures_v1.csv")
        runtimes = pd.read_csv(ROOT / "results" / "measure_runtimes_v1.csv")
        wide = long.pivot_table(index="did", columns="measure", values="value")
        groups = long.drop_duplicates("measure").set_index("measure").group
        return gold, wide, runtimes.assign(measure_group=runtimes.group), groups
    # Fake measures: noise, with a few measures made to track a *shuffled* gold so the test cannot leak results.
    rng = np.random.default_rng(1)
    fake_gold = rng.permutation(gold.gold_auc.to_numpy())
    wide = pd.DataFrame(index=gold.did)
    for m, d in DIRECTION.items():
        signal = 0.8 if m in ("tabpfn_auc", "emp_mean_auc", "lm_logreg_auc") else 0.0
        wide[m] = d * (signal * fake_gold + rng.normal(0, 0.05, len(gold)))
    wide.iloc[rng.integers(0, len(wide), 5), 0] = np.nan
    groups = pd.Series({m: m.split("_")[0] for m in DIRECTION})
    runtimes = pd.DataFrame({"did": np.repeat(gold.did, 3), "group": ["aidrin", "relevance", "tabpfn"] * len(gold),
                             "seconds": rng.exponential(10, 3 * len(gold))})
    return gold.assign(gold_auc=fake_gold, secondary_gold_auc=fake_gold), wide, runtimes, groups


def subset(gold: pd.DataFrame, code: str) -> tuple[pd.DataFrame, str]:
    prim = gold.category == "primary"
    flag = gold.flag.fillna("")
    sets = {
        "primary": (prim, "gold_auc"),
        "S1_no_time_ordered": (prim & (flag != "time-ordered"), "gold_auc"),
        "S2_no_anonymized": (prim & (flag != "anonymized"), "gold_auc"),
        "S3_no_check_provenance": (prim & ~flag.isin(["check", "provenance", "odd-target", "near-ceiling",
                                                      "simulated-physics"]), "gold_auc"),
        "S4_plus_binarized": (gold.category.isin(["primary", "binarized"]), "gold_auc"),
        "S5_plus_synthetic": (gold.category.isin(["primary", "synthetic"]), "gold_auc"),
        "S6_secondary_gold": (prim, "secondary_gold_auc"),
        "S8_tabrepo_only": (prim & (gold.source == "tabrepo"), "gold_auc"),
    }
    mask, col = sets[code]
    return gold[mask], col


# ---------------------------------------------------------------- analyses
def rho_table(g: pd.DataFrame, gold_col: str, wide: pd.DataFrame, groups: pd.Series, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    w = wide.reindex(g.did)
    gv = g[gold_col].to_numpy()
    rows = []
    for m in w.columns:
        r = oriented_rho(w[m].to_numpy(dtype=float), gv, DIRECTION[m], n_boot, rng)
        rows.append({"measure": m, "group": groups.get(m, ""), "set": "Q" if m in Q else "T" if m in T else "",
                     **r})
    return pd.DataFrame(rows).sort_values("rho", ascending=False)


def h_tests(g: pd.DataFrame, wide: pd.DataFrame, rho: pd.DataFrame) -> dict:
    r = rho.set_index("measure")
    out = {}
    # H1
    comp = {m: {k: r.loc[m, k] for k in ("rho", "lo", "hi", "n")} for m in COMPOSITES if m in r.index}
    q_avail = [m for m in Q if m in r.index and not np.isnan(r.loc[m, "rho"])]
    share_weak = float(np.mean([abs(r.loc[m, "rho"]) < 0.3 for m in q_avail])) if q_avail else np.nan
    comp_ok = all(v["rho"] < 0.3 and v["hi"] < 0.5 for v in comp.values()) and len(comp) == len(COMPOSITES)
    out["H1"] = {"composites": comp, "share_Q_abs_rho_below_0.3": share_weak, "n_Q": len(q_avail),
                 "supported": bool(comp_ok and share_weak >= 0.8)}
    # H2: selection 24 / test 56
    rng = np.random.default_rng(SEED)
    dids = g.did.to_numpy()
    perm = rng.permutation(len(dids))
    sel, test = dids[perm[:24]], dids[perm[24:]]
    gs = g.set_index("did").gold_auc

    def best(cands, ids):
        vals = {m: oriented_rho(wide.loc[ids, m].to_numpy(float), gs.loc[ids].to_numpy(), DIRECTION[m], 1, rng)["rho"]
                for m in cands if m in wide.columns}
        vals = {m: v for m, v in vals.items() if not np.isnan(v)}
        return max(vals, key=vals.get), vals

    bt, _ = best(T, sel)
    bq, _ = best(Q, sel)
    xt, xq, gt = (wide.loc[test, bt].to_numpy(float) * DIRECTION[bt], wide.loc[test, bq].to_numpy(float) * DIRECTION[bq],
                  gs.loc[test].to_numpy())
    ok = ~np.isnan(xt) & ~np.isnan(xq)
    xt, xq, gt = xt[ok], xq[ok], gt[ok]
    idx = np.random.default_rng(SEED + 1).integers(0, len(gt), size=(B_PRIMARY, len(gt)))
    diff = spearman_rows(xt[idx], gt[idx]) - spearman_rows(xq[idx], gt[idx])
    point = float(spearman_rows(xt[None], gt[None])[0] - spearman_rows(xq[None], gt[None])[0])
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    out["H2"] = {"best_T_on_selection": bt, "best_Q_on_selection": bq, "n_selection": 24, "n_test": int(ok.sum()),
                 "rho_diff": point, "lo": float(lo), "hi": float(hi), "supported": bool(lo > 0)}
    # H3
    e = r.loc[EMPIRICAL] if EMPIRICAL in r.index else None
    out["H3"] = {"rho": float(e.rho), "lo": float(e.lo), "hi": float(e.hi), "n": int(e.n),
                 "supported": bool(e.rho >= 0.85)} if e is not None else {"supported": None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    out = ROOT / "outputs" / ("phase4_synthetic" if args.synthetic else "phase4")
    out.mkdir(parents=True, exist_ok=True)
    gold, wide, runtimes, groups = load(args.synthetic)

    g, col = subset(gold, "primary")
    rho = rho_table(g, col, wide, groups, B_PRIMARY)
    rho.to_csv(out / "rho_primary.csv", index=False)
    hyp = h_tests(g, wide, rho)
    (out / "hypotheses.json").write_text(json.dumps(hyp, indent=2, default=float))

    rng = np.random.default_rng(SEED)
    feas = []
    for thr in (0.75, 0.80, 0.85):  # 0.80 primary; 0.75 / 0.85 = S7
        for m in wide.columns:
            r = feasibility_auroc(wide.reindex(g.did)[m].to_numpy(float), g.gold_auc.to_numpy(), DIRECTION[m], thr,
                                  1000, rng)
            feas.append({"threshold": thr, "measure": m, **r})
    pd.DataFrame(feas).to_csv(out / "feasibility.csv", index=False)

    sens = []
    for code in ["S1_no_time_ordered", "S2_no_anonymized", "S3_no_check_provenance", "S4_plus_binarized",
                 "S5_plus_synthetic", "S6_secondary_gold", "S8_tabrepo_only"]:
        gs, c = subset(gold, code)
        t = rho_table(gs, c, wide, groups, B_SENSITIVITY).assign(analysis=code, n_datasets=len(gs))
        sens.append(t)
    sens = pd.concat(sens)
    if EMPIRICAL_F1 in rho.measure.values:  # S9
        sens = pd.concat([sens, rho[rho.measure == EMPIRICAL_F1].assign(analysis="S9_empirical_macro_f1",
                                                                         n_datasets=len(g))])
    sens.to_csv(out / "sensitivity.csv", index=False)

    cost = runtimes.groupby("group").seconds.agg(["median", lambda s: s.quantile(0.9), "max"])
    cost.columns = ["median_s", "p90_s", "max_s"]
    cost.to_csv(out / "cost.csv")

    lines = ["# Phase 4 results" + (" (SYNTHETIC TEST, not real results)" if args.synthetic else ""), "",
             f"Primary pool: {len(g)} datasets; plan: git tag prereg-v1.", "", "## Hypotheses", "",
             "```json", json.dumps(hyp, indent=2, default=float), "```", "",
             "## Oriented Spearman rho with the gold standard (primary pool)", "",
             rho.round(3).to_markdown(index=False), "", "## Runtime per measure group (s)", "",
             cost.round(1).to_markdown()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(hyp, indent=2, default=float))
    print(rho.head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
