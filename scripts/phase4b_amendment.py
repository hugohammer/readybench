"""Amendment 1 analyses (plan section 10, git tag prereg-v1.1).

A1  secondary gold metrics AP, nAP, IG: rho tables, P1-P3, bottom-quartile feasibility (exploratory)
A2  thesis baselines on independent folds 3-5: P4
A3  incremental value of classical measures beyond a probe (exploratory)
A4  common-hardware timing benchmark summary (exploratory)

Inputs: results/gold_v1.csv, results/gold_secondary_v1.csv, results/measures_v1.csv,
results/measures_emp345_v1.csv, results/raw/timing_benchmark.csv.
Outputs: outputs/phase4b/ (tables + hypotheses_amendment.json).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import phase4_analysis as p4  # noqa: E402  (frozen Phase 4 functions)
from readybench.measures import DIRECTION  # noqa: E402

METRICS = ["ap", "nap", "ig"]
OUT = ROOT / "outputs" / "phase4b"
COMPOSITES_A3 = ["thesis_manual_score", "icirca_dqi", "dreamer_Total Quality", "aidrin_completeness"]
PROBES_A3 = ["tabpfn_auc", "lm_logreg_auc"]


def h2_for(g: pd.DataFrame, gold_col: str, wide: pd.DataFrame) -> dict:
    """H2 procedure of plan 2a with an arbitrary gold column (same seed and split)."""
    rng = np.random.default_rng(p4.SEED)
    dids = g.did.to_numpy()
    perm = rng.permutation(len(dids))
    sel, test = dids[perm[:24]], dids[perm[24:]]
    gs = g.set_index("did")[gold_col]

    def best(cands, ids):
        vals = {m: p4.oriented_rho(wide.loc[ids, m].to_numpy(float), gs.loc[ids].to_numpy(), DIRECTION[m], 1, rng)["rho"]
                for m in cands if m in wide.columns}
        vals = {m: v for m, v in vals.items() if not np.isnan(v)}
        return max(vals, key=vals.get)

    bt, bq = best(p4.T, sel), best(p4.Q, sel)
    xt = wide.loc[test, bt].to_numpy(float) * DIRECTION[bt]
    xq = wide.loc[test, bq].to_numpy(float) * DIRECTION[bq]
    gt = gs.loc[test].to_numpy()
    ok = ~np.isnan(xt) & ~np.isnan(xq) & ~np.isnan(gt)
    xt, xq, gt = xt[ok], xq[ok], gt[ok]
    idx = np.random.default_rng(p4.SEED + 1).integers(0, len(gt), size=(p4.B_PRIMARY, len(gt)))
    diff = p4.spearman_rows(xt[idx], gt[idx]) - p4.spearman_rows(xq[idx], gt[idx])
    point = float(p4.spearman_rows(xt[None], gt[None])[0] - p4.spearman_rows(xq[None], gt[None])[0])
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    return {"best_T": bt, "best_Q": bq, "n_test": int(ok.sum()), "rho_diff": point, "lo": float(lo), "hi": float(hi),
            "supported": bool(lo > 0)}


def h1_rule(rho: pd.DataFrame) -> dict:
    r = rho.set_index("measure")
    comp = {m: {k: float(r.loc[m, k]) for k in ("rho", "lo", "hi", "n")} for m in p4.COMPOSITES}
    q = [m for m in p4.Q if m in r.index and not np.isnan(r.loc[m, "rho"])]
    share = float(np.mean([abs(r.loc[m, "rho"]) < 0.3 for m in q]))
    ok = all(v["rho"] < 0.3 and v["hi"] < 0.5 for v in comp.values())
    return {"composites": comp, "share_Q_abs_rho_below_0.3": share, "supported": bool(ok and share >= 0.8)}


def partial_spearman(x, y, z) -> float:
    """Spearman partial correlation of x and y given z (Pearson on ranks, residualised on rank(z))."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def loo_mae(Xm: np.ndarray, y: np.ndarray) -> float:
    errs = []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        A = np.c_[np.ones(tr.sum()), Xm[tr]]
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        errs.append(abs(np.r_[1, Xm[i]] @ coef - y[i]))
    return float(np.mean(errs))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold = pd.read_csv(ROOT / "results" / "gold_v1.csv")
    sec = pd.read_csv(ROOT / "results" / "gold_secondary_v1.csv")
    gold = gold.merge(sec[["did"] + METRICS], on="did", how="left")
    long = pd.concat([pd.read_csv(ROOT / "results" / "measures_v1.csv"),
                      pd.read_csv(ROOT / "results" / "measures_emp345_v1.csv")])
    wide = long.pivot_table(index="did", columns="measure", values="value")
    groups = long.drop_duplicates("measure").set_index("measure").group
    g, _ = p4.subset(gold, "primary")
    result = {"n_primary": len(g), "n_with_secondary": int(g[METRICS].notna().all(axis=1).sum())}

    # ---- A1
    tables = []
    for m in METRICS:
        gm = g.dropna(subset=[m])
        rho = p4.rho_table(gm, m, wide, groups, p4.B_PRIMARY).assign(gold_metric=m)
        tables.append(rho)
        r = rho.set_index("measure")
        result[m] = {"H1_rule": h1_rule(rho), "H2_procedure": h2_for(gm, m, wide)}
        if m == "ap":
            result["P1"] = {k: {"rho": float(r.loc[k, "rho"]), "lo": float(r.loc[k, "lo"]), "hi": float(r.loc[k, "hi"])}
                            for k in ("aidrin_imbalance_degree", "thesis_c2_balance")}
            result["P1"]["supported"] = all(v["rho"] >= 0.3 for v in result["P1"].values())
    tables = pd.concat(tables)
    tables.to_csv(OUT / "rho_secondary_metrics.csv", index=False)
    result["P2"] = {"supported": result["nap"]["H1_rule"]["supported"] and result["ig"]["H1_rule"]["supported"]}
    result["P3"] = {"supported": all(result[m]["H2_procedure"]["supported"] for m in METRICS)}

    feas = []
    rng = np.random.default_rng(p4.SEED)
    for m in METRICS:
        gm = g.dropna(subset=[m])
        thr = gm[m].quantile(0.25)
        for meas in wide.columns:
            r = p4.feasibility_auroc(wide.reindex(gm.did)[meas].to_numpy(float), gm[m].to_numpy(), DIRECTION[meas], thr,
                                     1000, rng)
            feas.append({"gold_metric": m, "threshold": thr, "measure": meas, **r})
    pd.DataFrame(feas).to_csv(OUT / "feasibility_bottom_quartile.csv", index=False)

    # ---- A2
    rho_auc = p4.rho_table(g, "gold_auc", wide, groups, p4.B_PRIMARY).set_index("measure")
    result["P4"] = {k: {c: float(rho_auc.loc[k, c]) for c in ("rho", "lo", "hi", "n")}
                    for k in ("emp345_mean_auc", "emp_mean_auc", "emp345_mean_macro_f1") if k in rho_auc.index}
    result["P4"]["supported"] = bool(rho_auc.loc["emp345_mean_auc", "rho"] >= 0.85)

    # ---- A3
    inc = []
    gi = g.set_index("did")
    for probe in PROBES_A3:
        for comp in COMPOSITES_A3:
            d = pd.DataFrame({"gold": gi.gold_auc, "probe": wide[probe] * DIRECTION[probe],
                              "comp": wide[comp] * DIRECTION[comp]}).reindex(g.did).dropna()
            inc.append({"probe": probe, "classical": comp, "n": len(d),
                        "rho_classical": float(spearmanr(d.comp, d.gold)[0]),
                        "partial_rho_given_probe": partial_spearman(d.comp, d.gold, d.probe)})
        d = pd.DataFrame({"gold": gi.gold_auc, "probe": wide[probe], **{c: wide[c] for c in COMPOSITES_A3}}).reindex(
            g.did).dropna()
        inc.append({"probe": probe, "classical": "LOO linear regression", "n": len(d),
                    "mae_probe_only": loo_mae(d[["probe"]].to_numpy(), d.gold.to_numpy()),
                    "mae_probe_plus_classical": loo_mae(d[["probe"] + COMPOSITES_A3].to_numpy(), d.gold.to_numpy())})
    pd.DataFrame(inc).to_csv(OUT / "incremental_value.csv", index=False)

    # ---- A4
    tb = ROOT / "results" / "raw" / "timing_benchmark.csv"
    if tb.exists():
        t = pd.read_csv(tb)
        # DREAMER's own timeout exits normally: its result rows (scratch/timing_dreamer) say ok=False
        flags = {}
        for f in sorted((ROOT / "scratch" / "timing_dreamer").glob("*.csv")):
            r = pd.read_csv(f)
            if len(r):
                flags[int(r.did.iloc[0])] = bool(r.ok.iloc[0])
        is_dreamer = t.group == "dreamer_100x10"
        t.loc[is_dreamer & t.did.map(flags).eq(False), "status"] = "limit"
        t.to_csv(OUT / "timing_benchmark_corrected.csv", index=False)
        summary = t.groupby("group").agg(n=("did", "nunique"), median_s=("seconds", "median"),
                                         max_s=("seconds", "max"), n_limit=("status", lambda s: (s == "limit").sum()))
        summary.sort_values("median_s").to_csv(OUT / "timing_common_hardware.csv")

    (OUT / "hypotheses_amendment.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({k: v for k, v in result.items() if k in ("n_primary", "n_with_secondary", "P1", "P2", "P3", "P4")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
