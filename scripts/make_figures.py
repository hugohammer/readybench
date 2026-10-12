"""Figures for the paper (static PDF), built from the committed result tables.

  python scripts/make_figures.py [--out DIR]
Fig. 1  agreement of each measure with the gold standard (Study A)
Fig. 2  cost vs agreement per measure group (single-core timing, A4)
Fig. 3  harm of each degradation (Study B, median dAUC at level 5)
Fig. 4  reaction of measures to each degradation vs the harm it causes (Study B)

Palette: dataviz reference palette, first three categorical slots (validated
all-pairs, light), blue<->red diverging with a gray midpoint.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase4_analysis import Q, T  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
FAMILY = {"classical": ("Classical quality / readiness scores", BLUE),
          "other": ("Complexity & tool relevance", ORANGE),
          "probe": ("Target-aware probes", AQUA)}

LABEL = {
    "thesis_manual_score": "Thesis manual score (C1–C4)", "thesis_c1_completeness": "Thesis C1 completeness",
    "thesis_c2_balance": "Thesis C2 class balance", "thesis_c3_sample_size": "Thesis C3 sample size",
    "thesis_c4_feature_quality": "Thesis C4 feature quality", "icirca_dqi": "ICIRCA quality index (DQI)",
    "aidrin_completeness": "AIDRIN completeness", "aidrin_duplicity": "AIDRIN duplicity",
    "aidrin_outliers": "AIDRIN outliers", "aidrin_imbalance_degree": "AIDRIN imbalance degree",
    "aidrin_max_pairwise_corr": "AIDRIN max pairwise correlation", "dreamer_Total Quality": "DREAMER total quality",
    "aidrin_relevance_max": "AIDRIN feature relevance (max)", "cx_n3": "Complexity N3 (1-NN error)",
    "cx_f1": "Complexity F1 (Fisher ratio)", "rel_max_single_auc": "Best single-feature AUC",
    "rel_max_norm_mi": "Max mutual information (normalized)", "lm_stump_auc": "Landmarker: decision stump",
    "lm_naive_bayes_auc": "Landmarker: naive Bayes", "lm_logreg_auc": "Landmarker: logistic regression",
    "lm_nn1_auc": "Landmarker: 1-nearest neighbour", "tabpfn_auc": "TabPFN probe",
    "emp_mean_auc": "Thesis empirical score (5 baselines)", "empcv_mean_auc": "Thesis empirical score (5 baselines)",
}
FIG1 = ["icirca_dqi", "thesis_manual_score", "thesis_c1_completeness", "thesis_c2_balance", "thesis_c3_sample_size",
        "thesis_c4_feature_quality", "aidrin_completeness", "aidrin_duplicity", "aidrin_outliers",
        "aidrin_imbalance_degree", "aidrin_max_pairwise_corr", "dreamer_Total Quality",
        "aidrin_relevance_max", "cx_f1", "cx_n3",
        "rel_max_single_auc", "rel_max_norm_mi", "lm_stump_auc", "lm_naive_bayes_auc", "lm_logreg_auc", "lm_nn1_auc",
        "emp_mean_auc", "tabpfn_auc"]


def family(m: str) -> str:
    if m in Q or m == "dreamer_Total Quality":
        return "classical"
    if m in T or m.startswith(("emp_", "empcv_")):
        return "probe"
    return "other"


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig1(out: Path) -> None:
    r = pd.read_csv(ROOT / "results" / "phase4" / "rho_primary.csv").set_index("measure").loc[FIG1].reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 6.5))
    y = np.arange(len(r))[::-1]
    for fam, (name, col) in FAMILY.items():
        sel = r.measure.map(family) == fam
        ax.hlines(y[sel], r.lo[sel], r.hi[sel], color=col, linewidth=2, alpha=0.55)
        ax.plot(r.rho[sel], y[sel], "o", color=col, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=name)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.axvspan(-0.3, 0.3, color="#f0efec", zorder=0)
    ax.set_yticks(y, [LABEL[m] for m in r.measure], fontsize=8, color=INK)
    ax.set_xlim(-0.45, 1.0)
    ax.set_xlabel("Spearman ρ with gold-standard AUC (oriented; 80 datasets, 95% CI)", fontsize=8.5, color=INK)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.32, -0.085), ncol=3, fontsize=7, frameon=False,
              handletextpad=0.2, columnspacing=1.0)
    ax.set_ylim(-0.8, len(r) - 0.4)
    ax.text(0, -0.65, "weak: |ρ| < 0.3", ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig1_agreement.pdf")
    plt.close(fig)


def fig2(out: Path) -> None:
    rho = pd.read_csv(ROOT / "results" / "phase4" / "rho_primary.csv").set_index("measure").rho
    t = pd.read_csv(ROOT / "results" / "phase4b" / "timing_common_hardware.csv").set_index("group").median_s
    groups = [  # (timing group, representative measure, label)
        ("icirca", "icirca_dqi", "ICIRCA DQI"), ("thesis", "thesis_manual_score", "Thesis manual score"),
        ("dreamer_100x10", "dreamer_Total Quality", "DREAMER (extrapolated)"),
        ("aidrin", "aidrin_relevance_max", "AIDRIN (best metric)"), ("complexity", "cx_n3", "Complexity (best: N3)"),
        ("relevance", "rel_max_norm_mi", "Relevance (best: MI)"), ("landmarkers", "lm_nn1_auc", "Landmarkers (best: 1-NN)"),
        ("empirical", "emp_mean_auc", "Thesis empirical score"), ("tabpfn_cpu", "tabpfn_auc", "TabPFN (1 CPU core)"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for g, m, lab in groups:
        fam = family(m)
        x = max(t[g], 0.01)
        ax.plot(x, rho[m], "o", color=FAMILY[fam][1], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        off = {"TabPFN (1 CPU core)": (7, 2), "Thesis empirical score": (7, -9),
               "Landmarkers (best: 1-NN)": (-6, 6), "Relevance (best: MI)": (-6, -11)}.get(lab, (7, -3))
        ax.annotate(lab, (x, rho[m]), xytext=off, textcoords="offset points", fontsize=7.5, color=INK,
                    ha="right" if off[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(0.006, 30000)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("Median seconds per dataset on one CPU core (log scale)", fontsize=8.5, color=INK)
    ax.set_ylabel("Agreement with gold standard (ρ)", fontsize=8.5, color=INK)
    style(ax, "both")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=n) for n, c in FAMILY.values()]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_cost.pdf")
    plt.close(fig)


DTYPE_LABEL = {"label_noise": "Label noise", "fewer_rows": "Fewer rows", "minority_removal": "Fewer positives",
               "label_noise_minority": "Minority label noise", "mnar": "Missing (MNAR)", "redundant": "Redundant features",
               "mar": "Missing (MAR)", "mcar": "Missing (MCAR)", "irrelevant": "Irrelevant features",
               "outliers": "Outliers", "duplicates": "Duplicates"}


def fig3(out: Path) -> list[str]:
    d = pd.read_csv(ROOT / "results" / "phase5" / "delta_perf_by_type.csv")
    d = d[d.level == 5].sort_values("50%")
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["50%"], height=0.6, color=BLUE)
    ax.hlines(y, d["25%"], d["75%"], color=INK2, linewidth=1.2)
    ax.set_yticks(y, [DTYPE_LABEL[t] for t in d.dtype], fontsize=8, color=INK)
    for yy, v in zip(y, d["50%"]):
        ax.text(0.004, yy, f"{v:+.3f}", va="center", ha="left", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlim(-0.24, 0.035)
    ax.set_xlabel("Change in test AUC at the most severe level (median and IQR over 24 datasets × 5 replications)",
                  fontsize=8, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig3_harm.pdf")
    plt.close(fig)
    return list(d.dtype)


def fig4(out: Path, order: list[str]) -> None:
    m = pd.read_csv(ROOT / "outputs" / "phase5_analysis" / "measures_long.csv")
    p = pd.read_csv(ROOT / "outputs" / "phase5_analysis" / "perf_long.csv")
    rows = ["aidrin_completeness", "aidrin_duplicity", "aidrin_outliers", "aidrin_imbalance_degree",
            "thesis_c3_sample_size", "thesis_manual_score", "icirca_dqi", "rel_max_norm_mi", "lm_nn1_auc", "tabpfn_auc"]
    lab = {**LABEL, "thesis_c3_sample_size": "Thesis C3 sample size"}
    h = m[(m.level == 5) & m.measure.isin(rows)].groupby(["measure", "dtype"]).dm.median().unstack().loc[rows, order]
    harm = p[p.level == 5].groupby("dtype").d_auc.median()[order]

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.8, 5.6), gridspec_kw={"height_ratios": [1, 3.2]}, sharex=True)
    x = np.arange(len(order))
    a0.bar(x, harm.values, width=0.6, color=BLUE)
    a0.set_ylabel("ΔAUC", fontsize=8, color=INK)
    a0.axhline(0, color=INK2, linewidth=0.8)
    style(a0, "y")
    a0.set_title("Harm caused (top) and how much each measure reacts (bottom), most severe level", fontsize=8.5,
                 color=INK, loc="left")
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", BLUE])
    lim = 4.5
    im = a1.imshow(h.values.clip(-lim, lim), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            if np.isfinite(v) and abs(v) >= 0.2:
                a1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color=SURFACE if abs(v) > 2.5 else INK)
    a1.set_yticks(range(len(rows)), [lab[r] for r in rows], fontsize=7.5, color=INK)
    a1.set_xticks(x, [DTYPE_LABEL[t] for t in order], rotation=40, ha="right", fontsize=7.5, color=INK)
    a1.axhline(6.5, color=SURFACE, linewidth=3)
    a1.tick_params(length=0)
    for s in a1.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=a1, fraction=0.03, pad=0.02)
    cb.set_label("Change in measure (Study A SD units; red = looks less ready)", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.savefig(out / "fig4_reactions.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig1(args.out)
    fig2(args.out)
    order = fig3(args.out)
    fig4(args.out, order)
    print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()
