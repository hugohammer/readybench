"""Phase 1, step 4: duplicate check and quality-dimension spread of the pool.

Reads the profiling results (default: outputs/phase1/, or --run-dir pointing at
a fetched CI run) and writes outputs/phase1/report.md plus
outputs/phase1/near_duplicates.csv.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench.paths import OUTPUTS  # noqa: E402

DIMENSIONS = [
    ("missing_cell_frac", "Missing cells (fraction)"),
    ("missing_row_frac", "Rows with any missing value"),
    ("duplicate_row_frac", "Duplicate rows (fraction)"),
    ("imbalance_ratio", "Imbalance ratio (majority/minority)"),
    ("n_rows", "Rows"),
    ("n_features", "Features"),
    ("frac_categorical", "Categorical features (fraction)"),
    ("frac_pairs_corr_gt_0.8", "Feature pairs with |r| > 0.8"),
    ("constant_features", "Constant features"),
    ("high_cardinality_categoricals", "Categoricals with > 30 levels"),
    ("mean_abs_skew", "Mean |skewness|"),
    ("outlier_cell_frac", "Outlier cells (1.5 IQR rule)"),
]
JACCARD_FLAG = 0.5


def near_duplicates(profile: pd.DataFrame, signatures: dict) -> pd.DataFrame:
    sets = {int(k): set(v) for k, v in signatures.items()}
    names = profile.set_index("did").name
    rows = []
    for a, b in itertools.combinations(sorted(sets), 2):
        if not sets[a] or not sets[b]:
            continue
        shared = len(sets[a] & sets[b])
        jac = shared / len(sets[a] | sets[b])
        if jac >= JACCARD_FLAG:
            rows.append({"did_a": a, "name_a": names.get(a), "did_b": b, "name_b": names.get(b),
                         "jaccard": round(jac, 3), "shared_columns": shared})
    return pd.DataFrame(rows, columns=["did_a", "name_a", "did_b", "name_b", "jaccard", "shared_columns"])


def spread_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in DIMENSIONS:
        if col not in df:
            continue
        s = df[col].dropna()
        rows.append({"dimension": label, "median": s.median(), "p10": s.quantile(0.1), "p90": s.quantile(0.9),
                     "max": s.max(), "share_nonzero": (s > 0).mean() if col not in ("n_rows", "n_features",
                                                                                      "imbalance_ratio") else None})
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=OUTPUTS / "phase1")
    args = parser.parse_args()

    profile = pd.read_csv(args.run_dir / "profile.csv")
    signatures = json.loads((args.run_dir / "signatures.json").read_text())
    failures = pd.read_csv(args.run_dir / "failures.csv") if (args.run_dir / "failures.csv").stat().st_size > 1 else pd.DataFrame()

    dups = near_duplicates(profile, signatures)
    out = OUTPUTS / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    dups.to_csv(out / "near_duplicates.csv", index=False)

    primary = profile[profile.category == "primary"]
    parts = [
        "# Phase 1 report: dataset pool\n",
        f"Profiled {len(profile)} datasets ({len(failures)} failed).\n",
        "## Pool by category and source\n",
        pd.crosstab(profile.category, profile.source, margins=True).to_markdown(),
        "\n## Possible duplicate datasets (column-fingerprint Jaccard >= 0.5)\n",
        fmt(dups) if len(dups) else "None found.",
        "\n## Spread of quality dimensions: primary datasets, TabRepo only\n",
        fmt(spread_table(primary[primary.source == "tabrepo"])),
        "\n## Spread of quality dimensions: primary datasets, TabRepo + extension\n",
        fmt(spread_table(primary)),
    ]
    if len(failures):
        parts += ["\n## Failures\n", fmt(failures[["did", "tid", "error"]])]
    (out / "report.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
