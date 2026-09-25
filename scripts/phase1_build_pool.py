"""Phase 1, step 2: assign every candidate a category and build the pool list.

Rules, in order:
  1. a manual entry in configs/pool_overrides.csv wins;
  2. an OpenML description containing "Binarized version" -> binarized;
  3. otherwise -> primary.

Categories used in the analysis plan: primary (main analysis); binarized and
synthetic (sensitivity analyses); duplicate, text, image (never used).
Writes configs/pool_v1.csv: the frozen pool, committed so later phases do not
depend on the live OpenML listing.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench.paths import OUTPUTS, ROOT  # noqa: E402  (OUTPUTS: candidates input)

USED = {"primary", "binarized", "synthetic"}


def main() -> None:
    cand = pd.read_csv(OUTPUTS / "phase1" / "candidates.csv")
    over = pd.read_csv(ROOT / "configs" / "pool_overrides.csv", comment="#").set_index("did")

    unknown = set(over.index) - set(cand.did)
    if unknown:
        sys.exit(f"overrides refer to dids that are not candidates: {sorted(unknown)}")

    auto_binarized = cand.description_160.fillna("").str.contains("Binarized version", case=False)
    cand["category"] = "primary"
    cand.loc[auto_binarized, "category"] = "binarized"
    cand["flag"] = ""
    cand["reason"] = ""
    cand.loc[auto_binarized, "reason"] = "OpenML description: binarized from a numeric/multi-class target"

    has_override = cand.did.isin(over.index)
    for col in ("category", "flag", "reason"):
        cand.loc[has_override, col] = cand.loc[has_override, "did"].map(over[col]).fillna("")

    cand["used"] = cand.category.isin(USED)
    cand.drop(columns=["description_160"]).to_csv(ROOT / "configs" / "pool_v1.csv", index=False)

    print(pd.crosstab(cand.category, cand.source, margins=True).to_string())
    flagged = cand[cand.used & (cand.flag != "")]
    print(f"\nflagged but used: {len(flagged)}")
    print(flagged[["did", "name", "category", "flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
