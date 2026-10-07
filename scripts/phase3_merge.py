"""Phase 3, final step: one long table of all readiness measures.

  python scripts/phase3_merge.py --measures-run outputs/runs/9276 --dreamer results/raw/dreamer_v1.csv \
      --tabpfn outputs/phase3/tabpfn.csv

Writes results/measures_v1.csv (did, group, measure, value) and
results/measure_runtimes_v1.csv (did, group, seconds; per measure group).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readybench.measures import DIRECTION, DREAMER_FIELDS  # noqa: E402


def phase3_dir(run: Path) -> Path:
    return run / "run" / "outputs" / "phase3" if (run / "run").exists() else run / "dreamer" / "outputs" / "phase3"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--measures-run", type=Path, required=True)
    p.add_argument("--dreamer", type=Path, required=True, help="DREAMER results CSV (one row per dataset)")
    p.add_argument("--tabpfn", type=Path, required=True)
    args = p.parse_args()

    m_dir = phase3_dir(args.measures_run)
    measures = pd.read_csv(m_dir / "measures.csv")
    timings = pd.read_csv(m_dir / "timings.csv").drop_duplicates(["did", "group"], keep="last")
    timings = timings[timings.ok][["did", "group", "seconds"]]

    d = pd.read_csv(args.dreamer).drop_duplicates("did", keep="last")
    d_ok = d[d.ok.astype(bool)]
    dreamer = d_ok.melt(id_vars="did", value_vars=[f"dreamer_{f}" for f in DREAMER_FIELDS if f"dreamer_{f}" in d],
                        var_name="measure", value_name="value").assign(group="dreamer")
    d_time = d[["did", "seconds"]].assign(group="dreamer")  # timeouts kept: they are the cost of trying

    t = pd.read_csv(args.tabpfn).drop_duplicates("did", keep="last")
    tabpfn = t[t.ok.astype(bool)][["did", "tabpfn_auc"]].rename(columns={"tabpfn_auc": "value"}).assign(
        measure="tabpfn_auc", group="tabpfn")
    t_time = t[["did", "seconds"]].assign(group="tabpfn")

    allm = pd.concat([measures, dreamer, tabpfn], ignore_index=True)[["did", "group", "measure", "value"]]
    unknown = set(allm.measure) - set(DIRECTION)
    if unknown:
        sys.exit(f"measures without a fixed direction: {sorted(unknown)}")
    out = ROOT / "results"
    allm.to_csv(out / "measures_v1.csv", index=False)
    pd.concat([timings, d_time, t_time], ignore_index=True).to_csv(out / "measure_runtimes_v1.csv", index=False)
    cov = allm.groupby("group").did.nunique()
    print(f"{allm.measure.nunique()} measures, {allm.did.nunique()} datasets; datasets per group:")
    print(cov.to_string())


if __name__ == "__main__":
    main()
