"""Phase 3, measure group R6: TabPFN v2 probe on a local GPU.

Runs one dataset at a time (keeps memory low on a laptop) and appends to
outputs/phase3/tabpfn.csv, so an interrupted run resumes where it stopped.
Environment: .venv-gpu (torch with CUDA, tabpfn==2.0.9, openml); see
ci/requirements-gpu.txt.

  .venv-gpu/bin/python scripts/phase3_tabpfn.py [--dids 37,1063] [--device cuda]
"""

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench import data, measures  # noqa: E402
from readybench.paths import OUTPUTS, ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dids", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    pool = pool[pool.used]
    if args.dids:
        pool = pool[pool.did.isin([int(d) for d in args.dids.split(",")])]
    out = OUTPUTS / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "tabpfn.csv"
    done = set(pd.read_csv(target).query("ok").did) if target.exists() else set()

    todo = pool[~pool.did.isin(done)].sort_values("NumberOfInstances")
    print(f"{len(todo)} datasets to run ({len(done)} done)", flush=True)
    for i, d in enumerate(todo.itertuples(), 1):
        start = time.perf_counter()
        row = {"did": int(d.did), "ok": False, "tabpfn_auc": float("nan"), "error": ""}
        try:
            data.fetch(int(d.did), int(d.tid))
            X, y = data.load(int(d.did))
            row.update(measures.tabpfn_probe(X, y, device=args.device), ok=True)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=2)[-200:]}".replace("\n", " ")
        row["seconds"] = round(time.perf_counter() - start, 1)
        pd.DataFrame([row]).to_csv(target, mode="a", header=not target.exists(), index=False)
        status = f"AUC {row['tabpfn_auc']:.3f}" if row["ok"] else f"FAILED {row['error'][:120]}"
        print(f"[{i}/{len(todo)}] {d.name}: {status} ({row['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
