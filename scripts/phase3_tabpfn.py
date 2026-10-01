"""Phase 3, measure group R6: TabPFN v2 probe on a local GPU.

Each dataset runs in a fresh subprocess (all GPU memory is released in
between) with a per-dataset timeout; results are appended to
outputs/phase3/tabpfn.csv, so an interrupted run resumes where it stopped.
Environment: .venv-gpu (torch with CUDA, tabpfn==2.0.9, openml); see
ci/requirements-gpu.txt. A first version without subprocesses stalled when
the 8 GB GPU memory filled up (after 53 datasets).

  .venv-gpu/bin/python scripts/phase3_tabpfn.py [--dids 37,1063] [--timeout-min 20]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readybench.paths import OUTPUTS  # noqa: E402


def child(did: int, tid: int, device: str) -> None:
    """Run one dataset and print one JSON line (used by the parent process)."""
    warnings.filterwarnings("ignore")
    from readybench import data, measures

    data.fetch(did, tid)
    X, y = data.load(did)
    print("RESULT " + json.dumps(measures.tabpfn_probe(X, y, device=device)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dids", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-min", type=float, default=20)
    parser.add_argument("--child", type=int, nargs=2, metavar=("DID", "TID"))
    args = parser.parse_args()
    if args.child:
        child(*args.child, args.device)
        return

    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    pool = pool[pool.used]
    if args.dids:
        pool = pool[pool.did.isin([int(d) for d in args.dids.split(",")])]
    out = OUTPUTS / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "tabpfn.csv"
    done = set(pd.read_csv(target).did) if target.exists() else set()  # ok, failed and timed-out are all final

    todo = pool[~pool.did.isin(done)].sort_values("NumberOfInstances")
    print(f"{len(todo)} datasets to run ({len(done)} done)", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for i, d in enumerate(todo.itertuples(), 1):
        start = time.perf_counter()
        row = {"did": int(d.did), "ok": False, "tabpfn_auc": float("nan"), "error": ""}
        try:
            proc = subprocess.run([sys.executable, __file__, "--child", str(d.did), str(d.tid), "--device", args.device],
                                  capture_output=True, text=True, timeout=args.timeout_min * 60, env=env)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line:
                row.update(json.loads(line[7:]), ok=True)
            else:
                row["error"] = (proc.stderr or proc.stdout)[-300:].replace("\n", " | ")
        except subprocess.TimeoutExpired:
            row["error"] = f"timeout after {args.timeout_min} min"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=1)[-150:]}".replace("\n", " ")
        row["seconds"] = round(time.perf_counter() - start, 1)
        pd.DataFrame([row]).to_csv(target, mode="a", header=not target.exists(), index=False)
        status = f"AUC {row['tabpfn_auc']:.3f}" if row["ok"] else f"FAILED {row['error'][:120]}"
        print(f"[{i}/{len(todo)}] {d.name}: {status} ({row['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
