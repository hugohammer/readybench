"""Study B, TabPFN probe on degraded training data (plan B4: replications 0-2, local GPU).

The degraded data are regenerated exactly (readybench.degrade is deterministic and
the base split mirrors phase5_studyB). One subprocess per base dataset (GPU memory
is released in between); results are appended to outputs/phase5/tabpfn.csv and
finished units are skipped on a rerun.

  .venv-gpu/bin/python scripts/phase5_tabpfn.py [--dids 1068] [--reps 0,1,2]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "phase5" / "tabpfn.csv"
COLS = ["did", "rep", "dtype", "level", "tabpfn_auc", "seconds", "error"]


def child(did: int, tid: int, reps: list[int], done: set) -> None:
    warnings.filterwarnings("ignore")
    import phase5_studyB as b
    from readybench import degrade, measures

    for rep in reps:
        Xtr, ytr, _, _ = b.base_split(did, tid, rep)
        units = [("clean", 0)] + [(t, lv) for t in degrade.TYPES for lv in range(1, 6)]
        for dtype, level in units:
            if (did, rep, dtype, level) in done:
                continue
            start = time.perf_counter()
            row = {"did": did, "rep": rep, "dtype": dtype, "level": level, "tabpfn_auc": None, "error": ""}
            try:
                Xd, yd = (Xtr, ytr) if dtype == "clean" else degrade.apply(Xtr, ytr, dtype, level, rep)
                row["tabpfn_auc"] = measures.tabpfn_probe(Xd, pd.Series(yd), device="cuda")["tabpfn_auc"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            row["seconds"] = round(time.perf_counter() - start, 2)
            print("ROW " + json.dumps(row), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dids", default="")
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--child", type=int, nargs=2)
    args = ap.parse_args()
    reps = [int(r) for r in args.reps.split(",")]
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        prev = prev[prev.error.isna() | (prev.error == "")]
        done = set(zip(prev.did, prev.rep, prev.dtype, prev.level))
    if args.child:
        child(args.child[0], args.child[1], reps, done)
        return

    import phase5_studyB as b
    base = b.select_base()
    if args.dids:
        base = base[base.did.isin([int(d) for d in args.dids.split(",")])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for d in base.itertuples():
        start = time.time()
        proc = subprocess.Popen([sys.executable, __file__, "--child", str(d.did), str(d.tid), "--reps", args.reps],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        n = 0
        for line in proc.stdout:
            if line.startswith("ROW "):
                pd.DataFrame([json.loads(line[4:])]).reindex(columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists(),
                                                                                  index=False)
                n += 1
        proc.wait()
        print(f"{d.name}: {n} units in {(time.time() - start) / 60:.1f} min (exit {proc.returncode})", flush=True)


if __name__ == "__main__":
    main()
