"""Amendment A4: common-hardware timing benchmark (exploratory).

One machine, one CPU core (the whole run is pinned with taskset by the caller),
20 primary datasets spread over size (cells = rows x features). Times every
measure group on the same data loading; DREAMER with 100 subtables in one run
(reported x10 as an extrapolation to 1,000); TabPFN on one CPU core and on the
GPU. Per-dataset limit for DREAMER and TabPFN: --limit-min (recorded as a lower bound).

  taskset -c 0 .venv/bin/python scripts/phase4c_timing.py
Writes results/raw/timing_benchmark.csv (did, group, seconds, status).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readybench import data, measures  # noqa: E402

OUT = ROOT / "results" / "raw" / "timing_benchmark.csv"
CPU_GROUPS = ["icirca", "thesis", "relevance", "landmarkers", "aidrin", "empirical", "complexity"]


def pick_datasets(n: int) -> pd.DataFrame:
    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    prim = pool[pool.used & (pool.category == "primary")].copy()
    prim["cells"] = prim.NumberOfInstances * prim.NumberOfFeatures
    prim = prim.sort_values("cells").reset_index(drop=True)
    idx = np.unique(np.round(np.linspace(0, len(prim) - 1, n)).astype(int))
    return prim.iloc[idx]


def timed_subprocess(cmd: list[str], env: dict, limit_s: float) -> tuple[float, str]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=limit_s)
        status = "ok" if proc.returncode == 0 else "error: " + (proc.stderr or proc.stdout)[-150:].replace("\n", " ")
    except subprocess.TimeoutExpired:
        status = "limit"
    return time.perf_counter() - start, status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--limit-min", type=float, default=20)
    args = ap.parse_args()
    limit = args.limit_min * 60
    done = set()
    if OUT.exists():  # errors are retried, ok and limit entries are final
        prev = pd.read_csv(OUT)
        prev = prev[~prev.status.str.startswith("error")]
        prev.to_csv(OUT, index=False)
        done = set(zip(prev.did, prev.group))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    for d in pick_datasets(args.n).itertuples():
        did, tid = int(d.did), int(d.tid)
        data.fetch(did, tid)
        rows = []
        X, y = data.load(did)
        for g in CPU_GROUPS:
            if (did, g) in done:
                continue
            start = time.perf_counter()
            try:
                fn = measures.GROUPS[g]
                fn(X, y, tid) if g == "empirical" else fn(X, y)
                status = "ok"
            except Exception as exc:
                status = f"error: {type(exc).__name__}"
            rows.append({"did": did, "group": g, "seconds": time.perf_counter() - start, "status": status})
        if (did, "dreamer_100x10") not in done:
            dreamer_out = ROOT / "scratch" / "timing_dreamer" / f"{did}_{time.time_ns()}.csv"
            sec, status = timed_subprocess(
                [str(ROOT / ".venv-dreamer/bin/python"), str(ROOT / "scripts/phase3_dreamer.py"), "--dids", str(did),
                 "--total-subtables", "100", "--runs", "1", "--timeout-min", str(args.limit_min),
                 "--out", str(dreamer_out)], env, limit + 60)
            if status == "ok" and dreamer_out.exists() and not pd.read_csv(dreamer_out).ok.astype(bool).all():
                status = "limit"  # DREAMER's own timeout exits normally; its result row says ok=False
            rows.append({"did": did, "group": "dreamer_100x10", "seconds": sec * 10, "status": status})
        for device in ("cpu", "cuda"):
            g = f"tabpfn_{device}"
            if (did, g) in done:
                continue
            sec, status = timed_subprocess(
                [str(ROOT / ".venv-gpu/bin/python"), str(ROOT / "scripts/phase3_tabpfn.py"), "--child", str(did), str(tid),
                 "--device", device], {**env, "TABPFN_ALLOW_CPU_LARGE_DATASET": "1"}, limit)
            rows.append({"did": did, "group": g, "seconds": sec, "status": status})
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"{d.name} ({int(d.NumberOfInstances)}x{int(d.NumberOfFeatures)}): "
              + ", ".join(f"{r['group']} {r['seconds']:.1f}s{'' if r['status'] == 'ok' else ' [' + r['status'][:12] + ']'}"
                          for r in rows), flush=True)


if __name__ == "__main__":
    main()
