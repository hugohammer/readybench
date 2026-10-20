"""Exploratory: peak memory (RAM) per measure group on common hardware.

Same 19 datasets and single-core setup as the timing benchmark (phase4c_timing.py).
Every (dataset, group) runs in a fresh process, so the peak is not inflated by
earlier groups. Peak = max of (a) the child's own ru_maxrss from wait4 and
(b) the summed PSS (proportional set size: shared pages split between processes) of
the child and all its descendants, polled every 250 ms when there is more than one process
(reading PSS is slow and shares the single core with the measure). (b) is needed for DREAMER, whose
Pool() forks one worker per CPU core even for a single run; summing RSS would count
the shared memory once per worker. The peak includes the
Python interpreter, the libraries and the loaded dataset, i.e. what a user needs
to run the measure; the group "load" (load the dataset, compute nothing) is the
reference for that fixed part.

DREAMER runs 20 subtables in one run: its memory per subtable does not
accumulate, so this measures the peak of a single-core DREAMER run (the timing
benchmark used 100). A watchdog kills any process tree above --cap-gb
(status "cap").

  taskset -c 0 .venv/bin/python scripts/phase4d_memory.py
Writes results/raw/memory_benchmark.csv (resumable).
"""

import argparse
import os
import subprocess
import tempfile
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "raw" / "memory_benchmark.csv"
CPU_GROUPS = ["load", "icirca", "thesis", "relevance", "landmarkers", "aidrin", "empirical", "complexity"]


def child(did: int, tid: int, group: str) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from readybench import data, measures

    X, y = data.load(did)
    if group != "load":
        fn = measures.GROUPS[group]
        fn(X, y, tid) if group == "empirical" else fn(X, y)


def tree_rss(pid: int) -> int:
    import psutil

    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        return 0
    if len(procs) == 1:  # a single process: its exact peak comes from wait4, and reading smaps would slow it down
        return 0
    total = 0
    for q in procs:  # PSS: shared pages split between processes (DREAMER forks one idle worker per core)
        try:
            total += q.memory_full_info().pss
        except psutil.Error:
            pass
    return total


def measure(cmd: list[str], env: dict, limit_s: float, cap_bytes: float) -> dict:
    import psutil

    start = time.perf_counter()
    log = tempfile.TemporaryFile(mode="w+")  # a pipe could fill up and block a chatty child
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    polled, status, ru, wstatus = 0, "ok", None, 0
    while True:
        pid, wstatus, ru = os.wait4(proc.pid, os.WNOHANG)  # reap ourselves: Popen.poll would lose the rusage
        if pid:
            break
        polled = max(polled, tree_rss(proc.pid))
        if status == "ok" and (polled > cap_bytes or time.perf_counter() - start > limit_s):
            status = "cap" if polled > cap_bytes else "limit"
            for q in psutil.Process(proc.pid).children(recursive=True) + [psutil.Process(proc.pid)]:
                try:
                    q.kill()
                except psutil.Error:
                    pass
        time.sleep(0.25)
    proc.returncode = os.waitstatus_to_exitcode(wstatus)
    log.seek(0)
    err = log.read()
    log.close()
    if status == "ok" and proc.returncode != 0:
        status = "error: " + err[-150:].replace("\n", " ")
    own = ru.ru_maxrss * 1024  # kB on Linux
    return {"peak_mb": max(own, polled) / 2**20, "own_maxrss_mb": own / 2**20, "polled_pss_mb": polled / 2**20,
            "seconds": time.perf_counter() - start, "status": status}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=3, metavar=("DID", "TID", "GROUP"))
    ap.add_argument("--limit-min", type=float, default=20)
    ap.add_argument("--cap-gb", type=float, default=12)
    args = ap.parse_args()
    if args.child:
        child(int(args.child[0]), int(args.child[1]), args.child[2])
        return

    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))
    from phase4c_timing import pick_datasets

    timed = set(pd.read_csv(ROOT / "results" / "raw" / "timing_benchmark.csv").did)  # the 19 datasets timed
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev.did, prev.group))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cap, limit = args.cap_gb * 2**30, args.limit_min * 60
    for d in pick_datasets(20).itertuples():
        did, tid = int(d.did), int(d.tid)
        if did not in timed:
            continue
        jobs = [(g, [sys.executable, __file__, "--child", str(did), str(tid), g], env) for g in CPU_GROUPS]
        jobs.append(("tabpfn_cpu", [str(ROOT / ".venv-gpu/bin/python"), str(ROOT / "scripts/phase3_tabpfn.py"),
                                    "--child", str(did), str(tid), "--device", "cpu"],
                     {**env, "TABPFN_ALLOW_CPU_LARGE_DATASET": "1"}))
        jobs.append(("dreamer", [str(ROOT / ".venv-dreamer/bin/python"), str(ROOT / "scripts/phase3_dreamer.py"),
                                 "--dids", str(did), "--total-subtables", "20", "--runs", "1",
                                 "--timeout-min", str(args.limit_min),
                                 "--out", str(ROOT / "scratch" / "memory_dreamer" / f"{did}_{time.time_ns()}.csv")], env))
        for g, cmd, e in jobs:
            if (did, g) in done:
                continue
            row = {"did": did, "group": g, **measure(cmd, e, limit + 60, cap)}
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"{did} {g}: {row['peak_mb']:.0f} MB, {row['seconds']:.1f} s, {row['status'][:40]}", flush=True)


if __name__ == "__main__":
    main()
