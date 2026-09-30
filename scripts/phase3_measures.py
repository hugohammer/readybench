"""Phase 3: compute the readiness measures (all groups except DREAMER) for the pool.

Work unit = (dataset, measure group), one thread each, per-unit timeout.
Results are appended as they finish (a rerun skips finished units):
  outputs/phase3/measures.csv   did, group, measure, value
  outputs/phase3/timings.csv    did, group, ok, seconds, error
  outputs/phase3/directions.csv measure, group, direction (fixed in readybench.measures)

  python scripts/phase3_measures.py --workers 50 [--groups aidrin,relevance] [--dids 37,1063]
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one core per unit; parallelism comes from the process pool

import argparse  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench import data, measures  # noqa: E402
from readybench.paths import OUTPUTS, ROOT  # noqa: E402

OUT = OUTPUTS / "phase3"


class UnitTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise UnitTimeout()


def work(did: int, tid: int, group: str, timeout_s: int) -> tuple[dict, list[dict]]:
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    start = time.perf_counter()
    try:
        X, y = data.load(did)
        fn = measures.GROUPS[group]
        values = fn(X, y, tid) if group == "empirical" else fn(X, y)
        timing = {"did": did, "group": group, "ok": True, "seconds": round(time.perf_counter() - start, 2), "error": ""}
        return timing, [{"did": did, "group": group, "measure": k, "value": v} for k, v in values.items()]
    except UnitTimeout:
        err = f"timeout after {timeout_s} s"
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=2)[-300:]}".replace("\n", " ")
    finally:
        signal.alarm(0)
    return {"did": did, "group": group, "ok": False, "seconds": round(time.perf_counter() - start, 2), "error": err}, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--groups", default=",".join(measures.GROUPS))
    parser.add_argument("--dids", default="")
    parser.add_argument("--timeout-min", type=float, default=60)
    args = parser.parse_args()

    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    pool = pool[pool.used]
    if args.dids:
        pool = pool[pool.did.isin([int(d) for d in args.dids.split(",")])]
    groups = args.groups.split(",")
    for d in pool.itertuples():
        data.fetch(int(d.did), int(d.tid))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"measure": m, "direction": s} for m, s in measures.DIRECTION.items()]).to_csv(
        OUT / "directions.csv", index=False)
    tfile, mfile = OUT / "timings.csv", OUT / "measures.csv"
    done = set()
    if tfile.exists():
        t = pd.read_csv(tfile)
        done = set(zip(t.loc[t.ok, "did"], t.loc[t.ok, "group"]))

    # biggest datasets first within each group, groups interleaved
    order = pool.assign(cells=pool.NumberOfInstances * pool.NumberOfFeatures).sort_values("cells", ascending=False)
    jobs = [(int(d.did), int(d.tid), g) for d in order.itertuples() for g in groups if (d.did, g) not in done]
    print(f"{len(jobs)} units ({len(done)} done) on {args.workers} workers", flush=True)

    start, last = time.time(), 0.0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, did, tid, g, int(args.timeout_min * 60)) for did, tid, g in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            timing, rows = fut.result()
            pd.DataFrame([timing]).to_csv(tfile, mode="a", header=not tfile.exists(), index=False)
            if rows:
                pd.DataFrame(rows).to_csv(mfile, mode="a", header=not mfile.exists(), index=False)
            if not timing["ok"]:
                print(f"FAILED did {timing['did']} {timing['group']}: {timing['error'][:160]}", flush=True)
            el = time.time() - start
            if el - last > 300 or i == len(jobs):
                last = el
                print(f"[{i}/{len(jobs)}] {el / 60:.0f} min elapsed", flush=True)


if __name__ == "__main__":
    main()
