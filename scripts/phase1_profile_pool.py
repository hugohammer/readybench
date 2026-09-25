"""Phase 1, step 3: download every pool dataset and profile its quality dimensions.

Runs one dataset per process (downloads are I/O bound, profiling is cheap).
Writes outputs/phase1/profile.csv, outputs/phase1/signatures.json and
outputs/phase1/failures.csv.  Options: --limit N (test on N datasets), --workers N.
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench import data, profile  # noqa: E402
from readybench.paths import OUTPUTS, ROOT  # noqa: E402


def work(did: int, tid: int) -> dict:
    start = time.perf_counter()
    try:
        facts = data.fetch(did, tid)
        X, y = data.load(did)
        prof = profile.quality_profile(X, y)
        sigs = profile.column_signatures(X)
        return {"ok": True, **facts, **prof, "seconds": round(time.perf_counter() - start, 1), "signatures": sigs}
    except Exception as exc:  # keep going; failures are reported, not fatal
        return {"ok": False, "did": did, "tid": tid, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3), "seconds": round(time.perf_counter() - start, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    pool = pd.read_csv(ROOT / "configs" / "pool_v1.csv")
    pool = pool[pool.used].sort_values("NumberOfInstances")
    if args.limit:
        pool = pool.head(args.limit)
    print(f"profiling {len(pool)} datasets with {args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, int(r.did), int(r.tid)) for r in pool.itertuples()]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            status = "ok" if res["ok"] else f"FAILED {res['error'][:80]}"
            print(f"[{i}/{len(pool)}] did {res['did']}: {status} ({res['seconds']} s)", flush=True)

    out = OUTPUTS / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r["ok"]]
    sigs = {str(r["did"]): r.pop("signatures") for r in ok}
    (out / "signatures.json").write_text(json.dumps(sigs))
    prof = pd.DataFrame([{k: v for k, v in r.items() if k != "ok"} for r in ok])
    pool[["did", "name", "source", "category", "flag"]].merge(prof, on="did").to_csv(out / "profile.csv", index=False)
    pd.DataFrame([r for r in results if not r["ok"]]).to_csv(out / "failures.csv", index=False)
    print(f"done: {len(ok)} ok, {len(results) - len(ok)} failed")


if __name__ == "__main__":
    main()
