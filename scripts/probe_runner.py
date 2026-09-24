"""Describe the machine a job runs on and time a few representative workloads.

Used to choose runners and to estimate the run time of Study A/B. Writes
outputs/probe/<label>.json. Run locally with:  python scripts/probe_runner.py --label laptop
"""

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench.paths import OUTPUTS, ROOT, setup_openml_cache  # noqa: E402


def read_first(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def system_info() -> dict:
    meminfo = read_first("/proc/meminfo") or ""
    mem_total_kb = next((int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemTotal")), None)
    disk = shutil.disk_usage(ROOT)
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in (read_first("/proc/cpuinfo") or "").splitlines()
                           if line.startswith("model name")), None),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "cgroup_cpu_max": read_first("/sys/fs/cgroup/cpu.max"),  # "quota period" or "max period"
        "mem_total_gb": round(mem_total_kb / 1024**2, 1) if mem_total_kb else None,
        "cgroup_mem_max": read_first("/sys/fs/cgroup/memory.max"),
        "disk_free_gb": round(disk.free / 1024**3, 1),
    }


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return round(time.perf_counter() - start, 2), result


def benchmarks() -> dict:
    import numpy as np
    import openml
    from lightgbm import LGBMClassifier
    from pymfe.mfe import MFE
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    setup_openml_cache()
    out = {}

    # 1. OpenML download: credit-g (task 31), one of the thesis datasets
    t, task = timed(lambda: openml.tasks.get_task(31, download_splits=True))
    t2, (X, y, _, _) = timed(lambda: task.get_dataset().get_data(target=task.target_name))
    out["openml_credit_g_download_s"] = t + t2
    X = X.apply(lambda c: c.cat.codes if str(c.dtype) == "category" else c)
    y = (y == y.cat.categories[0]).astype(int)

    # 2. LightGBM 5-fold CV on credit-g (small data, dominated by overhead)
    model = LGBMClassifier(n_estimators=200, verbose=-1)
    out["lgbm_credit_g_cv5_s"], auc = timed(lambda: cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean())
    out["lgbm_credit_g_auc"] = round(float(auc), 4)

    # 3. LightGBM on a 200k x 50 synthetic dataset (throughput, uses all cores)
    Xs, ys = make_classification(n_samples=200_000, n_features=50, n_informative=20, random_state=0)
    out["lgbm_200k_fit_s"], _ = timed(lambda: LGBMClassifier(n_estimators=300, verbose=-1).fit(Xs, ys))

    # 4. PyMFE complexity measures on credit-g (the slowest Study A measure group)
    mfe = MFE(groups=["complexity"], random_state=0)
    out["pymfe_complexity_credit_g_s"], _ = timed(lambda: (mfe.fit(X.to_numpy(float), y.to_numpy()), mfe.extract()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=platform.node())
    args = parser.parse_args()

    report = {"label": args.label, **system_info(), **benchmarks()}
    target = OUTPUTS / "probe" / f"{args.label}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
