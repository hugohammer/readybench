"""Download the TabRepo result tables used as the Study A gold standard.

Source: Hugging Face dataset Tabrepo/tabrepo, release 2023_11_14
(Salinas & Erickson, AutoML 2024). Only the metric tables are fetched (~26 MB);
the per-model prediction files are not needed for the gold standard.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench.paths import TABREPO_CACHE  # noqa: E402

RELEASE = "2023_11_14"
BASE = f"https://huggingface.co/datasets/Tabrepo/tabrepo/resolve/main/{RELEASE}"
FILES = ["configs.parquet", "baselines.parquet", "task_metadata.csv"]


def main() -> None:
    TABREPO_CACHE.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = TABREPO_CACHE / name
        if not target.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE}/{name}", target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
        print(f"{name}: {target.stat().st_size:,} bytes, sha256 {digest}")


if __name__ == "__main__":
    main()
