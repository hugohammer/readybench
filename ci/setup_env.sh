#!/usr/bin/env bash
# Prepare a fresh python:3.12-slim container for a ReadyBench job.
set -euo pipefail

mkdir -p logs outputs data_cache

# LightGBM's wheel needs the OpenMP runtime, which the slim image lacks.
apt-get update -qq
apt-get install -y -qq --no-install-recommends libgomp1 > /dev/null

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.lock
python -m pip freeze > logs/pip-freeze.txt

python - <<'EOF'
import os, platform
print(f"python {platform.python_version()} on {platform.node()}, cpus visible: {os.cpu_count()}")
import lightgbm, openml, pymfe, aidrin, sklearn  # fail early if anything is missing
print("all imports resolve")
EOF

# Gold-standard tables used by several phases (skipped when already cached)
python scripts/fetch_tabrepo_tables.py
