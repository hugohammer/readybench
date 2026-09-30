#!/usr/bin/env bash
# Environment for DREAMER with the dependency versions it was written for.
set -euo pipefail
mkdir -p logs outputs data_cache
python -m pip install -q --upgrade pip
python -m pip install -q -r ci/requirements-dreamer.txt
python -m pip freeze > logs/pip-freeze-dreamer.txt
python -c "import pandas, numpy, sklearn, numba; print('dreamer env:', pandas.__version__, numpy.__version__, sklearn.__version__, numba.__version__)"
