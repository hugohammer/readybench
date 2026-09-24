"""Central locations for caches and outputs.

Everything heavy lives under the project root (never in OneDrive). Override the
root with the READYBENCH_ROOT environment variable when running on a server.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("READYBENCH_ROOT", Path(__file__).resolve().parents[2]))
CACHE = ROOT / "data_cache"
OPENML_CACHE = CACHE / "openml"
TABREPO_CACHE = CACHE / "tabrepo"
OUTPUTS = ROOT / "outputs"


def setup_openml_cache() -> None:
    """Point the openml package at the project cache instead of ~/.openml."""
    import openml

    OPENML_CACHE.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(OPENML_CACHE))
