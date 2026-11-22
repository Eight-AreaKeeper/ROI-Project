import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# Tests import `src` the same way the app and pipeline do, from the repo root.
sys.path.insert(0, str(REPO))

from src import config  # noqa: E402

# The app reads well-level files that are gitignored, so a fresh clone without
# a pipeline run skips these tests instead of failing.
needs_outputs = pytest.mark.skipif(
    not (config.MODEL.exists() and (config.WELLS_OUT / "well_watchlist.csv").exists()),
    reason="run `python run_all.py` first")
