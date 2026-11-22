"""Shared settings and paths for every pipeline step.

Values come from `.env` in the repo root when present, otherwise from the
defaults below. Scripts must import settings from here instead of
hard-coding thresholds, dates or prices.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    # Small parser so the project does not need python-dotenv. Real
    # environment variables win over the file, so a one-off override such as
    # EVENT_GAP_DAYS=7 on the command line still works.
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(REPO_ROOT / ".env")


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Paths
DATA_DIR = Path(_get("DATA_DIR", str(REPO_ROOT / "data")))
OUTPUTS = REPO_ROOT / "outputs"
WELLS_OUT = OUTPUTS / "wells"

RAW_FILES = {
    "daily": DATA_DIR / "daily_production.parquet",
    "jobs": DATA_DIR / "well_jobs.parquet",
    "fails": DATA_DIR / "production_failures.parquet",
}

DAILY_CLEAN = WELLS_OUT / "daily_clean.parquet"
EVENTS = WELLS_OUT / "events.parquet"

# Cleaning thresholds
PLACEHOLDER_MIN = float(_get("PLACEHOLDER_MIN", "99999"))
MAX_RPM = float(_get("MAX_RPM", "1000"))
MAX_MOTOR_CURRENT_A = float(_get("MAX_MOTOR_CURRENT_A", "1000"))
MAX_PUMP_EFFICIENCY = float(_get("MAX_PUMP_EFFICIENCY", "1.5"))

# Scope: heavy-oil producers are the wells that run progressing cavity pumps.
SCOPE_FLUID = _get("SCOPE_FLUID", "Heavy Oil")
SCOPE_FLOW = _get("SCOPE_FLOW", "production")

# Events
EVENT_GAP_DAYS = int(_get("EVENT_GAP_DAYS", "4"))
EVENT_GAP_DAYS_SENSITIVITY = int(_get("EVENT_GAP_DAYS_SENSITIVITY", "7"))
MATCH_WINDOW_DAYS = int(_get("MATCH_WINDOW_DAYS", "7"))
VALIDATION_START = _get("VALIDATION_START", "2025-07-01")

# Sizing
QUICK_REPEAT_DAYS = int(_get("QUICK_REPEAT_DAYS", "30"))
REACTIVE_LOOKBACK_DAYS = int(_get("REACTIVE_LOOKBACK_DAYS", "14"))
REPAIR_JOB_CATEGORIES = tuple(
    s.strip() for s in _get("REPAIR_JOB_CATEGORIES", "Rod Truck,Workover").split(",")
)
BBL_PER_M3 = float(_get("BBL_PER_M3", "6.29"))
PARETO_CUTS = tuple(
    int(s) for s in _get("PARETO_CUTS", "5,10,20,30,40,50,100").split(",")
)
CURRENCY_LABEL = _get("CURRENCY_LABEL", "est., CAD assumed")
CHECK_TOLERANCE = float(_get("CHECK_TOLERANCE", "0.01"))


def rel(path: Path) -> str:
    """Path relative to the repo root, so printed output does not expose local folders."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def ensure_dirs() -> None:
    WELLS_OUT.mkdir(parents=True, exist_ok=True)


def check_raw_files() -> None:
    missing = [str(p) for p in RAW_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Raw data not found: " + ", ".join(missing)
            + ". Put the three parquet files in data/ or set DATA_DIR in .env."
        )


def connect():
    """DuckDB connection with raw views `daily`, `jobs`, `fails`."""
    import duckdb

    check_raw_files()
    con = duckdb.connect()
    for name, path in RAW_FILES.items():
        con.execute(
            f"create or replace view {name} as select * from read_parquet('{path.as_posix()}')"
        )
    return con
