"""Data and numbers behind the dashboard, kept free of Streamlit so they can be tested.

The app only reads files the pipeline already wrote (outputs/, outputs/wells/
and the cleaned daily parquet); nothing here recomputes a pipeline step.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src import config, plots, scoring

WATCHLIST = config.WELLS_OUT / "well_watchlist.csv"

# Shown in the "Limitations" expander. Adapted from the team README draft to
# the survival method actually used.
LIMITATIONS = [
    "Costs are field estimates (`field_est_total`), not final job costs; the currency is assumed to be C$.",
    "The official failure log is incomplete (it starts in December 2021 and covers only some areas), "
    "so failures are rebuilt from daily downtime codes and checked against the log where it exists.",
    "Repairs and failures share no ID. They are linked by well and date, so some links will be wrong.",
    "Cox hazard ratios are associations, not proof of cause: wells given a workover may differ in ways the data does not show.",
    "Several sensor columns are sparse or look forward-filled; missing values are filled with training medians.",
    "The oil value (C$51.04/bbl) is an assumption, which is why the app exposes it as a slider.",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_outputs() -> dict[str, pd.DataFrame]:
    return plots.load_all(config.OUTPUTS)


def load_watchlist() -> pd.DataFrame:
    return pd.read_csv(WATCHLIST, parse_dates=["run_start", "as_of"])


def load_runs() -> pd.DataFrame:
    return pd.read_parquet(config.RUNS)


def load_events() -> pd.DataFrame:
    ev = pd.read_parquet(config.EVENTS)
    ev["ev_start"], ev["ev_end"] = pd.to_datetime(ev.ev_start), pd.to_datetime(ev.ev_end)
    return ev


def load_model() -> dict:
    return scoring.load_model()


def as_of(watchlist: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(watchlist.as_of.iloc[0])


# --------------------------------------------------------------------------
# Overview numbers
# --------------------------------------------------------------------------
def headline_metrics(sizing: pd.DataFrame, oil_value: float = config.OIL_VALUE_PER_BBL) -> dict:
    s = sizing.set_index("metric").value
    repairs, oil = plots.annual_cost(sizing, oil_value)
    return {
        "failures_per_well_year": float(s["failures_per_well_year"]),
        "repeat_within_30d": float(s["quick_repeat_share_calendar"]),
        "reactive_spend_share": float(s["reactive_spend_share"]),
        "cost_per_year_m": float(repairs + oil),
    }


# --------------------------------------------------------------------------
# Well IDs
# --------------------------------------------------------------------------
def make_aliases(watchlist: pd.DataFrame) -> dict[str, str]:
    """Stable "Well 001" names in watchlist rank order; unscored wells come last."""
    order = watchlist.sort_values("rank", na_position="last", kind="stable").uwi
    width = max(3, len(str(len(order))))
    return {u: f"Well {i:0{width}d}" for i, u in enumerate(order, start=1)}


def show_id(uwi, aliases: dict[str, str], hide: bool):
    if isinstance(uwi, pd.Series):
        return uwi.map(aliases) if hide else uwi
    return aliases[uwi] if hide else uwi


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------
TABLE_COLUMNS = {
    "rank": "rank", "well": "well", "area": "area", "risk_90": "risk_90",
    "elapsed_prod_days": "elapsed producing days", "fails_prior_365": "failures in past year",
}


def watchlist_table(watchlist: pd.DataFrame, aliases: dict[str, str], hide: bool,
                    areas: list[str] | None = None, min_risk: float = 0.0) -> pd.DataFrame:
    """Scored wells, filtered, with the well column already aliased if IDs are hidden."""
    w = watchlist[watchlist.risk_90.notna() & (watchlist.risk_90 >= min_risk)]
    if areas:
        w = w[w.area.isin(areas)]
    w = w.sort_values("rank").assign(well=lambda d: show_id(d.uwi, aliases, hide))
    t = w[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
    t["rank"] = t["rank"].astype(int)
    return t.set_index(w.uwi.rename("uwi"))


def well_run(watchlist: pd.DataFrame, runs: pd.DataFrame, uwi: str) -> pd.DataFrame:
    """The well's open run with every model feature, plus its elapsed producing days."""
    w = watchlist[watchlist.uwi == uwi][["uwi", "run_start", "elapsed_prod_days", "risk_90"]]
    r = w.merge(runs, left_on=["uwi", "run_start"], right_on=["uwi", "start_d"], how="inner", validate="1:1")
    if len(r) != 1:
        raise ValueError("the watchlist row has no matching run in runs.parquet")
    return r


def well_survival(bundle: dict, run_row: pd.DataFrame) -> pd.Series:
    """Survival curve long enough to read 180 days past the elapsed producing days."""
    elapsed = int(run_row.elapsed_prod_days.iloc[0])
    return scoring.survival_curve(bundle, run_row, elapsed + 181)


def outlook_risk_90(survival: pd.Series, elapsed: float, horizon: int = config.HORIZON_DAYS) -> float:
    """The 90-day value plots.well_outlook draws, computed the same way."""
    s = survival.sort_index()
    return float(1 - np.interp(elapsed + horizon, s.index, s.values) / np.interp(elapsed, s.index, s.values))


def well_daily(uwi: str, end: pd.Timestamp, days: int = 365) -> pd.DataFrame:
    """One well's last `days` of daily rows, read with a filtered query, not the whole file."""
    start = (end - pd.Timedelta(days=days)).date()
    return duckdb.execute(
        f"""select production_date, oil_produced_m3, is_downhole
            from read_parquet('{config.DAILY_CLEAN.as_posix()}')
            where uwi = ? and production_date > ? and production_date <= ?
            order by production_date""",
        [uwi, start, end.date()]).df()


def well_events(events: pd.DataFrame, uwi: str, end: pd.Timestamp, days: int = 365) -> pd.DataFrame:
    start = end - pd.Timedelta(days=days)
    return events[(events.uwi == uwi) & (events.ev_end > start)]
