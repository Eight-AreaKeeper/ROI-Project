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


def load_replay() -> pd.DataFrame:
    return pd.read_parquet(config.WELLS_OUT / "replay.parquet")


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
def make_aliases(watchlist: pd.DataFrame, other_uwis=()) -> dict[str, str]:
    """Stable "Well 001" names in watchlist rank order; unscored wells come last.

    Wells that appear elsewhere (such as the replay) but not on the current
    watchlist are numbered after it, so watchlist names never shift.
    """
    order = watchlist.sort_values("rank", na_position="last", kind="stable").uwi
    extra = sorted(set(other_uwis) - set(order))
    order = list(order) + extra
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
    "expected_cost_90": "expected cost, 90 days", "cost_per_failure": "cost per failure",
    "main_driver": "main driver", "suggested_action": "suggested action",
    "workover_payback_yrs": "workover payback, yrs",
    "elapsed_prod_days": "elapsed producing days", "fails_prior_365": "failures in past year",
}
SORTS = {"expected cost": "expected_cost_90", "risk": "risk_90"}


def watchlist_table(watchlist: pd.DataFrame, aliases: dict[str, str], hide: bool,
                    areas: list[str] | None = None, min_risk: float = 0.0,
                    sort: str = "expected cost") -> pd.DataFrame:
    """Scored wells, filtered and sorted, with the well column aliased if IDs are hidden.

    `rank` stays the risk rank from the watchlist whatever the sort.
    """
    w = watchlist[watchlist.risk_90.notna() & (watchlist.risk_90 >= min_risk)]
    if areas:
        w = w[w.area.isin(areas)]
    w = (w.sort_values([SORTS[sort], "rank"], ascending=[False, True], kind="stable")
          .assign(well=lambda d: show_id(d.uwi, aliases, hide)))
    t = w[[c for c in TABLE_COLUMNS if c in w.columns]].rename(columns=TABLE_COLUMNS)
    t["rank"] = t["rank"].astype(int)
    return t.set_index(w.uwi.rename("uwi"))


# Continuous features that can be a well's main driver, worded for the
# direction that raises risk. Rod torque and previous downtime are left out:
# neither is significant in the Cox model, so neither is quoted as a driver.
DRIVER_LABELS = {
    "fails_prior_365": "frequent recent failures",
    "rpm_pre30": "high pump speed",
    "wcut_pre30": "high water cut",
    "oil_pre30": "high oil rate",
    "prev_gap_days": "short previous run",
    "disp": "large pump",
    "util_pre30": "high pump utilisation",
}
NO_DRIVER = "no single driver"
PAYBACK_ACTION_YEARS = 1.5
HIGH_RISK = 0.7


def open_run_features(watchlist: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Every watchlist well's open run with all model features, in watchlist order."""
    w = watchlist[["uwi", "run_start"]]
    r = w.merge(runs, left_on=["uwi", "run_start"], right_on=["uwi", "start_d"], how="left", validate="1:1")
    if r.start_d.isna().any():
        raise ValueError("some watchlist rows have no matching run in runs.parquet")
    return r


def driver_contributions(bundle: dict, run_rows: pd.DataFrame) -> pd.DataFrame:
    """Coefficient x standardised value per driver feature: each one's push on log-hazard."""
    x = scoring.Prep.from_state(bundle).transform(run_rows)[list(DRIVER_LABELS)]
    return x * bundle["model"].params_[list(DRIVER_LABELS)]


def main_driver(contrib: pd.DataFrame) -> pd.Series:
    """Label of the largest positive contribution per row, or NO_DRIVER if none is positive."""
    top = contrib.idxmax(axis=1).map(DRIVER_LABELS)
    return top.where(contrib.max(axis=1) > 0, NO_DRIVER)


def expected_cost(risk: pd.Series, cost_per_failure: pd.Series) -> pd.Series:
    return risk * cost_per_failure


def suggest_action(row, rpm_top_quartile: float) -> str:
    """First matching rule wins; the order is the order of priority."""
    if not (row.workover or row.new_string) and row.workover_payback_yrs <= PAYBACK_ACTION_YEARS:
        return f"At next failure: workover (pays back in {row.workover_payback_yrs:.1f} yr)"
    if row.main_driver == DRIVER_LABELS["rpm_pre30"] and row.rpm_pre30 >= rpm_top_quartile:
        return "Review pump speed"
    if row.risk_90 >= HIGH_RISK:
        return "Pre-plan rod-truck crew"
    return "Monitor"


def enrich_watchlist(watchlist: pd.DataFrame, runs: pd.DataFrame, events: pd.DataFrame,
                     cox: pd.DataFrame, model: dict, oil_value: float) -> pd.DataFrame:
    """Add cost per failure, expected 90-day cost, main driver, workover payback and an action."""
    w = watchlist.copy()
    inputs = plots.payback_inputs(runs, events, cox)
    # A well's own lost oil per failure if it has history, else the fleet mean
    # (the same mean the payback chart uses).
    oil_m3 = w.uwi.map(events.groupby("uwi").lost_oil_m3.mean()).fillna(inputs["oil_m3_per_event"])
    w["cost_per_failure"] = inputs["rodtruck_cost"] + oil_m3 * config.BBL_PER_M3 * oil_value
    w["expected_cost_90"] = expected_cost(w.risk_90, w.cost_per_failure)

    w["main_driver"] = main_driver(driver_contributions(model, open_run_features(w, runs))).to_numpy()
    # A well with no failure in the past year is costed at one a year, the
    # least it would take to trigger the workover being priced.
    w["workover_payback_yrs"], _ = plots.payback_years(np.maximum(w.fails_prior_365, 1),
                                                       oil_value=oil_value, **inputs)
    rpm_q75 = w.rpm_pre30.quantile(0.75)
    w["suggested_action"] = [suggest_action(r, rpm_q75) for r in w.itertuples()]
    return w


def top_summary(table: pd.DataFrame, k: int = config.TOP_K) -> str:
    top = table.head(k)
    n_wo = int(top["suggested action"].str.startswith("At next failure: workover").sum())
    return (f"Top {len(top)} wells: ${top['expected cost, 90 days'].sum():,.0f} expected failure cost "
            f"in the next {config.HORIZON_DAYS} days; {n_wo} flagged for a workover at next failure.")


# --------------------------------------------------------------------------
# What if?
# --------------------------------------------------------------------------
REPAIRS = {
    "Rod truck only": dict(workover=0, rodtruck=1, new_string=0),
    "Workover": dict(workover=1, rodtruck=0, new_string=0),
    "New rod string": dict(workover=0, rodtruck=1, new_string=1),
}
BASELINE_REPAIR = "Rod truck only"


def rpm_range(runs: pd.DataFrame) -> tuple[float, float]:
    """5th to 95th percentile of training RPM: the range the model has seen plenty of."""
    r = runs.loc[runs.split == "train", "rpm_pre30"]
    return float(r.quantile(0.05)), float(r.quantile(0.95))


def current_rpm(run_row: pd.DataFrame, bundle: dict) -> float:
    rpm = run_row.rpm_pre30.iloc[0]
    return float(bundle["medians"]["rpm_pre30"] if pd.isna(rpm) else rpm)


def next_run(run_row: pd.DataFrame, events: pd.DataFrame, as_of: pd.Timestamp,
             repair: str, rpm: float) -> pd.DataFrame:
    """The run that would follow if the well failed today and got `repair`.

    The failure that triggers the repair is added to the past-year count, and
    the previous gap becomes this run's start-to-start span, the same
    definition runs.py uses. Everything else stays at the well's current values.
    """
    r = run_row.copy()
    ev_start = events.loc[(events.uwi == r.uwi.iloc[0]) & (events.event_id == r.event_id.iloc[0]), "ev_start"]
    r["prev_gap_days"] = (as_of - pd.Timestamp(ev_start.iloc[0])).days
    r["fails_prior_365"] = r.fails_prior_365 + 1
    r["rpm_pre30"] = rpm
    for k, v in REPAIRS[repair].items():
        r[k] = v
    return r


def whatif(run_row: pd.DataFrame, bundle: dict, events: pd.DataFrame, as_of: pd.Timestamp,
           repair: str, rpm: float) -> pd.DataFrame:
    """90-day risk from restart for each repair option at the chosen RPM.

    Rows: one per repair option, plus the rod-truck-only baseline at the
    well's current RPM. `chosen` marks the selected option.
    """
    rows = {name: next_run(run_row, events, as_of, name, rpm) for name in REPAIRS}
    rows["baseline"] = next_run(run_row, events, as_of, BASELINE_REPAIR, current_rpm(run_row, bundle))
    x = scoring.Prep.from_state(bundle).transform(pd.concat(rows.values(), ignore_index=True))
    risk = scoring.risk_within(bundle["model"], x, bundle["horizon_days"], 0.0)
    return pd.DataFrame({"option": list(rows), "risk_90": risk,
                         "chosen": [n == repair for n in rows]})


def whatif_note(enriched_row: pd.Series, inputs: dict) -> str:
    extra = inputs["workover_cost"] - inputs["rodtruck_cost"]
    return (f"Extra cost ${extra:,.0f}; this well failed {int(enriched_row.fails_prior_365)} times in the past year; "
            f"a workover pays back in {enriched_row.workover_payback_yrs:.1f} years.")


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
