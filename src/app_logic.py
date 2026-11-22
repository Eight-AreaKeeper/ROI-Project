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


# Measured on a laptop on 2026-11-22: `python run_all.py` end to end, a cold
# `score_wells.main()`, and the app's first page load in a fresh process.
PRODUCTION_NOTES = """
- **Weekly batch:** new daily data → `python run_all.py` → refreshed watchlist. One scheduled job, no new sensors or hardware.
- **Speed:** the full pipeline runs in about 1.5 minutes (84 s) on 5 million well-days with DuckDB; scoring all 805 open wells takes about 2 s, and the app opens in about 2 s.
- **Scales to:** more wells or areas with no code change (settings live in `.env`). The SQL steps port directly to Databricks or Spark, and a historian feed (for example Canary) can replace the parquet files.
- **Cost to run:** a laptop or one small VM. The value side is about $40M a year of failure cost, half of it on the worst 10% of wells.
- **Who uses it:** a production engineer reviews the top 50 weekly and books workovers on flagged wells at their next failure.
- **Next steps:** confirm rebuilt events with field staff; pilot workover vs rod truck on watchlist wells; retrain quarterly.
"""

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
# The story: every number the app quotes, computed from the outputs
# --------------------------------------------------------------------------
def impact_summary(enriched: pd.DataFrame, model_eval: pd.DataFrame, cox: pd.DataFrame,
                   payback_inputs: dict, oil_value: float, sizing: pd.DataFrame,
                   exclude_thorough: bool = True) -> dict:
    """Headline impact: better targeting, and workovers that pay back in year one.

    A well is a workover candidate when one year of avoided failures is worth
    more than the workover's extra cost:
        net = fails_prior_365 x (1 - HR) x cost_per_failure - extra cost > 0
    With exclude_thorough, wells whose last repair was already a workover or
    a new rod string are left out, since "workover instead of a quick fix"
    does not apply to them.
    """
    ev = model_eval.set_index(["model", "metric"]).value
    k = config.TOP_K
    model, rule = ev[("cox", f"indep_top{k}_hits")], ev[("baseline_a", f"indep_top{k}_hits")]
    hr = payback_inputs["hazard_ratio"]
    extra = payback_inputs["workover_cost"] - payback_inputs["rodtruck_cost"]
    w = enriched
    net = w.fails_prior_365 * (1 - hr) * w.cost_per_failure - extra
    cand = net > 0
    if exclude_thorough:
        cand &= ~((w.workover == 1) | (w.new_string == 1))
    repairs, oil = plots.annual_cost(sizing, oil_value)
    return {
        "lift_pct": float((model - rule) / rule),
        "extra_hits": float(model - rule),
        "workover_wells": int(cand.sum()),
        "failures_avoided": float((w.fails_prior_365[cand] * (1 - hr)).sum()),
        "net_savings_year1": float(net[cand].sum()),
        "fleet_cost_per_year": float((repairs + oil) * 1e6),
    }


def story_numbers(d: dict[str, pd.DataFrame], payback_inputs: dict) -> dict:
    """The comparison numbers the pages quote, read from the output files."""
    s = d["sizing"].set_index("metric").value
    km = d["km_summary"].set_index(["group_var", "group"]).median_prod_days
    rpm = d["km_summary"][d["km_summary"].group_var == "rpm_quartile"]
    rpm_lo = rpm.loc[rpm.group.map(lambda g: float(g.strip("(").split(",")[0])).idxmin(), "median_prod_days"]
    rpm_hi = rpm.loc[rpm.group.map(lambda g: float(g.strip("(").split(",")[0])).idxmax(), "median_prod_days"]
    ev = d["model_eval"].set_index(["model", "metric"])
    k = config.TOP_K
    rule = ev.loc[("baseline_a", f"indep_top{k}_hits")]
    val = d["event_validation"]
    recall = val[(val.metric == "log_to_event_recall") & (val.gap_days == config.EVENT_GAP_DAYS)
                 & (val.group == "all")].value.iloc[0]
    years = plots.YEARS
    return {
        "wells": int(d["pareto"].wells_in_population.iloc[0]),
        "failures_per_year": float(s["events"] / years),
        "failures_per_well_year": float(s["failures_per_well_year"]),
        "producing_days_per_failure": float(s["producing_days_per_event"]),
        "repeat_30d": float(s["quick_repeat_share_calendar"]),
        "reactive_share": float(s["reactive_spend_share"]),
        "repair_spend": float(s["repair_spend"]),
        "top10_lost_oil": float(s["pareto_all_top10_lost_oil"]),
        "days_2plus": float(km[("fails_prior_365", "2+")]),
        "days_none": float(km[("fails_prior_365", "0")]),
        "days_fast": float(rpm_hi), "days_slow": float(rpm_lo),
        "days_workover": float(km[("repair_class", "workover")]),
        "days_rodtruck": float(km[("repair_class", "rod truck only")]),
        "workover_risk_cut": float(1 - payback_inputs["hazard_ratio"]),
        "workover_extra_cost": float(payback_inputs["workover_cost"] - payback_inputs["rodtruck_cost"]),
        "model_hits": float(ev.loc[("cox", f"indep_top{k}_hits"), "value"]),
        "rule_hits": float(rule.value), "rule_min": float(rule.ci_low), "rule_max": float(rule.ci_high),
        "random_hits": float(ev.loc[("independent_check", "base_rate"), "value"] * k),
        "c_index": float(ev.loc[("cox", "c_index"), "value"]),
        "c_index_rule": float(ev.loc[("baseline_a", "c_index"), "value"]),
        "auc": float(ev.loc[("cox", "indep_auc_90"), "value"]),
        "auc_rule": float(ev.loc[("baseline_a", "indep_auc_90"), "value"]),
        "brier": float(ev.loc[("cox", "brier_90"), "value"]),
        "brier_km": float(ev.loc[("baseline_c_km", "brier_90"), "value"]),
        "test_runs": int(ev.loc[("cox", "c_index"), "n"]),
        "check_wells": int(ev.loc[("independent_check", "wells"), "value"]),
        "recall": float(recall),
    }


def in_tens(share: float) -> str:
    return f"{round(share * 10):.0f} in 10"


def sidebar_story(n: dict, imp: dict) -> str:
    return (
        f"**Problem**  \nHeavy-oil wells break down about {n['failures_per_well_year']:.0f} times a year, "
        f"{in_tens(n['repeat_30d'])} break again within a month, and failures cost about "
        f"${imp['fleet_cost_per_year'] / 1e6:.0f}M a year.\n\n"
        "**Our question**  \nWhich conditions and repair choices make a well run longer between failures, "
        "and is a better repair worth its cost?\n\n"
        f"**Our answer**  \nRepeat-failure wells are predictable. A workover instead of a quick fix pays for "
        f"itself on the worst ~{imp['workover_wells']} wells, and our watchlist finds "
        f"{imp['lift_pct']:.0%} more failing wells than today's rule."
    )


def impact_headline(imp: dict) -> str:
    return (f"Same crews, better targets: our watchlist finds {imp['lift_pct']:.0%} more failing wells than "
            f"today's rule, and fixing ~{imp['workover_wells']} repeat-failure wells properly avoids "
            f"~{imp['failures_avoided']:.0f} failures a year.")


def share_words(share: float) -> str:
    """ "half of" near 50%, otherwise the percentage, for sentences like "cause half the lost oil"."""
    return "half" if 0.45 <= share <= 0.55 else f"{share:.0%} of"


def as_long(ratio: float) -> str:
    return "less than half as long" if ratio < 0.5 else f"{ratio:.0%} as long"


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
