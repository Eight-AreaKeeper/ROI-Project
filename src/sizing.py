"""Step 5: size the problem. Every number on the problem-statement slides comes from here.

Writes
    outputs/sizing.csv       metric, value, unit, period, denominator, note
    outputs/pareto.csv       cumulative share of failures and lost oil by top wells
    outputs/failure_mix.csv  official failure log counts by type and year

Each block prints its intermediate counts so the arithmetic can be followed,
and the run ends with a check of every headline value against the figure
quoted in the brief.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.events import FAILURE_TYPE_SQL

# Headline figures quoted in the brief. They are only compared against, never
# used in a calculation.
EXPECTED = {
    "events": 10_548,
    "producing_well_days": 1_789_018,
    "failures_per_well_year": 2.15,
    "producing_days_per_event": 169.6,
    "median_failed_run_producing_days": 60,
    "median_failed_run_calendar_days": 62,
    "restarts_with_next_failure": 9_357,
    "quick_repeat_share_calendar": 0.288,
    "quick_repeat_share_producing": 0.302,
    "repair_jobs": 14_061,
    "reactive_jobs": 11_312,
    "reactive_job_share": 0.804,
    "repair_spend": 212_652_408,
    "reactive_spend": 164_217_276,
    "reactive_spend_share": 0.772,
    "spend_Workover": 171.9e6,
    "reactive_spend_Workover": 128.8e6,
    "spend_Rod Truck": 40.7e6,
    "reactive_spend_Rod Truck": 35.4e6,
    "downhole_lost_oil_m3": 208_840,
    "downhole_lost_oil_bbl": 1.31e6,
    "all_downtime_lost_oil_m3": 568_031,
    "downhole_share_of_lost_oil": 0.37,
    "pareto_all_top10_failures": 0.408,
    "pareto_all_top10_lost_oil": 0.513,
    "pareto_event_top10_failures": 0.294,
    "pareto_event_top10_lost_oil": 0.395,
}


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return pad + text.replace("\n", "\n" + pad)


class Sizing:
    """Collects metric rows and prints each one as it is added."""

    def __init__(self, period: str):
        self.period = period
        self.rows: list[dict] = []

    def add(self, metric, value, unit, denominator="", note="", period=None):
        self.rows.append({"metric": metric, "value": value, "unit": unit,
                          "period": period or self.period,
                          "denominator": denominator, "note": note})
        shown = f"{value:.1%}" if unit == "share" else f"{value:,.2f}".rstrip("0").rstrip(".")
        print(f"    {metric:<38} {shown:>16}  {unit}")

    def get(self, metric):
        return next(r["value"] for r in self.rows if r["metric"] == metric)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def register_views(con) -> None:
    con.execute(f"create or replace view dc as select * from read_parquet('{config.DAILY_CLEAN.as_posix()}')")
    con.execute(f"create or replace view ev as select * from read_parquet('{config.EVENTS.as_posix()}')")


def failure_rate(con, s: Sizing) -> None:
    print("\n1. Failure rate")
    n_events = con.sql("select count(*) from ev").fetchall()[0][0]
    # Producing days, not calendar days, are the exposure: a well that is shut
    # in cannot fail, so counting those days would make the rate look lower.
    pdays = con.sql("select count(*) from dc where in_scope and is_producing").fetchall()[0][0]
    print(f"    events = {n_events:,}; producing well-days = {pdays:,}; "
          f"well-years = {pdays:,} / 365 = {pdays / 365:,.1f}")
    s.add("events", n_events, "events", "in-scope wells", "rebuilt downhole events")
    s.add("producing_well_days", pdays, "well-days", "in-scope wells", "days with hours_on_prod > 0")
    s.add("failures_per_well_year", n_events / (pdays / 365), "failures per well-year",
          "producing well-years",
          "fleet rate: events / (producing well-days / 365). This is the number to quote as a rate.")
    s.add("producing_days_per_event", pdays / n_events, "producing days per event",
          "events", "the same fleet rate expressed as exposure per failure (inverse of the rate)")


def runs_between_events(con) -> pd.DataFrame:
    """One row per restart: the gap from an event's end to the next event's start."""
    return con.sql("""
    with gaps as (
        select uwi, event_id, ev_end,
               lead(ev_start) over (partition by uwi order by ev_start) as next_start
        from ev
    ),
    known as (select * from gaps where next_start is not null)
    -- Range join instead of a correlated subquery: DuckDB turns it into one
    -- pass over the producing days rather than one lookup per restart.
    select k.uwi, k.event_id, k.ev_end, k.next_start,
           k.next_start - k.ev_end as calendar_gap,
           count(d.production_date) as producing_gap
    from known k
    left join dc d
      on d.uwi = k.uwi and d.in_scope and d.is_producing
     and d.production_date > k.ev_end and d.production_date < k.next_start
    group by all
    """).df()


def quick_repeats(con, s: Sizing) -> None:
    print("\n2. Quick repeat failures")
    runs = runs_between_events(con)
    n = len(runs)
    q = config.QUICK_REPEAT_DAYS
    cal = int((runs.calendar_gap <= q).sum())
    prod = int((runs.producing_gap <= q).sum())
    print(f"    restarts with a known next failure = {n:,}")
    print(f"    next failure within {q} calendar days  = {cal:,} / {n:,}")
    print(f"    next failure within {q} producing days = {prod:,} / {n:,}")
    s.add("restarts_with_next_failure", n, "restarts", "events with a later event on the same well",
          "the last event on each well has no known next failure and is excluded")
    s.add("quick_repeat_share_calendar", cal / n, "share", "restarts with a known next failure",
          f"next event starts within {q} calendar days of the previous event ending")
    s.add("quick_repeat_share_producing", prod / n, "share", "restarts with a known next failure",
          f"fewer than or equal to {q} producing days between the two events")
    # Medians over runs that ended in a failure are biased short, because long
    # runs are more likely to still be running at the end of the data.
    s.add("median_failed_run_producing_days", float(runs.producing_gap.median()), "producing days",
          "runs that ended in a failure only",
          "MEDIAN OVER FAILED RUNS ONLY, biased short; not a fleet rate. Use failures_per_well_year for that.")
    s.add("median_failed_run_calendar_days", float(runs.calendar_gap.median()), "calendar days",
          "runs that ended in a failure only",
          "MEDIAN OVER FAILED RUNS ONLY, biased short; not a fleet rate.")


def reactive_spend(con, s: Sizing) -> None:
    print("\n3. Reactive versus planned spend")
    cats = ", ".join(f"'{c}'" for c in config.REPAIR_JOB_CATEGORIES)
    lb = config.REACTIVE_LOOKBACK_DAYS
    jobs = con.sql(f"""
    select j.uwi, j.job_category, j.start_date, j.field_est_total,
           -- A repair is reactive when the well was already down for a
           -- downhole reason in the days before the crew arrived. The window
           -- includes the job start day itself because the rig often arrives
           -- on the first down day.
           exists (select 1 from dc
                   where dc.uwi = j.uwi and dc.in_scope and dc.is_downhole
                     and dc.production_date between j.start_date - {lb} and j.start_date) as reactive
    from jobs j
    where j.job_category in ({cats})
    """).df()
    label = config.CURRENCY_LABEL
    n, n_r = len(jobs), int(jobs.reactive.sum())
    spend, spend_r = jobs.field_est_total.sum(), jobs.loc[jobs.reactive, "field_est_total"].sum()
    print(f"    jobs = {n:,}; reactive = {n_r:,}; planned = {n - n_r:,}")
    print(f"    spend = ${spend:,.0f}; reactive = ${spend_r:,.0f}; planned = ${spend - spend_r:,.0f}  ({label})")

    s.add("repair_jobs", n, "jobs", ", ".join(config.REPAIR_JOB_CATEGORIES), "all jobs in these categories")
    s.add("reactive_jobs", n_r, "jobs", "repair jobs",
          f"in-scope downhole downtime on the same well in the {lb} days up to and including job start")
    s.add("planned_jobs", n - n_r, "jobs", "repair jobs", "no downhole downtime in the lookback window")
    s.add("reactive_job_share", n_r / n, "share", "repair jobs", "")
    s.add("repair_spend", spend, f"CAD ({label})", "repair jobs",
          "sum of field_est_total; zero-cost jobs kept in sums")
    s.add("reactive_spend", spend_r, f"CAD ({label})", "repair jobs", "")
    s.add("planned_spend", spend - spend_r, f"CAD ({label})", "repair jobs", "")
    s.add("reactive_spend_share", spend_r / spend, "share", "repair spend", "")

    # A $0 field estimate means the cost was not entered, not that the repair
    # was free, so zeros are dropped for medians but kept in the sums above.
    priced = jobs[jobs.field_est_total > 0]
    s.add("zero_cost_jobs", int((jobs.field_est_total == 0).sum()), "jobs", "repair jobs",
          "field_est_total = 0; treated as missing for medians")
    s.add("median_cost_reactive", float(priced.loc[priced.reactive, "field_est_total"].median()),
          f"CAD ({label})", "reactive jobs with cost > 0", "")
    s.add("median_cost_planned", float(priced.loc[~priced.reactive, "field_est_total"].median()),
          f"CAD ({label})", "planned jobs with cost > 0", "")

    for cat, g in jobs.groupby("job_category"):
        tot, rea = g.field_est_total.sum(), g.loc[g.reactive, "field_est_total"].sum()
        print(f"    {cat}: {len(g):,} jobs, {int(g.reactive.sum()):,} reactive; "
              f"spend ${tot:,.0f}, reactive ${rea:,.0f}")
        s.add(f"jobs_{cat}", len(g), "jobs", cat, "")
        s.add(f"reactive_jobs_{cat}", int(g.reactive.sum()), "jobs", cat, "")
        s.add(f"spend_{cat}", tot, f"CAD ({label})", cat, "")
        s.add(f"reactive_spend_{cat}", rea, f"CAD ({label})", cat, "")


def lost_oil(con, s: Sizing) -> None:
    print("\n4. Lost oil")
    dh = con.sql("select sum(lost_oil_m3) from ev").fetchall()[0][0]
    # Total lost oil over all downtime codes on in-scope rows, for contrast
    # only. It includes deliberate long-term shut-ins, which are not avoidable.
    total = con.sql("select sum(lost_oil_m3) from dc where in_scope and hours_down > 0").fetchall()[0][0]
    print(f"    downhole events = {dh:,.0f} m3 x {config.BBL_PER_M3} = {dh * config.BBL_PER_M3:,.0f} bbl")
    print(f"    all downtime causes = {total:,.0f} m3; downhole share = {dh / total:.1%}")
    s.add("downhole_lost_oil_m3", dh, "m3", "rebuilt downhole events", "lost_oil_m3 on event down days only")
    s.add("downhole_lost_oil_bbl", dh * config.BBL_PER_M3, "bbl", "rebuilt downhole events",
          f"{config.BBL_PER_M3} bbl per m3")
    s.add("all_downtime_lost_oil_m3", total, "m3", "in-scope down days, all causes",
          "includes long-term uneconomic shut-ins; contrast only")
    s.add("downhole_share_of_lost_oil", dh / total, "share", "all in-scope lost oil", "")


def pareto(con, s: Sizing) -> pd.DataFrame:
    print("\n5. Concentration (Pareto)")
    wells = con.sql("""
    select w.uwi,
           count(e.event_id) as failures,
           coalesce(sum(e.lost_oil_m3), 0) as lost_oil_m3
    -- Start from every in-scope well, so wells that never failed count in the
    -- 'all producers' denominator with zero failures.
    from (select distinct uwi from dc where in_scope) w
    left join ev e using (uwi)
    group by w.uwi
    """).df()

    out = []
    populations = {
        "all_in_scope_producers": wells,
        "wells_with_at_least_one_event": wells[wells.failures > 0],
    }
    for denom, pop in populations.items():
        n = len(pop)
        for measure in ("failures", "lost_oil_m3"):
            # Sort by uwi as a tie-breaker so the ranking is reproducible.
            ranked = pop.sort_values([measure, "uwi"], ascending=[False, True])
            total = ranked[measure].sum()
            cum = ranked[measure].cumsum().to_numpy()
            for pct in config.PARETO_CUTS:
                k = max(1, int(round(n * pct / 100)))
                out.append({"denominator": denom, "measure": measure, "top_pct_wells": pct,
                            "wells": k, "wells_in_population": n,
                            "cumulative_share": cum[k - 1] / total})
    par = pd.DataFrame(out)

    for denom in populations:
        sub = par[par.denominator == denom]
        n = sub.wells_in_population.iloc[0]
        print(f"    denominator = {denom} ({n:,} wells)")
        pivot = sub.pivot(index=["top_pct_wells", "wells"], columns="measure", values="cumulative_share")
        print(_indent(pivot.map(lambda v: f"{v:.1%}").to_string(), 6))

    def top10(denom, measure):
        return par[(par.denominator == denom) & (par.measure == measure)
                   & (par.top_pct_wells == 10)].iloc[0]

    for key, denom in (("all", "all_in_scope_producers"), ("event", "wells_with_at_least_one_event")):
        for measure, short in (("failures", "failures"), ("lost_oil_m3", "lost_oil")):
            r = top10(denom, measure)
            s.add(f"pareto_{key}_top10_{short}", r.cumulative_share, "share", denom,
                  f"top 10% = {r.wells} of {r.wells_in_population} wells, ranked by {measure}. "
                  f"Denominator: {denom}.")
    return par


def failure_mix(con) -> pd.DataFrame:
    print("\n6. Failure mix (official failure log)")
    mix = con.sql(f"""
    select year(failure_start_date) as year,
           {FAILURE_TYPE_SQL} as failure_type,
           count(*) as failures
    from fails
    group by all
    order by year, failures desc
    """).df()
    mix["note"] = ("Descriptive only, never a training label: the log is effectively empty "
                   "before mid-2025.")
    by_year = mix.groupby("year").failures.sum()
    print("    records by year: " + ", ".join(f"{y}: {n:,}" for y, n in by_year.items()))
    print("    the log is effectively empty before mid-2025, so this is descriptive only")
    print(_indent(mix.pivot_table(index="failure_type", columns="year", values="failures",
                                  aggfunc="sum", fill_value=0).to_string(), 4))
    return mix


def compare(s: Sizing) -> list[str]:
    print("\nCheck against the figures quoted in the brief "
          f"(flag when off by more than {config.CHECK_TOLERANCE:.0%})")
    flagged = []
    for metric, expected in EXPECTED.items():
        got = s.get(metric)
        diff = abs(got - expected) / abs(expected)
        status = "ok" if diff <= config.CHECK_TOLERANCE else "DIFFERS"
        if status != "ok":
            flagged.append(metric)
        print(f"    {status:<8} {metric:<38} computed {got:>16,.4f}  expected {expected:>16,.4f}  ({diff:.2%})")
    if flagged:
        print(f"    {len(flagged)} value(s) differ: {', '.join(flagged)}")
    else:
        print("    all values within tolerance")
    return flagged


def main() -> pd.DataFrame:
    config.ensure_dirs()
    con = config.connect()
    register_views(con)

    lo, hi = con.sql("select min(production_date), max(production_date) from dc where in_scope").fetchall()[0]
    s = Sizing(f"{lo} to {hi}")
    print(f"sizing: period {s.period}")

    failure_rate(con, s)
    quick_repeats(con, s)
    reactive_spend(con, s)
    lost_oil(con, s)
    par = pareto(con, s)
    mix = failure_mix(con)

    sizing = s.frame()
    config.write_csv(sizing, config.OUTPUTS / "sizing.csv")
    config.write_csv(par, config.OUTPUTS / "pareto.csv")
    config.write_csv(mix, config.OUTPUTS / "failure_mix.csv")

    compare(s)
    print(f"\nsizing: wrote sizing.csv, pareto.csv, failure_mix.csv to {config.rel(config.OUTPUTS)}/")
    return sizing


if __name__ == "__main__":
    main()
