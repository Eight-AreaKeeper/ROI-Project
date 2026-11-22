"""Step 4: one row per run between downhole failures, with pre-run features.

A run starts the day after an event ends and lasts until the next event on the
same well starts (event = 1), or until the well's last producing day plus one
if no later event exists (event = 0, censored). Its length is counted in
producing days, because a well that is shut in cannot fail.

This is the only module that builds model features. Every feature is measured
before the run starts, so nothing the model sees could have been caused by the
outcome it predicts.

Writes outputs/wells/runs.parquet.
"""
from __future__ import annotations

import pandas as pd

from src import config

FEATURES = [
    "oil_pre30", "wcut_pre30", "rpm_pre30", "torque_pre30", "util_pre30",
    "fails_prior_365", "prev_gap_days", "prev_down_days", "disp",
    "workover", "rodtruck", "new_string", "corod", "stick", "has_job", "repair_cost",
]

EXPECTED = {
    "runs": 10_364,
    "wells": 1_161,
    "failures": 9_350,
    "censored": 1_014,
    "median_failed_prod_days": 60,
    "has_job_share": 0.872,
    "workover_share_of_linked": 0.368,
    "missing_torque_pre30": 0.34,
    "missing_prev_gap_days": 0.11,
}

# Differences reviewed by the team and accepted, with the reason.
ACCEPTED = {
    "missing_torque_pre30": "34.3% vs 34%: the brief rounded to a whole percent",
    "missing_prev_gap_days": "11.2% vs 11%: the brief rounded to a whole percent",
}


def register_views(con) -> None:
    con.execute(f"""
    create or replace view dc as
    select uwi, production_date, is_producing, oil_rate_m3d, water_cut, pcp_pump_rpm,
           pcp_pump_rod_torque_ftlb, pump_utilisation, pcp_pump_displacement_m3d100rpm
    from read_parquet('{config.DAILY_CLEAN.as_posix()}')
    where in_scope
    """)
    con.execute(f"create or replace view ev as select * from read_parquet('{config.EVENTS.as_posix()}')")


def build_runs(con) -> pd.DataFrame:
    pre, lb = config.PRE_WINDOW_PROD_DAYS, config.FAILS_PRIOR_DAYS
    jb, ja, disp_lb = config.JOB_LINK_DAYS_BEFORE, config.JOB_LINK_DAYS_AFTER, config.DISP_LOOKBACK_DAYS
    cats = ", ".join(f"'{c}'" for c in config.REPAIR_JOB_CATEGORIES)

    return con.sql(f"""
    with last_prod as (
        select uwi, max(production_date) as last_prod_day
        from dc where is_producing group by uwi
    ),
    ev_seq as (
        select e.*,
               lead(ev_start) over w as next_start,
               lag(ev_start) over w as prev_start
        from ev e
        window w as (partition by uwi order by ev_start)
    ),
    bounds as (
        select s.uwi, s.event_id, s.area, s.ev_start, s.ev_end, s.down_days, s.prev_start,
               s.ev_end + 1 as start_d,
               -- The last event on a well has no next failure: the run is
               -- censored the day after the well last produced, which is the
               -- last day we know it was still running.
               coalesce(s.next_start, l.last_prod_day + 1) as end_d,
               (s.next_start is not null)::int as event
        from ev_seq s
        left join last_prod l using (uwi)
    ),
    lengths as (
        select b.uwi, b.event_id,
               count(d.production_date) as prod_days
        from bounds b
        left join dc d
          on d.uwi = b.uwi and d.is_producing
         and d.production_date >= b.start_d and d.production_date < b.end_d
        group by all
    ),
    prod_seq as (
        -- Numbering each well's producing days lets the "last 30 producing
        -- days" window be a cheap range on the sequence number instead of a
        -- join against every earlier day of the well.
        select *, row_number() over (partition by uwi order by production_date) as k
        from dc where is_producing
    ),
    anchor as (
        -- Sequence number of the last producing day strictly before ev_start.
        -- The window ends before the failure, not before the restart: during
        -- the outage the well is down, so nothing is observable then.
        select b.uwi, b.event_id, max(p.k) as k_last
        from bounds b
        join prod_seq p on p.uwi = b.uwi and p.production_date < b.ev_start
        group by all
    ),
    pre_window as (
        select a.uwi, a.event_id,
               avg(p.oil_rate_m3d) as oil_pre30,
               avg(p.water_cut) as wcut_pre30,
               -- Medians for the pump readings, because they are carried
               -- forward between field visits and a single bad entry would
               -- move a mean.
               median(p.pcp_pump_rpm) as rpm_pre30,
               median(p.pcp_pump_rod_torque_ftlb) as torque_pre30,
               median(p.pump_utilisation) as util_pre30
        from anchor a
        join prod_seq p on p.uwi = a.uwi and p.k between a.k_last - {pre - 1} and a.k_last
        group by all
    ),
    prior_fails as (
        select b.uwi, b.event_id, count(e2.event_id) as fails_prior_365
        from bounds b
        left join ev e2
          on e2.uwi = b.uwi and e2.ev_start >= b.ev_start - {lb} and e2.ev_start < b.ev_start
        group by all
    ),
    disp as (
        -- The days just before the restart cover the repair itself, so a pump
        -- swapped during the outage shows its new displacement here. That is
        -- known at restart, so it is still a pre-run value.
        select b.uwi, b.event_id, max(d.pcp_pump_displacement_m3d100rpm) as disp
        from bounds b
        join dc d on d.uwi = b.uwi
         and d.production_date >= b.start_d - {disp_lb} and d.production_date < b.start_d
        group by all
    ),
    jobs_linked as (
        -- Jobs from a few days before the failure to a few days after the
        -- restart: crews often book the job before the well is logged down,
        -- and paperwork can trail the restart by a day or two.
        select b.uwi, b.event_id,
               max((j.job_category = 'Workover')::int) as workover,
               max((j.job_category = 'Rod Truck')::int) as rodtruck,
               max((j.job_type ilike '%new string%')::int) as new_string,
               max((j.job_type ilike '%corod%')::int) as corod,
               max((j.job_type ilike '%stick%')::int) as stick,
               -- A $0 field estimate means the cost was not entered.
               sum(nullif(j.field_est_total, 0)) as repair_cost
        from bounds b
        join jobs j on j.uwi = b.uwi
         and j.job_category in ({cats})
         and j.start_date between b.ev_start - {jb} and b.start_d + {ja}
        group by all
    )
    select b.uwi, b.event_id, b.area, b.start_d, b.end_d,
           l.prod_days, b.end_d - b.start_d as cal_days, b.event,
           -- Features. Each comment says when the value is known.
           pw.oil_pre30,          -- known at failure: 30 producing days before ev_start
           pw.wcut_pre30,         -- known at failure: 30 producing days before ev_start
           pw.rpm_pre30,          -- known at failure: 30 producing days before ev_start
           pw.torque_pre30,       -- known at failure: 30 producing days before ev_start
           pw.util_pre30,         -- known at failure: 30 producing days before ev_start
           pf.fails_prior_365,    -- known at failure: events in the year before ev_start
           b.ev_start - b.prev_start as prev_gap_days,  -- known at failure: start to start
           b.down_days as prev_down_days,               -- known at restart: outage length
           dp.disp,               -- known at restart: max displacement in the 60 days before
           coalesce(j.workover, 0) as workover,      -- known at restart: repair done
           coalesce(j.rodtruck, 0) as rodtruck,      -- known at restart: repair done
           coalesce(j.new_string, 0) as new_string,  -- known at restart: repair done
           coalesce(j.corod, 0) as corod,            -- known at restart: repair done
           coalesce(j.stick, 0) as stick,            -- known at restart: repair done
           (j.event_id is not null)::int as has_job, -- known at restart: repair done
           j.repair_cost                             -- known at restart: field estimate
    from bounds b
    join lengths l using (uwi, event_id)
    left join pre_window pw using (uwi, event_id)
    left join prior_fails pf using (uwi, event_id)
    left join disp dp using (uwi, event_id)
    left join jobs_linked j using (uwi, event_id)
    -- A run with no producing days never started, so it says nothing about
    -- how long a repair lasts.
    where l.prod_days > 0
    order by b.uwi, b.start_d
    """).df()


def summarise(runs: pd.DataFrame) -> dict:
    linked = runs[runs.has_job == 1]
    return {
        "runs": len(runs),
        "wells": runs.uwi.nunique(),
        "failures": int(runs.event.sum()),
        "censored": int((runs.event == 0).sum()),
        "median_failed_prod_days": float(runs.loc[runs.event == 1, "prod_days"].median()),
        "has_job_share": runs.has_job.mean(),
        "workover_share_of_linked": linked.workover.mean(),
        "missing_torque_pre30": runs.torque_pre30.isna().mean(),
        "missing_prev_gap_days": runs.prev_gap_days.isna().mean(),
    }


def compare(got: dict, expected: dict, tol: float) -> list[str]:
    flagged = []
    for k, exp in expected.items():
        diff = abs(got[k] - exp) / abs(exp)
        status = "ok" if diff <= tol else ("accepted" if k in ACCEPTED else "DIFFERS")
        if status == "DIFFERS":
            flagged.append(k)
        reason = f"  {ACCEPTED[k]}" if status == "accepted" else ""
        print(f"    {status:<8} {k:<28} computed {got[k]:>12,.4f}  expected {exp:>12,.4f}  ({diff:.2%}){reason}")
    print("    all values within tolerance" if not flagged else f"    differs: {', '.join(flagged)}")
    return flagged


def main() -> pd.DataFrame:
    config.ensure_dirs()
    con = config.connect()
    register_views(con)
    runs = build_runs(con)
    # The split is by run start, so a run is tested only if it began after
    # everything the model was trained on.
    runs["split"] = (runs.start_d >= pd.Timestamp(config.TIME_CUT)).map({True: "test", False: "train"})
    runs.to_parquet(config.RUNS, index=False)

    print(f"runs: {len(runs):,} runs on {runs.uwi.nunique():,} wells; "
          f"{int(runs.event.sum()):,} failures, {int((runs.event == 0).sum()):,} censored")
    print("  share missing per feature:")
    miss = runs[FEATURES].isna().mean().sort_values(ascending=False)
    for f, v in miss.items():
        print(f"    {f:<18} {v:6.1%}")
    print("  split: " + ", ".join(f"{k} {v:,}" for k, v in runs.split.value_counts().items()))
    print(f"\nCheck against the brief (flag when off by more than {config.CHECK_TOLERANCE:.0%})")
    compare(summarise(runs), EXPECTED, config.CHECK_TOLERANCE)
    print(f"runs: wrote {config.rel(config.RUNS)}")
    return runs


if __name__ == "__main__":
    main()
