"""Step 3: rebuild downhole failure events from the cleaned daily table.

Writes
    outputs/wells/events.parquet       one row per rebuilt event
    outputs/event_validation.csv       agreement with the official failure log,
                                       at the configured gap and the sensitivity gap

The official failure log only covers mid-2025 onward, so events are rebuilt
from downtime codes to get a multi-year history. The log is used here to check
how well the rebuilt events agree with it, never as a label.
"""
from __future__ import annotations

import pandas as pd

from src import config

EVENT_COLS = ["uwi", "event_id", "area", "ev_start", "ev_end", "down_days",
              "lost_oil_m3", "matched_failure_log", "failure_type", "main_downtime_code"]

# The raw failure_type field is free text. Each spelling found in the log is
# mapped to a family explicitly, so a new spelling lands in Other where it is
# visible, instead of being guessed into a family by a substring match.
FAILURE_TYPE_MAP = {
    "Rods": ["Rods", "Rod", "|Rod", "Parted rod", "Pared rod"],
    "Tubing": ["Tubing", "TUBING", "Parted tbg", "Tubing Parted", "Hole in tubing",
               "Hole in tbg colar", "Parted tubing Broke"],
    "Pump": ["Pump", "PCP", "ES- PCP", "Broken insert"],
    "Packer": ["Packer", "packer", "PACKER", "Packer  repair", "Hole in mandral packer"],
}
FAILURE_TYPE_SQL = "case " + " ".join(
    "when failure_type in (" + ", ".join("'" + s.replace("'", "''") + "'" for s in spellings)
    + f") then '{family}'"
    for family, spellings in FAILURE_TYPE_MAP.items()
) + " else 'Other' end"

# Specific families win over Other when one well-day has two log records, so
# the kept record carries the most informative label.
FAMILY_PRIORITY_SQL = "case failure_type_family when 'Other' then 1 else 0 end"


def register_fails(con) -> None:
    con.execute(f"""
    create or replace view fails_typed as
    with typed as (
        select uwi, failure_start_date::date as fdate, failure_type as failure_type_raw,
               {FAILURE_TYPE_SQL} as failure_type_family
        from fails
    )
    -- One logged failure per well per day: the log holds a few same-day
    -- double entries, which would otherwise count one failure twice in recall.
    select uwi, fdate, failure_type_raw, failure_type_family as failure_type
    from typed
    qualify row_number() over (partition by uwi, fdate
                               order by {FAMILY_PRIORITY_SQL}, failure_type_family,
                                        failure_type_raw) = 1
    """)


def log_fails_cleaning(con) -> None:
    """Append the failure-log rules to cleaning_log.csv, which clean.py creates."""
    raw = con.sql("select count(*) from fails").fetchall()[0][0]
    kept = con.sql("select count(*) from fails_typed").fetchall()[0][0]
    mix = con.sql(f"""
        select {FAILURE_TYPE_SQL} as family, count(*) as n,
               count(*) filter (where failure_type is distinct from
                                ({FAILURE_TYPE_SQL})) as renamed
        from fails group by 1 order by 1
    """).df()
    rows = [{"step": "failure_log", "rule": f"failure_type mapped to {r.family}", "column": "failure_type",
             "rows_affected": int(r.renamed), "rows_after": raw,
             "note": f"{int(r.n)} records in family; rows_affected = spellings changed"}
            for r in mix.itertuples()]
    rows.append({"step": "failure_log", "rule": "drop same-well same-day duplicate records", "column": "*",
                 "rows_affected": raw - kept, "rows_after": kept,
                 "note": "used for validation only; keeps the most specific failure type"})
    path = config.OUTPUTS / "cleaning_log.csv"
    log = pd.read_csv(path) if path.exists() else pd.DataFrame()
    # Replace earlier failure-log rows so rerunning this step alone does not
    # append the same rules twice.
    if not log.empty:
        log = log[log.step != "failure_log"]
    config.write_csv(pd.concat([log, pd.DataFrame(rows)], ignore_index=True), path)


def build_events(con, gap_days: int, table: str = "events") -> None:
    """Merge downhole days closer than `gap_days` into one event per well."""
    w = config.MATCH_WINDOW_DAYS
    con.execute(f"""
    create or replace table {table} as
    with down as (
        select uwi, production_date as d, field_operating_area, lost_oil_m3, downtime_code_1
        from read_parquet('{config.DAILY_CLEAN.as_posix()}')
        where in_scope and is_downhole
    ),
    marked as (
        -- A down day starts a new event when the previous down day on the
        -- same well is more than gap_days earlier. A single failure often
        -- shows up as several separate down days while the crew waits for a
        -- rig, so a short gap is treated as the same failure.
        select *,
               case when d - lag(d) over (partition by uwi order by d) <= {gap_days}
                    then 0 else 1 end as starts_event
        from down
    ),
    numbered as (
        select *, sum(starts_event) over (partition by uwi order by d) as ev_seq
        from marked
    ),
    area_pick as (
        -- Area can change on a well's record; the most common value over the
        -- event's own days is the one the crew worked under. mode() breaks
        -- ties arbitrarily, so ties go to the value seen first, then by name,
        -- which keeps reruns identical.
        select uwi, ev_seq, field_operating_area as area
        from numbered
        group by uwi, ev_seq, field_operating_area
        qualify row_number() over (partition by uwi, ev_seq
                                   order by count(*) desc, min(d), field_operating_area) = 1
    ),
    code_pick as (
        select uwi, ev_seq, downtime_code_1 as main_downtime_code
        from numbered
        group by uwi, ev_seq, downtime_code_1
        qualify row_number() over (partition by uwi, ev_seq
                                   order by count(*) desc, min(d), downtime_code_1) = 1
    ),
    ev as (
        select n.uwi, n.ev_seq,
               any_value(a.area) as area,
               min(n.d) as ev_start, max(n.d) as ev_end,
               count(*) as down_days,
               coalesce(sum(n.lost_oil_m3), 0) as lost_oil_m3,
               any_value(c.main_downtime_code) as main_downtime_code
        from numbered n
        join area_pick a using (uwi, ev_seq)
        join code_pick c using (uwi, ev_seq)
        group by n.uwi, n.ev_seq
    ),
    nearest_log as (
        -- The log date is a repair or logging date, often a few days after
        -- the well went down, so the match uses a symmetric window around the
        -- event start and keeps the closest logged failure.
        select e.uwi, e.ev_seq, f.failure_type,
               row_number() over (partition by e.uwi, e.ev_seq
                                  order by abs(f.fdate - e.ev_start), f.fdate, f.failure_type) as rn
        from ev e
        join fails_typed f
          on f.uwi = e.uwi and abs(f.fdate - e.ev_start) <= {w}
    )
    select e.uwi,
           row_number() over (order by e.uwi, e.ev_start) as event_id,
           e.area, e.ev_start, e.ev_end, e.down_days, e.lost_oil_m3,
           n.failure_type is not null as matched_failure_log,
           n.failure_type,
           e.main_downtime_code
    from ev e
    left join nearest_log n on n.uwi = e.uwi and n.ev_seq = e.ev_seq and n.rn = 1
    order by e.uwi, e.ev_start
    """)


def validate(con, table: str, gap_days: int) -> tuple[list[dict], pd.DataFrame]:
    """Recall and precision of rebuilt events against the official log."""
    w, start = config.MATCH_WINDOW_DAYS, config.VALIDATION_START
    rows: list[dict] = []

    def add(metric, group, num, den, note=""):
        rows.append({"gap_days": gap_days, "metric": metric, "group": group,
                     "numerator": int(num), "denominator": int(den),
                     "value": num / den if den else None, "note": note})

    def add_value(metric, value, note=""):
        rows.append({"gap_days": gap_days, "metric": metric, "group": "all",
                     "numerator": None, "denominator": None, "value": float(value), "note": note})

    s = con.sql(f"""
        select count(*) n_events, count(distinct uwi) n_wells, median(down_days) med_days,
               avg(down_days) mean_days, sum(down_days) total_days
        from {table}
    """).df().iloc[0]
    add_value("events", s["n_events"], "rebuilt events")
    add_value("event_wells", s["n_wells"], "wells with at least one event")
    add_value("median_down_days", s["med_days"])
    add_value("mean_down_days", s["mean_days"])
    add_value("total_down_days", s["total_days"])

    # Recall is limited to wells that appear in the event table, because a
    # logged failure on an out-of-scope well can never be matched by design.
    recall = con.sql(f"""
        select f.failure_type,
               exists (select 1 from {table} e
                       where e.uwi = f.uwi and abs(e.ev_start - f.fdate) <= {w}) as hit
        from fails_typed f
        where f.fdate >= date '{start}'
          and f.uwi in (select uwi from {table})
    """).df()
    add("log_to_event_recall", "all", recall.hit.sum(), len(recall),
        f"logged failures since {start} with a rebuilt event starting within {w} days")
    for ftype, g in recall.groupby("failure_type"):
        add("log_to_event_recall", ftype, g.hit.sum(), len(g), "")

    precision = con.sql(f"""
        select e.main_downtime_code,
               exists (select 1 from fails_typed f
                       where f.uwi = e.uwi and abs(e.ev_start - f.fdate) <= {w}) as hit
        from {table} e
        where e.ev_start >= date '{start}'
    """).df()
    add("event_to_log_precision", "all", precision.hit.sum(), len(precision),
        f"rebuilt events since {start} with a logged failure within {w} days")

    unmatched = (precision[~precision.hit]
                 .groupby("main_downtime_code", dropna=False).size()
                 .sort_values(ascending=False).rename("events").reset_index())
    unmatched["share_of_unmatched"] = unmatched.events / unmatched.events.sum()
    for r in unmatched.head(8).itertuples():
        add("unmatched_event_code", str(r.main_downtime_code), r.events, len(precision[~precision.hit]),
            "most common downtime code on events with no logged failure")
    return rows, unmatched


def main() -> pd.DataFrame:
    config.ensure_dirs()
    con = config.connect()
    register_fails(con)
    log_fails_cleaning(con)

    gap, alt = config.EVENT_GAP_DAYS, config.EVENT_GAP_DAYS_SENSITIVITY
    build_events(con, gap, "events")
    con.execute(f"copy events to '{config.EVENTS.as_posix()}' (format parquet)")
    rows, unmatched = validate(con, "events", gap)

    # The sensitivity build lives in its own table so it never overwrites
    # the events file that downstream steps read.
    build_events(con, alt, "events_alt")
    alt_rows, _ = validate(con, "events_alt", alt)

    val = pd.DataFrame(rows + alt_rows)
    # Summary rows have no numerator, so use the nullable integer type to keep
    # counts as whole numbers in the CSV.
    val[["numerator", "denominator"]] = val[["numerator", "denominator"]].astype("Int64")
    val_path = config.OUTPUTS / "event_validation.csv"
    config.write_csv(val, val_path)

    def get(df, gap_days, metric, group="all"):
        r = df[(df.gap_days == gap_days) & (df.metric == metric) & (df.group == group)]
        return r.value.iloc[0]

    print(f"events: EVENT_GAP_DAYS={gap}")
    print(f"  events {get(val, gap, 'events'):,.0f} on {get(val, gap, 'event_wells'):,.0f} wells; "
          f"down days median {get(val, gap, 'median_down_days'):.0f}, "
          f"mean {get(val, gap, 'mean_down_days'):.1f}, total {get(val, gap, 'total_down_days'):,.0f}")
    rec = val[(val.gap_days == gap) & (val.metric == "log_to_event_recall")]
    print("  log -> event recall (within "
          f"{config.MATCH_WINDOW_DAYS} days, since {config.VALIDATION_START}):")
    for r in rec.itertuples():
        print(f"    {r.group:<14} {r.numerator:>6,} / {r.denominator:<6,} = {r.value:.1%}")
    p = val[(val.gap_days == gap) & (val.metric == "event_to_log_precision")].iloc[0]
    print(f"  event -> log precision: {p.numerator:,} / {p.denominator:,} = {p.value:.1%}")
    print("  top downtime codes on unmatched events:")
    for r in unmatched.head(8).itertuples():
        print(f"    {r.events:>5,}  {r.share_of_unmatched:5.1%}  {r.main_downtime_code}")
    print(f"  sensitivity EVENT_GAP_DAYS={alt}: events {get(val, alt, 'events'):,.0f}, "
          f"recall {get(val, alt, 'log_to_event_recall'):.1%}, "
          f"precision {get(val, alt, 'event_to_log_precision'):.1%}")
    print(f"events: wrote {config.rel(config.EVENTS)} and {config.rel(val_path)}")
    return val


if __name__ == "__main__":
    main()
