"""Step 2: clean the daily production file.

Writes
    outputs/wells/daily_clean.parquet  one row per well-day, every well kept
    outputs/cleaning_log.csv           one row per rule with the rows it touched

No rows are filtered out except exact duplicates. Downstream steps choose
their own population with the `in_scope` flag.

Daily derived columns (is_producing, water_cut, oil_rate_m3d,
rated_throughput_m3d, pump_utilisation) are defined here once so every later
step uses the same definition. Model features do not belong here.
"""
from __future__ import annotations

import time

import pandas as pd

from src import config

PLACEHOLDER_COLS = [
    "tubing_pressure_kpa",
    "casing_pressure_kpa",
    "injection_pressure_kpa",
    "hydraulic_pressure_psi",
    "pcp_pump_rod_torque_ftlb",
]
VOLUME_COLS = ["oil_produced_m3", "water_produced_m3", "gas_produced_e3m3"]

# A downhole day is identified from the downtime codes. Disposal-well repairs
# carry a "Downhole Equipment" code but are not a producing pump failure, so
# they are removed here rather than in every downstream query.
IS_DOWNHOLE_SQL = """(
    hours_down > 0
    and (downtime_code_2 in ('Downhole Equipment', 'Downhole')
         or downtime_code_1 like 'Downhole%')
    and coalesce(downtime_code_1, '') not ilike '%disposal well%'
)"""


def _in_scope_sql() -> str:
    # Scope is set per row from primary_fluid_type and flow_direction, not per
    # well, so days a well spent as an injector or a gas well stay out even if
    # the same uwi produced heavy oil at another time.
    return (f"(primary_fluid_type = '{config.SCOPE_FLUID}' "
            f"and flow_direction = '{config.SCOPE_FLOW}')")


def register_views(con) -> None:
    """Create the views that clean the raw daily table step by step."""
    p = config.PLACEHOLDER_MIN

    # The 29 duplicates are identical in every column, so `distinct` removes
    # them without having to choose which copy to keep.
    con.execute("create or replace view daily_dedup as select distinct * from daily")

    placeholder_exprs = ",\n        ".join(
        f"case when {c} >= {p} then null else {c} end as {c}" for c in PLACEHOLDER_COLS
    )
    # Negative volumes are allocation corrections, not physical production.
    # Nulling them keeps them out of rates and water cut without inventing a value.
    volume_exprs = ",\n        ".join(
        f"case when {c} < 0 then null else {c} end as {c}" for c in VOLUME_COLS
    )

    con.execute(f"""
    create or replace view daily_nulled as
    select
        * exclude ({", ".join(PLACEHOLDER_COLS + VOLUME_COLS)},
                   pcp_pump_rpm, pcp_pump_electric_current_a, pump_efficiency_pct,
                   surface_latitude, surface_longitude),
        surface_latitude::double as surface_latitude,
        surface_longitude::double as surface_longitude,
        {placeholder_exprs},
        {volume_exprs},
        -- RPM is stored as DECIMAL(5,0); casting to double avoids decimal
        -- overflow when it is multiplied by displacement below.
        case when pcp_pump_rpm > {config.MAX_RPM} then null
             else pcp_pump_rpm::double end as pcp_pump_rpm,
        case when pcp_pump_electric_current_a > {config.MAX_MOTOR_CURRENT_A} then null
             else pcp_pump_electric_current_a end as pcp_pump_electric_current_a,
        case when pump_efficiency_pct between 0 and {config.MAX_PUMP_EFFICIENCY}
             then pump_efficiency_pct end as pump_efficiency_pct
    from daily_dedup
    """)

    con.execute(f"""
    create or replace view daily_clean as
    select
        *,
        {_in_scope_sql()} as in_scope,
        {IS_DOWNHOLE_SQL} as is_downhole,
        -- The run clock counts producing days only, so a well shut in for
        -- market reasons does not look like a long healthy run.
        coalesce(hours_on_prod, 0) > 0 as is_producing,
        water_produced_m3 / nullif(oil_produced_m3 + water_produced_m3, 0) as water_cut,
        -- Null on non-producing days so averages are not dragged down by zeros.
        case when coalesce(hours_on_prod, 0) > 0 then oil_produced_m3 end as oil_rate_m3d,
        -- Handbook formula. The reported throughput column is this same
        -- rating, recomputed here from the cleaned RPM.
        pcp_pump_displacement_m3d100rpm * pcp_pump_rpm / 100 as rated_throughput_m3d,
        (case when coalesce(hours_on_prod, 0) > 0 then oil_produced_m3 end)
            / nullif(pcp_pump_displacement_m3d100rpm * pcp_pump_rpm / 100, 0) as pump_utilisation
    from daily_nulled
    """)


def build_log(con) -> pd.DataFrame:
    """Count the rows each rule touched, measured on the de-duplicated table."""
    raw_rows = con.sql("select count(*) from daily").fetchall()[0][0]
    dedup_rows = con.sql("select count(*) from daily_dedup").fetchall()[0][0]

    p = config.PLACEHOLDER_MIN
    checks = {
        **{f"placeholder_{c}": f"{c} >= {p}" for c in PLACEHOLDER_COLS},
        **{f"negative_{c}": f"{c} < 0" for c in VOLUME_COLS},
        "rpm_too_high": f"pcp_pump_rpm > {config.MAX_RPM}",
        "current_too_high": f"pcp_pump_electric_current_a > {config.MAX_MOTOR_CURRENT_A}",
        "efficiency_out_of_range": (f"pump_efficiency_pct is not null and not "
                                    f"(pump_efficiency_pct between 0 and {config.MAX_PUMP_EFFICIENCY})"),
        "in_scope": _in_scope_sql(),
        "is_downhole": IS_DOWNHOLE_SQL,
        "is_producing": "coalesce(hours_on_prod, 0) > 0",
    }
    # One scan with many count filters is much faster than one query per rule.
    sel = ", ".join(f"count(*) filter (where {cond}) as {name}" for name, cond in checks.items())
    counts = con.sql(f"select {sel} from daily_dedup").df().iloc[0]

    derived = con.sql("""
        select count(water_cut) water_cut, count(oil_rate_m3d) oil_rate_m3d,
               count(rated_throughput_m3d) rated_throughput_m3d,
               count(pump_utilisation) pump_utilisation
        from daily_clean
    """).df().iloc[0]

    rows = [
        ("load", "raw daily_production", "*", raw_rows, raw_rows, "rows read"),
        ("dedupe", "drop exact duplicate rows", "*", raw_rows - dedup_rows, dedup_rows,
         "duplicates are identical in every column"),
    ]
    for c in PLACEHOLDER_COLS:
        rows.append(("null", f"placeholder >= {p:,.0f} set to null", c,
                     int(counts[f"placeholder_{c}"]), dedup_rows, "sensor placeholder, not a reading"))
    for c in VOLUME_COLS:
        rows.append(("null", "negative volume set to null", c,
                     int(counts[f"negative_{c}"]), dedup_rows, "allocation correction, not production"))
    rows += [
        ("null", f"rpm > {config.MAX_RPM:,.0f} set to null", "pcp_pump_rpm",
         int(counts["rpm_too_high"]), dedup_rows, "physically implausible"),
        ("null", f"current > {config.MAX_MOTOR_CURRENT_A:,.0f} A set to null", "pcp_pump_electric_current_a",
         int(counts["current_too_high"]), dedup_rows, "physically implausible"),
        ("null", f"efficiency outside 0..{config.MAX_PUMP_EFFICIENCY} set to null", "pump_efficiency_pct",
         int(counts["efficiency_out_of_range"]), dedup_rows, "derived ratio broken by allocation noise"),
        ("flag", f"primary_fluid_type = {config.SCOPE_FLUID} and flow_direction = {config.SCOPE_FLOW}",
         "in_scope", int(counts["in_scope"]), dedup_rows, "rows flagged true; no rows removed"),
        ("flag", "hours_down > 0 and downhole downtime code, excluding disposal-well repairs",
         "is_downhole", int(counts["is_downhole"]), dedup_rows, "rows flagged true; all wells"),
        ("derive", "hours_on_prod > 0", "is_producing", int(counts["is_producing"]), dedup_rows,
         "rows flagged true"),
        ("derive", "water / (oil + water), null when the sum is 0", "water_cut",
         int(derived["water_cut"]), dedup_rows, "rows with a non-null value"),
        ("derive", "oil_produced_m3 on producing days, else null", "oil_rate_m3d",
         int(derived["oil_rate_m3d"]), dedup_rows, "rows with a non-null value"),
        ("derive", "displacement * rpm / 100", "rated_throughput_m3d",
         int(derived["rated_throughput_m3d"]), dedup_rows, "rows with a non-null value"),
        ("derive", "oil_rate_m3d / rated_throughput_m3d", "pump_utilisation",
         int(derived["pump_utilisation"]), dedup_rows, "rows with a non-null value"),
        ("write", "daily_clean.parquet", "*", dedup_rows, dedup_rows, "no rows filtered"),
    ]
    return pd.DataFrame(rows, columns=["step", "rule", "column", "rows_affected", "rows_after", "note"])


def main() -> pd.DataFrame:
    t0 = time.time()
    config.ensure_dirs()
    con = config.connect()
    register_views(con)

    # Sorting by well and date lets later window functions read the file in
    # order and keeps parquet row-group statistics useful for uwi filters.
    con.execute(f"""
        copy (select * from daily_clean order by uwi, production_date)
        to '{config.DAILY_CLEAN.as_posix()}' (format parquet)
    """)
    log = build_log(con)
    log_path = config.OUTPUTS / "cleaning_log.csv"
    log.to_csv(log_path, index=False)

    print(log.to_string(index=False))
    print(f"clean: wrote {config.rel(config.DAILY_CLEAN)} and {config.rel(log_path)} in {time.time() - t0:.1f}s")
    return log


if __name__ == "__main__":
    main()
