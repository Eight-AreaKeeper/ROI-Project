"""Step 9: replay the watchlist on past dates and check it against the official log.

For each check date after the training cut, every well in an open run is
scored exactly as in the independent check (train_eval.score_check_date),
then labelled by whether the official log records a failure within the next
HORIZON_DAYS. Repeating this on several dates shows whether the result holds
over time rather than on one lucky date.

Writes outputs/wells/replay.parquet (well level, gitignored) and
outputs/replay_summary.csv (aggregates only).
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.scoring import beyond_model_range, load_model
from src.train_eval import score_check_date, topk_hits

CHECK_DATES = ["2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01"]
REPLAY = config.WELLS_OUT / "replay.parquet"
SUMMARY = config.OUTPUTS / "replay_summary.csv"


def check_dates_are_valid(con) -> None:
    """Each date must be after the training cut and have a full label window."""
    last_log = pd.Timestamp(con.sql("select max(failure_start_date) from fails").fetchall()[0][0])
    for d in map(pd.Timestamp, CHECK_DATES):
        if d < pd.Timestamp(config.TIME_CUT):
            raise ValueError(f"{d.date()} is before the training cut {config.TIME_CUT}")
        if d + pd.Timedelta(days=config.HORIZON_DAYS) > last_log:
            raise ValueError(f"{d.date()} has no full {config.HORIZON_DAYS}-day window in the log")


def replay_date(con, bundle: dict, runs: pd.DataFrame, day: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    o = score_check_date(con, bundle, runs, day)
    o = o.sort_values("cox_p", ascending=False, kind="stable").reset_index(drop=True)
    wells = pd.DataFrame({
        "date": day.date(), "uwi": o.uwi, "rank": o.index + 1, "risk": o.cox_p, "failed": o.y,
        "beyond_support": beyond_model_range(bundle, o.elapsed_prod_days),
    })
    k = config.TOP_K
    rule = topk_hits(o.base_a, o.y, k)
    summary = {
        "date": day.date(), "wells": len(o), "base_rate": o.y.mean(),
        "model_top50_hits": topk_hits(o.cox_p, o.y, k)["mean"],
        "rule_top50_mean": rule["mean"], "rule_top50_min": rule["min"], "rule_top50_max": rule["max"],
        "random_top50": o.y.mean() * k,
    }
    return wells, summary


def check_against_eval(summary: pd.DataFrame) -> list[str]:
    """The independent-check date must reproduce model_eval.csv exactly."""
    ev = pd.read_csv(config.OUTPUTS / "model_eval.csv").set_index(["model", "metric"]).value
    k = config.TOP_K
    row = summary.set_index("date").loc[pd.Timestamp(config.INDEPENDENT_CHECK_DATE).date()]
    pairs = {
        "wells": (row.wells, ev[("independent_check", "wells")]),
        "base_rate": (row.base_rate, ev[("independent_check", "base_rate")]),
        "model_top50_hits": (row.model_top50_hits, ev[("cox", f"indep_top{k}_hits")]),
        "rule_top50_mean": (row.rule_top50_mean, ev[("baseline_a", f"indep_top{k}_hits")]),
    }
    bad = []
    for name, (got, exp) in pairs.items():
        ok = abs(got - exp) <= 1e-3
        bad += [] if ok else [name]
        print(f"    {'ok' if ok else 'DIFFERS':<8} {name:<18} replay {got:,.4f}  model_eval {exp:,.4f}")
    return bad


def main() -> pd.DataFrame:
    config.ensure_dirs()
    con = config.connect()
    check_dates_are_valid(con)
    bundle, runs = load_model(), pd.read_parquet(config.RUNS)

    wells, rows = [], []
    for d in CHECK_DATES:
        w, s = replay_date(con, bundle, runs, pd.Timestamp(d))
        wells.append(w)
        rows.append(s)
    wells = pd.concat(wells, ignore_index=True)
    summary = pd.DataFrame(rows)
    wells.to_parquet(REPLAY, index=False)
    config.write_csv(summary, SUMMARY)

    print("replay: top-50 hits on the official log, model vs today's rule")
    for r in summary.itertuples():
        print(f"  {r.date}  wells {r.wells:>4}  base rate {r.base_rate:.3f}  model {r.model_top50_hits:>4.0f}  "
              f"rule {r.rule_top50_mean:>5.1f} ({r.rule_top50_min:.0f} to {r.rule_top50_max:.0f})  "
              f"random {r.random_top50:>4.1f}")
    print(f"  check: {config.INDEPENDENT_CHECK_DATE} against model_eval.csv")
    bad = check_against_eval(summary)
    if bad:
        raise SystemExit(f"replay does not reproduce the independent check: {', '.join(bad)}")
    print(f"replay: wrote {config.rel(SUMMARY)} and {config.rel(REPLAY)}")
    return summary


if __name__ == "__main__":
    main()
