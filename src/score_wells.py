"""Step 8: score every well currently in a run and write the watchlist.

Uses the model saved by train_eval.py and the shared scoring functions in
scoring.py, so each well's risk is computed exactly as in the validated
independent check: the chance of failing within the next HORIZON_DAYS
producing days, given the producing days the well has already survived.

Writes outputs/wells/well_watchlist.csv. It lists real wells, so it is
gitignored and must never be committed or shown in a public screenshot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.scoring import beyond_model_range, load_model, open_runs_at, score_open_runs

WATCHLIST = config.WELLS_OUT / "well_watchlist.csv"


def main() -> pd.DataFrame:
    config.ensure_dirs()
    con = config.connect()
    bundle = load_model()
    runs = pd.read_parquet(config.RUNS)

    # Score as of the last day in the data. A well is on the watchlist only if
    # it is in a run on that day, which means it produced on the final day.
    as_of = pd.Timestamp(con.sql(f"""
        select max(production_date) from read_parquet('{config.DAILY_CLEAN.as_posix()}') where in_scope
    """).fetchall()[0][0])
    wl = open_runs_at(con, runs, as_of)
    wl["risk_90"] = score_open_runs(bundle, wl)
    # Wells further into a run than any training run cannot be scored
    # honestly, so they are flagged with no risk instead of a false 0%.
    wl["beyond_model_range"] = beyond_model_range(bundle, wl.elapsed_prod_days)
    wl.loc[wl.beyond_model_range, "risk_90"] = np.nan

    wl = wl.sort_values("risk_90", ascending=False, kind="stable", na_position="last").reset_index(drop=True)
    wl["rank"] = (wl.index + 1).where(~wl.beyond_model_range)
    wl["as_of"] = as_of.date()
    cols = ["rank", "uwi", "area", "as_of", "start_d", "elapsed_prod_days", "risk_90", "beyond_model_range",
            "fails_prior_365", "prev_gap_days", "oil_pre30", "wcut_pre30", "rpm_pre30",
            "workover", "rodtruck", "new_string", "repair_cost"]
    wl = wl[cols].rename(columns={"start_d": "run_start"})
    config.write_csv(wl, WATCHLIST)

    # Only aggregates are printed, so console logs and notebook output never
    # show a well identifier.
    h, k = bundle["horizon_days"], config.TOP_K
    print(f"score_wells: {len(wl):,} wells in a run on {as_of.date()}, model trained to {bundle['time_cut']}")
    print(f"  {int(wl.beyond_model_range.sum())} wells are past the model support limit "
          f"({bundle['support_limit_days']:.0f} producing days) and are flagged unscored")
    print(f"  mean {h}-day risk {wl.risk_90.mean():.2f}; top {k} mean {wl.risk_90.head(k).mean():.2f}; "
          f"expected failures in the next {h} producing days {wl.risk_90.sum():,.0f}")
    print("  wells by risk band: " + ", ".join(
        f"{lbl} {n:,}" for lbl, n in pd.cut(wl.risk_90, [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True,
                                           labels=["<25%", "25-50%", "50-75%", "75%+"]).value_counts(sort=False).items()))
    print(f"  top {k} by area: " + ", ".join(f"{a} {n}" for a, n in wl.head(k).area.value_counts().items()))
    print(f"score_wells: wrote {config.rel(WATCHLIST)}")
    return wl


if __name__ == "__main__":
    main()
