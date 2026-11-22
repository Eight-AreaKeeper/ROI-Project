"""Step 6: survival analysis of run life, and evaluation on later data.

Duration is `prod_days` and the outcome is `event` (1 failed, 0 still running
when the data ended). Survival methods use the censored runs as "lasted at
least this long" instead of discarding them or calling them successes.

Writes
    outputs/km_curves.csv       Kaplan-Meier curves by group
    outputs/km_summary.csv      median run life and log-rank p per group
    outputs/cox_results.csv     hazard ratios per 1 SD, with PH test
    outputs/model_eval.csv      C-index, AUC, Brier, top-K hits, independent check
    outputs/calibration.csv     predicted vs observed 90-day risk by decile
    outputs/wells/model.pkl     the evaluated model and everything needed to score with it
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from src import config
from src.scoring import (BINARY_FEATURES, COX_FEATURES, LOG_FEATURES, Prep, beyond_model_range, open_runs_at,
                         risk_within, score_open_runs, support_limit)

EXPECTED = {
    "train_runs": 6_917, "train_events": 6_041, "test_runs": 3_446, "test_events": 2_641,
    "hr_fails_prior_365": 1.24, "hr_oil_pre30": 1.11, "hr_rpm_pre30": 1.06,
    "hr_wcut_pre30": 1.05, "hr_disp": 1.04, "hr_util_pre30": 1.02,
    "hr_prev_gap_days": 0.90,
    # Binary features are reported 0 to 1 (with versus without), not per SD.
    "hr_new_string": 0.432, "hr_workover": 0.546,
    "cindex_cox": 0.643, "cindex_baseline_a": 0.604, "cindex_baseline_b": 0.595,
    "auc90_cox": 0.720, "auc90_baseline_a": 0.658,
    "brier90_cox": 0.207, "brier90_base_rate": 0.241,
    "indep_wells": 806, "indep_base_rate": 0.32,
    # Revised from 0.706 by team decision: risk is scored conditional on the
    # producing days already survived, which is the method the watchlist uses.
    "indep_auc_cox": 0.72, "indep_auc_baseline_a": 0.674,
    "indep_top50_cox": 35, "indep_top50_baseline_a": 27.6,
}
# Differences reviewed by the team and accepted, with the reason. They are
# still printed, marked "accepted" instead of "DIFFERS".
ACCEPTED = {
    "hr_workover": "0.567 is the exact 0-to-1 conversion of the fitted per-SD ratio (0.768, SD 0.464)",
    "indep_top50_cox": "moved with the agreed check boundaries (docs/methods.md)",
    "indep_top50_baseline_a": "moved with the agreed check boundaries; mean over tie orders, range printed above",
}
EXPECTED_CALIBRATION = [(0.29, 0.25), (0.38, 0.40), (0.45, 0.42), (0.51, 0.51), (0.56, 0.60),
                        (0.61, 0.67), (0.66, 0.68), (0.72, 0.75), (0.77, 0.77), (0.87, 0.89)]


# ---------------------------------------------------------------- data

def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_parquet(config.RUNS)
    cut = pd.Timestamp(config.TIME_CUT)
    train = runs[runs.start_d < cut].copy()
    test = runs[runs.start_d >= cut].copy()

    # A training run still open at the cut must be censored there: its ending
    # happened after the cut, which the model is not allowed to know.
    open_at_cut = train.end_d > cut
    frac = (cut - train.start_d).dt.days / train.cal_days
    # Producing days are scaled by the calendar share before the cut, because
    # the daily table is not re-read here. Rounding keeps whole days.
    train.loc[open_at_cut, "prod_days"] = np.round(train.loc[open_at_cut, "prod_days"] * frac[open_at_cut])
    train.loc[open_at_cut, "event"] = 0
    train = train[train.prod_days > 0]
    return train, test


# ---------------------------------------------------------------- Kaplan-Meier

def km_groups(runs: pd.DataFrame) -> dict[str, pd.Series]:
    fp = runs.fails_prior_365
    rpm_q = pd.qcut(runs.rpm_pre30, 4)
    repair = np.select(
        [runs.workover == 1, runs.rodtruck == 1],
        ["workover", "rod truck only"], default="no linked job")
    return {
        "fails_prior_365": pd.Series(np.select([fp == 0, fp == 1], ["0", "1"], "2+"), index=runs.index),
        "rpm_quartile": rpm_q.astype(str).where(rpm_q.notna()),
        "water_cut_band": pd.cut(runs.wcut_pre30, [-np.inf, 0.7, 0.9, np.inf], right=False,
                                 labels=["<0.7", "0.7-0.9", "0.9+"]).astype(str).where(runs.wcut_pre30.notna()),
        "repair_class": pd.Series(repair, index=runs.index),
        "area": runs.area,
    }


def kaplan_meier(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves, summary = [], []
    for var, groups in km_groups(runs).items():
        ok = groups.notna()
        d, g = runs[ok], groups[ok]
        lr = multivariate_logrank_test(d.prod_days, g, d.event)
        for name in sorted(g.unique()):
            m = g == name
            km = KaplanMeierFitter().fit(d.prod_days[m], d.event[m])
            sf = km.survival_function_.iloc[:, 0]
            ci = km.confidence_interval_survival_function_
            at_risk = km.event_table.at_risk.reindex(sf.index)
            curves.append(pd.DataFrame({
                "group_var": var, "group": name, "t_days": sf.index,
                "survival": sf.values, "ci_low": ci.iloc[:, 0].values, "ci_high": ci.iloc[:, 1].values,
                "n_at_risk": at_risk.values}))
            summary.append({"group_var": var, "group": name, "n_runs": int(m.sum()),
                            "failures": int(d.event[m].sum()),
                            "median_prod_days": km.median_survival_time_,
                            "logrank_p": lr.p_value})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary)


# ---------------------------------------------------------------- Cox

def fit_cox(train: pd.DataFrame, prep: Prep) -> tuple[CoxPHFitter, pd.DataFrame]:
    x = prep.transform(train)
    df = pd.concat([x, train[["prod_days", "event", "uwi"]]], axis=1)
    # cluster_col gives robust standard errors: one well contributes many
    # runs, and those runs are not independent of each other.
    cph = CoxPHFitter(penalizer=config.COX_PENALIZER)
    cph.fit(df, duration_col="prod_days", event_col="event", cluster_col="uwi")
    return cph, df


def cox_table(cph: CoxPHFitter, df: pd.DataFrame, prep: Prep) -> pd.DataFrame:
    s = cph.summary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ph = proportional_hazard_test(cph, df, time_transform="rank")
    ph_p = ph.summary["p"]
    out = pd.DataFrame({
        "feature": s.index,
        "hazard_ratio": s["exp(coef)"].values,
        "ci_low": s["exp(coef) lower 95%"].values,
        "ci_high": s["exp(coef) upper 95%"].values,
        "p_value": s["p"].values,
        "ph_test_p": ph_p.reindex(s.index).values,
        # SD on the model scale (after log1p for logged features), so a reader
        # can convert a per-SD ratio back to the original unit.
        "sd": prep.sd.reindex(s.index).values,
    })
    out["transform"] = np.where(out.feature.isin(LOG_FEATURES), "log1p", "none")
    out["scale"] = np.where(out.feature.isin(BINARY_FEATURES), "0 to 1", "per SD")
    # A binary feature is not scaled, so it has no SD to convert back with.
    out.loc[out.scale == "0 to 1", "sd"] = np.nan
    out["significant"] = out.p_value < 0.05
    # Features that fail the PH test are kept: their ratio is then an average
    # effect over the run, which the Method page states.
    out["ph_violation"] = out.ph_test_p < config.PH_TEST_ALPHA
    return out


# ---------------------------------------------------------------- evaluation

def horizon_labels(df: pd.DataFrame, h: int) -> pd.Series:
    """1 failed within h producing days, 0 survived past h, NaN censored before h."""
    y = pd.Series(np.nan, index=df.index)
    y[(df.event == 1) & (df.prod_days <= h)] = 1
    y[df.prod_days > h] = 0
    return y


def topk_hits(score: pd.Series, y: pd.Series, k: int) -> dict:
    """Hits in the top k, with ties at the cut-off handled fairly.

    A score with many ties (such as a count of past failures) has no single
    top k: which tied wells make the list depends on sort order. Instead of
    picking one order, report the mean over all tie orders and the range.
    """
    order = score.sort_values(ascending=False)
    threshold = order.iloc[k - 1]
    above = score > threshold
    tied = score == threshold
    slots = k - int(above.sum())
    hits_above = int(y[above].sum())
    tied_hits, n_tied = int(y[tied].sum()), int(tied.sum())
    return {
        # Every tie order is equally likely, so each tied well fills a slot
        # with probability slots / n_tied.
        "mean": hits_above + slots * tied_hits / n_tied,
        "min": hits_above + max(0, slots - (n_tied - tied_hits)),
        "max": hits_above + min(slots, tied_hits),
        "n_tied": n_tied, "slots": slots,
    }


def topk_note(t: dict, base: str) -> str:
    if t["n_tied"] <= 1:
        return base
    return (f"{base}; mean over tie orders: {t['n_tied']} tied at the cut-off for {t['slots']} slots, "
            f"range {t['min']} to {t['max']} (ci_low/ci_high hold the range)")


def evaluate(cph, prep, train, test) -> tuple[list[dict], pd.DataFrame, dict]:
    h, k = config.HORIZON_DAYS, config.TOP_K
    rows = []

    def add(model, metric, value, n, ci=(None, None), note=""):
        rows.append({"model": model, "metric": metric, "value": float(value),
                     "ci_low": ci[0], "ci_high": ci[1], "n": int(n), "note": note})

    xt = prep.transform(test)
    test = test.assign(
        cox_risk=cph.predict_partial_hazard(xt).to_numpy(),
        cox_p90=risk_within(cph, xt, h),
        base_a=test.fails_prior_365.astype(float),
        # Missing gap means a first run; the training median stands in so the
        # baseline can still rank it.
        base_b=-test.prev_gap_days.fillna(train.prev_gap_days.median()),
    )

    # C-index: higher risk should mean a shorter run, hence the minus sign.
    def cidx(d, col):
        return concordance_index(d.prod_days, -d[col], d.event)

    c = {m: cidx(test, col) for m, col in
         [("cox", "cox_risk"), ("baseline_a", "base_a"), ("baseline_b", "base_b")]}

    # Resample wells, not runs: runs on one well are correlated, and resampling
    # runs would make the interval look tighter than it is.
    rng = np.random.default_rng(config.RANDOM_SEED)
    by_well = {w: g for w, g in test.groupby("uwi")}
    wells = np.array(list(by_well))
    diffs = []
    for _ in range(config.BOOTSTRAP_DRAWS):
        d = pd.concat([by_well[w] for w in rng.choice(wells, len(wells), replace=True)])
        diffs.append(cidx(d, "cox_risk") - cidx(d, "base_a"))
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    add("cox", "c_index", c["cox"], len(test))
    add("baseline_a", "c_index", c["baseline_a"], len(test), note="rank by fails_prior_365 (operator rule today)")
    add("baseline_b", "c_index", c["baseline_b"], len(test), note="rank by -prev_gap_days")
    add("cox_minus_baseline_a", "c_index_diff", c["cox"] - c["baseline_a"], len(wells), (lo, hi),
        f"{config.BOOTSTRAP_DRAWS} bootstrap draws resampling wells")

    # 90-day metrics use runs whose 90-day outcome is known: a run censored
    # before day 90 has no label and is left out.
    y = horizon_labels(test, h)
    lab = test[y.notna()].assign(y=y[y.notna()])
    km_train = KaplanMeierFitter().fit(train.prod_days, train.event)
    base_rate = 1 - float(km_train.predict(h))
    note = f"runs with a known {h}-day outcome"
    add("cox", f"auc_{h}", roc_auc_score(lab.y, lab.cox_p90), len(lab), note=note)
    add("baseline_a", f"auc_{h}", roc_auc_score(lab.y, lab.base_a), len(lab), note=note)
    add("baseline_b", f"auc_{h}", roc_auc_score(lab.y, lab.base_b), len(lab), note=note)
    add("cox", f"brier_{h}", np.mean((lab.cox_p90 - lab.y) ** 2), len(lab), note=note)
    add("baseline_c_km", f"brier_{h}", np.mean((base_rate - lab.y) ** 2), len(lab),
        note=f"{note}; predicts the training Kaplan-Meier rate {base_rate:.3f} for every run")
    for model, col in [("cox", "cox_p90"), ("baseline_a", "base_a"), ("baseline_b", "base_b")]:
        t = topk_hits(lab[col], lab.y, k)
        add(model, f"top{k}_hits", t["mean"], k, (t["min"], t["max"]),
            topk_note(t, f"failed within {h} producing days"))

    calib = calibration(test, h)
    return rows, calib, {"c": c, "diff_ci": (lo, hi), "base_rate": base_rate}


def calibration(test: pd.DataFrame, h: int) -> pd.DataFrame:
    test = test.assign(decile=pd.qcut(test.cox_p90.rank(method="first"), config.CALIBRATION_BINS,
                                      labels=False) + 1)
    out = []
    for d, g in test.groupby("decile"):
        # Observed risk comes from Kaplan-Meier within the decile, so runs
        # censored before day 90 still count for the time they were watched.
        km = KaplanMeierFitter().fit(g.prod_days, g.event)
        out.append({"model": "cox", "decile": int(d), "predicted": g.cox_p90.mean(),
                    "observed": 1 - float(km.predict(h)), "n": len(g)})
    return pd.DataFrame(out)


def independent_check(con, bundle: dict, runs: pd.DataFrame) -> list[dict]:
    """Score wells in an open run on the check date against the official log.

    Scoring goes through the saved bundle and the shared scoring functions,
    the same path score_wells.py uses, so this validates the watchlist itself.
    """
    day, h, k = pd.Timestamp(config.INDEPENDENT_CHECK_DATE), config.HORIZON_DAYS, config.TOP_K
    open_runs = open_runs_at(con, runs, day)

    # Labels come from the official failure log, which neither the model nor
    # the event rebuild used, so this is not a circular check. The window
    # includes day h itself (see docs/methods.md).
    logged = con.sql(f"""
        select distinct uwi from fails
        where failure_start_date >= date '{day.date()}'
          and failure_start_date <= date '{day.date()}' + interval {h} day
    """).df().uwi
    open_runs["y"] = open_runs.uwi.isin(set(logged)).astype(int)
    open_runs["cox_p"] = score_open_runs(bundle, open_runs)
    open_runs["base_a"] = open_runs.fails_prior_365.astype(float)

    note = "official log, unseen labels"

    def row(model, metric, value, n, ci=(None, None), extra=""):
        return {"model": model, "metric": metric, "value": float(value), "ci_low": ci[0],
                "ci_high": ci[1], "n": int(n), "note": note + (f"; {extra}" if extra else "")}

    # All open wells stay in the check so it matches the agreed population.
    # The count past the support limit is reported so readers know how many
    # of these scores the watchlist would leave blank.
    n_beyond = int(beyond_model_range(bundle, open_runs.elapsed_prod_days).sum())
    rows = [row("independent_check", "wells", len(open_runs), len(open_runs)),
            row("independent_check", "wells_beyond_support", n_beyond, len(open_runs),
                extra="scored here, left blank on the watchlist"),
            row("independent_check", "base_rate", open_runs.y.mean(), len(open_runs))]
    for model, col in [("cox", "cox_p"), ("baseline_a", "base_a")]:
        rows.append(row(model, f"indep_auc_{h}", roc_auc_score(open_runs.y, open_runs[col]), len(open_runs)))
        t = topk_hits(open_runs[col], open_runs.y, k)
        rows.append(row(model, f"indep_top{k}_hits", t["mean"], k, (t["min"], t["max"]),
                        topk_note(t, "").lstrip("; ")))
    return rows


# ---------------------------------------------------------------- reporting

def compare(got: dict) -> list[str]:
    print(f"\nCheck against the brief (flag when off by more than {config.CHECK_TOLERANCE:.0%})")
    flagged = []
    for key, exp in EXPECTED.items():
        diff = abs(got[key] - exp) / abs(exp)
        status = "ok" if diff <= config.CHECK_TOLERANCE else ("accepted" if key in ACCEPTED else "DIFFERS")
        if status == "DIFFERS":
            flagged.append(key)
        reason = f"  {ACCEPTED[key]}" if status == "accepted" else ""
        print(f"    {status:<8} {key:<24} computed {got[key]:>10,.4f}  expected {exp:>10,.4f}  ({diff:.2%}){reason}")
    print("    all values within tolerance" if not flagged else f"    differs: {', '.join(flagged)}")
    return flagged


def main() -> dict:
    config.ensure_dirs()
    con = config.connect()
    train, test = load_split()
    print(f"train_eval: train {len(train):,} runs ({int(train.event.sum()):,} events), "
          f"test {len(test):,} runs ({int(test.event.sum()):,} events), cut {config.TIME_CUT}")

    # Descriptive curves use every run, uncensored at the cut: they describe
    # the fleet, they are not part of the evaluation.
    all_runs = pd.read_parquet(config.RUNS)
    km, km_sum = kaplan_meier(all_runs)
    config.write_csv(km, config.OUTPUTS / "km_curves.csv")
    config.write_csv(km_sum, config.OUTPUTS / "km_summary.csv")
    print("\nKaplan-Meier: median producing days per group, log-rank p per variable")
    for var, g in km_sum.groupby("group_var", sort=False):
        print(f"  {var} (log-rank p = {g.logrank_p.iloc[0]:.2g})")
        for r in g.itertuples():
            print(f"    {r.group:<26} n={r.n_runs:>5,}  median={r.median_prod_days:>6.0f}")

    prep = Prep().fit(train)
    cph, train_df = fit_cox(train, prep)
    cox = cox_table(cph, train_df, prep)
    config.write_csv(cox, config.OUTPUTS / "cox_results.csv")
    print("\nCox hazard ratios, per 1 SD or 0 to 1 for yes/no features (above 1 = shorter runs)")
    for r in cox.sort_values("hazard_ratio", ascending=False).itertuples():
        flags = ("" if r.significant else "  not significant") + ("  PH violated" if r.ph_violation else "")
        print(f"    {r.feature:<16} {r.scale:<7} HR {r.hazard_ratio:5.3f}  [{r.ci_low:.2f}, {r.ci_high:.2f}]  "
              f"p={r.p_value:.2g}  ph_p={r.ph_test_p:.2g}{flags}")
    print(f"    {int(cox.ph_violation.sum())} feature(s) fail the PH test at p<{config.PH_TEST_ALPHA}: "
          f"{', '.join(cox.feature[cox.ph_violation])}")

    # The scorer must use this exact fitted model; refitting elsewhere would
    # make the watchlist disagree with the evaluation without anyone noticing.
    bundle = {"model": cph, **prep.state(), "features": COX_FEATURES,
              "log_features": LOG_FEATURES, "time_cut": config.TIME_CUT,
              "horizon_days": config.HORIZON_DAYS,
              "support_limit_days": support_limit(train.prod_days, config.MIN_AT_RISK)}
    joblib.dump(bundle, config.MODEL)
    print(f"\nModel support: scored up to {bundle['support_limit_days']:.0f} producing days "
          f"(at least {config.MIN_AT_RISK} training runs still observed)")

    rows, calib, ev = evaluate(cph, prep, train, test)
    rows += independent_check(con, bundle, all_runs)
    model_eval = pd.DataFrame(rows)
    config.write_csv(model_eval, config.OUTPUTS / "model_eval.csv")
    config.write_csv(calib, config.OUTPUTS / "calibration.csv")

    print("\nTest-set scores")
    for r in model_eval.itertuples():
        label = "range" if "hits" in r.metric else "95% CI"
        ci = f"  {label} [{r.ci_low:.3f}, {r.ci_high:.3f}]" if pd.notna(r.ci_low) else ""
        print(f"    {r.model:<22} {r.metric:<16} {r.value:8.3f}  n={r.n:<5,}{ci}  {r.note}")
    print("\nCalibration (predicted / observed 90-day risk; brief's values in brackets)")
    for r, (ep, eo) in zip(calib.itertuples(), EXPECTED_CALIBRATION):
        print(f"    decile {r.decile:>2}: {r.predicted:.2f} / {r.observed:.2f}   [{ep:.2f} / {eo:.2f}]  n={r.n}")

    beats = ev["c"]["cox"] > ev["c"]["baseline_a"] and ev["diff_ci"][0] > 0
    print(f"\nShip decision: Cox {'BEATS' if beats else 'DOES NOT BEAT'} baseline A "
          f"(C-index difference CI {ev['diff_ci'][0]:.3f} to {ev['diff_ci'][1]:.3f})")

    me = model_eval.set_index(["model", "metric"]).value
    h, k = config.HORIZON_DAYS, config.TOP_K
    hr = cox.set_index("feature").hazard_ratio
    got = {
        "train_runs": len(train), "train_events": train.event.sum(),
        "test_runs": len(test), "test_events": test.event.sum(),
        **{f"hr_{f}": hr[f] for f in ["fails_prior_365", "oil_pre30", "rpm_pre30", "wcut_pre30", "disp",
                                       "util_pre30", "prev_gap_days", "new_string", "workover"]},
        "cindex_cox": me["cox", "c_index"], "cindex_baseline_a": me["baseline_a", "c_index"],
        "cindex_baseline_b": me["baseline_b", "c_index"],
        "auc90_cox": me["cox", f"auc_{h}"], "auc90_baseline_a": me["baseline_a", f"auc_{h}"],
        "brier90_cox": me["cox", f"brier_{h}"], "brier90_base_rate": me["baseline_c_km", f"brier_{h}"],
        "indep_wells": me["independent_check", "wells"], "indep_base_rate": me["independent_check", "base_rate"],
        "indep_auc_cox": me["cox", f"indep_auc_{h}"], "indep_auc_baseline_a": me["baseline_a", f"indep_auc_{h}"],
        "indep_top50_cox": me["cox", f"indep_top{k}_hits"],
        "indep_top50_baseline_a": me["baseline_a", f"indep_top{k}_hits"],
    }
    compare(got)
    print(f"\ntrain_eval: wrote km_curves.csv, km_summary.csv, cox_results.csv, model_eval.csv, "
          f"calibration.csv and {config.rel(config.MODEL)}")
    return got


if __name__ == "__main__":
    main()
