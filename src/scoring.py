"""Shared scoring code for the run-life model.

train_eval.py (the evaluation) and score_wells.py (the watchlist) both score
through these functions, so the watchlist always uses exactly the method that
was validated. Change scoring here, never in one of the callers.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src import config

# Features entering the Cox model, in the order they are saved with the model.
COX_FEATURES = [
    "fails_prior_365", "oil_pre30", "rpm_pre30", "wcut_pre30", "disp", "util_pre30",
    "torque_pre30", "prev_gap_days", "no_prev_gap", "prev_down_days",
    "workover", "new_string", "corod", "stick",
]
# Skewed positive features: logging them stops a few extreme wells from
# dominating the fit. log1p keeps zeros finite.
LOG_FEATURES = ["oil_pre30", "prev_gap_days", "prev_down_days"]
# Yes/no features stay as 0/1 so their hazard ratio reads as "with versus
# without", which is how the effect is quoted. Only continuous features are
# standardised.
BINARY_FEATURES = ["workover", "new_string", "corod", "stick", "has_job", "no_prev_gap"]


class Prep:
    """Fill-in and scaling fitted on training rows only, then applied to any rows."""

    def fit(self, df: pd.DataFrame) -> "Prep":
        x = self._raw(df)
        self.medians = x.median()
        x = x.fillna(self.medians)
        self.mean, self.sd = x.mean(), x.std(ddof=0)
        # A mean of 0 and SD of 1 leaves binary features unchanged by transform().
        binary = [c for c in COX_FEATURES if c in BINARY_FEATURES]
        self.mean[binary], self.sd[binary] = 0.0, 1.0
        return self

    @classmethod
    def from_state(cls, state: dict) -> "Prep":
        p = cls()
        p.medians, p.mean, p.sd = (pd.Series(state[k]) for k in ("medians", "mean", "sd"))
        return p

    @staticmethod
    def _raw(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        # A first run on a well has no previous gap. The indicator lets the
        # model treat it separately instead of pretending it was a median gap.
        x["no_prev_gap"] = x.prev_gap_days.isna().astype(float)
        for c in LOG_FEATURES:
            x[c] = np.log1p(x[c])
        return x[COX_FEATURES].astype(float)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._raw(df).fillna(self.medians)
        # Standardising makes every hazard ratio "per 1 SD", so effects on
        # different scales can be compared directly.
        return (x - self.mean) / self.sd

    def state(self) -> dict:
        return {"medians": self.medians.to_dict(), "mean": self.mean.to_dict(), "sd": self.sd.to_dict()}


def baseline_survival_at(cph, t) -> np.ndarray:
    """Baseline survival S0(t) as a step function, for any array of times."""
    s0 = cph.baseline_survival_.iloc[:, 0]
    idx = np.searchsorted(s0.index.to_numpy(), np.asarray(t, dtype=float), side="right") - 1
    vals = np.r_[1.0, s0.to_numpy()]
    return vals[idx + 1]


def risk_within(cph, x: pd.DataFrame, horizon: float, elapsed=0.0) -> np.ndarray:
    """P(fail within `horizon` more producing days | already survived `elapsed`).

    Conditioning matters: a well 200 days into a run faces a different next
    90 days than one just restarted, and the Cox curve is not a straight line.
    """
    ph = cph.predict_partial_hazard(x).to_numpy()
    elapsed = np.broadcast_to(np.asarray(elapsed, dtype=float), ph.shape)
    s_start = baseline_survival_at(cph, elapsed) ** ph
    s_end = baseline_survival_at(cph, elapsed + horizon) ** ph
    return 1 - s_end / np.clip(s_start, 1e-12, None)


def open_runs_at(con, runs: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Runs in progress on `day`, with the producing days already survived.

    A run starting on `day` itself is excluded: the well restarted that day,
    so it has no time in the run yet to be scored on.
    """
    o = runs[(runs.start_d < day) & (runs.end_d > day)].copy()
    con.register("open_runs_tmp", o[["uwi", "event_id", "start_d"]])
    elapsed = con.sql(f"""
        select o.uwi, o.event_id, count(d.production_date) as elapsed_prod_days
        from open_runs_tmp o
        left join read_parquet('{config.DAILY_CLEAN.as_posix()}') d
          on d.uwi = o.uwi and d.in_scope and d.is_producing
         and d.production_date >= o.start_d and d.production_date < date '{day.date()}'
        group by all
    """).df()
    con.unregister("open_runs_tmp")
    return o.merge(elapsed, on=["uwi", "event_id"])


def score_open_runs(bundle: dict, open_runs: pd.DataFrame) -> np.ndarray:
    """Conditional risk of failing within the model's horizon, from the saved bundle."""
    prep = Prep.from_state(bundle)
    return risk_within(bundle["model"], prep.transform(open_runs), bundle["horizon_days"],
                       open_runs.elapsed_prod_days.to_numpy())


def survival_curve(bundle: dict, run_row: pd.DataFrame, max_days: int) -> pd.Series:
    """One run's survival S(t) on every producing day 0..max_days.

    Built from the same step function as risk_within, so reading the curve at
    elapsed and elapsed + horizon gives exactly the watchlist risk.
    """
    cph = bundle["model"]
    ph = cph.predict_partial_hazard(Prep.from_state(bundle).transform(run_row)).iloc[0]
    t = np.arange(0, max_days + 1)
    return pd.Series(baseline_survival_at(cph, t) ** ph, index=t)


def beyond_model_range(bundle: dict, elapsed) -> np.ndarray:
    """True where the scoring window runs past the model's data support.

    Very few training runs last past a few years, so the tail of the baseline
    survival curve rests on a handful of runs and goes flat, which gives
    risks near 0 that are artefacts rather than estimates. The limit is saved
    with the model by train_eval.py.
    """
    return np.asarray(elapsed, dtype=float) + bundle["horizon_days"] > bundle["support_limit_days"]


def support_limit(train_prod_days: pd.Series, min_at_risk: int) -> float:
    """Longest duration at which at least `min_at_risk` training runs are still observed."""
    t = np.sort(train_prod_days.to_numpy())[::-1]
    return float(t[min_at_risk - 1]) if len(t) >= min_at_risk else 0.0


def load_model() -> dict:
    return joblib.load(config.MODEL)
