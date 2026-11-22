"""Unit tests for the watchlist logic on hand-made rows; no data files needed."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import needs_outputs

from src import app_logic as al


def test_expected_cost_is_risk_times_cost():
    risk = pd.Series([0.9, 0.5, 0.0, np.nan])
    cost = pd.Series([10_000.0, 8_000.0, 12_000.0, 9_000.0])
    got = al.expected_cost(risk, cost)
    assert got.iloc[:3].tolist() == [9_000.0, 4_000.0, 0.0]
    assert np.isnan(got.iloc[3])


def _row(**kw):
    base = dict(workover=0, new_string=0, workover_payback_yrs=5.0,
                main_driver="frequent recent failures", rpm_pre30=200.0, risk_90=0.3)
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize("row, expected", [
    # 1. not a thorough last repair, and a workover pays back fast enough
    (_row(workover_payback_yrs=1.2, main_driver="high pump speed", rpm_pre30=500, risk_90=0.9),
     "At next failure: workover (pays back in 1.2 yr)"),
    # 2. last repair was already a workover, so pump speed is the next lever
    (_row(workover=1, workover_payback_yrs=0.8, main_driver="high pump speed", rpm_pre30=500, risk_90=0.9),
     "Review pump speed"),
    # 3. high risk, speed not in the top quartile
    (_row(new_string=1, main_driver="high pump speed", rpm_pre30=300, risk_90=0.75),
     "Pre-plan rod-truck crew"),
    # 4. nothing applies
    (_row(risk_90=0.4), "Monitor"),
])
def test_action_rules(row, expected):
    assert al.suggest_action(row, rpm_top_quartile=400.0) == expected


def test_driver_is_largest_positive_contribution():
    contrib = pd.DataFrame({
        "fails_prior_365": [0.30, -0.10, -0.2],
        "rpm_pre30": [0.10, 0.25, -0.1],
        "prev_gap_days": [0.05, 0.20, -0.3],
    }).reindex(columns=list(al.DRIVER_LABELS), fill_value=-1.0)
    assert al.main_driver(contrib).tolist() == [
        "frequent recent failures", "high pump speed", al.NO_DRIVER]


@needs_outputs
def test_enrich_on_real_watchlist():
    d = al.load_outputs()
    w = al.enrich_watchlist(al.load_watchlist(), al.load_runs(), al.load_events(),
                            d["cox_results"], al.load_model(), 51.04)
    s = w[w.risk_90.notna()]
    assert np.allclose(s.expected_cost_90, s.risk_90 * s.cost_per_failure)
    assert s.main_driver.isin(list(al.DRIVER_LABELS.values()) + [al.NO_DRIVER]).all()
    assert s.suggested_action.notna().all()


@needs_outputs
def test_workover_lowers_risk_for_every_well():
    # The workover hazard ratio is below 1 (0.567), so for any well and any
    # pump speed a workover must give a lower risk than rod truck only.
    wl, runs, ev, bundle = al.load_watchlist(), al.load_runs(), al.load_events(), al.load_model()
    lo, hi = al.rpm_range(runs)
    as_of = al.as_of(wl)
    for uwi in wl[wl.risk_90.notna()].uwi:
        run = al.well_run(wl, runs, uwi)
        for rpm in (lo, hi):
            r = al.whatif(run, bundle, ev, as_of, "Workover", rpm).set_index("option").risk_90
            assert r["Workover"] < r["Rod truck only"]
