"""The well outlook chart must show the same 90-day risk as the watchlist."""
import numpy as np

from conftest import needs_outputs

from src import app_logic as al

pytestmark = needs_outputs


def test_outlook_matches_watchlist_risk():
    wl, runs, bundle = al.load_watchlist(), al.load_runs(), al.load_model()
    scored = wl[wl.risk_90.notna()]
    for uwi in scored.uwi:
        run = al.well_run(wl, runs, uwi)
        elapsed = float(run.elapsed_prod_days.iloc[0])
        got = al.outlook_risk_90(al.well_survival(bundle, run), elapsed)
        assert np.isclose(got, run.risk_90.iloc[0], atol=5e-4)
