"""Smoke test: every page of the dashboard opens and a well can be selected."""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import REPO, needs_outputs

from src import app_logic as al

APP = str(REPO / "app.py")
PAGES = ["Summary", "The problem", "What drives failure", "Is a workover worth it?", "Watchlist",
         "Does it work?"]

pytestmark = needs_outputs


def _open(page):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio(key="page").set_value(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_opens(page):
    at = _open(page)
    assert not at.exception, at.exception
    if page == "Summary":
        # The pitch headline is large markdown rather than a header.
        assert any("Same crews, better targets" in m.value for m in at.markdown)
    else:
        assert at.header[0].value


def test_select_well():
    at = _open("Watchlist")
    # Select by the underlying value; the displayed option is an alias.
    third = al.load_watchlist().sort_values("rank").uwi.iloc[2]
    at.selectbox(key="well").set_value(third).run()
    assert not at.exception, at.exception
    assert at.session_state["well"] == third


def test_hide_ids_default_on():
    at = _open("Watchlist")
    assert at.sidebar.toggle(key="hide_ids").value is True
    assert all(o.startswith("Well ") for o in at.selectbox(key="well").options)


def test_whatif_options():
    at = _open("Watchlist")
    for repair in ["Workover", "New rod string", "Rod truck only"]:
        at.radio(key="wi_repair").set_value(repair).run()
        assert not at.exception, at.exception


def test_replay_dates():
    at = _open("Does it work?")
    slider = at.select_slider(key="replay_date")
    for day in slider.options:
        at.select_slider(key="replay_date").set_value(day).run()
        assert not at.exception, at.exception


def test_summary_is_default_and_cards_navigate():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert at.sidebar.radio(key="page").value == "Summary"
    at.button(key="go_watchlist").click().run()
    assert not at.exception, at.exception
    assert at.sidebar.radio(key="page").value == "Watchlist"


def test_no_jargon_outside_technical_detail():
    banned = ["kaplan", "hazard ratio", "cox", "c-index", "calibration", "censor"]
    for page in PAGES:
        at = _open(page)
        visible = [e.value for e in [*at.header, *at.info, *at.metric, *at.caption]]
        visible += [m.label for m in at.metric]
        text = " ".join(str(v) for v in visible).lower()
        assert not [b for b in banned if b in text], (page, [b for b in banned if b in text])
