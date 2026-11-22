"""Smoke test: every page of the dashboard opens and a well can be selected."""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import REPO, needs_outputs

from src import app_logic as al

APP = str(REPO / "app.py")
PAGES = ["Overview", "What drives run life", "Is a workover worth it?", "Watchlist", "Does it work?"]

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
