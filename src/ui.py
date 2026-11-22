"""Shared page building blocks, so every page of the app has the same layout.

Each page is built in the same order: page_header, kpi_strip, so_what, the
main and secondary chart side by side, then "More detail" and
"Technical detail" expanders.
"""
from __future__ import annotations

import re

import plotly.graph_objects as go
import streamlit as st

PAGE_KEY = "page"
SUBTITLE_COLOR = "#52514e"
CHART_HEIGHT = 380

# One style block for the whole app.
#  - Content width is capped so lines and charts stay readable on wide screens.
#  - Metric values and captions wrap instead of being cut off with "..." in narrow cards.
#  - Metric captions are descriptions or comparisons ("vs 134 with none"), not
#    changes, so the up/down arrow Streamlit adds would wrongly suggest good or bad.
_CSS = f"""
<style>
.block-container {{max-width: 1280px; padding-top: 2.75rem;}}
[data-testid='stMetricDelta'] svg {{display: none;}}
[data-testid='stMetricValue'] {{font-size: 1.85rem !important;}}
[data-testid='stMetricValue'] > div, [data-testid='stMetricDelta'] > div {{white-space: normal !important; overflow: visible !important; text-overflow: clip !important;}}
.page-subtitle {{font-size: 1.05rem; color: {SUBTITLE_COLOR}; margin: -0.4rem 0 0.9rem;}}
</style>
"""


def base_style() -> None:
    # st.html applies a style-only block without taking up space; st.markdown drops <style> here.
    st.html(_CSS)


def md_safe(text: str) -> str:
    """Escape "$" so Streamlit markdown does not read text between two of them as maths."""
    return text.replace("$", "\\$")


def page_header(title: str, subtitle: str, top: bool = False) -> None:
    """The page's finding as a header (or the app title on the first page), with one plain subtitle."""
    (st.title if top else st.header)(title)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def kpi_strip(items: list[tuple[str, str, str]]) -> None:
    """One row of metric cards; each item is (label, value, caption)."""
    for col, (label, value, caption) in zip(st.columns(len(items)), items):
        with col.container(border=True):
            st.metric(label, value, delta=caption, delta_color="off")


def so_what(text: str) -> None:
    st.info(f"**So what:** {md_safe(text)}")


def _go(page: str) -> None:
    st.session_state[PAGE_KEY] = page


def insight_card(metric: str, body: str, page: str, button: str | None = None, key: str | None = None) -> None:
    """A bordered card: big number, one or two sentences, and a button to the page that proves it."""
    with st.container(border=True):
        st.markdown(f"<div style='font-size:1.7rem;font-weight:600;line-height:1.2'>{metric}</div>",
                    unsafe_allow_html=True)
        st.markdown(md_safe(body))
        st.button(button or page, key=key or f"go_{page}", on_click=_go, args=(page,))


def footer(text: str) -> None:
    st.caption(text)


def _subtitle(fig: go.Figure) -> str:
    m = re.search(r"<span[^>]*>(.*?)</span>", fig.layout.title.text or "")
    return m.group(1) if m else ""


def chart(fig: go.Figure, height: int | None = None, hide_title: bool = False, fixed_width: int | None = None) -> None:
    """Show a plots.py figure with the app's chart text sizes.

    hide_title drops a title that repeats the page header and shows the
    subtitle as a caption above the chart instead. The figure is copied, so
    the same function still gives the full title in docs/figures exports.
    """
    fig = go.Figure(fig)
    if hide_title:
        sub = _subtitle(fig)
        if sub:
            st.caption(md_safe(sub.replace("&#36;", "$")))
        fig.update_layout(title_text="", margin_t=25)
    fig.update_layout(font_size=14)
    fig.update_xaxes(tickfont_size=13)
    fig.update_yaxes(tickfont_size=13)
    if height:
        fig.update_layout(height=height)
    if fixed_width:
        fig.update_layout(width=fixed_width)
        st.plotly_chart(fig, width="content")
    else:
        st.plotly_chart(fig, width="stretch")
