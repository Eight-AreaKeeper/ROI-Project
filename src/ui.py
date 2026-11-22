"""Shared page building blocks, so every page of the app has the same layout.

Each page is built in the same order: page_header, kpi_strip, so_what, one
main chart, then "More detail" and "Technical detail" expanders.
"""
from __future__ import annotations

import streamlit as st

PAGE_KEY = "page"


def base_style() -> None:
    """Hide the arrow on metric captions and tighten the top padding.

    Metric captions here are descriptions or comparisons ("vs 134 with none"),
    not changes, so an up/down arrow would wrongly suggest good or bad.
    """
    st.markdown(
        "<style>"
        "[data-testid='stMetricDelta'] svg{display:none}"
        ".block-container{padding-top:2.2rem}"
        "</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, top: bool = False) -> None:
    """The page's finding as a header (or the app title on the first page), with one plain subtitle."""
    (st.title if top else st.header)(title)
    st.caption(subtitle)


def kpi_strip(items: list[tuple[str, str, str]]) -> None:
    """One row of metrics; each item is (label, value, caption)."""
    for col, (label, value, caption) in zip(st.columns(len(items)), items):
        col.metric(label, value, delta=caption, delta_color="off")


def so_what(text: str) -> None:
    st.info(f"**So what:** {text}")


def _go(page: str) -> None:
    st.session_state[PAGE_KEY] = page


def insight_card(metric: str, body: str, page: str, button: str | None = None, key: str | None = None) -> None:
    """A bordered card: big number, one or two sentences, and a button to the page that proves it."""
    with st.container(border=True):
        st.markdown(f"<div style='font-size:1.7rem;font-weight:600;line-height:1.2'>{metric}</div>",
                    unsafe_allow_html=True)
        st.markdown(body)
        st.button(button or f"{page} →", key=key or f"go_{page}", on_click=_go, args=(page,))


def footer(text: str) -> None:
    st.caption(text)
