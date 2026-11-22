"""Run Life dashboard. Layout and widgets only; numbers come from src/app_logic.py
and every chart from src/plots.py.

    streamlit run app.py
"""
import streamlit as st

from src import app_logic as al
from src import config, plots

st.set_page_config(layout="wide", page_title="Run Life", page_icon="🛢️")


# Loaders are cached so switching pages or moving a slider never rereads files.
@st.cache_data
def outputs():
    return al.load_outputs()


@st.cache_data
def watchlist():
    return al.load_watchlist()


@st.cache_data
def runs():
    return al.load_runs()


@st.cache_data
def events():
    return al.load_events()


@st.cache_resource
def model():
    return al.load_model()


@st.cache_data
def payback_inputs():
    return plots.payback_inputs(runs(), events(), outputs()["cox_results"])


@st.cache_data
def well_daily(uwi, end):
    return al.well_daily(uwi, end)


def chart(fig):
    st.plotly_chart(fig, use_container_width=True)


PAGES = ["Overview", "What drives run life", "Is a workover worth it?", "Watchlist", "Does it work?"]

d, wl = outputs(), watchlist()
aliases = al.make_aliases(wl)
with st.sidebar:
    st.title("Run Life")
    page = st.radio("Page", PAGES, key="page")
    hide = st.toggle("Hide well IDs", value=True, key="hide_ids")
    st.caption(f"Data as of {al.as_of(wl):%B %d, %Y}")


# --------------------------------------------------------------------------
if page == "Overview":
    st.header("Downhole failures cost about $40M a year, and 10% of wells cause half of it.")
    st.caption("Heavy-oil PCP wells, January 2021 to October 2026, failures rebuilt from daily downtime codes.")
    m = al.headline_metrics(d["sizing"])
    c = st.columns(4)
    c[0].metric("Failures per producing well-year", f"{m['failures_per_well_year']:.2f}")
    c[1].metric("Repeat failure within 30 days", f"{m['repeat_within_30d']:.1%}")
    c[2].metric("Reactive share of repair spend", f"{m['reactive_spend_share']:.0%}")
    c[3].metric("Failure cost per year", f"${m['cost_per_year_m']:.0f}M")
    chart(plots.cost_per_year(d["sizing"]))
    chart(plots.concentration(d["pareto"]))
    chart(plots.failures_per_year(events()))

# --------------------------------------------------------------------------
elif page == "What drives run life":
    st.header("Wells that fail keep failing; thorough repairs and slower pumps go with longer runs.")
    st.caption("Run life is the producing days from a repair to the next downhole failure.")
    labels = {"history": "Failure history", "rpm": "Pump speed", "water_cut": "Water cut", "repair": "Last repair"}
    by = st.radio("Group runs by", list(labels), format_func=labels.get, horizontal=True, key="by")
    chart(plots.run_life(d["km_curves"], d["km_summary"], by))
    st.caption("Each line is the share of wells still running after a given number of producing days. "
               "A line that stays higher means longer runs; the dot marks the day half have failed.")
    chart(plots.risk_drivers(d["cox_results"]))
    st.caption("Each dot is one factor's link to failure risk with the others held fixed. Left of zero goes "
               "with longer runs, right with shorter; the line is the 95% range, and grey means no clear effect.")

# --------------------------------------------------------------------------
elif page == "Is a workover worth it?":
    st.header("A workover pays back within a year on wells failing about 6+ times a year.")
    st.caption("A workover costs more than a rod-truck job but is followed by fewer failures; "
               "the payback depends on how often the well fails and what the lost oil is worth.")
    oil = st.slider("Oil value, C$/bbl", 35.0, 70.0, float(config.OIL_VALUE_PER_BBL), key="oil")
    inputs = payback_inputs()
    st.metric("Break-even failure rate", f"{plots.break_even_rate(inputs, oil):.1f} failures/yr",
              help="Above this rate a workover's extra cost pays back within one year.")
    left, right = st.columns(2)
    with left:
        chart(plots.run_life(d["km_curves"], d["km_summary"], "repair"))
    with right:
        chart(plots.workover_payback(inputs, oil_value=oil))

# --------------------------------------------------------------------------
elif page == "Watchlist":
    st.header("The wells to act on next.")
    st.caption(f"Chance each well in a run fails within the next {config.HORIZON_DAYS} producing days, "
               "given how long it has already run.")
    chart(plots.risk_distribution(wl, top_n=config.TOP_K))

    f1, f2 = st.columns([2, 1])
    areas = f1.multiselect("Area", sorted(wl.area.dropna().unique()), key="areas")
    min_risk = f2.slider("Minimum 90-day risk", 0.0, 1.0, 0.0, 0.05, key="min_risk")
    table = al.watchlist_table(wl, aliases, hide, areas, min_risk)
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"risk_90": st.column_config.ProgressColumn(
                     "90-day risk", format="%.2f", min_value=0.0, max_value=1.0)})
    unscored = int(wl.beyond_model_range.sum())
    st.caption(f"{len(table):,} wells shown. {unscored} wells have run longer than the model has data for "
               "and are not scored.")
    st.download_button("Download CSV", table.to_csv(index=False), "watchlist.csv", "text/csv")

    if len(table):
        uwis = list(table.index)
        uwi = st.selectbox("Well", uwis, format_func=lambda u: al.show_id(u, aliases, hide), key="well")
        name = al.show_id(uwi, aliases, hide)
        run = al.well_run(wl, runs(), uwi)
        elapsed = float(run.elapsed_prod_days.iloc[0])
        end = al.as_of(wl)
        c1, c2 = st.columns(2)
        with c1:
            chart(plots.well_outlook(al.well_survival(model(), run), elapsed, label=name))
        with c2:
            chart(plots.well_history(well_daily(uwi, end), al.well_events(events(), uwi, end)))

# --------------------------------------------------------------------------
elif page == "Does it work?":
    st.header("On the official failure log it never saw, our top 50 caught 37 failures; today's rule caught 28.")
    st.caption("The model was trained on runs before 2025 and checked on later data it never saw.")
    check = st.radio("Check", ["official", "test"], horizontal=True, key="check",
                     format_func={"official": "Official failure log", "test": "Held-out runs"}.get)
    chart(plots.model_vs_rule(d["model_eval"], check))
    c1, c2 = st.columns(2)
    with c1:
        chart(plots.calibration(d["calibration"]))
    with c2:
        chart(plots.what_breaks(d["event_validation"]))
    with st.expander("Limitations"):
        st.markdown("\n".join(f"- {x}" for x in al.LIMITATIONS))
