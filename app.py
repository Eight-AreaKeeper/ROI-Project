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


@st.cache_data
def replay():
    return al.load_replay()


@st.cache_resource
def model():
    return al.load_model()


@st.cache_data
def payback_inputs():
    return plots.payback_inputs(runs(), events(), outputs()["cox_results"])


@st.cache_data
def enriched_watchlist(oil_value):
    return al.enrich_watchlist(watchlist(), runs(), events(), outputs()["cox_results"], model(), oil_value)


@st.cache_data
def rpm_range():
    return al.rpm_range(runs())


@st.cache_data
def well_daily(uwi, end):
    return al.well_daily(uwi, end)


def chart(fig):
    st.plotly_chart(fig, use_container_width=True)


PAGES = ["Overview", "What drives run life", "Is a workover worth it?", "Watchlist", "Does it work?"]

d, wl = outputs(), watchlist()
aliases = al.make_aliases(wl, replay().uwi)
# The oil value is set on the workover page but also prices the watchlist, so
# it is kept outside the slider's own state, which is dropped on other pages.
st.session_state.setdefault("oil_value", float(config.OIL_VALUE_PER_BBL))
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
    oil = st.slider("Oil value, C$/bbl", 35.0, 70.0, st.session_state["oil_value"], key="oil")
    st.session_state["oil_value"] = oil
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

    oil = st.session_state["oil_value"]
    ew = enriched_watchlist(oil)
    f1, f2, f3 = st.columns([2, 1, 1])
    areas = f1.multiselect("Area", sorted(wl.area.dropna().unique()), key="areas")
    min_risk = f2.slider("Minimum 90-day risk", 0.0, 1.0, 0.0, 0.05, key="min_risk")
    sort = f3.radio("Sort by", list(al.SORTS), horizontal=True, key="sort")
    table = al.watchlist_table(ew, aliases, hide, areas, min_risk, sort)
    money = st.column_config.NumberColumn(format="dollar")
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        "risk_90": st.column_config.ProgressColumn("90-day risk", format="%.2f", min_value=0.0, max_value=1.0),
        "expected cost, 90 days": money, "cost per failure": money,
        "workover payback, yrs": st.column_config.NumberColumn(format="%.1f"),
    })
    st.markdown(f"**{al.top_summary(table)}**")
    unscored = int(wl.beyond_model_range.sum())
    st.caption(f"{len(table):,} wells shown. Costs at C${oil:.2f}/bbl (set on \"Is a workover worth it?\"). "
               f"Expected cost = 90-day risk x cost of one failure (rod-truck repair + this well's usual lost oil). "
               f"{unscored} wells have run longer than the model has data for and are not scored.")
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

        with st.container(border=True):
            st.subheader(f"What if? Next repair on {name}")
            lo, hi = rpm_range()
            now_rpm = al.current_rpm(run, model())
            w1, w2 = st.columns(2)
            repair = w1.radio("Repair at the next failure", list(al.REPAIRS), horizontal=True, key="wi_repair")
            rpm = w2.slider("Pump speed after restart, rpm", lo, hi, min(max(now_rpm, lo), hi), 1.0,
                            key=f"wi_rpm_{uwi}")  # per well, so it starts at that well's speed
            wi = al.whatif(run, model(), events(), end, repair, rpm).set_index("option")
            m1, m2, m3 = st.columns([1, 1, 2])
            m1.metric(f"90-day risk: {repair.lower()} at {rpm:.0f} rpm", f"{wi.risk_90[repair]:.0%}",
                      delta=f"{(wi.risk_90[repair] - wi.risk_90['baseline']) * 100:+.0f} pts",
                      delta_color="inverse")
            m2.metric(f"Rod truck only at {now_rpm:.0f} rpm (today's speed)", f"{wi.risk_90['baseline']:.0%}")
            with m3:
                chart(plots.repair_options(wi.reset_index()))
            st.markdown(al.whatif_note(ew.set_index("uwi").loc[uwi], payback_inputs()))
            st.caption("Model estimate based on similar past runs; associations, not guaranteed effects.")

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
    st.subheader("Replay: what the watchlist would have said")
    st.caption("Each date is after the training cut. Every open well is scored as it stood that day, "
               "then checked against the official failure log for the next 90 days.")
    summ, rp = d["replay_summary"], replay()
    dates = [str(x) for x in summ.date]
    day = st.select_slider("Check date", dates, value=dates[0], key="replay_date")
    r = summ[summ.date.astype(str) == day].iloc[0]
    top = rp[rp.date.astype(str) == day].assign(label=lambda x: al.show_id(x.uwi, aliases, hide))
    g1, g2 = st.columns([3, 2])
    with g1:
        chart(plots.replay_top50(top, top_n=config.TOP_K))
    with g2:
        k1, k2, k3 = st.columns(3)
        k1.metric("Our model", f"{r.model_top50_hits:.0f}")
        k2.metric("Today's rule", f"{r.rule_top50_mean:.1f}",
                  help=f"Mean over tie orders; range {r.rule_top50_min:.0f} to {r.rule_top50_max:.0f}")
        k3.metric("Random 50", f"{r.random_top50:.1f}")
        st.caption(f"{int(r.wells):,} wells in a run on {day}; {r.base_rate:.0%} failed within 90 days.")
        chart(plots.replay_trend(summ, top_n=config.TOP_K))

    with st.expander("Limitations"):
        st.markdown("\n".join(f"- {x}" for x in al.LIMITATIONS))
