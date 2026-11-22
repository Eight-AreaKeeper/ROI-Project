"""Run Life dashboard. Layout and widgets only; numbers come from src/app_logic.py
and every chart from src/plots.py. No number in this file is typed in.

    streamlit run app.py
    streamlit run app.py -- then open ?page=watchlist to land on a page
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
def impact(oil_value):
    d = outputs()
    return al.impact_summary(enriched_watchlist(oil_value), d["model_eval"], d["cox_results"],
                             payback_inputs(), oil_value, d["sizing"])


@st.cache_data
def numbers():
    return al.story_numbers(outputs(), payback_inputs())


@st.cache_data
def rpm_range():
    return al.rpm_range(runs())


@st.cache_data
def well_daily(uwi, end):
    return al.well_daily(uwi, end)


def chart(fig):
    st.plotly_chart(fig, width="stretch")


def page_top(header, so_what):
    st.header(header)
    st.info(f"**So what:** {so_what}")


def goto(page):
    st.session_state["page"] = page


PAGES = {"summary": "Summary", "problem": "The problem", "drivers": "What drives failure",
         "workover": "Is a workover worth it?", "watchlist": "Watchlist", "proof": "Does it work?"}

# A ?page= link lands on that page; used for shared links and screenshots.
if "page" not in st.session_state:
    st.session_state["page"] = PAGES.get(st.query_params.get("page", "summary"), "Summary")
# The oil value is set on the workover page but also prices the watchlist and
# the impact numbers, so it lives outside the slider's own state.
st.session_state.setdefault("oil_value", float(config.OIL_VALUE_PER_BBL))
oil = st.session_state["oil_value"]

d, wl, n = outputs(), watchlist(), numbers()
imp = impact(oil)
aliases = al.make_aliases(wl, replay().uwi)
k = config.TOP_K

# Metric deltas here are comparisons ("vs 134 with none"), not changes, so
# the up/down arrow would mislead; only the text is kept.
st.markdown("<style>[data-testid='stMetricDelta'] svg{display:none}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Run Life")
    st.markdown(al.sidebar_story(n, imp))
    st.divider()
    page = st.radio("Page", list(PAGES.values()), key="page")
    hide = st.toggle("Hide well IDs", value=True, key="hide_ids")
    st.caption(f"Data as of {al.as_of(wl):%B %d, %Y}")


# --------------------------------------------------------------------------
if page == "Summary":
    st.markdown("<style>.block-container{padding-top:2.2rem}</style>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.9rem;font-weight:700;line-height:1.25;margin:0 0 1rem'>"
                f"{al.impact_headline(imp)}</p>", unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1.container(border=True):
        st.markdown("**The problem**")
        st.markdown(
            f"Strathcona's {n['wells']:,} heavy-oil wells have about {round(n['failures_per_year'], -2):,.0f} "
            f"downhole failures a year. {n['reactive_share']:.0%} of repair spend is reactive, and 10% of the "
            f"wells cause {al.share_words(n['top10_lost_oil'])} the lost oil. Total: about ${imp['fleet_cost_per_year'] / 1e6:.0f}M a year.")
    with p2.container(border=True):
        st.markdown("**Our research question**")
        st.markdown("*Which operating conditions and repair choices shorten or extend the time a heavy-oil PCP "
                    "well runs between downhole failures, and is the extra run time from a more thorough repair "
                    "worth its cost?*")

    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.metric("What drives failure", f"{n['days_2plus']:.0f} vs {n['days_none']:.0f} days")
        st.markdown(f"Wells that failed 2+ times last year run {al.as_long(n['days_2plus'] / n['days_none'])} "
                    "after a repair. Faster pumps and wetter wells also fail sooner.")
        st.button("What drives failure →", key="go_drivers", on_click=goto, args=(PAGES["drivers"],))
    with c2.container(border=True):
        st.metric("Is a better repair worth it?", f"−{n['workover_risk_cut']:.0%} risk")
        st.markdown(f"After a workover, wells fail {n['workover_risk_cut']:.0%} less often. It pays for itself "
                    f"within a year on wells failing {plots.break_even_rate(payback_inputs(), oil):.0f}+ times "
                    f"a year: ~{imp['workover_wells']} wells today.")
        st.button("Is a workover worth it? →", key="go_workover", on_click=goto, args=(PAGES["workover"],))
    with c3.container(border=True):
        st.metric("Which wells next", f"{n['model_hits']:.0f} of {k}")
        st.markdown(f"Of the {k} wells we flagged, {n['model_hits']:.0f} failed within 90 days, against "
                    f"{n['rule_hits']:.0f} for today's rule and {n['random_hits']:.0f} at random.")
        st.button("See the watchlist →", key="go_watchlist", on_click=goto, args=(PAGES["watchlist"],))

    st.markdown("**What to do on Monday**")
    st.markdown(f"1. Review the top {k} wells on the watchlist every week.\n"
                f"2. On the ~{imp['workover_wells']} flagged wells, do a workover at their next failure instead "
                "of a rod-truck fix.\n"
                "3. Check pump speed on high-RPM wells that keep failing.")
    chart(plots.cost_per_year(d["sizing"], oil_value=oil))

# --------------------------------------------------------------------------
elif page == "The problem":
    page_top(f"10% of wells cause {al.share_words(n['top10_lost_oil'])} the lost oil.",
             "Focus on a short list of wells, not the whole fleet.")
    m = st.columns(3)
    m[0].metric("Failures per producing well-year", f"{n['failures_per_well_year']:.2f}",
                f"1 every {n['producing_days_per_failure']:.0f} producing days", delta_color="off")
    m[1].metric("Fail again within 30 days", f"{n['repeat_30d']:.0%}", "of repaired wells", delta_color="off")
    m[2].metric("Reactive repair spend", f"{n['reactive_share']:.0%}",
                f"of ${n['repair_spend'] / 1e6:.0f}M total repair spend", delta_color="off")
    chart(plots.concentration(d["pareto"]))
    with st.expander("More detail"):
        chart(plots.failures_per_year(events()))
        chart(plots.cost_per_year(d["sizing"], oil_value=oil))

# --------------------------------------------------------------------------
elif page == "What drives failure":
    page_top("Wells that keep failing, run fast or run wet fail sooner.",
             "A well's last year predicts its next 90 days, so the history alone can rank wells.")
    m = st.columns(3)
    m[0].metric("2+ failures last year", f"{n['days_2plus']:.0f} days",
                f"vs {n['days_none']:.0f} with none", delta_color="off",
                help="Median producing days from a repair to the next failure")
    m[1].metric("Fastest pumps", f"{n['days_fast']:.0f} days", f"vs {n['days_slow']:.0f} slowest",
                delta_color="off", help="Fastest vs slowest quarter of pump speeds")
    m[2].metric("After a workover", f"{n['days_workover']:.0f} days",
                f"vs {n['days_rodtruck']:.0f} after rod truck only", delta_color="off")
    labels = {"history": "Past failures", "rpm": "Pump speed", "water_cut": "Water cut", "repair": "Last repair"}
    by = st.radio("Compare wells by", list(labels), format_func=labels.get, horizontal=True, key="by")
    chart(plots.run_life(d["km_curves"], d["km_summary"], by))
    st.caption("Each line shows the share of wells still running N days after a repair. "
               "Lower and steeper means they break sooner.")
    with st.expander("More detail"):
        chart(plots.risk_drivers(d["cox_results"]))
        st.caption("How much each factor changes the risk of failure, holding the others fixed.")
    with st.expander("Technical detail"):
        st.markdown(
            "- Run-life curves are Kaplan-Meier estimates. Runs still going when the data ends are censored, "
            "so they count up to that day instead of being dropped. Group differences are tested with the "
            "log-rank test (all p < 0.001).\n"
            "- The factor chart is a Cox proportional-hazards model on runs before 2025, with yes/no factors "
            "compared yes vs no and measured factors per one standard deviation. Hazard ratios are "
            "associations, not proof of cause.")

# --------------------------------------------------------------------------
elif page == "Is a workover worth it?":
    page_top("A workover pays for itself on wells that fail "
             f"{plots.break_even_rate(payback_inputs(), oil):.0f}+ times a year.",
             f"Do a workover instead of a quick fix on the ~{imp['workover_wells']} flagged wells. "
             "It's not worth it fleet-wide.")
    # The callback stores the new value before the rerun, so the header and
    # "So what" above already use it.
    oil = st.slider("Oil value, C$/bbl", 35.0, 70.0, oil, key="oil",
                    on_change=lambda: st.session_state.update(oil_value=st.session_state["oil"]))
    inputs = payback_inputs()
    m = st.columns(3)
    m[0].metric("Failure risk after a workover", f"−{n['workover_risk_cut']:.0%}", "vs rod truck only",
                delta_color="off")
    m[1].metric("Extra cost of a workover", f"+${n['workover_extra_cost'] / 1e3:.0f}k",
                "vs a rod-truck job (median)", delta_color="off")
    m[2].metric("Break-even", f"{plots.break_even_rate(inputs, oil):.1f} failures/yr",
                f"at C${oil:.2f}/bbl", delta_color="off")
    chart(plots.workover_payback(inputs, oil_value=oil))
    with st.expander("More detail"):
        chart(plots.run_life(d["km_curves"], d["km_summary"], "repair"))
    with st.expander("Technical detail"):
        imp_now = impact(oil)
        st.markdown(
            f"- One failure costs the median rod-truck repair (${inputs['rodtruck_cost']:,.0f}) plus the lost oil "
            f"(fleet mean {inputs['oil_m3_per_event']:.1f} m³ per failure × 6.29 bbl/m³ × the oil value).\n"
            f"- A workover's extra cost is the median workover (${inputs['workover_cost']:,.0f}) minus the median "
            f"rod-truck job. The Cox hazard ratio for a workover as the last repair is {inputs['hazard_ratio']:.3f}, "
            f"so a well failing f times a year avoids f × {1 - inputs['hazard_ratio']:.2f} failures a year.\n"
            "- Payback = extra cost / (avoided failures × cost per failure). This is an association from past "
            "runs, not a controlled trial.\n"
            f"- Flagged wells: open wells whose last repair was not already a workover or new rod string and "
            f"where one year of avoided failures (at their own lost oil) beats the extra cost: "
            f"{imp_now['workover_wells']} wells, ~{imp_now['failures_avoided']:.0f} failures avoided a year.\n"
            f"- Year-one net saving on those wells: about ${imp_now['net_savings_year1'] / 1e6:.1f}M after paying "
            "for the workovers; the benefit continues after year one.")

# --------------------------------------------------------------------------
elif page == "Watchlist":
    ew = enriched_watchlist(oil)
    # The header and tiles describe the filtered table, so they are drawn into
    # this slot after the filters below have been read.
    head = st.container()
    with st.expander("Filters and sorting", expanded=False):
        f1, f2, f3 = st.columns([2, 1, 1])
        areas = f1.multiselect("Area", sorted(wl.area.dropna().unique()), key="areas")
        min_risk = f2.slider("Minimum 90-day risk", 0.0, 1.0, 0.0, 0.05, key="min_risk")
        sort = f3.radio("Sort by", list(al.SORTS), horizontal=True, key="sort")
        all_cols = st.checkbox("Show all columns", key="all_cols")
    table = al.watchlist_table(ew, aliases, hide, areas, min_risk, sort)
    top = table.head(k)
    n_wo = int(top["suggested action"].str.startswith("At next failure: workover").sum())
    with head:
        page_top(f"The {len(top)} wells to act on this week.",
                 f"These {len(top)} wells carry ${top['expected cost, 90 days'].sum():,.0f} of expected failure "
                 f"cost in the next 90 days; {n_wo} should get a workover at their next failure.")
        m = st.columns(3)
    m[0].metric("Wells flagged", f"{len(top)}", f"of {int(ew.risk_90.notna().sum()):,} wells scored",
                delta_color="off")
    m[1].metric("Expected failures", f"{top.risk_90.sum():.0f} of {len(top)}", "in the next 90 days",
                delta_color="off")
    m[2].metric("Expected failure cost", f"${top['expected cost, 90 days'].sum() / 1e3:,.0f}k",
                f"next 90 days at C${oil:.0f}/bbl", delta_color="off")

    money = st.column_config.NumberColumn(format="$%.0f")
    default_cols = ["well", "area", "risk_90", "expected cost, 90 days", "main driver", "suggested action"]
    # Shown as whole percent and whole dollars; the download keeps full precision.
    st.dataframe(table.assign(risk_90=table.risk_90 * 100), hide_index=True, width="stretch",
                 column_order=None if all_cols else default_cols, column_config={
                     "risk_90": st.column_config.ProgressColumn("risk %", format="%.0f%%",
                                                                min_value=0.0, max_value=100.0),
                     "expected cost, 90 days": money, "cost per failure": money,
                     "workover payback, yrs": st.column_config.NumberColumn(format="%.1f"),
                 })
    st.caption(f"{len(table):,} wells. Sorted by {sort}. Risk = chance of failing in the next 90 producing days; "
               f"expected cost = risk × cost of one failure at C${oil:.2f}/bbl. "
               f"{int(wl.beyond_model_range.sum())} wells have run longer than the model has data for and "
               "are not scored.")
    st.download_button("Download CSV", table.to_csv(index=False), "watchlist.csv", "text/csv")

    if len(table):
        uwi = st.selectbox("Well", list(table.index), format_func=lambda u: al.show_id(u, aliases, hide),
                           key="well")
        name = al.show_id(uwi, aliases, hide)
        run = al.well_run(wl, runs(), uwi)
        elapsed = float(run.elapsed_prod_days.iloc[0])
        end = al.as_of(wl)
        c1, c2 = st.columns(2)
        with c1:
            chart(plots.well_outlook(al.well_survival(model(), run), elapsed, label=name))
        with c2.container(border=True):
            st.markdown(f"**What if? Next repair on {name}**")
            lo, hi = rpm_range()
            now_rpm = al.current_rpm(run, model())
            repair = st.radio("Repair at the next failure", list(al.REPAIRS), horizontal=True, key="wi_repair")
            rpm = st.slider("Pump speed after restart, rpm", lo, hi, min(max(now_rpm, lo), hi), 1.0, format="%.0f",
                            key=f"wi_rpm_{uwi}")  # per well, so it starts at that well's speed
            wi = al.whatif(run, model(), events(), end, repair, rpm).set_index("option")
            w1, w2 = st.columns(2)
            w1.metric(f"{repair} at {rpm:.0f} rpm", f"{wi.risk_90[repair]:.0%}",
                      delta=f"{(wi.risk_90[repair] - wi.risk_90['baseline']) * 100:+.0f} pts",
                      delta_color="inverse", help="Chance of failing in the first 90 producing days")
            w2.metric(f"Rod truck only at {now_rpm:.0f} rpm", f"{wi.risk_90['baseline']:.0%}", "today's speed",
                      delta_color="off")
            chart(plots.repair_options(wi.reset_index()))
            st.markdown(al.whatif_note(ew.set_index("uwi").loc[uwi], payback_inputs()))
            st.caption("Model estimate based on similar past runs; associations, not guaranteed effects.")
        with st.expander("More detail"):
            chart(plots.well_history(well_daily(uwi, end), al.well_events(events(), uwi, end)))
            chart(plots.risk_distribution(wl, top_n=k))

# --------------------------------------------------------------------------
elif page == "Does it work?":
    page_top(f"On failures it never saw, our top {k} caught {n['model_hits']:.0f}; "
             f"today's rule caught {n['rule_hits']:.0f}.",
             f"Same {k} well visits, about {imp['extra_hits']:.0f} more of the right wells: "
             f"{imp['lift_pct']:.0%} better targeting.")
    m = st.columns(3)
    m[0].metric("Our model", f"{n['model_hits']:.0f} of {k}", "failed within 90 days", delta_color="off")
    m[1].metric("Today's rule", f"{n['rule_hits']:.0f} of {k}",
                f"range {n['rule_min']:.0f} to {n['rule_max']:.0f}", delta_color="off",
                help="Most failures last year; many wells tie, so the result depends on which tied wells are picked")
    m[2].metric("Random wells", f"{n['random_hits']:.0f} of {k}", "for comparison", delta_color="off")
    chart(plots.model_vs_rule(d["model_eval"], "official"))

    st.subheader("Replay: what the watchlist would have said")
    st.caption("Each date is after the model was built. Every open well is scored as it stood that day, "
               "then checked against the official failure log for the next 90 days.")
    summ, rp = d["replay_summary"], replay()
    dates = [str(x) for x in summ.date]
    day = st.select_slider("Check date", dates, value=dates[0], key="replay_date")
    r = summ[summ.date.astype(str) == day].iloc[0]
    top = rp[rp.date.astype(str) == day].assign(label=lambda x: al.show_id(x.uwi, aliases, hide))
    g1, g2 = st.columns([3, 2])
    with g1:
        chart(plots.replay_top50(top, top_n=k))
    with g2:
        k1, k2, k3 = st.columns(3)
        k1.metric("Our model", f"{r.model_top50_hits:.0f}")
        k2.metric("Today's rule", f"{r.rule_top50_mean:.1f}",
                  help=f"Average over tied wells; range {r.rule_top50_min:.0f} to {r.rule_top50_max:.0f}")
        k3.metric("Random 50", f"{r.random_top50:.1f}")
        st.caption(f"{int(r.wells):,} wells in a run on {day}; {r.base_rate:.0%} failed within 90 days.")
        chart(plots.replay_trend(summ, top_n=k))

    with st.expander("More detail"):
        chart(plots.calibration(d["calibration"]))
        st.caption("When we say 70%, about 70% fail.")
        chart(plots.what_breaks(d["event_validation"]))
        st.caption(f"Our failure records match the official log {n['recall']:.0%} of the time.")
        st.markdown("**How this runs in production**")
        st.markdown(al.PRODUCTION_NOTES)
    with st.expander("Technical detail"):
        st.markdown(
            f"- **Time split:** the model is trained on runs that started before {config.TIME_CUT} and tested on "
            f"the {n['test_runs']:,} runs that started after.\n"
            f"- **Ranking (C-index):** {n['c_index']:.3f} for the model vs {n['c_index_rule']:.3f} for today's rule.\n"
            f"- **Official-log check ({config.INDEPENDENT_CHECK_DATE}, {n['check_wells']} wells):** AUC "
            f"{n['auc']:.3f} vs {n['auc_rule']:.3f} for the rule.\n"
            f"- **Probability accuracy (Brier, 90 days):** {n['brier']:.3f} vs {n['brier_km']:.3f} for a flat "
            "Kaplan-Meier rate (lower is better); calibration by risk decile is in More detail.\n"
            "- **Model:** Cox proportional hazards on pre-run features, scored as the chance of failing in the "
            "next 90 producing days given the days already survived.")
        chart(plots.model_vs_rule(d["model_eval"], "test"))
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {x}" for x in al.LIMITATIONS))
