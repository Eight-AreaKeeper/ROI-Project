"""Run Life dashboard. Layout and widgets only; numbers come from src/app_logic.py,
every chart from src/plots.py and the page building blocks from src/ui.py.
No number in this file is typed in.

    streamlit run app.py          (add ?page=watchlist to the URL to land on a page)
"""
import streamlit as st

from src import app_logic as al
from src import config, plots, ui

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
def model_facts():
    return al.model_facts(model(), runs(), al.as_of(watchlist()))


@st.cache_data
def rpm_range():
    return al.rpm_range(runs())


@st.cache_data
def well_daily(uwi, end):
    return al.well_daily(uwi, end)


chart = ui.chart
H = ui.CHART_HEIGHT


PAGES = {"summary": "Summary", "problem": "The problem", "drivers": "What drives failure",
         "workover": "Is a workover worth it?", "watchlist": "Watchlist", "proof": "Does it work?"}

# A ?page= link lands on that page; used for shared links and screenshots.
if ui.PAGE_KEY not in st.session_state:
    st.session_state[ui.PAGE_KEY] = PAGES.get(st.query_params.get("page", "summary"), "Summary")
# The oil value is set on the workover page but also prices the watchlist and
# the impact numbers, so it lives outside the slider's own state.
st.session_state.setdefault("oil_value", float(config.OIL_VALUE_PER_BBL))
oil = st.session_state["oil_value"]

d, wl, n = outputs(), watchlist(), numbers()
imp = impact(oil)
aliases = al.make_aliases(wl, replay().uwi)
k = config.TOP_K
ui.base_style()

with st.sidebar:
    page = st.radio("Page", list(PAGES.values()), key=ui.PAGE_KEY)
    hide = st.toggle("Hide well IDs", value=True, key="hide_ids")
    st.caption(f"Data as of {al.as_of(wl):%B %d, %Y}")

N = imp["workover_wells"]
cut_pct = f"−{n['workover_risk_cut']:.0%}"

# --------------------------------------------------------------------------
if page == "Summary":
    mf = model_facts()
    ui.page_header("Run Life", "Predicting which heavy-oil wells will fail again, and which repair keeps them "
                               "running longer.", top=True)
    ui.kpi_strip([
        ("Annual failure cost", f"~${imp['fleet_cost_per_year'] / 1e6:.0f}M",
         f"{n['reactive_share']:.0%} of repair spend is reactive"),
        ("Failures per year", f"~{round(n['failures_per_year'], -2):,.0f}", f"across {n['wells']:,} wells"),
        ("Run life after repeat failures", f"{n['days_2plus']:.0f} vs {n['days_none']:.0f} days",
         "2+ failures last year vs none"),
        ("Workover target wells", f"~{N} wells", f"workover pays back in year 1 · {cut_pct} risk"),
    ])

    with st.container(border=True):
        st.markdown("#### The model")
        st.markdown(al.MODEL_SENTENCE)
        steps = [
            ("What it reads", "failures in the past year  \nlength of the last run  \nhow it was last repaired  \n"
                              "pump speed, water cut, oil rate"),
            ("How it learned", f"{mf['train_runs']:,} past runs between failures ({mf['train_years']}), "
                               "keeping runs still going at the cut-off."),
            ("What it gives", f"a {config.HORIZON_DAYS}-day failure probability per well, plus its main risk driver."),
            ("What you do with it", "a ranked weekly watchlist with an expected cost and a suggested repair."),
        ]
        cols = st.columns([9, 1, 9, 1, 9, 1, 9], vertical_alignment="center")
        for i, (head, body) in enumerate(steps):
            with cols[2 * i].container(border=True, height=185):
                st.markdown(f"**{head}**  \n{body}")
            if i < len(steps) - 1:
                cols[2 * i + 1].markdown("<div style='text-align:center;font-size:1.6rem'>→</div>",
                                         unsafe_allow_html=True)
        st.caption(f"Tested on {mf['test_years']} runs and the official failure log, neither used in training: "
                   f"{n['model_hits']:.0f} of the top {k} failed within 90 days, against "
                   f"{n['rule_hits']:.0f} for today's rule.")

    left, right = st.columns([3, 2])
    with left:
        chart(plots.cost_per_year(d["sizing"], oil_value=oil))
        st.markdown("**What to do on Monday**")
        st.markdown(f"- **Review the top {k}:** check the watchlist every week.\n"
                    f"- **Workover, not a quick fix:** on the ~{N} target wells, at their next failure.\n"
                    "- **Check pump speed:** on high-RPM wells that keep failing.")
    with right:
        ui.insight_card(f"{n['days_2plus']:.0f} vs {n['days_none']:.0f} days",
                        f"Wells that failed 2+ times last year run {al.as_long(n['days_2plus'] / n['days_none'])} "
                        "after a repair.", PAGES["drivers"], key="go_drivers")
        ui.insight_card(f"{cut_pct} risk",
                        f"Wells fail {n['workover_risk_cut']:.0%} less often after a workover; it pays back in "
                        f"year 1 on the ~{N} target wells.", PAGES["workover"], key="go_workover")
        ui.insight_card(f"{n['model_hits']:.0f} of {k}",
                        f"Of our top {k} flagged wells, {n['model_hits']:.0f} failed within 90 days, vs "
                        f"{n['rule_hits']:.0f} for today's rule.", PAGES["watchlist"],
                        button="See the watchlist", key="go_watchlist")

    st.divider()
    ui.footer(f"Research question: {al.RESEARCH_QUESTION}")
    ui.footer("Method: survival analysis (Kaplan-Meier, Cox model) on events rebuilt from daily downtime codes. "
              "Details on 'Does it work?'.")

# --------------------------------------------------------------------------
elif page == "The problem":
    ui.page_header(f"10% of wells cause {al.share_words(n['top10_lost_oil'])} the lost oil.",
                   "How often wells fail, what it costs, and where the damage sits.")
    ui.kpi_strip([
        ("Failures per producing well-year", f"{n['failures_per_well_year']:.2f}",
         f"1 every {n['producing_days_per_failure']:.0f} producing days"),
        ("Fail again within 30 days", f"{n['repeat_30d']:.0%}", "of repairs"),
        ("Reactive repair spend", f"{n['reactive_share']:.0%}",
         f"of ${n['repair_spend'] / 1e6:.0f}M total repair spend"),
    ])
    ui.so_what("Focus on a short list of wells, not the whole fleet.")
    left, right = st.columns([3, 2])
    with left:
        chart(plots.concentration(d["pareto"]), height=H, hide_title=True)
    with right:
        chart(plots.cost_per_year(d["sizing"], oil_value=oil), height=H)
    with st.expander("More detail"):
        chart(plots.failures_per_year(events()))
    with st.expander("Technical detail"):
        st.markdown("- Failures are rebuilt from daily downtime codes: downhole days less than "
                    f"{config.EVENT_GAP_DAYS + 1} days apart count as one failure.\n"
                    "- Wells are ranked worst first by failures or by lost oil, over all in-scope producers.")

# --------------------------------------------------------------------------
elif page == "What drives failure":
    ui.page_header("Wells that keep failing, run fast or run wet fail sooner.",
                   "Producing days from a repair to the next failure, compared across groups of wells.")
    ui.kpi_strip([
        ("2+ failures last year", f"{n['days_2plus']:.0f} days", f"vs {n['days_none']:.0f} with none"),
        ("Fastest pumps", f"{n['days_fast']:.0f} days", f"vs {n['days_slow']:.0f} slowest"),
        ("After a workover", f"{n['days_workover']:.0f} days", f"vs {n['days_rodtruck']:.0f} after rod truck only"),
    ])
    ui.so_what("A well's last year predicts its next 90 days, so the history alone can rank wells.")
    left, right = st.columns([3, 2])
    with left:
        labels = {"history": "Past failures", "rpm": "Pump speed", "water_cut": "Water cut", "repair": "Last repair"}
        by = st.radio("Compare wells by", list(labels), format_func=labels.get, horizontal=True, key="by")
        chart(plots.run_life(d["km_curves"], d["km_summary"], by), height=H)
        st.caption("Each line shows the share of wells still running N days after a repair. "
                   "Lower and steeper means they break sooner.")
        st.caption("Rods are the weak link: each rotation loads the rod string, so faster pumps mean more cycles "
                   "and earlier fatigue.")
    with right:
        # In a narrow column the bar labels need more room on the left and a shorter axis title.
        drivers_fig = plots.risk_drivers(d["cox_results"])
        drivers_fig.update_xaxes(range=[-100, 45], title="change in risk (← longer runs · shorter →)")
        chart(drivers_fig, height=H + 60)
        st.caption("How much each factor changes the risk of failure, holding the others fixed.")
    with st.expander("Technical detail"):
        st.markdown(
            "- Run-life curves are Kaplan-Meier estimates; runs still going when the data ends are censored, "
            "not dropped. Group differences are tested with the log-rank test (all p < 0.001).\n"
            "- The factor chart is a Cox proportional-hazards model on runs before 2025. Hazard ratios are "
            "associations, not proof of cause.")

# --------------------------------------------------------------------------
elif page == "Is a workover worth it?":
    inputs = payback_inputs()
    ui.page_header(f"A workover pays for itself on wells that fail {plots.break_even_rate(inputs, oil):.0f}+ "
                   "times a year.", "Extra cost of a workover against the failures it avoids, at an oil value you set.")
    ui.kpi_strip([
        ("Failure risk after a workover", cut_pct, "vs rod truck only"),
        ("Extra cost of a workover", f"+${n['workover_extra_cost'] / 1e3:.0f}k", "vs a rod-truck job (median)"),
        ("Break-even", f"{plots.break_even_rate(inputs, oil):.1f} failures/yr", f"at C${oil:.2f}/bbl"),
    ])
    ui.so_what(f"Do a workover instead of a quick fix on the ~{N} target wells. It's not worth it fleet-wide.")
    left, right = st.columns([3, 2])
    with left:
        # The callback stores the new value before the rerun, so the header,
        # tiles and "So what" above already use it.
        st.slider("Oil value, C$/bbl", 35.0, 70.0, oil, key="oil",
                  on_change=lambda: st.session_state.update(oil_value=st.session_state["oil"]))
        payback_fig = plots.workover_payback(inputs, oil_value=oil)
        # At this width the 1-year label collides with the break-even label, so it moves to the left end.
        payback_fig.update_annotations(selector=dict(text="pays back within a year"), x=0, xanchor="left")
        chart(payback_fig, height=H, hide_title=True)
        st.caption("A workover replaces worn tubing and downhole equipment; a rod-truck job only swaps rods and "
                   "pump, so the underlying wear stays.")
    with right:
        chart(plots.run_life(d["km_curves"], d["km_summary"], "repair"), height=H + 75)
    with st.expander("Technical detail"):
        st.markdown(ui.md_safe(
            f"- One failure costs the median rod-truck repair (${inputs['rodtruck_cost']:,.0f}) plus the lost oil "
            f"(fleet mean {inputs['oil_m3_per_event']:.1f} m³ per failure × 6.29 bbl/m³ × the oil value).\n"
            f"- Extra cost = median workover (${inputs['workover_cost']:,.0f}) minus median rod-truck job. The Cox "
            f"hazard ratio for a workover is {inputs['hazard_ratio']:.3f}, so a well failing f times a year avoids "
            f"f × {1 - inputs['hazard_ratio']:.2f} failures a year.\n"
            "- Payback = extra cost / (avoided failures × cost per failure). This is an association from past "
            "runs, not a controlled trial.\n"
            f"- Target wells: open wells whose last repair was not already a workover or new rod string, and where "
            f"one year of avoided failures at their own cost per failure beats the extra cost: {N} wells, "
            f"~{imp['failures_avoided']:.0f} failures avoided a year.\n"
            f"- Year-one net saving on those wells: about ${imp['net_savings_year1'] / 1e6:.1f}M after paying for "
            "the workovers; the benefit continues after year one."))

# --------------------------------------------------------------------------
elif page == "Watchlist":
    ew = enriched_watchlist(oil)
    # The header, tiles and "So what" describe the filtered table, so they are
    # drawn into this slot after the filters below have been read.
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
    cost_top = top["expected cost, 90 days"].sum()
    with head:
        ui.page_header(f"The {len(top)} wells to act on this week.",
                       "Wells in a run today, ranked by expected failure cost over the next 90 producing days.")
        ui.kpi_strip([
            ("Wells flagged", f"{len(top)}", f"of {int(ew.risk_90.notna().sum()):,} wells scored"),
            ("Expected failures", f"{top.risk_90.sum():.0f} of {len(top)}", "in the next 90 days"),
            ("Expected failure cost", f"${cost_top / 1e3:,.0f}k", f"next 90 days at C${oil:.0f}/bbl"),
        ])
        ui.so_what(f"These {len(top)} wells carry ${cost_top / 1e3:,.0f}k of expected failure cost in the next "
                   f"90 days; {n_wo} should get a workover at their next failure.")

    default_cols = ["well", "area", "risk_90", "expected cost, 90 days", "main driver", "suggested action"]
    # Shown as whole percent and dollars with separators; the download keeps
    # full precision. Money is preformatted text because Streamlit's number
    # formats cannot add thousands separators without also adding cents.
    shown = table.assign(risk_90=table.risk_90 * 100)
    for c in ["expected cost, 90 days", "cost per failure"]:
        shown[c] = shown[c].map(lambda v: f"${v:,.0f}")
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_order=None if all_cols else default_cols, column_config={
                     "risk_90": st.column_config.ProgressColumn("risk %", format="%.0f%%",
                                                                min_value=0.0, max_value=100.0),
                     "workover payback, yrs": st.column_config.NumberColumn(format="%.1f"),
                 })
    st.caption(f"{len(table):,} wells, sorted by {sort}; expected cost = risk × cost of one failure at "
               f"C${oil:.2f}/bbl. {int(wl.beyond_model_range.sum())} wells have run longer than the model has "
               "data for and are not scored.")
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
            chart(plots.well_outlook(al.well_survival(model(), run), elapsed, label=name), height=H)
        with c2.container(border=True):
            st.markdown("**After its next repair: what if?**")
            lo, hi = rpm_range()
            now_rpm = al.current_rpm(run, model())
            repair = st.radio("Repair at the next failure", list(al.REPAIRS), horizontal=True, key="wi_repair")
            rpm = st.slider("Pump speed after restart, rpm", lo, hi, min(max(now_rpm, lo), hi), 1.0, format="%.0f",
                            key=f"wi_rpm_{uwi}")  # per well, so it starts at that well's speed
            wi = al.whatif(run, model(), events(), end, repair, rpm).set_index("option")
            st.markdown(al.whatif_sentence(wi))
            chart(plots.repair_options(wi.reset_index()))
            st.markdown(ui.md_safe(al.whatif_note(ew.set_index("uwi").loc[uwi], payback_inputs())))
            st.caption("Model estimate based on similar past runs; associations, not guaranteed effects.")
        with st.expander("More detail"):
            chart(plots.well_history(well_daily(uwi, end), al.well_events(events(), uwi, end)))
            chart(plots.risk_distribution(wl, top_n=k))
    with st.expander("Technical detail"):
        st.markdown(
            "- Risk is the model's chance of failing within the next 90 producing days, given the producing days "
            "the well has already run.\n"
            "- The main driver is the measured factor adding most to this well's risk; the suggested action is "
            "the first matching rule: workover if it pays back within 1.5 years, review pump speed, pre-plan a "
            "crew at 70%+ risk, else monitor.")

# --------------------------------------------------------------------------
elif page == "Does it work?":
    mf = model_facts()
    ui.page_header(f"On failures it never saw, our top {k} caught {n['model_hits']:.0f}; "
                   f"today's rule caught {n['rule_hits']:.0f}.",
                   "Our top 50 against today's rule, checked on the official failure log the model never used.")
    ui.kpi_strip([
        ("Our model", f"{n['model_hits']:.0f} of {k}", "failed within 90 days"),
        ("Today's rule", f"{n['rule_hits']:.0f} of {k}", f"range {n['rule_min']:.0f} to {n['rule_max']:.0f}"),
        ("Random wells", f"{n['random_hits']:.0f} of {k}", "for comparison"),
    ])
    ui.so_what(f"Same {k} well visits, about {imp['extra_hits']:.0f} more of the right wells: "
               f"{imp['lift_pct']:.0%} better targeting.")
    left, right = st.columns([3, 2])
    with left:
        chart(plots.model_vs_rule(d["model_eval"], "official"), height=H, hide_title=True)
    with right:
        chart(plots.calibration(d["calibration"]), height=H + 55)
        st.caption("When we say 70%, about 70% fail.")

    st.subheader("Replay: what the watchlist would have said")
    st.caption("Each date is after the model was built; every open well is scored as it stood that day and "
               "checked against the official log for the next 90 days.")
    summ, rp = d["replay_summary"], replay()
    dates = [str(x) for x in summ.date]
    day = st.select_slider("Check date", dates, value=dates[0], key="replay_date")
    r = summ[summ.date.astype(str) == day].iloc[0]
    top = rp[rp.date.astype(str) == day].assign(label=lambda x: al.show_id(x.uwi, aliases, hide))
    g1, g2 = st.columns([3, 2])
    with g1:
        chart(plots.replay_top50(top, top_n=k), fixed_width=520)
    with g2:
        ui.kpi_strip([
            ("Our model", f"{r.model_top50_hits:.0f}", f"of {k}"),
            ("Today's rule", f"{r.rule_top50_mean:.0f}", f"range {r.rule_top50_min:.0f} to {r.rule_top50_max:.0f}"),
            ("Random 50", f"{r.random_top50:.0f}", f"{r.base_rate:.0%} base rate"),
        ])
        chart(plots.replay_trend(summ, top_n=k))

    st.subheader("How this runs in production")
    st.markdown(ui.md_safe(al.PRODUCTION_NOTES))

    with st.expander("More detail"):
        chart(plots.what_breaks(d["event_validation"]))
        st.caption(f"Our failure records match the official log {n['recall']:.0%} of the time.")
    with st.expander("Technical detail"):
        st.markdown(f"**Research question:** {al.RESEARCH_QUESTION}")
        st.markdown(f"**The model:** {al.MODEL_SENTENCE}")
        st.markdown(
            f"- **Time split:** trained on {mf['train_runs']:,} runs that started before {config.TIME_CUT} "
            f"(runs still open at the cut are censored there), tested on the {n['test_runs']:,} runs that "
            "started after.\n"
            f"- **Ranking (C-index):** {n['c_index']:.3f} for the model vs {n['c_index_rule']:.3f} for today's rule.\n"
            f"- **Official-log check ({config.INDEPENDENT_CHECK_DATE}, {n['check_wells']} wells):** AUC "
            f"{n['auc']:.3f} vs {n['auc_rule']:.3f} for the rule.\n"
            f"- **Probability accuracy (Brier, 90 days):** {n['brier']:.3f} vs {n['brier_km']:.3f} for a flat "
            "Kaplan-Meier rate (lower is better).\n"
            "- **Model:** Cox proportional hazards on pre-run features, scored as the chance of failing in the "
            "next 90 producing days given the days already survived.")
        chart(plots.model_vs_rule(d["model_eval"], "test"))
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {x}" for x in al.LIMITATIONS))
