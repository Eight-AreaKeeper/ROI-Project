"""Every chart in the project, built with Plotly.

The notebook, the README figures and the Streamlit app all call these
functions, so a number can never differ between them. Each function takes
DataFrames (usually from load_all) and returns a plotly Figure.

    figs = plots.load_all()
    st.plotly_chart(plots.run_life(figs["km_curves"], figs["km_summary"], "history"))
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Colours. Categorical slots (blue, orange, aqua) are a validated
# colour-blind-safe set; use at most three in one chart. Ordered groups
# (RPM quartiles, failure history, water cut) use one blue ramp, light to
# dark, because the groups have an order. Grey is "the comparison".
# --------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_LIGHT = "#8a8985", "#d6d5d0"
INK, INK_2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
RAMP = {3: ["#86b6ef", "#2a78d6", "#104281"],
        4: ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]}

BBL_PER_M3 = 6.29
YEARS = 5.815          # 2021-01-01 to 2026-10-26
DEFAULT_OIL = 51.04    # C$/bbl, Strathcona Q3 2026 netback (docs/methods.md)


def _style(fig, title, subtitle=None, height=420):
    """One look for every chart: light surface, recessive grid, left title."""
    # Two "$" in one string make the image export render it as LaTeX, so
    # dollar signs are written as an HTML entity.
    title = title.replace("$", "&#36;")
    text = f"<b>{title}</b>"
    if subtitle:
        subtitle = subtitle.replace("$", "&#36;")
        text += f"<br><span style='font-size:13px;color:{INK_2}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=text, x=0, xanchor="left", font=dict(size=18, color=INK)),
        height=height, margin=dict(l=60, r=30, t=80 if subtitle else 55, b=55),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, Arial, sans-serif", size=13, color=INK_2),
        hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=GREY_LIGHT, ticks="outside",
                     tickcolor=GREY_LIGHT, zeroline=False)
    fig.update_yaxes(gridcolor="#ecebe7", zeroline=False, linecolor=GREY_LIGHT)
    return fig


def load_all(out_dir="outputs"):
    """Read every public output CSV into a dict keyed by file stem."""
    out = Path(out_dir)
    return {p.stem: pd.read_csv(p) for p in sorted(out.glob("*.csv"))}


# ==========================================================================
# Page 1. Overview: how big is the problem?
# ==========================================================================
def failures_per_year(events, last_day="2026-10-26"):
    """Bar per year of rebuilt downhole events. 2026 is marked as partial."""
    yr = pd.to_datetime(events["ev_start"]).dt.year.value_counts().sort_index()
    last = pd.Timestamp(last_day)
    colors = [BLUE] * len(yr)
    labels = [f"{v:,}" for v in yr.values]
    if yr.index[-1] == last.year:
        colors[-1] = "#9ec5f4"
        labels[-1] += f"<br>(to {last:%b %d})"
    fig = go.Figure(go.Bar(
        x=yr.index.astype(str), y=yr.values, marker_color=colors,
        text=labels, textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:,} failures<extra></extra>"))
    fig.update_yaxes(title="downhole failures", rangemode="tozero")
    return _style(fig, "Downhole failures every year",
                  "Rebuilt from daily downtime codes, in-scope heavy-oil wells")


def concentration(pareto, denominator="all_in_scope_producers"):
    """Lorenz-style curve: the worst X% of wells carry Y% of the damage."""
    p = pareto[pareto.denominator == denominator]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                             line=dict(color=GREY, dash="dot", width=1.5),
                             name="if spread evenly", hoverinfo="skip"))
    for measure, color, name in [("failures", BLUE, "failures"),
                                 ("lost_oil_m3", ORANGE, "lost oil")]:
        g = p[p.measure == measure].sort_values("top_pct_wells")
        x = np.r_[0, g.top_pct_wells]
        y = np.r_[0, g.cumulative_share * 100]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=name,
            line=dict(color=color, width=2.5), marker=dict(size=8),
            hovertemplate=f"worst %{{x}}% of wells → %{{y:.0f}}% of {name}<extra></extra>"))
        y10 = float(g.loc[g.top_pct_wells == 10, "cumulative_share"].iloc[0]) * 100
        fig.add_annotation(x=10, y=y10, text=f"<b>{y10:.0f}%</b> of {name}",
                           showarrow=True, arrowhead=0, arrowcolor=color,
                           ax=-45 if measure == "lost_oil_m3" else 60,
                           ay=-35 if measure == "lost_oil_m3" else 40,
                           font=dict(color=color, size=13))
    fig.add_vline(x=10, line=dict(color=GREY_LIGHT, width=1))
    fig.update_xaxes(title="worst % of wells", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(title="share of the total", range=[0, 102], ticksuffix="%")
    n = int(p.wells_in_population.iloc[0])
    return _style(fig, "10% of wells cause half the lost oil",
                  f"Wells ranked worst first, {n:,} in-scope producers")


def annual_cost(sizing, oil_value=DEFAULT_OIL, years=YEARS):
    """Reactive repair spend and lost-oil value per year, in C$ millions."""
    s = sizing.set_index("metric").value
    return (s["reactive_spend"] / years / 1e6,
            s["downhole_lost_oil_bbl"] * oil_value / years / 1e6)


def cost_per_year(sizing, oil_value=DEFAULT_OIL, years=YEARS):
    """Single stacked bar: reactive repair spend + lost oil value, $M/year."""
    repairs, oil = annual_cost(sizing, oil_value, years)
    fig = go.Figure()
    for val, name, color in [(repairs, "reactive repairs", BLUE),
                             (oil, f"lost oil at C${oil_value:.0f}/bbl", ORANGE)]:
        fig.add_trace(go.Bar(
            y=["per year"], x=[val], orientation="h", name=name,
            marker=dict(color=color, line=dict(color=SURFACE, width=2)),
            text=f"{name}<br><b>${val:.1f}M</b>", textposition="inside",
            insidetextanchor="middle", textfont=dict(color="white"),
            hovertemplate=f"{name}: $%{{x:.1f}}M a year<extra></extra>"))
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_xaxes(title="C$ millions a year (est.)", tickprefix="$", ticksuffix="M")
    fig.update_yaxes(showticklabels=False)
    return _style(fig, f"Downhole failures cost about ${repairs + oil:.0f}M a year",
                  "Reactive rod-truck and workover spend, plus oil not produced", height=230)


# ==========================================================================
# Page 2. What drives run life? (the survival-analysis core)
# ==========================================================================
RUN_LIFE_GROUPS = {
    "history": ("fails_prior_365", "Failures in the past year",
                {"0": "no failures", "1": "1 failure", "2+": "2 or more"}),
    "rpm": ("rpm_quartile", "Pump speed (RPM quartile)", None),
    "water_cut": ("water_cut_band", "Water cut",
                  {"<0.7": "below 70%", "0.7-0.9": "70 to 90%", "0.9+": "above 90%"}),
    "repair": ("repair_class", "Last repair", None),
}
_ORDER = {"water_cut_band": ["<0.7", "0.7-0.9", "0.9+"],
          "repair_class": ["rod truck only", "no linked job", "workover"]}


def _rpm_label(g):
    lo, hi = g.strip("(]").split(",")
    return f"{float(lo):.0f} to {float(hi):.0f} rpm"


def run_life(km_curves, km_summary, by="history", max_days=365, show_ci=True):
    """Kaplan-Meier curves: share of wells still running after N producing days.

    by: history | rpm | water_cut | repair. Ordered groups use a blue ramp;
    repair type uses categorical colours. Each curve is labelled with its
    median run life, and a dashed line marks the 50% level the medians read off.
    """
    var, title, names = RUN_LIFE_GROUPS[by]
    k = km_curves[(km_curves.group_var == var) & (km_curves.t_days <= max_days)]
    summ = km_summary[km_summary.group_var == var].set_index("group")
    groups = _ORDER.get(var, sorted(k.group.unique(), key=lambda g: (len(str(g)), str(g))))
    if var == "rpm_quartile":
        groups = sorted(k.group.unique(), key=lambda g: float(g.strip("(").split(",")[0]))
        colors = RAMP[4]
    elif var == "repair_class":
        colors = [ORANGE, GREY, BLUE]
    else:
        colors = RAMP[len(groups)]

    fig = go.Figure()
    fig.add_hline(y=50, line=dict(color=GREY_LIGHT, dash="dash", width=1))
    for g, color in zip(groups, colors):
        d = k[k.group == g].sort_values("t_days")
        label = names.get(g, g) if names else (_rpm_label(g) if var == "rpm_quartile" else g)
        med = summ.loc[g, "median_prod_days"] if g in summ.index else np.nan
        n = int(summ.loc[g, "n_runs"]) if g in summ.index else 0
        if show_ci:
            fig.add_trace(go.Scatter(
                x=np.r_[d.t_days, d.t_days[::-1]],
                y=np.r_[d.ci_high, d.ci_low[::-1]] * 100,
                fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0),
                line_shape="hv", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=d.t_days, y=d.survival * 100, mode="lines", line_shape="hv",
            line=dict(color=color, width=2.5),
            name=f"{label} · median {med:.0f} days ({n:,} runs)",
            customdata=d.n_at_risk,
            hovertemplate=(f"<b>{label}</b><br>day %{{x:.0f}}: %{{y:.0f}}% still running"
                           "<br>%{customdata:,} runs still observed<extra></extra>")))
        if med <= max_days:
            fig.add_trace(go.Scatter(
                x=[med], y=[50], mode="markers", marker=dict(color=color, size=9,
                line=dict(color=SURFACE, width=2)), showlegend=False,
                hovertemplate=f"{label}: half have failed by day {med:.0f}<extra></extra>"))
    fig.update_xaxes(title="producing days since the repair", range=[0, max_days])
    fig.update_yaxes(title="wells still running", range=[0, 101], ticksuffix="%")
    _style(fig, f"Run life by {title.lower()}",
           "Kaplan-Meier: runs still going when the data ends are counted, not dropped",
           height=460)
    fig.update_layout(legend=dict(orientation="v", y=0.98, x=0.98, xanchor="right",
                                  yanchor="top", bgcolor="rgba(252,252,251,0.9)"))
    return fig


DRIVER_LABELS = {
    "workover": "Workover as last repair",
    "new_string": "New rod string",
    "fails_prior_365": "Failures in past year (+2.7)",
    "oil_pre30": "Oil rate (higher)",
    "rpm_pre30": "Pump speed (+124 rpm)",
    "wcut_pre30": "Water cut (+17 pts)",
    "disp": "Pump displacement (larger)",
    "util_pre30": "Pump utilisation (higher)",
    "torque_pre30": "Rod torque (higher)",
    "prev_gap_days": "Previous run (longer)",
    "no_prev_gap": "First run on record",
    "prev_down_days": "Previous downtime (longer)",
    "corod": "Continuous rod",
    "stick": "Stick rods",
}


def risk_drivers(cox, only_significant=False):
    """Forest plot as % change in failure risk, with 95% CI.

    Bars left of zero (blue) mean longer runs; right (orange) shorter runs.
    Not-significant factors are grey. Yes/no factors compare yes vs no;
    measured factors are per one standard deviation.
    """
    c = cox.copy()
    if only_significant:
        c = c[c.significant]
    c["pct"] = (c.hazard_ratio - 1) * 100
    c["lo"] = (c.ci_low - 1) * 100
    c["hi"] = (c.ci_high - 1) * 100
    c = c.sort_values("pct")
    c["label"] = c.feature.map(DRIVER_LABELS).fillna(c.feature)
    c["color"] = np.where(~c.significant, GREY_LIGHT, np.where(c.pct < 0, BLUE, ORANGE))
    scale = c.get("scale", pd.Series("", index=c.index)).fillna("")
    fig = go.Figure(go.Scatter(
        x=c.pct, y=c.label, mode="markers",
        marker=dict(color=c.color, size=11, line=dict(color=SURFACE, width=1.5)),
        error_x=dict(type="data", symmetric=False, array=c.hi - c.pct,
                     arrayminus=c.pct - c.lo, color=GREY, thickness=1.5, width=0),
        customdata=np.c_[c.hazard_ratio, c.ci_low, c.ci_high, scale],
        hovertemplate=("<b>%{y}</b><br>%{x:+.0f}% failure risk"
                       "<br>hazard ratio %{customdata[0]:.2f} (%{customdata[1]:.2f} to %{customdata[2]:.2f})"
                       "<br>%{customdata[3]}<extra></extra>")))
    for _, r in c[c.significant & (c.pct.abs() >= 20)].iterrows():
        fig.add_annotation(x=r.lo if r.pct < 0 else r.hi, y=r.label, text=f"<b>{r.pct:+.0f}%</b>",
                           showarrow=False, xanchor="right" if r.pct < 0 else "left",
                           xshift=-6 if r.pct < 0 else 6, font=dict(color=r.color))
    fig.add_vline(x=0, line=dict(color=INK_2, width=1))
    fig.update_xaxes(title="change in failure risk  (← longer runs · shorter runs →)",
                     ticksuffix="%", range=[-75, 45])
    fig.update_yaxes(showgrid=False)
    return _style(fig, "What goes with shorter and longer runs",
                  "Cox model, each factor holding the others fixed; grey = not significant",
                  height=60 + 34 * len(c))


# ==========================================================================
# Page 3. Is a workover worth it?
# ==========================================================================
def payback_inputs(runs, events, cox):
    """Median repair costs, lost oil per event and the workover hazard ratio."""
    paid = runs[runs.repair_cost > 0]
    wo = paid.loc[paid.workover == 1, "repair_cost"].median()
    rt = paid.loc[(paid.rodtruck == 1) & (paid.workover == 0), "repair_cost"].median()
    return dict(workover_cost=float(wo), rodtruck_cost=float(rt),
                oil_m3_per_event=float(events.lost_oil_m3.mean()),
                hazard_ratio=float(cox.set_index("feature").loc["workover", "hazard_ratio"]))


def payback_years(fails_per_year, workover_cost, rodtruck_cost, oil_m3_per_event,
                  hazard_ratio, oil_value=DEFAULT_OIL):
    """Years for a workover's extra cost to pay back through avoided failures."""
    per_failure = rodtruck_cost + oil_m3_per_event * BBL_PER_M3 * oil_value
    saved_per_year = np.asarray(fails_per_year) * (1 - hazard_ratio) * per_failure
    return (workover_cost - rodtruck_cost) / saved_per_year, per_failure


def break_even_rate(inputs, oil_value=DEFAULT_OIL):
    """Failures per year at which a workover's extra cost pays back in one year."""
    _, per_failure = payback_years([1.0], oil_value=oil_value, **inputs)
    return (inputs["workover_cost"] - inputs["rodtruck_cost"]) / ((1 - inputs["hazard_ratio"]) * per_failure)


def workover_payback(inputs, oil_value=DEFAULT_OIL, fleet_rate=2.15, max_rate=10):
    """Payback time against how often a well fails, with the 1-year line.

    The app's oil-price slider just calls this again with a new oil_value.
    """
    f = np.linspace(0.75, max_rate, 200)
    yrs, per_failure = payback_years(f, oil_value=oil_value, **inputs)
    extra = inputs["workover_cost"] - inputs["rodtruck_cost"]
    be = break_even_rate(inputs, oil_value)
    fleet_yrs = payback_years([fleet_rate], oil_value=oil_value, **inputs)[0][0]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=1, fillcolor=AQUA, opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(
        x=f, y=np.minimum(yrs, 8), mode="lines", line=dict(color=BLUE, width=3),
        hovertemplate="%{x:.1f} failures a year → pays back in %{y:.1f} years<extra></extra>",
        showlegend=False))
    fig.add_hline(y=1, line=dict(color=AQUA, width=1.5, dash="dash"))
    fig.add_annotation(x=max_rate, y=1, text="pays back within a year", showarrow=False,
                       xanchor="right", yshift=10, font=dict(color="#127a55"))
    for x, y, txt, col in [(be, 1, f"break-even<br><b>{be:.1f} failures/yr</b>", INK),
                           (fleet_rate, fleet_yrs, f"fleet average ({fleet_rate})<br><b>{fleet_yrs:.1f} years</b>", GREY)]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=11, color=col, line=dict(color=SURFACE, width=2))))
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, xanchor="left", xshift=10,
                           yshift=18, align="left", font=dict(color=col))
    fig.update_xaxes(title="how often the well fails (failures per year)", range=[0, max_rate])
    fig.update_yaxes(title="years to pay back", range=[0, 8])
    return _style(fig, f"A workover pays back on wells failing {be:.0f}+ times a year",
                  f"Extra cost ${extra/1e3:,.0f}k · one failure ${per_failure/1e3:,.1f}k at C${oil_value:.0f}/bbl"
                  f" · {1 - inputs['hazard_ratio']:.0%} fewer failures after a workover")


# ==========================================================================
# Page 4. Watchlist
# ==========================================================================
def risk_distribution(watchlist, top_n=50):
    """Histogram of 90-day risk across open wells; the top-N cut is marked."""
    r = watchlist.risk_90.dropna() * 100
    cut = r.sort_values(ascending=False).iloc[top_n - 1]
    fig = go.Figure()
    for mask, color, name in [(r < cut, GREY_LIGHT, "other open wells"),
                              (r >= cut, ORANGE, f"top {top_n} (watchlist)")]:
        fig.add_trace(go.Histogram(x=r[mask], xbins=dict(start=0, end=100, size=2.5),
                                   marker=dict(color=color, line=dict(color=SURFACE, width=1)),
                                   name=name, hovertemplate="%{x}% risk: %{y} wells<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="chance of failing in the next 90 producing days", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(title="open wells")
    return _style(fig, f"Where the {len(r):,} open wells sit on risk",
                  f"Watchlist = the {top_n} wells at {cut:.0f}% risk or more")


def well_outlook(survival, elapsed_days, horizon=180, label="this well"):
    """One well's chance of failing from today, given it has run elapsed_days.

    survival: pd.Series of S(t) indexed by producing day t, e.g.
    cph.predict_survival_function(row).iloc[:, 0]. Plots
    1 - S(elapsed + x) / S(elapsed) for x = 0..horizon, and marks 90 days.
    """
    s = survival.sort_index()
    t = np.arange(0, horizon + 1)
    s_now = np.interp(elapsed_days, s.index, s.values)
    s_fut = np.interp(elapsed_days + t, s.index, s.values)
    risk = (1 - s_fut / s_now) * 100
    r90 = risk[90] if horizon >= 90 else np.nan
    fig = go.Figure(go.Scatter(
        x=t, y=risk, mode="lines", line=dict(color=ORANGE, width=3), fill="tozeroy",
        fillcolor="rgba(235,104,52,0.10)",
        hovertemplate="within %{x} days: %{y:.0f}% chance of failure<extra></extra>"))
    fig.add_vline(x=90, line=dict(color=GREY, dash="dash", width=1))
    fig.add_trace(go.Scatter(x=[90], y=[r90], mode="markers+text", text=[f"<b>{r90:.0f}%</b> in 90 days"],
                             textposition="middle left", marker=dict(size=10, color=ORANGE),
                             showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title="producing days from today", range=[0, horizon])
    fig.update_yaxes(title="chance it has failed", ticksuffix="%", range=[0, 100])
    fig.update_layout(showlegend=False)
    return _style(fig, f"Outlook for {label}", f"Already running {elapsed_days:.0f} producing days",
                  height=340)


def repair_options(df):
    """One bar per repair option: 90-day risk from restart, chosen option in blue.

    df: option, risk_90, chosen (from app_logic.whatif; a "baseline" row is ignored).
    """
    d = df[df.option != "baseline"].iloc[::-1]
    fig = go.Figure(go.Bar(
        y=d.option, x=d.risk_90 * 100, orientation="h",
        marker_color=[BLUE if c else GREY_LIGHT for c in d.chosen],
        text=[f"<b>{x:.0f}%</b>" for x in d.risk_90 * 100], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.0f}% chance of failing within 90 days<extra></extra>"))
    fig.update_xaxes(title="chance of failing in the first 90 producing days", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(showlegend=False)
    return _style(fig, "Risk after the next repair", "Same well, same pump speed, different repair", height=260)


def well_history(daily, events=None):
    """Last 12 months of one well: oil rate, with downhole events shaded.

    daily: rows for one well with production_date, oil_produced_m3, is_downhole.
    """
    d = daily.sort_values("production_date")
    fig = go.Figure(go.Scatter(
        x=d.production_date, y=d.oil_produced_m3, mode="lines", line=dict(color=BLUE, width=1.5),
        hovertemplate="%{x|%b %d %Y}: %{y:.1f} m3<extra></extra>", name="oil"))
    if events is not None:
        for _, e in events.iterrows():
            fig.add_vrect(x0=e.ev_start, x1=e.ev_end, fillcolor=ORANGE, opacity=0.25, line_width=0)
    fig.update_yaxes(title="oil, m3/day", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _style(fig, "Last 12 months", "Orange = downhole failure", height=300)


# ==========================================================================
# Page 5. Does it work?
# ==========================================================================
def model_vs_rule(model_eval, check="official"):
    """Top-50 hits: model vs today's rule vs picking 50 wells at random.

    check: "official" (unseen failure log) or "test" (held-out runs).
    The rule's bar shows the range over tie orders.
    """
    m = model_eval.set_index(["model", "metric"])
    if check == "official":
        metric, base = "indep_top50_hits", m.loc[("independent_check", "base_rate"), "value"]
        sub = "Official failure log, never used in training: did the well fail within 90 days?"
    else:
        metric, base = "top50_hits", None
        sub = "Held-out 2025 to 2026 runs: did the run fail within 90 producing days?"
    cox = m.loc[("cox", metric), "value"]
    rule = m.loc[("baseline_a", metric)]
    xs, ys, cols, err_hi, err_lo = ["Our model", "Today's rule<br>(most failures last year)"], \
        [cox, rule.value], [BLUE, GREY], [0, rule.ci_high - rule.value], [0, rule.value - rule.ci_low]
    if base is not None:
        xs.append("50 wells<br>at random"); ys.append(base * 50); cols.append(GREY_LIGHT)
        err_hi.append(0); err_lo.append(0)
    fig = go.Figure(go.Bar(
        x=xs, y=ys, marker_color=cols, width=0.55,
        text=[f"<b>{y:.0f}</b> / 50" for y in ys], textposition="inside",
        insidetextanchor="start",
        textfont=dict(color=["white", "white", INK][:len(ys)], size=16),
        hovertemplate="%{x}: %{y:.1f} of 50 failed<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[xs[1]], y=[rule.value], mode="markers", marker=dict(size=1, color=INK_2),
        error_y=dict(type="data", array=[err_hi[1]], arrayminus=[err_lo[1]],
                     color=INK_2, thickness=1.5, width=10),
        hovertemplate=f"tie orders: {rule.ci_low:.0f} to {rule.ci_high:.0f}<extra></extra>"))
    fig.add_annotation(x=xs[1], y=rule.ci_high, text=f"range {rule.ci_low:.0f} to {rule.ci_high:.0f}",
                       showarrow=False, yshift=12, font=dict(color=INK_2, size=12))
    fig.update_yaxes(title="of the 50 wells flagged, how many failed", range=[0, 50])
    fig.update_layout(showlegend=False)
    return _style(fig, f"Our top 50 caught {cox:.0f} failures; today's rule caught {rule.value:.0f}",
                  sub + " Rule range = tie orders.")


def calibration(cal):
    """Predicted vs observed 90-day failure share, by risk decile."""
    c = cal[cal.model == "cox"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=GREY, dash="dot"),
                             name="perfect", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=c.predicted * 100, y=c.observed * 100, mode="markers+lines", name="our model",
        line=dict(color=BLUE, width=1.5), marker=dict(size=10, color=BLUE, line=dict(color=SURFACE, width=2)),
        customdata=np.c_[c.decile, c.n],
        hovertemplate=("risk group %{customdata[0]}: we said %{x:.0f}%, "
                       "%{y:.0f}% failed (%{customdata[1]} runs)<extra></extra>")))
    gap = (c.predicted - c.observed).abs().max() * 100
    fig.update_xaxes(title="predicted chance of failing in 90 days", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(title="share that actually failed", ticksuffix="%", range=[0, 100])
    return _style(fig, "When the model says 70%, about 70% fail",
                  f"Held-out runs in ten risk groups; every group within {gap:.0f} points", height=440)


def what_breaks(validation, gap_days=4):
    """How many logged failures our rebuilt events catch, by failure type."""
    v = validation[(validation.metric == "log_to_event_recall") & (validation.gap_days == gap_days)
                   & validation.group.isin(["Rods", "Pump", "Tubing", "all"])].copy()
    v["label"] = v.group.replace({"all": "All failures"})
    v["order"] = v.group.map({"Tubing": 0, "Pump": 1, "Rods": 2, "all": 3})
    v = v.sort_values("order")
    fig = go.Figure(go.Bar(
        y=v.label, x=v.value * 100, orientation="h",
        marker_color=[INK_2 if g == "all" else BLUE for g in v.group],
        text=[f"{x:.0%} of {int(d):,}" for x, d in zip(v.value, v.denominator)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.0f}% matched<extra></extra>"))
    fig.update_xaxes(title="logged failures matched by a rebuilt event", ticksuffix="%", range=[0, 100])
    fig.update_layout(showlegend=False)
    return _style(fig, "Our rebuilt failures match the official log",
                  "Logged failures since July 2025 with an event within 7 days", height=300)


def replay_top50(df, top_n=50, per_row=10):
    """The model's top-N wells on a past date as a grid, in rank order.

    df: rank, risk, failed, label (well name or alias), for one date.
    Orange squares failed within 90 days of the date; grey did not.
    """
    d = df.sort_values("rank").head(top_n).reset_index(drop=True)
    col, row = d.index % per_row, d.index // per_row
    hits = int(d.failed.sum())
    fig = go.Figure(go.Scatter(
        x=col, y=row, mode="markers",
        marker=dict(symbol="square", size=34, color=np.where(d.failed == 1, ORANGE, GREY_LIGHT),
                    line=dict(color=SURFACE, width=2)),
        customdata=np.c_[d["rank"], d.label, d.risk * 100, np.where(d.failed == 1, "failed", "did not fail")],
        hovertemplate=("#%{customdata[0]} %{customdata[1]}<br>risk %{customdata[2]:.0f}%"
                       "<br>%{customdata[3]} within 90 days<extra></extra>")))
    fig.update_xaxes(visible=False, range=[-0.6, per_row - 0.4])
    fig.update_yaxes(visible=False, range=[(top_n - 1) // per_row + 0.6, -0.6])
    fig.update_layout(showlegend=False)
    return _style(fig, f"{hits} of our top {top_n} failed within 90 days",
                  "Ranked left to right, top row first; orange = failed (official log)", height=330)


def replay_trend(summary, top_n=50):
    """Model vs today's rule top-N hits on every replay date, rule range shaded."""
    s = summary.sort_values("date")
    x = pd.to_datetime(s.date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.r_[x, x[::-1]], y=np.r_[s.rule_top50_max, s.rule_top50_min[::-1]], mode="lines", fill="toself",
        fillcolor=GREY_LIGHT, opacity=0.5, line=dict(width=0), hoverinfo="skip", name="rule range (tie orders)"))
    for col, name, color in [("rule_top50_mean", "today's rule (mean)", GREY),
                             ("model_top50_hits", "our model", BLUE)]:
        fig.add_trace(go.Scatter(
            x=x, y=s[col], mode="lines+markers", name=name, line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(color=SURFACE, width=2)),
            hovertemplate=f"%{{x|%b %d %Y}}: {name} %{{y:.1f}} of {top_n}<extra></extra>"))
    fig.update_xaxes(tickformat="%b %Y", tickvals=x)
    fig.update_yaxes(title=f"failures caught in the top {top_n}", range=[0, top_n])
    return _style(fig, "The gap holds on every replay date",
                  f"Top {top_n} on four dates after the training cut, official failure log", height=320)


# ==========================================================================
# Export the README / slide figures (no well IDs in any of these)
# ==========================================================================
def export_static(out_dir="outputs", fig_dir="docs/figures", events=None, runs=None):
    """Write PNGs of the aggregate charts for the README and slides."""
    d = load_all(out_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "concentration": concentration(d["pareto"]),
        "cost_per_year": cost_per_year(d["sizing"]),
        "run_life_history": run_life(d["km_curves"], d["km_summary"], "history"),
        "run_life_rpm": run_life(d["km_curves"], d["km_summary"], "rpm"),
        "run_life_repair": run_life(d["km_curves"], d["km_summary"], "repair"),
        "risk_drivers": risk_drivers(d["cox_results"]),
        "model_vs_rule": model_vs_rule(d["model_eval"]),
        "calibration": calibration(d["calibration"]),
        "what_breaks": what_breaks(d["event_validation"]),
    }
    if events is not None:
        figs["failures_per_year"] = failures_per_year(events)
    if events is not None and runs is not None:
        figs["workover_payback"] = workover_payback(payback_inputs(runs, events, d["cox_results"]))
    for name, fig in figs.items():
        fig.write_image(fig_dir / f"{name}.png", width=900, height=fig.layout.height or 420, scale=2)
    return list(figs)
