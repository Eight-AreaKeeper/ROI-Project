# Run Life: Predicting Repeat Downhole Failures in Heavy-Oil PCP Wells

**DSMLC Datathon 2026 · Canary Labs x Chirality Research · Strathcona Resources dataset**

| | |
|---|---|
| **Team** | Byte Fight: Alec Ferrer, Nusrat Tabassum Anika, Oghenetano Ojekudo, Zhinan Dai |
| **Solution form** | Analytical dashboard backed by a survival model |
| **Slides** | [Google Slides](https://docs.google.com/presentation/d/1U50o5F3LQFfQnVLlqSyMyUr6qeUqL4kdHQcP6DT20mI/edit?usp=sharing) |

**In one line:** our model ranks wells by their chance of failing in the next 90 days, and it caught 37 of 50 real failures on data it never saw, against 28 for the rule used today.

---

## 1. Research question, motivation and findings

### Research question

> Which operating conditions and repair choices shorten or extend the time a heavy-oil PCP well runs between downhole failures, and is the extra run time from a more thorough repair worth its cost?

### Motivation

Strathcona's 1,863 heavy-oil wells lift oil with progressing-cavity pumps (PCPs) driven by a rod string. When something downhole breaks, usually the rods, the well stops producing until a crew repairs it.

- **Frequent:** 2.15 downhole failures per producing well-year, about 1,800 a year.
- **Repeating:** 29% of repairs are followed by another failure within 30 days.
- **Reactive:** 77% of rod-truck and workover spend ($164M of $213M) follows a failure.
- **Costly:** about **$40M a year** in reactive repairs plus lost oil. 10% of wells cause half the lost oil.

### Our solution

**The model.** A survival model gives every producing well a **probability of failing again within the next 90 producing days**, using only what is known about it today:
- failures in the past year
- the length of its last run
- how it was last repaired
- pump speed, water cut and oil rate

**The dashboard.** The scores feed a weekly watchlist. Each well gets an expected failure cost, its main risk driver and a suggested repair. A what-if panel compares repair options for any well.

### Summary of findings

1. **Wells that fail keep failing.** Wells with 2+ failures in the past year run a median of **55 days** after a repair, against **134** for wells with none.
2. **Faster and wetter wells fail sooner.** The median run falls from 91 days for the slowest pumps to **56** for the fastest. Wells with water cut above 90% run 64 days, against 97 below 70%.
3. **A thorough repair goes with longer runs, but it only pays on the worst wells.**
   - Holding other factors fixed, failure risk is **43% lower** after a workover (hazard ratio 0.57) and **56% lower** after a new rod string (0.44).
   - A workover costs about $29k more than a rod-truck job.
   - It **pays back within a year on wells that fail about 6 or more times a year**, about 29 wells today. It does not pay fleet-wide.
4. **The model finds failing wells better than today's rule.** Checked against the official failure log, which the model never saw, **37 of our top 50 wells failed within 90 days**. Ranking by "most failures last year" caught 28, and 50 random wells caught 16.

---

## 2. Source code and project files

```bash
pip install -r requirements.txt
cp .env.example .env              # set DATA_DIR to the folder with the 3 parquet files
python run_all.py                 # clean → events → sizing → runs → model → watchlist → replay
streamlit run app.py              # the dashboard (local only; well IDs are hidden by default)
```

| Path | What it is |
|---|---|
| `app.py` | Streamlit dashboard: Summary, The problem, What drives failure, Is a workover worth it?, Watchlist, Does it work? |
| `src/clean.py` | Cleaning rules (placeholders, negative volumes, impossible values), logged in `outputs/cleaning_log.csv` |
| `src/events.py` | Rebuilds failure events from daily downtime codes and checks them against the official log |
| `src/sizing.py` | Problem size: failure rate, repeats, reactive spend, lost oil, concentration |
| `src/runs.py` | One row per run between failures, with features known on the restart day |
| `src/train_eval.py` | Kaplan-Meier curves, Cox model, evaluation against baselines |
| `src/score_wells.py`, `src/replay.py` | Watchlist scoring and the back-in-time replay |
| `src/plots.py`, `src/app_logic.py`, `src/ui.py` | Charts, watchlist logic and UI helpers shared by the app and notebooks |
| `notebooks/` | Exploration and the story notebook |
| `outputs/*.csv` | Aggregate results (no well IDs) |
| `docs/` | Figures, full methods, AI disclosure |

The raw data and all well-level files (`data/`, `*.parquet`, `outputs/wells/`, the fitted model) are not in this repo, because the dataset is licensed for this datathon only.

---

## 3. Slides

[Presentation slides (Google Slides)](https://docs.google.com/presentation/d/1U50o5F3LQFfQnVLlqSyMyUr6qeUqL4kdHQcP6DT20mI/edit?usp=sharing)

---

## 4. Methods, validation, resources and AI disclosure

### Methods

- **Scope.** Heavy-oil producing wells: 1,863 wells, 1.79M producing well-days, 2021-01-01 to 2026-10-26.
- **Failure events.**
  - A downhole failure is a day with downtime coded as downhole ("Downhole" or "Downhole Equipment", excluding disposal-well repairs).
  - Down days 4 or fewer days apart merge into one event. This gives 10,548 events on 1,191 wells.
  - We rebuilt events because the official failure log is incomplete: it starts in December 2021 and holds 2,873 records, far fewer than the downtime codes show.
- **Runs.**
  - A run lasts from a restart to the next failure, counted in producing days, so shut-ins don't count.
  - Runs with no failure yet are kept as censored: we know they lasted *at least* that long. Survival analysis uses them instead of dropping them.
  - This gives 10,364 runs.
- **Features.** Only information known on the restart day:
  - oil rate, water cut and pump speed over the 30 producing days before the failure;
  - failures in the past year;
  - previous run length;
  - repair type, from the job records.
- **Models.**
  - **Kaplan-Meier:** run life by group.
  - **Cox proportional hazards:** the effect of each factor with the others held fixed, with standard errors clustered by well.
- **Scoring.** An open well's risk depends on how long it has already run: risk = 1 − S(t + 90) / S(t), where S is the survival curve and t is days run so far.
- **Economics.**
  - Payback = extra workover cost ÷ (failures per year × 43% × cost of one failure).
  - Cost of one failure = a rod-truck repair + lost oil valued at C$51.04/bbl (Strathcona's Q3 2026 netback).

### Validation

| Check | Result |
|---|---|
| Time split: train on runs before 2025-01-01 (6,917), test on 2025 to 2026 (3,446) | C-index **0.641** vs 0.603 for today's rule; AUC at 90 days **0.717** vs 0.659 |
| Independent check: official failure log (never used in training), 806 wells scored on 2025-07-01 | Top 50: **37** failed, vs 28 for the rule (21 to 36 depending on how ties are broken) and 16 at random |
| Replay on 4 dates (Jul 2025 to Apr 2026) | The model beats the rule's average on every date: 37, 33, 36, 30 vs 28, 26, 29, 25 (`outputs/replay_summary.csv`) |
| Calibration | Predicted and observed 90-day failure shares agree within about 7 points in every risk decile |
| Event rebuild vs official log | 77% of logged failures match a rebuilt event within 7 days (rods 82%); a 7-day merge gap gives 73% |

**Limitations**
- Events are rebuilt from downtime codes, not taken from a confirmed failure list.
- Costs are field estimates.
- The workover effect is an association, not proof of cause: wells chosen for workovers may differ in ways the data doesn't show.
- The effects of workover, pump speed, pump size, utilisation and previous run length change somewhat over a run, so their hazard ratios are averages.

**Future work**
- Confirm events with field staff.
- Pilot workovers on flagged wells.
- Retrain quarterly on new data.

Full detail: [docs/methods.md](docs/methods.md).

### External resources

- Strathcona Resources Q3 2026 results (oil value per barrel)
- DSMLC Datathon Dataset Handbook (PCP, rod string, downtime definitions)
- [lifelines](https://lifelines.readthedocs.io/) (Kaplan-Meier, Cox model), scikit-learn, DuckDB, pandas, Plotly, Streamlit
- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Kaplan, E. L. and Meier, P. (1958). Nonparametric estimation from incomplete observations. *JASA*, 53(282), 457–481.

### AI assistance

We used Claude (Anthropic): Claude in chat for planning and review, and Claude Code for writing code.

- **Our decisions:** the problem, the research question, survival analysis as the approach, the dashboard form, the findings to present, and the app's layout and wording.
- **Mainly AI, reviewed by us:**
  - the Cox model with clustered errors
  - the time split and the leakage rule
  - conditional risk scoring
  - the independent check against the official log
  - the event merge rules
  - the oil value and payback formula
  - most of the code (pipeline, `plots.py`, app, tests)
  - first drafts of the docs
- **How we verified it:**
  - We reran the pipeline and checked every reported number against the output CSVs.
  - We validated events against the official log.
  - We caught and fixed one AI error: binary features had been standardised, which understated the workover effect (−24% instead of −43%).

Full disclosure: [docs/ai_disclosure.md](docs/ai_disclosure.md).
