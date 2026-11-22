# Methods and full results

Everything behind the summary in the [README](../README.md): every table, every number and where it comes from. All figures are over 2021-01-01 to 2026-10-26 unless stated.

---

## 1. Problem size

Heavy-oil PCP wells fail downhole often, fail again soon after repair, and are almost always repaired reactively. The cost and lost oil are concentrated in a minority of wells.

| Finding | Value |
|---|---|
| Failure rate | **2.15 downhole failures per producing well-year** (one every 170 producing days) |
| Quick repeats | **28.8%** of restarts fail again within 30 calendar days |
| Reactive work | **80.4%** of rod-truck and workover jobs, and **77.2%** of their $212.7M spend, follow a downhole failure |
| Lost oil | **208,840 m3 (1.31M bbl)** on downhole events, 37% of all lost oil |
| Concentration | The top 10% of in-scope producers (186 of 1,863) account for **40.8% of failures** and **51.3% of downhole lost oil** |
| Yearly cost | About **$28M** in reactive repairs plus **$12M** in lost oil (at C$51.04/bbl), so about **$40M a year** |

All dollar figures are `field_est_total` sums (est., CAD assumed). Source: `outputs/sizing.csv`, `outputs/pareto.csv`.

---

## 2. Findings

### 2.1 Wells that fail keep failing

Median run life (Kaplan-Meier, producing days) by failures in the previous 365 days:

| Failures in past year | Runs | Median run life |
|---|---|---|
| 0 | 1,851 | **134 days** |
| 1 | 1,817 | **99 days** |
| 2 or more | 6,696 | **55 days** |

### 2.2 Operating conditions matter

| Group | Median run life (producing days) |
|---|---|
| Pump RPM, lowest to highest quartile | 91 · 73 · 62 · 56 |
| Water cut below 0.7 · 0.7 to 0.9 · above 0.9 | 97 · 77 · 64 |
| Last repair: rod truck only · no linked job · workover | 52 · 70 · 121 |

All differences are significant (log-rank p < 0.001). Source: `outputs/km_summary.csv`, `outputs/km_curves.csv`.

### 2.3 Risk drivers, holding the others fixed (Cox model)

A hazard ratio above 1 means a shorter run; below 1 means a longer run. Yes/no factors compare yes against no. Measured factors are per one standard deviation (SD).

| Factor | Hazard ratio (95% CI) | Reads as |
|---|---|---|
| Workover as the last repair | **0.567** (0.531 to 0.605) | 43% lower failure risk |
| New rod string installed | **0.435** (0.347 to 0.546) | 56% lower failure risk |
| Failures in past 365 days, per SD (2.7 failures) | 1.241 (1.203 to 1.280) | 24% higher risk |
| Oil rate (log), per SD | 1.115 (1.081 to 1.151) | 11% higher risk |
| Pump RPM, per SD (124 rpm) | 1.060 (1.027 to 1.095) | faster pumps fail sooner |
| Water cut, per SD (0.17) | 1.058 (1.017 to 1.100) | wetter wells fail sooner |
| Previous run length (log), per SD | 0.895 (0.867 to 0.924) | long runs follow long runs |
| Stick rods (vs other rod types) | 1.101 (1.013 to 1.196) | slightly higher risk |

Not significant: rod torque, continuous rod (corod), previous down days. Full table: `outputs/cox_results.csv`.

### 2.4 A workover pays back on repeat-failure wells

- Median workover costs about **$34,400** and a rod-truck-only repair about **$5,000**, so the workover costs about **$29,400 more**.
- One avoided failure saves about **$11,400**: a rod-truck repair plus the average lost oil per event (19.8 m3 at C$51.04/bbl).
- With 43% fewer failures, the simple payback is:

| Well's failures per year | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| Years to pay back the extra cost | 6.0 | 3.0 | 1.5 | **1.0** | 0.7 |

Break-even within one year is at about **6 failures a year**. The fleet average is 2.15, so a blanket workover policy does not pay, but a targeted one on the worst wells does. The dashboard slider redoes this at any oil price from C$35 to C$70.

**Workover target wells (29).** Open wells whose last repair was not already a workover or new rod string, and where one year of avoided failures at the well's own cost per failure beats the extra cost: `fails_prior_365 × (1 − 0.567) × cost_per_failure > $29,396`. At C$51.04/bbl that is 29 wells, about 90 failures avoided a year and about $0.25M year-one net saving. Computed live in `src/app_logic.py` (`impact_summary`).

---

## 3. Model results

**Model:** Cox proportional hazards (lifelines), robust standard errors clustered by well.
**Split:** fitted on 6,917 runs that started before 2025-01-01 (6,918 runs started before the cut; runs still going at the cut are censored there, and one had no producing days left) and tested on **3,446 runs** that started in 2025 to 2026.
**Today's rule (baseline):** rank wells by failures in the past 365 days.

| Metric (test set) | Cox model | Today's rule | Same risk for everyone |
|---|---|---|---|
| C-index (ranking of all runs) | **0.641** | 0.603 | 0.500 |
| AUC, failure within 90 producing days | **0.717** | 0.659 | 0.500 |
| Brier score at 90 days (lower is better) | **0.208** | n/a | 0.241 |
| Top-50 runs that failed within 90 days | **45** | 44.5 (42 to 46) | n/a |

- The C-index gain is **0.039 (95% CI 0.028 to 0.050)**, from 200 bootstrap draws that resample wells, not runs.
- On the internal top-50 the two are almost tied. Many wells share the same failure count, so the rule's top 50 depends on tie order, and we report the mean and range over tie orders. The model's real edge is in ranking the whole fleet (C-index, AUC) and on the independent check below.
- **Calibration:** in every risk decile, predicted and observed failure shares are within about 6.5 points (`outputs/calibration.csv`).

### 3.1 Independent check against the official failure log

The official failure log was never used to build events or train the model. It starts in December 2021 but is incomplete. On 2025-07-01 we scored every open well and checked the log for a failure in the next 90 days.

| Metric | Cox model | Today's rule |
|---|---|---|
| Wells scored | 806 | 806 |
| Base rate (failed within 90 days) | 31.9% | 31.9% |
| AUC | **0.726** | 0.675 |
| Top-50 wells with a logged failure | **37 / 50 (74%)** | 28.1 / 50 (range 21 to 36) |

The model's top 50 hit more than twice the base rate and beat the rule on every tie order.

**Conventions:**

- **Open-run population:** a well is scored on the check date only if its run started strictly before that date, so a well restarting on the day itself is excluded.
- **Label window:** a logged failure counts if its `failure_start_date` falls from the check date through the check date plus 90 days, inclusive of day 90.
- **Model support limit:** a well is left unscored on the watchlist when its run would extend past the longest duration at which at least 30 training runs were still observed (917 producing days), because the survival curve beyond that rests on too few runs. The independent check keeps these wells scored (32 of the 806).

### 3.2 Replay on four dates

The same check repeated on four dates after the training cut (`src/replay.py`, `outputs/replay_summary.csv`). The rule's value is the mean over tie orders, with its range.

| Date | Wells | Base rate | Model top 50 | Rule top 50 (range) | Random |
|---|---|---|---|---|---|
| 2025-07-01 | 806 | 0.319 | **37** | 28.1 (21 to 36) | 15.9 |
| 2025-10-01 | 821 | 0.307 | **33** | 25.6 (20 to 33) | 15.3 |
| 2026-01-01 | 795 | 0.323 | **36** | 28.7 (25 to 34) | 16.2 |
| 2026-04-01 | 804 | 0.294 | **30** | 24.5 (18 to 33) | 14.7 |

The model beats the rule's mean on every date. On 2025-10-01 and 2026-04-01 the rule's best tie order (33) matches or beats the model.

### 3.3 Watchlist

As of 2026-10-26: 805 open wells, 762 scored and ranked by 90-day failure risk. The top 50 average an 80% risk. 43 wells are left unscored because their runs are longer than the model has enough training data for (917 producing days). The watchlist is well-level and stays local (`outputs/wells/`, gitignored).

---

## 4. Method

### 4.1 Data and scope

Three files: daily production (5.08M well-days), well jobs (15,610) and the official failure log (2,873). We keep heavy-oil production rows: 1,863 wells and 1.79M producing well-days.

### 4.2 Cleaning (`src/clean.py`)

- **Rows:** no rows filtered. We removed 29 exact duplicate rows. Every well is kept, and later steps choose their population with an `in_scope` flag.
- **Values:** bad values are set to null, not dropped:
  - sensor placeholders of 99,999 or more
  - negative volumes (allocation corrections)
  - RPM or motor current above 1,000
  - pump efficiency outside 0 to 1.5
- **Failure log:** 30 `failure_type` spellings normalised to 5 families, and 9 same-well same-day duplicates dropped.

Every rule and its row count is in `outputs/cleaning_log.csv`.

### 4.3 Failure events (`src/events.py`)

- **Downhole day:** a day with `hours_down > 0` and a downhole downtime code ("Downhole" or "Downhole Equipment"), excluding disposal-well repairs.
- **Merging:** downhole days on one well at most 4 days apart merge into one event, because one failure often shows up as several down days while the crew waits for a rig.
- **Result:** 10,548 events on 1,191 wells.
- **Check against the log:** 76.6% of logged failures since 2025-07-01 have an event starting within 7 days (rods 82%). 65.4% of our events in that period appear in the log, so the log misses some failures that the downtime codes record.
- **Sensitivity:** with a 7-day merge gap, there are 10,082 events and 73.1% recall.

Source: `outputs/event_validation.csv`.

### 4.4 Runs and features (`src/runs.py`)

- A run starts on the first producing day after an event and ends when the next event starts.
- A run with no following failure is **censored**: we know it lasted at least this long. There are 1,014 censored runs out of 10,364. Survival analysis keeps these instead of throwing them away.
- Length is counted in **producing days**, so long shut-ins do not make a well look healthy.
- **Features:** all known on the restart day, taken from the 30 producing days before the failure:
  - oil rate, water cut, RPM, torque and pump utilisation
  - failures in the past 365 days
  - previous run length and down days
  - pump displacement
  - repair type (workover, rod truck, new rod string, rod type) from jobs near the event

### 4.5 Model (`src/train_eval.py`, `src/scoring.py`)

- **Split:** time-based, never random.
- **Preprocessing:** fitted on training data only. Missing values are filled with training medians; skewed features (oil rate, previous run, previous down days) are logged; continuous features are standardised; yes/no features are left as 0 or 1, so their hazard ratios read yes against no.
- **Survival curves:** Kaplan-Meier curves by group.
- **Model:** Cox model (penalizer 0.01), robust SEs clustered by well.
- **Comparison:** against today's rule, a recency rule and a no-information baseline.

### 4.6 Scoring (`src/score_wells.py`, `src/scoring.py`)

For each open well, risk is conditional on how long it has already run: P(fail in the next 90 days | survived t days) = 1 − S(t + 90) / S(t). The official-log check, the replay and the watchlist all score wells through the same functions in `src/scoring.py`.

### 4.7 Oil value

**C$51.04 per bbl** (`OIL_VALUE_PER_BBL`): Strathcona Q3 2026 results, consolidated per boe: sales net of blending 99.47 − royalties 22.97 − production and operating 5.77 (energy) and 10.56 (non-energy) − transportation and processing 9.13 = C$51.04. Company-wide including thermal, used as a conservative proxy for Lloydminster conventional heavy oil.

---

## 5. Limitations and future work

- **Events are a proxy.** They are rebuilt from downtime codes, not a confirmed failure list, and match the log 77% of the time.
- **Associations, not causes.** Wells chosen for a workover may differ in ways the data does not show. The 43% is what a workover goes with after adjusting for the measured factors; it is not a guaranteed effect.
- **Costs are field estimates** (`field_est_total`), and 13% of events have no matching job.
- **Proportional hazards** is violated for a few factors: workover, RPM, previous run length, pump displacement and utilisation (Schoenfeld test). Their effects change somewhat over a run, so read the hazard ratio as an average over the run.
- **Payback** uses median costs and assumes the risk reduction holds for a full year.

**Future work:** confirm events with the field team; a matched or randomised workover trial on the top watchlist wells; time-varying effects for the factors that break proportional hazards; scoring the watchlist weekly from live data.

---

## 6. How this meets the datathon brief

| The brief asks teams to | Where it is |
|---|---|
| Explore the data: relationships, patterns, limitations | `notebooks/01_cleaning_and_analysis.ipynb`, `outputs/data_audit.csv` |
| Identify a problem backed by the data | Problem size, `outputs/sizing.csv` |
| Develop a specific, measurable research question | README, section 1 |
| Propose a solution using DS, ML, statistics and domain knowledge | Survival model plus the dashboard |
| Validate findings, name pitfalls and limitations | Model results, independent check, replay, limitations |
| Communicate the analysis and why it matters | `app.py`, `notebooks/story.ipynb`, slides, video |

---

## 7. Repository, outputs and privacy

The full pipeline runs in about 1.5 minutes on a laptop (`python run_all.py`). `notebooks/story.ipynb` walks through the findings one chart at a time.

```
├── README.md
├── requirements.txt
├── .env.example            settings (data path, event gap, time cut, oil value)
├── run_all.py              runs every step in order
├── app.py                  Streamlit dashboard
├── src/
│   ├── config.py           paths and settings from .env
│   ├── explore.py          raw-file audit
│   ├── clean.py            cleaning rules, derived daily columns
│   ├── events.py           failure events, check against the log
│   ├── sizing.py           problem size, Pareto, failure mix
│   ├── runs.py             one row per run, features before restart
│   ├── train_eval.py       Kaplan-Meier, Cox model, evaluation
│   ├── scoring.py          conditional risk scoring shared by every check and the watchlist
│   ├── score_wells.py      watchlist
│   ├── replay.py           back-in-time replay on four dates
│   ├── plots.py            every chart, shared by notebook and app
│   ├── app_logic.py        watchlist enrichment, what-if, impact numbers
│   └── ui.py               shared page layout helpers for the app
├── tests/                  pytest suite (app smoke tests and logic tests)
├── notebooks/
│   ├── 01_cleaning_and_analysis.ipynb
│   └── story.ipynb
├── outputs/                aggregate CSVs only (no well IDs)
│   └── wells/              well-level files and model (gitignored)
├── docs/
│   ├── methods.md
│   ├── ai_disclosure.md
│   └── figures/
└── data/                   raw parquet files (gitignored)
```

**Public outputs** (`outputs/`):

| File | Contents |
|---|---|
| `cleaning_log.csv` | each cleaning rule and its row count |
| `data_audit.csv` | null share and range per raw column |
| `event_validation.csv` | event counts, recall and precision against the log, 4- and 7-day gaps |
| `sizing.csv`, `pareto.csv`, `failure_mix.csv` | problem size and concentration |
| `km_curves.csv`, `km_summary.csv` | run-life curves and medians by group |
| `cox_results.csv` | hazard ratios, CIs, PH test |
| `model_eval.csv`, `calibration.csv` | test metrics and the independent check |
| `replay_summary.csv` | top-50 hits on four replay dates |

**Data and privacy.** The Strathcona data is licensed for this datathon only. Raw files, `*.parquet`, `outputs/wells/`, the fitted model and `.env` are gitignored and never committed. No well identifiers appear in the README, slides, figures or committed notebooks. The dashboard reads well-level files and hides well IDs by default ("Hide well IDs"), so it is demoed locally and never deployed.

**Libraries.** [lifelines](https://lifelines.readthedocs.io/) (Cox model, Kaplan-Meier, PH test), scikit-learn (AUC), DuckDB, pandas, Plotly, Streamlit.
