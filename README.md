# Run Life: Predicting Repeat Downhole Failures in Heavy-Oil PCP Wells

**DSMLC Datathon 2026 · Canary Labs x Chirality Research · Strathcona Resources dataset**

**Team:** [name], [name], [name], [name]
**Solution form:** analytical dashboard backed by a survival model
**Slides:** [docs/slides.pdf](docs/slides.pdf) 

---

## In one paragraph

Strathcona's heavy-oil wells lift oil with progressing-cavity pumps (PCPs) driven by a rod string. When something downhole breaks (usually the rods), the well stops until a crew repairs it. These failures are frequent, they repeat, and they are almost always fixed after the fact. We rebuilt five years of failures from the daily downtime codes, measured how long each well ran between failures, and used survival analysis to find what goes with shorter and longer runs. A workover goes with about 43% lower failure risk than a lighter repair. It costs about $29k more, so it pays back within a year only on wells that fail about 6 or more times a year. The same model ranks active wells by their 90-day failure risk. On the official failure log, which the model never saw, 37 of its top 50 wells failed within 90 days, against about 28 for the rule operators use today.

---

## Problem

Heavy-oil PCP wells fail downhole often, fail again soon after repair, and are almost always repaired reactively. The cost and lost oil are concentrated in a minority of wells.

| Finding | Value |
|---|---|
| Failure rate | **2.15 downhole failures per producing well-year** (one every 170 producing days) |
| Quick repeats | **28.8%** of restarts fail again within 30 calendar days |
| Reactive work | **80.4%** of rod-truck and workover jobs, and **77.2%** of their $212.7M spend, follow a downhole failure |
| Lost oil | **208,840 m3 (1.31M bbl)** on downhole events, 37% of all lost oil |
| Concentration | The top 10% of in-scope producers (186 of 1,863) account for **40.8% of failures** and **51.3% of downhole lost oil** |
| Yearly cost | About **$28M** in reactive repairs plus **$12M** in lost oil (at C$51.04/bbl), so about **$40M a year** |

All dollar figures are `field_est_total` sums (est., CAD assumed), over 2021-01-01 to 2026-10-26. Source: `outputs/sizing.csv`, `outputs/pareto.csv`.

## Research question

> **Which operating conditions and repair choices shorten or extend the time a heavy-oil PCP well runs between downhole failures, and is the extra run time from a more thorough repair worth its cost?**

It is measurable: run life in producing days, failure risk within 90 days, and dollars per year.

## Proposed solution

An analytical dashboard (`app.py`, Streamlit) backed by a Cox survival model, answering three sub-questions and ending in an action list:

1. **Size:** how big is the problem, and where is it concentrated?
2. **Drivers:** which conditions, known on the day a well restarts, go with shorter runs?
3. **Repair choice:** does a workover buy enough extra run time to pay for itself?
4. **Watchlist:** active wells ranked by their risk of failing in the next 90 producing days.

**The model.** We built a survival model (Cox proportional hazards) that gives every producing well a probability of failing again within the next 90 producing days. It reads what is known about the well today: failures in the past year, the length of its last run, how it was last repaired, pump speed, water cut and oil rate. It was trained on 6,917 runs between failures (2021 to 2024) and tested on 2025 to 2026 and on the official failure log. The scores rank a weekly watchlist, each well with an expected cost and a suggested repair.

Dashboard pages: **Overview** · **What drives run life** (switchable run-life curves) · **Is a workover worth it?** (oil-price slider that redraws the payback chart) · **Watchlist** (ranked wells, pick one to see its last 12 months) · **Does it work?** (model against today's rule, plus limitations).

---

## What we found

### 1. Wells that fail keep failing

Median run life (Kaplan-Meier, producing days) by failures in the previous 365 days:

| Failures in past year | Runs | Median run life |
|---|---|---|
| 0 | 1,851 | **134 days** |
| 1 | 1,817 | **99 days** |
| 2 or more | 6,696 | **55 days** |

### 2. Operating conditions matter

| Group | Median run life (producing days) |
|---|---|
| Pump RPM, lowest to highest quartile | 91 · 73 · 62 · 56 |
| Water cut below 0.7 · 0.7 to 0.9 · above 0.9 | 97 · 77 · 64 |
| Last repair: rod truck only · no linked job · workover | 52 · 70 · 121 |

All differences are significant (log-rank p < 0.001). Source: `outputs/km_summary.csv`, `outputs/km_curves.csv`.

### 3. Risk drivers, holding the others fixed (Cox model)

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

### 4. A workover pays back on repeat-failure wells

- Median workover costs about **$34,400** and a rod-truck-only repair about **$5,000**, so the workover costs about **$29,400 more**.
- One avoided failure saves about **$11,400**: a rod-truck repair plus the average lost oil per event (19.8 m3 at C$51.04/bbl).
- With 43% fewer failures, the simple payback is:

| Well's failures per year | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| Years to pay back the extra cost | 6.0 | 3.0 | 1.5 | **1.0** | 0.7 |

Break-even within one year is at about **6 failures a year**. The fleet average is 2.15, so a blanket workover policy does not pay, but a targeted one on the worst wells does. The dashboard slider redoes this at any oil price from C$35 to C$70.

### 5. The model ranks wells better than today's rule

See the next section.

---

## Model results

**Model:** Cox proportional hazards (lifelines), robust standard errors clustered by well.
**Split:** trained on 6,918 runs that started before 2025-01-01 and tested on **3,446 runs** that started in 2025 to 2026. Training runs still going at the cut are censored there.
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

### Independent check against the official failure log

The official failure log starts in mid-2025 and was never used to build events or train the model. On 2025-07-01 we scored every open well and checked the log for a failure in the next 90 days.

| Metric | Cox model | Today's rule |
|---|---|---|
| Wells scored | 806 | 806 |
| Base rate (failed within 90 days) | 31.9% | 31.9% |
| AUC | **0.726** | 0.675 |
| Top-50 wells with a logged failure | **37 / 50 (74%)** | 28.1 / 50 (range 21 to 36) |

The model's top 50 hit more than twice the base rate and beat the rule on every tie order.

### Watchlist

As of 2026-10-26: 805 open wells, 762 scored and ranked by 90-day failure risk. The top 50 average an 80% risk. 43 wells are left unscored because their runs are longer than the model has enough training data for (917 producing days). The watchlist is well-level and stays local (`outputs/wells/`, gitignored).

---

## Method

### Data and scope

Three files: daily production (5.08M well-days), well jobs (15,610) and the official failure log (2,873). We keep heavy-oil production rows: 1,863 wells and 1.79M producing well-days.

### Cleaning (`src/clean.py`)

- **Rows:** no rows filtered. We removed 29 exact duplicate rows. Every well is kept, and later steps choose their population with an `in_scope` flag.
- **Values:** bad values are set to null, not dropped:
  - sensor placeholders of 99,999 or more
  - negative volumes (allocation corrections)
  - RPM or motor current above 1,000
  - pump efficiency outside 0 to 1.5
- **Failure log:** 30 `failure_type` spellings normalised to 5 families, and 9 same-well same-day duplicates dropped.

Every rule and its row count is in `outputs/cleaning_log.csv`.

### Failure events (`src/events.py`)

- **Downhole day:** a day with `hours_down > 0` and a downhole downtime code, excluding disposal-well repairs.
- **Merging:** downhole days on one well at most 4 days apart merge into one event, because one failure often shows up as several down days while the crew waits for a rig.
- **Result:** 10,548 events on 1,191 wells.
- **Check against the log:** 76.6% of logged failures since 2025-07-01 have an event starting within 7 days (rods 82%). 65.4% of our events in that period appear in the log, so the log misses some failures that the downtime codes record.
- **Sensitivity:** with a 7-day merge gap, there are 10,082 events and 73.1% recall.

Source: `outputs/event_validation.csv`.

### Runs and features (`src/runs.py`)

- A run starts on the first producing day after an event and ends when the next event starts.
- A run with no following failure is **censored**: we know it lasted at least this long. There are 1,014 censored runs out of 10,364. Survival analysis keeps these instead of throwing them away.
- Length is counted in **producing days**, so long shut-ins do not make a well look healthy.
- **Features:** all known on the restart day, taken from the 30 producing days before the failure:
  - oil rate, water cut, RPM, torque and pump utilisation
  - failures in the past 365 days
  - previous run length and down days
  - pump displacement
  - repair type (workover, rod truck, new rod string, rod type) from jobs near the event

### Model (`src/train_eval.py`)

- **Split:** time-based, never random.
- **Preprocessing:** fitted on training data only. Continuous features are clipped and standardised; yes/no features are left as 0 or 1, so their hazard ratios read yes against no.
- **Survival curves:** Kaplan-Meier curves by group.
- **Model:** Cox model, robust SEs clustered by well.
- **Comparison:** against today's rule, a recency rule and a no-information baseline.

### Scoring (`src/score_wells.py`)

For each open well, risk is conditional on how long it has already run: P(fail in the next 90 days | survived t days) = 1 − S(t + 90) / S(t). The official-log check scores wells the same way.

### Oil value

C$51.04/bbl, from Strathcona's Q3 2026 consolidated netback per boe: 99.47 sales net of blending, minus 22.97 royalties, minus 16.33 operating, minus 9.13 transportation. Details: `docs/methods.md`.

---

## Limitations

- **Events are a proxy.** They are rebuilt from downtime codes, not a confirmed failure list, and match the log 77% of the time.
- **Associations, not causes.** Wells chosen for a workover may differ in ways the data does not show. The 43% is what a workover goes with after adjusting for the measured factors; it is not a guaranteed effect.
- **Costs are field estimates** (`field_est_total`), and 13% of events have no matching job.
- **Proportional hazards** is violated for a few factors, including workover, RPM and previous run length (Schoenfeld test p < 0.05). Their effects change somewhat over a run, so read the hazard ratio as an average over the run.
- **Payback** uses median costs and assumes the risk reduction holds for a full year.

**Future work:** confirm events with the field team; a matched or randomised workover trial on the top watchlist wells; time-varying effects for the factors that break proportional hazards; scoring the watchlist weekly from live data.

---

## How this meets the datathon brief

| The brief asks teams to | Where it is |
|---|---|
| Explore the data: relationships, patterns, limitations | `notebooks/01_cleaning_and_analysis.ipynb`, `outputs/data_audit.csv` |
| Identify a problem backed by the data | Problem section, `outputs/sizing.csv` |
| Develop a specific, measurable research question | Research question section |
| Propose a solution using DS, ML, statistics and domain knowledge | Survival model plus the dashboard |
| Validate findings, name pitfalls and limitations | Model results, independent check, Limitations |
| Communicate the analysis and why it matters | `app.py`, `notebooks/story.ipynb`, slides, video |

---

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env            # set DATA_DIR if the data is not in data/
# put the three source files in data/ (not included, licensed for the datathon only)
python run_all.py               # clean → events → sizing → runs → model → watchlist
streamlit run app.py            # dashboard, run locally only
```

The full pipeline runs in a few minutes on a laptop. `notebooks/story.ipynb` walks through the findings one chart at a time.

## Repository layout

```
run-life/
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
│   ├── score_wells.py      watchlist
│   └── plots.py            every chart, shared by notebook and app
├── notebooks/
│   ├── 01_cleaning_and_analysis.ipynb
│   └── story.ipynb
├── outputs/                aggregate CSVs only (no well IDs)
│   └── wells/              well-level files and model (gitignored)
├── docs/
│   ├── slides.pdf
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

## Data and privacy

The Strathcona data is licensed for this datathon only. Raw files, `*.parquet`, `outputs/wells/`, the fitted model and `.env` are gitignored and never committed. No well identifiers appear in the README, slides, figures or committed notebooks. The dashboard shows well IDs, so it is demoed locally and never deployed.

## AI use

We used AI assistants (Claude) for planning, code drafting, debugging and writing. Every number in this README was produced by our code and checked against the output CSVs, and every team member can explain the method and the code. Details: `docs/ai_disclosure.md`.

## External resources

- [lifelines](https://lifelines.readthedocs.io/): Cox model, Kaplan-Meier, PH test
- scikit-learn (AUC), DuckDB, pandas, matplotlib, Streamlit
- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society, Series B*, 34(2), 187 to 220.
- Kaplan, E. L. and Meier, P. (1958). Nonparametric estimation from incomplete observations. *JASA*, 53(282), 457 to 481.
- Strathcona Resources Q3 2026 results (oil value)
