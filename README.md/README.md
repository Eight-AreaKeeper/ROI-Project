# Repair ROI: which well repairs pay for themselves, and which wells are next?

**Chirality Research x Canary Labs Datathon 2026** | Case: Strathcona Resources | Theme: Optimization

**Team:** [FILL names, one line each with role]
**Presentation slides:** [FILL link] (also in `docs/slides.pdf`)
**5-minute summary video:** [FILL link]
**Live dashboard:** [FILL link, or "run locally: see How to run"]

---

## Research question

[CONFIRM wording] Which well repairs pay for themselves, by how much, and which wells should be repaired next?

## Motivation

Heavy-oil wells in this dataset run on progressing cavity pumps, and rod and pump jobs are the most common work in the job log (9,359 Rod Truck jobs out of 15,610). Each job costs money, yet the data contains no final job cost and an incomplete failure log, so the company cannot easily tell which repairs are worth it.

A simple before/after comparison also overstates repair value. Wells are usually serviced when they are doing badly, and some of those wells would have partly recovered anyway. We compare repaired wells with similar wells that were not touched, then turn the difference into dollars and a ranked list of wells to visit.

## Summary of findings

> Fill these in from `outputs/` after the final run. Only write numbers that the code produces, and keep each one traceable to a file.

1. **Size of the problem:** [FILL, from `sizing.csv` and `pareto.csv`: for example how much lost oil or repair spend sits in how many wells or job types.]
2. **Failure patterns:** [FILL, from `failure_mix.csv`, `km_curves.csv`, `cox_results.csv`: which failure types dominate, how long pumps and rods run before failing, and which factors are linked to shorter runs.]
3. **Which repairs pay back:** [FILL, from `repair_choice.csv`: control-adjusted gain per repair type, cost, payback.]
4. **Naive vs. controlled:** [FILL: how much a plain before/after comparison overstates the benefit.]
5. **Risk model:** [FILL, from `model_eval.csv` and `calibration.csv`: how well the model ranks wells on held-out later data, and whether its probabilities are calibrated.]
6. **Watchlist:** [FILL: how many wells the dashboard flags and the expected value at stake, at the stated assumptions.]

## What the dashboard shows (`app.py`)

[CONFIRM against the final app.] A field team can see the size of the problem, which repairs work, which wells to repair next, and whether each repair is worth doing at an adjustable oil price. Sliders recalculate the dollar results.

---

## Method

The pipeline runs in numbered steps. `run_all.py` runs steps 2 to 8 in order, and `app.py` is step 9. Step 1 is exploration in `notebooks/`, which the app never depends on.

| Step | File | What it does |
|---|---|---|
| 2 | `src/clean.py` | Loads the three parquet files and fixes known issues (dates and RPM stored as text, 999999 pressure placeholders, negative job durations, empty columns). Every step is logged with row counts to `outputs/cleaning_log.csv`. |
| 3 | `src/events.py` | Builds the event table: repairs and failures per well with dates, joined on `uwi`. |
| 4 | `src/lookup_repairs.py`, `src/runs.py` | Labels each job with a repair type, and builds runs (the time a well runs between events) for survival analysis. [CONFIRM definitions] |
| 5 | `src/sizing.py` | Sizes the problem: spend, lost oil and failure counts, with a Pareto view of where they concentrate. |
| 6 | `src/train_eval.py` | Trains a failure-risk model and evaluates it on later data than it was trained on. Writes discrimination and calibration results. |
| 7 | `src/repair_choice.py` | Compares repaired wells with matched control wells, reports covariate balance, and converts the effect of each repair type into oil, dollars and payback. |
| 8 | `src/score_wells.py` | Scores current wells and produces the watchlist. |
| 9 | `app.py` | Streamlit dashboard reading the files in `outputs/`. |

Tools: DuckDB and pandas for data handling, lifelines for survival analysis (Kaplan-Meier and Cox), scikit-learn for the risk model, Plotly and Streamlit for the dashboard.

### Key design decisions

- **Controls instead of raw before/after.** Control wells are similar untouched wells matched on oil rate and recent trend [CONFIRM matching variables]. `balance.csv` shows whether matching made the groups comparable.
- **Cost = field estimate.** `actual_final_job_cost` is empty for every job, so `field_est_total` is the cost, and it is an estimate. Zero-cost rows are reported separately, not treated as free repairs.
- **Dollars rest on assumptions.** Oil price and effect duration are assumptions, not data. The dashboard exposes them as sliders. [CONFIRM defaults and currency of `field_est_total`.]
- **No leakage.** The risk model uses only information from before the prediction date. Downtime and lost oil that follow a failure are not used as inputs.

## Validation

- **Time-based evaluation:** the risk model is trained on earlier periods and tested on later ones. Results: `outputs/model_eval.csv` [FILL summary].
- **Calibration:** `outputs/calibration.csv` compares predicted and observed failure rates [FILL summary].
- **Matching quality:** `outputs/balance.csv` [FILL summary].
- **Survival analysis:** Kaplan-Meier curves and Cox results in `outputs/km_curves.csv` and `outputs/cox_results.csv`. Hazard ratios are associations, not proof of cause.
- **Cleaning audit:** every cleaning step and its row counts are in `outputs/cleaning_log.csv`.

## Limitations

- Costs are field estimates, not final costs, and the handbook does not state their currency [CONFIRM and state].
- The failure log is incomplete: it starts in December 2021, covers 17 of 26 operating areas, and has far fewer records than there are rod jobs. "No failure logged" does not mean a healthy well.
- Repairs and failures have no shared ID. Linking them uses well ID and dates, so some links will be wrong.
- Controls are matched on observed variables only. Reservoir quality, nearby injection and other unobserved factors can still bias results.
- Several sensor columns are sparse or look forward-filled, so sensor features cover only part of the fleet [CONFIRM which features the model uses].
- The effect window and the dollar duration are assumptions. Results are not a controlled experiment.

## How to run

```
pip install -r requirements.txt
# put the three parquet files in data/ (they are not in this repo)
python run_all.py
streamlit run app.py
```

`run_all.py` regenerates everything in `outputs/`. The app reads those files only.

## Data and privacy

The dataset belongs to Strathcona Resources and may be used only for this competition, so it is **not** in this repository.

- `data/`, `outputs/wells/` and `*.parquet` are gitignored. Well-level files (events, runs, watchlist, well history) stay local.
- Committed files in `outputs/` are aggregates only.
- Figures in `docs/figures/` contain no well identifiers (UWIs).

## External resources

- Dataset Handbook and Research Topic Guide provided by the organizers.
- PetroWiki, "Progressing cavity pump systems"; SLB Oilfield Review, "The Defining Series: Progressing Cavity Pumps" (domain background on pumps, per the handbook's resource list).
- Libraries: DuckDB, pandas, lifelines, scikit-learn, Streamlit, Plotly.
- [FILL any other source, such as an oil-price reference for the default price.]

## AI disclosure

Full detail is in `docs/ai_disclosure.md`. In short: [EDIT to match what your team actually did.] We used Claude (Anthropic) for planning the project, drafting code and documentation, and explaining concepts. All team members reviewed and ran the code, and can explain every step. We verified [FILL: for example the cleaning rules, the matching balance and each reported number] against the data ourselves.

## Repository layout

```
README.md            this file
requirements.txt     dependencies
run_all.py           runs steps 2 to 8
app.py               dashboard (step 9)
src/                 pipeline steps (see Method)
outputs/             aggregate CSVs (public); wells/ is gitignored
data/                raw parquet files (gitignored, never committed)
notebooks/           exploration only
docs/                slides.pdf, methods.md, ai_disclosure.md, figures/
.gitignore           data/  outputs/wells/  *.parquet  .env
```