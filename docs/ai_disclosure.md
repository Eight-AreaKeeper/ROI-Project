# AI assistance

We used Claude (Anthropic) as a research and coding assistant: Claude in chat for planning and review, and Claude Code for writing code in the repository.

**Our team's decisions:** the problem (repeat downhole failures in heavy-oil PCP wells), the research question, survival analysis as the approach (time between failures, keeping wells still running), the analytical dashboard as our solution form, which findings to present, and the final layout and wording of the app.

**Produced mainly by AI, then reviewed by us:**
- **Method choices:** a Cox proportional hazards model with standard errors clustered by well; the time-based train/test split (before and after 2025-01-01); the rule that features come only from the 30 producing days before each failure; conditional 90-day risk for open wells; and the independent check against the official failure log, including averaging over tie orders for the baseline.
- **Event rebuild rules:** downhole downtime codes, merging down days 4 or fewer days apart, and the 7-day sensitivity check.
- **Business inputs:** sourcing the oil value from Strathcona's Q3 2026 report, and the workover payback formula.
- **Code:** the pipeline (`src/clean.py` to `src/score_wells.py`), the chart library (`src/plots.py`), the Streamlit app and its tests, written by Claude Code from our specifications.
- **Documentation:** first drafts of the README, the methods notes and this disclosure.

**How we verified it:**
- We reran the full pipeline from a clean checkout and checked every number in the README, slides and app against the output CSVs.
- We checked the rebuilt events against the official failure log (77% match).
- We reviewed the model output for errors. For example, binary features had first been standardised, which understated the workover effect (reported as −24% instead of −43%); we found and corrected it before submission.
- Every team member walked through the pipeline and the app, and can explain the method and justify each decision.

### Additional AI contributions

- **Shared scoring module** (`src/scoring.py`): one set of functions for feature preparation and conditional risk, used by the evaluation, the independent check, the replay and the watchlist, so all of them score wells the same way.
- **Model support limit** (`src/scoring.py`, `src/score_wells.py`): leaving wells unscored once their run passes 917 producing days, the longest duration with at least 30 training runs still observed, because the survival curve beyond that went flat and gave false near-zero risks.
- **Training censoring at the cut** (`src/train_eval.py`): training runs still open on 2025-01-01 are censored there, with their producing days scaled by the share of calendar time before the cut.
- **Deterministic event attributes and failure-log cleaning** (`src/events.py`): fixed tie-breaking for each event's area and main downtime code, the `failure_type` family mapping, and dropping same-well same-day duplicate log records.
- **Privacy safeguards** (`src/explore.py`, `src/score_wells.py`, `src/app_logic.py`): blanking text minimum/maximum values in `outputs/data_audit.csv` so no well ID is published, printing only aggregates in pipeline logs, and the "Hide well IDs" aliases in the app.
- **Back-in-time replay** (`src/replay.py`, `outputs/replay_summary.csv`): repeating the independent check on four dates, with a check that the first date reproduces `outputs/model_eval.csv`.
- **Watchlist enrichment** (`src/app_logic.py`): expected 90-day cost per well, the main risk driver (limited to the seven significant continuous factors), workover payback per well, the suggested-action rules, and the `impact_summary` numbers (including the 29 workover target wells).
- **What-if panel** (`src/app_logic.py`, `app.py`): rescoring a well's next run under each repair option and pump speed, with the previous-run gap set by the same definition `src/runs.py` uses.
- **App structure and tests** (`src/ui.py`, `tests/`): the shared page layout helpers, the app smoke tests (every page opens, a well can be selected, no jargon outside "Technical detail") and the unit tests for the watchlist logic.
- **Notebooks** (`notebooks/01_cleaning_and_analysis.ipynb`, `notebooks/story.ipynb`): the conclusion section of the cleaning notebook and the story notebook built on `src/plots.py`.
- **Figures and screenshots** (`docs/figures/`): exporting the static charts, and the app screenshots taken with a headless browser.
- **Documentation edits** (`README.md`, `docs/methods.md`): checking every number against the output files and merging the earlier README into the methods notes.
