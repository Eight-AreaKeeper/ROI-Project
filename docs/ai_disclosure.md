# AI disclosure

| Date | Who | Tool | Asked for | What was kept | Human check |
|---|---|---|---|---|---|
| 2026-11-21 | Alec Ferrer | Claude Code | Write `src/config.py`, `src/explore.py`, `src/clean.py` (with derived daily columns), `src/events.py` (event rebuild and validation against the failure log), `src/sizing.py` (problem sizing), `run_all.py` and `notebooks/01_cleaning_and_analysis.ipynb`, reproducing the figures in the team brief. | All of the above. Every headline figure is computed by `src/sizing.py` and checked against the brief within 1%. | Still needs a human check: definitions, SQL logic and every number before it goes on a slide. |
