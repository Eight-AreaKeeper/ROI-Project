# Methods

## Independent check conventions

- **Open-run population:** a well is scored on the check date only if its run started strictly before that date, so a well restarting on the day itself is excluded.
- **Label window:** a logged failure counts if its `failure_start_date` falls from the check date through the check date plus 90 days, inclusive of day 90.
- **Model support limit:** a well is left unscored on the watchlist when its run would extend past the longest duration at which at least 30 training runs were still observed (917 producing days), because the survival curve beyond that rests on too few runs.

## Oil value

- **C$51.04 per bbl** (`OIL_VALUE_PER_BBL`): Strathcona Q3 2026 results, consolidated per boe: sales net of blending 99.47 − royalties 22.97 − production and operating 5.77 (energy) and 10.56 (non-energy) − transportation and processing 9.13 = C$51.04. Company-wide including thermal, used as a conservative proxy for Lloydminster conventional heavy oil.
