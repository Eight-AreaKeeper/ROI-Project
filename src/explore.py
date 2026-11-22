"""Step 1: audit the three raw files.

Writes outputs/data_audit.csv with one row per (file, column): type, null
share, distinct count, min and max. Nothing downstream reads this file; it is
the evidence behind the cleaning rules in clean.py.
"""
from __future__ import annotations

import pandas as pd

from src import config


def audit(con) -> pd.DataFrame:
    frames = []
    for name in config.RAW_FILES:
        # SUMMARIZE scans each column once, which is much cheaper than one
        # query per column on the 5M-row daily file.
        s = con.sql(f"summarize {name}").df()
        rows = con.sql(f"select count(*) from {name}").fetchall()[0][0]
        s.insert(0, "file", name)
        s["rows"] = rows
        frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"column_name": "column", "column_type": "type",
                              "approx_unique": "approx_distinct"})
    out["null_pct"] = pd.to_numeric(out["null_percentage"], errors="coerce")
    # Min and max of a text column are real values such as a UWI, well name or
    # comment, which identify wells. The audit is committed to a public repo,
    # so ranges are kept for numeric and date columns only.
    is_text = out["type"].str.upper().str.startswith("VARCHAR")
    out.loc[is_text, ["min", "max"]] = None
    return out[["file", "column", "type", "rows", "null_pct", "approx_distinct", "min", "max"]]


def main() -> pd.DataFrame:
    config.ensure_dirs()
    con = config.connect()
    out = audit(con)
    path = config.OUTPUTS / "data_audit.csv"
    out.to_csv(path, index=False)

    print("explore: rows per file")
    print(out.groupby("file")["rows"].first().to_string())
    empty = out[out["null_pct"] >= 100]
    print(f"explore: {len(empty)} columns are entirely null:")
    for r in empty.itertuples():
        print(f"  {r.file}.{r.column}")
    print(f"explore: wrote {config.rel(path)}")
    return out


if __name__ == "__main__":
    main()
