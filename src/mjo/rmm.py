"""BoM RMM (Real-time Multivariate MJO) index parser.

The file at http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt has a
small text header followed by whitespace-separated columns:

    year  month  day  RMM1  RMM2  phase  amplitude  Final_value

A reprocessing change is documented by BoM: for dates up to 2013-12-31 the
calculation removed SST1 variability, ENSO and the 120-day mean; from
2014-01-01 onward only the 120-day mean is removed. Composites that span both
eras can therefore mix two slightly different RMM definitions. This is recorded
in the README and as a column in the output (`processing_era`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RMM_COLUMNS = ["year", "month", "day", "RMM1", "RMM2", "phase", "amplitude"]
BOM_ERA_CHANGE_DATE = pd.Timestamp("2014-01-01")
MISSING_SENTINELS = {1.0e36, 999.0, 99.0}


def parse_bom_rmm(path: Path | str) -> pd.DataFrame:
    """Read the BoM RMM ascii file into a tidy DataFrame.

    The header lines start with non-numeric characters; we detect the first
    data line by trying to parse the first whitespace-separated token as an int.
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()
    skip = 0
    for i, line in enumerate(lines):
        toks = line.split()
        if len(toks) >= 7 and toks[0].isdigit():
            skip = i
            break

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=skip,
        usecols=list(range(7)),
        names=RMM_COLUMNS,
        engine="python",
    )

    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    # Replace BoM missing sentinels with NaN
    for col in ("RMM1", "RMM2", "amplitude"):
        df.loc[df[col].isin(MISSING_SENTINELS), col] = pd.NA
    df["phase"] = pd.to_numeric(df["phase"], errors="coerce").astype("Int64")
    df.loc[(df["phase"] < 1) | (df["phase"] > 8), "phase"] = pd.NA

    df["processing_era"] = df["date"].apply(
        lambda d: "pre2014" if d < BOM_ERA_CHANGE_DATE else "post2014"
    )

    return df[["date", "RMM1", "RMM2", "phase", "amplitude", "processing_era"]]


def add_active_flags(
    df: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.5, 1.0, 1.5),
) -> pd.DataFrame:
    """Add boolean active-MJO flags for one or more amplitude thresholds."""
    out = df.copy()
    for thr in thresholds:
        # Replace decimal point in column name to keep it identifier-friendly.
        suffix = str(thr).replace(".", "").rstrip("0") or "0"
        out[f"active_amp{suffix}"] = (out["amplitude"] >= thr).fillna(False)
    return out


def filter_window(
    df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    """Restrict to [start, end] inclusive on the `date` column."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    mask = (df["date"] >= s) & (df["date"] <= e)
    return df.loc[mask].reset_index(drop=True)
