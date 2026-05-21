"""NOAA CPC ONI parser and JJA-based ENSO categorization.

The ONI ascii file (https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt)
has columns:

    SEAS  YR  TOTAL  ANOM

where SEAS is a 3-letter overlapping season label (DJF, JFM, ..., NDJ).
JJA refers to Jun-Jul-Aug centered on July; we use the JJA value of year Y
to categorize summer Y.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_oni(path: Path | str) -> pd.DataFrame:
    """Read the CPC ONI ascii file into a long DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    # CPC sometimes labels columns differently; normalize.
    df.columns = [c.upper() for c in df.columns]
    df = df.rename(columns={"YR": "year", "SEAS": "season", "ANOM": "oni"})
    return df[["season", "year", "oni"]]


def jja_oni(df: pd.DataFrame, season_label: str = "JJA") -> pd.DataFrame:
    """Pull the JJA ONI value for each year."""
    sel = df.loc[df["season"] == season_label, ["year", "oni"]].copy()
    sel = sel.rename(columns={"oni": f"{season_label}_ONI"})
    return sel.reset_index(drop=True)


def classify(
    df: pd.DataFrame, threshold: float = 0.5, value_col: str = "JJA_ONI"
) -> pd.DataFrame:
    """Add a 3-category ENSO classification column."""
    def label(x: float) -> str:
        if x >= threshold:
            return "ElNino"
        if x <= -threshold:
            return "LaNina"
        return "Neutral"

    out = df.copy()
    out["ENSO_category"] = out[value_col].apply(label)
    return out
