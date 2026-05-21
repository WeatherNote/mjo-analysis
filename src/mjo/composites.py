"""MJO phase composites with optional lag.

Composite definition:
    response_date d is in the requested season,
    RMM phase on (d - lag) equals the requested phase, and
    RMM amplitude on (d - lag) >= threshold.

The composite mean is taken over the response-date anomaly values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class CompositeResult:
    mean: xr.DataArray  # (lat, lon)
    n: int


def select_response_dates(
    rmm: pd.DataFrame,
    season_months: list[int],
    phase: int,
    lag: int,
    amp_threshold: float = 1.0,
) -> pd.DatetimeIndex:
    """Return response dates d satisfying the composite criteria.

    `rmm` must have columns: date, phase, amplitude.
    """
    df = rmm.set_index("date").sort_index()
    # Phase / amplitude at the *reference* date d - lag; we shift to align with d.
    ref_phase = df["phase"].shift(lag)
    ref_amp = df["amplitude"].shift(lag)

    in_season = df.index.month.isin(season_months)
    cond = (
        in_season
        & (ref_phase == phase)
        & (ref_amp >= amp_threshold)
    )
    return df.index[cond]


def composite_mean(
    anom: xr.DataArray, response_dates: pd.DatetimeIndex
) -> CompositeResult:
    """Mean of `anom` over the given response dates."""
    if len(response_dates) == 0:
        empty = anom.isel(time=0).where(False)
        return CompositeResult(mean=empty.drop_vars("time", errors="ignore"), n=0)
    sub = anom.sel(time=response_dates.intersection(anom["time"].to_index()))
    return CompositeResult(mean=sub.mean("time"), n=sub.sizes["time"])


def build_all_composites(
    anom: xr.DataArray,
    rmm: pd.DataFrame,
    seasons: dict[str, list[int]],
    phases: list[int],
    lags: list[int],
    amp_threshold: float = 1.0,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """Loop over (season, phase, lag), assemble a 5-D composite DataArray and
    a sample-count DataFrame.

    Returns:
        composites: xr.DataArray with dims (season, phase, lag, lat, lon)
        counts: DataFrame with columns season, phase, lag, n_days
    """
    season_names = list(seasons)
    nlat = anom.sizes["lat"]
    nlon = anom.sizes["lon"]
    shape = (len(season_names), len(phases), len(lags), nlat, nlon)
    arr = np.full(shape, np.nan, dtype=np.float32)
    counts_rows = []

    for si, sname in enumerate(season_names):
        for pi, phase in enumerate(phases):
            for li, lag in enumerate(lags):
                dates = select_response_dates(
                    rmm, seasons[sname], phase, lag, amp_threshold
                )
                res = composite_mean(anom, dates)
                if res.n > 0:
                    arr[si, pi, li] = res.mean.values
                counts_rows.append(
                    dict(season=sname, phase=phase, lag=lag, n_days=res.n)
                )

    out = xr.DataArray(
        arr,
        dims=("season", "phase", "lag", "lat", "lon"),
        coords={
            "season": season_names,
            "phase": phases,
            "lag": lags,
            "lat": anom["lat"],
            "lon": anom["lon"],
        },
        name=anom.name,
        attrs={
            **anom.attrs,
            "composite_definition": (
                "mean of anomaly on response date d where "
                "phase(d-lag)==phase and amplitude(d-lag)>=threshold"
            ),
            "amp_threshold": amp_threshold,
        },
    )
    counts = pd.DataFrame(counts_rows)
    return out, counts
