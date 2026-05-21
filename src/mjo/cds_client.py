"""CDS API client for ERA5 daily statistics on single levels.

Submits one request per (variable, year) and writes yearly NetCDFs into
`data/raw/era5/`. Resumes by skipping years whose file already exists and
contains the expected number of JJA days.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import xarray as xr

from .io import ERA5_VARS, ensure_dir

DATASET = "reanalysis-era5-single-levels-daily-statistics"
EXPECTED_JJA_DAYS = 92  # 30 + 31 + 31

log = logging.getLogger(__name__)


def build_request(
    var: str,
    year: int,
    area: list[int],
    grid: float,
    months: list[int] | None = None,
) -> dict:
    """Build a single-year CDS request dict for one ERA5 variable.

    `area` is [N, W, S, E].
    """
    if months is None:
        months = [6, 7, 8]
    meta = ERA5_VARS[var]
    return {
        "product_type": "reanalysis",
        "variable": meta["cds_name"],
        "daily_statistic": meta["daily_statistic"],
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": area,
        "format": "netcdf",
        "grid": [grid, grid],
    }


def _is_complete(path: Path, expected_days: int) -> bool:
    """Return True if `path` exists and contains at least `expected_days` time steps."""
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as ds:
            time_dim = next(
                (d for d in ("time", "valid_time") if d in ds.dims), None
            )
            if time_dim is None:
                return False
            return ds.sizes[time_dim] >= expected_days
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to open %s for completeness check: %s", path, exc)
        return False


def submit_year(
    client,
    var: str,
    year: int,
    out_dir: Path,
    area: list[int],
    grid: float,
    months: list[int] | None = None,
    max_retries: int = 4,
) -> Path:
    """Submit one (var, year) request and write the result to `out_dir`.

    Skips the request if the output file already exists and looks complete.
    Retries with exponential backoff on transient errors.
    """
    ensure_dir(out_dir)
    out_path = out_dir / f"{var}_{year}.nc"
    expected = EXPECTED_JJA_DAYS if months is None or set(months) == {6, 7, 8} else None

    if expected is not None and _is_complete(out_path, expected):
        log.info("Skipping %s (already complete)", out_path.name)
        return out_path

    req = build_request(var, year, area=area, grid=grid, months=months)
    log.info("CDS request: %s %d", var, year)

    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            client.retrieve(DATASET, req, str(out_path))
            return out_path
        except Exception as exc:
            log.warning(
                "Attempt %d/%d failed for %s %d: %s",
                attempt, max_retries, var, year, exc,
            )
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")
