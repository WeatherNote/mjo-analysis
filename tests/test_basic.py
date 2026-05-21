"""Synthetic-data smoke tests for the analysis pipeline.

These tests do not require ERA5 data or any network access. Run with:

    pytest tests/

or as plain scripts:

    python tests/test_basic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import xarray as xr

from mjo.anomalies import daily_anomaly, linear_detrend_per_cell
from mjo.climatology import build_climatology
from mjo.composites import build_all_composites, select_response_dates
from mjo.regions import area_weighted_mean
from mjo.rmm import add_active_flags, filter_window, parse_bom_rmm


BOM_SAMPLE = """RMM1, RMM2, phase, amplitude.
The fortran format is (i4,2i3,3f12.6,i3,f12.6).
Missing value for RMM1, RMM2 and amplitude is 1.E36.
Missing value for phase is 999.
==================================================================
1979  6  1    1.500    0.500     1    1.581139   0.0
1979  6  2    1.300    1.000     1    1.640122   0.0
1979  6  3    0.100    0.100     8    0.141421   0.0
1979  6  4    2.000    2.000     2    2.828427   0.0
2014  1  1    1.000    0.000     5    1.000000   0.0
"""


def test_rmm_parser(tmp_path: Path = Path("/tmp")) -> None:
    f = tmp_path / "rmm_sample.txt"
    f.write_text(BOM_SAMPLE)
    df = parse_bom_rmm(f)
    assert len(df) == 5
    assert set(df.columns) == {"date", "RMM1", "RMM2", "phase", "amplitude", "processing_era"}
    assert df.loc[df["date"] < pd.Timestamp("2014-01-01"), "processing_era"].eq("pre2014").all()
    assert df.loc[df["date"] >= pd.Timestamp("2014-01-01"), "processing_era"].eq("post2014").all()

    df = filter_window(df, "1979-05-01", "2023-09-30")
    df = add_active_flags(df, thresholds=(0.5, 1.0, 1.5))
    assert df["active_amp1"].sum() == 4  # rows 0, 1, 3, 4
    assert df["active_amp15"].sum() == 3  # rows 0, 1, 3


def test_response_date_lag() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("1979-06-01", periods=10, freq="D"),
            "phase": [1, 1, 8, 2, 5, 1, 6, 7, 3, 4],
            "amplitude": [1.5, 1.6, 0.1, 2.8, 0.7, 1.2, 1.5, 2.0, 0.3, 1.1],
        }
    )
    dates = select_response_dates(df, [6, 7, 8], phase=1, lag=0, amp_threshold=1.0)
    assert set(dates) == {pd.Timestamp("1979-06-01"), pd.Timestamp("1979-06-02"), pd.Timestamp("1979-06-06")}

    dates_lag1 = select_response_dates(df, [6, 7, 8], phase=1, lag=1, amp_threshold=1.0)
    # d-1 is phase 1 on dates 1979-06-02 (d-1 = 06-01) and 1979-06-03 (d-1 = 06-02) and 1979-06-07
    assert pd.Timestamp("1979-06-02") in set(dates_lag1)
    assert pd.Timestamp("1979-06-03") in set(dates_lag1)
    assert pd.Timestamp("1979-06-07") in set(dates_lag1)


def test_full_xarray_pipeline() -> None:
    years = range(2010, 2015)
    dates = []
    for y in years:
        dates.extend(pd.date_range(f"{y}-06-01", f"{y}-08-31", freq="D"))
    times = pd.DatetimeIndex(dates)
    lat = np.arange(20.0, 45.0, 5.0)
    lon = np.arange(120.0, 150.0, 5.0)

    rng = np.random.default_rng(42)
    seasonal = 5 * np.sin(2 * np.pi * (times.dayofyear.to_numpy() - 152) / 92.0)
    trend = 0.5 * (times.year.to_numpy() - 2010)
    mean_lat = (45 - lat) * 0.2
    data = (
        seasonal[:, None, None]
        + trend[:, None, None]
        + mean_lat[None, :, None]
        + rng.normal(0, 0.5, size=(len(times), len(lat), len(lon)))
    )
    da = xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
        name="t2m",
    )

    clim = build_climatology(da, base_years=(2010, 2013), method="harmonic", n_harmonics=3)
    assert "dayofyear" in clim.dims

    anom = daily_anomaly(da, clim)
    anom.name = "t2m"
    anom_dt = linear_detrend_per_cell(anom)
    assert abs(float(anom_dt.mean())) < 1e-6  # detrend zeroed mean

    # Per-cell slope after detrend should be effectively zero.
    x = (anom_dt.time.values.astype("datetime64[D]") - anom_dt.time.values[0].astype("datetime64[D]")).astype(float)
    flat = anom_dt.values.reshape(anom_dt.shape[0], -1)
    slopes = np.polyfit(x, flat, 1)[0]
    assert np.max(np.abs(slopes)) < 1e-10

    # RMM (synthetic) + composites
    rmm_dates = pd.date_range("2010-06-01", "2014-08-31", freq="D")
    rng2 = np.random.default_rng(123)
    rmm = pd.DataFrame({
        "date": rmm_dates,
        "phase": rng2.integers(1, 9, size=len(rmm_dates)),
        "amplitude": rng2.uniform(0.2, 2.5, size=len(rmm_dates)),
    })
    comp, counts = build_all_composites(
        anom=anom_dt,
        rmm=rmm,
        seasons={"JJA": [6, 7, 8], "JJ": [6, 7], "AUG": [8]},
        phases=list(range(1, 9)),
        lags=[0, 5, 10],
        amp_threshold=1.0,
    )
    assert comp.dims == ("season", "phase", "lag", "lat", "lon")
    assert counts["n_days"].sum() > 0

    box = area_weighted_mean(comp.sel(season="JJA", phase=5, lag=0), [30, 46], [128, 146])
    assert np.isfinite(float(box))


if __name__ == "__main__":
    test_rmm_parser()
    test_response_date_lag()
    test_full_xarray_pipeline()
    print("All tests passed.")
