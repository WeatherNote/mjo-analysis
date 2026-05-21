"""Daily climatology and harmonic smoothing.

We fit the seasonal cycle in the climatology by truncating its Fourier series
at `n_harmonics` (default 3). For JJA-only data the dayofyear span is 92 days
(152..243) so the truncated Fourier expansion is interpreted as a fit over the
JJA window only — adequate for removing day-to-day climatology noise without
imposing the full annual cycle.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def raw_daily_climatology(da: xr.DataArray, base_years: tuple[int, int]) -> xr.DataArray:
    """Mean over `base_years` for each calendar dayofyear."""
    y0, y1 = base_years
    sel = da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    return sel.groupby("time.dayofyear").mean("time")


def harmonic_smooth(da_doy: xr.DataArray, n: int = 3, dim: str = "dayofyear") -> xr.DataArray:
    """Truncate the Fourier expansion of `da_doy` along `dim` at `n` harmonics.

    Implementation: rfft -> zero coefficients with k > n -> irfft. Works on
    arrays whose `dim` length is not a power of two.
    """
    arr = da_doy.transpose(dim, ...).values
    axis = 0
    spec = np.fft.rfft(arr, axis=axis)
    # Keep coefficients 0..n (DC + n harmonics). Higher modes -> 0.
    if n + 1 < spec.shape[axis]:
        spec[n + 1 :] = 0
    smoothed = np.fft.irfft(spec, n=arr.shape[axis], axis=axis)
    out = xr.DataArray(
        smoothed,
        dims=da_doy.transpose(dim, ...).dims,
        coords={
            d: da_doy[d]
            for d in da_doy.transpose(dim, ...).dims
            if d in da_doy.coords
        },
        name=da_doy.name,
        attrs={**da_doy.attrs, "smoothing": f"harmonic_n={n}"},
    )
    # Return in the original dim order
    return out.transpose(*da_doy.dims)


def running_mean_climatology(
    da: xr.DataArray, base_years: tuple[int, int], window: int = 31
) -> xr.DataArray:
    """Alternative smoothing: running-mean over dayofyear with circular padding."""
    clim = raw_daily_climatology(da, base_years)
    # Circular pad along dayofyear, take rolling mean, slice back.
    pad = window // 2
    padded = xr.concat(
        [clim.isel(dayofyear=slice(-pad, None)), clim, clim.isel(dayofyear=slice(0, pad))],
        dim="dayofyear",
    )
    smoothed = padded.rolling(dayofyear=window, center=True).mean()
    return smoothed.isel(dayofyear=slice(pad, pad + clim.sizes["dayofyear"]))


def build_climatology(
    da: xr.DataArray,
    base_years: tuple[int, int],
    method: str = "harmonic",
    n_harmonics: int = 3,
    running_window: int = 31,
) -> xr.DataArray:
    """Compute the daily climatology with the requested smoothing method."""
    if method == "raw":
        return raw_daily_climatology(da, base_years)
    if method == "harmonic":
        return harmonic_smooth(raw_daily_climatology(da, base_years), n=n_harmonics)
    if method == "running":
        return running_mean_climatology(da, base_years, window=running_window)
    raise ValueError(f"Unknown climatology method: {method}")
