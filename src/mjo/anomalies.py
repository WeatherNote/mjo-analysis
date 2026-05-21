"""Daily anomalies and grid-cell linear detrending."""
from __future__ import annotations

import numpy as np
import xarray as xr


def daily_anomaly(da: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Subtract `clim` (indexed by dayofyear) from `da` (indexed by time).

    Returned array keeps the original time axis.
    """
    return da.groupby("time.dayofyear") - clim


def linear_detrend_per_cell(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Subtract the per-(lat, lon) linear trend in `time` from `da`.

    Trend fitted over all available time steps. Returned array has zero linear
    trend per grid cell. NaN cells are preserved.
    """
    t = da[time_dim]
    # Convert times to days since first sample for a stable x-axis.
    x = (t.values.astype("datetime64[D]") - t.values[0].astype("datetime64[D]")).astype(float)

    # Move time to the first axis for vectorised polyfit.
    da_t = da.transpose(time_dim, ...)
    arr = da_t.values
    flat = arr.reshape(arr.shape[0], -1)

    mask = ~np.isnan(flat).any(axis=0)
    coef = np.zeros((2, flat.shape[1]), dtype=flat.dtype)
    if mask.any():
        coef[:, mask] = np.polyfit(x, flat[:, mask], deg=1)
    # coef shape (2, ncells): row 0 = slope, row 1 = intercept
    trend = (coef[0:1, :] * x[:, None] + coef[1:2, :]).reshape(arr.shape)
    detrended = arr - trend

    out = xr.DataArray(
        detrended,
        dims=da_t.dims,
        coords=da_t.coords,
        name=da.name,
        attrs={**da.attrs, "detrend": "linear_per_gridcell"},
    )
    return out.transpose(*da.dims)
