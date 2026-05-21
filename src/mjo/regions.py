"""Region box selection and cos(lat)-weighted area averages."""
from __future__ import annotations

import numpy as np
import xarray as xr


def select_box(da: xr.DataArray, lat_bnds: list[float], lon_bnds: list[float]) -> xr.DataArray:
    """Select a lat/lon box. Assumes ascending lat and 0-360 lon convention."""
    lat0, lat1 = sorted(lat_bnds)
    lon0, lon1 = sorted(lon_bnds)
    return da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))


def area_weighted_mean(
    da: xr.DataArray, lat_bnds: list[float], lon_bnds: list[float]
) -> xr.DataArray:
    """cos(lat)-weighted mean over a lat/lon box."""
    sub = select_box(da, lat_bnds, lon_bnds)
    w = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(w).mean(dim=("lat", "lon"))
