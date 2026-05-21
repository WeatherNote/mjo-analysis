"""I/O helpers: config loading, paths, NetCDF read/write, unit conversion."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # avoid forcing xarray on consumers of load_config/repo_path
    import xarray as xr  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

# ERA5 variable metadata: short_name, long_name, daily_statistic, unit_conversion
ERA5_VARS = {
    "t2m": {
        "cds_name": "2m_temperature",
        "daily_statistic": "daily_mean",
        "raw_unit": "K",
        "out_unit": "degC",
        "scale": 1.0,
        "offset": -273.15,
    },
    "tp": {
        "cds_name": "total_precipitation",
        "daily_statistic": "daily_sum",
        "raw_unit": "m",
        "out_unit": "mm/day",
        "scale": 1000.0,
        "offset": 0.0,
    },
}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config. Defaults to configs/config.yaml in repo root."""
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def repo_path(*parts: str) -> Path:
    """Build an absolute path under the repo root."""
    return REPO_ROOT.joinpath(*parts)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def convert_units(da, var: str):
    """Apply ERA5 unit conversion: K -> degC or m/day -> mm/day."""
    meta = ERA5_VARS[var]
    out = da * meta["scale"] + meta["offset"]
    out.attrs["units"] = meta["out_unit"]
    out.attrs["long_name"] = meta["cds_name"]
    return out


def open_era5_yearly(var: str, year: int, raw_dir: Path):
    """Open one yearly ERA5 NetCDF for the given variable. No unit conversion."""
    import xarray as xr  # lazy import

    fp = raw_dir / f"{var}_{year}.nc"
    return xr.open_dataset(fp)


def standardize_era5(ds, var: str):
    """
    Standardize an ERA5 daily-statistics Dataset to a single DataArray named `var`,
    with dims (time, lat, lon), monotonically increasing lat, applying unit conversion.

    The CDS daily-statistics product may name the variable using the short_name
    (e.g. "t2m", "tp") or a different identifier; we pick the only data variable.
    """
    # Pick the variable: prefer the expected short name, otherwise pick the first.
    if var in ds.data_vars:
        name = var
    elif len(ds.data_vars) == 1:
        name = list(ds.data_vars)[0]
    else:
        raise KeyError(f"Cannot find variable {var} in dataset: {list(ds.data_vars)}")
    da = ds[name]

    # Rename coords to (time, lat, lon)
    rename = {}
    for src, tgt in (
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("valid_time", "time"),
    ):
        if src in da.coords:
            rename[src] = tgt
    if rename:
        da = da.rename(rename)

    # Drop singleton expver/number if present
    for extra in ("expver", "number"):
        if extra in da.coords:
            da = da.drop_vars(extra, errors="ignore")

    # Make lat ascending (some CDS outputs are descending)
    if "lat" in da.dims and float(da.lat[0]) > float(da.lat[-1]):
        da = da.sortby("lat")

    da = convert_units(da, var)
    da.name = var
    return da


def to_netcdf(da, path: Path, **encoding_overrides) -> None:
    """Write NetCDF with sensible compression defaults."""
    import xarray as xr  # lazy import

    ensure_dir(path.parent)
    if isinstance(da, xr.DataArray):
        ds = da.to_dataset()
    else:
        ds = da
    encoding = {
        name: {"zlib": True, "complevel": 4} for name in ds.data_vars
    }
    for name, override in encoding_overrides.items():
        encoding.setdefault(name, {}).update(override)
    ds.to_netcdf(path, encoding=encoding)
