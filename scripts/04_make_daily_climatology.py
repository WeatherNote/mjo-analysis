"""Build a smoothed daily climatology from ERA5 yearly NetCDFs.

Steps:
    1. Open yearly raw files data/raw/era5/{var}_{YYYY}.nc and concatenate.
    2. Convert units (K -> degC, m/day -> mm/day) and standardize dim names.
    3. Save the concatenated dataset to data/interim/era5/{var}_{file_tag}_{y0}_{y1}.nc.
    4. Restrict to the base period (1991-2020), compute the daily climatology
       and smooth with the configured method (default: first 3 harmonics).
    5. Save to data/interim/clim/{var}_clim_{by0}_{by1}_{file_tag}.nc.

The ``file_tag`` is read from ``period.file_tag`` in the config (default: ``jja``).

Usage:
    python scripts/04_make_daily_climatology.py --variable t2m
    python scripts/04_make_daily_climatology.py --variable all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xarray as xr  # noqa: E402

from mjo.climatology import build_climatology  # noqa: E402
from mjo.io import ERA5_VARS, ensure_dir, load_config, repo_path, standardize_era5, to_netcdf  # noqa: E402


def concat_yearly(var: str, raw_dir: Path, years: range) -> xr.DataArray:
    """Open all yearly NetCDFs for `var` and concatenate along time."""
    paths = [raw_dir / f"{var}_{y}.nc" for y in years]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} yearly file(s) for {var}; first: {missing[0]}"
        )
    parts = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            parts.append(standardize_era5(ds, var).load())
    # Concatenate along time and drop any duplicates from leap-day overlaps.
    da = xr.concat(parts, dim="time").sortby("time")
    _, idx = xr.set_options(keep_attrs=True), None
    da = da.drop_duplicates("time")
    return da


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", required=True, choices=["t2m", "tp", "all"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = repo_path(cfg["paths"]["raw_era5"])
    interim_era5 = ensure_dir(repo_path(cfg["paths"]["interim_era5"]))
    interim_clim = ensure_dir(repo_path(cfg["paths"]["interim_clim"]))

    y0 = int(str(cfg["period"]["start"])[:4])
    y1 = int(str(cfg["period"]["end"])[:4])
    by0, by1 = cfg["period"]["base_climatology"]
    file_tag = cfg["period"].get("file_tag", "jja")

    method = cfg["climatology"]["method"]
    n_harm = int(cfg["climatology"]["n_harmonics"])
    win = int(cfg["climatology"]["running_window"])

    vars_to_run = ["t2m", "tp"] if args.variable == "all" else [args.variable]

    for var in vars_to_run:
        logging.info("[%s] concatenating yearly ERA5 files", var)
        da = concat_yearly(var, raw_dir, range(y0, y1 + 1))

        merged_path = interim_era5 / f"{var}_{file_tag}_{y0}_{y1}.nc"
        logging.info("[%s] writing %s", var, merged_path)
        to_netcdf(da, merged_path)

        logging.info(
            "[%s] computing %s climatology over %d-%d (n_harm=%d)",
            var, method, by0, by1, n_harm,
        )
        clim = build_climatology(
            da, base_years=(by0, by1), method=method,
            n_harmonics=n_harm, running_window=win,
        )
        clim_path = interim_clim / f"{var}_clim_{by0}_{by1}_{file_tag}.nc"
        logging.info("[%s] writing %s", var, clim_path)
        to_netcdf(clim, clim_path)


if __name__ == "__main__":
    main()
