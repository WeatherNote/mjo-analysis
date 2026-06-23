"""Compute daily anomalies (and optionally detrend) for ERA5 fields.

Reads:
    data/interim/era5/{var}_{file_tag}_{y0}_{y1}.nc
    data/interim/clim/{var}_clim_{by0}_{by1}_{file_tag}.nc

Writes:
    data/interim/anom/{var}_anom_daily_{file_tag}_{y0}_{y1}.nc

The ``file_tag`` is read from ``period.file_tag`` in the config (default: ``jja``).

Detrend is per grid cell, linear in time, applied only to variables flagged
in `detrend:` in the config (default: t2m only).

Usage:
    python scripts/05_make_daily_anomalies.py --variable t2m
    python scripts/05_make_daily_anomalies.py --variable all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xarray as xr  # noqa: E402

from mjo.anomalies import daily_anomaly, linear_detrend_per_cell  # noqa: E402
from mjo.io import ensure_dir, load_config, repo_path, to_netcdf  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", required=True, choices=["t2m", "tp", "all"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    interim_era5 = repo_path(cfg["paths"]["interim_era5"])
    interim_clim = repo_path(cfg["paths"]["interim_clim"])
    interim_anom = ensure_dir(repo_path(cfg["paths"]["interim_anom"]))

    y0 = int(str(cfg["period"]["start"])[:4])
    y1 = int(str(cfg["period"]["end"])[:4])
    by0, by1 = cfg["period"]["base_climatology"]
    file_tag = cfg["period"].get("file_tag", "jja")
    detrend_flags = cfg["detrend"]

    vars_to_run = ["t2m", "tp"] if args.variable == "all" else [args.variable]

    for var in vars_to_run:
        data_path = interim_era5 / f"{var}_{file_tag}_{y0}_{y1}.nc"
        clim_path = interim_clim / f"{var}_clim_{by0}_{by1}_{file_tag}.nc"
        logging.info("[%s] reading %s and %s", var, data_path.name, clim_path.name)
        with xr.open_dataset(data_path) as ds_data, xr.open_dataset(clim_path) as ds_clim:
            da = ds_data[var].load()
            clim = ds_clim[var].load()
        anom = daily_anomaly(da, clim)
        anom.name = var

        if detrend_flags.get(var, False):
            logging.info("[%s] applying linear detrend per grid cell", var)
            anom = linear_detrend_per_cell(anom)
        else:
            logging.info("[%s] detrend disabled", var)

        out = interim_anom / f"{var}_anom_daily_{file_tag}_{y0}_{y1}.nc"
        logging.info("[%s] writing %s", var, out)
        to_netcdf(anom, out)


if __name__ == "__main__":
    main()
