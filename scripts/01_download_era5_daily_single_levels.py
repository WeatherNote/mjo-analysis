"""Download ERA5 post-processed daily statistics on single levels.

Usage:
    python scripts/01_download_era5_daily_single_levels.py --variable 2m_temperature
    python scripts/01_download_era5_daily_single_levels.py --variable all

The CDS dataset used here is `reanalysis-era5-single-levels-daily-statistics`.
Submitting one request per (variable, year) keeps each request small and
allows simple resume: if `data/raw/era5/{var}_{year}.nc` already exists and
already contains all requested months, the request is skipped.

The months downloaded are controlled by ``period.download_months`` in the
config (default: [6, 7, 8]).

Run this locally where ~/.cdsapirc is configured. CDS jobs queue and may take
several minutes each; the script polls until each retrieve() returns.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjo.cds_client import submit_year  # noqa: E402
from mjo.io import ERA5_VARS, ensure_dir, load_config, repo_path  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variable",
        required=True,
        choices=["2m_temperature", "total_precipitation", "all"],
        help="ERA5 variable to download. 'all' downloads both t2m and tp.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (default: configs/config.yaml).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    area = [cfg["area"]["north"], cfg["area"]["west"], cfg["area"]["south"], cfg["area"]["east"]]
    grid = cfg["grid"]
    download_months = list(cfg["period"].get("download_months", [6, 7, 8]))

    out_dir = ensure_dir(repo_path(cfg["paths"]["raw_era5"]))

    if args.variable == "all":
        vars_to_run = ["2m_temperature", "total_precipitation"]
    else:
        vars_to_run = [args.variable]

    # Translate CDS variable name to internal short name
    cds_to_short = {v["cds_name"]: k for k, v in ERA5_VARS.items()}

    import cdsapi  # local import so the module-level import of this script does not fail without cdsapi installed.

    client = cdsapi.Client()

    start_year = int(str(cfg["period"]["start"])[:4])
    end_year = int(str(cfg["period"]["end"])[:4])

    for cds_name in vars_to_run:
        var = cds_to_short[cds_name]
        for year in range(start_year, end_year + 1):
            submit_year(
                client=client,
                var=var,
                year=year,
                out_dir=out_dir,
                area=area,
                grid=grid,
                months=download_months,
            )

    logging.info("Done.")


if __name__ == "__main__":
    main()
