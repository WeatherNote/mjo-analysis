"""Build MJO phase composites for one or more variables, seasons, and lags.

Reads:
    data/interim/anom/{var}_anom_daily_jja_{y0}_{y1}.nc
    data/raw/mjo/rmm_daily_1979_2023.csv

Writes:
    data/processed/composites/composite_{var}_mjo_amp{a}_lags.nc
    data/processed/composites/sample_count_{var}_mjo_amp{a}.csv

Usage:
    python scripts/06_make_mjo_composites.py --variable t2m
    python scripts/06_make_mjo_composites.py --variable all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mjo.composites import build_all_composites  # noqa: E402
from mjo.io import ensure_dir, load_config, repo_path, to_netcdf  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", required=True, choices=["t2m", "tp", "all"])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="Active-MJO amplitude threshold (default: config mjo.active_amplitude).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Subset of season names to process (e.g. JJA JJ AUG). Default: all.",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=None,
        help="Lag days (default: config lags).",
    )
    parser.add_argument(
        "--enso",
        default=None,
        choices=["ElNino", "Neutral", "LaNina"],
        help="Restrict RMM to years of specified ENSO category.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    amp = float(args.amplitude if args.amplitude is not None else cfg["mjo"]["active_amplitude"])
    lags = args.lags if args.lags is not None else cfg["lags"]

    seasons_all = cfg["seasons"]
    if args.seasons:
        seasons = {k: seasons_all[k] for k in args.seasons}
    else:
        seasons = seasons_all

    interim_anom = repo_path(cfg["paths"]["interim_anom"])
    composites_dir = ensure_dir(repo_path(cfg["paths"]["composites"]))
    rmm_path = repo_path(cfg["paths"]["raw_mjo"]) / "rmm_daily_1979_2023.csv"

    y0 = int(str(cfg["period"]["start"])[:4])
    y1 = int(str(cfg["period"]["end"])[:4])

    logging.info("Loading RMM from %s", rmm_path)
    rmm = pd.read_csv(rmm_path, parse_dates=["date"])

    enso_tag = ""
    if args.enso:
        enso_csv = repo_path(cfg["paths"]["raw_enso"]) / "enso_category_1979_2023.csv"
        enso_df = pd.read_csv(enso_csv)
        enso_years = set(enso_df.loc[enso_df["ENSO_category"] == args.enso, "year"])
        rmm = rmm[rmm["date"].dt.year.isin(enso_years)].reset_index(drop=True)
        enso_tag = f"_{args.enso}"
        logging.info("ENSO filter: %s → %d RMM rows", args.enso, len(rmm))

    vars_to_run = ["t2m", "tp"] if args.variable == "all" else [args.variable]
    phases = list(range(1, 9))
    amp_tag = f"amp{str(amp).replace('.', '')}"

    for var in vars_to_run:
        anom_path = interim_anom / f"{var}_anom_daily_jja_{y0}_{y1}.nc"
        logging.info("[%s] reading %s", var, anom_path.name)
        with xr.open_dataset(anom_path) as ds:
            anom = ds[var].load()

        logging.info(
            "[%s] building composites: seasons=%s phases=1..8 lags=%s threshold=%.1f",
            var, list(seasons), lags, amp,
        )
        comp, counts = build_all_composites(
            anom=anom,
            rmm=rmm,
            seasons=seasons,
            phases=phases,
            lags=lags,
            amp_threshold=amp,
        )

        nc_path = composites_dir / f"composite_{var}_mjo_{amp_tag}{enso_tag}_lags.nc"
        csv_path = composites_dir / f"sample_count_{var}_mjo_{amp_tag}{enso_tag}.csv"
        logging.info("[%s] writing %s", var, nc_path)
        to_netcdf(comp, nc_path)
        counts.to_csv(csv_path, index=False)
        logging.info("[%s] writing %s", var, csv_path)


if __name__ == "__main__":
    main()
