"""Compute Japan-region area-weighted mean anomalies per (season, phase, lag)
and plot phase x lag heatmaps.

Two implementations are available:

    Path A (fast, climatological):
        Average the composite field over each region box.

    Path B (per-event):
        Average per-day anomalies over each region box first, then composite
        the box-averaged time series over MJO phases / lags.

The default is Path A because it directly summarizes the composite NetCDFs
that step 06 produced.

Reads:
    data/processed/composites/composite_{var}_mjo_amp{a}_lags.nc
    data/processed/composites/sample_count_{var}_mjo_amp{a}.csv

Writes:
    data/processed/regions/japan_region_{var}_phase_lag.csv
    figures/heatmaps/heatmap_{region}_{var}_{season}.png

Usage:
    python scripts/08_make_japan_region_summary.py --variable all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mjo.io import ensure_dir, load_config, repo_path  # noqa: E402
from mjo.plotting import phase_lag_heatmap  # noqa: E402
from mjo.regions import area_weighted_mean  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", required=True, choices=["t2m", "tp", "all"])
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--enso", default=None, choices=["ElNino", "Neutral", "LaNina"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    amp = float(args.amplitude if args.amplitude is not None else cfg["mjo"]["active_amplitude"])
    amp_tag = f"amp{str(amp).replace('.', '')}"
    enso_tag = f"_{args.enso}" if args.enso else ""
    enso_label = f"  [{args.enso}]" if args.enso else ""

    composites_dir = repo_path(cfg["paths"]["composites"])
    out_dir = ensure_dir(repo_path(cfg["paths"]["regions"]))
    heatmap_dir = ensure_dir(repo_path(cfg["paths"]["figures_heatmap"]))

    regions = cfg["regions"]
    seasons = list(cfg["seasons"])
    lags = cfg["lags"]
    phases = list(range(1, 9))

    vars_to_run = ["t2m", "tp"] if args.variable == "all" else [args.variable]

    for var in vars_to_run:
        nc_path = composites_dir / f"composite_{var}_mjo_{amp_tag}{enso_tag}_lags.nc"
        csv_path = composites_dir / f"sample_count_{var}_mjo_{amp_tag}{enso_tag}.csv"
        with xr.open_dataset(nc_path) as ds:
            comp = ds[var].load()
        counts_df = pd.read_csv(csv_path)

        rows = []
        for region, bounds in regions.items():
            for season in seasons:
                for phase in phases:
                    for lag in lags:
                        da = comp.sel(season=season, phase=phase, lag=lag)
                        val = area_weighted_mean(da, bounds["lat"], bounds["lon"]).item()
                        n = int(
                            counts_df.query(
                                "season == @season and phase == @phase and lag == @lag"
                            )["n_days"].iloc[0]
                        )
                        rows.append(
                            dict(
                                region=region,
                                season=season,
                                phase=phase,
                                lag=lag,
                                n_days=n,
                                anom_mean=val,
                            )
                        )

        df = pd.DataFrame(rows)
        out_csv = out_dir / f"japan_region_{var}_phase_lag{enso_tag}.csv"
        df.to_csv(out_csv, index=False)
        logging.info("Wrote %s", out_csv)

        # Heatmaps per (region, season)
        for region in regions:
            for season in seasons:
                sub = df.query("region == @region and season == @season")
                mat = (
                    sub.pivot(index="phase", columns="lag", values="anom_mean")
                    .reindex(index=phases, columns=lags)
                    .values
                )
                counts_mat = (
                    sub.pivot(index="phase", columns="lag", values="n_days")
                    .reindex(index=phases, columns=lags)
                    .values
                )
                title = f"{region}  {var}  {season}  (amp>={amp}){enso_label}"
                out_png = heatmap_dir / f"heatmap_{region}_{var}_{season}{enso_tag}.png"
                phase_lag_heatmap(
                    matrix=mat,
                    phases=phases,
                    lags=lags,
                    counts=counts_mat,
                    var=var,
                    title=title,
                    savepath=out_png,
                )
                logging.info("Wrote %s", out_png.name)


if __name__ == "__main__":
    main()
