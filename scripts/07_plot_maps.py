"""Plot 8-panel MJO phase composite maps.

Reads:
    data/processed/composites/composite_{var}_mjo_amp{a}_lags.nc
    data/processed/composites/sample_count_{var}_mjo_amp{a}.csv

Writes:
    figures/maps/fig_{var}_{season}_lag{L}.png

Usage:
    python scripts/07_plot_maps.py --variable t2m --season JJA --lag 0
    python scripts/07_plot_maps.py --variable all --season all --lag all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from mjo.io import load_config, repo_path  # noqa: E402
from mjo.plotting import eight_panel_map  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", required=True, choices=["t2m", "tp", "all"])
    parser.add_argument("--season", default="all")
    parser.add_argument("--lag", default="all", help="Integer lag or 'all'.")
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    amp = float(args.amplitude if args.amplitude is not None else cfg["mjo"]["active_amplitude"])
    amp_tag = f"amp{str(amp).replace('.', '')}"

    composites_dir = repo_path(cfg["paths"]["composites"])
    figures_main = repo_path(cfg["paths"]["figures_maps"])
    figures_lag = repo_path(cfg["paths"]["figures_lag"])

    vars_to_run = ["t2m", "tp"] if args.variable == "all" else [args.variable]
    season_keys = list(cfg["seasons"]) if args.season == "all" else [args.season]
    lag_keys = cfg["lags"] if args.lag == "all" else [int(args.lag)]

    for var in vars_to_run:
        nc_path = composites_dir / f"composite_{var}_mjo_{amp_tag}_lags.nc"
        csv_path = composites_dir / f"sample_count_{var}_mjo_{amp_tag}.csv"
        with xr.open_dataset(nc_path) as ds:
            comp = ds[var].load()
        counts_df = pd.read_csv(csv_path)

        for season in season_keys:
            for lag in lag_keys:
                da_phase = comp.sel(season=season, lag=lag)
                count_lookup = (
                    counts_df.query("season == @season and lag == @lag")
                    .set_index("phase")["n_days"]
                    .to_dict()
                )
                out_dir = figures_lag if lag != 0 else figures_main
                out_path = out_dir / f"fig_{var}_{season}_lag{lag:+d}.png"
                title = f"{var} anomaly  {season}  lag={lag:+d}  (amp>={amp})"
                logging.info("Plotting %s", out_path.name)
                eight_panel_map(
                    da_phase=da_phase,
                    counts=count_lookup,
                    var=var,
                    title=title,
                    savepath=out_path,
                    japan_box=None,
                )


if __name__ == "__main__":
    main()
