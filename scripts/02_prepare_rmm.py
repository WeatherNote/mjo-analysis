"""Download and parse the BoM RMM index into a tidy CSV.

Usage:
    python scripts/02_prepare_rmm.py [--no-download]

Writes:
    data/raw/mjo/rmm.74toRealtime.txt    # original ascii (downloaded once)
    data/raw/mjo/rmm_daily_1979_2023.csv # cleaned, with active-amplitude flags
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from mjo.io import ensure_dir, load_config, repo_path  # noqa: E402
from mjo.rmm import add_active_flags, filter_window, parse_bom_rmm  # noqa: E402


def fetch(url: str, dest: Path) -> Path:
    logging.info("Downloading %s -> %s", url, dest)
    headers = {"User-Agent": "mjo-analysis/0.1 (research; ERA5 + RMM compositing)"}
    r = requests.get(url, timeout=60, headers=headers)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download if the ascii file is already present.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ensure_dir(repo_path(cfg["paths"]["raw_mjo"]))
    ascii_path = raw_dir / "rmm.74toRealtime.txt"

    if not ascii_path.exists() or not args.no_download:
        try:
            fetch(cfg["mjo"]["rmm_url"], ascii_path)
        except Exception as exc:
            if ascii_path.exists():
                logging.warning("Download failed (%s); using cached %s", exc, ascii_path)
            else:
                raise

    df = parse_bom_rmm(ascii_path)
    df = filter_window(df, cfg["period"]["rmm_buffer_start"], cfg["period"]["rmm_buffer_end"])
    df = add_active_flags(df, thresholds=(0.5, 1.0, 1.5))

    out_csv = raw_dir / "rmm_daily_1979_2023.csv"
    df.to_csv(out_csv, index=False)
    logging.info("Wrote %s (%d rows)", out_csv, len(df))

    # Quick summary for the log
    n_active = int(df["active_amp1"].sum())
    by_phase = df.loc[df["active_amp1"], "phase"].value_counts().sort_index()
    logging.info("Active days (amp>=1.0): %d", n_active)
    logging.info("Phase counts (amp>=1.0):\n%s", by_phase.to_string())


if __name__ == "__main__":
    main()
