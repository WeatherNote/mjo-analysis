"""Download NOAA CPC ONI and classify each year by its JJA value.

Usage:
    python scripts/03_prepare_enso.py [--no-download]

Writes:
    data/raw/enso/oni.ascii.txt
    data/raw/enso/enso_category_1979_2023.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from mjo.enso import classify, jja_oni, parse_oni  # noqa: E402
from mjo.io import ensure_dir, load_config, repo_path  # noqa: E402


def fetch(url: str, dest: Path) -> Path:
    logging.info("Downloading %s -> %s", url, dest)
    headers = {"User-Agent": "mjo-analysis/0.1 (research; ERA5 + ENSO classification)"}
    r = requests.get(url, timeout=60, headers=headers)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = ensure_dir(repo_path(cfg["paths"]["raw_enso"]))
    ascii_path = raw_dir / "oni.ascii.txt"

    if not ascii_path.exists() or not args.no_download:
        try:
            fetch(cfg["enso"]["oni_url"], ascii_path)
        except Exception as exc:
            if ascii_path.exists():
                logging.warning("Download failed (%s); using cached %s", exc, ascii_path)
            else:
                raise

    df = parse_oni(ascii_path)
    season_label = cfg["enso"]["season_label"]
    df_jja = jja_oni(df, season_label=season_label)
    threshold = float(cfg["enso"]["threshold"])
    df_class = classify(df_jja, threshold=threshold, value_col=f"{season_label}_ONI")

    y0 = int(str(cfg["period"]["start"])[:4])
    y1 = int(str(cfg["period"]["end"])[:4])
    df_class = df_class[(df_class["year"] >= y0) & (df_class["year"] <= y1)].reset_index(drop=True)

    out_csv = raw_dir / "enso_category_1979_2023.csv"
    df_class.to_csv(out_csv, index=False)
    logging.info("Wrote %s (%d rows)", out_csv, len(df_class))
    logging.info("Category counts:\n%s", df_class["ENSO_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
