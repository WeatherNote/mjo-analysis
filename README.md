# MJO Phase Composite Analysis (ERA5, Japan/East Asia summer)

This repository builds composite maps of ERA5 daily 2 m temperature and total
precipitation anomalies stratified by MJO RMM phase (1–8) over East Asia and
the western North Pacific (0–60 N, 90–180 E) for boreal summer 1979–2023. The
purpose is to evaluate how well a published table of "MJO phase-dependent
Japan summer cool / wet / hot tendencies" is supported by ERA5.

## Data sources

- **ERA5 daily statistics on single levels** (Copernicus CDS):
  `reanalysis-era5-single-levels-daily-statistics`. Variables: 2 m temperature
  (daily mean) and total precipitation (daily sum). Period: 1979-06-01 to
  2023-08-31, JJA only. Domain: 0–60 N, 90–180 E. Grid: 0.5°. Downloaded
  yearly per variable (90 requests total).
- **Bureau of Meteorology RMM index** (real-time MJO): `rmm.74toRealtime.txt`.
  *Reprocessing note:* for dates up to 2013-12-31 the calculation removed SST1
  variability, ENSO and the 120-day mean; from 2014-01-01 only the 120-day
  mean is removed. We record the processing era as a column in the cleaned CSV.
- **NOAA CPC ONI** (`oni.ascii.txt`) for ENSO categorization (used in
  step 03 / step 09 — out of the initial implementation scope).

## Design decisions

- **Climatology base period**: 1991–2020. Daily climatology is smoothed by
  truncating its Fourier expansion at 3 harmonics (configurable to running
  mean or raw).
- **Detrending**: per grid cell linear detrend in time, applied to 2 m
  temperature only. Precipitation is kept untouched.
- **MJO active days**: amplitude ≥ 1.0 (configurable). Sensitivity threshold
  1.5 is also flagged in the cleaned RMM CSV.
- **Lags**: response date d, with RMM phase / amplitude taken at d − lag.
  Default lags: 0, +5, +10 days.
- **Seasons**: JJA, JJ (Jun–Jul), AUG.

## Repository layout

```
configs/config.yaml                  # all knobs (period, area, regions, lags, ...)
src/mjo/                             # importable Python package
data/raw/{era5,mjo,enso}/            # downloaded inputs
data/interim/{era5,clim,anom}/       # standardized data, climatology, anomalies
data/processed/{composites,regions}/ # composite NetCDFs, Japan-box CSVs
figures/maps/                        # 8-panel composite maps (lag 0)
figures/maps/lag/                    # 8-panel composite maps (lag +5, +10)
figures/heatmaps/                    # phase x lag region-mean heatmaps
scripts/01..08_*.py                  # CLI entry points (run in numerical order)
```

## Quick start (Figure 1 in one t2m run)

```bash
pip install -r requirements.txt
# Configure ~/.cdsapirc with your CDS UID and API key.

python scripts/02_prepare_rmm.py
python scripts/01_download_era5_daily_single_levels.py --variable 2m_temperature
python scripts/04_make_daily_climatology.py --variable t2m
python scripts/05_make_daily_anomalies.py --variable t2m
python scripts/06_make_mjo_composites.py --variable t2m --seasons JJA --lags 0
python scripts/07_plot_maps.py --variable t2m --season JJA --lag 0
# => figures/maps/fig_t2m_JJA_lag+0.png
```

Then add precipitation and other seasons/lags:

```bash
python scripts/01_download_era5_daily_single_levels.py --variable total_precipitation
python scripts/04_make_daily_climatology.py --variable tp
python scripts/05_make_daily_anomalies.py --variable tp
python scripts/06_make_mjo_composites.py --variable all
python scripts/07_plot_maps.py --variable all --season all --lag all
python scripts/08_make_japan_region_summary.py --variable all
```

## Initial deliverables (Step 1–11 of the plan)

1. JJA t2m anomaly map, phase 1–8, lag 0
2. JJA precipitation anomaly map, phase 1–8, lag 0
3. JJ t2m / precip maps, phase 1–8, lag 0
4. AUG t2m / precip maps, phase 1–8, lag 0
5. Japan-region phase × lag heatmap for t2m
6. Japan-region phase × lag heatmap for precipitation
7. Sample-count table

The lag +5 / +10 maps and ENSO / circulation / bootstrap layers are scoped for
follow-up runs.

## Reading the maps

- 8 panels = MJO phases 1..8 (top row 1–4, bottom row 5–8).
- Shading is composite anomaly; colorbar levels match `LEVELS` in
  `src/mjo/plotting.py`. Defaults: ±3 °C for t2m, ±6 mm/day for tp.
- Panel titles show `n = <number of response days>` for that phase.
- The Japan box (30–46 N, 128–146 E) is outlined in black.

## Verification checks

- `df.amplitude.describe()` and per-phase counts in step 02 log
- JJA climatology magnitudes (e.g., ~25 °C around Tokyo latitudes)
- t2m anomaly trend by year ≈ 0 after detrend
- Each (season, phase, lag) bucket should have N ≳ 50 active days

## Out of scope (planned follow-ups)

- ENSO-stratified composites (step 09)
- Z500 / 850 hPa wind / 200 hPa wind overlays
- Bootstrap significance (year-resampled)
- Phase-group aggregation (1–2, 3–4, 5–6, 7–8)
- 0.25° high-resolution rerun
