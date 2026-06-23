# Running the JAS (July-August-September) Season Analysis

This document describes how to produce MJO composite results for the
JAS (July-August-September) season using the updated pipeline.

The config (`configs/config.yaml`) has already been updated with:

- `period.end: 2023-09-30` (extended from August to include September)
- `period.download_months: [6, 7, 8, 9]` (download June through September)
- `period.file_tag: jjas` (interim files tagged `jjas` instead of `jja`)
- New seasons: `JAS: [7, 8, 9]`, `JA: [7, 8]`, `SEP: [9]`

Existing JJA files (`*_jja_*.nc`) are unaffected; the new pipeline writes
`*_jjas_*.nc` files alongside them.

---

## Prerequisites

Ensure your CDS API key is configured:

```
~/.cdsapirc
```

It should contain your UID and API key in the standard format. See
<https://cds.climate.copernicus.eu/api-how-to> for details.

---

## Step-by-step instructions

### 1. Download ERA5 data (June-September)

```bash
python scripts/01_download_era5_daily_single_levels.py --variable all
```

This downloads months 6-9 for every year in the configured period
(1979-2023). Years whose output file already exists and contains all
expected days are skipped automatically.

### 2. Build the daily climatology

```bash
python scripts/04_make_daily_climatology.py --variable all
```

Writes:
- `data/interim/era5/{var}_jjas_1979_2023.nc`
- `data/interim/clim/{var}_clim_1991_2020_jjas.nc`

### 3. Compute daily anomalies

```bash
python scripts/05_make_daily_anomalies.py --variable all
```

Writes:
- `data/interim/anom/{var}_anom_daily_jjas_1979_2023.nc`

### 4. Build MJO composites (all-year ENSO)

```bash
python scripts/06_make_mjo_composites.py --variable all --seasons JAS JA SEP
```

Writes composite NetCDFs and sample-count CSVs under
`data/processed/composites/`.

### 5. Plot maps

```bash
python scripts/07_plot_maps.py --variable all --seasons JAS JA SEP
```

### 6. Japan region summary

```bash
python scripts/08_make_japan_region_summary.py --variable all
```

---

## ENSO-stratified versions

Repeat steps 4-6 for each ENSO category:

```bash
# El Nino
python scripts/06_make_mjo_composites.py --variable all --seasons JAS JA SEP --enso ElNino
python scripts/07_plot_maps.py           --variable all --seasons JAS JA SEP --enso ElNino
python scripts/08_make_japan_region_summary.py --variable all --enso ElNino

# La Nina
python scripts/06_make_mjo_composites.py --variable all --seasons JAS JA SEP --enso LaNina
python scripts/07_plot_maps.py           --variable all --seasons JAS JA SEP --enso LaNina
python scripts/08_make_japan_region_summary.py --variable all --enso LaNina

# Neutral
python scripts/06_make_mjo_composites.py --variable all --seasons JAS JA SEP --enso Neutral
python scripts/07_plot_maps.py           --variable all --seasons JAS JA SEP --enso Neutral
python scripts/08_make_japan_region_summary.py --variable all --enso Neutral
```

---

## Notes

- The JJA analysis (original pipeline) is unaffected. If you need to
  regenerate JJA results from a config that has `file_tag: jjas`, run
  the scripts with a JJA-specific config or temporarily set
  `period.file_tag: jja` and `period.end: 2023-08-31` in the config.
- The `file_tag` key defaults to `jja` when absent from the config, so
  older configs remain fully backward-compatible.
