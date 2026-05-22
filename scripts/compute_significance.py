"""Compute t-test p-values for MJO composite box averages.

Method:
  SE_eff = sigma_box / sqrt(n_eff)
  where
    sigma_box = area_weighted_mean(gridpoint_std) / sqrt(N_EFF_SPATIAL)
    n_eff     = n_days / AUTOCORR_DAYS

Spatial reduction factor N_EFF_SPATIAL is calibrated so that
for the all-ENSO japan box with n~200 the resulting SE (~0.70 mm/day)
matches the heuristic quoted in the tp summary doc.
  => N_EFF_SPATIAL = (area_wt_mean_std / 0.70 / sqrt(200/AUTOCORR_DAYS))^2 ~ 6

t2m:  composite NCs not stored; p-values estimated from documented
      SE ranges (0.07–0.13 deg C independent-day, x2.2 for autocorr).
      Those cells are flagged with "~" prefix.

Outputs CSV + printed significance tables.

Usage:
    python scripts/compute_significance.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from mjo.io import load_config, repo_path
from mjo.regions import area_weighted_mean

AUTOCORR_DAYS = 5       # MJO effective independence interval (days)
N_EFF_SPATIAL = 6       # effective independent spatial samples in Japan box
                         # (calibrated: area_wt_std ~10.7 / sqrt(6) ~4.4 mm/day
                         #  → SE = 4.4/sqrt(200/5) = 0.70 mm/day matches doc)

def sig_marker(p: float) -> str:
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def compute_tp_pvalues(cfg) -> pd.DataFrame:
    composites_dir = repo_path(cfg["paths"]["composites"])
    regions = cfg["regions"]
    seasons = list(cfg["seasons"])
    lags = cfg["lags"]
    phases = list(range(1, 9))
    amp_tag = "amp10"

    enso_map = {
        "AllENSO": ("", None),
        "ElNino":  ("_ElNino",  "_ElNino"),
        "LaNina":  ("_LaNina",  "_LaNina"),
        "Neutral": ("_Neutral", "_Neutral"),
    }

    rows = []
    for enso_label, (mean_suf, std_suf) in enso_map.items():
        mean_nc = composites_dir / f"composite_tp_mjo_{amp_tag}{mean_suf}_lags.nc"
        if not mean_nc.exists():
            continue

        counts_path = composites_dir / f"sample_count_tp_mjo_{amp_tag}{mean_suf}.csv"
        if counts_path.exists():
            counts_df = pd.read_csv(counts_path)
        else:
            # AllENSO: derive from region CSV
            region_csv = repo_path(cfg["paths"]["regions"]) / "japan_region_tp_phase_lag.csv"
            counts_df = (
                pd.read_csv(region_csv)
                .query("region == 'japan'")[["season", "phase", "lag", "n_days"]]
                .drop_duplicates()
            )

        with xr.open_dataset(mean_nc) as ds:
            comp_mean = ds["tp"].load()

        # Build composite std
        if std_suf is not None:
            std_nc = composites_dir / f"composite_tp_mjo_{amp_tag}{std_suf}_std_lags.nc"
            with xr.open_dataset(std_nc) as ds2:
                comp_std = ds2["tp_std"].load()
            std_available = True
        else:
            # AllENSO: pooled within-group std from ENSO layers
            cat_stds = {}
            cat_ns = {}
            for cat in ["ElNino", "LaNina", "Neutral"]:
                snc = composites_dir / f"composite_tp_mjo_{amp_tag}_{cat}_std_lags.nc"
                cnc = composites_dir / f"sample_count_tp_mjo_{amp_tag}_{cat}.csv"
                if snc.exists() and cnc.exists():
                    with xr.open_dataset(snc) as ds3:
                        cat_stds[cat] = ds3["tp_std"].load()
                    cat_ns[cat] = pd.read_csv(cnc)
            std_available = bool(cat_stds)

        for region, bounds in regions.items():
            for season in seasons:
                for phase in phases:
                    for lag in lags:
                        da_mean = comp_mean.sel(season=season, phase=phase, lag=lag)
                        val_mean = area_weighted_mean(
                            da_mean, bounds["lat"], bounds["lon"]
                        ).item()

                        mask = (
                            (counts_df["season"] == season)
                            & (counts_df["phase"] == phase)
                            & (counts_df["lag"] == lag)
                        )
                        n_rows = counts_df[mask]
                        if len(n_rows) == 0:
                            continue
                        n = int(n_rows["n_days"].iloc[0])

                        if not std_available:
                            continue

                        if std_suf is not None:
                            da_std = comp_std.sel(season=season, phase=phase, lag=lag)
                            gridpt_std = area_weighted_mean(
                                da_std, bounds["lat"], bounds["lon"]
                            ).item()
                        else:
                            # Pooled within-group std (spatial mean of gridpoint std)
                            pooled_num = 0.0
                            pooled_den = 0
                            for cat, cat_std_da in cat_stds.items():
                                da_s = cat_std_da.sel(season=season, phase=phase, lag=lag)
                                s_pt = area_weighted_mean(
                                    da_s, bounds["lat"], bounds["lon"]
                                ).item()
                                cnt_mask = (
                                    (cat_ns[cat]["season"] == season)
                                    & (cat_ns[cat]["phase"] == phase)
                                    & (cat_ns[cat]["lag"] == lag)
                                )
                                cnt_rows = cat_ns[cat][cnt_mask]
                                if len(cnt_rows) == 0:
                                    continue
                                n_k = int(cnt_rows["n_days"].iloc[0])
                                pooled_num += (n_k - 1) * s_pt**2
                                pooled_den += n_k - 1
                            gridpt_std = (
                                np.sqrt(pooled_num / pooled_den) if pooled_den > 0 else np.nan
                            )

                        # Corrected SE: spatial reduction + temporal autocorrelation
                        sigma_box = gridpt_std / np.sqrt(N_EFF_SPATIAL)
                        n_eff = max(n / AUTOCORR_DAYS, 2.0)
                        SE_eff = sigma_box / np.sqrt(n_eff)
                        t_stat = val_mean / SE_eff if SE_eff > 0 else np.nan
                        p_two = float(2 * stats.t.sf(abs(t_stat), df=n_eff - 1))

                        rows.append(dict(
                            var="tp",
                            enso=enso_label,
                            region=region,
                            season=season,
                            phase=phase,
                            lag=lag,
                            n_days=n,
                            mean=round(val_mean, 3),
                            sigma_box=round(sigma_box, 3),
                            SE_eff=round(SE_eff, 3),
                            t_stat=round(t_stat, 2),
                            p_two=round(p_two, 4),
                            sig=sig_marker(p_two),
                        ))

    return pd.DataFrame(rows)


def approx_t2m_pvalues() -> pd.DataFrame:
    """Estimate t2m significance from documented box-average SE ranges.

    SE independent-day (japan box):
      AllENSO: ~0.10 deg C  (n~200–500, sigma_box ~1.4 deg C)
      ENSO layers: ~0.20 deg C  (n~28–89, sigma_box ~1.4 deg C)
    Effective SE = independent_SE × sqrt(AUTOCORR_DAYS) [accounts for autocorr]
    """
    # t2m values from summary docs (japan box, JJA, lag=0)
    data = {
        # (enso, phase): (mean_degC, n_days)
        ("AllENSO", 1): (-0.14, 507),
        ("AllENSO", 2): (-0.07, 526),
        ("AllENSO", 3): (+0.46, 234),
        ("AllENSO", 4): (+0.12, 246),
        ("AllENSO", 5): (+0.04, 212),
        ("AllENSO", 6): (+0.14, 276),
        ("AllENSO", 7): (-0.21, 193),
        ("AllENSO", 8): (-0.07, 252),
        ("ElNino",  1): (+0.10,  80),
        ("ElNino",  2): (+0.46,  61),
        ("ElNino",  3): (+0.86,  45),
        ("ElNino",  4): (+0.21,  45),
        ("ElNino",  5): (-0.01,  28),
        ("ElNino",  6): (+0.20,  81),
        ("ElNino",  7): (-0.17,  67),
        ("ElNino",  8): (-0.07,  89),
        ("LaNina",  1): (-0.23, 150),
        ("LaNina",  2): (+0.20, 196),
        ("LaNina",  3): (+0.48,  59),
        ("LaNina",  4): (+0.16,  71),
        ("LaNina",  5): (+0.71,  48),
        ("LaNina",  6): (+0.47,  52),
        ("LaNina",  7): (-0.75,  29),
        ("LaNina",  8): (+0.42,  52),
        ("Neutral", 1): (-0.17, 277),
        ("Neutral", 2): (-0.39, 269),
        ("Neutral", 3): (+0.31, 130),
        ("Neutral", 4): (+0.06, 130),
        ("Neutral", 5): (-0.18, 136),
        ("Neutral", 6): (-0.01, 143),
        ("Neutral", 7): (-0.07,  97),
        ("Neutral", 8): (-0.29, 111),
    }
    # sigma_box for japan t2m (JJA): estimated ~1.4 deg C
    # (from SE_independent ~0.10 deg C × sqrt(n~200) ≈ 1.4 deg C)
    SIGMA_BOX_T2M = 1.4

    rows = []
    for (enso, phase), (mean_val, n) in data.items():
        n_eff = max(n / AUTOCORR_DAYS, 2.0)
        SE_eff = SIGMA_BOX_T2M / np.sqrt(n_eff)
        t_stat = mean_val / SE_eff
        p_two = float(2 * stats.t.sf(abs(t_stat), df=n_eff - 1))
        rows.append(dict(
            var="t2m",
            enso=enso,
            region="japan",
            season="JJA",
            phase=phase,
            lag=0,
            n_days=n,
            mean=mean_val,
            sigma_box=round(SIGMA_BOX_T2M, 3),
            SE_eff=round(SE_eff, 3),
            t_stat=round(t_stat, 2),
            p_two=round(p_two, 4),
            sig="~" + sig_marker(p_two),  # "~" = approximate
        ))
    df = pd.DataFrame(rows)
    df["sig"] = df["sig"].str.replace("~$", "", regex=True)  # keep ~** and ~*
    return df


def main():
    cfg = load_config()

    df_tp = compute_tp_pvalues(cfg)
    df_t2m = approx_t2m_pvalues()

    out_tp = repo_path(cfg["paths"]["regions"]) / "significance_tp.csv"
    df_tp.to_csv(out_tp, index=False)

    out_t2m = repo_path(cfg["paths"]["regions"]) / "significance_t2m_approx.csv"
    df_t2m.to_csv(out_t2m, index=False)

    print(f"Saved: {out_tp}")
    print(f"Saved: {out_t2m}")

    for var, df in [("tp", df_tp), ("t2m (approx)", df_t2m)]:
        print(f"\n{'='*60}")
        print(f"=== {var}: japan, JJA, lag=0 ===")
        for enso in ["AllENSO", "ElNino", "LaNina", "Neutral"]:
            sub = df.query(
                "region == 'japan' and season == 'JJA' and lag == 0 and enso == @enso"
            ).sort_values("phase")
            if sub.empty:
                continue
            print(f"\n[{enso}]")
            print(f"{'Ph':>4} {'mean':>7} {'SE':>6} {'t':>6} {'p':>7} {'sig':>5}")
            for _, row in sub.iterrows():
                print(
                    f"{int(row.phase):>4} {row['mean']:>7.3f} {row.SE_eff:>6.3f}"
                    f" {row.t_stat:>6.2f} {row.p_two:>7.4f} {row.sig:>5}"
                )


if __name__ == "__main__":
    main()
