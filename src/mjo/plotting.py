"""Plotting helpers for 8-panel MJO phase composite maps and phase x lag heatmaps."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Discrete contour levels per variable
LEVELS = {
    "t2m": [-3, -2, -1, -0.5, 0.5, 1, 2, 3],
    "tp":  [-6, -3, -1.5, -0.5, 0.5, 1.5, 3, 6],
}
CMAPS = {"t2m": "RdBu_r", "tp": "BrBG"}
UNITS = {"t2m": "°C", "tp": "mm/day"}


def _cartopy_ax(ax):
    """Add coastlines/borders to a cartopy GeoAxes if available."""
    try:
        import cartopy.feature as cfeature  # type: ignore

        ax.coastlines(resolution="110m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    except Exception:
        pass


def _japan_box(ax, lat_bnds, lon_bnds, **kwargs):
    """Overlay a Japan box rectangle."""
    lat0, lat1 = sorted(lat_bnds)
    lon0, lon1 = sorted(lon_bnds)
    ax.plot(
        [lon0, lon1, lon1, lon0, lon0],
        [lat0, lat0, lat1, lat1, lat0],
        linewidth=1.0,
        color="black",
        **kwargs,
    )


def eight_panel_map(
    da_phase: xr.DataArray,  # dims (phase, lat, lon)
    counts: dict[int, int],
    var: str,
    title: str,
    savepath: Path,
    japan_box: tuple[list[float], list[float]] | None = None,
) -> Path:
    """Draw an 8-panel composite map (phases 1..8) and save to `savepath`.

    Uses cartopy if available; otherwise falls back to plain matplotlib axes.
    """
    try:
        import cartopy.crs as ccrs  # type: ignore

        proj = ccrs.PlateCarree()
        use_cartopy = True
    except Exception:
        proj = None
        use_cartopy = False

    levels = LEVELS[var]
    cmap = CMAPS[var]
    unit = UNITS[var]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 7),
        subplot_kw={"projection": proj} if use_cartopy else {},
        constrained_layout=True,
    )

    lons = da_phase["lon"].values
    lats = da_phase["lat"].values
    cs = None
    for i, phase in enumerate(da_phase["phase"].values.tolist()):
        ax = axes.flat[i]
        field = da_phase.sel(phase=phase).values
        plot_kwargs = dict(
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        if use_cartopy:
            plot_kwargs["transform"] = ccrs.PlateCarree()
            cs = ax.contourf(lons, lats, field, **plot_kwargs)
            ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
        else:
            cs = ax.contourf(lons, lats, field, **plot_kwargs)
        _cartopy_ax(ax)
        if japan_box is not None:
            _japan_box(ax, japan_box[0], japan_box[1])
        n = counts.get(int(phase), 0)
        ax.set_title(f"Phase {phase}  n={n}", fontsize=10)

    cbar = fig.colorbar(
        cs, ax=axes.ravel().tolist(), orientation="horizontal",
        shrink=0.6, pad=0.04, label=f"anomaly ({unit})",
    )
    cbar.set_ticks(levels)
    fig.suptitle(title, fontsize=12)

    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return savepath


def phase_lag_heatmap(
    matrix: np.ndarray,  # shape (n_phase, n_lag)
    phases: list[int],
    lags: list[int],
    counts: np.ndarray | None,
    var: str,
    title: str,
    savepath: Path,
) -> Path:
    """Plot a phase (rows) x lag (cols) heatmap of area-mean anomalies."""
    levels = LEVELS[var]
    cmap = CMAPS[var]
    unit = UNITS[var]
    vmax = max(abs(levels[0]), abs(levels[-1]))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        origin="upper",
    )
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f"+{l}" if l >= 0 else str(l) for l in lags])
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases)
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("MJO phase")
    ax.set_title(title)

    for i in range(len(phases)):
        for j in range(len(lags)):
            val = matrix[i, j]
            txt = f"{val:+.2f}"
            if counts is not None:
                txt = f"{txt}\nn={int(counts[i, j])}"
            ax.text(
                j, i, txt,
                ha="center", va="center", fontsize=8,
                color="black" if abs(val) < 0.6 * vmax else "white",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, label=f"anomaly ({unit})")
    cbar.set_ticks(levels)

    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return savepath
