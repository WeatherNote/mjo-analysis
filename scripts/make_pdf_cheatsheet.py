"""Generate Japanese and English MJO cheat sheet PDFs with figures.

Outputs:
    docs/cheatsheet_JA.pdf
    docs/cheatsheet_EN.pdf

Usage:
    python scripts/make_pdf_cheatsheet.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from mjo.io import load_config, repo_path

# ── fonts ────────────────────────────────────────────────────────────────────
FONT_JP = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
fm.fontManager.addfont(FONT_JP)
JP_FAMILY = fm.FontProperties(fname=FONT_JP).get_name()  # "IPAGothic"

FIG_DIR   = Path("figures")
MAPS_DIR  = FIG_DIR / "maps"
HEAT_DIR  = FIG_DIR / "heatmaps"
DOCS_DIR  = Path("docs")

# ── colour palette ───────────────────────────────────────────────────────────
C_HEADER  = "#1a3a5c"   # dark navy
C_HOT     = "#d62728"   # red
C_COOL    = "#1f77b4"   # blue
C_WET     = "#2ca02c"   # green
C_DRY     = "#ff7f0e"   # orange
C_SIG2    = "#333333"   # bold ** text
C_BG_ALT  = "#f5f8fc"   # alternating row bg
C_BG_HEAD = "#e0e8f0"   # column header bg

# ── text content (EN + JA) ───────────────────────────────────────────────────
CONVECTION = {
    1: {"en": "W. Hemisphere\n/ Africa",   "ja": "西半球・アフリカ"},
    2: {"en": "Western\nIndian Ocean",      "ja": "インド洋西部"},
    3: {"en": "Eastern\nIndian Ocean",      "ja": "インド洋東部"},
    4: {"en": "Maritime\nContinent (W)",    "ja": "海洋大陸西部"},
    5: {"en": "MC / Philippine\nSea",       "ja": "海洋大陸〜フィリピン海"},
    6: {"en": "Western\nPacific",           "ja": "西部太平洋"},
    7: {"en": "Central\nPacific",           "ja": "中部太平洋"},
    8: {"en": "Eastern Pacific\n/ W. Hem.", "ja": "東部太平洋〜西半球"},
}

# ── data tables ──────────────────────────────────────────────────────────────
# Format: (t2m, t2m_sig, tp, tp_sig, tendency_en, tendency_ja)
ALL_ENSO = {
    1: (-0.14, "",   -0.16, "*",  "Slightly cool / E. Japan dry",           "弱涼 / 東日本少雨"),
    2: (-0.07, "",   +0.21, "",   "Near-neutral / Slightly wet",             "弱涼 / 弱多雨"),
    3: (+0.46, "**", -0.55, "",   "HOTTEST / Dry  ★",                       "最暑 / 少雨  ★"),
    4: (+0.12, "",   +0.61, "",   "Slightly warm / Wet",                     "弱暑 / 多雨"),
    5: (+0.04, "",   +0.32, "",   "Neutral / Slightly wet",                  "中立 / 弱多雨"),
    6: (+0.14, "",   -0.61, "**", "Slightly warm / W. Japan DRY  ★",        "弱暑 / 西日本少雨  ★"),
    7: (-0.21, "",   +0.24, "",   "Cool (N. Japan) / Slightly wet",          "北日本涼 / 弱多雨"),
    8: (-0.07, "",   +1.06, "**", "Slightly cool / WETTEST  ★",             "弱涼 / 最強多雨  ★"),
}

EL_NINO = {
    1: (+0.10, "",   -0.60, "",   "Near-neutral / Dry",                      "中立 / 少雨"),
    2: (+0.46, "",   +0.31, "",   "Warm / Wet  ★",                           "暑 / 多雨  ★"),
    3: (+0.86, "",   +0.18, "",   "HOTTEST / Near-neutral  ★",               "最暑 / 中立  ★"),
    4: (+0.21, "",   +0.85, "",   "Slightly warm / Wet",                     "弱暑 / 多雨"),
    5: (-0.01, "",   +0.53, "",   "Neutral / Slightly wet",                  "中立 / 弱多雨"),
    6: (+0.20, "",   -0.34, "",   "Slightly warm / Dry (dampened)",          "弱暑 / 少雨（減衰）"),
    7: (-0.17, "",   +0.57, "",   "Slightly cool / Slightly wet",            "弱涼 / 弱多雨"),
    8: (-0.07, "",   +1.11, "",   "Near-neutral / WETTEST  ★",               "中立 / 最強多雨  ★"),
}

LA_NINA = {
    1: (-0.23, "",   +0.36, "",   "Slightly cool / Slightly wet",            "弱涼 / 弱多雨"),
    2: (+0.20, "",   +0.25, "",   "Slightly warm / Slightly wet",            "弱暑 / 弱多雨"),
    3: (+0.48, "",   -0.18, "",   "Warm / Near-neutral",                     "暑 / 中立"),
    4: (+0.16, "",   +0.34, "",   "Slightly warm / Slightly wet",            "弱暑 / 弱多雨"),
    5: (+0.71, "",   -0.14, "",   "HOTTEST / Near-neutral  ★",               "最暑 / 中立  ★"),
    6: (+0.47, "",   -0.80, "**", "Warm / DRIEST  ★",                        "暑 / 最強少雨  ★"),
    7: (-0.75, "",   +0.68, "",   "COOLEST (fragile[!]) / Wet",             "最涼（脆弱[!]） / 多雨"),
    8: (+0.42, "",   +1.75, "",   "Slightly warm / WETTEST  ★",              "弱暑 / 最強多雨  ★"),
}

NEUTRAL = {
    1: (-0.17, "",   -0.29, "**", "Slightly cool / E. Japan dry",            "弱涼 / 東日本少雨"),
    2: (-0.39, "**", +0.15, "",   "COOLEST / Slightly wet  ★",               "最涼 / 弱多雨  ★"),
    3: (+0.31, "",   -1.06, "**", "Warm / DRIEST (Neutral-only)  ★",        "暑 / 最強少雨（Neutral限定）★"),
    4: (+0.06, "",   +0.57, "",   "Near-neutral / Wet",                      "中立 / 多雨"),
    5: (-0.18, "",   +0.29, "",   "Slightly cool / Slightly wet",            "弱涼 / 弱多雨"),
    6: (-0.01, "",   -0.70, "**", "Neutral / W. Japan dry",                  "中立 / 西日本少雨"),
    7: (-0.07, "",   -0.10, "",   "Near-neutral",                            "ほぼ中立"),
    8: (-0.29, "",   +0.72, "",   "Slightly cool / Wet",                     "弱涼 / 多雨"),
}

# ── helper: draw one regime table ────────────────────────────────────────────
def draw_regime_table(ax, data: dict, lang: str, regime_label: str, font_family: str):
    """Draw a styled phase × (t2m, tp, tendency) table in the given axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_rows = 8
    col_x  = [0.00, 0.07, 0.20, 0.33, 0.44, 0.53, 1.00]
    # columns: Phase | Convection | t2m | sig | tp | sig | Tendency
    col_labels_en = ["Ph", "Convection", "t2m\n(°C)", "", "tp\n(mm/d)", "", "Japan tendency"]
    col_labels_ja = ["位相", "対流活発域",  "t2m\n(°C)", "", "tp\n(mm/d)", "", "日本の傾向"]
    col_labels = col_labels_en if lang == "en" else col_labels_ja

    row_h = 0.82 / n_rows
    head_h = 0.10
    y_top = 0.97

    # header background
    ax.add_patch(FancyBboxPatch((0, y_top - head_h), 1, head_h,
                                boxstyle="round,pad=0", linewidth=0,
                                facecolor=C_BG_HEAD, zorder=1))

    # header text
    for ci, (lx, label) in enumerate(zip(col_x, col_labels)):
        ax.text(lx + 0.01, y_top - head_h / 2, label,
                ha="left", va="center", fontsize=5.5,
                fontfamily=font_family, fontweight="bold", color=C_HEADER,
                linespacing=1.2, zorder=2)

    # rows
    for ri, ph in enumerate(range(1, 9)):
        t2m, t2m_sig, tp, tp_sig, tend_en, tend_ja = data[ph]
        tend = tend_en if lang == "en" else tend_ja
        conv = CONVECTION[ph][lang]
        y_row = y_top - head_h - (ri + 1) * row_h

        # alternating background
        bg = C_BG_ALT if ri % 2 == 1 else "white"
        ax.add_patch(FancyBboxPatch((0, y_row), 1, row_h,
                                    boxstyle="round,pad=0", linewidth=0,
                                    facecolor=bg, zorder=1))

        # ── Phase number
        ax.text(col_x[0] + 0.005, y_row + row_h * 0.5, str(ph),
                ha="left", va="center", fontsize=7,
                fontfamily="DejaVu Sans", fontweight="bold",
                color=C_HEADER, zorder=2)

        # ── Convection
        ax.text(col_x[1] + 0.005, y_row + row_h * 0.5, conv,
                ha="left", va="center", fontsize=5,
                fontfamily=font_family, linespacing=1.2, color="#333333", zorder=2)

        # ── t2m value
        t2m_col = C_HOT if t2m > 0.15 else (C_COOL if t2m < -0.15 else "#555555")
        fw = "bold" if t2m_sig == "**" else "normal"
        ax.text(col_x[2] + 0.005, y_row + row_h * 0.5,
                f"{t2m:+.2f}",
                ha="left", va="center", fontsize=6.5,
                fontfamily="DejaVu Sans", fontweight=fw, color=t2m_col, zorder=2)

        # ── t2m sig
        if t2m_sig:
            ax.text(col_x[3], y_row + row_h * 0.5, t2m_sig,
                    ha="left", va="center", fontsize=6, fontweight="bold",
                    color=C_SIG2, zorder=2)

        # ── tp value
        tp_col = C_WET if tp > 0.3 else (C_DRY if tp < -0.3 else "#555555")
        fw2 = "bold" if tp_sig == "**" else "normal"
        ax.text(col_x[4] + 0.005, y_row + row_h * 0.5,
                f"{tp:+.2f}",
                ha="left", va="center", fontsize=6.5,
                fontfamily="DejaVu Sans", fontweight=fw2, color=tp_col, zorder=2)

        # ── tp sig
        if tp_sig:
            ax.text(col_x[5], y_row + row_h * 0.5, tp_sig,
                    ha="left", va="center", fontsize=6, fontweight="bold",
                    color=C_SIG2, zorder=2)

        # ── Tendency
        is_hot  = "★" in tend or "HOT" in tend.upper() or "最暑" in tend
        is_cool = "COOL" in tend.upper() or "最涼" in tend or "涼" in tend
        is_wet  = "WET" in tend.upper() or "多雨" in tend
        is_dry  = "DRY" in tend.upper() or "少雨" in tend
        tend_col = (C_HOT if is_hot else
                    C_COOL if is_cool else
                    C_WET if is_wet else
                    C_DRY if is_dry else "#333333")
        fw3 = "bold" if "★" in tend else "normal"
        ax.text(col_x[6] + 0.005, y_row + row_h * 0.5, tend,
                ha="left", va="center", fontsize=5.5,
                fontfamily=font_family, fontweight=fw3, color=tend_col,
                linespacing=1.2, zorder=2)

    # outer border
    ax.add_patch(FancyBboxPatch((0, y_top - head_h - n_rows * row_h), 1,
                                head_h + n_rows * row_h,
                                boxstyle="round,pad=0.005", linewidth=0.8,
                                edgecolor=C_HEADER, facecolor="none", zorder=3))


def add_map_image(ax, path: Path, title: str, font_family: str):
    if path.exists():
        img = np.array(Image.open(path))
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "figure\nnot found", ha="center", va="center",
                color="gray", fontsize=8, fontfamily=font_family)
        ax.set_facecolor("#f0f0f0")
    ax.set_title(title, fontsize=7, fontfamily=font_family,
                 pad=3, color=C_HEADER, fontweight="bold")
    ax.axis("off")


def section_header(fig, y_frac: float, text: str, font_family: str):
    fig.text(0.05, y_frac, text, fontsize=11, fontfamily=font_family,
             fontweight="bold", color=C_HEADER,
             bbox=dict(facecolor=C_BG_HEAD, edgecolor="none",
                       boxstyle="round,pad=0.3"))


# ── per-language strings ──────────────────────────────────────────────────────
STRINGS = {
    "en": {
        "title":        "Japan Summer MJO Cheat Sheet",
        "subtitle":     "ERA5 JJA  |  t2m 1994–2023 / tp 1979–2023  |  RMM amp ≥ 1.0  |  lag = 0",
        "box_note":     "Box: Japan 30–46 N, 128–146 E  |  ** p<0.05   * p<0.10 (two-sided t-test, n_eff = n/5)\n"
                        "t2m sig: Japan box-mean (σ=1.4°C)   tp sig: best same-sign sub-region",
        "all_enso":     "ALL ENSO",
        "el_nino":      "EL NIÑO REGIME",
        "la_nina":      "LA NIÑA REGIME",
        "neutral":      "NEUTRAL REGIME",
        "map_all":      "tp anomaly (mm/day) — All ENSO, JJA, lag=0",
        "map_enino":    "tp anomaly — El Niño, JJA, lag=0",
        "map_lanina":   "tp anomaly — La Niña, JJA, lag=0",
        "map_neutral":  "tp anomaly — Neutral, JJA, lag=0",
        "heat_all":     "Heatmap: Japan box tp (mm/day)",
        "heat_wj":      "Heatmap: West Japan tp",
        "heat_ej":      "Heatmap: East Japan tp",
        "sig_note":     ("Significance notes:\n"
                         "  ** p<0.05   * p<0.10  (two-sided t-test)\n"
                         "  t2m: Japan box-mean, sigma_box=1.4C, n_eff=n/5\n"
                         "  tp: best same-sign sub-region (W./E./N. Japan\n"
                         "       or Okinawa), sigma_box=gridpt sigma/sqrt(6)\n"
                         "  El Nino/La Nina layers (n=28-89): no cells\n"
                         "       formally significant\n"
                         "  [!] La Nina Phase 7: n=19, directional only"),
        "key_findings": ("Key findings:\n"
                         "  P3 (E. Indian Ocean): Hottest + Dry\n"
                         "     robust in ALL ENSO (t2m +0.46C **)\n"
                         "  P8 (E. Pacific): Strongest wet\n"
                         "     ALL ENSO (tp +1.06** ; La Nina +1.75)\n"
                         "  P6 (W. Pacific): Driest - all ENSO\n"
                         "     W. Japan -1.71 **\n"
                         "  P7 (Central Pac.): Coolest N. Japan\n"
                         "  El Nino: amplifies P3 heat (+0.86C),\n"
                         "     P4 wet (+0.85 mm/d)\n"
                         "  La Nina: hot shifts to P5-6;\n"
                         "     P8 precip peaks (+1.75 mm/d)\n"
                         "  Neutral: P3 dry Neutral-only (-1.06 **)\n"
                         "     P2 cool (-0.39C **)"),
    },
    "ja": {
        "title":        "日本夏季 MJO チートシート",
        "subtitle":     "ERA5 JJA  |  t2m 1994–2023 / tp 1979–2023  |  RMM振幅 ≥ 1.0  |  lag = 0",
        "box_note":     "ボックス: Japan (30–46N, 128–146E)  |  ** p<0.05   * p<0.10（両側t検定, n_eff = n/5）\n"
                        "t2m有意性: Japanボックス平均（σ=1.4°C）   tp有意性: 最大シグナルのサブ領域",
        "all_enso":     "全 ENSO",
        "el_nino":      "El Niño 層",
        "la_nina":      "La Niña 層",
        "neutral":      "Neutral 層",
        "map_all":      "tp アノマリ (mm/day) — 全ENSO, JJA, lag=0",
        "map_enino":    "tp アノマリ — El Niño, JJA, lag=0",
        "map_lanina":   "tp アノマリ — La Niña, JJA, lag=0",
        "map_neutral":  "tp アノマリ — Neutral, JJA, lag=0",
        "heat_all":     "ヒートマップ: Japan ボックス tp",
        "heat_wj":      "ヒートマップ: 西日本 tp",
        "heat_ej":      "ヒートマップ: 東日本 tp",
        "sig_note":     ("有意性の注記:\n"
                         "  ** p<0.05  * p<0.10（両側t検定）\n"
                         "  t2m: Japanボックス平均, σ=1.4°C, n_eff=n/5\n"
                         "  tp: 最大シグナルの同符号サブ領域\n"
                         "       （西・東・北日本 or 沖縄）, n_eff=n/5\n"
                         "  El Nino/La Nina層 (n=28-89):\n"
                         "       形式的に有意なセルなし\n"
                         "  [!] La Nina Phase 7: n=19、方向性のみ参考"),
        "key_findings": ("主な知見:\n"
                         "  P3（インド洋東部）: 最暑＋少雨\n"
                         "     全ENSO層で安定（t2m +0.46C **）\n"
                         "  P8（東部太平洋）: 最強多雨\n"
                         "     全ENSO層（tp +1.06**; La Nina +1.75）\n"
                         "  P6（西部太平洋）: 最強少雨、全ENSO安定\n"
                         "     西日本 -1.71 **\n"
                         "  P7（中部太平洋）: 北日本涼、持続的\n"
                         "  El Nino: P3暑を増幅（+0.86C）\n"
                         "     P4多雨を増幅（+0.85 mm/d）\n"
                         "  La Nina: 暑位相がP5-6へシフト\n"
                         "     P8多雨が最大（+1.75 mm/d）\n"
                         "  Neutral: P3少雨はNeutral限定（-1.06 **）\n"
                         "     P2涼（-0.39C **）"),
    },
}


# ── build one PDF ─────────────────────────────────────────────────────────────
def build_pdf(lang: str, out_path: Path):
    S = STRINGS[lang]
    ff = JP_FAMILY if lang == "ja" else "DejaVu Sans"

    with PdfPages(out_path) as pdf:

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Title + All ENSO table + All ENSO map
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.patch.set_facecolor("white")

        # --- title block ---
        fig.text(0.5, 0.97, S["title"],
                 ha="center", va="top", fontsize=18,
                 fontfamily=ff, fontweight="bold", color=C_HEADER)
        fig.text(0.5, 0.935, S["subtitle"],
                 ha="center", va="top", fontsize=8,
                 fontfamily="DejaVu Sans", color="#555555")
        fig.text(0.5, 0.905, S["box_note"],
                 ha="center", va="top", fontsize=7,
                 fontfamily=ff, color="#555555", linespacing=1.5)

        # decorative line
        fig.add_artist(plt.Line2D([0.04, 0.96], [0.893, 0.893],
                                   color=C_HEADER, linewidth=1.5,
                                   transform=fig.transFigure))

        # --- layout: left = table, right = map ---
        gs = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.04, right=0.97,
                               top=0.88, bottom=0.04,
                               wspace=0.04,
                               width_ratios=[1.05, 1])

        ax_table = fig.add_subplot(gs[0])
        ax_map   = fig.add_subplot(gs[1])

        # section label above table
        ax_table.set_title(S["all_enso"], fontsize=9, fontfamily=ff,
                           fontweight="bold", color="white", pad=4,
                           bbox=dict(facecolor=C_HEADER, edgecolor="none",
                                     boxstyle="round,pad=0.3"))

        draw_regime_table(ax_table, ALL_ENSO, lang, S["all_enso"], ff)
        add_map_image(ax_map, MAPS_DIR / "fig_tp_JJA_lag+0.png",
                      S["map_all"], ff)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2 — El Niño table + map
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")

        fig.text(0.5, 0.97, S["title"],
                 ha="center", va="top", fontsize=14,
                 fontfamily=ff, fontweight="bold", color=C_HEADER)

        fig.add_artist(plt.Line2D([0.04, 0.96], [0.945, 0.945],
                                   color=C_HEADER, linewidth=1,
                                   transform=fig.transFigure))

        gs = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.04, right=0.97,
                               top=0.93, bottom=0.04,
                               wspace=0.04, width_ratios=[1.05, 1])

        ax_t = fig.add_subplot(gs[0])
        ax_m = fig.add_subplot(gs[1])

        ax_t.set_title(S["el_nino"], fontsize=9, fontfamily=ff,
                       fontweight="bold", color="white", pad=4,
                       bbox=dict(facecolor="#c44e52", edgecolor="none",
                                 boxstyle="round,pad=0.3"))

        # small warning note
        warn_en = "[!]  n = 28-89 per phase (El Nino years); no cells formally significant"
        warn_ja = "[!]  El Nino: 各位相 n = 28-89; 形式的に有意なセルなし"
        fig.text(0.04, 0.965, warn_en if lang == "en" else warn_ja,
                 fontsize=6.5, fontfamily=ff, color="#c44e52", style="italic")

        draw_regime_table(ax_t, EL_NINO, lang, S["el_nino"], ff)
        add_map_image(ax_m, MAPS_DIR / "fig_tp_JJA_lag+0_ElNino.png",
                      S["map_enino"], ff)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 3 — La Niña table + map
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")

        fig.text(0.5, 0.97, S["title"],
                 ha="center", va="top", fontsize=14,
                 fontfamily=ff, fontweight="bold", color=C_HEADER)
        fig.add_artist(plt.Line2D([0.04, 0.96], [0.945, 0.945],
                                   color=C_HEADER, linewidth=1,
                                   transform=fig.transFigure))

        gs = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.04, right=0.97,
                               top=0.93, bottom=0.04,
                               wspace=0.04, width_ratios=[1.05, 1])

        ax_t = fig.add_subplot(gs[0])
        ax_m = fig.add_subplot(gs[1])

        ax_t.set_title(S["la_nina"], fontsize=9, fontfamily=ff,
                       fontweight="bold", color="white", pad=4,
                       bbox=dict(facecolor="#4c72b0", edgecolor="none",
                                 boxstyle="round,pad=0.3"))

        warn_en = "[!]  Phase 7 n=19 (La Nina) -- directional only, do not over-weight"
        warn_ja = "[!]  La Nina Phase 7 n=19 -- 方向性のみ参考"
        fig.text(0.04, 0.965, warn_en if lang == "en" else warn_ja,
                 fontsize=6.5, fontfamily=ff, color="#4c72b0", style="italic")

        draw_regime_table(ax_t, LA_NINA, lang, S["la_nina"], ff)
        add_map_image(ax_m, MAPS_DIR / "fig_tp_JJA_lag+0_LaNina.png",
                      S["map_lanina"], ff)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 4 — Neutral + heatmaps
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")

        fig.text(0.5, 0.97, S["title"],
                 ha="center", va="top", fontsize=14,
                 fontfamily=ff, fontweight="bold", color=C_HEADER)
        fig.add_artist(plt.Line2D([0.04, 0.96], [0.945, 0.945],
                                   color=C_HEADER, linewidth=1,
                                   transform=fig.transFigure))

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               left=0.04, right=0.97,
                               top=0.93, bottom=0.04,
                               wspace=0.06, hspace=0.15,
                               height_ratios=[1, 0.85])

        ax_t  = fig.add_subplot(gs[0, 0])
        ax_mn = fig.add_subplot(gs[0, 1:])
        ax_h1 = fig.add_subplot(gs[1, 0])
        ax_h2 = fig.add_subplot(gs[1, 1])
        ax_h3 = fig.add_subplot(gs[1, 2])

        ax_t.set_title(S["neutral"], fontsize=9, fontfamily=ff,
                       fontweight="bold", color="white", pad=4,
                       bbox=dict(facecolor="#55a868", edgecolor="none",
                                 boxstyle="round,pad=0.3"))

        draw_regime_table(ax_t, NEUTRAL, lang, S["neutral"], ff)
        add_map_image(ax_mn, MAPS_DIR / "fig_tp_JJA_lag+0_Neutral.png",
                      S["map_neutral"], ff)
        add_map_image(ax_h1, HEAT_DIR / "heatmap_japan_tp_JJA.png",
                      S["heat_all"], ff)
        add_map_image(ax_h2, HEAT_DIR / "heatmap_west_japan_tp_JJA.png",
                      S["heat_wj"], ff)
        add_map_image(ax_h3, HEAT_DIR / "heatmap_east_japan_tp_JJA.png",
                      S["heat_ej"], ff)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 5 — Key findings + ENSO × MJO matrix + significance notes
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")

        fig.text(0.5, 0.97, S["title"],
                 ha="center", va="top", fontsize=14,
                 fontfamily=ff, fontweight="bold", color=C_HEADER)
        fig.add_artist(plt.Line2D([0.04, 0.96], [0.945, 0.945],
                                   color=C_HEADER, linewidth=1,
                                   transform=fig.transFigure))

        # ENSO × MJO matrix (t2m and tp)
        # t2m matrix
        t2m_data = {
            "AllENSO": [-0.14, -0.07, +0.46, +0.12, +0.04, +0.14, -0.21, -0.07],
            "ElNino":  [+0.10, +0.46, +0.86, +0.21, -0.01, +0.20, -0.17, -0.07],
            "LaNina":  [-0.23, +0.20, +0.48, +0.16, +0.71, +0.47, -0.75, +0.42],
            "Neutral": [-0.17, -0.39, +0.31, +0.06, -0.18, -0.01, -0.07, -0.29],
        }
        tp_data = {
            "AllENSO": [-0.16, +0.21, -0.55, +0.61, +0.32, -0.61, +0.24, +1.06],
            "ElNino":  [-0.60, +0.31, +0.18, +0.85, +0.53, -0.34, +0.57, +1.11],
            "LaNina":  [+0.36, +0.25, -0.18, +0.34, -0.14, -0.80, +0.68, +1.75],
            "Neutral": [-0.29, +0.15, -1.06, +0.57, +0.29, -0.70, -0.10, +0.72],
        }

        enso_keys = ["AllENSO", "ElNino", "LaNina", "Neutral"]
        enso_labels_en = ["All", "El Niño", "La Niña", "Neutral"]
        enso_labels_ja = ["全ENSO", "El Niño", "La Niña", "Neutral"]
        enso_labels = enso_labels_en if lang == "en" else enso_labels_ja

        gs = gridspec.GridSpec(2, 2, figure=fig,
                               left=0.04, right=0.97,
                               top=0.92, bottom=0.04,
                               wspace=0.12, hspace=0.3)

        def draw_matrix(ax, data_dict, title, fmt=".2f", cmap="RdBu_r",
                        vmax=None, unit=""):
            rows = [data_dict[k] for k in enso_keys]
            arr = np.array(rows)
            vm = vmax or np.nanmax(np.abs(arr))
            im = ax.imshow(arr, cmap=cmap, vmin=-vm, vmax=vm, aspect="auto")

            ax.set_xticks(range(8))
            ax.set_xticklabels([f"P{i}" for i in range(1, 9)],
                               fontsize=7, fontfamily="DejaVu Sans")
            ax.set_yticks(range(4))
            ax.set_yticklabels(enso_labels, fontsize=7, fontfamily=ff)

            for ri, row in enumerate(rows):
                for ci, val in enumerate(row):
                    ax.text(ci, ri, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=6.5, fontfamily="DejaVu Sans",
                            color="white" if abs(val) > vm * 0.55 else "black",
                            fontweight="bold" if abs(val) > vm * 0.55 else "normal")

            ax.set_title(title, fontsize=8, fontfamily=ff, pad=4,
                         color=C_HEADER, fontweight="bold")
            plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04,
                         label=unit).ax.tick_params(labelsize=6)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

        t2m_title = "t2m anomaly (°C) — Japan box, JJA, lag=0" if lang == "en" else "t2m アノマリ (°C) — Japanボックス"
        tp_title  = "tp anomaly (mm/day) — Japan box, JJA, lag=0"  if lang == "en" else "tp アノマリ (mm/day) — Japanボックス"

        ax_t2m = fig.add_subplot(gs[0, 0])
        ax_tp  = fig.add_subplot(gs[0, 1])
        draw_matrix(ax_t2m, t2m_data, t2m_title, fmt=".2f", vmax=0.9, unit="°C")
        draw_matrix(ax_tp,  tp_data,  tp_title,  fmt=".2f", vmax=1.8, unit="mm/d")

        # Key findings text
        ax_kf = fig.add_subplot(gs[1, 0])
        ax_kf.axis("off")
        ax_kf.text(0.02, 0.98, S["key_findings"],
                   ha="left", va="top", fontsize=6.5,
                   fontfamily=ff, linespacing=1.7,
                   transform=ax_kf.transAxes, color="#222222",
                   bbox=dict(facecolor="#f0f4f8", edgecolor=C_HEADER,
                             linewidth=0.5, boxstyle="round,pad=0.5"))

        # Significance notes
        ax_sn = fig.add_subplot(gs[1, 1])
        ax_sn.axis("off")
        ax_sn.text(0.02, 0.98, S["sig_note"],
                   ha="left", va="top", fontsize=6.5,
                   fontfamily=ff, linespacing=1.7,
                   transform=ax_sn.transAxes, color="#222222",
                   bbox=dict(facecolor="#f8f4f0", edgecolor="#aa7733",
                             linewidth=0.5, boxstyle="round,pad=0.5"))

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # metadata
        d = pdf.infodict()
        d["Title"]   = S["title"]
        d["Author"]  = "ERA5 MJO Analysis"
        d["Subject"] = "Japan Summer MJO Phase Composites"

    print(f"Saved: {out_path}")


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    build_pdf("ja", DOCS_DIR / "cheatsheet_JA.pdf")
    build_pdf("en", DOCS_DIR / "cheatsheet_EN.pdf")


if __name__ == "__main__":
    main()
