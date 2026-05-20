"""
Streamlit GUI around `gem2_simpeg_invert.py`.

Run with:

  streamlit run gem2_simpeg_app.py

or, with a specific environment's python on Windows:

  & "<path-to-env>\\python.exe" -m streamlit run "gem2_simpeg_app.py"

Each numbered section in the sidebar maps 1:1 to a CLI flag in
`gem2_simpeg_invert.py`, so anything you can dial in here you can also script.
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LogNorm, Normalize

# ---------------------------------------------------------------------------
# Import sibling inversion module
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gem2_simpeg_invert import (  # noqa: E402
    build_layer_thicknesses,
    get_freq_channel_columns,
    invert_one_sounding,
    read_gem2_xyz,
    stack_line_rolling_median,
)
from gem2_simpeg_lci import (  # noqa: E402
    alpha_x_from_constraint_factor,
    invert_one_line_lci,
)
from gem_csv_to_xyz import convert_gem_csv_to_xyz  # noqa: E402
from gem_inversion_xyz_export import inversion_wide_to_long_xyz  # noqa: E402


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

# Optional: prefill the sidebar path inputs by setting environment variables
# `GEM2_DEFAULT_XYZ` and `GEM2_DEFAULT_CSV` to absolute paths on your machine.
# Otherwise the inputs start blank and you can paste a path or use Upload.
DEFAULT_XYZ = os.environ.get("GEM2_DEFAULT_XYZ", "")
DEFAULT_STUDENT_GEM_CSV = os.environ.get("GEM2_DEFAULT_CSV", "")


def _ensure_xyz_from_gem_csv(csv_path: Path) -> Path:
    """Write ``*.xyz`` next to the CSV if missing or older than the CSV."""
    xyz_path = csv_path.with_suffix(".xyz")
    need = True
    if xyz_path.is_file():
        need = csv_path.stat().st_mtime > xyz_path.stat().st_mtime
    if need:
        convert_gem_csv_to_xyz(csv_path, xyz_path, skip_bad_xy=True)
    return xyz_path


def _ss_init() -> None:
    defaults = {
        "df_raw": None,
        "df_raw_path": None,
        "df_stacked": None,
        "inv_df": None,
        "misfit_df": None,
        "layer_thicknesses": None,
        "frequencies_full": None,
        "channel_cols_full": None,
        "config_used": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Cached IO
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Reading XYZ file...")
def _cached_read_xyz(path_str: str, mtime: float) -> pd.DataFrame:
    """`mtime` is part of the cache key so edits invalidate the cache."""
    return read_gem2_xyz(Path(path_str))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_line_section(
    inv_df: pd.DataFrame,
    line_id: int,
    layer_thicknesses: np.ndarray,
    plot_max_depth: Optional[float] = None,
    color_scale: str = "log",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip_pct: tuple = (5.0, 95.0),
    show_channels_strip: bool = True,
) -> plt.Figure:
    """Pcolormesh of recovered sigma vs distance-along-line, depth below a flat
    zero ground surface (GPS/DEM elevations are intentionally ignored).

    A small strip above the section traces `n_channels_used` per sounding so
    you can see at a glance whether visual artefacts in the section line up
    with channel-count drops along the line."""
    sec = inv_df[inv_df["Line"] == line_id].copy().sort_values("Sample")
    if sec.empty:
        raise ValueError(f"Line {line_id} not in result.")
    sigma_cols = [c for c in inv_df.columns if c.startswith("sigma_layer_")]
    sigma = sec[sigma_cols].to_numpy(dtype=float).T  # (n_layers, n_soundings)
    sigma = np.clip(sigma, 1e-6, None)

    x = sec["X"].to_numpy(dtype=float)
    y = sec["Y"].to_numpy(dtype=float)
    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    along = np.r_[0.0, np.cumsum(step)]

    nlay = sigma.shape[0]
    if nlay == len(layer_thicknesses):
        plot_thicknesses = np.asarray(layer_thicknesses, dtype=float)
    elif nlay == len(layer_thicknesses) + 1:
        plot_thicknesses = np.r_[
            np.asarray(layer_thicknesses, dtype=float), float(layer_thicknesses[-1])
        ]
    else:
        total = float(np.sum(layer_thicknesses))
        plot_thicknesses = np.full(nlay, total / nlay)
    edges = np.r_[0.0, np.cumsum(plot_thicknesses)]
    centres = 0.5 * (edges[:-1] + edges[1:])

    sigma_finite = sigma[np.isfinite(sigma)]
    auto_vmin = max(float(np.nanpercentile(sigma_finite, float(clip_pct[0]))), 1e-6)
    auto_vmax = float(np.nanpercentile(sigma_finite, float(clip_pct[1])))
    if auto_vmax <= auto_vmin:
        auto_vmax = auto_vmin * 10.0
    eff_vmin = float(vmin) if vmin is not None and vmin > 0 else auto_vmin
    eff_vmax = float(vmax) if vmax is not None and vmax > eff_vmin else auto_vmax
    if color_scale == "log":
        norm = LogNorm(vmin=eff_vmin, vmax=eff_vmax)
    else:
        norm = Normalize(vmin=eff_vmin, vmax=eff_vmax)

    if show_channels_strip and "n_channels_used" in sec.columns:
        fig, (ax_top, ax) = plt.subplots(
            2, 1, figsize=(11, 5.2),
            gridspec_kw={"height_ratios": [0.6, 4.5], "hspace": 0.05},
            sharex=True,
        )
        n_used = sec["n_channels_used"].to_numpy(dtype=float)
        n_total_per = sec["n_channels_total"].to_numpy(dtype=float)
        ax_top.fill_between(along, 0, n_used, color="tab:gray", alpha=0.85, step="mid")
        ax_top.plot(along, n_total_per, color="black", linewidth=0.8, linestyle=":")
        ax_top.set_ylabel("ch used", fontsize=9)
        ax_top.set_ylim(0, max(1.0, float(np.nanmax(n_total_per)) + 0.5))
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(11, 4.5))

    pm = ax.pcolormesh(along, centres, sigma, shading="nearest", cmap="RdYlBu_r", norm=norm)
    ax.axhline(0.0, color="k", linewidth=1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Distance along profile (m)")
    ax.set_ylabel("Depth below ground (m)")

    total_finite = float(np.sum(plot_thicknesses))
    depth_to_show = float(plot_max_depth) if plot_max_depth is not None else total_finite
    ax.set_ylim(depth_to_show, 0.0)

    title_extra = ""
    if "doi_m" in sec.columns:
        doi_y = pd.to_numeric(sec["doi_m"], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isfinite(doi_y)):
            ax.plot(
                along,
                doi_y,
                color="lime",
                linewidth=2.0,
                linestyle="-",
                solid_capstyle="round",
                zorder=10,
                label="DOI (m)",
            )
            ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
            title_extra = " | lime line = DOI depth"

    ax.set_title(
        f"Line {line_id}: recovered sigma (S/m, {color_scale}; "
        f"vmin {eff_vmin:.3g}, vmax {eff_vmax:.3g}){title_extra}"
    )

    cbar = fig.colorbar(pm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("sigma (S/m)")
    fig.tight_layout()
    return fig


def _plot_sounding_profile(
    inv_df: pd.DataFrame, obs_id: int, layer_thicknesses: np.ndarray
) -> plt.Figure:
    """Stairstep sigma(z) for one sounding."""
    row = inv_df[inv_df["obs_id"] == obs_id]
    if row.empty:
        raise ValueError(f"obs_id {obs_id} not in result.")
    row = row.iloc[0]
    sigma_cols = [c for c in inv_df.columns if c.startswith("sigma_layer_")]
    sigma = row[sigma_cols].to_numpy(dtype=float)
    nlay = sigma.shape[0]
    if nlay == len(layer_thicknesses):
        plot_thicknesses = np.asarray(layer_thicknesses, dtype=float)
    elif nlay == len(layer_thicknesses) + 1:
        plot_thicknesses = np.r_[
            np.asarray(layer_thicknesses, dtype=float), float(layer_thicknesses[-1])
        ]
    else:
        plot_thicknesses = np.full(nlay, float(np.mean(layer_thicknesses)))
    edges = np.r_[0.0, np.cumsum(plot_thicknesses)]

    fig, ax = plt.subplots(figsize=(4, 6))
    for j in range(nlay):
        ax.plot([sigma[j], sigma[j]], [edges[j], edges[j + 1]], "b-", linewidth=2)
        if j + 1 < nlay:
            ax.plot([sigma[j], sigma[j + 1]], [edges[j + 1], edges[j + 1]], "b-", linewidth=2)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("sigma (S/m)")
    ax.set_ylabel("Depth below ground (m)")
    ax.grid(True, which="both", alpha=0.3)
    tt = (
        f"Line {int(row['Line'])} sample {int(row['Sample'])} | "
        f"chi^2 {float(row['chi2']):.2f} | {int(row['n_channels_used'])}/{int(row['n_channels_total'])} ch"
    )
    if "doi_m" in inv_df.columns:
        dm = pd.to_numeric(row.get("doi_m"), errors="coerce")
        if np.isfinite(dm):
            ax.axhline(float(dm), color="tab:red", linestyle="--", linewidth=1.8, zorder=5, label="DOI")
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
            tt += f" | DOI {float(dm):.2f} m"
    ax.set_title(tt)
    fig.tight_layout()
    return fig


def _plot_map(
    df: pd.DataFrame,
    color_col: str,
    log_color: bool = False,
    title: str = "",
) -> plt.Figure:
    """X/Y scatter coloured by a column, used for both raw coverage and inversion summaries."""
    fig, ax = plt.subplots(figsize=(8, 6))
    c = df[color_col].to_numpy(dtype=float)
    finite = np.isfinite(c)
    c_finite = c[finite]
    scatter_kwargs = {"cmap": "viridis", "s": 14}
    if log_color and (c_finite > 0).any():
        vmin = max(float(np.nanpercentile(c_finite[c_finite > 0], 5)), 1e-6)
        vmax = float(np.nanpercentile(c_finite[c_finite > 0], 95))
        if vmax <= vmin:
            vmax = vmin * 10
        # Newer matplotlib disallows passing a Normalize instance together
        # with vmin/vmax; the limits live inside the norm itself.
        scatter_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = float(np.nanpercentile(c_finite, 5)) if c_finite.size else 0.0
        vmax = float(np.nanpercentile(c_finite, 95)) if c_finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        scatter_kwargs["vmin"] = vmin
        scatter_kwargs["vmax"] = vmax
    sc = ax.scatter(df["X"], df["Y"], c=c, **scatter_kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(color_col)
    fig.tight_layout()
    return fig


def _plot_channel_distribution(
    df_raw: pd.DataFrame, channel_cols_full: List[str]
) -> plt.Figure:
    """Histogram of each Q/I channel, marking the negative-Q region in red."""
    n = len(channel_cols_full)
    n_col = 2
    n_row = int(np.ceil(n / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(11, 2.6 * n_row))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, channel_cols_full):
        v = pd.to_numeric(df_raw[col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(v, bins=60, color="tab:blue", alpha=0.7)
        if col.startswith("Q_"):
            n_neg = int((v < 0).sum())
            n_pos = int((v >= 0).sum())
            ax.axvspan(min(v.min(), -1.0), 0.0, color="tab:red", alpha=0.10)
            ax.set_title(f"{col} | neg {n_neg}, pos {n_pos}")
        else:
            ax.set_title(col)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def _plot_channel_misfit(misfit_df: pd.DataFrame, channel_cols: List[str]) -> plt.Figure:
    """Per-channel RMSE / median |residual| bar plot."""
    rmse = []
    med = []
    used = []
    for ch in channel_cols:
        col = f"res_{ch}"
        if col not in misfit_df.columns:
            rmse.append(np.nan)
            med.append(np.nan)
            used.append(0)
            continue
        v = pd.to_numeric(misfit_df[col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        used.append(len(v))
        if v.size == 0:
            rmse.append(np.nan)
            med.append(np.nan)
        else:
            rmse.append(float(np.sqrt(np.mean(v**2))))
            med.append(float(np.median(np.abs(v))))

    x = np.arange(len(channel_cols))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax1.bar(x, rmse, color="tab:blue", alpha=0.85)
    ax1.set_ylabel("RMSE (ppm)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Per-channel residual diagnostics")
    for xi, n_used in zip(x, used):
        ax1.text(xi, 0.0, f"n={n_used}", ha="center", va="bottom", fontsize=8, color="black")

    ax2.bar(x, med, color="tab:orange", alpha=0.85)
    ax2.set_ylabel("Median |residual| (ppm)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(channel_cols, rotation=45, ha="right")
    fig.tight_layout()
    return fig


def _along_track_distance_m(df_line: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Cumulative horizontal path length (m) along trace; same convention as stacking."""
    g = df_line.sort_values("Sample").reset_index(drop=True)
    dx = pd.to_numeric(g["X"], errors="coerce").to_numpy(dtype=float)
    dy = pd.to_numeric(g["Y"], errors="coerce").to_numpy(dtype=float)
    if len(dx) < 2:
        return np.zeros(len(dx), dtype=float), g
    step = np.sqrt(np.diff(dx) ** 2 + np.diff(dy) ** 2)
    along = np.r_[0.0, np.cumsum(step)]
    return along, g


def _plot_line_raw_vs_stacked(
    line_id: int,
    df_raw: pd.DataFrame,
    df_stacked: Optional[pd.DataFrame],
    channels: List[str],
    raw_rolling: int = 7,
    y_scale: str = "symlog",
    pct_low: float = 0.5,
    pct_high: float = 99.5,
) -> plt.Figure:
    """
    All EM channels on one axes: raw vs distance (scatter + rolling median) and
    stacked median with **error bars** = within-bin std (``<col>_std``).

    ``y_scale``: ``linear`` | ``symlog`` | ``percentile`` — the latter clips the
    display range to ``pct_low``–``pct_high`` percentiles of plotted finite ppm
    so outliers do not flatten the figure.
    """
    raw_line = df_raw[df_raw["Line"] == line_id].copy()
    if raw_line.empty:
        raise ValueError(f"No raw rows for Line {line_id}")

    if not channels:
        raise ValueError("No channels selected for plotting.")

    rw = int(max(2, raw_rolling))
    cmap = plt.cm.tab10(np.linspace(0, 1, min(10, max(len(channels), 1))))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    stack_present = (
        df_stacked is not None
        and not df_stacked.empty
        and "Line" in df_stacked.columns
    )
    stk = (
        df_stacked[df_stacked["Line"] == line_id].copy()
        if stack_present
        else pd.DataFrame()
    )

    all_y_for_limits: List[np.ndarray] = []

    for idx, ch in enumerate(channels):
        color = cmap[idx % len(cmap)]
        if ch not in raw_line.columns:
            continue

        along, g = _along_track_distance_m(raw_line)
        y_raw = pd.to_numeric(g[ch], errors="coerce").to_numpy(dtype=float)
        ser = pd.Series(y_raw)
        y_rm = ser.rolling(window=rw, center=True, min_periods=2).median().to_numpy(dtype=float)

        m_sc = np.isfinite(along) & np.isfinite(y_raw)
        if np.any(m_sc):
            all_y_for_limits.append(y_raw[m_sc])
            ax.scatter(
                along[m_sc],
                y_raw[m_sc],
                s=6,
                alpha=0.22,
                color=color,
                label=f"{ch} raw",
                zorder=1,
                rasterized=True,
            )

        m_roll = np.isfinite(along) & np.isfinite(y_rm)
        if np.any(m_roll):
            all_y_for_limits.append(y_rm[m_roll])
            ax.plot(
                along[m_roll],
                y_rm[m_roll],
                color=color,
                lw=1.1,
                linestyle="--",
                alpha=0.75,
                label=f"{ch} raw med (w={rw})",
                zorder=2,
            )

        if not stk.empty and ch in stk.columns and "stack_distance_m" in stk.columns:
            xs = stk["stack_distance_m"].to_numpy(dtype=float)
            ys = pd.to_numeric(stk[ch], errors="coerce").to_numpy(dtype=float)
            std_col = f"{ch}_std"
            if std_col in stk.columns:
                yss = pd.to_numeric(stk[std_col], errors="coerce").to_numpy(dtype=float)
            else:
                yss = np.zeros_like(ys)
            m = np.isfinite(xs) & np.isfinite(ys)
            if np.any(m):
                all_y_for_limits.append(ys[m])
                all_y_for_limits.append((ys - yss)[m])
                all_y_for_limits.append((ys + yss)[m])
                yerr = np.where(np.isfinite(yss), yss, 0.0)
                ax.errorbar(
                    xs[m],
                    ys[m],
                    yerr=yerr[m],
                    fmt=".-",
                    color=color,
                    markersize=4,
                    capsize=2,
                    capthick=1,
                    elinewidth=1,
                    label=f"{ch} stacked ± SD",
                    alpha=0.95,
                    zorder=5,
                )

    ax.set_xlabel("Distance along line (m)")
    ax.set_ylabel("Response (ppm)")
    ax.grid(True, alpha=0.3)

    y_scale_l = (y_scale or "symlog").lower()
    y_all = np.concatenate([a.ravel() for a in all_y_for_limits if a.size]) if all_y_for_limits else np.array([])
    fin = y_all[np.isfinite(y_all)]

    if fin.size:
        if y_scale_l == "symlog":
            linthresh = float(np.clip(np.nanmedian(np.abs(fin)) * 0.15, 1.0, 50.0))
            ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
        elif y_scale_l == "percentile":
            lo = float(np.percentile(fin, pct_low))
            hi = float(np.percentile(fin, pct_high))
            if lo >= hi:
                pad = max(abs(lo) * 0.05, 1.0)
                lo, hi = lo - pad, hi + pad
            ax.set_ylim(lo, hi)
        else:
            ax.set_yscale("linear")

    ax.set_title(
        f"Line {line_id}: all channels — raw (cloud + dashed median) vs stacked (lines with SD error bars)",
        fontsize=11,
    )
    h, lab = ax.get_legend_handles_labels()
    if h:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.92)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


st.set_page_config(page_title="GEM-2 SimPEG Inversion", layout="wide")
_ss_init()
st.title("GEM-2 1D SimPEG inversion")

# === Sidebar ===

with st.sidebar:
    st.header("1. Data")
    data_source = st.radio(
        "Input",
        options=["XYZ file path", "GEM CSV path", "Upload (.csv or .xyz)"],
        index=1,
        help=(
            "GEM CSV: raw exporter `*_gem.csv` (UTM X/Y, I/Q ppm columns). "
            "Rows with missing coordinates are skipped when building the sidecar `.xyz`. "
            "XYZ: file already converted with gem_csv_to_xyz.py."
        ),
    )
    if data_source == "XYZ file path":
        xyz_path_str = st.text_input(
            "GEM-2 XYZ path",
            value=DEFAULT_XYZ,
            help="Whitespace-column `.xyz` from gem_csv_to_xyz.py",
        )
        xyz_path = Path(xyz_path_str.strip().strip('"'))
    elif data_source == "GEM CSV path":
        csv_path_str = st.text_input(
            "GEM CSV path",
            value=DEFAULT_STUDENT_GEM_CSV,
            help="Raw GEM-2 exporter `*_gem.csv` with X/Y and I/Q ppm columns",
        )
        csv_path = Path(csv_path_str.strip().strip('"'))
        if not csv_path.is_file():
            st.error(f"CSV not found: {csv_path}")
            st.stop()
        try:
            xyz_path = _ensure_xyz_from_gem_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"CSV → XYZ conversion failed: {exc}")
            st.stop()
        st.caption(f"Using converted: `{xyz_path.name}` (next to the CSV)")
    else:
        uploaded = st.file_uploader(
            "Upload GEM CSV or XYZ",
            type=["csv", "xyz"],
            help="CSV is converted to XYZ in a temp folder; XYZ is read directly.",
        )
        if uploaded is None:
            st.info("Choose a file to load.")
            st.stop()
        tmp_root = Path(tempfile.gettempdir()) / "gem2_simpeg_app_uploads"
        tmp_root.mkdir(parents=True, exist_ok=True)
        safe = Path(uploaded.name).name
        dest = tmp_root / safe
        dest.write_bytes(uploaded.getbuffer())
        if dest.suffix.lower() == ".csv":
            try:
                xyz_path = _ensure_xyz_from_gem_csv(dest)
            except Exception as exc:  # noqa: BLE001
                st.error(f"CSV → XYZ conversion failed: {exc}")
                st.stop()
        else:
            xyz_path = dest

    st.session_state["data_source_used"] = data_source
    st.session_state["xyz_path_resolved"] = str(xyz_path)


# === Load data (between the two sidebar blocks so the multiselect can show
#     the actual frequencies present in this file) ===

if not xyz_path.is_file():
    st.error(f"File not found: {xyz_path}")
    st.stop()

mtime = xyz_path.stat().st_mtime
df_raw = _cached_read_xyz(str(xyz_path), mtime)
prev_path = st.session_state.get("df_raw_path")
prev_mtime = st.session_state.get("df_raw_mtime")
if prev_path != str(xyz_path) or prev_mtime != mtime:
    st.session_state.df_stacked = None
    st.session_state.inv_df = None
    st.session_state.misfit_df = None
    st.session_state.layer_thicknesses = None
    st.session_state.config_used = None
st.session_state.df_raw = df_raw
st.session_state.df_raw_path = str(xyz_path)
st.session_state.df_raw_mtime = mtime

with st.expander("Loaded dataset (quick check)", expanded=False):
    st.caption(f"XYZ file: `{xyz_path}`")
    st.metric("Soundings (rows)", len(df_raw))
    if "X" in df_raw.columns and "Y" in df_raw.columns:
        xy = pd.to_numeric(df_raw["X"], errors="coerce"), pd.to_numeric(df_raw["Y"], errors="coerce")
        n_ok = int(np.sum(np.isfinite(xy[0]) & np.isfinite(xy[1])))
        st.metric("Rows with finite X/Y", n_ok)
    try:
        _fq, _ch = get_freq_channel_columns(df_raw, use_channels="iq")
        st.write("Detected frequencies (Hz):", ", ".join(f"{float(f):.0f}" for f in _fq))
    except ValueError as _e:
        st.warning(str(_e))


with st.sidebar:
    st.header("2. Pre-processing")
    stack_out = st.number_input("Stack output spacing (m)", min_value=0.1, value=1.0, step=0.5)
    stack_window = st.number_input("Stack window (m)", min_value=0.1, value=2.0, step=0.5)
    min_stack = st.number_input("Min raw samples per bin", min_value=1, value=2, step=1)
    drop_neg_q = st.checkbox(
        "Drop negative-Q channels per sounding", value=True,
        help="Negative Q is unphysical for a sigma-only forward model; usually a "
             "magnetic-susceptibility signature. Drops that channel for the affected "
             "sounding only.",
    )

    st.header("3. Channels")
    use_channels = st.selectbox(
        "Components to invert",
        options=["q", "iq", "i"],
        index=0,
        help="'q' = quadrature only (recommended on magnetic clays); 'iq' = both.",
    )
    try:
        frequencies_full, channel_cols_full = get_freq_channel_columns(
            df_raw, use_channels=use_channels
        )
    except ValueError as _e:
        st.error(str(_e))
        st.stop()
    exclude_freqs = st.multiselect(
        "Exclude frequencies (Hz)",
        options=[float(f) for f in frequencies_full],
        default=[],
        format_func=lambda v: f"{v:.0f} Hz",
        help=(
            "Drop these frequencies entirely from inversion. Use this when a "
            "channel is dominated by magnetic-susceptibility (typical: very "
            "large RMSE on the lowest GEM-2 frequency in the per-channel "
            "residual diagnostics)."
        ),
    )
    _exclude_set = {float(v) for v in exclude_freqs}
    keep_mask = np.array(
        [float(f) not in _exclude_set for f in frequencies_full], dtype=bool
    )
    if not keep_mask.any():
        st.error("All frequencies excluded - leave at least one.")
        st.stop()
    frequencies_kept = frequencies_full[keep_mask]
    n_per_freq = 2 if use_channels == "iq" else 1
    channel_cols_kept: List[str] = []
    for _i, _k in enumerate(keep_mask):
        if _k:
            channel_cols_kept.extend(channel_cols_full[n_per_freq * _i : n_per_freq * (_i + 1)])
    if exclude_freqs:
        st.caption(
            f"-> {len(frequencies_kept)}/{len(frequencies_full)} freqs kept: "
            + ", ".join(f"{f:.0f}" for f in frequencies_kept)
            + f" Hz ({len(channel_cols_kept)} channels)"
        )
    # If use_channels changed (e.g. Q -> IQ added I channels) the stacked frame
    # is missing the new columns; clear it. Exclusion changes do NOT invalidate
    # the stacked frame because we always stack the full channel set anyway.
    _stacked = st.session_state.get("df_stacked")
    if _stacked is not None:
        _missing = [c for c in channel_cols_full if c not in _stacked.columns]
        if _missing:
            st.session_state.df_stacked = None
    st.session_state.frequencies_full = frequencies_full
    st.session_state.channel_cols_full = channel_cols_full
    st.session_state.frequencies_kept = frequencies_kept
    st.session_state.channel_cols_kept = channel_cols_kept

    st.header("4. Noise model")
    rel_err_pct = st.number_input("Relative error (%)", min_value=0.1, value=5.0, step=1.0)
    floor_ppm = st.number_input("Noise floor (ppm)", min_value=0.0, value=10.0, step=1.0)
    use_emp_std = st.checkbox(
        "Use within-window std as per-channel noise", value=True,
        help="Adds the empirical 1.253 * std/sqrt(n) as a lower bound on each "
             "channel's uncertainty, on top of the relative+floor model.",
    )

    st.header("5. Forward / mesh")
    coil_spacing = st.number_input("Coil spacing (m)", min_value=0.1, value=1.66, step=0.01)
    default_height = st.number_input(
        "Default sensor height AGL (m)",
        min_value=0.0, value=0.10, step=0.05,
        help="Used when SensorHeight(m) is missing or invalid in the XYZ.",
    )
    n_layers = st.slider(
        "Number of layers (incl. halfspace)", min_value=3, max_value=20, value=6,
    )
    max_depth = st.number_input(
        "Max model depth (m)",
        min_value=0.5, value=5.0, step=0.5,
        help="Cumulative thickness of the finite layers. The halfspace begins below this.",
    )
    first_thickness = st.number_input(
        "First (top) layer thickness (m)", min_value=0.05, value=0.4, step=0.05,
        help="The growth ratio between adjacent layers is then solved so the finite "
             "layers sum exactly to Max model depth.",
    )
    try:
        _preview_thicks = build_layer_thicknesses(
            float(max_depth), float(first_thickness), int(n_layers)
        )
        _actual_growth = (
            float(_preview_thicks[-1] / _preview_thicks[-2])
            if len(_preview_thicks) > 1 else 1.0
        )
        st.caption(
            f"-> {len(_preview_thicks) + 1} cells "
            f"({len(_preview_thicks)} finite + halfspace), "
            f"growth ratio {_actual_growth:.3f}, "
            f"top {_preview_thicks[0]:.2f} m -> last finite {_preview_thicks[-1]:.2f} m"
        )
    except ValueError as _e:
        st.warning(f"Layering invalid: {_e}")

    st.header("6. Regularisation / optimiser")
    inversion_mode = st.radio(
        "Inversion mode",
        options=["Per-sounding 1D", "LCI (laterally constrained)"],
        index=0,
        help=(
            "Per-sounding: each sounding is inverted independently (fast, but "
            "noisy when only 1-2 channels survive negative-Q dropping).\n\n"
            "LCI: all soundings on a line are inverted simultaneously with "
            "lateral coupling between same-index layers of neighbouring "
            "soundings (Auken & Christiansen 2004). Information from data-rich "
            "soundings spreads to data-poor neighbours via the lateral prior."
        ),
    )
    alpha_x = st.number_input(
        "alpha (vertical smoothness within sounding)",
        min_value=0.0, value=3.0, step=0.5,
        help="Penalises log-sigma roughness in depth (used in both modes).",
    )
    if inversion_mode.startswith("LCI"):
        lateral_constraint = st.number_input(
            "Lateral constraint factor c",
            min_value=1.001, value=1.2, step=0.05, format="%.3f",
            help=(
                "Auken & Christiansen 2004 lateral constraint: c=1.2 means "
                "neighbouring soundings may differ by ~20%% per layer. "
                "c=1.05 = very tight, 1.5 = loose. Translated to SimPEG "
                "alpha_x = 1 / log(c)^2."
            ),
        )
        try:
            alpha_x_lat_eff = alpha_x_from_constraint_factor(float(lateral_constraint))
            st.caption(f"-> SimPEG alpha_x = {alpha_x_lat_eff:.2f}")
        except ValueError:
            alpha_x_lat_eff = 30.0
    else:
        lateral_constraint = 1.2
        alpha_x_lat_eff = alpha_x_from_constraint_factor(lateral_constraint)
    alpha_s = st.number_input("alpha_s (smallness)", min_value=0.0, value=1e-4, format="%.0e")
    sigma_min = st.number_input("sigma_min (S/m)", min_value=1e-6, value=1e-4, format="%.0e")
    sigma_max = st.number_input("sigma_max (S/m)", min_value=0.01, value=5.0)
    max_iter = st.number_input("Max iterations", min_value=1, value=30, step=1)
    beta0 = st.number_input("beta0 ratio", min_value=0.1, value=5.0, step=0.5)
    cool_factor = st.number_input("Beta cooling factor", min_value=1.01, value=1.5, step=0.1)
    cool_rate = st.number_input("Beta cooling rate (iters)", min_value=1, value=2, step=1)
    chifact = st.number_input("Target chi-fact", min_value=0.01, value=1.0, step=0.1)

    st.header("7. Scope")
    only_line_str = st.text_input(
        "Only this Line ID (blank = all lines)", value="",
        help="Restrict to one survey line; useful for quick iteration.",
    )
    limit = st.number_input(
        "Limit number of stacked soundings (0 = all)", min_value=0, value=0, step=10
    )

    st.header("8. Parallelism")
    cpu_total = os.cpu_count() or 1
    if inversion_mode.startswith("LCI"):
        worker_help = (
            "Number of survey lines inverted in parallel. Each line is one big "
            "LCI inversion; effective concurrency is min(workers, n_lines). "
            "Single-line datasets gain nothing."
        )
    else:
        worker_help = (
            "Number of stacked soundings inverted in parallel. Each worker is a "
            "fresh Python process that imports SimPEG once at startup, so the "
            "speedup only pays off above ~50 soundings."
        )
    n_workers = st.slider(
        "Workers",
        min_value=1,
        max_value=max(1, cpu_total),
        value=1,
        help=worker_help,
    )

    st.divider()
    run_button = st.button("Run inversion", type="primary", use_container_width=True)


# === Main: tabs ===

tab_data, tab_pre, tab_inv = st.tabs(["1. Data", "2. Pre-processing", "3. Inversion results"])

with tab_data:
    st.subheader(f"{xyz_path.name}")
    cols = st.columns(4)
    cols[0].metric("Raw rows", f"{len(df_raw):,}")
    cols[1].metric(
        "Frequencies (kept / file)",
        f"{len(frequencies_kept)} / {len(frequencies_full)}",
    )
    cols[2].metric("Components used", use_channels.upper())
    cols[3].metric("Channels in inversion", f"{len(channel_cols_kept)}")

    st.write(f"Frequencies in file (Hz): {frequencies_full.tolist()}")
    if exclude_freqs:
        st.write(
            f"Frequencies kept for inversion (Hz): {frequencies_kept.tolist()} "
            f"(excluded: {sorted(int(f) for f in exclude_freqs)})"
        )

    if {"X", "Y", "Sample"}.issubset(df_raw.columns):
        st.pyplot(_plot_map(df_raw, color_col="Sample", title="Raw sample coverage"))

    with st.expander("Show first 200 raw rows"):
        st.dataframe(df_raw.head(200), use_container_width=True)


# === Pre-processing diagnostics ===

with tab_pre:
    st.subheader("Channel distributions (raw, before stacking)")
    st.caption(
        "Negative-Q channels (red shading) are usually a magnetic-susceptibility "
        "signature; with `Drop negative-Q` on, they are masked out per sounding."
    )
    st.pyplot(_plot_channel_distribution(df_raw, channel_cols_full))

    st.subheader("Stack the data")
    if st.button("Run rolling-median stacking", use_container_width=False):
        if only_line_str.strip():
            try:
                only_line = int(only_line_str)
            except ValueError:
                st.error("Line ID must be an integer")
                st.stop()
            df_in = df_raw[df_raw["Line"] == only_line]
            if df_in.empty:
                st.error(f"Line {only_line} not in raw data")
                st.stop()
        else:
            df_in = df_raw

        required = ["Line", "Sample", "X", "Y"] + list(channel_cols_full)
        for c in required:
            df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
        df_in = df_in.dropna(subset=required)

        extras = [c for c in ("GPSalt(m)", "DEM_m", "SensorHeight(m)", "sensor_height_m")
                  if c in df_in.columns]
        parts = []
        for line_id, line_df in df_in.sort_values(["Line", "Sample"]).groupby("Line", sort=True):
            parts.append(
                stack_line_rolling_median(
                    line_df,
                    measurement_cols=channel_cols_full,
                    out_spacing_m=stack_out,
                    window_m=stack_window,
                    extra_median_cols=extras,
                    min_samples_per_bin=min_stack,
                )
            )
        parts = [p for p in parts if len(p)]
        if not parts:
            st.error("Stacking produced no soundings (window too small? Track too short?).")
            st.stop()
        stacked = pd.concat(parts, ignore_index=True)
        if "SensorHeight(m)" in stacked.columns and "sensor_height_m" not in stacked.columns:
            stacked["sensor_height_m"] = stacked["SensorHeight(m)"]
        elif "sensor_height_m" not in stacked.columns:
            stacked["sensor_height_m"] = default_height
        st.session_state.df_stacked = stacked
        st.success(
            f"Stacked to {len(stacked)} soundings "
            f"(median {int(np.median(stacked['n_stack']))} raw samples per bin)"
        )

    df_stacked = st.session_state.df_stacked
    if df_stacked is not None:
        cols = st.columns(4)
        cols[0].metric("Stacked soundings", f"{len(df_stacked):,}")
        cols[1].metric("Median n in bin", int(np.median(df_stacked["n_stack"])))
        cols[2].metric("Lines", df_stacked["Line"].nunique())
        cols[3].metric(
            "Median sensor height (m)", f"{float(np.nanmedian(df_stacked['sensor_height_m'])):.2f}"
        )

        st.subheader("Stacked map")
        color_options = [c for c in channel_cols_full if c in df_stacked.columns]
        if color_options:
            color_col = st.selectbox("Map colour by", options=color_options, index=len(color_options) - 1)
            st.pyplot(_plot_map(df_stacked, color_col=color_col, title=f"Stacked: {color_col}"))

        st.subheader("Negative-Q rate per channel (stacked)")
        # Only consider Q channels that actually exist in the current stacked
        # frame; the frame may have been stacked before the user switched to a
        # file with a different frequency set.
        q_cols = [c for c in channel_cols_full
                  if c.startswith("Q_") and c in df_stacked.columns]
        missing = [c for c in channel_cols_full
                   if c.startswith("Q_") and c not in df_stacked.columns]
        if missing:
            st.warning(
                "Stacked data is from an older file or use-channels setting and "
                f"is missing {missing}. Re-run rolling-median stacking to refresh."
            )
        if q_cols:
            stats_rows = []
            for c in q_cols:
                v = pd.to_numeric(df_stacked[c], errors="coerce").to_numpy(dtype=float)
                fin = v[np.isfinite(v)]
                stats_rows.append({
                    "channel": c,
                    "n_finite": int(fin.size),
                    "n_negative": int((fin < 0).sum()),
                    "pct_negative": (
                        100.0 * float((fin < 0).sum()) / float(fin.size) if fin.size else 0.0
                    ),
                    "median": float(np.median(fin)) if fin.size else np.nan,
                })
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

        with st.expander("Show first 100 stacked soundings"):
            display_cols = ["Line", "Sample", "X", "Y", "stack_distance_m", "n_stack",
                            "sensor_height_m"] + channel_cols_full
            display_cols = [c for c in display_cols if c in df_stacked.columns]
            st.dataframe(df_stacked[display_cols].head(100), use_container_width=True)

    st.subheader("Line profiles: all channels (raw vs stacked)")
    st.caption(
        "Single plot per line: each channel has its own colour — raw samples (light), "
        "rolling median (dashed), stacked median with **SD as vertical error bars**. "
        "Use symmetric log or percentile clipping if one channel spikes and flattens the rest."
    )
    if "Line" in df_raw.columns and df_raw["Line"].notna().any():
        line_opts = sorted(pd.to_numeric(df_raw["Line"], errors="coerce").dropna().astype(int).unique().tolist())
        plot_lines = st.multiselect(
            "Lines to plot",
            options=line_opts,
            default=line_opts[:1] if line_opts else [],
        )
        cands = [c for c in channel_cols_full if c in df_raw.columns]
        plot_chans = st.multiselect(
            "Channels (default = all)",
            options=cands,
            default=cands,
        )
        raw_roll = st.slider(
            "Raw rolling window (samples, for dashed median)",
            min_value=3,
            max_value=51,
            value=7,
            step=2,
        )
        y_axis_mode = st.selectbox(
            "Y-axis (handle outliers)",
            options=[
                "Symmetric log",
                "Linear",
                "Linear + clip to percentiles",
            ],
            index=0,
            help=(
                "Symmetric log compresses large |ppm| so low-amplitude channels stay visible. "
                "Percentile clip sets y-limits from the chosen spread of finite ppm values."
            ),
        )
        mode_key = {
            "Symmetric log": "symlog",
            "Linear": "linear",
            "Linear + clip to percentiles": "percentile",
        }[y_axis_mode]
        pct_lo, pct_hi = 0.5, 99.5
        if mode_key == "percentile":
            pct_col1, pct_col2 = st.columns(2)
            with pct_col1:
                pct_lo = st.slider("Percentile low (clip)", 0.0, 25.0, 0.5, 0.5)
            with pct_col2:
                pct_hi = st.slider("Percentile high (clip)", 75.0, 100.0, 99.5, 0.5)

        df_stacked_plot = st.session_state.df_stacked
        if df_stacked_plot is None:
            st.info("Run rolling-median stacking above to show stacked curves with SD error bars.")
        for lid in plot_lines:
            if not plot_chans:
                st.warning("Select at least one channel.")
                break
            try:
                fig_prof = _plot_line_raw_vs_stacked(
                    int(lid),
                    df_raw,
                    df_stacked_plot,
                    plot_chans,
                    int(raw_roll),
                    y_scale=mode_key,
                    pct_low=float(pct_lo),
                    pct_high=float(pct_hi),
                )
                st.pyplot(fig_prof)
            except ValueError as e:
                st.error(str(e))
    else:
        st.caption("No usable `Line` column for per-line profiles.")


# === Inversion run ===

if run_button:
    df_stacked = st.session_state.df_stacked
    if df_stacked is None:
        # Auto-stack if not done.
        with st.status("Stacking before inversion...", expanded=False):
            df_in = df_raw.copy()
            if only_line_str.strip():
                try:
                    only_line = int(only_line_str)
                except ValueError:
                    st.error("Line ID must be an integer")
                    st.stop()
                df_in = df_in[df_in["Line"] == only_line]
            required = ["Line", "Sample", "X", "Y"] + list(channel_cols_full)
            for c in required:
                df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
            df_in = df_in.dropna(subset=required)
            extras = [c for c in ("GPSalt(m)", "DEM_m", "SensorHeight(m)", "sensor_height_m")
                      if c in df_in.columns]
            parts = []
            for _, line_df in df_in.sort_values(["Line", "Sample"]).groupby("Line", sort=True):
                parts.append(
                    stack_line_rolling_median(
                        line_df,
                        measurement_cols=channel_cols_full,
                        out_spacing_m=stack_out,
                        window_m=stack_window,
                        extra_median_cols=extras,
                        min_samples_per_bin=min_stack,
                    )
                )
            parts = [p for p in parts if len(p)]
            if not parts:
                st.error("Stacking produced no soundings.")
                st.stop()
            df_stacked = pd.concat(parts, ignore_index=True)
            if "SensorHeight(m)" in df_stacked.columns and "sensor_height_m" not in df_stacked.columns:
                df_stacked["sensor_height_m"] = df_stacked["SensorHeight(m)"]
            elif "sensor_height_m" not in df_stacked.columns:
                df_stacked["sensor_height_m"] = default_height
            st.session_state.df_stacked = df_stacked

    sub = df_stacked
    if only_line_str.strip():
        try:
            sub = sub[sub["Line"] == int(only_line_str)]
        except ValueError:
            st.error("Line ID must be an integer.")
            st.stop()
    if limit and limit > 0:
        sub = sub.head(int(limit))
    if sub.empty:
        st.error("No soundings to invert after applying scope filters.")
        st.stop()

    layer_thicknesses = build_layer_thicknesses(
        float(max_depth), float(first_thickness), int(n_layers)
    )
    st.session_state.layer_thicknesses = layer_thicknesses

    is_lci = inversion_mode.startswith("LCI")
    config = {
        "coil_spacing": float(coil_spacing),
        "default_sensor_height": float(default_height),
        "relative_error": float(rel_err_pct) / 100.0,
        "noise_floor_ppm": float(floor_ppm),
        "alpha_s": float(alpha_s),
        # In per-sounding 1D this is the depth-smoothness; in LCI mode it gets
        # passed through as alpha_y_vertical.
        "alpha_x": float(alpha_x),
        "alpha_y_vertical": float(alpha_x),
        "alpha_x_lateral": float(alpha_x_lat_eff),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "max_iterations": int(max_iter),
        "beta0_ratio": float(beta0),
        "beta_cooling_factor": float(cool_factor),
        "beta_cooling_rate": int(cool_rate),
        "chifact": float(chifact),
        "use_channels": use_channels,
        "noise_from_std": bool(use_emp_std),
        "drop_negative": bool(drop_neg_q),
        "inversion_mode": "LCI" if is_lci else "per_sounding",
        "lateral_constraint_factor": float(lateral_constraint),
    }
    st.session_state.config_used = config

    records = sub.to_dict("records")
    for i, r in enumerate(records):
        r["obs_id"] = i
    n_total = len(records)

    inv_out: List[dict] = []
    t0 = time.time()

    def _lci_out_to_per_sounding(out_dict, recs):
        """Translate one line's LCI output to the per-sounding dict schema."""
        if not out_dict.get("ok"):
            return []
        recs_by_local = {i: recs[i] for i in range(len(recs))}
        rows_out = []
        for r in out_dict["results"]:
            global_obs_id = int(recs_by_local[int(r["obs_id"])]["obs_id"])
            rows_out.append({
                "ok": True,
                "obs_id": global_obs_id,
                "line": int(r["Line"]),
                "sample": int(r["Sample"]),
                "x": float(r["X"]),
                "y": float(r["Y"]),
                "z_gps": float(r.get("GPSalt_m", float("nan"))),
                "dem_m": float(r.get("DEM_m", float("nan"))),
                "sensor_height_used_m": float(r["sensor_height_m"]),
                "sigma_warm": float(r.get("sigma_warm", float("nan"))),
                "chi2": float(r["chi2"]) if r["active"] else float("nan"),
                "active": bool(r["active"]),
                "n_channels_used": int(r["n_channels_used"]),
                "n_channels_total": int(r["n_channels_total"]),
                "sigma_layers": list(r["sigma_layers"]),
                "dobs": list(r["dobs"]) if r["dobs"] else [float("nan")] * len(channel_cols_kept),
                "dpred": list(r["dpred"]) if r["dpred"] else [float("nan")] * len(channel_cols_kept),
                "residual": (
                    list(r["residual"]) if r["residual"] else [float("nan")] * len(channel_cols_kept)
                ),
                "doi_m": float(r.get("doi_m", np.nan)),
                "doi_threshold": float(r.get("doi_threshold", np.nan)),
            })
        return rows_out

    if is_lci:
        line_ids_run = sorted({int(r["Line"]) for r in records})
        line_to_records = {li: [r for r in records if int(r["Line"]) == li] for li in line_ids_run}
        line_dfs = {li: pd.DataFrame(recs) for li, recs in line_to_records.items()}
        n_lines = len(line_ids_run)
        eff_workers = max(1, min(int(n_workers), n_lines))
        progress = st.progress(
            0.0,
            text=f"LCI: 0 / {n_lines} lines (workers={eff_workers}) ...",
        )

        def _update_lci_progress(done, line_id, out_dict):
            elapsed = time.time() - t0
            tag = f"chi^2 {out_dict['chi2_global']:.2f}" if out_dict.get("ok") else "FAILED"
            progress.progress(
                done / max(1, n_lines),
                text=(
                    f"LCI: {done} / {n_lines} lines (workers={eff_workers}) | "
                    f"last line {line_id}: {tag} | elapsed {int(elapsed)}s"
                ),
            )

        if eff_workers <= 1:
            for done, line_id in enumerate(line_ids_run, start=1):
                out = invert_one_line_lci(
                    line_dfs[line_id], channel_cols_kept, frequencies_kept,
                    layer_thicknesses, config,
                )
                if out.get("ok"):
                    inv_out.extend(_lci_out_to_per_sounding(out, line_to_records[line_id]))
                else:
                    st.error(f"LCI failed on line {line_id}: {out.get('error')}")
                _update_lci_progress(done, line_id, out)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=eff_workers) as ex:
                fut_to_lid = {
                    ex.submit(
                        invert_one_line_lci,
                        line_dfs[lid], channel_cols_kept, frequencies_kept,
                        layer_thicknesses, config,
                    ): lid
                    for lid in line_ids_run
                }
                done = 0
                for fut in concurrent.futures.as_completed(fut_to_lid):
                    lid = fut_to_lid[fut]
                    done += 1
                    try:
                        out = fut.result()
                    except Exception as e:
                        out = {"ok": False, "error": str(e), "results": []}
                    if out.get("ok"):
                        inv_out.extend(_lci_out_to_per_sounding(out, line_to_records[lid]))
                    else:
                        st.error(f"LCI failed on line {lid}: {out.get('error')}")
                    _update_lci_progress(done, lid, out)
        progress.empty()
    else:
        eff_workers = max(1, min(int(n_workers), n_total))
        progress = st.progress(
            0.0, text=f"Inverting 0 / {n_total} (workers={eff_workers}) ...",
        )

        def _update_sounding_progress(done):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (n_total - done) / rate if rate > 0 else 0.0
            progress.progress(
                done / max(1, n_total),
                text=(
                    f"Inverting {done} / {n_total} (workers={eff_workers}) | "
                    f"elapsed {int(elapsed)}s | ETA {int(eta)}s"
                ),
            )

        if eff_workers <= 1:
            for i, rec in enumerate(records, start=1):
                inv_out.append(
                    invert_one_sounding(
                        rec, channel_cols_kept, frequencies_kept, layer_thicknesses, config
                    )
                )
                _update_sounding_progress(i)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=eff_workers) as ex:
                futs = [
                    ex.submit(
                        invert_one_sounding,
                        rec, channel_cols_kept, frequencies_kept,
                        layer_thicknesses, config,
                    )
                    for rec in records
                ]
                done = 0
                for fut in concurrent.futures.as_completed(futs):
                    done += 1
                    try:
                        inv_out.append(fut.result())
                    except Exception as e:
                        inv_out.append({"ok": False, "error": str(e)})
                    _update_sounding_progress(done)
            inv_out.sort(key=lambda d: d.get("obs_id", -1))
        progress.empty()

    n_ok = sum(1 for r in inv_out if r.get("ok"))
    n_fail = n_total - n_ok
    n_cells = len(layer_thicknesses) + 1
    layer_top = np.r_[0.0, np.cumsum(layer_thicknesses)]

    rows, mrows = [], []
    for r in inv_out:
        if not r.get("ok"):
            continue
        row = {
            "obs_id": r["obs_id"],
            "Line": r["line"],
            "Sample": r["sample"],
            "X": r["x"],
            "Y": r["y"],
            "GPSalt_m": r["z_gps"],
            "DEM_m": r["dem_m"],
            "sensor_height_used_m": r["sensor_height_used_m"],
            "sigma_warm_S_per_m": r["sigma_warm"],
            "chi2": r["chi2"],
            "active": bool(r.get("active", True)),
            "n_channels_used": r["n_channels_used"],
            "n_channels_total": r["n_channels_total"],
            "doi_m": float(r.get("doi_m", np.nan)),
            "doi_threshold": float(r.get("doi_threshold", np.nan)),
        }
        for j, sigma in enumerate(r["sigma_layers"]):
            row[f"sigma_layer_{j + 1}"] = sigma
        for j in range(n_cells):
            row[f"depth_top_layer_{j + 1}_m"] = layer_top[j]
            row[f"depth_bottom_layer_{j + 1}_m"] = (
                layer_top[j + 1] if j + 1 < len(layer_top) else float("nan")
            )
        rows.append(row)

        m = {
            "obs_id": r["obs_id"],
            "Line": r["line"],
            "Sample": r["sample"],
            "chi2": r["chi2"],
            "n_channels_used": r["n_channels_used"],
            "n_channels_total": r["n_channels_total"],
        }
        for k, ch in enumerate(channel_cols_kept):
            m[f"dobs_{ch}"] = r["dobs"][k]
            m[f"dpred_{ch}"] = r["dpred"][k]
            m[f"res_{ch}"] = r["residual"][k]
        mrows.append(m)

    inv_df = pd.DataFrame(rows) if rows else None
    misfit_df = pd.DataFrame(mrows) if mrows else None
    st.session_state.inv_df = inv_df
    st.session_state.misfit_df = misfit_df

    summary_cols = st.columns(4)
    summary_cols[0].metric("Inverted", f"{n_ok} / {n_total}")
    summary_cols[1].metric("Failed", f"{n_fail}")
    if inv_df is not None and len(inv_df):
        summary_cols[2].metric("Median chi^2", f"{float(inv_df['chi2'].median()):.2f}")
        summary_cols[3].metric(
            "Median channels used",
            f"{int(inv_df['n_channels_used'].median())} / {int(inv_df['n_channels_total'].median())}",
        )

    if n_fail:
        with st.expander(f"{n_fail} failed soundings"):
            for r in inv_out:
                if not r.get("ok"):
                    st.write(f"Line {r.get('line')} sample {r.get('sample')}: {r.get('error')}")


# === Inversion results ===

with tab_inv:
    inv_df = st.session_state.inv_df
    misfit_df = st.session_state.misfit_df
    layer_thicknesses = st.session_state.layer_thicknesses
    cfg_used = st.session_state.config_used
    if inv_df is None or layer_thicknesses is None:
        st.info("Configure parameters in the sidebar and click **Run inversion**.")
    else:
        if cfg_used and cfg_used.get("inversion_mode") == "LCI":
            st.caption(
                f"Mode: **LCI** | lateral constraint c = {cfg_used['lateral_constraint_factor']:.3f} "
                f"(alpha_x = {cfg_used['alpha_x_lateral']:.2f}) | "
                f"alpha_y (vertical) = {cfg_used['alpha_y_vertical']:.2f}"
            )
        else:
            st.caption("Mode: **Per-sounding 1D** (independent inversions)")
        cols = st.columns(4)
        active_mask = inv_df.get("active", pd.Series(True, index=inv_df.index)).astype(bool)
        cols[0].metric(
            "Soundings (active / total)",
            f"{int(active_mask.sum())} / {len(inv_df):,}",
        )
        chi2_active = inv_df.loc[active_mask, "chi2"]
        cols[1].metric(
            "chi^2 median",
            f"{float(chi2_active.median()):.2f}" if len(chi2_active) else "n/a",
        )
        cols[2].metric(
            "chi^2 p90",
            f"{float(chi2_active.quantile(0.9)):.2f}" if len(chi2_active) else "n/a",
        )
        cols[3].metric("Lines in result", inv_df["Line"].nunique())

        st.subheader("Section per line")
        line_options = sorted(inv_df["Line"].unique().tolist())
        line_sel = st.selectbox("Line", options=line_options)
        sec_cols = st.columns([1, 1, 1, 2])
        plot_max_depth = sec_cols[0].number_input(
            "Max depth (m)",
            min_value=1.0,
            value=min(15.0, float(np.sum(layer_thicknesses)) * 1.2),
        )
        color_scale = sec_cols[1].selectbox("Colour scale", options=["log", "linear"], index=0)
        clip_lo, clip_hi = sec_cols[2].slider(
            "Clip percentile",
            min_value=0.0, max_value=100.0,
            value=(5.0, 95.0), step=1.0,
            help="Auto-vmin / vmax come from these percentiles of the recovered "
                 "sigma. Widen the window (e.g. 1-99) to see whether the strong "
                 "tongues are still bright after avoiding clip saturation, "
                 "narrow it (e.g. 25-75) to focus on the bulk model.",
        )
        manual_clip = sec_cols[3].checkbox(
            "Manual vmin / vmax (overrides percentiles)", value=False,
        )
        if manual_clip:
            mc1, mc2 = sec_cols[3].columns(2)
            man_vmin = mc1.number_input("vmin (S/m)", min_value=1e-6, value=0.05, format="%.3g")
            man_vmax = mc2.number_input("vmax (S/m)", min_value=1e-5, value=1.0, format="%.3g")
        else:
            man_vmin = None
            man_vmax = None
        try:
            fig = _plot_line_section(
                inv_df, int(line_sel), layer_thicknesses, plot_max_depth,
                color_scale=color_scale,
                vmin=man_vmin, vmax=man_vmax,
                clip_pct=(float(clip_lo), float(clip_hi)),
                show_channels_strip=True,
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Section plot failed: {e}")

        st.subheader("Map of soundings (colour = recovered shallow sigma)")
        sigma_cols = [c for c in inv_df.columns if c.startswith("sigma_layer_")]
        if sigma_cols:
            shallow = inv_df[sigma_cols[0]]
            inv_df_map = inv_df.assign(sigma_layer1=shallow)
            st.pyplot(
                _plot_map(
                    inv_df_map, color_col="sigma_layer1", log_color=True,
                    title="Shallow sigma (S/m, log)"
                )
            )
        st.pyplot(_plot_map(inv_df, color_col="chi2", log_color=False, title="Per-sounding chi^2"))

        if "doi_m" in inv_df.columns and inv_df["doi_m"].notna().any():
            st.subheader("Map of DOI depth")
            st.caption("Per-sounding Jacobian-based DOI (m below surface); matches the lime line on the section plot.")
            st.pyplot(
                _plot_map(
                    inv_df, color_col="doi_m", log_color=False, title="DOI depth (m)"
                )
            )

        st.subheader("Single sounding profile")
        line_filt = st.selectbox("Line ", options=line_options, key="profile_line")
        sub = inv_df[inv_df["Line"] == int(line_filt)].sort_values("Sample")
        sample_options = sub["obs_id"].tolist()
        if sample_options:
            obs_idx = st.slider(
                "Sounding index along line",
                min_value=0,
                max_value=len(sample_options) - 1,
                value=0,
            )
            obs_id = int(sample_options[obs_idx])
            try:
                fig = _plot_sounding_profile(inv_df, obs_id, layer_thicknesses)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Profile plot failed: {e}")

        st.subheader("Per-channel residual diagnostics")
        if misfit_df is not None:
            st.pyplot(_plot_channel_misfit(misfit_df, channel_cols_kept))

        st.subheader("Downloads")
        st.caption(
            "Long XYZ files list one row per sounding × layer: **X_m**, **Y_m**, "
            "**depth_centre_m** (positive down), **sigma_S_m**, **resistivity_ohm_m**, "
            "**doi_m** / **doi_threshold** (repeated on each layer row), "
            "and **Z_elevation_m** (DEM − depth when DEM is available) for 3D display."
        )
        csv_bytes = inv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download inversion CSV",
            data=csv_bytes,
            file_name=f"inverted_{xyz_path.stem}.csv",
            mime="text/csv",
        )
        try:
            long_xyz = inversion_wide_to_long_xyz(
                inv_df,
                layer_thicknesses=(
                    np.asarray(layer_thicknesses, dtype=float)
                    if layer_thicknesses is not None
                    else None
                ),
            )
            buf = io.BytesIO()
            long_xyz.to_csv(buf, index=False, encoding="utf-8")
            st.download_button(
                "Download 3D point cloud (long CSV: X,Y,depth, sigma per layer)",
                data=buf.getvalue(),
                file_name=f"inverted_{xyz_path.stem}_xyz_long.csv",
                mime="text/csv",
            )
            buf_tab = io.BytesIO()
            long_xyz.to_csv(buf_tab, index=False, sep="\t", encoding="utf-8")
            st.download_button(
                "Download 3D point cloud (tab-separated .txt)",
                data=buf_tab.getvalue(),
                file_name=f"inverted_{xyz_path.stem}_xyz_long.txt",
                mime="text/plain",
            )
        except ValueError as e:
            st.caption(f"Long XYZ export unavailable: {e}")
        if "doi_m" in inv_df.columns:
            doi_cols = [c for c in ("obs_id", "Line", "Sample", "X", "Y", "doi_m", "doi_threshold") if c in inv_df.columns]
            if doi_cols:
                doi_only = inv_df[doi_cols].copy()
                st.download_button(
                    "Download DOI summary (one row per sounding)",
                    data=doi_only.to_csv(index=False).encode("utf-8"),
                    file_name=f"doi_{xyz_path.stem}.csv",
                    mime="text/csv",
                )
        if misfit_df is not None:
            mb = misfit_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download misfit CSV",
                data=mb,
                file_name=f"misfit_{xyz_path.stem}.csv",
                mime="text/csv",
            )

        with st.expander("Inversion result table"):
            st.dataframe(inv_df, use_container_width=True)
