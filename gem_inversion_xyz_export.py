#!/usr/bin/env python3
"""
Flatten GEM-2 SimPEG inversion tables (wide CSV: one row per sounding) into a
long-format XYZ point cloud: one row per (sounding × layer) with depth and
recovered electrical properties for 3D viewers (CloudCompare, ParaView, GIS).

Typical inputs are CSV files written by ``gem2_simpeg_invert.py`` or the
Streamlit app: columns ``sigma_layer_*``, ``depth_top_layer_*_m``,
``depth_bottom_layer_*_m``, plus ``X``, ``Y``, ``obs_id``, etc.

Example CLI::

    python gem_inversion_xyz_export.py inverted.csv -o model_points.csv

``-o`` defaults to ``<stem>_xyz_long.csv`` next to the input file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


_SIGMA_RE = re.compile(r"^sigma_layer_(\d+)$")
_META_EXCLUDE = re.compile(
    r"^(sigma_layer_\d+|depth_top_layer_\d+_m|depth_bottom_layer_\d+_m)$"
)


def _sigma_layer_indices(columns: List[str]) -> List[int]:
    idx: List[int] = []
    for c in columns:
        m = _SIGMA_RE.match(str(c))
        if m:
            idx.append(int(m.group(1)))
    return sorted(set(idx))


def _mesh_thickness_for_plot(layer_thicknesses: np.ndarray, n_layers: int) -> np.ndarray:
    """
    Match ``gem2_simpeg_app._plot_line_section``: duplicate the last finite
    thickness so the half-space cell gets a finite thickness for bin edges /
    centres.
    """
    lt = np.asarray(layer_thicknesses, dtype=float).ravel()
    if n_layers == lt.size:
        return lt
    if n_layers == lt.size + 1:
        return np.r_[lt, float(lt[-1])]
    total = float(np.sum(lt)) if lt.size else 1.0
    return np.full(n_layers, total / max(n_layers, 1))


def inversion_wide_to_long_xyz(
    inv_df: pd.DataFrame,
    *,
    layer_thicknesses: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Expand wide inversion rows to a long point cloud.

    Output columns (subset may be NaN if absent from input):

    * Geometry: ``X_m``, ``Y_m``, ``depth_top_m``, ``depth_bottom_m``,
      ``depth_centre_m`` (positive downward from mesh / flat surface),
      ``thickness_m``
    * Properties: ``sigma_S_m`` (conductivity), ``log10_sigma_S_m``,
      ``resistivity_ohm_m``
    * Optional sounding metadata repeated on every layer row (everything in the
      wide CSV except layer sigma / depth columns): ``chi2``, ``DEM_m``, …
    * When ``DEM_m`` is present: ``Z_elevation_m`` = DEM − depth_centre (Z up).
    * When present on the wide table: ``doi_m``, ``doi_threshold`` (same value on
      every layer row for that sounding).

    ``layer_thicknesses``: finite layer thicknesses from the inversion mesh.
    When given, centres and thicknesses match the section plot in
    ``gem2_simpeg_app``. When omitted, centres use CSV tops/bottoms and a
    simple fallback for the half-space cell.
    """
    if inv_df is None or inv_df.empty:
        return pd.DataFrame()

    df = inv_df.copy()
    layer_ids = _sigma_layer_indices(list(df.columns))
    if not layer_ids:
        raise ValueError(
            "No columns matching 'sigma_layer_<n>' found; not a GEM2 inversion table."
        )

    sound_meta_cols = [c for c in df.columns if not _META_EXCLUDE.match(str(c))]

    n_layers = len(layer_ids)
    plot_thick: Optional[np.ndarray] = None
    centres_mesh: Optional[np.ndarray] = None
    if layer_thicknesses is not None:
        plot_thick = _mesh_thickness_for_plot(
            np.asarray(layer_thicknesses, dtype=float), n_layers
        )
        edges = np.r_[0.0, np.cumsum(plot_thick)]
        centres_mesh = 0.5 * (edges[:-1] + edges[1:])

    rows_out: List[dict] = []

    for _, rec in df.iterrows():
        dem = float(rec["DEM_m"]) if "DEM_m" in df.columns and pd.notna(rec.get("DEM_m")) else np.nan

        for j in layer_ids:
            sig_col = f"sigma_layer_{j}"
            top_col = f"depth_top_layer_{j}_m"
            bot_col = f"depth_bottom_layer_{j}_m"
            if sig_col not in df.columns:
                continue

            sigma = pd.to_numeric(rec[sig_col], errors="coerce")
            d_top = pd.to_numeric(rec[top_col], errors="coerce") if top_col in df.columns else np.nan
            d_bot = pd.to_numeric(rec[bot_col], errors="coerce") if bot_col in df.columns else np.nan

            if centres_mesh is not None and 1 <= j <= len(centres_mesh):
                d_centre = float(centres_mesh[j - 1])
            elif np.isfinite(d_top) and np.isfinite(d_bot):
                d_centre = float(0.5 * (d_top + d_bot))
            elif np.isfinite(d_top):
                if layer_thicknesses is not None and len(layer_thicknesses):
                    pad = float(np.nanmedian(np.asarray(layer_thicknesses, dtype=float)))
                else:
                    pad = 1.0
                d_centre = float(d_top + 0.5 * max(pad, 0.1))
            else:
                d_centre = np.nan

            if plot_thick is not None and 1 <= j <= len(plot_thick):
                thk = float(plot_thick[j - 1])
            elif np.isfinite(d_top) and np.isfinite(d_bot):
                thk = float(d_bot - d_top)
            else:
                thk = np.nan

            sig_f = float(sigma) if pd.notna(sigma) else np.nan

            row: dict = {
                "obs_id": rec["obs_id"] if "obs_id" in df.columns else np.nan,
                "Line": rec["Line"] if "Line" in df.columns else np.nan,
                "Sample": rec["Sample"] if "Sample" in df.columns else np.nan,
                "layer_index": int(j),
                "X_m": pd.to_numeric(rec["X"], errors="coerce") if "X" in df.columns else np.nan,
                "Y_m": pd.to_numeric(rec["Y"], errors="coerce") if "Y" in df.columns else np.nan,
                "depth_top_m": d_top,
                "depth_bottom_m": d_bot,
                "depth_centre_m": d_centre,
                "thickness_m": thk,
                "sigma_S_m": sig_f,
                "log10_sigma_S_m": np.log10(max(sig_f, 1e-30))
                if np.isfinite(sig_f) and sig_f > 0
                else np.nan,
                "resistivity_ohm_m": (1.0 / sig_f) if np.isfinite(sig_f) and sig_f > 0 else np.nan,
            }

            if np.isfinite(dem) and np.isfinite(d_centre):
                row["Z_elevation_m"] = float(dem) - float(d_centre)
            else:
                row["Z_elevation_m"] = np.nan

            skip_dup = {
                "obs_id", "Line", "Sample", "X", "Y",
            }
            for c in sound_meta_cols:
                if c in skip_dup:
                    continue
                row[str(c)] = rec[c]

            rows_out.append(row)

    out = pd.DataFrame(rows_out)
    if out.empty:
        return out

    first = [
        "obs_id",
        "Line",
        "Sample",
        "layer_index",
        "X_m",
        "Y_m",
        "depth_top_m",
        "depth_bottom_m",
        "depth_centre_m",
        "thickness_m",
        "Z_elevation_m",
        "sigma_S_m",
        "log10_sigma_S_m",
        "resistivity_ohm_m",
        "doi_m",
        "doi_threshold",
    ]
    rest = [c for c in out.columns if c not in first]
    out = out[[c for c in first if c in out.columns] + rest]
    return out


def write_xyz_long_csv(
    long_df: pd.DataFrame,
    path: Path | str,
    *,
    sep: str = ",",
) -> None:
    """Write long-format table (UTF-8). Comma default for easy GIS import."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(path, index=False, sep=sep, encoding="utf-8")


def write_xyz_long_txt(long_df: pd.DataFrame, path: Path | str) -> None:
    """Tab-separated variant (classic XYZ / CloudCompare)."""
    write_xyz_long_csv(long_df, path, sep="\t")


def main() -> None:
    p = argparse.ArgumentParser(description="GEM-2 inversion wide CSV → long XYZ point cloud.")
    p.add_argument("input_csv", type=Path, help="Wide inversion CSV (per sounding).")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input stem>_xyz_long.csv).",
    )
    p.add_argument(
        "--txt",
        action="store_true",
        help="Also write tab-separated file with .txt extension.",
    )
    args = p.parse_args()

    inp = args.input_csv
    if not inp.is_file():
        raise SystemExit(f"Not found: {inp}")

    inv_df = pd.read_csv(inp)
    long_df = inversion_wide_to_long_xyz(inv_df, layer_thicknesses=None)

    out = args.output
    if out is None:
        out = inp.with_name(f"{inp.stem}_xyz_long.csv")

    write_xyz_long_csv(long_df, out)
    print(f"Wrote {len(long_df)} layer points to {out}")

    if args.txt:
        txt_path = out.with_suffix(".txt")
        write_xyz_long_txt(long_df, txt_path)
        print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
