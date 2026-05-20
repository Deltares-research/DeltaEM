#!/usr/bin/env python3
"""
GEM-2 1D layered inversion using SimPEG.

Reads a Hedwige-style GEM-2 XYZ file produced by gem_csv_to_xyz.py, stacks per
line by distance, and runs a per-sounding SimPEG `Simulation1DLayered`
inversion. Outputs a CSV of recovered conductivity layers, per-channel misfit,
and an optional inversion section plot.

Differences from the previous Hedwige_Polder/src/invert.py:
  * Uses each sounding's actual `SensorHeight(m)` (or `GPSalt - DEM` if a DEM is
    given) instead of a single hard-coded survey elevation.
  * Mesh has the same number of cells as `Simulation1DLayered` layers
    (n_thicknesses + 1), no extra dummy cell, so smoothness is not applied
    across the halfspace boundary in a misleading way.
  * Noise model passed via `Data(standard_deviation=...)` with a 5% relative
    error and 10 ppm absolute floor as defaults.
  * Apparent-conductivity warm start per sounding from the lowest-frequency Q
    channel, instead of a global 0.5 S/m starting model.
  * Smooth L2 regularization on log-conductivity with `alpha_x = 1.0` and a
    very small reference-model penalty, similar to AarhusINV's smooth model.
  * Directives in the recommended order:
    UpdateSensitivityWeights -> UpdatePreconditioner -> BetaEstimate_ByEig
    -> BetaSchedule -> TargetMisfit. Cooling factor 1.5, rate 2.
  * Optional DEM sampling is only imported (rasterio) if the user passes
    `--dem-path`, so the script runs without rasterio in lighter envs.

Example:
  python gem2_simpeg_invert.py input.xyz \
    --output-csv inverted.csv \
    --stack-distance 3.0 --n-workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import simpeg.electromagnetics.frequency_domain as fdem
from discretize import TensorMesh
from simpeg import (
    data_misfit,
    directives,
    inversion,
    inverse_problem,
    maps,
    optimization,
    regularization,
)
from simpeg.data import Data

from gem2_doi import christiansen_auken_doi, summarize_doi_for_csv
from simpeg.electromagnetics.frequency_domain.survey import Survey


# ---------------------------------------------------------------------------
# XYZ reading
# ---------------------------------------------------------------------------


def read_gem2_xyz(path: Path) -> pd.DataFrame:
    """
    Read a Hedwige-style XYZ produced by gem_csv_to_xyz.py.

    Multi-token column names like `/Econd_1525Hz (mS/m)` are joined back into a
    single column name; data rows are then parsed by whitespace into the
    correct number of fields.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    header_idx = None
    for i, line in enumerate(text):
        if line.lstrip().startswith("/"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No '/'-prefixed header row found in {path}")

    raw_tokens = text[header_idx].split()
    columns: List[str] = []
    for tok in raw_tokens:
        if tok.startswith("/"):
            columns.append(tok[1:])
        elif columns:
            columns[-1] = columns[-1] + " " + tok
        else:
            columns.append(tok)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=header_idx + 1,
        header=None,
        names=columns,
        engine="python",
        na_values=["*"],
    )
    return df


def get_freq_channel_columns(
    df: pd.DataFrame, use_channels: str = "iq"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Return (frequencies, ordered SimPEG receiver channels, list of dataframe columns).

    `use_channels` controls which receiver components are kept:
      * "iq" -> [I_f1, Q_f1, I_f2, Q_f2, ...] (real + imag; matches the default
        `PointMagneticFieldSecondary` real+imag pair).
      * "q"  -> [Q_f1, Q_f2, ...] (imaginary only). Standard practice for shallow
        GEM-2 conductivity inversion when soils have any magnetic susceptibility,
        because the in-phase channels are then dominated by `mu_r` and not `sigma`.
      * "i"  -> [I_f1, I_f2, ...] (real only; mostly diagnostic).

    Returns: `(frequencies, channel_cols)`. With "iq" each frequency has two
    consecutive entries in channel_cols (I then Q); with "i" or "q" only one.
    """
    if use_channels not in ("iq", "i", "q"):
        raise ValueError(f"use_channels must be one of 'iq','i','q' (got '{use_channels}')")
    freq_to_i: Dict[float, str] = {}
    freq_to_q: Dict[float, str] = {}
    for col in df.columns:
        name = col.strip()
        m = re.match(r"I_(\d+(?:\.\d+)?)Hz\(ppm\)", name)
        if m:
            freq_to_i[float(m.group(1))] = col
            continue
        m = re.match(r"Q_(\d+(?:\.\d+)?)Hz\(ppm\)", name)
        if m:
            freq_to_q[float(m.group(1))] = col
    if use_channels == "iq":
        freqs = sorted(set(freq_to_i).intersection(freq_to_q))
    elif use_channels == "i":
        freqs = sorted(freq_to_i)
    else:
        freqs = sorted(freq_to_q)
    if not freqs:
        raise ValueError(f"No I/Q channels found for use_channels='{use_channels}'.")
    cols: List[str] = []
    for f in freqs:
        if use_channels in ("iq", "i"):
            cols.append(freq_to_i[f])
        if use_channels in ("iq", "q"):
            cols.append(freq_to_q[f])
    return np.array(freqs, dtype=float), cols


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


def stack_line_rolling_median(
    line_df: pd.DataFrame,
    measurement_cols: Sequence[str],
    out_spacing_m: float = 1.0,
    window_m: float = 2.0,
    extra_median_cols: Sequence[str] = (),
    min_samples_per_bin: int = 2,
) -> pd.DataFrame:
    """
    Median-stack one line on a regular spatial grid using a rolling window.

    Output soundings are placed every `out_spacing_m` metres along the GPS
    track. Each output uses all raw samples within +/- `window_m / 2` of its
    bin centre, so e.g. (out_spacing_m=1, window_m=2) means "one sounding per
    metre, each smoothing 2 m of raw data".

    For every measurement column we export the median, an empirical sample
    standard deviation (`<col>_std`), and the within-window sample count
    (`<col>_n`) so the inversion can build a per-sounding noise model directly
    from the stack instead of guessing a fixed relative error.
    """
    g = line_df.copy().sort_values("Sample").reset_index(drop=True)
    if len(g) < min_samples_per_bin:
        return g.iloc[0:0]

    dx = g["X"].to_numpy(dtype=float)
    dy = g["Y"].to_numpy(dtype=float)
    step = np.sqrt(np.diff(dx) ** 2 + np.diff(dy) ** 2)
    along = np.r_[0.0, np.cumsum(step)]
    track_len = float(along.max()) if len(along) else 0.0
    if track_len < out_spacing_m:
        return g.iloc[0:0]

    half = float(window_m) / 2.0
    bin_centres = np.arange(out_spacing_m / 2.0, track_len + out_spacing_m / 2.0, out_spacing_m)

    samples = g["Sample"].to_numpy()
    line_id = int(g["Line"].iloc[0])
    arr_meas = {c: g[c].to_numpy(dtype=float) for c in measurement_cols}
    arr_extra = {c: g[c].to_numpy(dtype=float) for c in extra_median_cols if c in g.columns}

    rows: List[dict] = []
    for bc in bin_centres:
        sel = np.abs(along - bc) <= half
        n_sel = int(sel.sum())
        if n_sel < min_samples_per_bin:
            continue
        row = {
            "Line": line_id,
            "Sample": int(np.median(samples[sel])),
            "X": float(np.median(dx[sel])),
            "Y": float(np.median(dy[sel])),
            "stack_distance_m": float(bc),
            "n_stack": n_sel,
        }
        for c, arr in arr_extra.items():
            v = arr[sel]
            v = v[np.isfinite(v)]
            row[c] = float(np.median(v)) if len(v) else float("nan")
        for c in measurement_cols:
            v = arr_meas[c][sel]
            v = v[np.isfinite(v)]
            n = len(v)
            row[c] = float(np.median(v)) if n else float("nan")
            row[f"{c}_std"] = float(np.std(v, ddof=1)) if n >= 2 else 0.0
            row[f"{c}_n"] = int(n)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optional DEM sampling
# ---------------------------------------------------------------------------


def sample_dem_to_dataframe(
    df: pd.DataFrame,
    dem_path: Path,
    survey_crs: str = "EPSG:32631",
    x_col: str = "X",
    y_col: str = "Y",
    dem_col: str = "DEM_m",
) -> pd.DataFrame:
    """Lazily import rasterio and sample a DEM at survey points."""
    import rasterio
    from rasterio import warp as rio_warp

    out = df.copy()
    x = out[x_col].to_numpy(dtype=float)
    y = out[y_col].to_numpy(dtype=float)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        x_s, y_s = x, y
        survey_crs_obj = rasterio.crs.CRS.from_user_input(survey_crs)
        if dem_crs is not None and survey_crs_obj != dem_crs:
            x_s, y_s = rio_warp.transform(
                survey_crs_obj, dem_crs, x.tolist(), y.tolist()
            )
            x_s = np.asarray(x_s, dtype=float)
            y_s = np.asarray(y_s, dtype=float)
        nodata = src.nodata
        sampled = list(src.sample(list(zip(x_s, y_s))))
    vals = np.array(
        [float(v[0]) if v is not None and len(v) > 0 else np.nan for v in sampled],
        dtype=float,
    )
    if nodata is not None and np.isfinite(nodata):
        vals[np.isclose(vals, float(nodata))] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    vals[np.abs(vals) > 1e4] = np.nan
    out[dem_col] = vals
    return out


# ---------------------------------------------------------------------------
# Forward survey + model utilities
# ---------------------------------------------------------------------------


def build_gem2_hcp_survey(
    frequencies_hz: Sequence[float],
    sensor_height_m: float,
    coil_spacing_m: float = 1.66,
    use_channels: str = "iq",
) -> Survey:
    """
    Build a GEM-2 HCP survey: vertical magnetic dipole Tx/Rx at given AGL.

    `use_channels` selects which receiver components to include and must match
    `get_freq_channel_columns(..., use_channels=...)` so SimPEG returns the
    same number of predicted values, in the same order, as the observed data.
    """
    if use_channels not in ("iq", "i", "q"):
        raise ValueError(f"use_channels must be one of 'iq','i','q' (got '{use_channels}')")
    src_loc = np.array([0.0, 0.0, float(sensor_height_m)])
    rx_loc = np.array([float(coil_spacing_m), 0.0, float(sensor_height_m)])
    rx_list: List = []
    if use_channels in ("iq", "i"):
        rx_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_loc, orientation="z", data_type="ppm", component="real"
            )
        )
    if use_channels in ("iq", "q"):
        rx_list.append(
            fdem.receivers.PointMagneticFieldSecondary(
                rx_loc, orientation="z", data_type="ppm", component="imag"
            )
        )
    src_list = [
        fdem.sources.MagDipole(
            receiver_list=rx_list,
            frequency=float(f),
            location=src_loc,
            orientation="z",
            moment=1.0,
        )
        for f in frequencies_hz
    ]
    return Survey(src_list)


def build_layer_thicknesses(
    max_depth: float,
    first_thickness: float,
    n_layers: int,
) -> np.ndarray:
    """
    Geometric finite-layer thicknesses (length `n_layers - 1`, halfspace
    implicit) growing from `first_thickness` so that the cumulative finite-
    layer thickness equals `max_depth`.

    The growth ratio between adjacent layers is solved numerically; if the
    requested combination implies non-growing layers (`first_thickness *
    (n_layers - 1) > max_depth`) a ValueError is raised.
    """
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2 (at least one finite layer + halfspace)")
    if first_thickness <= 0:
        raise ValueError("first_thickness must be > 0")
    if max_depth <= first_thickness:
        raise ValueError("max_depth must be > first_thickness")

    n_finite = n_layers - 1
    uniform_sum = first_thickness * n_finite
    if max_depth + 1e-9 < uniform_sum:
        raise ValueError(
            f"max_depth={max_depth} is too small for n_layers={n_layers} with "
            f"first_thickness={first_thickness}: would need shrinking layers "
            f"(uniform thickness = {uniform_sum:.3f} m). "
            f"Reduce n_layers or first_thickness, or increase max_depth."
        )
    if abs(max_depth - uniform_sum) < 1e-6:
        return np.full(n_finite, first_thickness)

    from scipy.optimize import brentq

    def _residual(g: float) -> float:
        if abs(g - 1.0) < 1e-9:
            return first_thickness * n_finite - max_depth
        return first_thickness * (g**n_finite - 1.0) / (g - 1.0) - max_depth

    g_lo, g_hi = 1.0001, 10.0
    f_lo, f_hi = _residual(g_lo), _residual(g_hi)
    if f_lo * f_hi > 0:
        return np.full(n_finite, max_depth / n_finite)
    g_solved = float(brentq(_residual, g_lo, g_hi))
    return first_thickness * g_solved ** np.arange(n_finite)


def estimate_apparent_sigma(
    obs: np.ndarray,
    frequencies_hz: np.ndarray,
    coil_spacing_m: float,
    use_channels: str = "iq",
) -> float:
    """
    Low-induction-number apparent conductivity from the lowest-frequency Q.

    H_s/H_p ~= i * omega * mu_0 * sigma * s^2 / 4 at the LIN limit, hence
    sigma ~= 4 * (Q_ppm * 1e-6) / (omega * mu_0 * s^2). With `use_channels="iq"`
    `obs` is laid out as [I_f1, Q_f1, ...], so Q at the lowest frequency is
    `obs[1]`. With `use_channels="q"` it is `obs[0]`.
    Returns a non-negative value clamped to [1e-3, 1.0] S/m so the warm start
    never produces an absurd reference model.
    """
    mu0 = 4.0e-7 * np.pi
    omega = 2.0 * np.pi * float(frequencies_hz[0])
    q_lowest_ppm = float(obs[1] if use_channels == "iq" else obs[0]) if use_channels != "i" else 0.0
    sigma = 4.0 * (q_lowest_ppm * 1e-6) / (omega * mu0 * coil_spacing_m**2)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return 0.1
    return float(np.clip(sigma, 1e-3, 1.0))


# ---------------------------------------------------------------------------
# Per-sounding inversion
# ---------------------------------------------------------------------------


def _frequencies_after_mask(
    frequencies: np.ndarray, channel_cols: Sequence[str], mask: np.ndarray, use_channels: str
) -> Tuple[np.ndarray, str, List[str]]:
    """
    After dropping some channels, recompute (frequencies, use_channels, channel_cols).

    For "q" (or "i") modes each channel maps to one frequency, so the surviving
    frequencies are those whose channel is kept. For "iq" mode we currently
    only support per-frequency drops (drop both I and Q together) so that the
    survey definition stays consistent.
    """
    cols_kept = [c for c, k in zip(channel_cols, mask) if k]
    if use_channels in ("q", "i"):
        freqs_kept = frequencies[mask]
        return freqs_kept, use_channels, list(cols_kept)
    # use_channels == "iq": collapse pairs.
    pair_keep = mask[0::2] & mask[1::2]
    cols_kept = []
    for i, k in enumerate(pair_keep):
        if k:
            cols_kept.append(channel_cols[2 * i])
            cols_kept.append(channel_cols[2 * i + 1])
    return frequencies[pair_keep], "iq", cols_kept


def invert_one_sounding(
    record: dict,
    channel_cols: Sequence[str],
    frequencies: np.ndarray,
    layer_thicknesses: np.ndarray,
    config: dict,
) -> dict:
    """Run a 1D smooth-model SimPEG inversion for one stacked sounding."""
    try:
        sensor_height = float(record.get("sensor_height_m", config["default_sensor_height"]))
        if not np.isfinite(sensor_height) or sensor_height <= 0.0:
            sensor_height = float(config["default_sensor_height"])

        obs_full = np.array([float(record[c]) for c in channel_cols], dtype=float)
        rel_err = float(config["relative_error"])
        floor_ppm = float(config["noise_floor_ppm"])
        std_full = rel_err * np.abs(obs_full) + floor_ppm
        if config.get("noise_from_std", False):
            empirical = np.array(
                [float(record.get(f"{c}_std", 0.0) or 0.0) for c in channel_cols], dtype=float
            )
            n_stack = np.array(
                [int(record.get(f"{c}_n", 0) or 0) for c in channel_cols], dtype=float
            )
            n_safe = np.maximum(n_stack, 1.0)
            std_of_median = 1.253 * empirical / np.sqrt(n_safe)
            std_full = np.maximum(std_full, std_of_median)

        mask = np.ones(len(obs_full), dtype=bool)
        if config.get("drop_negative", False):
            for j, c in enumerate(channel_cols):
                if c.startswith("Q_") and not (np.isfinite(obs_full[j]) and obs_full[j] > 0.0):
                    mask[j] = False
        finite_mask = np.isfinite(obs_full) & np.isfinite(std_full) & (std_full > 0.0)
        mask = mask & finite_mask

        freqs_kept, use_channels_kept, cols_kept = _frequencies_after_mask(
            frequencies, channel_cols, mask, config["use_channels"]
        )
        if len(freqs_kept) == 0 or not np.any(mask):
            return {
                "ok": False, "obs_id": int(record.get("obs_id", -1)),
                "line": int(record.get("Line", -1)),
                "sample": int(record.get("Sample", -1)),
                "error": "all channels masked (negative Q or non-finite)",
            }
        # Rebuild obs/std consistent with possibly reduced (freqs, channels).
        kept_idx = [i for i, c in enumerate(channel_cols) if c in cols_kept]
        obs = obs_full[kept_idx]
        std = std_full[kept_idx]

        survey = build_gem2_hcp_survey(
            frequencies_hz=freqs_kept,
            sensor_height_m=sensor_height,
            coil_spacing_m=config["coil_spacing"],
            use_channels=use_channels_kept,
        )

        n_cells = len(layer_thicknesses) + 1  # finite layers + halfspace
        mesh_thicknesses = np.r_[layer_thicknesses, layer_thicknesses[-1]]
        mesh = TensorMesh([mesh_thicknesses], "0")

        sigma_map = maps.ExpMap(nP=n_cells)
        simulation = fdem.Simulation1DLayered(
            survey=survey,
            thicknesses=layer_thicknesses,
            sigmaMap=sigma_map,
        )

        sigma_warm = estimate_apparent_sigma(
            obs=obs,
            frequencies_hz=freqs_kept,
            coil_spacing_m=config["coil_spacing"],
            use_channels=use_channels_kept,
        )
        starting_model = np.log(sigma_warm * np.ones(n_cells))

        data_object = Data(survey=survey, dobs=obs.flatten(), standard_deviation=std.flatten())
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)

        reg = regularization.WeightedLeastSquares(
            mesh,
            mapping=maps.IdentityMap(nP=n_cells),
            alpha_s=float(config["alpha_s"]),
            alpha_x=float(config["alpha_x"]),
        )
        reg.reference_model = starting_model

        opt = optimization.ProjectedGNCG(
            maxIter=int(config["max_iterations"]),
            maxIterLS=20,
        )
        opt.lower = np.log(float(config["sigma_min"])) * np.ones(n_cells)
        opt.upper = np.log(float(config["sigma_max"])) * np.ones(n_cells)

        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(
            inv_prob,
            [
                directives.UpdateSensitivityWeights(),
                directives.UpdatePreconditioner(),
                directives.BetaEstimate_ByEig(beta0_ratio=float(config["beta0_ratio"])),
                directives.BetaSchedule(
                    coolingFactor=float(config["beta_cooling_factor"]),
                    coolingRate=int(config["beta_cooling_rate"]),
                ),
                directives.TargetMisfit(chifact=float(config["chifact"])),
            ],
        )

        recovered = inv.run(starting_model)
        sigma_layers = (sigma_map * recovered).tolist()
        pred = simulation.dpred(recovered).astype(float)
        residual = (obs - pred).astype(float)
        chi2 = float(np.sum((residual / std) ** 2) / max(1, len(residual)))

        try:
            doi_pack = christiansen_auken_doi(
                simulation,
                recovered,
                std,
                layer_thicknesses,
                d_pred=pred,
                doi_threshold=0.8,
            )
            doi_flat = summarize_doi_for_csv(doi_pack)
        except Exception:
            doi_flat = {"doi_m": float("nan"), "doi_threshold": 0.8}

        # Re-expand obs/pred/residual back to the full original channel list so
        # the misfit CSV has one column per (channel, dobs/dpred/res), with NaN
        # for masked-out entries.
        full_obs = np.full(len(channel_cols), np.nan)
        full_pred = np.full(len(channel_cols), np.nan)
        full_res = np.full(len(channel_cols), np.nan)
        for k, src_idx in enumerate(kept_idx):
            full_obs[src_idx] = obs[k]
            full_pred[src_idx] = pred[k]
            full_res[src_idx] = residual[k]

        return {
            "ok": True,
            "obs_id": int(record.get("obs_id", -1)),
            "line": int(record.get("Line", -1)),
            "sample": int(record.get("Sample", -1)),
            "x": float(record.get("X", np.nan)),
            "y": float(record.get("Y", np.nan)),
            "z_gps": float(record.get("GPSalt(m)", np.nan)),
            "dem_m": float(record.get("DEM_m", np.nan)),
            "sensor_height_used_m": sensor_height,
            "sigma_warm": sigma_warm,
            "chi2": chi2,
            "n_channels_used": int(mask.sum()),
            "n_channels_total": int(len(mask)),
            "sigma_layers": sigma_layers,
            "dobs": full_obs.tolist(),
            "dpred": full_pred.tolist(),
            "residual": full_res.tolist(),
            **doi_flat,
        }
    except Exception as exc:
        return {
            "ok": False,
            "obs_id": int(record.get("obs_id", -1)),
            "line": int(record.get("Line", -1)),
            "sample": int(record.get("Sample", -1)),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    sec = int(max(0, round(seconds)))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def main() -> None:
    p = argparse.ArgumentParser(description="GEM-2 1D SimPEG inversion (smooth-model L2).")
    p.add_argument("input_xyz", type=Path, help="Path to GEM-2 XYZ from gem_csv_to_xyz.py")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Output CSV path (default: alongside input as inverted_<stem>.csv)")
    p.add_argument("--misfit-csv", type=Path, default=None,
                   help="Optional per-sounding misfit CSV (default: misfit_<stem>.csv next to input)")

    p.add_argument("--coil-spacing", type=float, default=1.66, help="Tx-Rx separation in m (GEM-2 = 1.66)")
    p.add_argument("--default-sensor-height", type=float, default=0.1,
                   help="Fallback AGL sensor height (m) when SensorHeight column is missing/invalid")

    p.add_argument("--stack-out-spacing", type=float, default=1.0,
                   help="Output spacing of stacked soundings along the line (m, default 1)")
    p.add_argument("--stack-window", type=float, default=2.0,
                   help="Rolling window length for the median stack (m, default 2)")
    p.add_argument("--min-stack-samples", type=int, default=2,
                   help="Drop stacked soundings with fewer than this many raw samples in window")
    p.add_argument("--noise-from-std", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the empirical within-window std as the per-channel noise model "
                        "(default on; uses 1.253 * std / sqrt(n) as sigma of the median, "
                        "lower-bounded by --relative-error and --noise-floor-ppm).")
    p.add_argument("--drop-negative-q", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop Q channels that are negative (physically impossible for a "
                        "sigma-only forward model; usually a magnetic-susceptibility signature). "
                        "Default on. Negative-Q channels are excluded per sounding.")
    p.add_argument("--decimation", type=int, default=1, help="After stacking, take every Nth sounding")

    p.add_argument("--use-channels", choices=["iq", "q", "i"], default="q",
                   help="Which receiver components to invert. Default 'q' (quadrature only) "
                        "is recommended for shallow GEM-2 inversion when soil has any magnetic "
                        "susceptibility, because in-phase channels are then dominated by mu_r and "
                        "cannot be matched by a sigma-only forward model. Use 'iq' if your site "
                        "is non-magnetic and you want both components.")
    p.add_argument("--exclude-freq", type=float, nargs="*", default=[],
                   help="Frequencies in Hz to drop. Example: --exclude-freq 425 1525")
    p.add_argument("--only-line", type=int, default=None, help="Invert only this Line ID")
    p.add_argument("--limit", type=int, default=None, help="Invert only the first N stacked soundings (for testing)")

    p.add_argument("--relative-error", type=float, default=0.05, help="Relative noise (5%% default)")
    p.add_argument("--noise-floor-ppm", type=float, default=10.0, help="Absolute ppm noise floor")

    p.add_argument("--n-layers", type=int, default=6,
                   help="Total number of model cells including the halfspace. Default 6 is "
                        "suited to the typical 2-3 surviving Q channels.")
    p.add_argument("--max-depth", type=float, default=5.0,
                   help="Cumulative thickness of the finite layers in m (the halfspace starts "
                        "below this). Default 5 m is roughly the GEM-2 sensitivity depth above "
                        "~10 mS/m clays.")
    p.add_argument("--first-thickness", type=float, default=0.4,
                   help="Top layer thickness (m). The growth ratio is solved so the finite "
                        "layers fill --max-depth exactly.")

    p.add_argument("--sigma-min", type=float, default=1e-4, help="Lower conductivity bound (S/m)")
    p.add_argument("--sigma-max", type=float, default=5.0, help="Upper conductivity bound (S/m)")
    p.add_argument("--alpha-s", type=float, default=1e-4, help="Smallness weight (small = weak ref-model pull)")
    p.add_argument("--alpha-x", type=float, default=3.0,
                   help="Smoothness weight along depth (default 3 to keep underdetermined "
                        "2-channel solutions from oscillating)")

    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--beta0-ratio", type=float, default=5.0)
    p.add_argument("--beta-cooling-factor", type=float, default=1.5)
    p.add_argument("--beta-cooling-rate", type=int, default=2)
    p.add_argument("--chifact", type=float, default=1.0)

    p.add_argument("--dem-path", type=Path, default=None,
                   help="Optional DEM raster; if given, sensor height is GPSalt - DEM (clamped 0.05-3 m)")
    p.add_argument("--survey-crs", type=str, default="EPSG:32631")

    p.add_argument("--n-workers", type=int, default=1, help="Parallel workers for per-sounding inversion")

    args = p.parse_args()

    if not args.input_xyz.is_file():
        raise SystemExit(f"Input not found: {args.input_xyz}")

    out_csv = args.output_csv or args.input_xyz.with_name(f"inverted_{args.input_xyz.stem}.csv")
    misfit_csv = args.misfit_csv or args.input_xyz.with_name(f"misfit_{args.input_xyz.stem}.csv")

    print(f"Reading {args.input_xyz}")
    df = read_gem2_xyz(args.input_xyz)
    print(f"  {len(df)} raw rows, {len(df.columns)} columns")

    frequencies, channel_cols = get_freq_channel_columns(df, use_channels=args.use_channels)
    n_per_freq = 2 if args.use_channels == "iq" else 1
    if args.exclude_freq:
        keep = [i for i, f in enumerate(frequencies) if float(f) not in set(args.exclude_freq)]
        if not keep:
            raise SystemExit("All frequencies excluded; nothing to invert.")
        frequencies = frequencies[keep]
        channel_cols = [c for i in keep for c in channel_cols[n_per_freq * i: n_per_freq * (i + 1)]]
        print(f"  Excluded: {sorted(set(args.exclude_freq))}")
    print(f"  Using {len(frequencies)} frequencies (Hz): {frequencies.tolist()}")
    print(f"  Using channels: {args.use_channels.upper()} ({len(channel_cols)} obs per sounding)")

    required = ["Line", "Sample", "X", "Y"] + list(channel_cols)
    optional = [c for c in ("GPSalt(m)", "SensorHeight(m)") if c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    for c in required + optional:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"  {len(df)} rows with valid Line/Sample/X/Y/channels")

    if args.dem_path is not None:
        if not args.dem_path.is_file():
            raise SystemExit(f"DEM not found: {args.dem_path}")
        print(f"  Sampling DEM: {args.dem_path}")
        df = sample_dem_to_dataframe(df, args.dem_path, survey_crs=args.survey_crs)
        if "GPSalt(m)" in df.columns:
            df["sensor_height_m"] = (df["GPSalt(m)"] - df["DEM_m"]).clip(lower=0.05, upper=3.0)
            n_ok = int(df["sensor_height_m"].notna().sum())
            med = float(df["sensor_height_m"].median()) if n_ok else float("nan")
            print(f"  Sensor height from GPS-DEM: median {med:.2f} m ({n_ok} valid)")
    if "sensor_height_m" not in df.columns:
        if "SensorHeight(m)" in df.columns:
            df["sensor_height_m"] = df["SensorHeight(m)"]
        else:
            df["sensor_height_m"] = args.default_sensor_height
        med = float(df["sensor_height_m"].median())
        print(f"  Sensor height from XYZ column: median {med:.2f} m")

    if args.only_line is not None:
        df = df[df["Line"] == args.only_line].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"Line {args.only_line} not present in input.")

    extra = [c for c in ("GPSalt(m)", "DEM_m", "sensor_height_m") if c in df.columns]
    stacked_parts = []
    for line_id, line_df in df.sort_values(["Line", "Sample"]).groupby("Line", sort=True):
        stacked_parts.append(
            stack_line_rolling_median(
                line_df,
                channel_cols,
                out_spacing_m=args.stack_out_spacing,
                window_m=args.stack_window,
                extra_median_cols=extra,
                min_samples_per_bin=args.min_stack_samples,
            )
        )
    stacked_parts = [p for p in stacked_parts if len(p)]
    if not stacked_parts:
        raise SystemExit("Stacking produced no soundings (track too short / too few samples per bin).")
    stacked = pd.concat(stacked_parts, ignore_index=True)
    n_stack_med = float(np.median(stacked["n_stack"]))
    print(f"  Stacked to {len(stacked)} soundings "
          f"(out spacing {args.stack_out_spacing:.1f} m, window {args.stack_window:.1f} m, "
          f"median {n_stack_med:.0f} raw samples per bin)")

    if args.decimation > 1:
        stacked = stacked.iloc[:: args.decimation].reset_index(drop=True)
        print(f"  Decimated to {len(stacked)} soundings (every {args.decimation}th)")
    if args.limit is not None:
        stacked = stacked.head(args.limit).reset_index(drop=True)
        print(f"  Limited to first {len(stacked)} soundings")

    layer_thicknesses = build_layer_thicknesses(
        args.max_depth, args.first_thickness, args.n_layers
    )
    layer_top_depths = np.r_[0.0, np.cumsum(layer_thicknesses)]
    actual_growth = float(layer_thicknesses[-1] / layer_thicknesses[-2]) if len(layer_thicknesses) > 1 else 1.0
    n_total_cells = len(layer_thicknesses) + 1
    print(f"  Model: {n_total_cells} cells "
          f"(top {layer_thicknesses[0]:.2f} m -> last finite {layer_thicknesses[-1]:.2f} m, "
          f"finite-layer base {layer_top_depths[-1]:.2f} m, actual growth ratio {actual_growth:.3f})")

    config = {
        "coil_spacing": args.coil_spacing,
        "default_sensor_height": args.default_sensor_height,
        "relative_error": args.relative_error,
        "noise_floor_ppm": args.noise_floor_ppm,
        "alpha_s": args.alpha_s,
        "alpha_x": args.alpha_x,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "max_iterations": args.max_iterations,
        "beta0_ratio": args.beta0_ratio,
        "beta_cooling_factor": args.beta_cooling_factor,
        "beta_cooling_rate": args.beta_cooling_rate,
        "chifact": args.chifact,
        "use_channels": args.use_channels,
        "noise_from_std": bool(args.noise_from_std),
        "drop_negative": bool(args.drop_negative_q),
    }

    records = stacked.to_dict("records")
    for i, r in enumerate(records):
        r["obs_id"] = i
    n_total = len(records)

    print(f"  Inverting {n_total} stacked soundings (n_workers={args.n_workers}) ...")
    t0 = time.time()
    n_workers = max(1, int(args.n_workers))

    def _print_progress(done: int) -> None:
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_total - done) / rate if rate > 0 else float("inf")
        eta_txt = format_duration(eta) if np.isfinite(eta) else "--:--:--"
        print(f"    progress {done}/{n_total} | elapsed {format_duration(elapsed)} | ETA {eta_txt}")

    inv_out: List[dict] = []
    if n_workers == 1:
        for i, r in enumerate(records, start=1):
            inv_out.append(invert_one_sounding(r, channel_cols, frequencies, layer_thicknesses, config))
            if i == 1 or i % 25 == 0 or i == n_total:
                _print_progress(i)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [
                ex.submit(invert_one_sounding, r, channel_cols, frequencies, layer_thicknesses, config)
                for r in records
            ]
            for done, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
                inv_out.append(fut.result())
                if done == 1 or done % 25 == 0 or done == n_total:
                    _print_progress(done)

    inv_out.sort(key=lambda d: d.get("obs_id", -1))
    n_ok = sum(1 for r in inv_out if r.get("ok"))
    n_fail = n_total - n_ok
    print(f"  Done in {format_duration(time.time() - t0)} | success={n_ok} fail={n_fail}")

    rows: List[dict] = []
    misfit_rows: List[dict] = []
    n_cells = len(layer_thicknesses) + 1
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
            "n_channels_used": r["n_channels_used"],
            "n_channels_total": r["n_channels_total"],
            "doi_m": r.get("doi_m", np.nan),
            "doi_threshold": r.get("doi_threshold", np.nan),
        }
        for j, sigma in enumerate(r["sigma_layers"]):
            row[f"sigma_layer_{j + 1}"] = sigma
        for j in range(n_cells):
            top = layer_top_depths[j]
            bot = layer_top_depths[j + 1] if j + 1 < len(layer_top_depths) else float("nan")
            row[f"depth_top_layer_{j + 1}_m"] = top
            row[f"depth_bottom_layer_{j + 1}_m"] = bot
        rows.append(row)

        m = {
            "obs_id": r["obs_id"],
            "Line": r["line"],
            "Sample": r["sample"],
            "chi2": r["chi2"],
            "n_channels_used": r["n_channels_used"],
            "n_channels_total": r["n_channels_total"],
        }
        for k, ch in enumerate(channel_cols):
            m[f"dobs_{ch}"] = r["dobs"][k]
            m[f"dpred_{ch}"] = r["dpred"][k]
            m[f"res_{ch}"] = r["residual"][k]
        misfit_rows.append(m)

    if rows:
        out_df = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print(f"  Saved inversion: {out_csv} ({len(out_df)} soundings)")
        chi2 = out_df["chi2"].to_numpy(dtype=float)
        chi2_med = float(np.nanmedian(chi2))
        print(f"  chi^2 (per-sounding, normalized): "
              f"median {chi2_med:.2f}, p10 {np.nanpercentile(chi2, 10):.2f}, "
              f"p90 {np.nanpercentile(chi2, 90):.2f}")
    if misfit_rows:
        mdf = pd.DataFrame(misfit_rows)
        misfit_csv.parent.mkdir(parents=True, exist_ok=True)
        mdf.to_csv(misfit_csv, index=False)
        print(f"  Saved misfit: {misfit_csv}")

    if n_fail:
        for r in inv_out:
            if not r.get("ok"):
                print(f"    FAIL Line={r.get('line')} Sample={r.get('sample')}: {r.get('error')}")


if __name__ == "__main__":
    main()
