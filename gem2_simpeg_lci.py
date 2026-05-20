"""
Laterally constrained 1D inversion (LCI) of GEM-2 data using SimPEG.

This module implements the Auken & Christiansen (2004) LCI scheme on top of
SimPEG's `MetaSimulation`. Each survey line is processed as one section: every
stacked sounding contributes its own `Simulation1DLayered` sub-simulation, all
sharing one global flat model `m = log(sigma)` of length
`n_soundings * n_layers`. A 2D `TensorMesh(n_soundings, n_layers)` carries
`WeightedLeastSquares` regularization with `alpha_x` for *lateral* coupling
(Auken's R_rho constraint between same-index layers of neighbouring soundings)
and `alpha_y` for *vertical* smoothness (within each 1D model). Layer
thicknesses are shared and fixed across the line, which is the standard Aarhus
simplification (no thickness coupling R_d).

Re-uses readers and the per-sounding survey builder from `gem2_simpeg_invert`.

CLI usage:

  python gem2_simpeg_lci.py <input.xyz> [options...]

Translation between Auken's "constraint factor" c (e.g. 1.2 = "20% lateral
variation allowed") and SimPEG `alpha_x` in log-conductivity space:

  alpha_x = 1 / (log(c))^2           # so c=1.05 -> ~420, c=1.1 -> ~110,
                                     #    c=1.2  -> ~30,  c=1.3 -> ~14
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
)
from simpeg.data import Data
from simpeg.meta import MetaSimulation

from gem2_doi import christiansen_auken_doi, summarize_doi_for_csv
from gem2_simpeg_invert import (
    build_gem2_hcp_survey,
    build_layer_thicknesses,
    estimate_apparent_sigma,
    get_freq_channel_columns,
    read_gem2_xyz,
    stack_line_rolling_median,
    _frequencies_after_mask,
)


# ---------------------------------------------------------------------------
# Per-sounding artefact assembly
# ---------------------------------------------------------------------------


def _build_sounding_inputs(
    record: dict,
    channel_cols_full: Sequence[str],
    frequencies_full: np.ndarray,
    config: dict,
) -> Optional[Dict]:
    """
    Prepare one sounding's survey, dobs, std, mask, and warm-start sigma.

    Returns None when every channel of this sounding gets masked out by the
    negative-Q drop or non-finite check. The caller then keeps the sounding's
    slot in the global LCI model (so the lateral regularizer fills it in) but
    excludes it from the `MetaSimulation`.
    """
    obs_full = np.array([float(record[c]) for c in channel_cols_full], dtype=float)
    rel_err = float(config["relative_error"])
    floor_ppm = float(config["noise_floor_ppm"])
    std_full = rel_err * np.abs(obs_full) + floor_ppm
    if config.get("noise_from_std", False):
        empirical = np.array(
            [float(record.get(f"{c}_std", 0.0) or 0.0) for c in channel_cols_full], dtype=float
        )
        n_stack = np.array(
            [int(record.get(f"{c}_n", 0) or 0) for c in channel_cols_full], dtype=float
        )
        std_med = 1.253 * empirical / np.sqrt(np.maximum(n_stack, 1.0))
        std_full = np.maximum(std_full, std_med)

    mask = np.ones(len(obs_full), dtype=bool)
    if config.get("drop_negative", False):
        for k, c in enumerate(channel_cols_full):
            if c.startswith("Q_") and not (np.isfinite(obs_full[k]) and obs_full[k] > 0.0):
                mask[k] = False
    mask = mask & np.isfinite(obs_full) & np.isfinite(std_full) & (std_full > 0.0)
    if not np.any(mask):
        return None

    freqs_kept, use_channels_kept, cols_kept = _frequencies_after_mask(
        frequencies_full, list(channel_cols_full), mask, config["use_channels"]
    )
    if len(freqs_kept) == 0:
        return None
    kept_idx = [i for i, c in enumerate(channel_cols_full) if c in cols_kept]
    obs = obs_full[kept_idx]
    std = std_full[kept_idx]

    sensor_h = float(record.get("sensor_height_m", config["default_sensor_height"]))
    if not (np.isfinite(sensor_h) and sensor_h > 0.0):
        sensor_h = float(config["default_sensor_height"])

    survey = build_gem2_hcp_survey(
        frequencies_hz=freqs_kept,
        sensor_height_m=sensor_h,
        coil_spacing_m=config["coil_spacing"],
        use_channels=use_channels_kept,
    )

    sigma_warm = estimate_apparent_sigma(
        obs=obs,
        frequencies_hz=freqs_kept,
        coil_spacing_m=config["coil_spacing"],
        use_channels=use_channels_kept,
    )

    return {
        "survey": survey,
        "obs": obs,
        "std": std,
        "kept_idx": kept_idx,
        "sensor_height_m": sensor_h,
        "sigma_warm": sigma_warm,
    }


# ---------------------------------------------------------------------------
# Per-line LCI
# ---------------------------------------------------------------------------


def invert_one_line_lci(
    line_df: pd.DataFrame,
    channel_cols_full: Sequence[str],
    frequencies_full: np.ndarray,
    layer_thicknesses: np.ndarray,
    config: dict,
) -> dict:
    """Run a single laterally-constrained inversion over one survey line."""
    n_layers_total = len(layer_thicknesses) + 1
    line_df = line_df.sort_values("stack_distance_m").reset_index(drop=True)
    n_soundings = len(line_df)
    if n_soundings < 2:
        return {"ok": False, "error": "LCI needs >= 2 soundings", "results": []}
    n_global = n_soundings * n_layers_total
    line_id = int(line_df["Line"].iloc[0])

    sub_sims: List = []
    sub_mappings: List = []
    dobs_parts: List[np.ndarray] = []
    std_parts: List[np.ndarray] = []
    sounding_meta: List[dict] = []
    sigma_warm_per_sounding = np.full(n_soundings, np.nan, dtype=float)

    for j, rec in enumerate(line_df.to_dict("records")):
        prepared = _build_sounding_inputs(rec, channel_cols_full, frequencies_full, config)
        meta = {
            "obs_id": j,
            "Line": line_id,
            "Sample": int(rec.get("Sample", j)),
            "X": float(rec.get("X", np.nan)),
            "Y": float(rec.get("Y", np.nan)),
            "GPSalt_m": float(rec.get("GPSalt(m)", np.nan)) if "GPSalt(m)" in rec else np.nan,
            "DEM_m": float(rec.get("DEM_m", np.nan)) if "DEM_m" in rec else np.nan,
            "stack_distance_m": float(rec.get("stack_distance_m", float(j))),
            "n_stack": int(rec.get("n_stack", 0) or 0),
            "active": prepared is not None,
            "kept_idx": prepared["kept_idx"] if prepared else [],
            "sensor_height_m": (
                prepared["sensor_height_m"]
                if prepared
                else float(rec.get("sensor_height_m", config["default_sensor_height"]))
            ),
            "sigma_warm": prepared["sigma_warm"] if prepared else float("nan"),
        }
        sounding_meta.append(meta)
        if prepared is None:
            continue
        sigma_warm_per_sounding[j] = prepared["sigma_warm"]

        # Map global m -> sigma for this sounding's 1D simulation:
        # SimPEG layout for TensorMesh([hx, hy]) is m[i + j*nx] where i is x-index
        # (sounding) and j is y-index (layer). So sounding j's layer-l slot is
        # at global index j + l*n_soundings, with layer 0 at the top.
        slice_indices = j + np.arange(n_layers_total) * n_soundings
        sub_sims.append(
            fdem.Simulation1DLayered(
                survey=prepared["survey"],
                thicknesses=layer_thicknesses,
                sigmaMap=maps.IdentityMap(nP=n_layers_total),
            )
        )
        sub_mappings.append(maps.ExpMap(nP=n_layers_total) * maps.Projection(n_global, slice_indices))
        dobs_parts.append(prepared["obs"])
        std_parts.append(prepared["std"])

    if not sub_sims:
        return {
            "ok": False,
            "error": "All soundings on this line had every channel masked.",
            "results": [],
        }

    meta_sim = MetaSimulation(simulations=sub_sims, mappings=sub_mappings)
    dobs = np.concatenate(dobs_parts)
    std = np.concatenate(std_parts)

    # 2D regularization mesh: x = lateral (one cell per sounding), y = layer.
    # Use the actual along-track distances so SimPEG's 1/dx weighting in Wx
    # gives weaker coupling between far-apart soundings.
    distances = line_df["stack_distance_m"].to_numpy(dtype=float)
    if np.all(np.diff(distances) > 0):
        midpoints = 0.5 * (distances[:-1] + distances[1:])
        hx = np.r_[
            distances[1] - distances[0],
            np.diff(midpoints),
            distances[-1] - distances[-2],
        ]
        hx = np.maximum(hx, 1e-3)
    else:
        hx = np.ones(n_soundings)
    hy = np.r_[layer_thicknesses, layer_thicknesses[-1]]
    mesh_2d = TensorMesh([hx, hy], "00")

    # Starting model: per-sounding warm start, fall back to median for empties.
    fallback_sigma = float(np.nanmedian(sigma_warm_per_sounding))
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        fallback_sigma = 0.1
    sigma_per_sounding = np.where(
        np.isfinite(sigma_warm_per_sounding), sigma_warm_per_sounding, fallback_sigma
    )
    starting_model = np.zeros(n_global, dtype=float)
    for j in range(n_soundings):
        slice_indices = j + np.arange(n_layers_total) * n_soundings
        starting_model[slice_indices] = np.log(sigma_per_sounding[j])

    # MetaSimulation exposes a stitched `survey` property that concatenates
    # the sub-sim surveys; SimPEG's Data object needs that for shape checks.
    data_object = Data(survey=meta_sim.survey, dobs=dobs, standard_deviation=std)
    dmis = data_misfit.L2DataMisfit(simulation=meta_sim, data=data_object)

    reg = regularization.WeightedLeastSquares(
        mesh_2d,
        mapping=maps.IdentityMap(nP=n_global),
        alpha_s=float(config["alpha_s"]),
        alpha_x=float(config["alpha_x_lateral"]),
        alpha_y=float(config["alpha_y_vertical"]),
    )
    reg.reference_model = starting_model

    opt = optimization.ProjectedGNCG(
        maxIter=int(config["max_iterations"]),
        maxIterLS=20,
    )
    opt.lower = np.log(float(config["sigma_min"])) * np.ones(n_global)
    opt.upper = np.log(float(config["sigma_max"])) * np.ones(n_global)

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

    t0 = time.time()
    recovered = inv.run(starting_model)
    elapsed = time.time() - t0

    sigma_global = np.exp(recovered)
    pred_full = meta_sim.dpred(recovered).astype(float)

    results: List[dict] = []
    pred_cursor = 0
    for j, meta in enumerate(sounding_meta):
        slice_indices = j + np.arange(n_layers_total) * n_soundings
        sigma_layers = sigma_global[slice_indices].tolist()
        if not meta["active"]:
            results.append(
                {
                    **meta,
                    "ok": True,
                    "active": False,
                    "chi2": float("nan"),
                    "sigma_layers": sigma_layers,
                    "dobs": [],
                    "dpred": [],
                    "residual": [],
                    "n_channels_used": 0,
                    "n_channels_total": len(channel_cols_full),
                    "doi_m": float("nan"),
                    "doi_threshold": float("nan"),
                }
            )
            continue
        n_d = len(meta["kept_idx"])
        obs_j = dobs[pred_cursor: pred_cursor + n_d]
        std_j = std[pred_cursor: pred_cursor + n_d]
        pred_j = pred_full[pred_cursor: pred_cursor + n_d]
        pred_cursor += n_d
        res_j = obs_j - pred_j
        chi2_j = float(np.sum((res_j / std_j) ** 2) / max(1, n_d))

        full_obs = np.full(len(channel_cols_full), np.nan)
        full_pred = np.full(len(channel_cols_full), np.nan)
        full_res = np.full(len(channel_cols_full), np.nan)
        for k, src_idx in enumerate(meta["kept_idx"]):
            full_obs[src_idx] = obs_j[k]
            full_pred[src_idx] = pred_j[k]
            full_res[src_idx] = res_j[k]

        doi_flat = {"doi_m": float("nan"), "doi_threshold": 0.8}
        try:
            prepared_doi = _build_sounding_inputs(
                line_df.iloc[j].to_dict(),
                channel_cols_full,
                frequencies_full,
                config,
            )
            if prepared_doi is not None:
                m_log = recovered[slice_indices]
                doi_sim = fdem.Simulation1DLayered(
                    survey=prepared_doi["survey"],
                    thicknesses=layer_thicknesses,
                    sigmaMap=maps.ExpMap(nP=n_layers_total),
                )
                doi_pack = christiansen_auken_doi(
                    doi_sim,
                    m_log,
                    prepared_doi["std"],
                    layer_thicknesses,
                    d_pred=pred_j,
                    doi_threshold=0.8,
                )
                doi_flat = summarize_doi_for_csv(doi_pack)
        except Exception:
            pass

        results.append(
            {
                **meta,
                "ok": True,
                "active": True,
                "chi2": chi2_j,
                "sigma_layers": sigma_layers,
                "dobs": full_obs.tolist(),
                "dpred": full_pred.tolist(),
                "residual": full_res.tolist(),
                "n_channels_used": int(n_d),
                "n_channels_total": int(len(channel_cols_full)),
                **doi_flat,
            }
        )

    chi2_arr = np.array([r["chi2"] for r in results if r["active"]], dtype=float)
    n_data_total = int(len(dobs))
    chi2_global = float(np.sum(((dobs - pred_full) / std) ** 2) / max(1, n_data_total))
    return {
        "ok": True,
        "line": line_id,
        "n_soundings": n_soundings,
        "n_active": int(sum(1 for r in results if r["active"])),
        "n_data_total": n_data_total,
        "elapsed_s": elapsed,
        "chi2_global": chi2_global,
        "chi2_median": float(np.nanmedian(chi2_arr)) if chi2_arr.size else float("nan"),
        "chi2_p10": float(np.nanpercentile(chi2_arr, 10)) if chi2_arr.size else float("nan"),
        "chi2_p90": float(np.nanpercentile(chi2_arr, 90)) if chi2_arr.size else float("nan"),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Auken-style constraint factor helper
# ---------------------------------------------------------------------------


def alpha_x_from_constraint_factor(c: float) -> float:
    """Auken's lateral constraint factor c -> SimPEG alpha_x.

    The Auken & Christiansen (2004) "constraint factor" is the multiplicative
    factor by which a layer's resistivity is allowed to vary between adjacent
    soundings (e.g. c=1.2 means 20% lateral variation). In log-space the
    standard deviation of the inter-sounding difference is `log(c)`, so an
    L2 prior `||Wx m / sigma_lat||^2` with `sigma_lat = log(c)` is equivalent
    to alpha_x * ||Wx m||^2 with alpha_x = 1 / log(c)^2.
    """
    if c <= 1.0:
        raise ValueError("Constraint factor must be > 1.0")
    return 1.0 / (np.log(float(c)) ** 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_duration(s: float) -> str:
    sec = int(max(0, round(s)))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def main() -> None:
    p = argparse.ArgumentParser(description="GEM-2 1D LCI inversion (laterally constrained).")
    p.add_argument("input_xyz", type=Path)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--misfit-csv", type=Path, default=None)

    p.add_argument("--coil-spacing", type=float, default=1.66)
    p.add_argument("--default-sensor-height", type=float, default=0.1)

    p.add_argument("--stack-out-spacing", type=float, default=1.0)
    p.add_argument("--stack-window", type=float, default=2.0)
    p.add_argument("--min-stack-samples", type=int, default=2)
    p.add_argument("--noise-from-std", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-negative-q", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--use-channels", choices=["iq", "q", "i"], default="q")
    p.add_argument("--exclude-freq", type=float, nargs="*", default=[])
    p.add_argument("--only-line", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="(Test mode) restrict to first N soundings of the chosen line.")

    p.add_argument("--relative-error", type=float, default=0.05)
    p.add_argument("--noise-floor-ppm", type=float, default=10.0)

    p.add_argument("--n-layers", type=int, default=6,
                   help="Total number of model cells including the halfspace.")
    p.add_argument("--max-depth", type=float, default=5.0,
                   help="Cumulative thickness of the finite layers in m (the halfspace starts "
                        "below this).")
    p.add_argument("--first-thickness", type=float, default=0.4,
                   help="Top layer thickness (m). The growth ratio is solved so the finite "
                        "layers fill --max-depth exactly.")

    p.add_argument("--sigma-min", type=float, default=1e-4)
    p.add_argument("--sigma-max", type=float, default=5.0)
    p.add_argument("--alpha-s", type=float, default=1e-4)
    p.add_argument("--alpha-y-vertical", type=float, default=3.0,
                   help="Within-sounding (depth) smoothness; same role as the per-sounding "
                        "script's --alpha-x.")
    p.add_argument(
        "--lateral-constraint",
        type=float,
        default=1.2,
        help="Auken-style lateral constraint factor c (e.g. 1.2 = 20% lateral variation; "
             "1.05 = very tight, 1.5 = loose). Internally converted to SimPEG alpha_x.",
    )
    p.add_argument(
        "--alpha-x-lateral",
        type=float,
        default=None,
        help="If set, use this SimPEG alpha_x directly (overrides --lateral-constraint).",
    )

    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--beta0-ratio", type=float, default=5.0)
    p.add_argument("--beta-cooling-factor", type=float, default=1.5)
    p.add_argument("--beta-cooling-rate", type=int, default=2)
    p.add_argument("--chifact", type=float, default=1.0)

    p.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel processes for multi-line LCI. Each line is one "
             "task; effective concurrency is min(n_workers, n_lines). Single-line "
             "datasets gain nothing from this (one big inversion per line).",
    )

    args = p.parse_args()

    if not args.input_xyz.is_file():
        raise SystemExit(f"Input not found: {args.input_xyz}")
    out_csv = args.output_csv or args.input_xyz.with_name(f"lci_inverted_{args.input_xyz.stem}.csv")
    misfit_csv = args.misfit_csv or args.input_xyz.with_name(f"lci_misfit_{args.input_xyz.stem}.csv")

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
    print(f"  Using {len(frequencies)} frequencies (Hz): {frequencies.tolist()}")
    print(f"  Using channels: {args.use_channels.upper()} ({len(channel_cols)} obs per sounding)")

    required = ["Line", "Sample", "X", "Y"] + list(channel_cols)
    optional = [c for c in ("GPSalt(m)", "SensorHeight(m)") if c in df.columns]
    for c in required + optional:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"  {len(df)} rows with valid Line/Sample/X/Y/channels")

    if "sensor_height_m" not in df.columns:
        if "SensorHeight(m)" in df.columns:
            df["sensor_height_m"] = df["SensorHeight(m)"]
        else:
            df["sensor_height_m"] = args.default_sensor_height

    if args.only_line is not None:
        df = df[df["Line"] == args.only_line].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"Line {args.only_line} not present.")

    extras = [c for c in ("GPSalt(m)", "DEM_m", "SensorHeight(m)", "sensor_height_m")
              if c in df.columns]
    stacked_parts = []
    for line_id, line_df in df.sort_values(["Line", "Sample"]).groupby("Line", sort=True):
        stacked_parts.append(
            stack_line_rolling_median(
                line_df,
                channel_cols,
                out_spacing_m=args.stack_out_spacing,
                window_m=args.stack_window,
                extra_median_cols=extras,
                min_samples_per_bin=args.min_stack_samples,
            )
        )
    stacked_parts = [p for p in stacked_parts if len(p)]
    if not stacked_parts:
        raise SystemExit("Stacking produced no soundings.")
    stacked = pd.concat(stacked_parts, ignore_index=True)
    if "sensor_height_m" not in stacked.columns:
        if "SensorHeight(m)" in stacked.columns:
            stacked["sensor_height_m"] = stacked["SensorHeight(m)"]
        else:
            stacked["sensor_height_m"] = args.default_sensor_height
    n_med = float(np.median(stacked["n_stack"]))
    print(f"  Stacked to {len(stacked)} soundings (median {n_med:.0f} raw samples per bin)")

    if args.limit is not None:
        stacked = stacked.head(args.limit).reset_index(drop=True)
        print(f"  Limited to first {len(stacked)} soundings")

    layer_thicknesses = build_layer_thicknesses(args.max_depth, args.first_thickness, args.n_layers)
    layer_top_depths = np.r_[0.0, np.cumsum(layer_thicknesses)]
    n_layers_total = len(layer_thicknesses) + 1
    actual_growth = float(layer_thicknesses[-1] / layer_thicknesses[-2]) if len(layer_thicknesses) > 1 else 1.0
    print(f"  Model: {n_layers_total} cells "
          f"(top {layer_thicknesses[0]:.2f} m -> last finite {layer_thicknesses[-1]:.2f} m, "
          f"finite-layer base {layer_top_depths[-1]:.2f} m, actual growth ratio {actual_growth:.3f})")

    if args.alpha_x_lateral is not None:
        alpha_x = float(args.alpha_x_lateral)
        c_eff = float(np.exp(1.0 / np.sqrt(alpha_x))) if alpha_x > 0 else float("inf")
        print(f"  Lateral coupling: alpha_x={alpha_x:.2f} (~constraint factor {c_eff:.3f})")
    else:
        alpha_x = alpha_x_from_constraint_factor(float(args.lateral_constraint))
        print(f"  Lateral coupling: constraint factor {args.lateral_constraint:.2f} -> alpha_x={alpha_x:.2f}")

    config = {
        "coil_spacing": args.coil_spacing,
        "default_sensor_height": args.default_sensor_height,
        "relative_error": args.relative_error,
        "noise_floor_ppm": args.noise_floor_ppm,
        "alpha_s": args.alpha_s,
        "alpha_x_lateral": alpha_x,
        "alpha_y_vertical": args.alpha_y_vertical,
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

    line_ids = sorted(stacked["Line"].unique().tolist())
    n_workers = max(1, int(args.n_workers))
    effective_workers = min(n_workers, len(line_ids))
    print(f"  Running LCI on {len(line_ids)} line(s) with {effective_workers} parallel "
          f"worker(s): {line_ids}")
    all_rows: List[dict] = []
    all_misfit: List[dict] = []
    t0 = time.time()

    line_subs = {lid: stacked[stacked["Line"] == lid].copy() for lid in line_ids}
    line_outs: Dict[int, dict] = {}

    if effective_workers <= 1:
        for li, line_id in enumerate(line_ids, start=1):
            sub = line_subs[line_id]
            print(f"  [Line {line_id}, {li}/{len(line_ids)}] inverting {len(sub)} soundings ...")
            line_outs[line_id] = invert_one_line_lci(
                sub, channel_cols, frequencies, layer_thicknesses, config
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as ex:
            future_to_lid = {
                ex.submit(invert_one_line_lci, line_subs[lid], channel_cols, frequencies,
                          layer_thicknesses, config): lid
                for lid in line_ids
            }
            done = 0
            for fut in concurrent.futures.as_completed(future_to_lid):
                lid = future_to_lid[fut]
                done += 1
                try:
                    line_outs[lid] = fut.result()
                except Exception as e:
                    line_outs[lid] = {"ok": False, "error": str(e), "results": []}
                ok = line_outs[lid].get("ok", False)
                tag = (
                    f"global chi^2 {line_outs[lid]['chi2_global']:.2f}"
                    if ok else f"FAILED: {line_outs[lid].get('error')}"
                )
                print(f"  [Line {lid}, {done}/{len(line_ids)}] {tag}")

    for line_id in line_ids:
        out = line_outs.get(line_id)
        if out is None or not out.get("ok"):
            print(f"  Line {line_id}: FAILED ({out.get('error') if out else 'no output'})")
            continue
        if effective_workers <= 1:
            print(f"    Done in {_format_duration(out['elapsed_s'])} | "
                  f"global chi^2 {out['chi2_global']:.2f} over {out['n_data_total']} obs | "
                  f"per-sounding median {out['chi2_median']:.2f} "
                  f"(p10 {out['chi2_p10']:.2f}, p90 {out['chi2_p90']:.2f}) | "
                  f"active soundings {out['n_active']}/{out['n_soundings']}")

        for r in out["results"]:
            row = {
                "Line": r["Line"],
                "obs_id_in_line": r["obs_id"],
                "Sample": r["Sample"],
                "X": r["X"],
                "Y": r["Y"],
                "GPSalt_m": r["GPSalt_m"],
                "DEM_m": r["DEM_m"],
                "stack_distance_m": r["stack_distance_m"],
                "n_stack": r["n_stack"],
                "sensor_height_used_m": r["sensor_height_m"],
                "active": r["active"],
                "sigma_warm_S_per_m": r["sigma_warm"],
                "chi2": r["chi2"],
                "n_channels_used": r["n_channels_used"],
                "n_channels_total": r["n_channels_total"],
                "doi_m": r.get("doi_m", np.nan),
                "doi_threshold": r.get("doi_threshold", np.nan),
            }
            for j_lay, sigma in enumerate(r["sigma_layers"]):
                row[f"sigma_layer_{j_lay + 1}"] = sigma
            for j_lay in range(n_layers_total):
                row[f"depth_top_layer_{j_lay + 1}_m"] = layer_top_depths[j_lay]
                row[f"depth_bottom_layer_{j_lay + 1}_m"] = (
                    layer_top_depths[j_lay + 1] if j_lay + 1 < len(layer_top_depths) else float("nan")
                )
            all_rows.append(row)

            m = {
                "Line": r["Line"],
                "obs_id_in_line": r["obs_id"],
                "Sample": r["Sample"],
                "active": r["active"],
                "chi2": r["chi2"],
                "n_channels_used": r["n_channels_used"],
                "n_channels_total": r["n_channels_total"],
            }
            for k_ch, ch in enumerate(channel_cols):
                if r["dobs"]:
                    m[f"dobs_{ch}"] = r["dobs"][k_ch]
                    m[f"dpred_{ch}"] = r["dpred"][k_ch]
                    m[f"res_{ch}"] = r["residual"][k_ch]
                else:
                    m[f"dobs_{ch}"] = float("nan")
                    m[f"dpred_{ch}"] = float("nan")
                    m[f"res_{ch}"] = float("nan")
            all_misfit.append(m)

    total_elapsed = time.time() - t0
    print(f"\n  Total elapsed: {_format_duration(total_elapsed)} | "
          f"saved {len(all_rows)} sounding rows")

    if all_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"  Saved inversion: {out_csv}")
    if all_misfit:
        misfit_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_misfit).to_csv(misfit_csv, index=False)
        print(f"  Saved misfit:    {misfit_csv}")


if __name__ == "__main__":
    main()
