#!/usr/bin/env python3
"""
Depth of Investigation (DOI) after Christiansen & Auken (Jacobian / cumulative
sensitivity), adapted to SimPEG GEM-2 inversions.

Uses the **same fixed thickness mesh** as the inversion (no rediscretization).
Model parameters are **natural log conductivity** per layer, matching
``Simulation1DLayered`` + ``maps.ExpMap``.

Sensitivity chain:

1. ``J = ∂d / ∂ log σ`` — built column-wise with ``simulation.Jvec(m, e_j)``.
2. ``G_ij ≈ ∂ log(|d_i|) / ∂ log σ_j`` — multiply rows by ``1 / max(|d_pred_i|, ε)``
   so GEM ppm data never passes through ``log`` of a negative value (Quadrature
   is usually positive after masking; see notes).
3. ``s_j = Σ_i G_ij / Δd_i`` with ``Δd_i = std_i`` from the inversion noise model.
4. ``s′_j = s_j / t_j`` with finite ``t_j``; the semi-infinite bottom cell has no
   ``s′`` (NaN).
5. **DOI (depth at which cumulative sensitivity from the surface reaches a
   fraction of the total)** — practical counterpart to the paper's threshold on
   cumulative curves: depth to the **bottom** of the first layer for which
   ``Σ_{k=0..j} |s_k| / Σ |s_k| ≥ threshold`` (default **0.8**).

The empirical global threshold (e.g. 0.8 in the paper) was tuned for their
definition of cumulative ``S``; treat ``doi_threshold`` as tunable when using
this variant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def dpred_dlogsigma_matrix(simulation, m_log_sigma: np.ndarray) -> np.ndarray:
    """Dense Jacobian ``∂d / ∂ log σ`` with shape (n_data, n_params)."""
    m = np.asarray(m_log_sigma, dtype=float).ravel()
    n = m.size
    cols = []
    eye = np.eye(n)
    for j in range(n):
        cols.append(np.asarray(simulation.Jvec(m, eye[:, j]), dtype=float).ravel())
    return np.column_stack(cols)


def christiansen_auken_doi(
    simulation,
    m_log_sigma: np.ndarray,
    std: np.ndarray,
    layer_thicknesses: np.ndarray,
    *,
    d_pred: Optional[np.ndarray] = None,
    doi_threshold: float = 0.8,
    eps_log_amp: float = 1e-12,
    use_abs_s_mass: bool = True,
) -> Dict[str, Any]:
    """
    Compute column sensitivities ``s_j``, thickness-normalized ``s′``, and a
    scalar **DOI depth (m)** below surface.

    Parameters
    ----------
    simulation
        ``fdem.Simulation1DLayered`` already configured like the inversion run.
    m_log_sigma
        Recovered model vector ``log(σ)`` (natural log, S/m).
    std
        Per-data standard deviations (same layout as ``d_pred`` / inversion).
    layer_thicknesses
        1D array of **finite** layer thicknesses (m); half-space is implicit.
    d_pred
        Predicted data; if None, ``simulation.dpred(m_log_sigma)`` is used.
    doi_threshold
        Cumulative **|s|** mass from the surface (finite layers only for mass
        normalization when excluding half-space — see implementation).
    eps_log_amp
        Floor when scaling rows by ``1/|d_pred|`` for log-amplitude derivatives.
    use_abs_s_mass
        If True, use ``|s_j|`` for cumulative mass (stable when signed columns
        cancel). If False, use signed ``s_j``.

    Returns
    -------
    dict with arrays ``depth_layer_top_m``, ``depth_layer_bottom_m``,
    ``depth_layer_centre_m``, ``s``, ``s_prime``, ``G_log_amp``, ``cum_mass``,
    and scalars ``doi_depth_m``, ``doi_threshold``.
    """
    m = np.asarray(m_log_sigma, dtype=float).ravel()
    std = np.asarray(std, dtype=float).ravel()

    if d_pred is None:
        d_pred = np.asarray(simulation.dpred(m), dtype=float).ravel()
    else:
        d_pred = np.asarray(d_pred, dtype=float).ravel()

    if std.size != d_pred.size:
        raise ValueError(f"d_pred ({d_pred.size}) and std ({std.size}) length mismatch")

    J = dpred_dlogsigma_matrix(simulation, m)
    if J.shape != (d_pred.size, m.size):
        raise ValueError(f"Unexpected J shape {J.shape}, expected nD={d_pred.size}, nP={m.size}")

    scale = 1.0 / np.maximum(np.abs(d_pred), eps_log_amp)
    G = scale[:, np.newaxis] * J

    s = np.sum(G / std[:, np.newaxis], axis=0)

    lt = np.asarray(layer_thicknesses, dtype=float).ravel()
    n_cells = m.size
    if n_cells != len(lt) + 1:
        raise ValueError(
            f"Model length {n_cells} should be len(layer_thicknesses)+1 "
            f"(got {len(lt)} thicknesses)."
        )

    tops = np.r_[0.0, np.cumsum(lt)]
    bottoms = np.empty(n_cells, dtype=float)
    centres = np.empty(n_cells, dtype=float)
    s_prime = np.full(n_cells, np.nan, dtype=float)
    thk = np.empty(n_cells, dtype=float)

    for j in range(n_cells):
        if j < len(lt):
            thk[j] = float(lt[j])
            bottoms[j] = float(tops[j + 1])
            centres[j] = float(tops[j] + 0.5 * thk[j])
            s_prime[j] = float(s[j] / thk[j]) if thk[j] > 0 else np.nan
        else:
            thk[j] = np.nan
            bottoms[j] = np.nan
            centres[j] = float(tops[j] + 0.5 * float(lt[-1]))
            s_prime[j] = np.nan

    mass = np.abs(s) if use_abs_s_mass else s
    # Cumulative information from surface downward using **finite** layers only
    # for the mass budget (half-space column gets ambiguous thickness).
    n_fin = len(lt)
    mass_fin = mass[:n_fin]
    total = float(np.sum(mass_fin))
    if total <= 0.0 or not np.isfinite(total):
        doi_m = float("nan")
        cum_mass = np.full(n_fin, np.nan)
    else:
        cum_mass = np.cumsum(mass_fin) / total
        idx = int(np.searchsorted(cum_mass, float(doi_threshold), side="left"))
        if idx >= n_fin:
            doi_m = float(tops[-1])
        else:
            doi_m = float(bottoms[idx])

    out: Dict[str, Any] = {
        "doi_depth_m": doi_m,
        "doi_threshold": float(doi_threshold),
        "depth_layer_top_m": tops[:n_cells],
        "depth_layer_bottom_m": bottoms,
        "depth_layer_centre_m": centres,
        "layer_thickness_m": thk,
        "s": s,
        "s_prime": s_prime,
        "G_log_amp": G,
        "cum_mass_surface_to_bottom_of_layer": cum_mass if total > 0 else np.full(n_fin, np.nan),
    }
    return out


def summarize_doi_for_csv(doi_dict: Dict[str, Any]) -> Dict[str, float]:
    """Flatten minimal fields for inversion CSV export."""
    return {
        "doi_m": float(doi_dict.get("doi_depth_m", np.nan)),
        "doi_threshold": float(doi_dict.get("doi_threshold", np.nan)),
    }
