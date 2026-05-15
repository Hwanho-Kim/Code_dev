#!/usr/bin/env python3
"""Phase H: Jacobian eigenvalue analysis at 2.6 kV quasi-SS.

User analysis (2026-05-06):
  - R32 alone (mutual annihilation) cannot generate sustained limit cycle.
    2-species Jacobian: trace<0, det>0 → eigenvalues always negative real part
    → damped only. Best case is decaying oscillation, not sustained.
  - Sustained limit cycle requires Hopf bifurcation: positive real eigenvalue
    + nonzero imaginary part somewhere in chemistry network.
  - This test identifies whether such mode exists.

Method:
  1. Load 2.6 kV baseline cache, extract bulk-only avg y_cell at quasi-SS (t=400s+).
  2. Build chemistry RHS Jacobian via finite difference (25-dim state).
  3. Compute eigenvalues, plot Re vs Im.
  4. Check Hopf condition: any eigenvalue with Re > 0 AND |Im| > 0?

Limitations:
  - 0D chemistry only (no diffusion coupling). Spatial-coupled PDE Jacobian is
    49×25=1225 dim, expensive but possible follow-up.
  - Linearization around single point — true nonlinear dynamics may differ.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

from chemistry_1d import AqueousChemistry1D  # noqa: E402
from config_1d import AQUEOUS_SPECIES  # noqa: E402

print = functools.partial(print, flush=True)


def get_qss_state(voltage: str, t_target: float = 400.0) -> tuple:
    """Load cache and extract bulk-only volume-averaged y_cell at t_target."""
    fp = (_root / "Figures" / "DIW results"
          / f"{voltage}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])  # (nt, N_z, N_s)
    dz = np.asarray(d["dz_cells"])
    z_mm = np.asarray(d["z_centers"]) * 1e3

    si = int(np.argmin(np.abs(snap_t - t_target)))
    # Volume-weighted bulk-only avg (z > 0.1 mm)
    mask = z_mm > 0.1
    dz_b = dz[mask]
    Lb = float(dz_b.sum())
    y_avg = np.sum(snap_y[si, mask, :] * dz_b[:, None], axis=0) / Lb
    return y_avg, snap_y[si, 0, :], snap_t[si]


def compute_chemistry_jacobian(chem, y_cell: np.ndarray,
                                eps: float = 1e-8) -> np.ndarray:
    """Numerical Jacobian of chemistry RHS (no diffusion).

    J[i,j] = ∂(dy_i/dt)/∂y_j

    Use central difference for accuracy. eps relative perturbation.
    """
    N = len(y_cell)
    f0 = chem.compute_rates_numba(y_cell)
    J = np.zeros((N, N))
    for j in range(N):
        # Adaptive perturbation: max(eps × |y_j|, eps × 1e-15) to handle trace species
        h = max(eps * abs(y_cell[j]), eps * 1e-15)
        y_plus = y_cell.copy()
        y_plus[j] += h
        f_plus = chem.compute_rates_numba(y_plus)
        y_minus = y_cell.copy()
        y_minus[j] = max(y_cell[j] - h, 0.0)  # avoid negative
        h_actual = y_plus[j] - y_minus[j]
        f_minus = chem.compute_rates_numba(y_minus)
        J[:, j] = (f_plus - f_minus) / max(h_actual, 1e-300)
    return J


def main():
    print("=" * 80)
    print("Phase H: Jacobian eigenvalue analysis (2.6 kV bulk-only quasi-SS)")
    print("=" * 80)

    chem = AqueousChemistry1D(saline_mode=False)
    N_s = len(chem.species_idx)
    print(f"  N_species = {N_s}")
    print(f"  species: {AQUEOUS_SPECIES}")

    results = {}
    for V in ["2.6kV", "3.6kV"]:
        print(f"\n--- {V} bulk-only quasi-SS ---")
        y_bulk, y_surf, t_at = get_qss_state(V, t_target=400.0)
        print(f"  Loaded at t={t_at:.0f}s")
        # Build Jacobian for bulk-only avg state
        # Clip to chemistry valid range first
        y_bulk_clip = np.clip(y_bulk, chem.trace, 1.0)
        y_bulk_clip[chem.species_idx["H+"]] = max(y_bulk_clip[chem.species_idx["H+"]], 1e-14)
        print(f"  H+: {y_bulk_clip[chem.species_idx['H+']]:.3e} M, "
              f"pH = {-np.log10(max(y_bulk_clip[chem.species_idx['H+']], 1e-14)):.2f}")
        print(f"  O3: {y_bulk_clip[chem.species_idx['O3']]:.3e} M")
        print(f"  HONO_total: {y_bulk_clip[chem.species_idx['HONO_total']]:.3e} M")

        print(f"  computing 0D chemistry Jacobian (bulk avg, {N_s}×{N_s})...")
        J_bulk = compute_chemistry_jacobian(chem, y_bulk_clip, eps=1e-7)

        # Eigenvalues
        eigvals_bulk = np.linalg.eigvals(J_bulk)

        # Surface state Jacobian for comparison
        y_surf_clip = np.clip(y_surf, chem.trace, 1.0)
        y_surf_clip[chem.species_idx["H+"]] = max(y_surf_clip[chem.species_idx["H+"]], 1e-14)
        J_surf = compute_chemistry_jacobian(chem, y_surf_clip, eps=1e-7)
        eigvals_surf = np.linalg.eigvals(J_surf)

        results[V] = {
            "y_bulk": y_bulk_clip,
            "y_surf": y_surf_clip,
            "J_bulk": J_bulk,
            "J_surf": J_surf,
            "eig_bulk": eigvals_bulk,
            "eig_surf": eigvals_surf,
        }

        # Hopf check: eigenvalues with Re > 0 AND |Im| > 0
        hopf_bulk = eigvals_bulk[(eigvals_bulk.real > 1e-15) &
                                  (np.abs(eigvals_bulk.imag) > 1e-15)]
        hopf_surf = eigvals_surf[(eigvals_surf.real > 1e-15) &
                                  (np.abs(eigvals_surf.imag) > 1e-15)]
        print(f"\n  === EIGENVALUE SUMMARY (bulk-only avg state) ===")
        print(f"  Total {N_s} eigenvalues")
        print(f"  Re < 0:        {np.sum(eigvals_bulk.real < -1e-15)}")
        print(f"  Re ≈ 0:        {np.sum(np.abs(eigvals_bulk.real) <= 1e-15)}")
        print(f"  Re > 0:        {np.sum(eigvals_bulk.real > 1e-15)}")
        print(f"  Re > 0 + complex (Hopf candidates): {len(hopf_bulk)}")
        if len(hopf_bulk) > 0:
            print(f"  ★★ Hopf candidates (positive Re + nonzero Im):")
            for ev in hopf_bulk:
                period = 2 * np.pi / abs(ev.imag)
                print(f"     λ = {ev.real:+.3e} ± {abs(ev.imag):.3e}i, "
                      f"period {period:.2f}s, growth τ = {1/ev.real:.2e}s")
        else:
            print(f"  → NO Hopf candidates: bulk chemistry CANNOT generate "
                  f"sustained limit cycle on its own (linear analysis)")

        print(f"\n  Surface state — {len(hopf_surf)} Hopf candidates")
        if len(hopf_surf) > 0:
            for ev in hopf_surf[:5]:
                period = 2 * np.pi / abs(ev.imag)
                print(f"     λ = {ev.real:+.3e} ± {abs(ev.imag):.3e}i, "
                      f"period {period:.2f}s")

        # Most positive real part
        most_pos = eigvals_bulk[np.argmax(eigvals_bulk.real)]
        print(f"\n  Bulk: most-positive Re eigenvalue: "
              f"{most_pos.real:+.3e} + {most_pos.imag:.3e}i")
        most_pos_s = eigvals_surf[np.argmax(eigvals_surf.real)]
        print(f"  Surface: most-positive Re eigenvalue: "
              f"{most_pos_s.real:+.3e} + {most_pos_s.imag:.3e}i")

    # Plot eigenvalue spectra
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ci, V in enumerate(["2.6kV", "3.6kV"]):
        d = results[V]
        # Bulk
        ax = axes[0, ci]
        ev = d["eig_bulk"]
        ax.scatter(ev.real, ev.imag, s=30, c="#1f77b4", alpha=0.7,
                   label="all eigvals")
        # Highlight positive Re
        pos_re = ev[ev.real > 1e-15]
        if len(pos_re) > 0:
            ax.scatter(pos_re.real, pos_re.imag, s=80, c="#d62728",
                       label=f"Re>0 ({len(pos_re)})", marker="x")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Re(λ) [s⁻¹]")
        ax.set_ylabel("Im(λ) [s⁻¹]")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"({chr(97+2*ci)}) {V} bulk-only avg quasi-SS",
                     fontweight="bold", loc="left")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

        # Surface
        ax = axes[1, ci]
        ev = d["eig_surf"]
        ax.scatter(ev.real, ev.imag, s=30, c="#1f77b4", alpha=0.7,
                   label="all eigvals")
        pos_re = ev[ev.real > 1e-15]
        if len(pos_re) > 0:
            ax.scatter(pos_re.real, pos_re.imag, s=80, c="#d62728",
                       label=f"Re>0 ({len(pos_re)})", marker="x")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Re(λ) [s⁻¹]")
        ax.set_ylabel("Im(λ) [s⁻¹]")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"({chr(98+2*ci)}) {V} surface (z=0)",
                     fontweight="bold", loc="left")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle("Phase H: Chemistry Jacobian eigenvalues — "
                 "Hopf bifurcation candidate identification",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_h.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"\nsaved: {p}")


if __name__ == "__main__":
    main()
