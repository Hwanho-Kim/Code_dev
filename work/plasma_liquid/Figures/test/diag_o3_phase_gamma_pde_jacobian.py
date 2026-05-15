#!/usr/bin/env python3
"""Phase γ: Full PDE Jacobian (1225 dim) eigenvalue analysis.

User analysis (2026-05-06): 0D chemistry has no Hopf (Phase H confirmed).
Sustained oscillation could come from spatial-temporal Hopf in 1D PDE coupling.
Full 1225×1225 Jacobian eigenvalue spectrum check.

Method:
  1. Load 2.6 kV baseline at quasi-SS (t=300s, after wall formation).
  2. Build full PDE rhs Jacobian via finite difference (1225 perturbations).
  3. scipy linalg.eig → 1225 eigenvalues.
  4. Hopf check: any eigenvalue with Re > 0 AND |Im| > 0?

Cost: ~1225 rhs evaluations × ~10ms each = ~12s + linalg.eig ~30s.
Total ~1 min.

If finds Hopf candidates → spatial-temporal limit cycle (physical)
If no Hopf → sustained oscillation must be numerical or forced by gas BC variation
"""
from __future__ import annotations

import functools
import sys
import time as time_mod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

from chemistry_1d import AqueousChemistry1D  # noqa: E402
from config_1d import AQUEOUS_SPECIES  # noqa: E402
from pde_solver import PDESolver1D  # noqa: E402

print = functools.partial(print, flush=True)


def build_pde_jacobian(solver, y0, t0, eps_rel=1e-7):
    """Numerical Jacobian of solver.rhs at (t0, y0).

    Returns (N_total, N_total) matrix.
    Use central difference for accuracy.
    """
    N = len(y0)
    print(f"  Building Jacobian: {N}×{N}, ~{2*N} rhs evaluations...")
    t_start = time_mod.time()

    f0 = solver.rhs(t0, y0)
    J = np.zeros((N, N))

    for j in range(N):
        h = max(eps_rel * abs(y0[j]), eps_rel * 1e-15)
        y_plus = y0.copy()
        y_plus[j] += h
        f_plus = solver.rhs(t0, y_plus)
        y_minus = y0.copy()
        y_minus[j] = max(y0[j] - h, 0.0)
        h_actual = y_plus[j] - y_minus[j]
        f_minus = solver.rhs(t0, y_minus)
        J[:, j] = (f_plus - f_minus) / max(h_actual, 1e-300)

        if (j + 1) % 200 == 0:
            print(f"    progress {j+1}/{N} ({time_mod.time()-t_start:.0f}s)")

    print(f"  Jacobian built in {time_mod.time()-t_start:.0f}s")
    return J


def main():
    print("=" * 80)
    print("Phase γ: Full PDE Jacobian eigenvalue analysis")
    print("=" * 80)

    for V in ("2.6kV", "3.6kV"):
        print(f"\n--- {V} quasi-SS at t=300s ---")
        import gen_all_figures as gaf

        gaf.IS_SALINE = False
        gaf.DEFAULT_GAS_SHEET = V
        gaf.SOLUTION_LABEL = "DIW"
        gaf.FIXED_CATION_CONC = 0.0
        gaf.CONDITION_LABEL = "Humid_fitting"
        gaf.EXP = gaf.EXP_DIW_ALL[V]

        times, gas_conc = gaf.load_gas_data()
        HONO_GAS = gas_conc["NO2"] * gaf.RH80_RATIOS[V]["HONO_NO2"]
        HONO2_GAS = gas_conc["N2O5"] * gaf.HONO2_RATIO
        H2O2_GAS = gas_conc["O3"] * gaf.H2O2_RATIO

        chem = AqueousChemistry1D(saline_mode=False)
        solver = PDESolver1D(
            chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
            saline_mode=False, bc_type="three_film", alpha_b=None,
            delta_gas=0.01, delta_liq=1e-4,
        )
        solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                            hono_gas=HONO_GAS, hono2_gas=HONO2_GAS,
                            h2o2_gas=H2O2_GAS)

        # Initial state from cache at t=300s
        fp = (_root / "Figures" / "DIW results"
              / f"{V}_Humid_fitting_three_film_HONOvar"
              / "cache" / "three_film_abspecies_dg0.0100.npz")
        d = dict(np.load(fp, allow_pickle=True))
        snap_t_cache = np.asarray(d["snap_t"])
        snap_y_cache = np.asarray(d["snap_y"])
        si = int(np.argmin(np.abs(snap_t_cache - 300)))
        y0 = snap_y_cache[si].ravel()

        N_total = solver.N_total
        N_z = solver.N_z
        N_s = solver.N_s
        print(f"  N_total = {N_total} (N_z={N_z}, N_s={N_s})")

        J = build_pde_jacobian(solver, y0, t0=300.0, eps_rel=1e-7)

        print(f"  Computing eigenvalues...")
        t_start = time_mod.time()
        eigvals = np.linalg.eigvals(J)
        print(f"  Eigenvalues computed in {time_mod.time()-t_start:.0f}s")

        # Statistics
        re_neg = np.sum(eigvals.real < -1e-15)
        re_zero = np.sum(np.abs(eigvals.real) <= 1e-15)
        re_pos = np.sum(eigvals.real > 1e-15)

        # Hopf candidates: Re > 0 AND |Im| > 0
        hopf = eigvals[(eigvals.real > 1e-15) &
                       (np.abs(eigvals.imag) > 1e-15)]

        # Sort by real part descending
        sorted_eigs = eigvals[np.argsort(-eigvals.real)]

        print(f"\n  === EIGENVALUE SUMMARY ({V}) ===")
        print(f"  Total: {N_total}")
        print(f"  Re < 0:  {re_neg}")
        print(f"  Re ≈ 0:  {re_zero}")
        print(f"  Re > 0:  {re_pos}")
        print(f"  ★ Hopf candidates (Re>0 AND Im≠0): {len(hopf)}")
        if len(hopf) > 0:
            print(f"  Top 5 Hopf candidates:")
            hopf_sorted = hopf[np.argsort(-hopf.real)][:5]
            for ev in hopf_sorted:
                period = 2 * np.pi / abs(ev.imag)
                growth = 1 / ev.real
                print(f"    λ = {ev.real:+.3e} ± {abs(ev.imag):.3e}i, "
                      f"period {period:.2f}s, growth τ {growth:.2e}s")
        print(f"  Top 5 most-positive Re eigenvalues:")
        for ev in sorted_eigs[:5]:
            print(f"    λ = {ev.real:+.3e} + {ev.imag:+.3e}i")

        # Save figure
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(eigvals.real, eigvals.imag, s=8, c="#1f77b4", alpha=0.5,
                   label=f"all {N_total} eigenvalues")
        # Highlight positive Re
        pos_mask = eigvals.real > 1e-15
        if pos_mask.any():
            ax.scatter(eigvals.real[pos_mask], eigvals.imag[pos_mask],
                       s=40, c="#d62728", marker="x",
                       label=f"Re>0 ({pos_mask.sum()})")
        # Highlight Hopf candidates
        hopf_mask = (eigvals.real > 1e-15) & (np.abs(eigvals.imag) > 1e-15)
        if hopf_mask.any():
            ax.scatter(eigvals.real[hopf_mask], eigvals.imag[hopf_mask],
                       s=120, facecolors='none', edgecolors='red', lw=2,
                       label=f"Hopf candidate ({hopf_mask.sum()})")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Re(λ) [s⁻¹]")
        ax.set_ylabel("Im(λ) [s⁻¹]")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"Phase γ: Full PDE Jacobian eigenvalues — {V} t=300s "
                     f"(N={N_total})", fontweight="bold", loc="left")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = Path(__file__).parent
        for ext in ("png", "pdf"):
            p = out / f"fig_diag_o3_gamma_{V}.{ext}"
            fig.savefig(p, dpi=200 if ext == "png" else None,
                        bbox_inches="tight")
            print(f"  saved: {p}")


if __name__ == "__main__":
    main()
