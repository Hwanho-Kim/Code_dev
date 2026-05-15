#!/usr/bin/env python3
"""Cl- evolution + total Cl atom mass conservation diagnostic.

Goals:
  1. Pre/post Cl- profile (spatial and time-evolution)
  2. Total Cl atom mass at t=0 vs t=600s (conservation check)
  3. Where the +1112 µM Cl- bulk drift comes from
     (chemistry vs numerical projection vs electroneutrality)
  4. Activation level of Cl chemistry — how much Cl- got "redirected"
     into transient species (Cl2, Cl2-, HClO, ClNO2, etc.)
"""
from __future__ import annotations
import functools, sys
from pathlib import Path
import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))
sys.path.insert(0, str(_root / 'Figures'))

import gen_all_figures as gaf
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

print = functools.partial(print, flush=True)

# Cl atom count per species (matches _enforce_cl_conservation in pde_solver)
CL_ATOM_COUNT = {
    'Cl-': 1, 'HOCl-': 1, 'Cl': 1, 'HOClH': 1, 'HCl': 1,
    'HClO_total': 1, 'HClO2_total': 1, 'ClO': 1, 'ClO2': 1,
    'ClO3': 1, 'ClNO2': 1, 'ClO3-': 1, 'ClO4-': 1,
    'Cl2-': 2, 'Cl2': 2, 'Cl2O': 2, 'Cl2O2': 2, 'Cl2O3': 2,
    'Cl2O4': 2, 'Cl2O5': 2, 'Cl2O6': 2,
    'Cl3-': 3,
}


def total_cl_per_cell(y_2d: np.ndarray, idx_map: dict) -> np.ndarray:
    """Sum n_Cl × [species] per cell."""
    N_z = y_2d.shape[0]
    out = np.zeros(N_z)
    for sp, n in CL_ATOM_COUNT.items():
        if sp in idx_map:
            out += n * y_2d[:, idx_map[sp]]
    return out


def vol_avg(arr: np.ndarray, dz: np.ndarray, L: float) -> float:
    return float(np.sum(arr * dz) / L)


def main():
    gaf.DEFAULT_GAS_SHEET = '3.2kV'
    gaf.CONDITION_LABEL = 'Humid_fitting'
    times, gas_conc = gaf.load_gas_data()

    chem = AqueousChemistry1D(saline_mode=True)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=True, fixed_cation_conc=0.154,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=gaf.HONO_GAS, hono2_gas=gaf.HONO2_GAS,
        h2o2_gas=gaf.H2O2_GAS,
    )

    idx = solver.species_idx
    dz = solver.dz_cells
    L = solver.L
    N_z = solver.N_z

    y0_flat = solver.build_initial_condition(initial_pH=7.0)
    y0_2d = y0_flat.reshape(N_z, solver.N_s)

    # Snapshot timing
    t_end = float(times[-1])
    t_eval = np.array([2.0, 30.0, 60.0, 120.0, 300.0, 600.0])
    t_eval = t_eval[t_eval <= t_end]

    print("=" * 80)
    print("Cl evolution diagnostic — Saline 3.2 kV Humid_fitting")
    print("=" * 80)

    # ----- Initial state -----
    cl_per_cell_init = total_cl_per_cell(y0_2d, idx)
    cl_total_init_M = vol_avg(cl_per_cell_init, dz, L)
    cl_minus_init = vol_avg(y0_2d[:, idx['Cl-']], dz, L)
    print(f"\n[INITIAL t=0]")
    print(f"  Cl- bulk-avg     = {cl_minus_init*1e3:.4f} mM")
    print(f"  Total Cl atoms   = {cl_total_init_M*1e3:.4f} mM (bulk-avg)")
    print(f"  Cl- range over cells: "
          f"{y0_2d[:, idx['Cl-']].min()*1e3:.4f} - "
          f"{y0_2d[:, idx['Cl-']].max()*1e3:.4f} mM")

    # ----- Run -----
    print(f"\nRunning ({N_z} cells, t_end={t_end}s)...")
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval, y0=y0_flat,
                          verbose=False, dt_poisson=None)

    y_final = result['y_final']
    if y_final.ndim == 1:
        y_final = y_final.reshape(N_z, solver.N_s)

    # ----- Final state -----
    cl_per_cell_final = total_cl_per_cell(y_final, idx)
    cl_total_final_M = vol_avg(cl_per_cell_final, dz, L)
    cl_minus_final = vol_avg(y_final[:, idx['Cl-']], dz, L)

    print(f"\n[FINAL t=600s]")
    print(f"  Cl- bulk-avg     = {cl_minus_final*1e3:.4f} mM    "
          f"Δ vs init = {(cl_minus_final - cl_minus_init)*1e6:+.2f} µM")
    print(f"  Total Cl atoms   = {cl_total_final_M*1e3:.4f} mM    "
          f"Δ vs init = {(cl_total_final_M - cl_total_init_M)*1e6:+.2f} µM"
          f" (drift = {(cl_total_final_M/cl_total_init_M - 1)*100:+.4f}%)")

    # Per-cell Cl- distribution (final)
    cl_minus_final_arr = y_final[:, idx['Cl-']]
    print(f"\n  Cl- per-cell range (final): "
          f"{cl_minus_final_arr.min()*1e3:.4f} - "
          f"{cl_minus_final_arr.max()*1e3:.4f} mM")
    print(f"  Surface cell Cl- = {cl_minus_final_arr[0]*1e3:.4f} mM "
          f"(cell 0, dz={dz[0]*1e6:.1f}µm)")
    print(f"  Bulk-end cell Cl- = {cl_minus_final_arr[-1]*1e3:.4f} mM "
          f"(cell {N_z-1}, z={solver.z_centers[-1]*1e3:.2f}mm)")

    # ----- Final Cl species partitioning -----
    print(f"\n[FINAL Cl species partitioning, bulk-avg]")
    print(f"  {'species':12s} {'n_Cl':>4s} {'concentration':>20s} {'Cl atoms':>15s}")
    print("  " + "-" * 55)
    cl_atoms_breakdown = {}
    for sp, n in sorted(CL_ATOM_COUNT.items()):
        if sp in idx:
            c_avg = vol_avg(y_final[:, idx[sp]], dz, L)
            cl_atoms = n * c_avg
            cl_atoms_breakdown[sp] = cl_atoms
            if c_avg > 1e-15:
                if c_avg > 1e-3:
                    s = f"{c_avg*1e3:.4f} mM"
                elif c_avg > 1e-6:
                    s = f"{c_avg*1e6:.4f} µM"
                elif c_avg > 1e-9:
                    s = f"{c_avg*1e9:.4f} nM"
                elif c_avg > 1e-12:
                    s = f"{c_avg*1e12:.4f} pM"
                else:
                    s = f"{c_avg:.3e} M"
                cl_atom_str = (f"{cl_atoms*1e3:.4f} mM" if cl_atoms > 1e-3
                               else f"{cl_atoms*1e6:.3e} µM")
                print(f"  {sp:12s} {n:>4d} {s:>20s} {cl_atom_str:>15s}")

    # Sum
    cl_in_minus = cl_atoms_breakdown.get('Cl-', 0)
    cl_in_others = sum(v for k, v in cl_atoms_breakdown.items() if k != 'Cl-')
    print("  " + "-" * 55)
    print(f"  {'Cl- only':12s} {'':>4s} "
          f"{cl_in_minus*1e3:>20.4f} mM")
    print(f"  {'all OTHERS':12s} {'':>4s} "
          f"{cl_in_others*1e3:>20.6f} mM  "
          f"({cl_in_others/cl_total_final_M*100:.4f}% of total Cl)")

    # ----- Time evolution of Cl- volume-avg -----
    print(f"\n[Cl- and Cl atom time evolution]")
    print(f"  {'t (s)':>8s} {'Cl- avg (mM)':>15s} {'total Cl (mM)':>16s} "
          f"{'drift % vs t=0':>15s}")
    sol_t = np.array(result.get('t_eval', [t_end]))
    sol_y_list = result.get('y_eval', None)
    if not sol_y_list:
        print(f"  {'600':>8s} {cl_minus_final*1e3:>15.4f} "
              f"{cl_total_final_M*1e3:>16.4f} "
              f"{(cl_total_final_M/cl_total_init_M-1)*100:>14.4f}%")
    else:
        for k, t in enumerate(sol_t):
            y_flat_k = sol_y_list[k] if k < len(sol_y_list) else sol_y_list[-1]
            y_k = np.asarray(y_flat_k).reshape(N_z, solver.N_s)
            cl_minus_k = vol_avg(y_k[:, idx['Cl-']], dz, L)
            cl_total_k = vol_avg(total_cl_per_cell(y_k, idx), dz, L)
            print(f"  {t:>8.1f} {cl_minus_k*1e3:>15.4f} "
                  f"{cl_total_k*1e3:>16.4f} "
                  f"{(cl_total_k/cl_total_init_M-1)*100:>14.4f}%")

    # ----- Cl chemistry activation summary -----
    print(f"\n[Cl chemistry activation gauge]")
    cl_redistributed = cl_in_others
    activation_pct = cl_redistributed / cl_total_init_M * 100
    print(f"  Initial Cl atoms (all in Cl-)     : {cl_total_init_M*1e3:.4f} mM")
    print(f"  Final Cl atoms in transient species: "
          f"{cl_redistributed*1e3:.6f} mM ({activation_pct:.4f}%)")
    print(f"  → Only {activation_pct:.4f}% of Cl was 'activated' "
          f"(converted to non-Cl- transient species)")
    print(f"  → Remaining {100-activation_pct:.4f}% stays as Cl- ion")

    # ----- Pre/post visual summary —----
    print(f"\n[Per-cell Cl- profile (first 10 + last 5 cells)]")
    print(f"  {'cell':>5s} {'z (mm)':>10s} {'Cl- init':>12s} {'Cl- final':>14s} "
          f"{'Δ µM':>10s}")
    show_cells = list(range(min(10, N_z))) + (
        [N_z - 1] if N_z > 10 else [])
    for j in show_cells:
        z_mm = solver.z_centers[j] * 1e3
        c_i = y0_2d[j, idx['Cl-']] * 1e3
        c_f = y_final[j, idx['Cl-']] * 1e3
        d = (c_f - c_i) * 1e3
        print(f"  {j:>5d} {z_mm:>10.4f} {c_i:>12.4f} {c_f:>14.4f} {d:>+10.2f}")


if __name__ == '__main__':
    main()
