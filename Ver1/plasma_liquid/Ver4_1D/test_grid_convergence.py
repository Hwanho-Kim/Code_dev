#!/usr/bin/env python3
"""
Grid convergence test: vary dz_min (surface resolution) with fixed stretch_ratio.

Checks whether bulk-averaged and surface concentrations are converged
with respect to the near-surface grid spacing.

O3 reactive penetration depth ≈ 34 µm → need sufficient cells within this zone.

Cases:
  dz_min = 20, 10, 5, 2, 1 µm  (stretch_ratio=1.12, L=10mm)
  → N_z ≈ 37, 43, 49, 57, 63 cells

Run:
    Ver3/.venv/bin/python Ver4_1D/test_grid_convergence.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_1d import (
    PHYSICAL, MASS_TRANSFER, GAS_TO_AQUEOUS_MAP,
    N2O4_EQ,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    Path(__file__).parent.parent
    / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

# Grid cases: dz_min in meters
GRID_CASES = [20e-6, 10e-6, 5e-6, 2e-6, 1e-6]
STRETCH_RATIO = 1.12
ALPHA_B = 0.03


def load_gas_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float) * 2.0

    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))

    if 'N2O4' not in df.columns or np.all(gas_conc['N2O4'] == 0):
        import math
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298) +
            (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        factor = PHYSICAL.KB_T_OVER_P * T
        gas_conc['N2O4'] = Kp * factor * (no2 ** 2)

    return times, gas_conc


def run_case(dz_min, times, gas_conc):
    """Run one DIW case with given dz_min."""
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=dz_min,
        stretch_ratio=STRETCH_RATIO,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='film_alpha',
        alpha_b=ALPHA_B,
    )
    solver.set_gas_data(
        times=times,
        gas_conc_molecules=gas_conc,
        hono_gas=0,
        hono2_gas=0,
        h2o2_gas=0,
    )
    t_end = float(times[-1])

    t0 = time.time()
    result = solver.solve(
        t_span=(0, t_end),
        t_eval=np.array([0, t_end]),
        verbose=True,
        dt_poisson=None,  # single BDF (DIW)
    )
    wall = time.time() - t0

    return result, solver, wall


def count_cells_in_zone(solver, depth_m):
    """Count how many cells have their center within `depth_m` of the surface."""
    return int(np.sum(solver.z_centers < depth_m))


def main():
    times, gas_conc = load_gas_data(DEFAULT_CSV)

    print("=" * 80)
    print("GRID CONVERGENCE TEST — DIW, Film+α_b=0.03, measured species only")
    print("=" * 80)
    print(f"  stretch_ratio = {STRETCH_RATIO}")
    print(f"  α_b = {ALPHA_B}")
    print(f"  t_end = {times[-1]:.0f}s")
    print(f"  O3 penetration depth ≈ 34 µm")
    print()

    results = {}

    for dz_min in GRID_CASES:
        tag = f"dz_min={dz_min*1e6:.0f}µm"
        print("=" * 80)
        print(f"  Case: {tag}")
        print("=" * 80)

        result, solver, wall = run_case(dz_min, times, gas_conc)

        n_cells = solver.N_z
        n_in_34um = count_cells_in_zone(solver, 34e-6)
        n_in_100um = count_cells_in_zone(solver, 100e-6)

        avg = result['spatial_avg']
        sfc = result.get('surface', {})

        info = {
            'dz_min_um': dz_min * 1e6,
            'N_z': n_cells,
            'cells_in_34um': n_in_34um,
            'cells_in_100um': n_in_100um,
            'dz_min_actual': solver.dz_cells[0] * 1e6,
            'dz_max': solver.dz_cells[-1] * 1e6,
            'wall_s': wall,
            'success': result['success'],
            # Bulk averages
            'pH': result['pH_avg'],
            'NO3_uM': avg.get('NO3-', 0) * 1e6,
            'NO2_uM': avg.get('NO2-', 0) * 1e6,
            'H2O2_uM': avg.get('H2O2', 0) * 1e6,
            'O3_nM': avg.get('O3', 0) * 1e9,
            'OH_pM': avg.get('OH', 0) * 1e12,
            'HO2_pM': avg.get('HO2', 0) * 1e12,
            # Surface values
            'pH_sfc': result.get('pH_surface', 0),
            'O3_sfc_uM': sfc.get('O3', 0) * 1e6,
            'OH_sfc_nM': sfc.get('OH', 0) * 1e9,
            'NO3_sfc_uM': sfc.get('NO3-', 0) * 1e6,
        }
        results[dz_min] = info

        print(f"  → N_z={n_cells}, cells in 34µm zone={n_in_34um}, "
              f"cells in 100µm={n_in_100um}")
        print(f"  → pH={info['pH']:.3f}, NO3⁻={info['NO3_uM']:.1f}µM, "
              f"O3={info['O3_nM']:.1f}nM, wall={wall:.1f}s")
        print()

    # ---- Summary table ----
    print()
    print("=" * 80)
    print("GRID CONVERGENCE SUMMARY")
    print("=" * 80)

    # Table 1: Grid info
    print("\n  [Grid Structure]")
    h1 = f"  {'dz_min(µm)':>10s}  {'N_z':>5s}  {'34µm':>5s}  {'100µm':>6s}  {'dz_max(µm)':>10s}  {'Time(s)':>8s}"
    print(h1)
    print("  " + "─" * (len(h1) - 2))
    for dz in GRID_CASES:
        r = results[dz]
        print(f"  {r['dz_min_um']:10.0f}  {r['N_z']:5d}  {r['cells_in_34um']:5d}  "
              f"{r['cells_in_100um']:6d}  {r['dz_max']:10.0f}  {r['wall_s']:8.1f}")

    # Table 2: Bulk averages
    print("\n  [Bulk-Averaged Results]")
    h2 = f"  {'dz_min(µm)':>10s}  {'pH':>6s}  {'NO3⁻(µM)':>10s}  {'O3(nM)':>8s}  {'OH(pM)':>8s}  {'HO2(pM)':>9s}  {'H2O2(µM)':>9s}"
    print(h2)
    print("  " + "─" * (len(h2) - 2))
    for dz in GRID_CASES:
        r = results[dz]
        print(f"  {r['dz_min_um']:10.0f}  {r['pH']:6.3f}  {r['NO3_uM']:10.1f}  "
              f"{r['O3_nM']:8.1f}  {r['OH_pM']:8.1f}  {r['HO2_pM']:9.1f}  {r['H2O2_uM']:9.4f}")
    print(f"  {'실험':>10s}  {3.61:6.2f}  {63.0:10.1f}  {'–':>8s}  {'–':>8s}  {'–':>9s}  {11.0:9.2f}")

    # Table 3: Surface values
    print("\n  [Surface (z=0) Values]")
    h3 = f"  {'dz_min(µm)':>10s}  {'pH_sfc':>7s}  {'O3_sfc(µM)':>11s}  {'OH_sfc(nM)':>11s}  {'NO3⁻_sfc(µM)':>13s}"
    print(h3)
    print("  " + "─" * (len(h3) - 2))
    for dz in GRID_CASES:
        r = results[dz]
        print(f"  {r['dz_min_um']:10.0f}  {r['pH_sfc']:7.3f}  {r['O3_sfc_uM']:11.3f}  "
              f"{r['OH_sfc_nM']:11.3f}  {r['NO3_sfc_uM']:13.1f}")

    # Convergence analysis: relative change vs finest grid
    print("\n  [Convergence vs Finest Grid (dz_min=1µm)]")
    ref = results[1e-6]
    h4 = f"  {'dz_min(µm)':>10s}  {'ΔpH':>8s}  {'ΔNO3⁻(%)':>10s}  {'ΔO3(%)':>8s}  {'ΔOH(%)':>8s}"
    print(h4)
    print("  " + "─" * (len(h4) - 2))
    for dz in GRID_CASES:
        r = results[dz]
        d_pH = r['pH'] - ref['pH']
        d_NO3 = (r['NO3_uM'] - ref['NO3_uM']) / max(ref['NO3_uM'], 1e-10) * 100
        d_O3 = (r['O3_nM'] - ref['O3_nM']) / max(ref['O3_nM'], 1e-10) * 100
        d_OH = (r['OH_pM'] - ref['OH_pM']) / max(ref['OH_pM'], 1e-10) * 100
        mark = " ←ref" if dz == 1e-6 else ""
        print(f"  {r['dz_min_um']:10.0f}  {d_pH:+8.4f}  {d_NO3:+10.1f}  "
              f"{d_O3:+8.1f}  {d_OH:+8.1f}{mark}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
