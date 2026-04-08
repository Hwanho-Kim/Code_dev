#!/usr/bin/env python3
"""
Grid convergence test #2: vary stretch_ratio with fixed dz_min=2µm.

Isolates the effect of bulk resolution (diffusion zone ~1.2mm)
from surface resolution (reaction zone ~200µm).

Cases:
  ratio = 1.02, 1.05, 1.08, 1.12  (dz_min=2µm, L=10mm)

Run:
    Ver3/.venv/bin/python Ver4_1D/test_grid_convergence2.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_1d import (
    PHYSICAL, N2O4_EQ,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    Path(__file__).parent.parent
    / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

DZ_MIN = 5e-6  # 5 µm (matches baseline)
RATIO_CASES = [1.02, 1.04, 1.06, 1.08, 1.12]
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


def run_case(ratio, times, gas_conc):
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=DZ_MIN,
        stretch_ratio=ratio,
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
        dt_poisson=None,
    )
    wall = time.time() - t0

    return result, solver, wall


def cells_in_zone(solver, depth_m):
    return int(np.sum(solver.z_centers < depth_m))


def dz_at_depth(solver, depth_m):
    """Cell size at given depth."""
    idx = np.searchsorted(solver.z_centers, depth_m)
    idx = min(idx, solver.N_z - 1)
    return solver.dz_cells[idx]


def main():
    times, gas_conc = load_gas_data(DEFAULT_CSV)

    print("=" * 85)
    print("GRID CONVERGENCE #2 — vary stretch_ratio, dz_min=2µm fixed")
    print("=" * 85)
    print(f"  dz_min = {DZ_MIN*1e6:.0f} µm")
    print(f"  α_b = {ALPHA_B}")
    print(f"  t_end = {times[-1]:.0f}s")
    print()

    results = {}

    for ratio in RATIO_CASES:
        tag = f"ratio={ratio}"
        print("=" * 85)
        print(f"  Case: {tag}")
        print("=" * 85)

        result, solver, wall = run_case(ratio, times, gas_conc)

        avg = result['spatial_avg']
        sfc = result.get('surface', {})

        info = {
            'ratio': ratio,
            'N_z': solver.N_z,
            'cells_34um': cells_in_zone(solver, 34e-6),
            'cells_200um': cells_in_zone(solver, 200e-6),
            'cells_1mm': cells_in_zone(solver, 1e-3),
            'cells_2mm': cells_in_zone(solver, 2e-3),
            'dz_at_200um': dz_at_depth(solver, 200e-6) * 1e6,
            'dz_at_1mm': dz_at_depth(solver, 1e-3) * 1e6,
            'dz_at_2mm': dz_at_depth(solver, 2e-3) * 1e6,
            'dz_max': solver.dz_cells[-1] * 1e6,
            'wall_s': wall,
            'success': result['success'],
            'pH': result['pH_avg'],
            'NO3_uM': avg.get('NO3-', 0) * 1e6,
            'NO2_uM': avg.get('NO2-', 0) * 1e6,
            'H2O2_uM': avg.get('H2O2', 0) * 1e6,
            'O3_nM': avg.get('O3', 0) * 1e9,
            'OH_pM': avg.get('OH', 0) * 1e12,
            'HO2_pM': avg.get('HO2', 0) * 1e12,
            'pH_sfc': result.get('pH_surface', 0),
            'O3_sfc_uM': sfc.get('O3', 0) * 1e6,
            'OH_sfc_nM': sfc.get('OH', 0) * 1e9,
            'NO3_sfc_uM': sfc.get('NO3-', 0) * 1e6,
        }
        results[ratio] = info

        print(f"  → N_z={solver.N_z}, pH={info['pH']:.3f}, "
              f"NO3⁻={info['NO3_uM']:.1f}µM, O3={info['O3_nM']:.1f}nM, "
              f"wall={wall:.1f}s")
        print()

    # ---- Summary tables ----
    print()
    print("=" * 85)
    print("SUMMARY")
    print("=" * 85)

    # Grid structure
    print("\n  [Grid Structure]")
    h = (f"  {'ratio':>6s}  {'N_z':>5s}  {'34µm':>5s}  {'200µm':>6s}  "
         f"{'1mm':>5s}  {'2mm':>5s}  "
         f"{'dz@200µm':>9s}  {'dz@1mm':>7s}  {'dz@2mm':>7s}  {'dz_max':>7s}  {'Time':>6s}")
    print(h)
    print("  " + "─" * (len(h) - 2))
    for r in RATIO_CASES:
        d = results[r]
        print(f"  {d['ratio']:6.2f}  {d['N_z']:5d}  {d['cells_34um']:5d}  "
              f"{d['cells_200um']:6d}  {d['cells_1mm']:5d}  {d['cells_2mm']:5d}  "
              f"{d['dz_at_200um']:8.1f}µ  {d['dz_at_1mm']:6.1f}µ  "
              f"{d['dz_at_2mm']:6.1f}µ  {d['dz_max']:6.0f}µ  {d['wall_s']/60:5.1f}m")

    # Bulk results
    print("\n  [Bulk-Averaged Results]")
    h2 = f"  {'ratio':>6s}  {'pH':>6s}  {'NO3⁻(µM)':>10s}  {'O3(nM)':>8s}  {'OH(pM)':>8s}  {'HO2(pM)':>9s}"
    print(h2)
    print("  " + "─" * (len(h2) - 2))
    for r in RATIO_CASES:
        d = results[r]
        print(f"  {d['ratio']:6.2f}  {d['pH']:6.3f}  {d['NO3_uM']:10.1f}  "
              f"{d['O3_nM']:8.1f}  {d['OH_pM']:8.1f}  {d['HO2_pM']:9.1f}")
    print(f"  {'실험':>6s}  {3.61:6.2f}  {63.0:10.1f}  {'–':>8s}  {'–':>8s}  {'–':>9s}")

    # Surface values
    print("\n  [Surface (z=0)]")
    h3 = f"  {'ratio':>6s}  {'pH_sfc':>7s}  {'O3(µM)':>8s}  {'OH(nM)':>8s}  {'NO3⁻(µM)':>10s}"
    print(h3)
    print("  " + "─" * (len(h3) - 2))
    for r in RATIO_CASES:
        d = results[r]
        print(f"  {d['ratio']:6.2f}  {d['pH_sfc']:7.3f}  {d['O3_sfc_uM']:8.3f}  "
              f"{d['OH_sfc_nM']:8.3f}  {d['NO3_sfc_uM']:10.1f}")

    # Convergence vs finest (1.02)
    ref = results[1.02]
    print("\n  [Convergence vs ratio=1.02]")
    h4 = f"  {'ratio':>6s}  {'ΔpH':>8s}  {'ΔNO3⁻(%)':>10s}  {'ΔO3(%)':>8s}  {'ΔOH(%)':>8s}"
    print(h4)
    print("  " + "─" * (len(h4) - 2))
    for r in RATIO_CASES:
        d = results[r]
        dp = d['pH'] - ref['pH']
        dn = (d['NO3_uM'] - ref['NO3_uM']) / max(ref['NO3_uM'], 1e-10) * 100
        do = (d['O3_nM'] - ref['O3_nM']) / max(ref['O3_nM'], 1e-10) * 100
        doh = (d['OH_pM'] - ref['OH_pM']) / max(ref['OH_pM'], 1e-10) * 100
        mark = " ←ref" if r == 1.02 else ""
        print(f"  {r:6.2f}  {dp:+8.4f}  {dn:+10.1f}  {do:+8.1f}  {doh:+8.1f}{mark}")

    print()
    print("=" * 85)
    print("DONE")
    print("=" * 85)


if __name__ == '__main__':
    main()
