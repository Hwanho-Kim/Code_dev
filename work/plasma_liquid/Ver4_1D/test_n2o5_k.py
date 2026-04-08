#!/usr/bin/env python3
"""Quick test: N2O5 hydrolysis k=5e9 vs k=1e7 comparison (DIW, α_b=0.03)."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_1d import PHYSICAL, MASS_TRANSFER, GRID
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    Path(__file__).parent.parent
    / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)


def load_gas_data(csv_path):
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))
    if 'N2O4' not in df.columns or np.all(gas_conc['N2O4'] == 0):
        from config_1d import N2O4_EQ, PHYSICAL as P
        import math
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298) +
            (N2O4_EQ.DELTA_H / P.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        factor = P.KB_T_OVER_P * T
        gas_conc['N2O4'] = Kp * factor * (no2 ** 2)
    return times, gas_conc


def run_case(k_n2o5, alpha_b, times, gas_conc):
    """Run DIW case, patching R98 k value."""
    chem = AqueousChemistry1D(saline_mode=False)

    # Patch R98 k value
    for i, rxn in enumerate(chem.reactions):
        label = rxn.get('label', '')
        if 'R98' in label and 'N2O5' in label:
            old_k = rxn['k']
            rxn['k'] = k_n2o5
            print(f"  Patched R98: k={float(old_k):.1e} → {k_n2o5:.1e}")
            break

    # Re-precompute after patching
    chem._precompute_reaction_data()
    chem._precompute_numba_arrays()

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.02,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='film_alpha',
        alpha_b=alpha_b,
    )
    solver.set_gas_data(
        times=times,
        gas_conc_molecules=gas_conc,
        hono_gas=0, hono2_gas=0, h2o2_gas=0,
    )
    t_end = float(times[-1])

    t0 = time.time()
    result = solver.solve(
        t_span=(0, t_end),
        t_eval=np.array([0, t_end / 4, t_end / 2, 3 * t_end / 4, t_end]),
        verbose=True,
        dt_poisson=10.0,
    )
    wall = time.time() - t0
    return result, wall


def main():
    times, gas_conc = load_gas_data(DEFAULT_CSV)
    alpha_b = 0.03

    k_cases = [5.0e9, 4.0e7, 1.0e7]

    print("=" * 70)
    print("N2O5 HYDROLYSIS k SENSITIVITY TEST")
    print(f"  α_b={alpha_b}, DIW, Film+α_b BC")
    print(f"  k cases: {[f'{k:.0e}' for k in k_cases]}")
    print("=" * 70)

    results = {}
    for k_val in k_cases:
        print(f"\n{'='*70}")
        print(f"  Running k(R98) = {k_val:.1e}")
        print(f"{'='*70}")
        result, wall = run_case(k_val, alpha_b, times, gas_conc)
        avg = result['spatial_avg']

        pH = result['pH_avg']
        no3 = avg.get('NO3-', 0) * 1e6 + avg.get('HONO2', 0) * 1e6
        no2 = avg.get('NO2-', 0) * 1e6 + avg.get('HONO', 0) * 1e6
        h2o2 = avg.get('H2O2', 0) * 1e6 + avg.get('HO2-', 0) * 1e6
        n2o5 = avg.get('N2O5', 0)
        o3 = avg.get('O3', 0)

        results[k_val] = {
            'pH': pH, 'NO3-': no3, 'NO2-': no2, 'H2O2': h2o2,
            'N2O5_avg': n2o5, 'O3_avg': o3, 'wall': wall,
        }
        print(f"  → pH={pH:.3f}, NO3⁻={no3:.1f}µM, H2O2={h2o2:.2f}µM, "
              f"N2O5={n2o5:.3e}M, time={wall:.0f}s")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'k(R98)':>10s}  {'pH':>6s}  {'NO3⁻(µM)':>10s}  {'H2O2(µM)':>9s}  {'N2O5(M)':>10s}  {'O3(M)':>10s}  {'Time':>6s}"
    print(header)
    print("─" * len(header))
    for k_val in k_cases:
        r = results[k_val]
        print(f"{k_val:10.1e}  {r['pH']:6.3f}  {r['NO3-']:10.1f}  "
              f"{r['H2O2']:9.2f}  {r['N2O5_avg']:10.3e}  {r['O3_avg']:10.3e}  "
              f"{r['wall']/60:5.1f}m")
    print("─" * len(header))
    print(f"{'실험':>10s}  {3.61:6.2f}  {63.0:10.1f}  {11.0:9.2f}")


if __name__ == '__main__':
    main()
