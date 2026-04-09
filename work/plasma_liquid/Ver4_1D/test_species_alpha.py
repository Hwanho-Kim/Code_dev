#!/usr/bin/env python3
"""
Compare single α_b vs species-specific α_b.

Case 1: uniform α_b = 0.03 (baseline)
Case 2: species-specific α_b from literature

Run:
    .venv/bin/python Ver4_1D/test_species_alpha.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_1d import PHYSICAL, N2O4_EQ, MASS_TRANSFER
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    Path(__file__).parent.parent
    / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

SPECIES_ALPHA = {
    'N2O5':  0.03,
    'O3':    0.05,
    'H2O2':  0.1,
    'NO':    0.001,
    'NO2':   0.03,
    'NO3':   0.03,
    'N2O4':  0.03,
    'HONO':  0.05,
    'HONO2': 0.07,
}


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


def run_case(times, gas_conc, alpha_b_uniform=None):
    """Run DIW case. alpha_b_uniform=float → uniform, None → species-specific."""
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='film_alpha',
        alpha_b=alpha_b_uniform,
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


def print_result(label, result, solver, wall):
    avg = result['spatial_avg']
    sfc = result.get('surface', {})
    print(f"\n  [{label}]")
    print(f"  N_z={solver.N_z}, wall={wall:.1f}s ({wall/60:.1f}min)")
    print(f"  pH_avg    = {result['pH_avg']:.3f}")
    print(f"  pH_sfc    = {result.get('pH_surface', 0):.3f}")
    print(f"  NO3⁻      = {avg.get('NO3-', 0)*1e6:.1f} µM  (exp: 63)")
    print(f"  NO2⁻      = {avg.get('NO2-', 0)*1e6:.2f} µM  (exp: 3)")
    print(f"  H2O2      = {avg.get('H2O2', 0)*1e6:.4f} µM  (exp: 11)")
    print(f"  O3        = {avg.get('O3', 0)*1e9:.1f} nM")
    print(f"  OH        = {avg.get('OH', 0)*1e12:.1f} pM")
    print(f"  HO2       = {avg.get('HO2', 0)*1e12:.1f} pM")
    print(f"  O3_sfc    = {sfc.get('O3', 0)*1e6:.3f} µM")
    print(f"  OH_sfc    = {sfc.get('OH', 0)*1e9:.3f} nM")


def main():
    times, gas_conc = load_gas_data(DEFAULT_CSV)

    print("=" * 70)
    print("SPECIES-SPECIFIC α_b TEST — DIW, Film+α_b, measured species only")
    print("=" * 70)

    # Print alpha_b comparison
    print("\n  [α_b values]")
    print(f"  {'Species':>8s}  {'Uniform':>8s}  {'Per-species':>11s}  {'Ratio':>6s}")
    print("  " + "─" * 40)
    for sp in ['N2O5', 'O3', 'H2O2', 'NO', 'NO2', 'NO3', 'N2O4', 'HONO', 'HONO2']:
        uniform = 0.03
        specific = SPECIES_ALPHA.get(sp, 0.03)
        ratio = specific / uniform
        print(f"  {sp:>8s}  {uniform:8.3f}  {specific:11.3f}  {ratio:6.1f}x")

    # Case 1: uniform
    print("\n" + "=" * 70)
    print("  Case 1: Uniform α_b = 0.03")
    print("=" * 70)
    r1, s1, w1 = run_case(times, gas_conc, alpha_b_uniform=0.03)
    print_result("Uniform α_b=0.03", r1, s1, w1)

    # Case 2: species-specific (alpha_b=None → uses config per-species)
    print("\n" + "=" * 70)
    print("  Case 2: Species-specific α_b")
    print("=" * 70)
    r2, s2, w2 = run_case(times, gas_conc, alpha_b_uniform=None)
    print_result("Species-specific α_b", r2, s2, w2)

    # Comparison
    a1, a2 = r1['spatial_avg'], r2['spatial_avg']
    print("\n" + "=" * 70)
    print("  COMPARISON (species-specific vs uniform)")
    print("=" * 70)
    print(f"  {'Metric':>12s}  {'Uniform':>10s}  {'Per-species':>11s}  {'Change':>10s}")
    print("  " + "─" * 50)
    metrics = [
        ('pH_avg', r1['pH_avg'], r2['pH_avg'], ''),
        ('pH_sfc', r1.get('pH_surface', 0), r2.get('pH_surface', 0), ''),
        ('NO3⁻ µM', a1.get('NO3-', 0)*1e6, a2.get('NO3-', 0)*1e6, ''),
        ('H2O2 µM', a1.get('H2O2', 0)*1e6, a2.get('H2O2', 0)*1e6, ''),
        ('O3 nM', a1.get('O3', 0)*1e9, a2.get('O3', 0)*1e9, ''),
        ('OH pM', a1.get('OH', 0)*1e12, a2.get('OH', 0)*1e12, ''),
        ('HO2 pM', a1.get('HO2', 0)*1e12, a2.get('HO2', 0)*1e12, ''),
    ]
    for name, v1, v2, _ in metrics:
        if abs(v1) > 1e-15:
            pct = (v2 - v1) / abs(v1) * 100
            print(f"  {name:>12s}  {v1:10.4f}  {v2:11.4f}  {pct:+9.1f}%")
        else:
            print(f"  {name:>12s}  {v1:10.4f}  {v2:11.4f}  {'N/A':>10s}")

    print(f"\n  Experimental: pH=3.61, NO3⁻=63µM, NO2⁻=3µM, H2O2=11µM")
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
