"""
N-atom budget diagnostic for NO3⁻ overproduction analysis.

Calculates species-wise N influx through the gas-liquid interface
using actual CSV data over 720s, and δ_gas/δ_liq sensitivity.

Usage:
    Ver3/.venv/bin/python Ver4_1D/diagnose_n_budget.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- project imports --
sys.path.insert(0, str(Path(__file__).parent))
from config_1d import (
    HENRY_CONSTANTS,
    GAS_DIFFUSIVITY,
    LIQUID_DIFFUSIVITY,
    D_GAS_DEFAULT,
    D_LIQ_DEFAULT,
    PHYSICAL,
    N2O4_EQ,
)
from pde_solver import compute_k_mt

# =====================================================================
# Constants
# =====================================================================

# N atoms per dissolved molecule
N_ATOMS = {
    'NO':   1,
    'NO2':  1,
    'NO3':  1,
    'N2O4': 2,
    'N2O5': 2,
    'HONO': 1,
    'HONO2': 1,
}

# cm⁻³ → mol/L conversion
CONV = 1000.0 / PHYSICAL.AVOGADRO  # 1.66e-21

# Experimental target
EXP_NO3_UM = 102.0   # µM (saline)
T_END = 720.0         # seconds

CSV_PATH = Path(__file__).parent.parent / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'


def load_gas_data(csv_path: Path):
    """Load gas-phase data and compute N2O4 from equilibrium if missing."""
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float) * 2.0  # 2-second intervals

    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5', 'HONO', 'HONO2']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))

    # N2O4 from equilibrium if not in CSV
    if 'N2O4' not in df.columns or np.all(gas_conc['N2O4'] == 0):
        import math
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298)
            + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        factor = PHYSICAL.KB_T_OVER_P * T
        gas_conc['N2O4'] = Kp * factor * (no2 ** 2)

    return times, gas_conc


def compute_species_budget(times, gas_conc, delta_gas, delta_liq):
    """
    For each N-bearing species, compute:
      C_eq(t) = H × C_gas(t) × CONV
      flux(t) = k_mt × C_eq(t)   (assume C_surface ≈ 0 for fast-reacting species)
      N_total = n_N × ∫ flux(t) dt  [mol/L·s → mol/L over integration]

    Returns dict of {species: {k_mt, avg_ceq, total_N_mol_L, total_N_uM}}
    """
    dt = times[1] - times[0] if len(times) > 1 else 2.0
    results = {}

    for sp, n_N in N_ATOMS.items():
        H = HENRY_CONSTANTS.get(sp, 1.0)
        k_mt = compute_k_mt(sp, delta_gas, delta_liq)

        # gas conc in mol/L
        c_gas_molar = gas_conc.get(sp, np.zeros_like(times)) * CONV

        # C_eq = H × C_gas_molar
        c_eq = H * c_gas_molar

        # flux = k_mt × C_eq  [m/s × mol/L = mol/(L·s) per unit interface...
        # but in PDE the BC is: dC/dt(j=0) += (k_mt/dz) × (C_eq - C)
        # For budget, total dissolved = ∫ k_mt × C_eq dt  [m/s × mol/L × s = m × mol/L]
        # Then divide by liquid_depth to get average concentration increase:
        # ΔC_avg = (1/L) × ∫ k_mt × C_eq dt

        # Actually simpler: total moles entering per unit area = k_mt × ∫ C_eq dt [mol/m³ · m/s · s = mol/m²]
        # Spread over liquid depth L:
        # ΔC = k_mt × ∫ C_eq dt / L  ... but C_eq is in mol/L = 1000 mol/m³
        # Let's just work in mol/L consistently.
        # flux = k_mt [m/s] × (C_eq [mol/L] × 1000 [L/m³]) = mol/(m²·s)
        # total = ∫ flux dt / L / 1000 = mol/L

        liquid_depth = 0.01  # 10 mm
        flux = k_mt * c_eq * 1000.0  # mol/(m²·s)
        total_mol_m2 = np.trapezoid(flux, times)  # mol/m²
        total_mol_L = total_mol_m2 / liquid_depth / 1000.0  # mol/L

        total_N_mol_L = n_N * total_mol_L

        # Also compute gas-side limited flux for comparison
        D_g = GAS_DIFFUSIVITY.get(sp, D_GAS_DEFAULT)
        k_gas_only = D_g / delta_gas  # gas-side only [m/s]

        results[sp] = {
            'n_N': n_N,
            'H': H,
            'k_mt': k_mt,
            'k_gas_only': k_gas_only,
            'ratio_k': k_mt / k_gas_only if k_gas_only > 0 else 0,
            'avg_ceq_uM': np.mean(c_eq) * 1e6,
            'max_ceq_uM': np.max(c_eq) * 1e6,
            'avg_cgas_cm3': np.mean(gas_conc.get(sp, np.zeros_like(times))),
            'total_species_uM': total_mol_L * 1e6,
            'total_N_uM': total_N_mol_L * 1e6,
        }

    return results


def print_budget(results, delta_gas, delta_liq):
    """Print species N budget table."""
    print(f"\n{'='*90}")
    print(f"  N-atom Budget (δ_gas={delta_gas*1e3:.1f}mm, δ_liq={delta_liq*1e6:.0f}µm, t={T_END:.0f}s)")
    print(f"{'='*90}")

    # Header
    print(f"  {'Species':<8} {'n_N':>3} {'H_cc':>10} {'k_mt':>10} {'k_gas':>10} "
          f"{'k/k_gas':>7} {'<C_eq>':>10} {'ΔC':>10} {'N_in':>10} {'%':>6}")
    print(f"  {'':<8} {'':>3} {'':>10} {'[m/s]':>10} {'[m/s]':>10} "
          f"{'':>7} {'[µM]':>10} {'[µM]':>10} {'[µM]':>10} {'':>6}")
    print(f"  {'-'*86}")

    total_N = sum(r['total_N_uM'] for r in results.values())

    # Sort by N contribution
    sorted_sp = sorted(results.keys(), key=lambda s: results[s]['total_N_uM'], reverse=True)

    for sp in sorted_sp:
        r = results[sp]
        pct = r['total_N_uM'] / total_N * 100 if total_N > 0 else 0
        print(f"  {sp:<8} {r['n_N']:>3} {r['H']:>10.2e} {r['k_mt']:>10.2e} {r['k_gas_only']:>10.2e} "
              f"{r['ratio_k']:>7.4f} {r['avg_ceq_uM']:>10.1f} {r['total_species_uM']:>10.1f} "
              f"{r['total_N_uM']:>10.1f} {pct:>5.1f}%")

    print(f"  {'-'*86}")
    print(f"  {'TOTAL':<8} {'':>3} {'':>10} {'':>10} {'':>10} {'':>7} {'':>10} "
          f"{'':>10} {total_N:>10.1f} {'100.0':>5}%")
    print(f"\n  → 만약 모든 N이 NO3⁻로: {total_N:.1f} µM (실험값 {EXP_NO3_UM} µM, {total_N/EXP_NO3_UM:.1f}배)")


def sensitivity_analysis(times, gas_conc):
    """δ_gas, δ_liq sensitivity sweep."""
    print(f"\n{'='*90}")
    print(f"  δ Sensitivity Analysis")
    print(f"{'='*90}")

    delta_gas_values = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]   # 1,5,10,20,50 mm
    delta_liq_values = [1e-6, 1e-5, 1e-4, 1e-3]           # 1,10,100,1000 µm

    # Find the dominant species first (at default δ)
    results_default = compute_species_budget(times, gas_conc, 0.01, 0.0001)
    sorted_sp = sorted(results_default.keys(), key=lambda s: results_default[s]['total_N_uM'], reverse=True)
    top_sp = sorted_sp[0]
    print(f"  Dominant species: {top_sp}")

    # --- δ_gas sweep (fixed δ_liq=100µm) ---
    print(f"\n  δ_gas sweep (δ_liq=100µm fixed):")
    print(f"  {'δ_gas':>10} {'k_mt('+top_sp+')':>14} {'Total N':>10} {'NO3⁻ ratio':>12}")
    for dg in delta_gas_values:
        res = compute_species_budget(times, gas_conc, dg, 1e-4)
        total_N = sum(r['total_N_uM'] for r in res.values())
        print(f"  {dg*1e3:>8.1f}mm {res[top_sp]['k_mt']:>14.3e} {total_N:>10.0f}µM {total_N/EXP_NO3_UM:>10.1f}×")

    # --- δ_liq sweep (fixed δ_gas=10mm) ---
    print(f"\n  δ_liq sweep (δ_gas=10mm fixed):")
    print(f"  {'δ_liq':>10} {'k_mt('+top_sp+')':>14} {'Total N':>10} {'NO3⁻ ratio':>12}")
    for dl in delta_liq_values:
        res = compute_species_budget(times, gas_conc, 1e-2, dl)
        total_N = sum(r['total_N_uM'] for r in res.values())
        print(f"  {dl*1e6:>8.0f}µm {res[top_sp]['k_mt']:>14.3e} {total_N:>10.0f}µM {total_N/EXP_NO3_UM:>10.1f}×")

    # --- Combined: what δ_gas gets us to ~102 µM? ---
    print(f"\n  Target: NO3⁻ ≈ {EXP_NO3_UM} µM")
    print(f"  Searching δ_gas that gives total N ≈ {EXP_NO3_UM} µM (δ_liq=100µm)...")
    from scipy.optimize import brentq

    def residual(log_dg):
        dg = 10**log_dg
        res = compute_species_budget(times, gas_conc, dg, 1e-4)
        total_N = sum(r['total_N_uM'] for r in res.values())
        return total_N - EXP_NO3_UM

    # Check if target is bracketed
    low = residual(np.log10(1e-3))
    high = residual(np.log10(1.0))
    if low * high < 0:
        log_dg_opt = brentq(residual, np.log10(1e-3), np.log10(1.0))
        dg_opt = 10**log_dg_opt
        print(f"  → δ_gas = {dg_opt*1e3:.1f} mm gives ~{EXP_NO3_UM} µM (physical? typically 1-10mm)")
    else:
        print(f"  → No δ_gas in [1mm, 1000mm] gives {EXP_NO3_UM} µM")
        print(f"    δ_gas=1mm → {low + EXP_NO3_UM:.0f} µM, δ_gas=1000mm → {high + EXP_NO3_UM:.0f} µM")


def gas_side_analysis(times, gas_conc):
    """Check if transfer is gas-side limited (k_mt ≈ k_gas)."""
    print(f"\n{'='*90}")
    print(f"  Gas-Side Limitation Check (δ_gas=10mm, δ_liq=100µm)")
    print(f"{'='*90}")
    print(f"  If k_mt/k_gas ≈ 1.0, species is gas-side limited → changing H or δ_liq has no effect")
    print(f"  If k_mt/k_gas << 1.0, species is liquid-side limited → H matters")
    print()

    for sp in N_ATOMS:
        H = HENRY_CONSTANTS.get(sp, 1.0)
        D_g = GAS_DIFFUSIVITY.get(sp, D_GAS_DEFAULT)
        D_l = LIQUID_DIFFUSIVITY.get(sp, D_LIQ_DEFAULT)
        k_mt = compute_k_mt(sp, 0.01, 0.0001)
        k_gas = D_g / 0.01

        # Resistance analysis: 1/k_mt = δ_g/(D_g) + H×δ_l/(D_l)
        R_gas = 0.01 / D_g
        R_liq = H * 0.0001 / D_l
        R_total = R_gas + R_liq
        frac_gas = R_gas / R_total * 100
        frac_liq = R_liq / R_total * 100

        limiting = "GAS-LIMITED" if frac_gas > 90 else ("LIQ-LIMITED" if frac_liq > 90 else "MIXED")
        print(f"  {sp:<8}: R_gas={R_gas:.2e}s/m  R_liq={R_liq:.2e}s/m  "
              f"gas%={frac_gas:5.1f}%  liq%={frac_liq:5.1f}%  → {limiting}")


def reaction_check():
    """Verify N2O5 and related reaction rates."""
    print(f"\n{'='*90}")
    print(f"  N2O5 Reaction Check")
    print(f"{'='*90}")

    reactions = [
        ('R95', 'N2O4 + H2O → NO2⁻ + NO3⁻ + 2H⁺', 1e3, 'hydrolysis'),
        ('R98', 'N2O5 + H2O → 2NO3⁻ + 2H⁺', 5e9, 'hydrolysis (instant)'),
        ('R94', 'N2O3 + H2O → 2NO2⁻ + 2H⁺', 5.3e5, 'hydrolysis'),
        ('R102', 'NO2 + NO3 → N2O5', 1.7e9, 'aq. formation'),
    ]

    print(f"  {'ID':<6} {'Reaction':<35} {'k':>10} {'Note':<25}")
    print(f"  {'-'*80}")
    for rid, rxn, k, note in reactions:
        t_half = 0.693 / k if k > 0 else float('inf')
        print(f"  {rid:<6} {rxn:<35} {k:>10.1e} {note:<25} t½={t_half:.1e}s")

    print(f"\n  N2O5 t½=0.14ns → mass transfer limited (hydrolysis rate irrelevant)")
    print(f"  Key question: is N2O5 gas-phase concentration / mass transfer rate correct?")


def main():
    print("=" * 90)
    print("  NO3⁻ Overproduction Diagnostic — N-atom Budget Analysis")
    print("=" * 90)

    times, gas_conc = load_gas_data(CSV_PATH)
    # Trim to T_END
    mask = times <= T_END
    times = times[mask]
    for sp in gas_conc:
        gas_conc[sp] = gas_conc[sp][:len(times)]

    print(f"  CSV: {len(times)} points, t=[0, {times[-1]:.0f}]s")
    print(f"  Experimental NO3⁻ = {EXP_NO3_UM} µM (saline)")

    # --- Step 1: Species N budget ---
    results = compute_species_budget(times, gas_conc, 0.01, 0.0001)
    print_budget(results, 0.01, 0.0001)

    # --- Step 2: Gas-side limitation ---
    gas_side_analysis(times, gas_conc)

    # --- Step 3: δ sensitivity ---
    sensitivity_analysis(times, gas_conc)

    # --- Step 4: Reaction check ---
    reaction_check()

    # --- Summary ---
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    total_N = sum(r['total_N_uM'] for r in results.values())
    sorted_sp = sorted(results.keys(), key=lambda s: results[s]['total_N_uM'], reverse=True)
    print(f"  Total N influx: {total_N:.0f} µM (실험 {EXP_NO3_UM} µM, {total_N/EXP_NO3_UM:.0f}×)")
    print(f"  Top contributors:")
    for i, sp in enumerate(sorted_sp[:3]):
        r = results[sp]
        pct = r['total_N_uM'] / total_N * 100
        print(f"    {i+1}. {sp}: {r['total_N_uM']:.0f} µM ({pct:.1f}%)")

    # Actionable insights
    top = results[sorted_sp[0]]
    if top['ratio_k'] > 0.9:
        print(f"\n  ⚠ {sorted_sp[0]}은 gas-side limited → H 변경 무의미, δ_gas가 핵심 파라미터")
    else:
        print(f"\n  ⚠ {sorted_sp[0]}은 liquid-side limited → H 값 검증 필요")


if __name__ == '__main__':
    main()
