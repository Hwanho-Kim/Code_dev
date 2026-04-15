#!/usr/bin/env python3
"""
Diagnostic: pH gap and NO2- depletion in 1D validation.

Runs 1D simulations with best 0D-optimized gas inputs and analyzes:
  1. Full charge balance — what anions/cations exist at 12 min?
  2. NO2- rate budget — dominant sinks
  3. HONO2 sensitivity sweep — does adding HONO2 close the pH gap?
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import (PHYSICAL, N2O4_EQ, HENRY_CONSTANTS,
                        ACID_BASE_PAIRS, GAS_TO_AQUEOUS_MAP)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

# Best 0D-optimized parameters
BASE_PARAMS = {
    'hono_gas': 1.401e15,
    'hono2_gas': 2.082e9,
    'h2o2_gas': 2.258e15,
    'delta_gas': 13.1e-3,  # m
}

EXP = {'pH': 3.61, 'NO2-': 3.0, 'NO3-': 63.0, 'H2O2': 11.0}


# ═══════════════════════════════════════════════════════════════════════
# Gas data loading (copy from gen_all_figures)
# ═══════════════════════════════════════════════════════════════════════

def preprocess(vals):
    out = vals.copy()
    n = len(vals)
    run_start, run_len = -1, 0
    stable_start = n
    for i in range(n):
        if vals[i] > 0:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= 5:
                stable_start = run_start
                break
        else:
            run_len = 0
    if stable_start >= n:
        return np.maximum(out, 0.0)
    nz_after = [(i, vals[i]) for i in range(stable_start, n) if vals[i] > 0]
    if len(nz_after) >= 2:
        nz_idx = np.array([x[0] for x in nz_after])
        nz_vals = np.array([x[1] for x in nz_after])
        for i in range(stable_start, n):
            if out[i] <= 0:
                out[i] = np.interp(i, nz_idx, nz_vals)
    sg_win = 15
    stable_region = out[stable_start:]
    if len(stable_region) >= sg_win:
        stable_region = savgol_filter(stable_region, window_length=sg_win,
                                      polyorder=3)
        out[stable_start:] = np.maximum(stable_region, 0.0)
    first_val = out[stable_start]
    for i in range(stable_start):
        out[i] = first_val * (i / max(stable_start, 1))
    return np.maximum(out, 0.0)


def load_gas():
    csv_path = (_project_root / 'empty chamber' / 'empty chamber'
                / '1kHz3.2kVpp.csv')
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = preprocess(
                np.maximum(df[col].values.astype(float), 0.0))
        else:
            gas_conc[col] = np.zeros(len(df))
    if np.all(gas_conc.get('N2O4', np.zeros(1)) == 0):
        T = 298.15
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298)
            + (N2O4_EQ.DELTA_H / PHYSICAL.R)
            * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * gas_conc['NO2']**2
    return times, gas_conc


# ═══════════════════════════════════════════════════════════════════════
# 1D runner
# ═══════════════════════════════════════════════════════════════════════

def run_1d(hono_gas, hono2_gas, h2o2_gas, delta_gas, label=""):
    times, gas_conc = load_gas()

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='gas_alpha',
        alpha_b=None,
        delta_gas=delta_gas,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=hono_gas, hono2_gas=hono2_gas, h2o2_gas=h2o2_gas,
    )

    t_end = float(times[-1])
    t_eval = np.array([t_end])
    y0 = solver.build_initial_condition(initial_pH=7.0)

    t0 = time.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval, y0=y0,
        verbose=False, dt_poisson=None,
    )
    wall = time.time() - t0
    print(f"  [{label}] t={wall:.1f}s, pH={result['pH_avg']:.3f}")

    return solver, result


# ═══════════════════════════════════════════════════════════════════════
# Charge balance analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_charge_balance(solver, result, label=""):
    """Compute full ion inventory and charge balance at final time."""
    avg = result['spatial_avg']
    pH = result['pH_avg']
    h_plus = 10**(-pH)

    # Charge assignments for acid-base species at current pH
    # From ACID_BASE_PAIRS: total → (acid, base, pKa)
    cations = {'H+': h_plus}
    anions = {'OH-': 1e-14 / h_plus}

    # Acid-base speciation for total variables
    for total_name, (acid, base, pKa) in ACID_BASE_PAIRS.items():
        if total_name in avg:
            Ka = 10**(-pKa)
            total = avg[total_name]
            frac_base = Ka / (Ka + h_plus)
            base_conc = total * frac_base
            if base_conc > 1e-15:
                anions[base] = base_conc

    # Direct anions
    for sp in ['O-', 'O3-']:
        if sp in avg:
            anions[sp] = avg[sp]

    print(f"\n  ─── Charge Balance: {label} ───")
    print(f"  pH = {pH:.3f}, [H+] = {h_plus*1e6:.2f} µM")
    print(f"\n  Cations [µM]:")
    total_cat = 0.0
    for sp, c in sorted(cations.items(), key=lambda x: -x[1]):
        print(f"    {sp:>12s}: {c*1e6:12.4f}")
        total_cat += c
    print(f"    {'Σ cations':>12s}: {total_cat*1e6:12.4f}")

    print(f"\n  Anions [µM]:")
    total_an = 0.0
    for sp, c in sorted(anions.items(), key=lambda x: -x[1]):
        if c > 1e-12:
            print(f"    {sp:>12s}: {c*1e6:12.4f}")
            total_an += c
    print(f"    {'Σ anions':>12s}: {total_an*1e6:12.4f}")

    gap = total_cat - total_an
    print(f"\n  Charge gap (cat-an): {gap*1e6:+.4f} µM "
          f"({gap/max(total_cat,1e-15)*100:+.2f}%)")

    return cations, anions, gap


# ═══════════════════════════════════════════════════════════════════════
# NO2- rate budget
# ═══════════════════════════════════════════════════════════════════════

def analyze_no2_budget(solver, result, label=""):
    """Compute rate of each reaction affecting NO2- at final time.

    NO2- is part of HONO_total. Track net stoichiometric contribution to
    HONO_total via HONO and NO2- appearances in reactants/products.
    """
    avg = result['spatial_avg']
    pH = result['pH_avg']
    h_plus = 10**(-pH)

    Ka_hono = 10**(-3.4)
    hono_total = avg.get('HONO_total', 0.0)
    no2_minus = hono_total * Ka_hono / (Ka_hono + h_plus)
    print(f"\n  ─── NO2- Budget: {label} ───")
    print(f"  [HONO_total] = {hono_total*1e6:.4f} µM")
    print(f"  [NO2-]       = {no2_minus*1e6:.4f} µM  (at pH {pH:.2f})")
    print(f"  [HONO]       = {(hono_total-no2_minus)*1e6:.4f} µM")

    # Get final state
    y_final = result['y_final']
    C = y_final.reshape(solver.N_z, solver.N_s)
    dz = solver.dz_cells
    vol_w = dz / dz.sum()

    chem = solver.chem

    # Accumulate per-reaction contribution to dHONO_total/dt
    rxn_contrib = {}
    for cell_j in range(solver.N_z):
        c_cell = C[cell_j, :].copy()
        speciated = chem.speciate(c_cell)

        for k, rxn_d in enumerate(chem._rxn_data):
            rate = chem._compute_single_rate(rxn_d, c_cell, speciated)
            if abs(rate) < 1e-30:
                continue

            # Count HONO + NO2- stoichiometry (both map to HONO_total)
            d_hono_total = 0
            for sp_name, coeff, _ in rxn_d['products']:
                if sp_name in ('HONO', 'NO2-', 'HONO_total'):
                    d_hono_total += coeff
            for sp_name, coeff, _ in rxn_d['reactants']:
                if sp_name in ('HONO', 'NO2-', 'HONO_total'):
                    d_hono_total -= coeff

            if d_hono_total == 0:
                continue

            label_rxn = chem.reactions[k].get('label', f'rxn{k}')
            contrib = d_hono_total * rate * vol_w[cell_j]
            rxn_contrib[label_rxn] = rxn_contrib.get(label_rxn, 0.0) + contrib

    sorted_contrib = sorted(rxn_contrib.items(), key=lambda x: -abs(x[1]))
    print(f"\n  Top contributions to d[HONO_total]/dt [M/s]:")
    print(f"  {'Reaction':<50s} {'Rate':>14s}")
    for label_rxn, rate in sorted_contrib[:15]:
        if abs(rate) > 1e-20:
            sign = '+' if rate > 0 else '-'
            print(f"    {label_rxn[:48]:<48s}  {sign}{abs(rate):.3e}")

    # Sum of sources and sinks
    sources = sum(r for _, r in sorted_contrib if r > 0)
    sinks = sum(r for _, r in sorted_contrib if r < 0)
    print(f"\n  Σ sources = +{sources:.3e} M/s")
    print(f"  Σ sinks   = {sinks:.3e} M/s")
    print(f"  Net       = {sources+sinks:+.3e} M/s")


# ═══════════════════════════════════════════════════════════════════════
# HONO2 sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════

def hono2_sensitivity():
    """Sweep HONO2 gas input and observe pH response."""
    print("\n" + "=" * 72)
    print("  HONO2 Sensitivity Sweep")
    print("=" * 72)

    hono2_values = [0.0, 1e12, 1e13, 1e14, 1e15]
    results = []

    for h2 in hono2_values:
        label = f"HONO2={h2:.0e}"
        _, result = run_1d(
            hono_gas=BASE_PARAMS['hono_gas'],
            hono2_gas=h2,
            h2o2_gas=BASE_PARAMS['h2o2_gas'],
            delta_gas=BASE_PARAMS['delta_gas'],
            label=label,
        )
        avg = result['spatial_avg']
        results.append({
            'HONO2_gas': h2,
            'pH': result['pH_avg'],
            'NO3-': avg.get('NO3-', 0) * 1e6,
            'NO2-': avg.get('NO2-', 0) * 1e6,
            'H2O2': avg.get('H2O2', 0) * 1e6,
        })

    print(f"\n  {'HONO2 [cm⁻³]':>15s}  {'pH':>6s}  "
          f"{'NO₃⁻ [µM]':>10s}  {'NO₂⁻ [µM]':>10s}  {'H₂O₂ [µM]':>10s}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['HONO2_gas']:>15.2e}  {r['pH']:>6.3f}  "
              f"{r['NO3-']:>10.2f}  {r['NO2-']:>10.4f}  {r['H2O2']:>10.2f}")
    print(f"  {'Experiment':>15s}  {EXP['pH']:>6.2f}  "
          f"{EXP['NO3-']:>10.2f}  {EXP['NO2-']:>10.2f}  {EXP['H2O2']:>10.2f}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  DIAGNOSTIC: pH gap and NO2- depletion")
    print("=" * 72)
    print(f"  Base params: HONO={BASE_PARAMS['hono_gas']:.2e}, "
          f"HONO2={BASE_PARAMS['hono2_gas']:.2e}")
    print(f"                H2O2={BASE_PARAMS['h2o2_gas']:.2e}, "
          f"δ_gas={BASE_PARAMS['delta_gas']*1e3:.1f}mm")

    # Stage 1: Baseline run with full analysis
    print("\n" + "=" * 72)
    print("  Stage 1: Baseline 1D Analysis")
    print("=" * 72)

    solver, result = run_1d(**BASE_PARAMS, label="baseline")
    analyze_charge_balance(solver, result, label="baseline")
    analyze_no2_budget(solver, result, label="baseline")

    # Stage 2: HONO2 sensitivity
    hono2_sensitivity()


if __name__ == '__main__':
    main()
