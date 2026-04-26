#!/usr/bin/env python3
"""H2O2 rate budget diagnostic — DIW vs Saline at 3.2 kV Humid_fitting.

Goal: Identify which reactions consume H2O2 in saline 25-80x faster than DIW.
H2O2 source is dominated by gas-phase mass transfer (identical for DIW/Saline),
so the drop must come from sink-side amplification.

Runs two sims to t=600s, extracts per-reaction rates at the final state,
and prints a ranked source/sink budget for H2O2_total in both cases.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))
sys.path.insert(0, str(_root / 'Figures'))

import gen_all_figures as gaf
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from run_alpha_analysis import (
    compute_per_reaction_rates,
    compute_mass_transfer_flux,
    species_rate_budget,
)

print = functools.partial(print, flush=True)

VOLTAGE = '3.2kV'


def load_humid_fit(voltage: str):
    gaf.DEFAULT_GAS_SHEET = voltage
    gaf.CONDITION_LABEL = 'Humid_fitting'
    times, gas_conc = gaf.load_gas_data()
    return times, gas_conc, gaf.HONO_GAS, gaf.HONO2_GAS, gaf.H2O2_GAS


def run_case(mode: str):
    is_saline = (mode == 'Saline')
    print()
    print("=" * 72)
    print(f"Running {mode} {VOLTAGE} Humid_fitting ...")
    print("=" * 72)

    times, gas, hono_g, hono2_g, h2o2_g = load_humid_fit(VOLTAGE)
    chem = AqueousChemistry1D(saline_mode=is_saline)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=is_saline,
        fixed_cation_conc=(0.154 if is_saline else 0.0),
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas,
        hono_gas=hono_g, hono2_gas=hono2_g, h2o2_gas=h2o2_g,
    )
    t_end = float(times[-1])
    t_eval = np.arange(2.0, t_end + 0.1, 2.0)
    y0 = solver.build_initial_condition(initial_pH=7.0)

    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval, y0=y0,
        verbose=False, dt_poisson=None,
    )

    y_final_flat = result['y_final']
    if y_final_flat.ndim == 1:
        y_final = y_final_flat.reshape(solver.N_z, solver.N_s)
    else:
        y_final = y_final_flat

    rxn_rates = compute_per_reaction_rates(solver, y_final)
    mt_flux = compute_mass_transfer_flux(solver, y_final)

    # Volume-averaged bulk concentrations of interest
    dz = solver.dz_cells
    L = solver.L
    idx = solver.species_idx

    def avg(sp):
        if sp not in idx:
            return 0.0
        return float(np.sum(y_final[:, idx[sp]] * dz) / L)

    bulk = {
        'H2O2_total': avg('H2O2') + avg('HO2-'),
        'H2O2': avg('H2O2'),
        'HO2-': avg('HO2-'),
        'OH': avg('OH'),
        'HO2': avg('HO2'),
        'O2-': avg('O2-'),
        'O3': avg('O3'),
        'Cl-': avg('Cl-') if is_saline else 0.0,
        'Cl': avg('Cl') if is_saline else 0.0,
        'Cl2': avg('Cl2') if is_saline else 0.0,
        'Cl2-': avg('Cl2-') if is_saline else 0.0,
        'HOCl-': avg('HOCl-') if is_saline else 0.0,
    }
    surface = {
        'H2O2_total': float(y_final[0, idx['H2O2']] + y_final[0, idx.get('HO2-', 0)]) if 'H2O2' in idx else 0.0,
        'OH': float(y_final[0, idx['OH']]) if 'OH' in idx else 0.0,
    }

    return {
        'mode': mode,
        'rxn_rates': rxn_rates,
        'mt_flux': mt_flux,
        'bulk': bulk,
        'surface': surface,
        'solver': solver,
        'y_final': y_final,
    }


def print_budget(case, species='H2O2'):
    mode = case['mode']
    print()
    print("=" * 90)
    print(f"  H2O2 rate budget — {mode} {VOLTAGE} Humid_fitting")
    print("=" * 90)

    bulk = case['bulk']
    surf = case['surface']
    mt = case['mt_flux']
    print(f"  Bulk avg:  H2O2_total={bulk['H2O2_total']*1e6:.3f} µM  "
          f"H2O2={bulk['H2O2']*1e6:.3f}  HO2-={bulk['HO2-']*1e6:.3e}")
    print(f"  OH={bulk['OH']*1e12:.3e} pM  HO2={bulk['HO2']*1e12:.3e} pM  "
          f"O2-={bulk['O2-']*1e12:.3e} pM  O3={bulk['O3']*1e6:.3f} µM")
    if mode == 'Saline':
        print(f"  Cl-={bulk['Cl-']*1e3:.3f} mM  Cl={bulk['Cl']*1e12:.3e} pM  "
              f"Cl2={bulk['Cl2']*1e12:.3e} pM  Cl2-={bulk['Cl2-']*1e12:.3e} pM  "
              f"HOCl-={bulk['HOCl-']*1e9:.3e} nM")
    print(f"  Surface OH={surf['OH']*1e12:.3e} pM")

    # Mass transfer flux for H2O2 (gas-phase name = 'H2O2')
    mt_h2o2 = mt.get('H2O2', 0.0)
    print(f"\n  Mass transfer flux (H2O2 gas → aq): {mt_h2o2:.3e} M/s  "
          f"(vol-avg [M/s])")

    prod, cons, total_prod, total_cons, _mt_inner = species_rate_budget(
        case['rxn_rates'], species, mt_flux=None,
    )

    print(f"\n  Chemistry production (top 10):  sum = {total_prod:.3e} M/s")
    for e in prod[:10]:
        frac = e['rate'] / total_prod * 100 if total_prod > 0 else 0
        print(f"    {e['label']:40s}  {e['rate']:+.3e} M/s  ({frac:5.1f}%)")

    print(f"\n  Chemistry consumption (top 10):  sum = {total_cons:.3e} M/s")
    for e in cons[:10]:
        frac = abs(e['rate']) / total_cons * 100 if total_cons > 0 else 0
        print(f"    {e['label']:40s}  {e['rate']:+.3e} M/s  ({frac:5.1f}%)")

    net_chem = total_prod - total_cons
    residual = net_chem + mt_h2o2
    print(f"\n  Net (chem): {net_chem:+.3e} M/s")
    print(f"  Net (chem + MT): {residual:+.3e} M/s   (≈0 at steady state)")


def main():
    diw = run_case('DIW')
    sal = run_case('Saline')

    print_budget(diw, species='H2O2')
    print_budget(sal, species='H2O2')

    # Direct comparison: ratio of key sinks
    print()
    print("=" * 90)
    print("  Sink comparison: Saline / DIW ratio for each reaction")
    print("=" * 90)

    def sink_dict(case):
        _, cons, _, _, _ = species_rate_budget(
            case['rxn_rates'], 'H2O2', mt_flux=None,
        )
        return {e['label']: abs(e['rate']) for e in cons}

    diw_sinks = sink_dict(diw)
    sal_sinks = sink_dict(sal)
    all_labels = set(diw_sinks) | set(sal_sinks)

    rows = []
    for lab in all_labels:
        d = diw_sinks.get(lab, 0.0)
        s = sal_sinks.get(lab, 0.0)
        ratio = s / d if d > 1e-30 else float('inf')
        rows.append((lab, d, s, ratio))
    rows.sort(key=lambda r: r[2], reverse=True)  # sort by saline rate desc

    print(f"  {'reaction':40s}  {'DIW [M/s]':>12s}  {'Sal [M/s]':>12s}  {'ratio':>10s}")
    print("  " + "-" * 80)
    for lab, d, s, r in rows[:15]:
        r_str = "inf" if r == float('inf') else f"{r:.2f}"
        print(f"  {lab:40s}  {d:>12.3e}  {s:>12.3e}  {r_str:>10s}")


if __name__ == '__main__':
    main()
