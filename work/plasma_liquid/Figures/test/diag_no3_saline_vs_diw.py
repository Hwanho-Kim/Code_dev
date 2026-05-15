#!/usr/bin/env python3
"""NO3- rate budget — DIW vs Saline at 3.2 kV Humid_fitting.

Goal: Identify why saline NO3- ≈ DIW NO3- (sim Sal/DIW = 1.00) when
experimental ratio is 1.62. Saline-specific NO3- pathways (S55, S56, S58)
should produce extra NO3- from N2O5 + Cl-, ClNO2 hydrolysis chain.
This script enumerates which reactions actually fire.

Reports:
  - NO3- production budget for both cases (top reactions)
  - Saline-specific reactions involving Cl-, ClNO2 (S55-S60 etc.)
  - Bulk concentrations of Cl chemistry intermediates
  - Mass transfer flux (gas N2O5 -> aq, etc.)
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
    y_final = result['y_final']
    if y_final.ndim == 1:
        y_final = y_final.reshape(solver.N_z, solver.N_s)

    rxn_rates = compute_per_reaction_rates(solver, y_final)
    mt_flux = compute_mass_transfer_flux(solver, y_final)

    dz = solver.dz_cells
    L = solver.L
    idx = solver.species_idx

    def avg(sp):
        return float(np.sum(y_final[:, idx[sp]] * dz) / L) if sp in idx else 0.0

    bulk = {sp: avg(sp) for sp in
            ['NO3-', 'NO2-', 'H2O2', 'HO2-', 'OH', 'O3', 'N2O5',
             'Cl-', 'Cl', 'Cl2', 'Cl2-', 'HOCl-', 'HClO', 'ClO-',
             'ClNO2', 'HONO']}

    spatial_avg = result['spatial_avg']
    return {
        'mode': mode,
        'rxn_rates': rxn_rates,
        'mt_flux': mt_flux,
        'bulk': bulk,
        'spatial_avg': spatial_avg,
        'avg': {sp: float(spatial_avg.get(sp, 0))
                for sp in ['NO3-', 'NO2-', 'H2O2', 'Cl-', 'OH', 'HClO_total',
                           'HONO_total', 'H2O2_total', 'HONO2_total']},
    }


def print_budget(case, species='NO3-'):
    mode = case['mode']
    print()
    print("=" * 100)
    print(f"  {species} rate budget — {mode} {VOLTAGE} Humid_fitting")
    print("=" * 100)
    val = case['avg'].get(species, 0)
    if species == 'Cl-':
        print(f"  Bulk-avg [{species}] = {val*1e3:.4f} mM  "
              f"(initial 154.0 mM)  Δ = {(val-0.154)*1e6:+.2f} µM")
    else:
        print(f"  Bulk-avg [{species}] = {val*1e6:.3f} µM")

    bulk = case['bulk']
    print(f"\n  Key bulk-avg concentrations:")
    print(f"    N2O5 = {bulk['N2O5']*1e9:.3e} nM   "
          f"O3 = {bulk['O3']*1e6:.3f} µM   "
          f"OH = {bulk['OH']*1e12:.3e} pM")
    if mode == 'Saline':
        print(f"    Cl-    = {bulk['Cl-']*1e3:.3f} mM    "
              f"Cl     = {bulk['Cl']*1e12:.3e} pM    "
              f"Cl2    = {bulk['Cl2']*1e12:.3e} pM")
        print(f"    Cl2-   = {bulk['Cl2-']*1e12:.3e} pM    "
              f"HOCl-  = {bulk['HOCl-']*1e9:.3e} nM    "
              f"HClO   = {bulk['HClO']*1e9:.3e} nM")
        print(f"    ClO-   = {bulk['ClO-']*1e9:.3e} nM    "
              f"ClNO2  = {bulk['ClNO2']*1e12:.3e} pM    "
              f"HONO   = {bulk['HONO']*1e9:.3e} nM")

    # Mass transfer fluxes for relevant gas species
    mt = case['mt_flux']
    print(f"\n  Mass transfer flux (gas→aq, vol-avg M/s):")
    for gas_sp in ['N2O5', 'NO2', 'NO3', 'O3', 'HONO', 'HONO2']:
        v = mt.get(gas_sp, 0)
        print(f"    {gas_sp:8s}: {v:+.3e}")

    prod, cons, total_prod, total_cons, _ = species_rate_budget(
        case['rxn_rates'], species, mt_flux=None,
    )

    print(f"\n  {species} production (top 10):  sum = {total_prod:.3e} M/s")
    for e in prod[:10]:
        frac = e['rate'] / total_prod * 100 if total_prod > 0 else 0
        print(f"    {e['label']:50s}  {e['rate']:+.3e}  ({frac:5.1f}%)")

    print(f"\n  {species} consumption (top 5):  sum = {total_cons:.3e} M/s")
    for e in cons[:5]:
        frac = abs(e['rate']) / total_cons * 100 if total_cons > 0 else 0
        print(f"    {e['label']:50s}  {e['rate']:+.3e}  ({frac:5.1f}%)")


def print_saline_specific(saline_case):
    """List Cl-related reactions and their rates (saline only)."""
    print()
    print("=" * 100)
    print("  Saline-specific Cl chemistry — non-zero rates ranked")
    print("=" * 100)
    cl_keywords = ['Cl', 'ClO', 'ClNO']
    cl_rxns = []
    for r in saline_case['rxn_rates']:
        if any(k in r['label'] for k in cl_keywords):
            if abs(r['rate']) > 1e-20:
                cl_rxns.append(r)
    cl_rxns.sort(key=lambda x: abs(x['rate']), reverse=True)
    print(f"  {'rxn label':60s}  {'rate [M/s]':>14s}")
    print("  " + "-" * 80)
    for r in cl_rxns[:25]:
        print(f"  {r['label']:60s}  {r['rate']:>+14.3e}")


def main():
    diw = run_case('DIW')
    sal = run_case('Saline')

    for sp in ['NO3-', 'NO2-', 'H2O2']:
        print_budget(diw, species=sp)
        print_budget(sal, species=sp)

    # Cl- budget — saline only
    print_budget(sal, species='Cl-')

    print_saline_specific(sal)

    # Direct DIW vs Saline NO3- production comparison
    print()
    print("=" * 100)
    print("  NO3- production — DIW vs Saline (top reactions, ratio)")
    print("=" * 100)

    def prod_dict(case):
        prod, _, _, _, _ = species_rate_budget(
            case['rxn_rates'], 'NO3-', mt_flux=None,
        )
        return {e['label']: e['rate'] for e in prod}

    d = prod_dict(diw)
    s = prod_dict(sal)
    all_labels = set(d) | set(s)
    rows = []
    for lab in all_labels:
        dv, sv = d.get(lab, 0), s.get(lab, 0)
        ratio = sv / dv if abs(dv) > 1e-30 else float('inf')
        rows.append((lab, dv, sv, ratio))
    rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])), reverse=True)

    print(f"  {'reaction':50s}  {'DIW':>12s}  {'Saline':>12s}  {'Sal/DIW':>10s}")
    print("  " + "-" * 90)
    for lab, dv, sv, r in rows[:20]:
        r_str = "Sal-only" if r == float('inf') else f"{r:.3f}"
        print(f"  {lab:50s}  {dv:>+12.3e}  {sv:>+12.3e}  {r_str:>10s}")


if __name__ == '__main__':
    main()
