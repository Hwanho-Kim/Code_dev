#!/usr/bin/env python3
"""
α_b sensitivity analysis: Run DIW simulations for α_b = 0.01, 0.03, 0.05.

Outputs:
  1. Basic results (pH, NO2⁻, NO3⁻, H2O2) per case
  2. Radical concentration comparison (OH, O3, HO2, O2⁻, ONOO⁻, NO2, etc.)
  3. Dominant reaction rates for RONS (NO2⁻, NO3⁻) and H2O2
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_1d import (
    PHYSICAL, MASS_TRANSFER, GRID,
    GAS_TO_AQUEOUS_MAP, ACID_BASE_PAIRS,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    Path(__file__).parent.parent
    / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

ALPHA_CASES = [0.01, 0.03, 0.05]


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


def run_case(alpha_b, times, gas_conc):
    """Run one DIW case with film_alpha BC. Returns (result_dict, solver)."""
    chem = AqueousChemistry1D(saline_mode=False)
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
        hono_gas=0,
        hono2_gas=0,
        h2o2_gas=0,
    )
    t_end = float(times[-1])

    # Save initial state for ΔC/Δt verification
    y0 = solver.build_initial_condition(initial_pH=7.0)
    y0_2d = y0.reshape(solver.N_z, solver.N_s)
    init_avg = solver._compute_spatial_average(y0_2d)

    t0 = time.time()
    result = solver.solve(
        t_span=(0, t_end),
        t_eval=np.array([0, t_end / 4, t_end / 2, 3 * t_end / 4, t_end]),
        verbose=True,
        dt_poisson=10.0,
    )
    result['wall_s'] = time.time() - t0
    result['init_avg'] = init_avg
    result['t_end'] = t_end
    return result, solver


def extract_species(result, solver):
    """Extract detailed species concentrations from final state."""
    avg = result['spatial_avg']
    sfc = result['surface']

    # Key species to report
    pH_sfc = result.get('pH_surface', None)
    if pH_sfc is None:
        pH_sfc = -np.log10(max(sfc.get('H+', 1e-7), 1e-14))
    species_of_interest = {
        # Measured products
        'pH': result['pH_avg'],
        'pH_surface': pH_sfc,
        # Bulk avg (µM)
        'NO2- (avg)': avg.get('NO2-', 0) * 1e6,
        'NO3- (avg)': avg.get('NO3-', 0) * 1e6,
        'H2O2 (avg)': avg.get('H2O2', 0) * 1e6,
        'HO2- (avg)': avg.get('HO2-', 0) * 1e6,
        # Radicals — bulk avg
        'OH (avg)': avg.get('OH', 0),
        'O3 (avg)': avg.get('O3', 0),
        'HO2 (avg)': avg.get('HO2', 0),
        'O2- (avg)': avg.get('O2-', 0),
        'NO2 (avg)': avg.get('NO2', 0),
        'NO (avg)': avg.get('NO', 0),
        'ONOOH (avg)': avg.get('ONOOH', 0),
        'ONOO- (avg)': avg.get('ONOO-', 0),
        'O2NOOH (avg)': avg.get('O2NOOH', 0),
        'O2NOO- (avg)': avg.get('O2NOO-', 0),
        'O3- (avg)': avg.get('O3-', 0),
        'N2O5 (avg)': avg.get('N2O5', 0),
        'N2O4 (avg)': avg.get('N2O4', 0),
        'N2O3 (avg)': avg.get('N2O3', 0),
        # Surface concentrations
        'OH (sfc)': sfc.get('OH', 0),
        'O3 (sfc)': sfc.get('O3', 0),
        'HO2 (sfc)': sfc.get('HO2', 0),
        'NO2 (sfc)': sfc.get('NO2', 0),
    }
    return species_of_interest


def compute_per_reaction_rates(solver, y_final):
    """Compute volume-averaged rate for each reaction at the final state.

    Returns list of (label, rate_avg [M/s], reactant_str, product_str).
    """
    chem = solver.chem
    N_z = solver.N_z
    dz = solver.dz_cells
    L = solver.L

    rxn_rates = []
    n_rxn = len(chem.reactions)

    for ri in range(n_rxn):
        rxn = chem.reactions[ri]
        rxn_d = chem._rxn_data[ri]
        label = rxn.get('label', f'R{ri}')

        rates_cells = np.zeros(N_z)
        for j in range(N_z):
            y_cell = y_final[j, :].copy()
            # Sanitize
            y_cell = np.clip(y_cell, chem.trace, 1.0)
            h_idx = chem.species_idx['H+']
            y_cell[h_idx] = max(y_cell[h_idx], 1e-14)
            speciated = chem.speciate(y_cell)
            rate = chem._compute_single_rate(rxn_d, y_cell, speciated)
            rates_cells[j] = rate

        # Volume-weighted average
        rate_avg = np.sum(rates_cells * dz) / L

        # Build reaction string
        r_str = ' + '.join(
            f'{int(c)}{s}' if int(c) > 1 else str(s)
            for s, c in rxn['reactants'].items()
        )
        p_str = ' + '.join(
            f'{int(c)}{s}' if int(c) > 1 else str(s)
            for s, c in rxn.get('products', {}).items()
        )

        rxn_rates.append({
            'label': label,
            'rate': rate_avg,
            'abs_rate': abs(rate_avg),
            'reactants_str': r_str,
            'products_str': p_str,
            'reactants': rxn['reactants'],
            'products': rxn.get('products', {}),
        })

    return rxn_rates


def compute_mass_transfer_flux(solver, y_final):
    """Compute volume-averaged mass transfer flux for each interface species.

    Returns dict: {aqueous_species_name: flux_vol_avg [M/s]}.
    Flux > 0 means gas → liquid (production in liquid).
    """
    t_end = solver._dt_gas * (solver._n_times - 1)

    mt_flux = {}
    L = solver.L
    idx_to_name = {v: k for k, v in solver.species_idx.items()}
    for aq_idx, k_mt, gas_sp, H_val, Ka in solver._interface_species:
        C_eq = solver._get_C_eq_fast(gas_sp, t_end)
        C_0 = y_final[0, aq_idx]  # surface cell concentration
        # flux in surface cell: k_mt/dz[0] * (C_eq - C_0) [M/s]
        # volume-averaged: flux * dz[0] / L = k_mt * (C_eq - C_0) / L
        flux_vol = k_mt * (C_eq - C_0) / L
        aq_name = idx_to_name[aq_idx]
        mt_flux[aq_name] = flux_vol
    return mt_flux



def species_rate_budget(rxn_rates, species_name, mt_flux=None):
    """Compute full mass balance budget for ONE species.

    Includes: chemical reactions + mass transfer flux.
    dC/dt = Σ(rxn) + MT  →  residual = Σ(rxn) + MT = dC/dt (accumulation).
    Returns (prod_list, cons_list, total_prod, total_cons, mt_val).
    """
    # Map speciated species to total variable name used in reaction defs
    spec_to_total = {
        'HONO': 'HONO_total', 'NO2-': 'HONO_total',
        'HONO2': 'HONO2_total', 'NO3-': 'HONO2_total',
        'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
        'HO2': 'HO2_total', 'O2-': 'HO2_total',
        'ONOOH': 'ONOOH_total', 'ONOO-': 'ONOOH_total',
        'O2NOOH': 'O2NOOH_total', 'O2NOO-': 'O2NOOH_total',
        'HClO': 'HClO_total', 'ClO-': 'HClO_total',
    }

    # Names to match in reaction reactants/products
    match_names = {species_name}
    if species_name in spec_to_total:
        match_names.add(spec_to_total[species_name])

    prod_list = []
    cons_list = []

    for r in rxn_rates:
        in_reactants = set(r['reactants'].keys()) & match_names
        in_products = set(r['products'].keys()) & match_names

        if not in_reactants and not in_products:
            continue

        # Net contribution to this species: prod(+) / cons(-)
        net = 0.0
        for sp in in_products:
            net += int(r['products'][sp]) * r['rate']
        for sp in in_reactants:
            net -= int(r['reactants'][sp]) * r['rate']

        if abs(net) < 1e-30:
            continue

        rxn_str = f"{r['reactants_str']} → {r['products_str']}"
        entry = {'label': r['label'], 'rate': net, 'rxn_str': rxn_str}

        if net > 0:
            prod_list.append(entry)
        else:
            cons_list.append(entry)

    prod_list.sort(key=lambda x: x['rate'], reverse=True)
    cons_list.sort(key=lambda x: x['rate'])  # most negative first

    total_prod = sum(e['rate'] for e in prod_list)
    total_cons = sum(abs(e['rate']) for e in cons_list)

    # Mass transfer flux for this species (or its total variable)
    mt_val = 0.0
    if mt_flux:
        for name in match_names:
            mt_val += mt_flux.get(name, 0.0)

    return prod_list, cons_list, total_prod, total_cons, mt_val


def compute_actual_dCdt(species_name, results_by_alpha, alpha_cases):
    """Compute actual ΔC/Δt = (C_final_avg - C_init_avg) / t_end per α_b.

    Handles acid-base pair totals: e.g. NO3⁻ → HONO2_total = HONO2 + NO3⁻.
    """
    # Map species to the set of components that make up its "total"
    total_components = {
        'NO3-': ['HONO2', 'NO3-'],
        'NO2-': ['HONO', 'NO2-'],
        'H2O2': ['H2O2', 'HO2-'],
        'O3': ['O3'],
        'NO': ['NO'],
        'OH': ['OH'],
        'HO2': ['HO2', 'O2-'],
    }
    components = total_components.get(species_name, [species_name])

    actual = {}
    for ab in alpha_cases:
        r = results_by_alpha[ab]
        init_avg = r['init_avg']
        final_avg = r['spatial_avg']
        t_end = r['t_end']

        c_init = sum(init_avg.get(sp, 0.0) for sp in components)
        c_final = sum(final_avg.get(sp, 0.0) for sp in components)
        actual[ab] = (c_final - c_init) / t_end
    return actual


def print_species_budget(species_name, rxn_rates_by_alpha, alpha_cases,
                         mt_flux_by_alpha, actual_dCdt_by_alpha,
                         pct_threshold=1.0):
    """Print full mass balance budget for one species across α_b cases.

    Balance: dC/dt = Σ(rxn sources) - Σ(rxn sinks) + MT
    Cross-check: actual ΔC/Δt = (C_final - C_init) / t_end
    """
    print(f"\n  ── {species_name} ──")

    for ab in alpha_cases:
        mt_flux = mt_flux_by_alpha.get(ab, {})
        prod, cons, tot_p, tot_c, mt_val = species_rate_budget(
            rxn_rates_by_alpha[ab], species_name, mt_flux)

        # dC/dt = net_rxn + MT = (tot_p - tot_c) + mt_val
        net_rxn = tot_p - tot_c
        dCdt_rate = net_rxn + mt_val

        # Actual ΔC/Δt from simulation
        dCdt_actual = actual_dCdt_by_alpha.get(ab, 0.0)

        # All sources vs all sinks
        all_sources = tot_p + max(mt_val, 0)
        all_sinks = tot_c + max(-mt_val, 0)
        turnover = max(all_sources, all_sinks, 1e-30)

        print(f"\n  α_b={ab}")
        print(f"    ┌─ Sources ──────────────────────────────────")
        if tot_p > 0:
            for e in prod:
                pct = e['rate'] / turnover * 100
                if pct >= pct_threshold:
                    print(f"    │  {e['label']:<14s}  {e['rate']:+.3e}  "
                          f"({pct:5.1f}%)  {e['rxn_str']}")
        if mt_val > 0:
            mt_pct = mt_val / turnover * 100
            print(f"    │  {'MT(gas→liq)':<14s}  {mt_val:+.3e}  "
                  f"({mt_pct:5.1f}%)")
        print(f"    │  {'Σ sources':<14s}  {all_sources:+.3e} M/s")
        print(f"    ├─ Sinks ────────────────────────────────────")
        if tot_c > 0:
            for e in cons:
                pct = abs(e['rate']) / turnover * 100
                if pct >= pct_threshold:
                    print(f"    │  {e['label']:<14s}  {e['rate']:+.3e}  "
                          f"({pct:5.1f}%)  {e['rxn_str']}")
        if mt_val < 0:
            mt_pct = abs(mt_val) / turnover * 100
            print(f"    │  {'MT(liq→gas)':<14s}  {mt_val:+.3e}  "
                  f"({mt_pct:5.1f}%)")
        print(f"    │  {'Σ sinks':<14s}  {-all_sinks:+.3e} M/s")
        print(f"    └─ Balance ──────────────────────────────────")
        print(f"       rate budget dC/dt = {dCdt_rate:+.3e} M/s")
        print(f"       actual  ΔC/Δt    = {dCdt_actual:+.3e} M/s")
        if abs(dCdt_actual) > 1e-30:
            ratio = dCdt_rate / dCdt_actual
            print(f"       ratio (rate/actual) = {ratio:.3f}  "
                  f"({'OK' if 0.5 < ratio < 2.0 else 'MISMATCH'})")
        else:
            print(f"       (actual ≈ 0, skip ratio)")


def main():
    params_path = Path(__file__).parent / 'optimal_params_1d.yaml'
    with open(params_path) as f:
        yaml.safe_load(f)

    times, gas_conc = load_gas_data(DEFAULT_CSV)

    print("=" * 78)
    print("α_b SENSITIVITY ANALYSIS — DIW 1D (Film+α_b BC)")
    print("=" * 78)
    print(f"  Cases: α_b = {ALPHA_CASES}")
    print(f"  CSV: {len(times)} timesteps, t_end={times[-1]:.0f}s")
    print(f"  Gas unmeasured: HONO=0, HONO2=0, H2O2=0 (측정종만)")
    print()

    # ---- Run simulations ----
    all_results = {}
    all_solvers = {}
    all_species = {}
    all_rxn_rates = {}

    for ab in ALPHA_CASES:
        print("=" * 78)
        print(f"  Running α_b = {ab}")
        print("=" * 78)
        result, solver = run_case(ab, times, gas_conc)
        all_results[ab] = result
        all_solvers[ab] = solver

        sp = extract_species(result, solver)
        all_species[ab] = sp
        print(f"  → pH={sp['pH']:.3f}, NO3⁻={sp['NO3- (avg)']:.1f}µM, "
              f"H2O2={sp['H2O2 (avg)']:.2f}µM, "
              f"time={result['wall_s']:.0f}s")
        print()

    # ---- 1. Basic results comparison ----
    print()
    print("=" * 78)
    print("1. BASIC RESULTS COMPARISON")
    print("=" * 78)
    header = f"{'α_b':>6s}  {'pH':>6s}  {'NO2⁻(µM)':>9s}  {'NO3⁻(µM)':>10s}  {'H2O2(µM)':>9s}  {'Time':>6s}"
    print(header)
    print("─" * len(header))
    for ab in ALPHA_CASES:
        sp = all_species[ab]
        w = all_results[ab]['wall_s']
        print(f"{ab:6.2f}  {sp['pH']:6.3f}  {sp['NO2- (avg)']:9.1f}  "
              f"{sp['NO3- (avg)']:10.1f}  {sp['H2O2 (avg)']:9.2f}  {w/60:5.1f}m")
    print("─" * len(header))
    print(f"{'실험':>6s}  {3.61:6.2f}  {3.0:9.1f}  {63.0:10.1f}  {11.0:9.2f}")
    print()

    # ---- 2. Radical concentration comparison ----
    print("=" * 78)
    print("2. RADICAL / KEY SPECIES CONCENTRATION COMPARISON")
    print("=" * 78)

    # Radicals in M (scientific notation)
    rad_keys = [
        ('OH (avg)', 'M'),
        ('O3 (avg)', 'M'),
        ('HO2 (avg)', 'M'),
        ('O2- (avg)', 'M'),
        ('NO2 (avg)', 'M'),
        ('NO (avg)', 'M'),
        ('ONOOH (avg)', 'M'),
        ('ONOO- (avg)', 'M'),
        ('O2NOOH (avg)', 'M'),
        ('O2NOO- (avg)', 'M'),
        ('O3- (avg)', 'M'),
        ('N2O5 (avg)', 'M'),
        ('N2O3 (avg)', 'M'),
    ]

    header2 = f"{'Species':<18s}"
    for ab in ALPHA_CASES:
        header2 += f"  {'α_b='+str(ab):>14s}"
    print(header2)
    print("─" * len(header2))

    for key, unit in rad_keys:
        row = f"{key:<18s}"
        for ab in ALPHA_CASES:
            val = all_species[ab][key]
            row += f"  {val:14.3e}"
        print(row)

    # Surface radicals
    print()
    print("  --- Surface (z=0) ---")
    sfc_keys = [('OH (sfc)', 'M'), ('O3 (sfc)', 'M'),
                ('HO2 (sfc)', 'M'), ('NO2 (sfc)', 'M')]
    for key, unit in sfc_keys:
        row = f"{key:<18s}"
        for ab in ALPHA_CASES:
            val = all_species[ab][key]
            row += f"  {val:14.3e}"
        print(row)
    print()

    # ---- 3. Per-species FULL MASS BALANCE ----
    print("=" * 78)
    print("3. PER-SPECIES MASS BALANCE (rxn + mass transfer + accumulation)")
    print("   volume-avg, final state, rxn ≥1%")
    print("=" * 78)

    # Compute reaction rates and mass transfer flux for each case
    all_mt_flux = {}
    for ab in ALPHA_CASES:
        print(f"\n  Computing rates for α_b={ab}...")
        y_final = all_results[ab]['y_final']
        solver = all_solvers[ab]
        if y_final.ndim == 1:
            y_final = y_final.reshape(solver.N_z, solver.N_s)
        rxn_rates = compute_per_reaction_rates(solver, y_final)
        all_rxn_rates[ab] = rxn_rates
        all_mt_flux[ab] = compute_mass_transfer_flux(solver, y_final)

    target_species = ['NO3-', 'NO2-', 'NO', 'O3', 'H2O2']

    # Compute actual ΔC/Δt for cross-check
    all_actual_dCdt = {}
    for sp_name in target_species:
        all_actual_dCdt[sp_name] = compute_actual_dCdt(
            sp_name, all_results, ALPHA_CASES)

    for sp_name in target_species:
        print()
        print("=" * 78)
        print_species_budget(sp_name, all_rxn_rates, ALPHA_CASES,
                             all_mt_flux, all_actual_dCdt[sp_name],
                             pct_threshold=1.0)

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == '__main__':
    main()
