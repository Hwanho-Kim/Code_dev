#!/usr/bin/env python3
"""
Collect monolithic BDF data for Figures 1, 3, 4.

Runs:
  - BC comparison: Two-film, Dirichlet, Film(α_b=1), Film+0.05, Film+0.01
  - α_b sensitivity: 0.01, 0.03, 0.05
  - Detailed rates for mass balance (α_b=0.03)

Outputs data to stdout for updating plot_bc_results.py.
"""

import sys
import time as time_mod
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import (
    PHYSICAL, MASS_TRANSFER, N2O4_EQ,
    GAS_TO_AQUEOUS_MAP, ACID_BASE_PAIRS,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)


def load_gas_data():
    df = pd.read_csv(DEFAULT_CSV)
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
            math.log(N2O4_EQ.KP_298)
            + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)
    return times, gas_conc


def run_case(times, gas_conc, bc_type, alpha_b, label):
    """Run one DIW case, return result dict."""
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type=bc_type,
        alpha_b=alpha_b,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=0, hono2_gas=0, h2o2_gas=0)
    t_end = float(times[-1])

    print(f"\n{'='*60}")
    print(f"Running: {label} (bc={bc_type}, α_b={alpha_b})")
    print(f"{'='*60}")
    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end),
        t_eval=np.array([0, t_end]),
        verbose=True,
        dt_poisson=None,
    )
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    r = {
        'label': label,
        'bc_type': bc_type,
        'alpha_b': alpha_b,
        'pH': result['pH_avg'],
        'NO3': avg.get('NO3-', 0) * 1e6,
        'NO2': avg.get('NO2-', 0) * 1e6,
        'H2O2': avg.get('H2O2', 0) * 1e6,
        'OH': avg.get('OH', 0),
        'O3': avg.get('O3', 0),
        'HO2': avg.get('HO2', 0),
        'O2-': avg.get('O2-', 0),
        'ONOOH': avg.get('ONOOH', 0),
        'O2NOOH': avg.get('O2NOOH', 0),
        'ONOO-': avg.get('ONOO-', 0),
        'O3-': avg.get('O3-', 0),
        'N2O5': avg.get('N2O5', 0),
        'NO2_aq': avg.get('NO2', 0),
        'O2NOO-': avg.get('O2NOO-', 0),
        'wall': wall,
        'success': result['success'],
    }
    print(f"  pH={r['pH']:.3f}, NO3-={r['NO3']:.1f}uM, "
          f"H2O2={r['H2O2']:.2f}uM, wall={wall:.0f}s")
    return r


def compute_mass_balance(times, gas_conc, alpha_b=0.03):
    """Run α_b=0.03 and compute per-reaction rates at final snapshot."""
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='film_alpha',
        alpha_b=alpha_b,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=0, hono2_gas=0, h2o2_gas=0)
    t_end = float(times[-1])

    print(f"\n{'='*60}")
    print(f"Mass balance: α_b={alpha_b}")
    print(f"{'='*60}")
    result = solver.solve(
        t_span=(0, t_end),
        t_eval=np.array([0, t_end]),
        verbose=True,
        dt_poisson=None,
    )

    y_final = result['y_final']
    N_z, N_s = solver.N_z, solver.N_s
    dz, L = solver.dz_cells, solver.L

    # Per-reaction volume-averaged rates at final state
    SPEC_TO_TOTAL = {
        'HONO': 'HONO_total', 'NO2-': 'HONO_total',
        'HONO2': 'HONO2_total', 'NO3-': 'HONO2_total',
        'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
        'HO2': 'HO2_total', 'O2-': 'HO2_total',
        'ONOOH': 'ONOOH_total', 'ONOO-': 'ONOOH_total',
        'O2NOOH': 'O2NOOH_total', 'O2NOO-': 'O2NOOH_total',
    }

    h_idx = chem.species_idx['H+']
    n_rxn = len(chem.reactions)
    rates_avg = np.zeros(n_rxn)
    for j in range(N_z):
        yc = np.clip(y_final[j, :].copy(), chem.trace, 1.0)
        yc[h_idx] = max(yc[h_idx], 1e-14)
        spec = chem.speciate(yc)
        for ri, rxn_d in enumerate(chem._rxn_data):
            rates_avg[ri] += chem._compute_single_rate(rxn_d, yc, spec) * dz[j]
    rates_avg /= L

    # MT flux
    t_idx = max(0, min(int(t_end / solver._dt_gas), solver._n_times - 1))
    idx_to_name = {v: k for k, v in solver.species_idx.items()}
    mt_flux = {}
    hp_idx = solver._h_plus_idx
    h_s = max(y_final[0, hp_idx], 1e-14) if hp_idx >= 0 else 1e-7
    for aq_idx, k_mt, gas_sp, _, Ka in solver._interface_species:
        C_eq = solver._get_C_eq_fast(gas_sp, t_idx)
        C_0 = y_final[0, aq_idx]
        c_eff = C_0 * h_s / (h_s + Ka) if Ka is not None else C_0
        mt_flux[idx_to_name[aq_idx]] = k_mt * (C_eq - c_eff) / L

    # Species breakdown
    for sp_name in ['NO3-', 'O3', 'NO2-', 'H2O2']:
        match_names = {sp_name}
        total = SPEC_TO_TOTAL.get(sp_name)
        if total:
            match_names.add(total)
            for s, t in SPEC_TO_TOTAL.items():
                if t == total:
                    match_names.add(s)

        print(f"\n  --- {sp_name} ---")
        contribs = []
        for ri in range(n_rxn):
            rxn = chem.reactions[ri]
            in_r = set(rxn['reactants'].keys()) & match_names
            in_p = set(rxn.get('products', {}).keys()) & match_names
            if not in_r and not in_p:
                continue
            net = 0.0
            for sp in in_p:
                net += int(rxn['products'][sp]) * rates_avg[ri]
            for sp in in_r:
                net -= int(rxn['reactants'][sp]) * rates_avg[ri]
            if abs(net) > 1e-30:
                contribs.append((rxn.get('label', f'R{ri}'), net))

        mt_val = sum(mt_flux.get(n, 0.0) for n in match_names)
        if abs(mt_val) > 1e-30:
            contribs.append(('MT', mt_val))

        total_src = sum(c for _, c in contribs if c > 0)
        total_snk = sum(c for _, c in contribs if c < 0)
        contribs.sort(key=lambda x: -abs(x[1]))

        for label, rate in contribs:
            if abs(rate) / max(abs(total_src), abs(total_snk), 1e-30) * 100 < 1.0:
                continue
            ref = total_src if rate > 0 else abs(total_snk)
            pct = abs(rate) / max(ref, 1e-30) * 100
            direction = 'src' if rate > 0 else 'snk'
            short = label.split(':')[0].strip() if ':' in label else label
            print(f"    {short:8s} {direction} {pct:6.1f}%  rate={rate:+.3e} M/s")
        print(f"    Σsrc={total_src:.3e}, Σsnk={total_snk:.3e}")


def main():
    import os
    os.chdir(_project_root)

    times, gas_conc = load_gas_data()

    # ── BC comparison (Fig 1) ──
    bc_cases = [
        ('Two-film',       'two_film',   1.0),
        ('Dirichlet',      'dirichlet',  1.0),
        ('Film (α_b=1)',   'film',       1.0),
        ('Film+α_b=0.05', 'film_alpha', 0.05),
        ('Film+α_b=0.01', 'film_alpha', 0.01),
    ]

    bc_results = []
    for label, bc_type, ab in bc_cases:
        r = run_case(times, gas_conc, bc_type, ab, label)
        bc_results.append(r)

    # ── α_b sensitivity (Fig 3) — 0.03 already in bc_results if film_alpha ──
    alpha_cases = [
        ('α_b=0.01', 'film_alpha', 0.01),
        ('α_b=0.03', 'film_alpha', 0.03),
        ('α_b=0.05', 'film_alpha', 0.05),
    ]
    alpha_results = []
    for label, bc_type, ab in alpha_cases:
        # Reuse if already computed
        existing = [r for r in bc_results if r['bc_type'] == bc_type and r['alpha_b'] == ab]
        if existing:
            alpha_results.append(existing[0])
        else:
            r = run_case(times, gas_conc, bc_type, ab, label)
            alpha_results.append(r)

    # ── Mass balance (Fig 4) ──
    compute_mass_balance(times, gas_conc, alpha_b=0.03)

    # ── Print summary for plot_bc_results.py ──
    print("\n" + "=" * 70)
    print("DATA FOR plot_bc_results.py")
    print("=" * 70)

    def _join(results, key, fmt):
        return ', '.join(fmt.format(r[key]) for r in results)

    print("\n# BC comparison (Fig 1)")
    print("BC_DATA = {")
    print(f"    'labels': {[r['label'] for r in bc_results]},")
    print(f"    'pH':     [{_join(bc_results, 'pH', '{:.2f}')}],")
    print(f"    'NO3':    [{_join(bc_results, 'NO3', '{:.1f}')}],")
    print(f"    'NO2':    [{_join(bc_results, 'NO2', '{:.1f}')}],")
    print(f"    'H2O2':   [{_join(bc_results, 'H2O2', '{:.2f}')}],")
    print("}")

    print("\n# α_b sensitivity (Fig 3)")
    print("ALPHA_DATA = {")
    print(f"    'alpha_b': [{_join(alpha_results, 'alpha_b', '{}')}],")
    print(f"    'pH':      [{_join(alpha_results, 'pH', '{:.3f}')}],")
    print(f"    'NO3':     [{_join(alpha_results, 'NO3', '{:.1f}')}],")
    print(f"    'H2O2':    [{_join(alpha_results, 'H2O2', '{:.2f}')}],")
    print(f"    'OH':      [{_join(alpha_results, 'OH', '{:.3e}')}],")
    print(f"    'O3':      [{_join(alpha_results, 'O3', '{:.3e}')}],")
    print(f"    'HO2':     [{_join(alpha_results, 'HO2', '{:.3e}')}],")
    print("}")

    # Detailed radicals for Fig 3 table
    print("\n# Detailed radicals (Fig 3 table)")
    rad_species = [
        ('O3', 'O₃'), ('O2NOOH', 'O₂NOOH'), ('NO2_aq', 'NO₂'),
        ('ONOOH', 'ONOOH'), ('O2NOO-', 'O₂NOO⁻'),
        ('HO2', 'HO₂'), ('O2-', 'O₂⁻'), ('OH', 'OH'),
        ('ONOO-', 'ONOO⁻'), ('O3-', 'O₃⁻'), ('N2O5', 'N₂O₅'),
    ]
    for key, name in rad_species:
        vals = [r.get(key, 0) for r in alpha_results]
        print(f"  {name:12s}: " + ", ".join(f"{v:.3e}" for v in vals))


if __name__ == '__main__':
    main()
