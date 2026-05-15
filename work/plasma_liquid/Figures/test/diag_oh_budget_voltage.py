#!/usr/bin/env python3
"""Dump full OH rate budget (top sinks + sources) at t=600s for 2.6 vs 3.6 kV.

Answers:
1. Is R32+R77 the ONLY lever for OH voltage gap? -> identify other contributors.
2. After R32+R77 OFF, why 4.67x gap remains? -> compare top sinks/sources at same case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))
sys.path.insert(0, str(_proj_root / 'Figures'))

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
import gen_all_figures as gaf


OUT_DIR = Path(__file__).resolve().parent
PREFIX = 'r27_r32_r77_sweep'
TOP_N = 12


def load_cache(volt: str, case: str) -> dict:
    return dict(np.load(OUT_DIR / f'{PREFIX}_{volt}_{case}.npz',
                        allow_pickle=True))


def build_solver(volt: str, disable_idx: list[int]) -> PDESolver1D:
    """Construct the same solver/chemistry used during the sweep so we can
    re-evaluate per-reaction rates from y_final."""
    gaf.DEFAULT_GAS_SHEET = volt
    times, gas_conc = gaf.load_gas_data()
    rh80 = gaf.RH80_RATIOS.get(volt, {})
    h2o2_ratio = rh80.get('H2O2_O3', 0.003)
    hono2_ratio = rh80.get('HONO2_N2O5', 0.83)
    hono_ratio = rh80.get('HONO_NO2', 0.097)
    no2_arr = gas_conc.get('NO2', np.zeros_like(times))
    n2o5_arr = gas_conc.get('N2O5', np.zeros_like(times))
    o3_arr = gas_conc.get('O3', np.zeros_like(times))
    hono_gas = no2_arr * hono_ratio
    hono2_gas = n2o5_arr * hono2_ratio
    h2o2_gas = o3_arr * h2o2_ratio

    chem = AqueousChemistry1D(saline_mode=False)
    for ri in disable_idx:
        chem.reactions[ri]['k'] = 0.0
        chem._rxn_data[ri]['k'] = 0.0
    if disable_idx:
        chem._precompute_numba_arrays()

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6, stretch_ratio=1.12,
        mass_transfer_eta=1.0, saline_mode=False, fixed_cation_conc=0.0,
        bc_type='three_film', alpha_b=None, delta_gas=0.01,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hono2_gas,
                        h2o2_gas=h2o2_gas)
    return solver


def oh_budget(solver: PDESolver1D, y_2d: np.ndarray, t: float):
    """For every reaction, compute net dOH/dt contribution (mol/L/s, vol-avg).
    Positive = OH source, negative = OH sink."""
    chem = solver.chem
    rxn_rates, _mt = gaf.compute_rates_snapshot(solver, y_2d, t)
    rows = []
    for r in rxn_rates:
        reac = r['reactants']
        prod = r['products']
        stoich = prod.get('OH', 0) - reac.get('OH', 0)
        if stoich == 0:
            continue
        net = stoich * r['rate']
        rows.append((r['label'], net, r['rate'], stoich))
    rows.sort(key=lambda x: -abs(x[1]))
    return rows


def fmt_rows(rows, n=TOP_N):
    out = []
    for label, net, rate, stoich in rows[:n]:
        sign = '+' if net > 0 else '-'
        out.append(f'  {sign}{abs(net):<10.2e} (stoich={stoich:+d}, |rate|={rate:.2e})  {label}')
    return '\n'.join(out)


def main():
    # cases to analyze
    cases = [
        ('2.6kV', 'baseline', []),
        ('3.6kV', 'baseline', []),
        ('2.6kV', 'R32_R77_off', [23, 68]),
        ('3.6kV', 'R32_R77_off', [23, 68]),
    ]

    budgets = {}
    for volt, case, disable in cases:
        print(f'\n=== {volt} {case} ===', flush=True)
        d = load_cache(volt, case)
        y_final = d['snap_y'][-1]
        t_final = float(d['snap_t'][-1])
        solver = build_solver(volt, disable)
        rows = oh_budget(solver, y_final, t_final)
        budgets[(volt, case)] = rows

        net_total = sum(r[1] for r in rows)
        src = sum(r[1] for r in rows if r[1] > 0)
        snk = sum(r[1] for r in rows if r[1] < 0)
        print(f'  total OH net dydt = {net_total:+.3e} (source {src:+.3e} '
              f'+ sink {snk:+.3e})', flush=True)
        print(f'  top |contributions| (vol-avg M/s):\n{fmt_rows(rows)}',
              flush=True)

    # Cross-voltage comparison: which reactions scale most strongly with V?
    for case in ['baseline', 'R32_R77_off']:
        print(f'\n=== Voltage scaling of OH-rxn rates ({case}) ===', flush=True)
        rows_26 = {r[0]: r[1] for r in budgets[('2.6kV', case)]}
        rows_36 = {r[0]: r[1] for r in budgets[('3.6kV', case)]}
        all_labels = set(rows_26) | set(rows_36)
        scaled = []
        for lbl in all_labels:
            v26 = rows_26.get(lbl, 0.0)
            v36 = rows_36.get(lbl, 0.0)
            mag_max = max(abs(v26), abs(v36))
            if mag_max < 1e-15:
                continue
            ratio = v26 / v36 if v36 != 0 else float('inf')
            scaled.append((lbl, v26, v36, ratio, mag_max))
        # rank by magnitude
        scaled.sort(key=lambda x: -x[4])
        print(f'  {"label":<46} {"2.6kV":<11} {"3.6kV":<11} {"V26/V36":<8}',
              flush=True)
        for lbl, v26, v36, ratio, _ in scaled[:TOP_N]:
            r_str = f'{ratio:+.2e}' if abs(ratio) < 1e6 else 'inf'
            print(f'  {lbl[:46]:<46} {v26:+.2e}  {v36:+.2e}  {r_str:>8}',
                  flush=True)


if __name__ == '__main__':
    main()
