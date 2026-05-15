#!/usr/bin/env python3
"""Run 3.6 kV DIW Humid_fitting three_film with R32 / R77 / both disabled.

Goal: identify the lever responsible for ~1700x OH suppression at 3.6 kV vs 2.6 kV.
- R32 (O3 + NO2- -> O2 + NO3-, idx=23, k=5e5): indirect lever via O3 depletion
- R77 (OH + NO2- -> OH- + NO2, idx=68, k=1e9): direct OH scavenger

Uses N_z=49 (stretch=1.12) for speed; mechanism comparison is relative so grid
should not change qualitative ordering. Baseline (all ON) is re-run on the same
grid to ensure apples-to-apples comparison.
"""
from __future__ import annotations

import sys
import time as time_mod
from pathlib import Path

import numpy as np

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))
sys.path.insert(0, str(_proj_root / 'Figures'))

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
import gen_all_figures as gaf


VOLTS = ['2.6kV', '3.6kV']
T_END = 600.0
DT_SNAP = 2.0
OUT_DIR = Path(__file__).resolve().parent
CACHE_PREFIX = 'r27_r32_r77_sweep'

# R27: O3+OH (idx 18, OH main sink in fig2b)
# R32: O3+NO2- (idx 23, NO2-mediated O3 depletion)
# R77: OH+NO2- (idx 68, direct OH scavenger by NO2-)
CASES = {
    'baseline':   [],
    'R27_off':    [18],
    'R32_off':    [23],
    'R77_off':    [68],
    'R32_R77_off': [23, 68],
}


def run_case(volt: str, case_name: str, disable_idx: list[int],
             rerun: bool = False) -> dict:
    cache = OUT_DIR / f'{CACHE_PREFIX}_{volt}_{case_name}.npz'
    if cache.exists() and not rerun:
        print(f'[{volt} {case_name}] loading cache {cache.name}', flush=True)
        return dict(np.load(cache, allow_pickle=True))

    print(f'\n[{volt} {case_name}] disabling reactions: {disable_idx}',
          flush=True)
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
        lbl = chem.reactions[ri].get('label', '')
        print(f'  disabling idx={ri}: {lbl}', flush=True)
        chem.reactions[ri]['k'] = 0.0
        chem._rxn_data[ri]['k'] = 0.0
    if disable_idx:
        chem._precompute_numba_arrays()

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        fixed_cation_conc=0.0,
        bc_type='three_film',
        alpha_b=None,
        delta_gas=0.01,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hono2_gas,
                        h2o2_gas=h2o2_gas)
    print(f'  N_z={solver.N_z}, N_s={solver.N_s}', flush=True)

    t_eval = np.arange(DT_SNAP, T_END + 0.1, DT_SNAP)
    y0 = solver.build_initial_condition(initial_pH=7.0)

    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, T_END), t_eval=t_eval, y0=y0,
        verbose=False, dt_poisson=None,
    )
    wall = time_mod.time() - t0
    print(f'  done: wall={wall:.1f}s, success={result["success"]}, '
          f'nfev={result.get("nfev", 0)}', flush=True)

    snap_t = np.array([0.0] + [float(tv) for tv in result['t_eval']])
    N_z, N_s = solver.N_z, solver.N_s
    snap_y = [y0.reshape(N_z, N_s).copy()]
    for yv in result['y_eval']:
        snap_y.append(np.array(yv).reshape(N_z, N_s))
    snap_y = np.array(snap_y)

    out = {
        'snap_t': snap_t, 'snap_y': snap_y,
        'z_centers': solver.z_centers, 'dz_cells': solver.dz_cells,
        'N_z': np.int64(N_z), 'N_s': np.int64(N_s), 'L': np.float64(solver.L),
        'wall_s': np.float64(wall),
        'disabled_idx': np.array(disable_idx, dtype=np.int64),
    }
    np.savez_compressed(cache, **out)
    print(f'  cached: {cache.name}', flush=True)
    return out


def summarize(volt: str, results: dict[str, dict]) -> None:
    chem = AqueousChemistry1D(saline_mode=False)
    Ka_HONO = 7.1e-4
    idx = chem.species_idx
    print('\n' + '='*78, flush=True)
    print(f'Final state at t=600s — vol-avg, three_film, {volt}', flush=True)
    print('='*78, flush=True)
    header = (f'{"case":<14} {"[OH] M":<11} {"[O3] M":<11} '
              f'{"[NO2-] M":<11} {"[HO2] M":<11} {"pH":<6}')
    print(header, flush=True)
    print('-'*78, flush=True)
    OH_base = None
    for name, d in results.items():
        y = d['snap_y'][-1]
        dz = d['dz_cells']
        L = float(d['L'])

        def vol_avg(arr):
            return float((arr * dz).sum() / L)

        OH = vol_avg(y[:, idx['OH']])
        O3 = vol_avg(y[:, idx['O3']])
        HO2t = vol_avg(y[:, idx['HO2_total']])
        HONOt = y[:, idx['HONO_total']]
        Hp = y[:, idx['H+']]
        NO2m = vol_avg(HONOt * Ka_HONO / (Hp + Ka_HONO))
        Hp_avg = vol_avg(Hp)
        pH = -np.log10(max(Hp_avg, 1e-14))
        print(f'{name:<14} {OH:<11.3e} {O3:<11.3e} {NO2m:<11.3e} '
              f'{HO2t:<11.3e} {pH:<6.2f}', flush=True)
        if name == 'baseline':
            OH_base = OH

    print()
    for name, d in results.items():
        if name == 'baseline':
            continue
        y = d['snap_y'][-1]
        dz = d['dz_cells']
        L = float(d['L'])
        OH = float((y[:, idx['OH']] * dz).sum() / L)
        ratio = OH / OH_base if OH_base else float('nan')
        print(f'  [OH]_{name} / [OH]_baseline = {ratio:.2e}', flush=True)


def main():
    print(f'Sweep voltages: {VOLTS}', flush=True)
    print(f'Cases: {list(CASES.keys())}', flush=True)
    print(f'Grid: dz_min=5um, stretch=1.12 (N_z=49) for speed.', flush=True)
    all_results = {}
    for volt in VOLTS:
        results = {}
        for name, disable_idx in CASES.items():
            results[name] = run_case(volt, name, disable_idx, rerun=False)
        all_results[volt] = results
        summarize(volt, results)

    print('\n' + '='*78, flush=True)
    print('Cross-voltage comparison: [OH]_vol_avg at t=600s', flush=True)
    print('='*78, flush=True)
    chem = AqueousChemistry1D(saline_mode=False)
    i_OH = chem.species_idx['OH']
    hdr = f'{"case":<14}' + ''.join(f'{v:<14}' for v in VOLTS) + 'low/high'
    print(hdr, flush=True)
    print('-'*78, flush=True)
    for name in CASES:
        row = f'{name:<14}'
        vals = []
        for volt in VOLTS:
            d = all_results[volt][name]
            y = d['snap_y'][-1]
            dz = d['dz_cells']
            L = float(d['L'])
            OH = float((y[:, i_OH] * dz).sum() / L)
            vals.append(OH)
            row += f'{OH:<14.3e}'
        row += f'{vals[0]/vals[-1]:<.2e}' if vals[-1] > 0 else 'n/a'
        print(row, flush=True)


if __name__ == '__main__':
    main()
