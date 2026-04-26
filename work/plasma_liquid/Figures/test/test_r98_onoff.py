#!/usr/bin/env python3
"""Step 0: R98 on/off diagnostic — is bulk N2O5 hydrolysis double-counting?

Tests:
  A. Baseline (gas_alpha + R98 on)
  B. R98 off (gas_alpha, N2O5 dissolves but no hydrolysis)
  C. N2O5 gas = 0 (only NO2/N2O4 hydrolysis + HNO3 direct uptake)
"""
import sys, time as time_mod, functools
from pathlib import Path
import numpy as np
import pandas as pd

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

print = functools.partial(print, flush=True)

GAS_XLSX = _root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
RH80 = {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054, 'NO3_O3': 0.00442}
EXP = {'pH': 3.61, 'NO3-': 62.74, 'NO2-': 3.58, 'H2O2': 11.21}


def load_gas():
    df = pd.read_excel(GAS_XLSX, sheet_name='3.2kV')
    times = df.iloc[:, 0].values.astype(float)
    gas = {}
    for cn, k in {'O3': 'O3', 'NO2': 'NO2', 'NO3': 'NO3', 'N2O5': 'N2O5'}.items():
        for c in df.columns:
            if cn in str(c):
                gas[k] = df[c].values.astype(float)
                break
    for sp in gas:
        arr = gas[sp].copy()
        arr[arr < 0] = 0
        cnt, si = 0, len(arr)
        for i in range(len(arr)):
            if arr[i] > 0:
                cnt += 1
                if cnt >= 5:
                    si = i - 4
                    break
            else:
                cnt = 0
        arr[:si] = 0
        if si < len(arr):
            nz = np.nonzero(arr)[0]
            if len(nz) > 1:
                arr = np.interp(np.arange(len(arr)), nz, arr[nz])
                arr[:si] = np.linspace(0, arr[si], si + 1)[:-1]
        gas[sp] = arr
    T = N2O4_EQ.REF_TEMP
    Kp = np.exp(np.log(N2O4_EQ.KP_298) +
                (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
    gas['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (gas['NO2'] ** 2)
    return times, gas


def apply_rh80(gas_dry, times):
    r = RH80
    mask = times >= (times[-1] - 100)
    ss = lambda a: max(np.mean(a[mask]), 1e-30)
    o3d, no2d, n2o5d, no3d = ss(gas_dry['O3']), ss(gas_dry['NO2']), ss(gas_dry['N2O5']), ss(gas_dry['NO3'])
    o3_80 = o3d * r['O3_scale']
    no2_80 = o3_80 * r['NO2_O3']
    n2o5_80 = no2_80 * r['N2O5_NO2']
    no3_80 = o3_80 * r['NO3_O3']
    g = {
        'O3': gas_dry['O3'] * (o3_80 / o3d),
        'NO2': gas_dry['NO2'] * (no2_80 / no2d),
        'N2O5': gas_dry['N2O5'] * (n2o5_80 / n2o5d),
        'NO3': gas_dry['NO3'] * (no3_80 / no3d),
    }
    T = N2O4_EQ.REF_TEMP
    Kp = np.exp(np.log(N2O4_EQ.KP_298) +
                (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
    g['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (g['NO2'] ** 2)
    hono = g['NO2'] * 0.00707
    hno3 = g['N2O5'] * 0.83
    h2o2 = g['O3'] * 0.03
    return g, hono, hno3, h2o2


def run(label, gas, times, hono, hno3, h2o2, r98_off=False, n2o5_zero=False):
    if n2o5_zero:
        gas = {k: (np.zeros_like(v) if k == 'N2O5' else v) for k, v in gas.items()}

    chem = AqueousChemistry1D(saline_mode=False)
    if r98_off:
        for rxn in chem.reactions:
            if 'R98' in rxn.get('label', ''):
                rxn['k'] = 0.0
                print(f"    R98 disabled: {rxn['label']}")
        chem._precompute_reaction_data()

    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type='gas_alpha', alpha_b=None,
        delta_gas=0.01,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=hono, hono2_gas=hno3, h2o2_gas=h2o2)
    t_end = float(times[-1])
    te = np.arange(2, t_end + 0.1, 2)
    te = te[te <= t_end + 0.1]
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0
    avg = result['spatial_avg']

    # N2O5(aq) surface concentration
    N_z, N_s = solver.N_z, solver.N_s
    y_final = result['y_final']
    if y_final.ndim == 1:
        y_final = y_final.reshape(N_z, N_s)
    n2o5_idx = solver.species_idx.get('N2O5', -1)
    n2o5_surface = y_final[0, n2o5_idx] if n2o5_idx >= 0 else 0.0
    n2o5_avg = avg.get('N2O5', 0.0)

    print(f"  {label:40s} | pH={result['pH_avg']:.3f} | "
          f"NO3⁻={avg['NO3-']*1e6:7.1f}µM | "
          f"NO2⁻={avg['NO2-']*1e6:7.4f}µM | "
          f"H2O2={avg['H2O2']*1e6:6.1f}µM | "
          f"N2O5(aq)_surf={n2o5_surface:.2e} | "
          f"N2O5(aq)_avg={n2o5_avg:.2e} | "
          f"{wall:.0f}s")


if __name__ == '__main__':
    times, gas_dry = load_gas()
    gas, hono, hno3, h2o2 = apply_rh80(gas_dry.copy(), times)

    print(f"TARGET: pH={EXP['pH']}, NO3⁻={EXP['NO3-']}µM\n")

    # A. Baseline
    print("--- A. Baseline (gas_alpha + R98 on) ---")
    run("Baseline", gas, times, hono, hno3, h2o2)

    # B. R98 off
    print("\n--- B. R98 off ---")
    run("R98 off", gas, times, hono, hno3, h2o2, r98_off=True)

    # C. N2O5 gas = 0
    print("\n--- C. N2O5 gas = 0 (other pathways only) ---")
    run("N2O5=0", gas, times, hono, hno3, h2o2, n2o5_zero=True)

    # D. R98 off + N2O5 = 0
    print("\n--- D. R98 off + N2O5 = 0 ---")
    run("R98 off + N2O5=0", gas, times, hono, hno3, h2o2, r98_off=True, n2o5_zero=True)
