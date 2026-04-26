#!/usr/bin/env python3
"""H2O2/O3 ratio sweep — how low does ratio need to go to fit experiment?

BC: three_film (adopted). dz_min=5µm. 3 voltages × 4 ratios = 12 sims.

Ratios:
  0.03  — current default (mid literature Sakiyama 2012)
  0.01  — Sakiyama 2012 dry air lower bound
  0.003 — below lit, toward fit
  0.001 — forced fit test
"""
import sys, functools, time as time_mod
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
EXP_BY_V = {
    '2.6kV': {'pH': 5.09, 'NO3-': 32.63, 'NO2-': 0.00, 'H2O2': 4.76},
    '3.2kV': {'pH': 3.61, 'NO3-': 62.74, 'NO2-': 3.58, 'H2O2': 11.21},
    '3.6kV': {'pH': 3.25, 'NO3-': 70.42, 'NO2-': 20.74, 'H2O2': 16.25},
}

RATIOS = [0.03, 0.01, 0.003, 0.001]
VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']


def load_gas(sheet):
    df = pd.read_excel(GAS_XLSX, sheet_name=sheet)
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
    Kp = np.exp(np.log(N2O4_EQ.KP_298)
                + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
    gas['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (gas['NO2'] ** 2)
    return times, gas


def apply_rh80_with_ratio(gas_dry, times, h2o2_ratio):
    r = RH80
    mask = times >= (times[-1] - 100)
    ss = lambda a: max(np.mean(a[mask]), 1e-30)
    o3d = ss(gas_dry['O3'])
    no2d = ss(gas_dry['NO2'])
    n2o5d = ss(gas_dry['N2O5'])
    no3d = ss(gas_dry['NO3'])
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
    Kp = np.exp(np.log(N2O4_EQ.KP_298)
                + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
    g['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (g['NO2'] ** 2)
    hono = g['NO2'] * 0.00707
    hno3 = g['N2O5'] * 0.83
    h2o2 = g['O3'] * h2o2_ratio
    return g, hono, hno3, h2o2


def run(voltage, h2o2_ratio, t_end=600.0):
    times, gas_dry = load_gas(voltage)
    gas, hono, hno3, h2o2 = apply_rh80_with_ratio(gas_dry.copy(), times, h2o2_ratio)
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type='three_film', alpha_b=None,
        delta_gas=0.01, delta_liq=1e-4,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=hono, hono2_gas=hno3, h2o2_gas=h2o2)
    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0
    avg = result['spatial_avg']
    return {
        'voltage': voltage, 'ratio': h2o2_ratio,
        'pH': result['pH_avg'],
        'NO3-': avg.get('NO3-', 0.0) * 1e6,
        'NO2-': avg.get('NO2-', 0.0) * 1e6,
        'H2O2': avg.get('H2O2', 0.0) * 1e6,
        'wall': wall,
    }


if __name__ == '__main__':
    results = {}
    print("=" * 110)
    print(f"H2O2/O3 ratio sweep (three_film BC, dz_min=5µm, 600s)")
    print("=" * 110)
    for v in VOLTAGES:
        for r in RATIOS:
            print(f"Running [{v}, ratio={r}]...", end=' ')
            res = run(v, r)
            results[(v, r)] = res
            print(f"H2O2={res['H2O2']:7.2f}µM, NO3-={res['NO3-']:6.2f}µM, "
                  f"pH={res['pH']:.3f}, wall={res['wall']:.0f}s")

    # H2O2 table
    print()
    print("=" * 110)
    print(f"{'H2O2 [µM]':>12s} |" + "".join(f" ratio={r:>6.3f} |" for r in RATIOS)
          + f" {'EXP':>7s} |")
    print('-' * 110)
    for v in VOLTAGES:
        exp = EXP_BY_V[v]['H2O2']
        row = f"{v:>12s} |"
        for r in RATIOS:
            val = results[(v, r)]['H2O2']
            row += f" {val:12.2f} |"
        row += f" {exp:7.2f} |"
        print(row)

    print()
    print("=" * 110)
    print(f"{'H2O2 sim/exp':>12s} |" + "".join(f" ratio={r:>6.3f} |" for r in RATIOS))
    print('-' * 100)
    for v in VOLTAGES:
        exp = EXP_BY_V[v]['H2O2']
        row = f"{v:>12s} |"
        for r in RATIOS:
            val = results[(v, r)]['H2O2']
            row += f" {val/exp:12.3f} |"
        print(row)

    # NO3- check — should be unchanged (H2O2 path independent)
    print()
    print("=" * 110)
    print("NO3- [µM] (should be essentially unchanged across ratios)")
    print(f"{'Voltage':>12s} |" + "".join(f" ratio={r:>6.3f} |" for r in RATIOS)
          + f" {'EXP':>7s} |")
    print('-' * 110)
    for v in VOLTAGES:
        exp = EXP_BY_V[v]['NO3-']
        row = f"{v:>12s} |"
        for r in RATIOS:
            val = results[(v, r)]['NO3-']
            row += f" {val:12.2f} |"
        row += f" {exp:7.2f} |"
        print(row)

    # pH
    print()
    print("=" * 110)
    print("pH_avg")
    print(f"{'Voltage':>12s} |" + "".join(f" ratio={r:>6.3f} |" for r in RATIOS)
          + f" {'EXP':>7s} |")
    print('-' * 110)
    for v in VOLTAGES:
        exp = EXP_BY_V[v]['pH']
        row = f"{v:>12s} |"
        for r in RATIOS:
            val = results[(v, r)]['pH']
            row += f" {val:12.3f} |"
        row += f" {exp:7.2f} |"
        print(row)

    # Recommended ratio per voltage (fit H2O2)
    print()
    print("=" * 110)
    print("Required ratio for exact H2O2 fit (linear scaling, H2O2 ∝ ratio)")
    print(f"{'Voltage':>12s} | {'Current ratio=0.03':>20s} | "
          f"{'Implied ratio':>15s} | {'Reduction':>10s}")
    print('-' * 100)
    for v in VOLTAGES:
        exp = EXP_BY_V[v]['H2O2']
        base = results[(v, 0.03)]['H2O2']
        implied = 0.03 * exp / base if base > 0 else 0
        print(f"{v:>12s} | {base:20.2f} | {implied:15.5f} | "
              f"{0.03/implied if implied > 0 else 0:10.1f}×")
