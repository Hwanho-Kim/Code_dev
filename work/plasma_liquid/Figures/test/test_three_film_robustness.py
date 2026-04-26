#!/usr/bin/env python3
"""three_film BC robustness: grid + voltage sweep.

Test 1 (grid convergence):  3.2 kV Humid, dz_min ∈ {1, 5, 20} µm.
Test 2 (voltage transfer):  2.6/3.2/3.6 kV Humid, dz_min = 5 µm.

Includes gas_alpha baseline for comparison at each voltage.
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
# Experimental values from (P-L) 액체활성종 농도, pH, conductivity.xlsx (DIW sheet)
EXP_BY_V = {
    '2.6kV': {'pH': 5.09, 'NO3-': 32.63, 'NO2-': 0.00, 'H2O2': 4.76},
    '3.2kV': {'pH': 3.61, 'NO3-': 62.74, 'NO2-': 3.58, 'H2O2': 11.21},
    '3.6kV': {'pH': 3.25, 'NO3-': 70.42, 'NO2-': 20.74, 'H2O2': 16.25},
}


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


def apply_rh80(gas_dry, times):
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
    h2o2 = g['O3'] * 0.03
    return g, hono, hno3, h2o2


def run(voltage, bc_type, dz_min, t_end=600.0):
    times, gas_dry = load_gas(voltage)
    gas, hono, hno3, h2o2 = apply_rh80(gas_dry.copy(), times)

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=dz_min, stretch_ratio=1.12,
        saline_mode=False, bc_type=bc_type, alpha_b=None,
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
        'voltage': voltage, 'bc': bc_type, 'dz_min': dz_min,
        'pH_avg': result['pH_avg'], 'pH_surf': result['pH_surface'],
        'NO3-': avg.get('NO3-', 0.0) * 1e6,
        'NO2-': avg.get('NO2-', 0.0) * 1e6,
        'H2O2': avg.get('H2O2', 0.0) * 1e6,
        'O3': avg.get('O3', 0.0) * 1e6,
        'N_z': solver.N_z,
        'wall': wall,
    }


def fmt(r):
    return (f"  pH={r['pH_avg']:.3f} | NO3-={r['NO3-']:7.2f}µM | "
            f"NO2-={r['NO2-']:6.3f}µM | H2O2={r['H2O2']:7.2f}µM | "
            f"O3={r['O3']:.3e}µM | N_z={r['N_z']} | wall={r['wall']:.0f}s")


if __name__ == '__main__':
    print("=" * 110)
    print("TEST 1: Grid convergence — 3.2 kV Humid fitting, three_film, dz_min sweep")
    print("=" * 110)
    grid_results = {}
    for dz in [1e-6, 5e-6, 20e-6]:
        r = run('3.2kV', 'three_film', dz, t_end=600.0)
        grid_results[dz] = r
        print(f"[three_film, dz_min={dz*1e6:.0f}µm]")
        print(fmt(r))

    # grid convergence analysis
    print()
    print(f"{'Metric':8s} | " + " | ".join(f"dz={dz*1e6:.0f}µm" for dz in [1e-6, 5e-6, 20e-6]))
    print('-' * 70)
    for key in ['pH_avg', 'NO3-', 'NO2-', 'H2O2']:
        vals = [grid_results[dz][key] for dz in [1e-6, 5e-6, 20e-6]]
        print(f"{key:8s} | " + " | ".join(f"{v:12.3f}" for v in vals))

    print()
    print("=" * 110)
    print("TEST 2: Voltage sweep — Humid fitting, three_film vs gas_alpha, dz_min = 5 µm")
    print("=" * 110)
    volt_results = {}
    for v in ['2.6kV', '3.2kV', '3.6kV']:
        for bc in ['gas_alpha', 'three_film']:
            r = run(v, bc, 5e-6, t_end=600.0)
            volt_results[(v, bc)] = r
            print(f"[{v}, {bc}]")
            print(fmt(r))

    # Comparison table
    print()
    print(f"{'Voltage':8s} | {'BC':14s} | {'NO3- µM':>10s} | {'NO2- µM':>10s} | {'H2O2 µM':>10s} | {'pH':>6s}")
    print('-' * 100)
    for v in ['2.6kV', '3.2kV', '3.6kV']:
        for bc in ['gas_alpha', 'three_film']:
            r = volt_results[(v, bc)]
            print(f"{v:8s} | {bc:14s} | {r['NO3-']:10.2f} | {r['NO2-']:10.3f} | "
                  f"{r['H2O2']:10.2f} | {r['pH_avg']:6.3f}")
        if '3.2kV' == v:
            exp = EXP_BY_V['3.2kV']
            print(f"{v:8s} | {'EXPERIMENT':14s} | {exp['NO3-']:10.2f} | {exp['NO2-']:10.3f} | "
                  f"{exp['H2O2']:10.2f} | {exp['pH']:6.3f}")

    print()
    print("=" * 110)
    print("three_film / gas_alpha ratios:")
    print(f"{'Voltage':8s} | {'NO3-':>10s} | {'NO2-':>10s} | {'H2O2':>10s}")
    for v in ['2.6kV', '3.2kV', '3.6kV']:
        ga = volt_results[(v, 'gas_alpha')]
        tf = volt_results[(v, 'three_film')]
        print(f"{v:8s} | {tf['NO3-']/max(ga['NO3-'],1e-30):10.3f} | "
              f"{tf['NO2-']/max(ga['NO2-'],1e-30):10.3f} | "
              f"{tf['H2O2']/max(ga['H2O2'],1e-30):10.3f}")
