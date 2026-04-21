#!/usr/bin/env python3
"""Quick 3-voltage Dry comparison after Henry constant bug fix."""
import sys, time as time_mod, functools
from pathlib import Path
import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

print = functools.partial(print, flush=True)

GAS_XLSX = _project_root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
DZ_MIN = 5e-6
STRETCH = 1.12
BC_TYPE = 'gas_alpha'
DELTA_GAS = 0.01
DT_SNAP = 2.0
MIN_STABLE = 5

EXP = {
    '2.6kV': {'pH': 5.09, 'NO2-': 0, 'NO3-': 32.63e-6, 'H2O2': 4.76e-6},
    '3.2kV': {'pH': 3.61, 'NO2-': 3.58e-6, 'NO3-': 62.74e-6, 'H2O2': 11.21e-6},
    '3.6kV': {'pH': 3.25, 'NO2-': 20.74e-6, 'NO3-': 70.42e-6, 'H2O2': 16.25e-6},
}


def load_gas(sheet):
    df = pd.read_excel(GAS_XLSX, sheet_name=sheet)
    times = df.iloc[:, 0].values.astype(float)
    gas = {}
    for col_name, key in {'O3': 'O3', 'NO2': 'NO2', 'NO3': 'NO3', 'N2O5': 'N2O5'}.items():
        for c in df.columns:
            if col_name in str(c):
                gas[key] = df[c].values.astype(float)
                break
    for sp in gas:
        arr = gas[sp].copy()
        arr[arr < 0] = 0
        count, start_idx = 0, len(arr)
        for i in range(len(arr)):
            if arr[i] > 0:
                count += 1
                if count >= MIN_STABLE:
                    start_idx = i - MIN_STABLE + 1
                    break
            else:
                count = 0
        arr[:start_idx] = 0
        if start_idx < len(arr):
            nz = np.nonzero(arr)[0]
            if len(nz) > 1:
                arr = np.interp(np.arange(len(arr)), nz, arr[nz])
                arr[:start_idx] = np.linspace(0, arr[start_idx], start_idx + 1)[:-1]
        gas[sp] = arr
    T = N2O4_EQ.REF_TEMP
    Kp = np.exp(np.log(N2O4_EQ.KP_298) +
                (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
    gas['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (gas['NO2'] ** 2)
    return times, gas


def run_one(sheet):
    times, gas = load_gas(sheet)
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type=BC_TYPE, alpha_b=None,
        delta_gas=DELTA_GAS,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=0.0, hono2_gas=0.0, h2o2_gas=0.0)
    t_end = float(times[-1])
    t_eval = np.arange(DT_SNAP, t_end + 0.1, DT_SNAP)
    t_eval = t_eval[t_eval <= t_end + 0.1]
    y0 = solver.build_initial_condition(initial_pH=7.0)

    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval, y0=y0,
                          verbose=True, dt_poisson=None)
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    return {
        'pH': result['pH_avg'],
        'NO3-': avg.get('NO3-', 0),
        'NO2-': avg.get('NO2-', 0),
        'H2O2': avg.get('H2O2', 0),
        'O3': avg.get('O3', 0),
        'OH': avg.get('OH', 0),
        'wall_s': wall,
    }


if __name__ == '__main__':
    print("="*100)
    print("3-voltage Dry comparison (after Henry constant fix)")
    print("="*100)

    for sheet in ['2.6kV', '3.2kV', '3.6kV']:
        print(f"\n--- {sheet} ---")
        r = run_one(sheet)
        e = EXP[sheet]
        print(f"  {'':20s} {'Sim':>10s}  {'Exp':>10s}  {'Error':>10s}")
        print(f"  {'pH':20s} {r['pH']:10.3f}  {e['pH']:10.2f}  {(r['pH']-e['pH'])/e['pH']*100:+8.1f}%")
        print(f"  {'NO3- (µM)':20s} {r['NO3-']*1e6:10.2f}  {e['NO3-']*1e6:10.2f}  {(r['NO3-']-e['NO3-'])/max(e['NO3-'],1e-10)*100:+8.1f}%")
        no2m_err = 'N/A' if e['NO2-'] == 0 else f"{(r['NO2-']-e['NO2-'])/e['NO2-']*100:+.1f}%"
        print(f"  {'NO2- (µM)':20s} {r['NO2-']*1e6:10.4f}  {e['NO2-']*1e6:10.2f}  {no2m_err:>10s}")
        print(f"  {'H2O2 (µM)':20s} {r['H2O2']*1e6:10.4f}  {e['H2O2']*1e6:10.2f}  {(r['H2O2']-e['H2O2'])/max(e['H2O2'],1e-10)*100:+8.1f}%")
        print(f"  {'O3 (nM)':20s} {r['O3']*1e9:10.1f}")
        print(f"  {'OH (pM)':20s} {r['OH']*1e12:10.2f}")
        print(f"  Wall time: {r['wall_s']:.0f}s")
