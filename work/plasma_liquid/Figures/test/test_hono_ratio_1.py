#!/usr/bin/env python3
"""
Quick test: Humid_fitting with HONO/NO2 = 1.0 (instead of 0.0071).
Compare against baseline Humid_fitting.
"""
import sys, time as time_mod
from pathlib import Path
import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ, GAS_TO_AQUEOUS_MAP
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

# --- Config ---
GAS_XLSX = _project_root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
SHEET = '3.2kV'
DZ_MIN = 5e-6
STRETCH = 1.12
BC_TYPE = 'gas_alpha'
DELTA_GAS = 0.01  # 10mm
DT_SNAP = 2.0
MIN_STABLE = 5

RH80 = {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054,
         'NO3_O3': 0.00442}
HONO2_RATIO = 0.83
H2O2_RATIO = 0.03

EXP = {'pH': 3.61, 'NO3-': 62.74, 'NO2-': 3.58, 'H2O2': 11.21}


def load_gas():
    df = pd.read_excel(GAS_XLSX, sheet_name=SHEET)
    times = df.iloc[:, 0].values.astype(float)
    gas = {}
    col_map = {'O3': 'O3', 'NO2': 'NO2', 'NO3': 'NO3', 'N2O5': 'N2O5'}
    for col_name, key in col_map.items():
        for c in df.columns:
            if col_name in str(c):
                gas[key] = df[c].values.astype(float)
                break

    # Onset filter
    for sp in gas:
        arr = gas[sp].copy()
        arr[arr < 0] = 0
        # find stable start
        count = 0
        start_idx = len(arr)
        for i in range(len(arr)):
            if arr[i] > 0:
                count += 1
                if count >= MIN_STABLE:
                    start_idx = i - MIN_STABLE + 1
                    break
            else:
                count = 0
        arr[:start_idx] = 0
        # linear interp zeros after start
        if start_idx < len(arr):
            nz = np.nonzero(arr)[0]
            if len(nz) > 1:
                arr = np.interp(np.arange(len(arr)), nz, arr[nz])
                arr[:start_idx] = np.linspace(0, arr[start_idx], start_idx + 1)[:-1]
        gas[sp] = arr

    # N2O4 equilibrium
    T = N2O4_EQ.REF_TEMP
    Kp = np.exp(
        np.log(N2O4_EQ.KP_298) +
        (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T)
    )
    gas['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (gas['NO2'] ** 2)

    return times, gas


def apply_rh80(gas, times, hono_ratio):
    """Apply RH80 scaling + given HONO/NO2 ratio."""
    r = RH80
    mask_ss = times >= (times[-1] - 100)
    def ss(arr): return max(np.mean(arr[mask_ss]), 1e-30)

    o3_ss_dry = ss(gas['O3'])
    o3_ss_80 = o3_ss_dry * r['O3_scale']
    no2_ss_dry = ss(gas['NO2'])
    no2_ss_80 = o3_ss_80 * r['NO2_O3']
    n2o5_ss_dry = ss(gas['N2O5'])
    n2o5_ss_80 = no2_ss_80 * r['N2O5_NO2']
    no3_ss_dry = ss(gas['NO3'])
    no3_ss_80 = o3_ss_80 * r['NO3_O3']

    g = {}
    g['O3'] = gas['O3'] * (o3_ss_80 / o3_ss_dry)
    g['NO2'] = gas['NO2'] * (no2_ss_80 / no2_ss_dry)
    g['N2O5'] = gas['N2O5'] * (n2o5_ss_80 / n2o5_ss_dry)
    g['NO3'] = gas['NO3'] * (no3_ss_80 / no3_ss_dry)

    T = N2O4_EQ.REF_TEMP
    Kp = np.exp(
        np.log(N2O4_EQ.KP_298) +
        (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T)
    )
    g['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (g['NO2'] ** 2)

    hono = g['NO2'] * hono_ratio
    hno3 = g['N2O5'] * HONO2_RATIO
    h2o2 = g['O3'] * H2O2_RATIO

    return g, hono, hno3, h2o2


def run_one(times, gas, hono, hno3, h2o2, label):
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type=BC_TYPE, alpha_b=None,
        delta_gas=DELTA_GAS,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=hono, hono2_gas=hno3, h2o2_gas=h2o2)
    t_end = float(times[-1])
    t_eval = np.arange(DT_SNAP, t_end + 0.1, DT_SNAP)
    t_eval = t_eval[t_eval <= t_end + 0.1]
    y0 = solver.build_initial_condition(initial_pH=7.0)

    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval, y0=y0,
                          verbose=True, dt_poisson=None)
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    print(f"\n=== {label} (wall={wall:.1f}s) ===")
    print(f"  pH       = {result['pH_avg']:.3f}  (exp {EXP['pH']})")
    print(f"  NO3-     = {avg['NO3-']*1e6:.2f} µM  (exp {EXP['NO3-']})")
    print(f"  NO2-     = {avg['NO2-']*1e6:.4f} µM  (exp {EXP['NO2-']})")
    print(f"  H2O2     = {avg['H2O2']*1e6:.2f} µM  (exp {EXP['H2O2']})")
    print(f"  O3       = {avg['O3']*1e9:.2f} nM")
    print(f"  OH       = {avg['OH']*1e12:.2f} pM")
    print(f"  ONOOH    = {avg.get('ONOOH', 0)*1e9:.3f} nM")

    # HONO SS input
    if isinstance(hono, np.ndarray):
        mask = times >= (times[-1] - 100)
        hono_ss = np.mean(hono[mask])
        print(f"  HONO gas SS = {hono_ss:.3e} cm-3 ({hono_ss/2.46e13:.1f} ppm)")

    return result


if __name__ == '__main__':
    times, gas_dry = load_gas()

    # --- Baseline: HONO/NO2 = 0.0071 (fitted) ---
    gas1, hono1, hno3_1, h2o2_1 = apply_rh80(gas_dry.copy(), times, hono_ratio=0.00707)
    run_one(times, gas1, hono1, hno3_1, h2o2_1, "Humid fitting (HONO/NO2=0.0071)")

    # --- Test: HONO/NO2 = 1.0 ---
    gas2, hono2, hno3_2, h2o2_2 = apply_rh80(gas_dry.copy(), times, hono_ratio=1.0)
    run_one(times, gas2, hono2, hno3_2, h2o2_2, "Humid fitting (HONO/NO2=1.0)")
