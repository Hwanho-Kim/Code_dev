#!/usr/bin/env python3
"""HONO/NO2 and HONO2/N2O5 ratio sweep.

three_film BC + H2O2/O3=0.003 fixed. 3 voltages × 2 sweeps × 4 ratios = 24 sims.

HONO/NO2 ∈ {0.007, 0.03, 0.07, 0.1}   (Sakiyama 2012 range 0.01-0.1)
HONO2/N2O5 ∈ {0.83, 2, 3, 5}          (0.83 current; deep research: humid >1, ~3-5)
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
# Voltage-specific RH80 ratios (matches gen_all_figures.py)
RH80_ALL = {
    '2.6kV': {'O3_scale': 0.493, 'NO2_O3': 0.222, 'N2O5_NO2': 0.043, 'HONO_NO2_default': 0.00915, 'NO3_O3': 0.0179},
    '3.2kV': {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054, 'HONO_NO2_default': 0.00707, 'NO3_O3': 0.00442},
    '3.6kV': {'O3_scale': 0.762, 'NO2_O3': 0.095, 'N2O5_NO2': 0.037, 'HONO_NO2_default': 0.00662, 'NO3_O3': 0.00337},
}
EXP = {
    '2.6kV': {'pH': 5.09, 'NO3': 32.63, 'NO2':  0.00, 'H2O2':  4.76},
    '3.2kV': {'pH': 3.61, 'NO3': 62.74, 'NO2':  3.58, 'H2O2': 11.21},
    '3.6kV': {'pH': 3.25, 'NO3': 70.42, 'NO2': 20.74, 'H2O2': 16.25},
}

H2O2_RATIO = 0.003
DEFAULT_HONO2_RATIO = 0.83

HONO_RATIOS = [0.007, 0.03, 0.07, 0.1]
HONO2_RATIOS = [0.83, 2.0, 3.0, 5.0]
VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']


def _preprocess(vals, min_run=5):
    out = np.maximum(vals.copy(), 0.0)
    n = len(out)
    run_start, run_len, stable_start = -1, 0, n
    for i in range(n):
        if out[i] > 0:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= min_run:
                stable_start = run_start
                break
        else:
            run_len = 0
    if stable_start >= n:
        return out
    nz = [(i, out[i]) for i in range(stable_start, n) if out[i] > 0]
    if len(nz) >= 2:
        idx = np.array([x[0] for x in nz])
        vs = np.array([x[1] for x in nz])
        for i in range(stable_start, n):
            if out[i] <= 0:
                out[i] = np.interp(i, idx, vs)
    if stable_start > 0:
        first = out[stable_start]
        out[:stable_start] = np.linspace(0, first, stable_start + 1)[:-1]
    return out


def load_gas(voltage):
    df = pd.read_excel(GAS_XLSX, sheet_name=voltage)
    times = df.iloc[:, 0].values.astype(float)
    gas = {}
    for sp in ['O3', 'NO2', 'NO3', 'N2O5']:
        for col in df.columns:
            if sp in str(col):
                gas[sp] = _preprocess(df[col].values.astype(float))
                break
    return times, gas


def apply_rh80(gas_dry, times, voltage, hono_ratio, hono2_ratio):
    r = RH80_ALL[voltage]
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
    # Unmeasured (swept)
    hono = g['NO2'] * hono_ratio
    hno3 = g['N2O5'] * hono2_ratio
    h2o2 = g['O3'] * H2O2_RATIO
    return g, hono, hno3, h2o2


def run(voltage, hono_ratio, hono2_ratio, t_end=600.0):
    times, gas_dry = load_gas(voltage)
    gas, hono, hno3, h2o2 = apply_rh80(gas_dry, times, voltage,
                                        hono_ratio, hono2_ratio)
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
        'pH':   result['pH_avg'],
        'NO3':  avg.get('NO3-', 0.0) * 1e6,
        'NO2':  avg.get('NO2-', 0.0) * 1e6,
        'H2O2': avg.get('H2O2', 0.0) * 1e6,
        'wall': wall,
    }


def print_table(title, sweep_key, sweep_values, fixed_label, results):
    print(f"\n{'='*115}")
    print(f"{title}  (fixed: {fixed_label})")
    print('=' * 115)
    for metric in ['pH', 'NO3', 'NO2', 'H2O2']:
        print(f"\n{metric}:")
        header = f"  {'V':>6} |" + "".join(f" {sweep_key}={r:>6.3f} |" for r in sweep_values) + f" {'EXP':>8} |"
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for v in VOLTAGES:
            exp = EXP[v][metric]
            row = f"  {v:>6} |"
            for r in sweep_values:
                val = results[(v, r)][metric]
                row += f" {val:13.3f} |"
            row += f" {exp:8.2f} |"
            print(row)


if __name__ == '__main__':
    # === Sweep 1: HONO/NO2 ===
    print("="*115)
    print(f"Sweep 1: HONO/NO2 ratio  (HONO2/N2O5 fixed at {DEFAULT_HONO2_RATIO})")
    print("="*115)
    hono_results = {}
    for v in VOLTAGES:
        for r in HONO_RATIOS:
            print(f"  Running [{v}, HONO={r}]...", end=' ')
            res = run(v, r, DEFAULT_HONO2_RATIO)
            hono_results[(v, r)] = res
            print(f"NO2-={res['NO2']:7.3f}, NO3-={res['NO3']:6.2f}, "
                  f"pH={res['pH']:.3f}, H2O2={res['H2O2']:6.2f}, {res['wall']:.0f}s")

    print_table("Sweep 1: HONO/NO2", "HONO", HONO_RATIOS,
                f"HONO2/N2O5={DEFAULT_HONO2_RATIO}", hono_results)

    # === Sweep 2: HONO2/N2O5 ===
    print("\n" + "="*115)
    print(f"Sweep 2: HONO2/N2O5 ratio  (HONO/NO2 = voltage-default)")
    print("="*115)
    hono2_results = {}
    for v in VOLTAGES:
        default_hono = RH80_ALL[v]['HONO_NO2_default']
        for r in HONO2_RATIOS:
            print(f"  Running [{v}, HONO2={r}, HONO={default_hono}]...", end=' ')
            res = run(v, default_hono, r)
            hono2_results[(v, r)] = res
            print(f"NO2-={res['NO2']:7.3f}, NO3-={res['NO3']:6.2f}, "
                  f"pH={res['pH']:.3f}, H2O2={res['H2O2']:6.2f}, {res['wall']:.0f}s")

    print_table("Sweep 2: HONO2/N2O5", "HONO2", HONO2_RATIOS,
                "HONO/NO2 = voltage-default", hono2_results)
