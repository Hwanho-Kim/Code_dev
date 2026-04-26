#!/usr/bin/env python3
"""Baseline (gas_alpha) vs three_film BC comparison.

3.2 kV Humid fitting, 600s. Verify hypothesis:
  - Adding liquid film (1/(H·k_L)) to Schwartz series
  - Expected: dramatic reduction for low-H species (O3/NO/NO2)
             modest reduction for mid-H (N2O5/NO3)
             negligible change for high-H (HNO3/H2O2)
"""
import sys, functools, time as time_mod
from pathlib import Path
import numpy as np
import pandas as pd

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D, compute_k_mt, MOLAR_MASS
from config_1d import HENRY_CONSTANTS, GAS_DIFFUSIVITY, LIQUID_DIFFUSIVITY

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


def print_k_mt_comparison(delta_gas, delta_liq):
    """Print K per species for each BC type."""
    species_list = ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5', 'HONO', 'HONO2', 'H2O2']
    alpha_dict = {'O3': 0.05, 'NO': 0.001, 'NO2': 0.03, 'NO3': 0.03,
                  'N2O4': 0.03, 'N2O5': 0.03, 'HONO': 0.05,
                  'HONO2': 0.07, 'H2O2': 0.1}
    print(f"\n{'='*110}")
    print(f"K [m/s] per species  (δ_gas={delta_gas*1000:.1f}mm, δ_liq={delta_liq*1e6:.1f}µm)")
    print(f"{'='*110}")
    print(f"{'Species':8s} | {'α':6s} | {'H_cc':12s} | "
          f"{'K_gas_alpha':15s} | {'K_three_film':15s} | {'Ratio (3f/ga)':14s}")
    print('-' * 110)
    for sp in species_list:
        alpha = alpha_dict[sp]
        H = HENRY_CONSTANTS.get(sp, 1.0)
        k_ga = compute_k_mt(sp, delta_gas, delta_liq,
                            bc_type='gas_alpha', alpha_b=alpha)
        k_3f = compute_k_mt(sp, delta_gas, delta_liq,
                            bc_type='three_film', alpha_b=alpha)
        ratio = k_3f / max(k_ga, 1e-30)
        print(f"{sp:8s} | {alpha:6.3f} | {H:12.4e} | "
              f"{k_ga:15.4e} | {k_3f:15.4e} | {ratio:14.4e}")


def run(label, gas, times, hono, hno3, h2o2, bc_type='gas_alpha', t_end=600.0):
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
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
    surf = result['surface']

    print(f"\n{label}:")
    print(f"  BC={bc_type}, pH_avg={result['pH_avg']:.3f}, pH_surf={result['pH_surface']:.3f}, wall={wall:.1f}s")
    print(f"  {'':20s} | {'Avg (µM)':>12s} | {'Surface (µM)':>14s} | {'Exp (µM)':>10s}")
    for sp, exp_val in [('NO3-', 62.74), ('NO2-', 3.58), ('H2O2', 11.21)]:
        avg_uM = avg.get(sp, 0.0) * 1e6
        surf_uM = surf.get(sp, 0.0) * 1e6
        ratio = avg_uM / exp_val if exp_val > 0 else 0
        print(f"  {sp:20s} | {avg_uM:12.2f} | {surf_uM:14.2f} | "
              f"{exp_val:10.2f} (×{ratio:.2f})")
    # Additional neutral species
    for sp in ['O3', 'H2O2_total', 'HONO_total', 'HONO2_total']:
        avg_uM = avg.get(sp, 0.0) * 1e6
        surf_uM = surf.get(sp, 0.0) * 1e6
        print(f"  {sp:20s} | {avg_uM:12.3e} | {surf_uM:14.3e} |")
    return result


if __name__ == '__main__':
    times, gas_dry = load_gas()
    gas, hono, hno3, h2o2 = apply_rh80(gas_dry.copy(), times)

    print(f"Target experimental: pH={EXP['pH']}, NO3-={EXP['NO3-']}µM, "
          f"NO2-={EXP['NO2-']}µM, H2O2={EXP['H2O2']}µM")

    # Print K comparison first
    print_k_mt_comparison(0.01, 1e-4)

    # Run baseline
    print(f"\n{'='*110}")
    print(f"Running baseline (gas_alpha) — 3.2 kV Humid fitting 600s")
    print(f"{'='*110}")
    r_ga = run("[gas_alpha]", gas, times, hono, hno3, h2o2,
               bc_type='gas_alpha', t_end=600.0)

    # Run three_film
    print(f"\n{'='*110}")
    print(f"Running three_film — 3.2 kV Humid fitting 600s")
    print(f"{'='*110}")
    r_3f = run("[three_film]", gas, times, hono, hno3, h2o2,
               bc_type='three_film', t_end=600.0)

    # Summary
    print(f"\n{'='*110}")
    print(f"SUMMARY")
    print(f"{'='*110}")
    print(f"{'Metric':12s} | {'Target':>10s} | {'gas_alpha':>12s} | {'three_film':>12s} | "
          f"{'Δ(3f/ga)':>10s} | {'Δ(3f/exp)':>10s}")
    print('-' * 110)
    avg_ga, avg_3f = r_ga['spatial_avg'], r_3f['spatial_avg']
    for sp, exp_val in [('NO3-', 62.74), ('NO2-', 3.58), ('H2O2', 11.21)]:
        ga_val = avg_ga.get(sp, 0.0) * 1e6
        tf_val = avg_3f.get(sp, 0.0) * 1e6
        r_3f_exp = tf_val / exp_val if exp_val > 0 else 0
        r_3f_ga = tf_val / ga_val if ga_val > 0 else 0
        print(f"{sp:12s} | {exp_val:10.2f} | {ga_val:12.2f} | {tf_val:12.2f} | "
              f"{r_3f_ga:10.3f} | {r_3f_exp:10.3f}")
    print(f"pH_avg       | {EXP['pH']:10.3f} | {r_ga['pH_avg']:12.3f} | "
          f"{r_3f['pH_avg']:12.3f} |")
