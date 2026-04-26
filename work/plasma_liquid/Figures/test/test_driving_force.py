#!/usr/bin/env python3
"""Driving force ratio diagnostic:
    ratio(t) = (C_eq(t) - c_eff_surface(t)) / C_eq(t)

  ratio → 1: no saturation, full driving force (gas-side limited + fast liquid sink)
  ratio → 0: liquid surface saturates at C_eq (liquid-side limited or slow bulk removal)

Output per-species time series for diagnosis."""
import sys, functools, time as time_mod
from pathlib import Path
import numpy as np
import pandas as pd

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))

from config_1d import (PHYSICAL, N2O4_EQ, HENRY_CONSTANTS, ACID_BASE_PAIRS,
                        GAS_TO_AQUEOUS_MAP, MASS_TRANSFER)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

print = functools.partial(print, flush=True)

GAS_XLSX = _root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
RH80 = {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054, 'NO3_O3': 0.00442}


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


def run_and_extract(t_end=60.0):
    times, gas_dry = load_gas()
    gas, hono, hno3, h2o2 = apply_rh80(gas_dry.copy(), times)

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type='gas_alpha', alpha_b=None,
        delta_gas=0.01,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=hono, hono2_gas=hno3, h2o2_gas=h2o2)

    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0
    print(f"Simulation {t_end}s completed in {wall:.1f}s")

    return solver, result, te


def compute_ratios(solver, result, te):
    """Compute driving force ratios from y_eval time series."""
    y_eval = result['y_eval']  # list of y arrays at te
    N_z, N_s = solver.N_z, solver.N_s

    # Gas species to analyze (all entries in GAS_TO_AQUEOUS_MAP)
    species_list = ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5', 'HONO', 'HONO2', 'H2O2']

    # Get Ka and aq_idx for each
    sp_info = {}
    for gas_sp in species_list:
        aq_sp = GAS_TO_AQUEOUS_MAP[gas_sp]
        if aq_sp not in solver.species_idx:
            continue
        aq_idx = solver.species_idx[aq_sp]
        H_cc = HENRY_CONSTANTS.get(gas_sp, 1.0)
        Ka = None
        if aq_sp in ACID_BASE_PAIRS:
            pKa = ACID_BASE_PAIRS[aq_sp][2]
            Ka = 10.0 ** (-pKa)
        sp_info[gas_sp] = (aq_idx, H_cc, Ka)

    h_idx = solver.species_idx.get('H+', -1)

    # Extract per time
    print(f"\n{'='*110}")
    print(f"Driving force ratio = (C_eq - c_eff_surf) / C_eq")
    print(f"  1.0 → full driving force (no surface saturation)")
    print(f"  0.0 → surface at Henry equilibrium (fully saturated)")
    print(f"{'='*110}")
    header = f"{'t [s]':>6s} |" + "".join(f" {sp:>18s} |" for sp in sp_info.keys())
    print(header)
    sub = f"{'':6s} |" + "".join(f" {'ratio(C_surf/C_eq)':>18s} |" for _ in sp_info)
    print(sub)
    print('-' * len(header))

    for i, t in enumerate(te):
        y = y_eval[i]
        if y.ndim == 1:
            y = y.reshape(N_z, N_s)
        h_surf = y[0, h_idx] if h_idx >= 0 else 1e-7
        row = f"{t:>6.1f} |"
        for gas_sp, (aq_idx, H_cc, Ka) in sp_info.items():
            C_eq = H_cc * solver._get_C_eq_fast(gas_sp, t) / H_cc  # code does H*c_gas, but we want same
            # Actually _get_C_eq_fast returns H*c_gas already
            # Let me re-get it properly
            C_eq = solver._get_C_eq_fast(gas_sp, t)
            c0 = y[0, aq_idx]
            c_eff = c0 * h_surf / (h_surf + Ka) if Ka is not None else c0
            if C_eq > 1e-30:
                ratio = (C_eq - c_eff) / C_eq
                csurf_over_ceq = c_eff / C_eq
            else:
                ratio = 1.0
                csurf_over_ceq = 0.0
            row += f" {csurf_over_ceq:18.3e} |"
        print(row)

    print()
    # Last: also print driving force ratios
    print(f"{'='*110}")
    print(f"Driving force = 1 - c_surf/C_eq   (1 = full, 0 = saturated)")
    print(f"{'='*110}")
    header = f"{'t [s]':>6s} |" + "".join(f" {sp:>18s} |" for sp in sp_info.keys())
    print(header)
    sub = f"{'':6s} |" + "".join(f" {'1 - c_surf/C_eq':>18s} |" for _ in sp_info)
    print(sub)
    print('-' * len(header))
    for i, t in enumerate(te):
        y = y_eval[i]
        if y.ndim == 1:
            y = y.reshape(N_z, N_s)
        h_surf = y[0, h_idx] if h_idx >= 0 else 1e-7
        row = f"{t:>6.1f} |"
        for gas_sp, (aq_idx, H_cc, Ka) in sp_info.items():
            C_eq = solver._get_C_eq_fast(gas_sp, t)
            c0 = y[0, aq_idx]
            c_eff = c0 * h_surf / (h_surf + Ka) if Ka is not None else c0
            if C_eq > 1e-30:
                ratio = (C_eq - c_eff) / C_eq
            else:
                ratio = 1.0
            row += f" {ratio:18.4f} |"
        print(row)

    # Print absolute values too
    print(f"\n{'='*110}")
    print(f"Absolute values at last time step:")
    print(f"{'='*110}")
    t = te[-1]
    y = y_eval[-1]
    if y.ndim == 1:
        y = y.reshape(N_z, N_s)
    h_surf = y[0, h_idx] if h_idx >= 0 else 1e-7
    print(f"t = {t}s, pH_surface = {-np.log10(max(h_surf,1e-14)):.3f}")
    for gas_sp, (aq_idx, H_cc, Ka) in sp_info.items():
        C_eq = solver._get_C_eq_fast(gas_sp, t)
        c0 = y[0, aq_idx]
        c_eff = c0 * h_surf / (h_surf + Ka) if Ka is not None else c0
        c_gas_OAS = C_eq / H_cc
        print(f"  {gas_sp:8s}: c_gas(OAS)={c_gas_OAS:.3e} M | "
              f"H={H_cc:.3e} | C_eq={C_eq:.3e} M | "
              f"c_total_surf={c0:.3e} M | c_eff_surf={c_eff:.3e} M | "
              f"driving_force={(C_eq-c_eff)/max(C_eq,1e-30):.4f}")


if __name__ == '__main__':
    solver, result, te = run_and_extract(t_end=60.0)
    compute_ratios(solver, result, te)
