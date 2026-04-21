#!/usr/bin/env python3
"""
TPA / hTPA alkaline scenario — OH radical quantification via Terephthalic Acid probe.

Simulates plasma-liquid treatment of 2 mM TPA + 10 mM NaOH (pH ~11.5) solution
at 3 voltages (2.6 / 3.2 / 3.6 kVpp) for 600 s.

Output: cumulative [hTPA] bulk average → compared to experimental fluorescence.
    Experiment (inner filter ×2 corrected):
        2.6 kVpp → 12.66 µM
        3.2 kVpp → 57.72 µM
        3.6 kVpp → 43.26 µM

Usage:
    python run_tpa_alkaline.py --voltage 3.2kV
    python run_tpa_alkaline.py --voltage 3.2kV --tpa 0   # control (no TPA)
    python run_tpa_alkaline.py --all                     # 3 voltages × TPA on/off
"""

import argparse
import math
import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_script_dir))

from config_1d import (
    PHYSICAL, N2O4_EQ, DEFAULTS, ODE_CONFIG,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D


GAS_XLSX = _project_root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
CACHE_DIR = _project_root / 'Figures' / 'cache' / 'tpa'

# Grid / solver config — same as gen_all_figures.py reference
DZ_MIN = 5e-6
STRETCH = 1.12
REF_BC = 'gas_alpha'
REF_ALPHA = None         # None → species-specific α_b from config
REF_DELTA_GAS = 0.01     # 10 mm
DT_SNAPSHOT = 2.0

# Unmeasured gas ratios (RH 80% fitting, from gen_all_figures.py)
RH80_RATIOS = {
    '2.6kV': {'O3_scale': 0.493, 'NO2_O3': 0.222, 'N2O5_NO2': 0.043,
              'HONO_NO2': 0.00915, 'NO3_O3': 0.0179},
    '3.2kV': {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054,
              'HONO_NO2': 0.00707, 'NO3_O3': 0.00442},
    '3.6kV': {'O3_scale': 0.762, 'NO2_O3': 0.095, 'N2O5_NO2': 0.037,
              'HONO_NO2': 0.00662, 'NO3_O3': 0.00337},
}
HONO2_RATIO = 0.83   # HNO₃/N₂O₅
H2O2_RATIO = 0.03    # H₂O₂/O₃
MIN_STABLE_RUN = 5

EXPERIMENT = {
    '2.6kV': 12.66,   # [hTPA] µM (inner-filter corrected)
    '3.2kV': 57.72,
    '3.6kV': 43.26,
}


# ─────────────────────────────────────────────────────────────────────────────
# Gas data preprocessing (adopted from gen_all_figures.py)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_below_lod(vals: np.ndarray) -> np.ndarray:
    from scipy.signal import savgol_filter
    out = vals.astype(float).copy()
    n = len(vals)
    run_start, run_len, stable_start = -1, 0, n
    for i in range(n):
        if vals[i] > 0:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= MIN_STABLE_RUN:
                stable_start = run_start
                break
        else:
            run_len = 0
    if stable_start >= n:
        return np.maximum(out, 0.0)
    nz_after = [(i, vals[i]) for i in range(stable_start, n) if vals[i] > 0]
    if len(nz_after) >= 2:
        nz_idx = np.array([x[0] for x in nz_after])
        nz_vals = np.array([x[1] for x in nz_after])
        for i in range(stable_start, n):
            if out[i] <= 0:
                out[i] = np.interp(i, nz_idx, nz_vals)
    sg_win = 15
    stable_region = out[stable_start:]
    if len(stable_region) >= sg_win:
        w = sg_win if sg_win % 2 == 1 else sg_win + 1
        stable_region = savgol_filter(stable_region, window_length=w, polyorder=3)
        out[stable_start:] = np.maximum(stable_region, 0.0)
    first_val = out[stable_start]
    for i in range(stable_start):
        out[i] = first_val * (i / max(stable_start, 1))
    return np.maximum(out, 0.0)


def load_gas_data(voltage: str, condition: str = 'Humid_fitting'):
    """Load OAS gas data for given voltage, apply RH-80 ratios for unmeasured species.

    Returns: (times, gas_conc_dict, hono_array, hono2_array, h2o2_array)
    """
    df = pd.read_excel(GAS_XLSX, sheet_name=voltage, skiprows=1, header=None)
    df.columns = (['t', 'O3', 'NO2', 'NO3', 'N2O5', 'unit'][:df.shape[1]])
    times = pd.to_numeric(df['t'], errors='coerce').values.astype(float)

    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            raw = np.maximum(pd.to_numeric(df[col], errors='coerce').fillna(0).values, 0.0)
            gas_conc[col] = _preprocess_below_lod(raw)
        else:
            gas_conc[col] = np.zeros(len(times))

    # N2O4 equilibrium
    if np.all(gas_conc.get('N2O4', np.array([0])) == 0):
        T = 298.15
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298)
            + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (gas_conc['NO2'] ** 2)

    r = RH80_RATIOS.get(voltage)
    if condition == 'Humid_fitting' and r is not None:
        mask_ss = times >= (times[-1] - 100)

        def ss(arr):
            return max(np.mean(arr[mask_ss]), 1e-30)

        o3_ss_dry = ss(gas_conc['O3'])
        o3_ss_80 = o3_ss_dry * r['O3_scale']
        no2_ss_80 = o3_ss_80 * r['NO2_O3']
        n2o5_ss_80 = no2_ss_80 * r['N2O5_NO2']
        no3_ss_80 = o3_ss_80 * r['NO3_O3']

        gas_conc['O3'] = gas_conc['O3'] * (o3_ss_80 / o3_ss_dry)
        gas_conc['NO2'] = gas_conc['NO2'] * (no2_ss_80 / ss(gas_conc['NO2']))
        gas_conc['N2O5'] = gas_conc['N2O5'] * (n2o5_ss_80 / ss(gas_conc['N2O5']))
        gas_conc['NO3'] = gas_conc['NO3'] * (no3_ss_80 / ss(gas_conc['NO3']))
        hono_gas = gas_conc['NO2'] * r['HONO_NO2']
    else:
        hono_gas = gas_conc['NO2'] * 0.33  # Humid median fallback

    hono2_gas = gas_conc['N2O5'] * HONO2_RATIO
    h2o2_gas = gas_conc['O3'] * H2O2_RATIO

    return times, gas_conc, hono_gas, hono2_gas, h2o2_gas


# ─────────────────────────────────────────────────────────────────────────────
# Initial condition for TPA/NaOH system
# ─────────────────────────────────────────────────────────────────────────────

def build_tpa_initial_condition(solver: PDESolver1D,
                                tpa_conc: float = 2e-3,
                                initial_pH: float = 12.0) -> np.ndarray:
    """2 mM TPA²⁻ (disodium terephthalate) + 10 mM NaOH initial state.

    [OH⁻] = 10 mM (from NaOH) → pH = 12 initial (slight drift expected).
    [Na⁺] = 14 mM passed via solver.fixed_cation_conc (NaOH 10 + 2·TPA 4).
    """
    y0 = solver.build_initial_condition(initial_pH=initial_pH)
    trace = DEFAULTS.trace_concentration
    tpa_idx = solver.species_idx.get('TPA', -1)
    htpa_idx = solver.species_idx.get('hTPA', -1)
    for j in range(solver.N_z):
        off = j * solver.N_s
        if tpa_idx >= 0:
            y0[off + tpa_idx] = tpa_conc
        if htpa_idx >= 0:
            y0[off + htpa_idx] = trace
    return y0


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_single(voltage: str, tpa_conc: float = 2e-3,
               initial_pH: float = 12.0, rerun: bool = False,
               verbose: bool = True, condition: str = 'Humid_fitting'):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tpa_tag = f"tpa{int(tpa_conc*1e6):d}uM" if tpa_conc > 0 else "notpa"
    cond_tag = condition.lower().replace('_', '')
    key = f"{voltage}_{tpa_tag}_{cond_tag}"
    cache_file = CACHE_DIR / f"{key}.npz"

    if cache_file.exists() and not rerun:
        print(f"[{key}] loading from cache")
        return dict(np.load(cache_file, allow_pickle=True))

    print(f"\n{'='*70}")
    print(f"TPA alkaline run: voltage={voltage}, [TPA]₀={tpa_conc*1e3:.2f} mM")
    print(f"{'='*70}")

    times, gas_conc, hono_gas, hono2_gas, h2o2_gas = load_gas_data(voltage, condition=condition)
    print(f"  Gas data: N={len(times)}, t=[{times[0]:.0f}, {times[-1]:.0f}] s")
    print(f"  O3 max = {gas_conc['O3'].max():.2e} cm⁻³")
    print(f"  NO2 max = {gas_conc['NO2'].max():.2e} cm⁻³")
    print(f"  N2O5 max = {gas_conc['N2O5'].max():.2e} cm⁻³")

    # Charge balance: Na⁺ = 10 mM (NaOH) + 2 × [TPA²⁻] + 2 × [hTPA²⁻]₀=0 = 10 + 2·2 = 14 mM
    na_conc = 10e-3 + 2.0 * tpa_conc

    chem = AqueousChemistry1D(saline_mode=False, tpa_mode=(tpa_conc > 0))
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=DZ_MIN,
        stretch_ratio=STRETCH,
        mass_transfer_eta=1.0,
        saline_mode=False,
        fixed_cation_conc=na_conc,
        bc_type=REF_BC,
        alpha_b=REF_ALPHA,
        delta_gas=REF_DELTA_GAS,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=hono_gas, hono2_gas=hono2_gas, h2o2_gas=h2o2_gas,
    )

    y0 = build_tpa_initial_condition(solver, tpa_conc=tpa_conc,
                                     initial_pH=initial_pH)
    t_end = float(times[-1])
    t_eval = np.arange(DT_SNAPSHOT, t_end + 0.1, DT_SNAPSHOT)
    t_eval = t_eval[t_eval <= t_end + 0.1]

    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval, y0=y0,
        verbose=verbose, dt_poisson=None,
    )
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    pH = result['pH_avg']
    htpa_uM = avg.get('hTPA', 0.0) * 1e6
    tpa_uM = avg.get('TPA', 0.0) * 1e6
    oh_M = avg.get('OH', 0.0)
    o3_M = avg.get('O3', 0.0)
    ho2m_M = avg.get('HO2-', 0.0)
    h2o2_M = avg.get('H2O2', 0.0)

    # OH equivalent from measured hTPA (experimental convention)
    oh_equiv_uM = htpa_uM / 0.35

    exp_htpa = EXPERIMENT.get(voltage, None)
    err = (htpa_uM - exp_htpa) / exp_htpa * 100 if exp_htpa else None

    print()
    print(f"  Wall: {wall:.1f} s | pH_avg = {pH:.3f} | success = {result['success']}")
    print(f"  [TPA]  : {tpa_uM*1e3:.1f} µM ({tpa_uM/(tpa_conc*1e6)*100:.1f}% of initial)"
          if tpa_conc > 0 else f"  [TPA]  : -- (control)")
    print(f"  [hTPA] : {htpa_uM:.2f} µM")
    print(f"  [OH]_total (hTPA/0.35) : {oh_equiv_uM:.2f} µM")
    if exp_htpa:
        print(f"  Experiment           : {exp_htpa:.2f} µM hTPA")
        print(f"  Error                : {err:+.1f}%")
    print(f"  [OH]      = {oh_M:.3e} M")
    print(f"  [O3]      = {o3_M:.3e} M")
    print(f"  [HO2-]    = {ho2m_M:.3e} M")
    print(f"  [H2O2]    = {h2o2_M:.3e} M")

    # Extract TPA + hTPA spatial profiles
    N_z, N_s = solver.N_z, solver.N_s
    snap_t = np.array([0.0] + [float(tv) for tv in result['t_eval']])
    snap_y = [y0.reshape(N_z, N_s).copy()]
    for yv in result['y_eval']:
        snap_y.append(np.array(yv).reshape(N_z, N_s))
    snap_y_arr = np.array(snap_y)

    data = {
        'voltage': voltage, 'tpa_conc': np.float64(tpa_conc),
        'pH_avg': np.float64(pH),
        'pH_surface': np.float64(result.get('pH_surface', 0)),
        'TPA_uM': np.float64(tpa_uM), 'hTPA_uM': np.float64(htpa_uM),
        'OH_equiv_uM': np.float64(oh_equiv_uM),
        'OH_M': np.float64(oh_M), 'O3_M': np.float64(o3_M),
        'HO2m_M': np.float64(ho2m_M), 'H2O2_M': np.float64(h2o2_M),
        'wall_s': np.float64(wall),
        'success': np.bool_(result['success']),
        'z_centers': solver.z_centers, 'dz_cells': solver.dz_cells,
        'L': np.float64(solver.L),
        'N_z': np.int64(N_z), 'N_s': np.int64(N_s),
        'snap_t': snap_t, 'snap_y': snap_y_arr,
        'species_idx_keys': np.array(list(solver.species_idx.keys())),
        'species_idx_vals': np.array(list(solver.species_idx.values())),
    }
    np.savez_compressed(cache_file, **data)
    print(f"  cached → {cache_file.name}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voltage', default='3.2kV',
                    choices=['2.6kV', '3.2kV', '3.6kV'])
    ap.add_argument('--tpa', type=float, default=2e-3,
                    help='Initial [TPA] in M (0 for control)')
    ap.add_argument('--pH', type=float, default=12.0, dest='initial_pH')
    ap.add_argument('--condition', default='Humid_fitting',
                    choices=['Dry', 'Humid_fitting', 'Humid_median'])
    ap.add_argument('--all', action='store_true',
                    help='Run all 3 voltages × (TPA on, off)')
    ap.add_argument('--rerun', action='store_true')
    args = ap.parse_args()

    if args.all:
        rows = []
        for v in ['2.6kV', '3.2kV', '3.6kV']:
            for tpa in [2e-3, 0.0]:
                d = run_single(v, tpa_conc=tpa, initial_pH=args.initial_pH,
                               rerun=args.rerun, verbose=False,
                               condition=args.condition)
                rows.append({
                    'voltage': v, 'tpa_mM': tpa * 1e3,
                    'pH': float(d['pH_avg']),
                    'hTPA_uM': float(d['hTPA_uM']),
                    'OH_equiv_uM': float(d['OH_equiv_uM']),
                    'exp_uM': EXPERIMENT.get(v, None),
                })
        print("\n" + "="*70)
        print(f"SUMMARY (condition={args.condition})")
        print("="*70)
        print(f"{'V':6} {'[TPA]₀':>8} {'pH':>7} {'hTPA':>10} {'OH_eq':>10} {'Exp':>8}")
        for r in rows:
            exp = f"{r['exp_uM']:.2f}" if r['exp_uM'] else '--'
            print(f"{r['voltage']:6} {r['tpa_mM']:7.2f} {r['pH']:7.3f} "
                  f"{r['hTPA_uM']:9.3f} {r['OH_equiv_uM']:9.3f} {exp:>8}")
    else:
        run_single(args.voltage, tpa_conc=args.tpa,
                   initial_pH=args.initial_pH, rerun=args.rerun,
                   condition=args.condition)


if __name__ == '__main__':
    main()
