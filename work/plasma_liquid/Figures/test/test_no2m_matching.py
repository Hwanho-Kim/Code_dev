#!/usr/bin/env python3
"""
Systematic NO₂⁻ matching: sweep physically tunable parameters.

NO₂⁻ budget (baseline):
  Sources: R19 (2NO₂ → NO₂⁻+NO₃⁻, 64%), R95 (N₂O₄+H₂O → NO₂⁻+NO₃⁻, 36%)
           HONO dissolution (acid-base → NO₂⁻, minor)
  Sinks:   R32 (O₃+NO₂⁻ → NO₃⁻, k=5e5, 97%), R92 (NO₃+NO₂⁻, 2.5%)

Approaches:
  A. R32 rate constant sweep (physical uncertainty: 3.5e5 ~ 5e5, test wider)
  B. Post-treatment (plasma-off) — O₃ decays, R32 stops → NO₂⁻ accumulates
  C. HONO ratio sweep (source enhancement)
  D. Combined: HONO + R32 reduction + post-treatment
"""
import sys, time as time_mod, itertools
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
T_TREAT = 600.0  # treatment time (s)

RH80 = {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054,
         'NO3_O3': 0.00442}
HONO2_RATIO = 0.83
H2O2_RATIO = 0.03

EXP = {'pH': 3.61, 'NO3-': 62.74e-6, 'NO2-': 3.58e-6, 'H2O2': 11.21e-6}


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


def apply_rh80(gas_dry, times, hono_ratio):
    r = RH80
    mask_ss = times >= (times[-1] - 100)
    def ss(arr): return max(np.mean(arr[mask_ss]), 1e-30)

    o3d, no2d, n2o5d, no3d = ss(gas_dry['O3']), ss(gas_dry['NO2']), ss(gas_dry['N2O5']), ss(gas_dry['NO3'])
    o3_80 = o3d * r['O3_scale']
    no2_80 = o3_80 * r['NO2_O3']
    n2o5_80 = no2_80 * r['N2O5_NO2']
    no3_80 = o3_80 * r['NO3_O3']

    g = {}
    g['O3']   = gas_dry['O3']  * (o3_80  / o3d)
    g['NO2']  = gas_dry['NO2'] * (no2_80 / no2d)
    g['N2O5'] = gas_dry['N2O5']* (n2o5_80/ n2o5d)
    g['NO3']  = gas_dry['NO3'] * (no3_80 / no3d)

    T = N2O4_EQ.REF_TEMP
    Kp = np.exp(np.log(N2O4_EQ.KP_298) +
                (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
    g['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (g['NO2'] ** 2)

    hono = g['NO2'] * hono_ratio
    hno3 = g['N2O5'] * HONO2_RATIO
    h2o2 = g['O3'] * H2O2_RATIO
    return g, hono, hno3, h2o2


def extend_gas_posttreatment(times, gas, hono, hno3, h2o2, t_extra):
    """Extend time series with gas=0 for post-treatment period."""
    if t_extra <= 0:
        return times, gas, hono, hno3, h2o2

    dt = float(times[1] - times[0]) if len(times) > 1 else 2.0
    n_extra = int(t_extra / dt)
    t_ext = np.arange(1, n_extra + 1) * dt + times[-1]
    times_ext = np.concatenate([times, t_ext])

    gas_ext = {}
    for sp in gas:
        gas_ext[sp] = np.concatenate([gas[sp], np.zeros(n_extra)])

    if isinstance(hono, np.ndarray):
        hono_ext = np.concatenate([hono, np.zeros(n_extra)])
    else:
        hono_ext = np.concatenate([np.full(len(times), float(hono)),
                                    np.zeros(n_extra)])
    if isinstance(hno3, np.ndarray):
        hno3_ext = np.concatenate([hno3, np.zeros(n_extra)])
    else:
        hno3_ext = np.concatenate([np.full(len(times), float(hno3)),
                                    np.zeros(n_extra)])
    if isinstance(h2o2, np.ndarray):
        h2o2_ext = np.concatenate([h2o2, np.zeros(n_extra)])
    else:
        h2o2_ext = np.concatenate([np.full(len(times), float(h2o2)),
                                    np.zeros(n_extra)])
    return times_ext, gas_ext, hono_ext, hno3_ext, h2o2_ext


def modify_r32(chem, scale_factor):
    """Scale R32 (O3+NO2- → NO3-) rate constant."""
    for rxn in chem.reactions:
        if 'label' in rxn and 'R32' in rxn['label']:
            rxn['k'] = 5.0e5 * scale_factor
            break
    chem._precompute_reaction_data()


def run_one(times, gas, hono, hno3, h2o2, label,
            r32_scale=1.0, delta_gas=DELTA_GAS):
    chem = AqueousChemistry1D(saline_mode=False)
    if r32_scale != 1.0:
        modify_r32(chem, r32_scale)

    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type=BC_TYPE, alpha_b=None,
        delta_gas=delta_gas,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=hono, hono2_gas=hno3, h2o2_gas=h2o2)
    t_end = float(times[-1])
    t_eval = np.arange(DT_SNAP, t_end + 0.1, DT_SNAP)
    t_eval = t_eval[t_eval <= t_end + 0.1]
    y0 = solver.build_initial_condition(initial_pH=7.0)

    print(f"    running {label}...", flush=True)
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval, y0=y0,
                          verbose=True, dt_poisson=None)
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    no2m = avg.get('NO2-', 0)
    no3m = avg.get('NO3-', 0)
    h2o2_val = avg.get('H2O2', 0)
    o3_val = avg.get('O3', 0)
    oh_val = avg.get('OH', 0)
    pH = result['pH_avg']

    return {
        'label': label,
        'pH': pH,
        'NO3-': no3m,
        'NO2-': no2m,
        'H2O2': h2o2_val,
        'O3': o3_val,
        'OH': oh_val,
        'wall_s': wall,
        'success': result['success'],
    }


def print_result(r):
    print(f"  {r['label']:50s} | pH={r['pH']:.3f} | "
          f"NO3⁻={r['NO3-']*1e6:7.2f}µM | "
          f"NO2⁻={r['NO2-']*1e6:7.4f}µM | "
          f"H2O2={r['H2O2']*1e6:6.2f}µM | "
          f"O3={r['O3']*1e9:6.1f}nM | "
          f"OH={r['OH']*1e12:5.2f}pM | "
          f"{r['wall_s']:.0f}s", flush=True)


if __name__ == '__main__':
    import functools
    print = functools.partial(print, flush=True)

    times_raw, gas_dry = load_gas()
    results = []

    # ==================================================================
    # A. R32 rate constant sweep (baseline Humid_fitting, HONO/NO2=0.0071)
    # ==================================================================
    print("\n" + "="*120)
    print("A. R32 rate constant sweep (k_R32 = 5e5 × factor)")
    print("="*120)

    gas_base, hono_b, hno3_b, h2o2_b = apply_rh80(gas_dry.copy(), times_raw, 0.00707)

    for factor in [1.0, 0.1, 0.01, 0.001]:
        label = f"R32×{factor}"
        r = run_one(times_raw, gas_base, hono_b, hno3_b, h2o2_b,
                    label, r32_scale=factor)
        results.append(r)
        print_result(r)

    # ==================================================================
    # B. Post-treatment (plasma-off) — gas=0 after 600s
    # ==================================================================
    print("\n" + "="*120)
    print("B. Post-treatment (plasma-off period, gas→0 after 600s)")
    print("="*120)

    for t_extra in [60, 120, 300, 600]:
        t_ext, g_ext, ho_ext, hn_ext, hp_ext = extend_gas_posttreatment(
            times_raw, gas_base.copy(), hono_b.copy(), hno3_b.copy(),
            h2o2_b.copy(), t_extra)
        label = f"Post-treatment +{t_extra}s"
        r = run_one(t_ext, g_ext, ho_ext, hn_ext, hp_ext, label)
        results.append(r)
        print_result(r)

    # ==================================================================
    # C. HONO ratio sweep (with baseline R32)
    # ==================================================================
    print("\n" + "="*120)
    print("C. HONO/NO₂ ratio sweep")
    print("="*120)

    for ratio in [0.00707, 0.1, 0.33, 1.0]:
        gas_c, hono_c, hno3_c, h2o2_c = apply_rh80(gas_dry.copy(), times_raw, ratio)
        label = f"HONO/NO2={ratio}"
        r = run_one(times_raw, gas_c, hono_c, hno3_c, h2o2_c, label)
        results.append(r)
        print_result(r)

    # ==================================================================
    # D. δ_gas sweep
    # ==================================================================
    print("\n" + "="*120)
    print("D. δ_gas sweep (mm)")
    print("="*120)

    for dg_mm in [1, 3, 5, 10, 20]:
        label = f"δ_gas={dg_mm}mm"
        r = run_one(times_raw, gas_base, hono_b, hno3_b, h2o2_b,
                    label, delta_gas=dg_mm * 1e-3)
        results.append(r)
        print_result(r)

    # ==================================================================
    # E. Combined: best candidates
    # ==================================================================
    print("\n" + "="*120)
    print("E. Combined parameter tests")
    print("="*120)

    combos = [
        # (HONO_ratio, R32_scale, t_extra, delta_gas_mm, label)
        (0.33, 0.1, 0, 10, "HONO=0.33 + R32×0.1"),
        (0.33, 0.01, 0, 10, "HONO=0.33 + R32×0.01"),
        (1.0,  0.1, 0, 10, "HONO=1.0 + R32×0.1"),
        (0.33, 1.0, 120, 10, "HONO=0.33 + post120s"),
        (0.33, 0.1, 120, 10, "HONO=0.33 + R32×0.1 + post120s"),
        (0.33, 1.0, 300, 10, "HONO=0.33 + post300s"),
        (0.33, 0.1, 300, 10, "HONO=0.33 + R32×0.1 + post300s"),
        (0.33, 0.01, 120, 10, "HONO=0.33 + R32×0.01 + post120s"),
    ]

    for hono_r, r32_s, t_ex, dg_mm, label in combos:
        gas_e, hono_e, hno3_e, h2o2_e = apply_rh80(gas_dry.copy(), times_raw, hono_r)
        if t_ex > 0:
            t_ext, g_ext, ho_ext, hn_ext, hp_ext = extend_gas_posttreatment(
                times_raw, gas_e, hono_e, hno3_e, h2o2_e, t_ex)
        else:
            t_ext, g_ext, ho_ext, hn_ext, hp_ext = times_raw, gas_e, hono_e, hno3_e, h2o2_e
        r = run_one(t_ext, g_ext, ho_ext, hn_ext, hp_ext,
                    label, r32_scale=r32_s, delta_gas=dg_mm*1e-3)
        results.append(r)
        print_result(r)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "="*120)
    print(f"TARGET: pH={EXP['pH']:.2f}, NO3⁻={EXP['NO3-']*1e6:.1f}µM, "
          f"NO2⁻={EXP['NO2-']*1e6:.2f}µM, H2O2={EXP['H2O2']*1e6:.1f}µM")
    print("="*120)
    print("\nTop matches by NO₂⁻ (closest to 3.58 µM):")
    ranked = sorted(results, key=lambda x: abs(x['NO2-'] - EXP['NO2-']))
    for i, r in enumerate(ranked[:10]):
        err = (r['NO2-'] - EXP['NO2-']) / EXP['NO2-'] * 100
        print(f"  #{i+1} {r['label']:50s} | NO2⁻={r['NO2-']*1e6:.4f}µM ({err:+.1f}%) | "
              f"pH={r['pH']:.3f} | NO3⁻={r['NO3-']*1e6:.1f}µM | H2O2={r['H2O2']*1e6:.1f}µM")
