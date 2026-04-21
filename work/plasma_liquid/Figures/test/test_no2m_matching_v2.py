#!/usr/bin/env python3
"""
NO₂⁻ matching v2: 2-stage approach (treatment → plasma-off).

Key finding from v1: R32×0.001 gave same NO₂⁻ as baseline!
Need to verify R32 modification is actually working.

Strategy:
  1. Verify R32 modification works (diagnostic)
  2. 2-stage post-treatment: run treatment, then continue with gas=0
  3. HONO sweep with post-treatment
"""
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
SHEET = '3.2kV'
DZ_MIN = 5e-6
STRETCH = 1.12
BC_TYPE = 'gas_alpha'
DELTA_GAS = 0.01
DT_SNAP = 2.0
MIN_STABLE = 5

RH80 = {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054,
         'NO3_O3': 0.00442}
HONO2_RATIO = 0.83
H2O2_RATIO = 0.03

EXP = {'pH': 3.61, 'NO3-': 62.74e-6, 'NO2-': 3.58e-6, 'H2O2': 11.21e-6}


def load_gas():
    df = pd.read_excel(GAS_XLSX, sheet_name=SHEET)
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


def apply_rh80(gas_dry, times, hono_ratio):
    r = RH80
    mask_ss = times >= (times[-1] - 100)
    def ss(arr): return max(np.mean(arr[mask_ss]), 1e-30)
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
                (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
    g['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (g['NO2'] ** 2)
    return g, g['NO2'] * hono_ratio, g['N2O5'] * HONO2_RATIO, g['O3'] * H2O2_RATIO


def modify_r32(chem, scale_factor):
    """Scale R32 rate constant and verify."""
    for rxn in chem.reactions:
        if 'label' in rxn and 'R32' in rxn['label']:
            orig = float(rxn['k'])
            rxn['k'] = 5.0e5 * scale_factor
            print(f"    R32 k: {orig:.1e} → {float(rxn['k']):.1e}")
            break
    chem._precompute_reaction_data()


def extract_results(result, solver):
    avg = result['spatial_avg']
    N_z, N_s = solver.N_z, solver.N_s
    y_final = result['y_final']
    if y_final.ndim == 1:
        y_final = y_final.reshape(N_z, N_s)
    return {
        'pH': result['pH_avg'],
        'NO3-': avg.get('NO3-', 0),
        'NO2-': avg.get('NO2-', 0),
        'H2O2': avg.get('H2O2', 0),
        'O3': avg.get('O3', 0),
        'OH': avg.get('OH', 0),
        'y_final': y_final,
    }


def print_row(label, r):
    print(f"  {label:55s} | pH={r['pH']:.3f} | "
          f"NO3⁻={r['NO3-']*1e6:7.2f}µM | "
          f"NO2⁻={r['NO2-']*1e6:7.4f}µM | "
          f"H2O2={r['H2O2']*1e6:6.2f}µM | "
          f"O3={r['O3']*1e9:6.1f}nM | "
          f"OH={r['OH']*1e12:5.2f}pM")


# ==========================================================================
# Stage 1: Treatment (600s)
# ==========================================================================
def run_treatment(times, gas, hono, hno3, h2o2, r32_scale=1.0):
    chem = AqueousChemistry1D(saline_mode=False)
    if r32_scale != 1.0:
        modify_r32(chem, r32_scale)
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
    print(f"    Stage 1 done: {wall:.0f}s")
    return extract_results(result, solver), solver


# ==========================================================================
# Stage 2: Post-treatment (gas=0, continue from y_final)
# ==========================================================================
def run_posttreatment(solver, y_final_2d, t_extra, r32_scale=1.0):
    """Continue simulation with all gas = 0."""
    N_z, N_s = solver.N_z, solver.N_s

    # Set gas data to zeros
    dt = 2.0
    n_pts = max(int(t_extra / dt) + 1, 2)
    t_post = np.linspace(0, t_extra, n_pts)
    gas_zero = {sp: np.zeros(n_pts) for sp in ['O3', 'NO2', 'NO3', 'N2O5', 'N2O4']}

    # Create fresh solver with same config but gas=0
    chem = AqueousChemistry1D(saline_mode=False)
    if r32_scale != 1.0:
        modify_r32(chem, r32_scale)
    solver2 = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type=BC_TYPE, alpha_b=None,
        delta_gas=DELTA_GAS,
    )
    solver2.set_gas_data(times=t_post, gas_conc_molecules=gas_zero,
                         hono_gas=0.0, hono2_gas=0.0, h2o2_gas=0.0)

    y0 = y_final_2d.flatten()
    t_eval = np.arange(dt, t_extra + 0.1, dt)
    t_eval = t_eval[t_eval <= t_extra + 0.1]

    t0 = time_mod.time()
    result = solver2.solve(t_span=(0, t_extra), t_eval=t_eval, y0=y0,
                           verbose=True, dt_poisson=None)
    wall = time_mod.time() - t0
    print(f"    Stage 2 (+{t_extra}s) done: {wall:.0f}s")
    return extract_results(result, solver2)


if __name__ == '__main__':
    times_raw, gas_dry = load_gas()

    # ==================================================================
    # 0. Verify R32 modification actually works
    # ==================================================================
    print("\n" + "="*120)
    print("0. R32 modification verification")
    print("="*120)
    chem_test = AqueousChemistry1D(saline_mode=False)
    for rxn in chem_test.reactions:
        if 'label' in rxn and 'R32' in rxn['label']:
            print(f"  Original R32 k = {float(rxn['k']):.1e}")
            break
    modify_r32(chem_test, 0.001)
    for rxn in chem_test.reactions:
        if 'label' in rxn and 'R32' in rxn['label']:
            print(f"  Modified R32 k = {float(rxn['k']):.1e}")
            break
    # Check precomputed data
    for i, rd in enumerate(chem_test._rxn_data):
        label = chem_test.reactions[i].get('label', '')
        if 'R32' in label:
            print(f"  _rxn_data[{i}]: {rd}")
            break

    # ==================================================================
    # A. R32 sweep (re-run with verification)
    # ==================================================================
    print("\n" + "="*120)
    print("A. R32 rate constant sweep (with verification)")
    print("="*120)
    gas_base, hono_b, hno3_b, h2o2_b = apply_rh80(gas_dry.copy(), times_raw, 0.00707)

    for factor in [1.0, 0.1, 0.01]:
        print(f"\n--- R32 × {factor} ---")
        r, solver = run_treatment(times_raw, gas_base, hono_b, hno3_b, h2o2_b,
                                   r32_scale=factor)
        print_row(f"R32×{factor}", r)

    # ==================================================================
    # B. 2-stage post-treatment
    # ==================================================================
    print("\n" + "="*120)
    print("B. 2-stage post-treatment (600s treatment → gas=0)")
    print("="*120)

    # Run baseline treatment once
    print("\n--- Stage 1: Treatment (600s) ---")
    r_treat, solver_treat = run_treatment(times_raw, gas_base, hono_b, hno3_b, h2o2_b)
    print_row("End of treatment (t=600s)", r_treat)

    # Stage 2: various post-treatment durations
    for t_extra in [30, 60, 120, 300, 600]:
        print(f"\n--- Stage 2: Post-treatment +{t_extra}s ---")
        r_post = run_posttreatment(solver_treat, r_treat['y_final'], t_extra)
        print_row(f"Post-treatment +{t_extra}s (t={600+t_extra}s)", r_post)

    # ==================================================================
    # C. HONO sweep + post-treatment 120s
    # ==================================================================
    print("\n" + "="*120)
    print("C. HONO/NO₂ ratio sweep + post-treatment 120s")
    print("="*120)

    for ratio in [0.00707, 0.1, 0.33, 1.0]:
        print(f"\n--- HONO/NO₂ = {ratio} ---")
        gas_c, hono_c, hno3_c, h2o2_c = apply_rh80(gas_dry.copy(), times_raw, ratio)
        r_c, slv_c = run_treatment(times_raw, gas_c, hono_c, hno3_c, h2o2_c)
        print_row(f"HONO={ratio} (t=600s)", r_c)
        r_cp = run_posttreatment(slv_c, r_c['y_final'], 120)
        print_row(f"HONO={ratio} + post120s", r_cp)

    # ==================================================================
    # D. HONO + R32 reduction + post-treatment
    # ==================================================================
    print("\n" + "="*120)
    print("D. Combined: HONO + R32 reduction + post-treatment")
    print("="*120)

    combos = [
        (0.33, 0.1, 120),
        (0.33, 0.01, 120),
        (1.0, 0.1, 120),
        (0.33, 0.1, 300),
    ]
    for hono_r, r32_s, t_ex in combos:
        print(f"\n--- HONO={hono_r}, R32×{r32_s}, post+{t_ex}s ---")
        gas_d, hono_d, hno3_d, h2o2_d = apply_rh80(gas_dry.copy(), times_raw, hono_r)
        r_d, slv_d = run_treatment(times_raw, gas_d, hono_d, hno3_d, h2o2_d,
                                    r32_scale=r32_s)
        print_row(f"HONO={hono_r},R32×{r32_s} (t=600s)", r_d)
        r_dp = run_posttreatment(slv_d, r_d['y_final'], t_ex, r32_scale=r32_s)
        print_row(f"HONO={hono_r},R32×{r32_s},post+{t_ex}s", r_dp)

    # Summary
    print("\n" + "="*120)
    print(f"TARGET: pH={EXP['pH']:.2f}, NO3⁻={EXP['NO3-']*1e6:.1f}µM, "
          f"NO2⁻={EXP['NO2-']*1e6:.2f}µM, H2O2={EXP['H2O2']*1e6:.1f}µM")
    print("="*120)
