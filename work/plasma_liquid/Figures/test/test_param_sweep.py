#!/usr/bin/env python3
"""
Multi-parameter sweep to match 3.2kV DIW experimental data.
Humid fitting base, vary: δ_gas, H2O2/O3, HONO2/N2O5, HONO/NO2.

Strategy:
  A. δ_gas sweep → find NO3- matching range
  B. H2O2/O3 ratio sweep at promising δ_gas values
  C. HONO2/N2O5 ratio sweep
  D. HONO/NO2 ratio sweep
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
MIN_STABLE = 5

RH80 = {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054,
         'NO3_O3': 0.00442}

EXP = {'pH': 3.61, 'NO3-': 62.74, 'NO2-': 3.58, 'H2O2': 11.21}


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


def apply_rh80(gas_dry, times, hono_ratio=0.00707, hono2_ratio=0.83, h2o2_ratio=0.03):
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
    hono = g['NO2'] * hono_ratio
    hno3 = g['N2O5'] * hono2_ratio
    h2o2 = g['O3'] * h2o2_ratio
    return g, hono, hno3, h2o2


def run(times, gas, hono, hno3, h2o2, delta_gas, label):
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type=BC_TYPE, alpha_b=None,
        delta_gas=delta_gas,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=hono, hono2_gas=hno3, h2o2_gas=h2o2)
    t_end = float(times[-1])
    t_eval = np.arange(2.0, t_end + 0.1, 2.0)
    t_eval = t_eval[t_eval <= t_end + 0.1]
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval, y0=y0,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0
    avg = result['spatial_avg']
    r = {
        'label': label,
        'pH': result['pH_avg'],
        'NO3-': avg.get('NO3-', 0) * 1e6,
        'NO2-': avg.get('NO2-', 0) * 1e6,
        'H2O2': avg.get('H2O2', 0) * 1e6,
        'O3': avg.get('O3', 0) * 1e9,
        'OH': avg.get('OH', 0) * 1e12,
        'wall': wall,
    }
    return r


HDR = f"  {'Label':45s} | {'pH':>6s} | {'NO3⁻':>8s} | {'NO2⁻':>8s} | {'H2O2':>8s} | {'O3':>7s} | {'OH':>6s} | {'t':>4s}"
SEP = "  " + "-"*110

def pr(r):
    print(f"  {r['label']:45s} | {r['pH']:6.3f} | {r['NO3-']:7.1f}µ | {r['NO2-']:7.4f}µ | {r['H2O2']:7.2f}µ | {r['O3']:6.1f}n | {r['OH']:5.2f}p | {r['wall']:3.0f}s")


if __name__ == '__main__':
    times_raw, gas_dry = load_gas()
    all_results = []

    print(f"\nTARGET: pH={EXP['pH']}, NO3⁻={EXP['NO3-']}µM, NO2⁻={EXP['NO2-']}µM, H2O2={EXP['H2O2']}µM")

    # ==========================================================================
    # A. δ_gas sweep (humid fitting base)
    # ==========================================================================
    print("\n" + "="*115)
    print("A. δ_gas sweep (HONO/NO2=0.007, HONO2/N2O5=0.83, H2O2/O3=0.03)")
    print("="*115)
    print(HDR)
    print(SEP)

    for dg_mm in [10, 20, 30, 50, 70, 100]:
        gas, ho, hn, hp = apply_rh80(gas_dry.copy(), times_raw)
        r = run(times_raw, gas, ho, hn, hp, dg_mm*1e-3, f"δg={dg_mm}mm")
        all_results.append(r)
        pr(r)

    # ==========================================================================
    # B. H2O2/O3 ratio sweep (at δ_gas=50mm which should be near NO3- match)
    # ==========================================================================
    print("\n" + "="*115)
    print("B. H2O2/O3 ratio sweep (δ_gas=50mm)")
    print("="*115)
    print(HDR)
    print(SEP)

    for h2o2_r in [0.03, 0.01, 0.005, 0.002, 0.001]:
        gas, ho, hn, hp = apply_rh80(gas_dry.copy(), times_raw, h2o2_ratio=h2o2_r)
        r = run(times_raw, gas, ho, hn, hp, 0.05, f"δg=50mm, H2O2/O3={h2o2_r}")
        all_results.append(r)
        pr(r)

    # ==========================================================================
    # C. HONO2/N2O5 ratio sweep (at δ_gas=50mm, H2O2/O3=0.005)
    # ==========================================================================
    print("\n" + "="*115)
    print("C. HONO2/N2O5 ratio sweep (δ_gas=50mm, H2O2/O3=0.005)")
    print("="*115)
    print(HDR)
    print(SEP)

    for hno3_r in [0.83, 0.5, 0.3, 0.1, 0.0]:
        gas, ho, hn, hp = apply_rh80(gas_dry.copy(), times_raw,
                                      hono2_ratio=hno3_r, h2o2_ratio=0.005)
        r = run(times_raw, gas, ho, hn, hp, 0.05, f"δg=50, HNO3/N2O5={hno3_r}, H2O2=0.005")
        all_results.append(r)
        pr(r)

    # ==========================================================================
    # D. HONO/NO2 ratio sweep (at δ_gas=50mm, H2O2/O3=0.005)
    # ==========================================================================
    print("\n" + "="*115)
    print("D. HONO/NO2 ratio sweep (δ_gas=50mm, H2O2/O3=0.005)")
    print("="*115)
    print(HDR)
    print(SEP)

    for hono_r in [0.00707, 0.05, 0.1, 0.33, 1.0]:
        gas, ho, hn, hp = apply_rh80(gas_dry.copy(), times_raw,
                                      hono_ratio=hono_r, h2o2_ratio=0.005)
        r = run(times_raw, gas, ho, hn, hp, 0.05, f"δg=50, HONO/NO2={hono_r}, H2O2=0.005")
        all_results.append(r)
        pr(r)

    # ==========================================================================
    # E. Fine-tuning combos
    # ==========================================================================
    print("\n" + "="*115)
    print("E. Fine-tuning combinations")
    print("="*115)
    print(HDR)
    print(SEP)

    combos = [
        # (δg_mm, hono_r, hno3_r, h2o2_r, label)
        (50,  0.33, 0.83, 0.003, "δ50 HO=0.33 HN=0.83 HP=0.003"),
        (50,  0.33, 0.5,  0.003, "δ50 HO=0.33 HN=0.5  HP=0.003"),
        (50,  0.33, 0.3,  0.003, "δ50 HO=0.33 HN=0.3  HP=0.003"),
        (70,  0.33, 0.83, 0.005, "δ70 HO=0.33 HN=0.83 HP=0.005"),
        (70,  0.33, 0.5,  0.005, "δ70 HO=0.33 HN=0.5  HP=0.005"),
        (70,  0.33, 0.83, 0.003, "δ70 HO=0.33 HN=0.83 HP=0.003"),
        (30,  0.33, 0.83, 0.002, "δ30 HO=0.33 HN=0.83 HP=0.002"),
        (30,  0.33, 0.5,  0.002, "δ30 HO=0.33 HN=0.5  HP=0.002"),
        (30,  0.1,  0.83, 0.002, "δ30 HO=0.1  HN=0.83 HP=0.002"),
        (50,  0.1,  0.83, 0.003, "δ50 HO=0.1  HN=0.83 HP=0.003"),
    ]

    for dg, ho_r, hn_r, hp_r, label in combos:
        gas, ho, hn, hp = apply_rh80(gas_dry.copy(), times_raw,
                                      hono_ratio=ho_r, hono2_ratio=hn_r,
                                      h2o2_ratio=hp_r)
        r = run(times_raw, gas, ho, hn, hp, dg*1e-3, label)
        all_results.append(r)
        pr(r)

    # ==========================================================================
    # Summary: top matches
    # ==========================================================================
    print("\n" + "="*115)
    print("RANKING: composite error = |ΔpH/pH| + |ΔNO3/NO3| + |ΔH2O2/H2O2|")
    print("="*115)

    def score(r):
        e_ph = abs(r['pH'] - EXP['pH']) / EXP['pH']
        e_no3 = abs(r['NO3-'] - EXP['NO3-']) / EXP['NO3-']
        e_h2o2 = abs(r['H2O2'] - EXP['H2O2']) / max(EXP['H2O2'], 0.01)
        return e_ph + e_no3 + e_h2o2

    ranked = sorted(all_results, key=score)
    print(HDR)
    print(SEP)
    for i, r in enumerate(ranked[:15]):
        s = score(r)
        pr(r)
    print(f"\n  Exp target:  pH={EXP['pH']}, NO3⁻={EXP['NO3-']}µM, NO2⁻={EXP['NO2-']}µM, H2O2={EXP['H2O2']}µM")
