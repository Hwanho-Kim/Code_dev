#!/usr/bin/env python3
"""Final processed gas input — voltage trend analysis.

Pipeline (matches gen_all_figures.py):
  1. Raw OAS (Dry) → SG smoothing (window=31)
  2. RH80 rescaling: voltage-specific O3_scale, NO2_O3, N2O5_NO2, NO3_O3
  3. Unmeasured: HONO = 0.10 × NO2, HONO2 = 0.83 × N2O5, H2O2 = 0.003 × O3
  4. N2O4 = Kp · (kT/P) · NO2² (post-rescaling, self-consistent)

각 species에 대해 t=500-600s steady-state 평균을 추출하고 voltage trend 비교.
"""
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ

GAS_XLSX = _root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'

RH80_RATIOS = {
    '2.6kV': {'O3_scale': 0.493, 'NO2_O3': 0.222, 'N2O5_NO2': 0.043, 'NO3_O3': 0.0179},
    '3.2kV': {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054, 'NO3_O3': 0.00442},
    '3.6kV': {'O3_scale': 0.762, 'NO2_O3': 0.095, 'N2O5_NO2': 0.037, 'NO3_O3': 0.00337},
}
HONO_NO2_UNIFORM = 0.10
HONO2_RATIO = 0.83
H2O2_RATIO = 0.003


def preprocess(vals, sg_win=31, min_run=5):
    out = np.maximum(vals.copy(), 0.0)
    n = len(out)
    run_len, run_start, stable_start = 0, -1, n
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
    sr = out[stable_start:]
    if len(sr) >= sg_win:
        w = sg_win if sg_win % 2 == 1 else sg_win + 1
        sr = savgol_filter(sr, window_length=w, polyorder=3)
        out[stable_start:] = np.maximum(sr, 0.0)
    if stable_start > 0 and out[stable_start] > 0:
        out[:stable_start] = np.linspace(0, out[stable_start], stable_start + 1)[:-1]
    return out


def load_and_process(voltage):
    """Returns dict {species: time series (molec/cm³)} after full pipeline."""
    df = pd.read_excel(GAS_XLSX, sheet_name=voltage)
    times = df.iloc[:, 0].values.astype(float)

    # Raw measured species
    raw = {}
    for sp in ['O3', 'NO2', 'NO3', 'N2O5']:
        for c in df.columns:
            if sp in str(c):
                raw[sp] = df[c].values.astype(float)
                break

    # SG smoothing
    sm = {sp: preprocess(raw[sp]) for sp in raw}

    # RH80 rescale
    r = RH80_RATIOS[voltage]
    mask = times >= (times[-1] - 100)
    ss = lambda a: max(np.mean(a[mask]), 1e-30)
    o3d, no2d = ss(sm['O3']), ss(sm['NO2'])
    n2o5d, no3d = ss(sm['N2O5']), ss(sm['NO3'])
    o3_80 = o3d * r['O3_scale']
    no2_80 = o3_80 * r['NO2_O3']
    n2o5_80 = no2_80 * r['N2O5_NO2']
    no3_80 = o3_80 * r['NO3_O3']

    out = {
        'O3':   sm['O3']   * (o3_80 / o3d),
        'NO2':  sm['NO2']  * (no2_80 / no2d),
        'N2O5': sm['N2O5'] * (n2o5_80 / n2o5d),
        'NO3':  sm['NO3']  * (no3_80 / no3d),
    }

    # N2O4 from rescaled NO2²
    T = N2O4_EQ.REF_TEMP
    Kp = math.exp(math.log(N2O4_EQ.KP_298)
                  + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
    out['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (out['NO2'] ** 2)

    # Unmeasured derived
    out['HONO']  = out['NO2']  * HONO_NO2_UNIFORM
    out['HONO2'] = out['N2O5'] * HONO2_RATIO
    out['H2O2']  = out['O3']   * H2O2_RATIO

    return times, out


def ss_mean(times, arr, window=100):
    """steady-state avg over last `window` seconds."""
    mask = times >= (times[-1] - window)
    return float(np.mean(arr[mask]))


def main():
    VOLTS = ['2.6kV', '3.2kV', '3.6kV']
    SPECIES = ['O3', 'NO2', 'NO3', 'N2O5', 'N2O4', 'HONO', 'HONO2', 'H2O2']

    # Collect data
    all_data = {}
    ss_vals = {sp: {} for sp in SPECIES}
    for v in VOLTS:
        times, gas = load_and_process(v)
        all_data[v] = (times, gas)
        for sp in SPECIES:
            ss_vals[sp][v] = ss_mean(times, gas[sp])

    # ─── Table 1: SS values ───
    print("=" * 110)
    print("Final processed gas input — Steady-state values (last 100s avg)")
    print("=" * 110)
    print(f"{'Species':<10} | {'2.6 kV':>14} | {'3.2 kV':>14} | {'3.6 kV':>14}  [molec/cm³]")
    print("-" * 110)
    for sp in SPECIES:
        v_str = " | ".join(f'{ss_vals[sp][v]:>14.3e}' for v in VOLTS)
        print(f"{sp:<10} | {v_str}")

    # ─── Table 2: Voltage scaling ratios ───
    print()
    print("=" * 110)
    print("Voltage scaling — ratios relative to 3.2 kV (baseline)")
    print("=" * 110)
    print(f"{'Species':<10} | {'2.6/3.2':>10} | {'3.2/3.2':>10} | {'3.6/3.2':>10} | {'3.6/2.6':>10} | trend type")
    print("-" * 110)
    for sp in SPECIES:
        base = ss_vals[sp]['3.2kV']
        if base < 1e-30:
            r_26 = r_36 = r_double = 0
        else:
            r_26 = ss_vals[sp]['2.6kV'] / base
            r_36 = ss_vals[sp]['3.6kV'] / base
            r_double = ss_vals[sp]['3.6kV'] / max(ss_vals[sp]['2.6kV'], 1e-30)

        if r_26 < 0.5 and r_36 > 1.5:
            trend = "★ strongly voltage-increasing"
        elif r_36 / r_26 > 2:
            trend = "voltage-increasing"
        elif r_36 / r_26 < 0.5:
            trend = "voltage-decreasing"
        else:
            trend = "weakly voltage-dependent"

        print(f"{sp:<10} | {r_26:>10.3f} | {1.000:>10.3f} | {r_36:>10.3f} | {r_double:>10.2f}× | {trend}")

    # ─── Table 3: Species ratios within voltage ───
    print()
    print("=" * 110)
    print("Species ratios within each voltage — composition comparison")
    print("=" * 110)
    print(f"{'Ratio':<25} | {'2.6 kV':>12} | {'3.2 kV':>12} | {'3.6 kV':>12}")
    print("-" * 110)

    pairs = [
        ('NO2/O3', 'NO2', 'O3'),
        ('N2O5/O3', 'N2O5', 'O3'),
        ('NO3/O3', 'NO3', 'O3'),
        ('N2O5/NO2', 'N2O5', 'NO2'),
        ('N2O4/NO2', 'N2O4', 'NO2'),
        ('HONO/NO2', 'HONO', 'NO2'),
        ('HONO2/N2O5', 'HONO2', 'N2O5'),
        ('H2O2/O3', 'H2O2', 'O3'),
    ]
    for label, num, den in pairs:
        rs = " | ".join(
            f'{ss_vals[num][v]/max(ss_vals[den][v],1e-30):>12.4e}'
            for v in VOLTS
        )
        print(f"{label:<25} | {rs}")

    # ─── Plot: time series for all species, 3 voltages overlaid ───
    n_sp = len(SPECIES)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    colors = {'2.6kV': '#1f77b4', '3.2kV': '#2ca02c', '3.6kV': '#d62728'}

    for i, sp in enumerate(SPECIES):
        ax = axes.flat[i]
        for v in VOLTS:
            times, gas = all_data[v]
            ax.plot(times, gas[sp], label=v, color=colors[v], lw=1.4)
        ax.set_yscale('log')
        ax.set_ylabel('molec/cm³')
        ax.set_title(sp, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        if i == 0:
            ax.legend(loc='best', fontsize=10)

    for ax in axes[1]:
        ax.set_xlabel('Time (s)')

    fig.suptitle(
        'Final processed gas inputs — voltage trend comparison '
        '(SG=31, RH80 rescale, HONO=0.10·NO2, HONO2=0.83·N2O5, H2O2=0.003·O3)',
        fontsize=13, fontweight='bold', y=0.995,
    )
    fig.tight_layout()

    out_dir = Path(__file__).parent
    for ext in ('png', 'pdf'):
        p = out_dir / f'diag_gas_input_voltage_trend.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
