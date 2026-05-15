#!/usr/bin/env python3
"""Gas input — Raw OAS vs Processed (Humid fitting) side-by-side comparison.

Generates:
  1) Time series overlay (raw vs processed) for each measured species
  2) Composition ratio comparison
  3) Steady-state transform factor table
  4) Voltage trend comparison (raw vs processed scaling)
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))
from config_1d import PHYSICAL, N2O4_EQ

GAS_XLSX = _root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'

RH80 = {
    '2.6kV': {'O3_scale': 0.493, 'NO2_O3': 0.222, 'N2O5_NO2': 0.043, 'NO3_O3': 0.0179},
    '3.2kV': {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054, 'NO3_O3': 0.00442},
    '3.6kV': {'O3_scale': 0.762, 'NO2_O3': 0.095, 'N2O5_NO2': 0.037, 'NO3_O3': 0.00337},
}
HONO_NO2 = 0.10
HONO2_R = 0.83
H2O2_R = 0.003
VOLTS = ['2.6kV', '3.2kV', '3.6kV']
COLORS = {'2.6kV': '#1f77b4', '3.2kV': '#2ca02c', '3.6kV': '#d62728'}


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


def load_voltage(voltage):
    """Returns dict: 'raw' (no smoothing/rescale), 'processed' (full pipeline)."""
    df = pd.read_excel(GAS_XLSX, sheet_name=voltage)
    times = df.iloc[:, 0].values.astype(float)

    raw = {}
    for sp in ['O3', 'NO2', 'NO3', 'N2O5']:
        for c in df.columns:
            if sp in str(c):
                v = df[c].values.astype(float)
                v[v < 0] = 0
                raw[sp] = v
                break

    sm = {sp: preprocess(raw[sp]) for sp in raw}

    r = RH80[voltage]
    mask = times >= (times[-1] - 100)
    ss = lambda a: max(np.mean(a[mask]), 1e-30)
    o3d, no2d = ss(sm['O3']), ss(sm['NO2'])
    n2o5d, no3d = ss(sm['N2O5']), ss(sm['NO3'])
    o3_80 = o3d * r['O3_scale']
    no2_80 = o3_80 * r['NO2_O3']
    n2o5_80 = no2_80 * r['N2O5_NO2']
    no3_80 = o3_80 * r['NO3_O3']

    proc = {
        'O3':   sm['O3']   * (o3_80 / o3d),
        'NO2':  sm['NO2']  * (no2_80 / no2d),
        'N2O5': sm['N2O5'] * (n2o5_80 / n2o5d),
        'NO3':  sm['NO3']  * (no3_80 / no3d),
    }
    T = N2O4_EQ.REF_TEMP
    Kp = math.exp(math.log(N2O4_EQ.KP_298)
                  + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
    proc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (proc['NO2'] ** 2)
    proc['HONO'] = proc['NO2'] * HONO_NO2
    proc['HONO2'] = proc['N2O5'] * HONO2_R
    proc['H2O2'] = proc['O3'] * H2O2_R
    return times, raw, proc


def ss_mean(times, arr, window=100):
    mask = times >= (times[-1] - window)
    return float(np.mean(arr[mask]))


def main():
    data = {v: load_voltage(v) for v in VOLTS}
    species_meas = ['O3', 'NO2', 'NO3', 'N2O5']
    species_derived = ['N2O4', 'HONO', 'HONO2', 'H2O2']

    # ─── Print SS transform table ───
    print("=" * 110)
    print("Raw vs Processed — Steady-state values (last 100s avg, molec/cm³)")
    print("=" * 110)
    print(f"{'Sp':<5} | {'V':<6} | {'RAW':>13} | {'PROC':>13} | {'PROC/RAW':>10} | comment")
    print('-' * 110)
    for sp in species_meas:
        for v in VOLTS:
            times, raw, proc = data[v]
            r_val = ss_mean(times, raw[sp])
            p_val = ss_mean(times, proc[sp])
            ratio = p_val / max(r_val, 1e-30)
            print(f"{sp:<5} | {v:<6} | {r_val:>13.3e} | {p_val:>13.3e} | {ratio:>10.3f} |")
        print()

    # ─── Voltage scaling comparison ───
    print("=" * 110)
    print("Voltage scaling 3.6/2.6 — RAW vs PROCESSED")
    print("=" * 110)
    print(f"{'Sp':<8} | {'RAW 3.6/2.6':>13} | {'PROC 3.6/2.6':>13} | difference")
    print('-' * 80)
    for sp in species_meas:
        r_26 = ss_mean(*data['2.6kV'][:1], data['2.6kV'][1][sp])
        r_36 = ss_mean(*data['3.6kV'][:1], data['3.6kV'][1][sp])
        p_26 = ss_mean(*data['2.6kV'][:1], data['2.6kV'][2][sp])
        p_36 = ss_mean(*data['3.6kV'][:1], data['3.6kV'][2][sp])
        raw_scale = r_36 / max(r_26, 1e-30)
        proc_scale = p_36 / max(p_26, 1e-30)
        amp = proc_scale / max(raw_scale, 1e-30)
        comment = "amplified" if amp > 1.1 else "dampened" if amp < 0.9 else "preserved"
        print(f"{sp:<8} | {raw_scale:>13.2f} | {proc_scale:>13.2f} | proc {amp:.2f}× ({comment})")

    # ─── Plot: 2x4 grid showing raw + processed time series for 4 measured species ───
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)

    for col, sp in enumerate(species_meas):
        ax_raw = axes[0, col]
        ax_proc = axes[1, col]
        for v in VOLTS:
            times, raw, proc = data[v]
            ax_raw.plot(times, raw[sp], color=COLORS[v], label=v, lw=1.2)
            ax_proc.plot(times, proc[sp], color=COLORS[v], label=v, lw=1.2)
        ax_raw.set_yscale('log')
        ax_proc.set_yscale('log')
        ax_raw.set_title(f'{sp} — RAW', fontweight='bold')
        ax_proc.set_title(f'{sp} — PROCESSED', fontweight='bold')
        ax_raw.set_ylabel('molec/cm³')
        ax_raw.grid(True, alpha=0.3, which='both')
        ax_proc.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax_raw.legend(loc='best', fontsize=9)
            ax_proc.legend(loc='best', fontsize=9)

    for ax in axes[1]:
        ax.set_xlabel('Time (s)')

    fig.suptitle(
        'Gas input: Raw OAS (Dry) vs Processed (SG smoothed + RH80 rescaled + unmeasured derived)',
        fontsize=13, fontweight='bold', y=0.995,
    )
    fig.tight_layout()
    out_dir = Path(__file__).parent
    for ext in ('png', 'pdf'):
        p = out_dir / f'diag_gas_raw_vs_processed.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'\nSaved: {p}')

    # ─── Plot 2: ratio + bar chart summary ───
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: transform factor (proc/raw) at SS
    ax = axes2[0]
    x = np.arange(len(species_meas))
    width = 0.25
    for i, v in enumerate(VOLTS):
        ratios = []
        for sp in species_meas:
            times, raw, proc = data[v]
            r_val = ss_mean(times, raw[sp])
            p_val = ss_mean(times, proc[sp])
            ratios.append(p_val / max(r_val, 1e-30))
        ax.bar(x + i*width - width, ratios, width, label=v, color=COLORS[v],
               edgecolor='black', lw=0.8, alpha=0.85)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(species_meas)
    ax.set_ylabel('Transform factor (PROC/RAW)')
    ax.set_title('(a) Steady-state transform per voltage', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, which='both')

    # Right: voltage scaling 3.6/2.6 — RAW vs PROC
    ax = axes2[1]
    x = np.arange(len(species_meas))
    width = 0.35
    raw_scales, proc_scales = [], []
    for sp in species_meas:
        r_26 = ss_mean(*data['2.6kV'][:1], data['2.6kV'][1][sp])
        r_36 = ss_mean(*data['3.6kV'][:1], data['3.6kV'][1][sp])
        p_26 = ss_mean(*data['2.6kV'][:1], data['2.6kV'][2][sp])
        p_36 = ss_mean(*data['3.6kV'][:1], data['3.6kV'][2][sp])
        raw_scales.append(r_36 / max(r_26, 1e-30))
        proc_scales.append(p_36 / max(p_26, 1e-30))
    ax.bar(x - width/2, raw_scales, width, label='RAW', color='#7f7f7f',
           edgecolor='black', lw=0.8, alpha=0.85)
    ax.bar(x + width/2, proc_scales, width, label='PROCESSED', color='#9467bd',
           edgecolor='black', lw=0.8, alpha=0.85)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(species_meas)
    ax.set_ylabel('Scaling factor 3.6 kV / 2.6 kV')
    ax.set_title('(b) Voltage scaling — RAW vs PROCESSED', fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    fig2.suptitle('Processing transform summary', fontsize=13, fontweight='bold', y=1.01)
    fig2.tight_layout()
    for ext in ('png', 'pdf'):
        p = out_dir / f'diag_gas_transform_summary.{ext}'
        fig2.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()
