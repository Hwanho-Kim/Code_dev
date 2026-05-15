#!/usr/bin/env python3
"""fig1b NO2/NO3 noise 단계별 진단.

각 단계의 시계열과 noise metric 추출:
  1) c_gas raw OAS (전처리 전)
  2) c_gas smoothed (SG window=31 후)
  3) c_gas after RH80 rescaling (+ time interp)
  4) C_eq = H × c_gas
  5) C_s (snap_y[0, idx])
  6) ΔC = C_eq − C_s
  7) flux = k_mt × ΔC / L

Noise metric: first-difference std, CV (coefficient of variation),
peak-to-peak / mean ratio.
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))

from config_1d import (PHYSICAL, N2O4_EQ, HENRY_CONSTANTS,
                        GAS_DIFFUSIVITY, LIQUID_DIFFUSIVITY,
                        AQUEOUS_SPECIES)
from pde_solver import MOLAR_MASS

GAS_XLSX = _root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'

# Three_film K parameters
DELTA_GAS = 0.01
DELTA_LIQ = 1e-4
ALPHA = {'O3': 0.05, 'NO2': 0.03, 'NO3': 0.03, 'N2O5': 0.03}

# RH80 (3.2 kV)
RH80 = {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054,
        'HONO_NO2': 0.10, 'NO3_O3': 0.00442}
H2O2_RATIO = 0.003
HONO2_RATIO = 0.83

VOLTAGE = '3.2kV'
CACHE = (_root / 'Figures/DIW results' /
         f'{VOLTAGE}_Humid_fitting_three_film/cache/three_film_abspecies_dg0.0100.npz')

SPECIES = ['O3', 'NO2', 'NO3', 'N2O5']      # N2O5 for comparison


def compute_k_mt_three_film(sp):
    H = HENRY_CONSTANTS[sp]
    D_g = GAS_DIFFUSIVITY[sp]
    D_l = LIQUID_DIFFUSIVITY[sp]
    M = MOLAR_MASS[sp]
    R = 8.314
    T = 298.15
    v = math.sqrt(8 * R * T / (math.pi * M * 1e-3))
    a = ALPHA[sp]
    inv_kg = DELTA_GAS / D_g
    inv_kint = 4 / (a * v)
    inv_kL = DELTA_LIQ / D_l
    return 1.0 / (H * (inv_kg + inv_kint) + inv_kL)


def preprocess_raw(vals, sg_win=31, min_run=5):
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
    stable_region = out[stable_start:]
    if len(stable_region) >= sg_win:
        w = sg_win if sg_win % 2 == 1 else sg_win + 1
        stable_region = savgol_filter(stable_region, window_length=w, polyorder=3)
        out[stable_start:] = np.maximum(stable_region, 0.0)
    if stable_start > 0 and out[stable_start] > 0:
        out[:stable_start] = np.linspace(0, out[stable_start], stable_start + 1)[:-1]
    return out


def noise_metric(arr):
    """Return (mean, std_diff, CV_diff%, p2p%)."""
    arr = np.asarray(arr)
    if arr.size < 2 or np.mean(np.abs(arr)) < 1e-30:
        return (0, 0, 0, 0)
    diff = np.diff(arr)
    mean_abs = np.mean(np.abs(arr))
    std_diff = np.std(diff)
    cv_diff = 100 * std_diff / mean_abs if mean_abs > 0 else 0
    p2p = (np.max(arr) - np.min(arr))
    p2p_rel = 100 * p2p / mean_abs if mean_abs > 0 else 0
    return (mean_abs, std_diff, cv_diff, p2p_rel)


def main():
    # ─── 1. Load raw OAS ───
    df = pd.read_excel(GAS_XLSX, sheet_name=VOLTAGE)
    times = df.iloc[:, 0].values.astype(float)
    raw = {}
    for sp in SPECIES:
        for c in df.columns:
            if sp in str(c):
                raw[sp] = df[c].values.astype(float)
                break

    # ─── 2. Preprocessed (raw → SG smoothed) ───
    smoothed = {sp: preprocess_raw(raw[sp]) for sp in SPECIES}

    # ─── 3. After RH80 rescaling ───
    mask = times >= (times[-1] - 100)
    ss = lambda a: max(np.mean(a[mask]), 1e-30)
    o3d, no2d, n2o5d, no3d = (ss(smoothed['O3']), ss(smoothed['NO2']),
                                ss(smoothed['N2O5']), ss(smoothed['NO3']))
    o3_80 = o3d * RH80['O3_scale']
    no2_80 = o3_80 * RH80['NO2_O3']
    n2o5_80 = no2_80 * RH80['N2O5_NO2']
    no3_80 = o3_80 * RH80['NO3_O3']
    rescaled = {
        'O3':   smoothed['O3']   * (o3_80 / o3d),
        'NO2':  smoothed['NO2']  * (no2_80 / no2d),
        'N2O5': smoothed['N2O5'] * (n2o5_80 / n2o5d),
        'NO3':  smoothed['NO3']  * (no3_80 / no3d),
    }

    # convert molec/cm³ → M
    conv = 1000.0 / PHYSICAL.AVOGADRO
    cgas_M = {sp: rescaled[sp] * conv for sp in SPECIES}

    # ─── 4. Load snap_y from cache ───
    d = np.load(CACHE)
    snap_t = d['snap_t']
    snap_y = d['snap_y']      # (n_t, N_z, N_s)
    sp_idx = {s: i for i, s in enumerate(AQUEOUS_SPECIES)}

    # Interpolate cgas to snap_t to align times
    cgas_at_snap = {}
    for sp in SPECIES:
        cgas_at_snap[sp] = np.interp(snap_t, times, cgas_M[sp])

    # ─── 5. Build flux components per species ───
    L = 0.01
    rows_data = {}
    for sp in SPECIES:
        H = HENRY_CONSTANTS[sp]
        Ceq = H * cgas_at_snap[sp]
        Cs  = snap_y[:, 0, sp_idx[sp]]   # surface cell, no acid-base for these
        dC  = Ceq - Cs
        k_mt = compute_k_mt_three_film(sp)
        flux = k_mt * dC / L      # M/s
        rows_data[sp] = {
            'cgas_raw': raw[sp],
            'cgas_smoothed': smoothed[sp],
            'cgas_M': cgas_at_snap[sp],
            'Ceq': Ceq,
            'Cs': Cs,
            'dC': dC,
            'flux': flux,
            'k_mt': k_mt,
        }

    # ─── 6. Print noise metrics table ───
    print('\n' + '=' * 110)
    print(f'fig1b noise propagation diagnostic — {VOLTAGE}, three_film, SG win=31')
    print('=' * 110)
    print(f"{'Species':>7} | {'Stage':>20} | {'<|x|>':>12} | "
          f"{'std(Δx)':>12} | {'CV%(std/<|x|>)':>14} | {'p2p%':>8}")
    print('-' * 110)
    for sp in SPECIES:
        D = rows_data[sp]
        for stage_name, arr in [
            ('1. cgas raw OAS',          D['cgas_raw']),
            ('2. cgas smoothed (SG31)',  D['cgas_smoothed']),
            ('3. cgas at snap_t [M]',    D['cgas_M']),
            ('4. C_eq = H·cgas',         D['Ceq']),
            ('5. C_s (surface)',         D['Cs']),
            ('6. ΔC = C_eq − C_s',       D['dC']),
            ('7. flux/L = k·ΔC/L [M/s]', D['flux']),
        ]:
            m, s, cv, p2p = noise_metric(arr)
            print(f'{sp:>7} | {stage_name:>20} | {m:12.3e} | '
                  f'{s:12.3e} | {cv:14.2f} | {p2p:8.1f}')
        print('-' * 110)

    # ─── 7. Plot ───
    n_sp = len(SPECIES)
    fig, axes = plt.subplots(7, n_sp, figsize=(4 * n_sp, 14), sharex='col')
    stages = [
        ('1. cgas raw [molec/cm³]',   'cgas_raw'),
        ('2. cgas smoothed [molec/cm³]', 'cgas_smoothed'),
        ('3. cgas at snap (M)',        'cgas_M'),
        ('4. C_eq = H·cgas (M)',        'Ceq'),
        ('5. C_s (M)',                   'Cs'),
        ('6. ΔC = Ceq − Cs (M)',         'dC'),
        ('7. flux/L (M/s)',              'flux'),
    ]
    for col, sp in enumerate(SPECIES):
        D = rows_data[sp]
        for row, (label, key) in enumerate(stages):
            ax = axes[row, col]
            arr = D[key]
            if key in ('cgas_raw', 'cgas_smoothed'):
                ax.plot(times, arr, lw=0.8, color='#3a6ea5')
            else:
                ax.plot(snap_t, arr, lw=0.8, color='#9467bd')
            if row == 0:
                ax.set_title(sp, fontweight='bold')
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        axes[-1, col].set_xlabel('Time (s)')

    fig.suptitle(
        f'Noise propagation through fig1b pipeline ({VOLTAGE}, three_film, SG=31)',
        fontsize=12, fontweight='bold', y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_dir = Path(__file__).parent
    for ext in ('png', 'pdf'):
        p = out_dir / f'diag_fig1b_noise.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()
