#!/usr/bin/env python3
"""Voltage comparison figure — uniform HONO/NO2=0.1, HONO2/N2O5=0.83.

HONO/NO2=0.1 (voltage-independent) + HONO2/N2O5=0.83 + H2O2/O3=0.003,
three_film BC, δ_gas=10mm, δ_liq=100µm.

3 voltage sims executed here (cached), then bar chart in the same style
as fig_voltage_comparison.png.
"""
import sys, time as time_mod
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

# ─────────────────────────────────────────────────────────────
GAS_XLSX = _project_root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'

RH80_ALL = {
    '2.6kV': {'O3_scale': 0.493, 'NO2_O3': 0.222, 'N2O5_NO2': 0.043, 'NO3_O3': 0.0179},
    '3.2kV': {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054, 'NO3_O3': 0.00442},
    '3.6kV': {'O3_scale': 0.762, 'NO2_O3': 0.095, 'N2O5_NO2': 0.037, 'NO3_O3': 0.00337},
}
HONO_NO2_UNIFORM = 0.10       # ★ uniform
HONO2_N2O5 = 0.83
H2O2_O3 = 0.003

VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
EXP = {
    '2.6kV': {'pH': 5.09, 'NO3': 32.63, 'NO2':  0.0,  'H2O2':  4.76},
    '3.2kV': {'pH': 3.61, 'NO3': 62.74, 'NO2':  3.58, 'H2O2': 11.21},
    '3.6kV': {'pH': 3.25, 'NO3': 70.42, 'NO2': 20.74, 'H2O2': 16.25},
}

CACHE_DIR = _script_dir / 'cache_hono010'
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Gas loading / RH80 (matches gen_all_figures.py behavior)
# ─────────────────────────────────────────────────────────────

def _preprocess(vals, min_run=5):
    out = np.maximum(vals.copy(), 0.0)
    n = len(out)
    run_len, stable_start = 0, n
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
    if stable_start > 0 and out[stable_start] > 0:
        out[:stable_start] = np.linspace(0, out[stable_start], stable_start + 1)[:-1]
    return out


def load_gas(voltage):
    df = pd.read_excel(GAS_XLSX, sheet_name=voltage)
    times = df.iloc[:, 0].values.astype(float)
    gas = {}
    for sp in ['O3', 'NO2', 'NO3', 'N2O5']:
        for c in df.columns:
            if sp in str(c):
                gas[sp] = _preprocess(df[c].values.astype(float))
                break
    return times, gas


def build_inputs(voltage, condition='humid'):
    """Build gas/unmeasured inputs.
    condition='dry':   raw OAS values, unmeasured (HONO/HONO2/H2O2) = 0
    condition='humid': RH80 rescaled + HONO/NO2=0.1, HONO2/N2O5=0.83, H2O2/O3=0.003
    """
    times, gas_dry = load_gas(voltage)

    if condition == 'dry':
        gas = {
            'O3':   gas_dry['O3'].copy(),
            'NO2':  gas_dry['NO2'].copy(),
            'N2O5': gas_dry['N2O5'].copy(),
            'NO3':  gas_dry['NO3'].copy(),
        }
        T = N2O4_EQ.REF_TEMP
        Kp = np.exp(np.log(N2O4_EQ.KP_298)
                    + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
        gas['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (gas['NO2'] ** 2)
        hono = np.zeros_like(gas['NO2'])
        hno3 = np.zeros_like(gas['N2O5'])
        h2o2 = np.zeros_like(gas['O3'])
        return times, gas, hono, hno3, h2o2

    # humid fitting path
    r = RH80_ALL[voltage]
    mask = times >= (times[-1] - 100)
    ss = lambda a: max(np.mean(a[mask]), 1e-30)
    o3d, no2d, n2o5d, no3d = ss(gas_dry['O3']), ss(gas_dry['NO2']), ss(gas_dry['N2O5']), ss(gas_dry['NO3'])

    o3_80 = o3d * r['O3_scale']
    no2_80 = o3_80 * r['NO2_O3']
    n2o5_80 = no2_80 * r['N2O5_NO2']
    no3_80 = o3_80 * r['NO3_O3']

    gas = {
        'O3':   gas_dry['O3']   * (o3_80 / o3d),
        'NO2':  gas_dry['NO2']  * (no2_80 / no2d),
        'N2O5': gas_dry['N2O5'] * (n2o5_80 / n2o5d),
        'NO3':  gas_dry['NO3']  * (no3_80 / no3d),
    }
    T = N2O4_EQ.REF_TEMP
    Kp = np.exp(np.log(N2O4_EQ.KP_298)
                + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
    gas['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (gas['NO2'] ** 2)

    hono = gas['NO2'] * HONO_NO2_UNIFORM   # ★ uniform 0.1
    hno3 = gas['N2O5'] * HONO2_N2O5
    h2o2 = gas['O3'] * H2O2_O3
    return times, gas, hono, hno3, h2o2


def run_voltage(voltage, condition='humid', rerun=False):
    tag = 'Dry' if condition == 'dry' else 'HONO0.1'
    cache_fp = CACHE_DIR / f'{voltage}_{tag}.npz'
    if cache_fp.exists() and not rerun:
        d = dict(np.load(cache_fp, allow_pickle=True))
        print(f'  [{voltage} / {condition}] loaded cache  → '
              f'pH={float(d["pH"]):.3f}, NO3={float(d["avg_NO3-"])*1e6:.2f}, '
              f'NO2={float(d["avg_NO2-"])*1e6:.3f}, H2O2={float(d["avg_H2O2"])*1e6:.2f}')
        return d

    times, gas, hono, hno3, h2o2 = build_inputs(voltage, condition)
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type='three_film', alpha_b=None,
        delta_gas=0.01, delta_liq=1e-4,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=hono, hono2_gas=hno3, h2o2_gas=h2o2)
    t_end = float(times[-1])
    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0
    avg = result['spatial_avg']

    data = {
        'pH': np.float64(result['pH_avg']),
        'avg_NO3-': np.float64(avg.get('NO3-', 0.0)),
        'avg_NO2-': np.float64(avg.get('NO2-', 0.0)),
        'avg_H2O2': np.float64(avg.get('H2O2', 0.0)),
        'avg_O3':   np.float64(avg.get('O3', 0.0)),
    }
    np.savez(cache_fp, **data)
    print(f'  [{voltage} / {condition}] run  → pH={float(data["pH"]):.3f}, '
          f'NO3={float(data["avg_NO3-"])*1e6:.2f} µM, '
          f'NO2={float(data["avg_NO2-"])*1e6:.3f} µM, '
          f'H2O2={float(data["avg_H2O2"])*1e6:.2f} µM, wall={wall:.0f}s')
    return data


# ─────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────

def _extract(d, key):
    if key == 'pH':
        return float(d['pH'])
    elif key == 'NO3':
        return float(d['avg_NO3-']) * 1e6
    elif key == 'NO2':
        return float(d['avg_NO2-']) * 1e6
    elif key == 'H2O2':
        return float(d['avg_H2O2']) * 1e6
    return 0.0


def make_figure(dry_results, humid_results):
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    metrics = [
        ('pH', 'pH', ''),
        ('NO₃⁻', 'NO3', ' (µM)'),
        ('NO₂⁻', 'NO2', ' (µM)'),
        ('H₂O₂', 'H2O2', ' (µM)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    n_v = len(VOLTAGES)
    n_groups = 3  # Dry, Humid, Exp
    width = 0.8 / n_groups
    x = np.arange(n_v)

    for i, (label, key, unit) in enumerate(metrics):
        ax = axes[i]

        dry_vals   = [_extract(dry_results[v],   key) for v in VOLTAGES]
        humid_vals = [_extract(humid_results[v], key) for v in VOLTAGES]
        exp_vals   = [EXP[v][key] for v in VOLTAGES]

        # Bar order: Dry (left), Humid (center), Exp (right)
        off = width * (n_groups - 1) / 2  # total offset span
        bars_dry   = ax.bar(x - off,             dry_vals,   width,
                            color='#4878a8', edgecolor='black', lw=0.8,
                            label='Dry', alpha=0.85)
        bars_humid = ax.bar(x - off + width,      humid_vals, width,
                            color='#9467bd', edgecolor='black', lw=0.8,
                            label='Humid fitting (HONO=0.1)', alpha=0.85)
        bars_exp   = ax.bar(x - off + 2*width,    exp_vals,   width,
                            color='#2ca02c', edgecolor='black', lw=0.8,
                            label='Experiment', alpha=0.85)

        # Value labels
        for bars, vals, color in [(bars_dry,   dry_vals,   '#2b4d7a'),
                                   (bars_humid, humid_vals, '#6a3d9a'),
                                   (bars_exp,   exp_vals,   '#2ca02c')]:
            for bar, val in zip(bars, vals):
                if val > 0.01:
                    fmt = f'{val:.1f}' if val >= 1 else f'{val:.2f}'
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            fmt, ha='center', va='bottom', fontsize=8, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{v}pp' for v in VOLTAGES])
        ax.set_ylabel(f'{label}{unit}')
        ax.set_title(f'({chr(97+i)}) {label}', fontweight='bold')
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(
        'Model vs Experiment — Dry / Humid (HONO/NO₂=0.1, HONO₂/N₂O₅=0.83)  '
        '(DIW, 10 min, three_film, δg=10mm, δl=100µm, H₂O₂/O₃=0.003)',
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    out_png = _script_dir / 'fig_voltage_comparison_HONO010.png'
    out_pdf = _script_dir / 'fig_voltage_comparison_HONO010.pdf'
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f'  -> {out_png.name} saved')
    print(f'  -> {out_pdf.name} saved')


if __name__ == '__main__':
    print('=== Dry condition ===')
    dry_results = {v: run_voltage(v, condition='dry') for v in VOLTAGES}
    print('=== Humid fitting (HONO=0.1) ===')
    humid_results = {v: run_voltage(v, condition='humid') for v in VOLTAGES}
    make_figure(dry_results, humid_results)
