#!/usr/bin/env python3
"""
2D sweep: HONO/NO₂ ratio × O₃ MT scaling.

Goal: find conditions where NO₂⁻ approaches experimental 3 µM.

Fixed: HNO₃/N₂O₅=0.83, H₂O₂/O₃=0.03
Sweep: HONO/NO₂ = [1, 3, 5, 10]  ×  O₃ MT scale = [0.1, 0.2, 0.3, 0.5, 1.0]
Total: 20 cases

Run:
    .venv/bin/python Figures/test/sweep/test_hono_o3_sweep.py
"""

import sys
import time as time_mod
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

DZ_MIN = 5e-6
STRETCH = 1.12
DELTA_GAS = 0.01

# Fixed ratios
HNO3_RATIO = 0.83
H2O2_RATIO = 0.03

# Sweep axes
HONO_RATIOS = [1.0, 3.0, 5.0, 10.0]
O3_SCALES = [0.1, 0.2, 0.3, 0.5, 1.0]

EXP = {'pH': 3.61, 'NO3-': 63.0, 'NO2-': 3.0, 'H2O2': 11.0}


def load_gas_data():
    df = pd.read_csv(DEFAULT_CSV)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0) if col in df.columns else np.zeros(len(df))
    if np.all(gas_conc['N2O4'] == 0):
        no2 = gas_conc['NO2']; T = 298.15
        Kp = math.exp(math.log(N2O4_EQ.KP_298) +
                       (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)
    return times, gas_conc


def run_case(times, gas_conc, hono_ratio, o3_scale):
    hono_gas = gas_conc['NO2'] * hono_ratio
    hno3_gas = gas_conc['N2O5'] * HNO3_RATIO
    h2o2_gas = gas_conc['O3'] * H2O2_RATIO  # based on original O3, not scaled

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type='gas_alpha', alpha_b=None,
        delta_gas=DELTA_GAS,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hno3_gas,
                        h2o2_gas=h2o2_gas)

    # Scale O3 k_mt
    if o3_scale != 1.0:
        solver._interface_species = [
            (a, k * o3_scale if g == 'O3' else k, g, H, Ka)
            for a, k, g, H, Ka in solver._interface_species
        ]

    t_end = float(times[-1])
    result = solver.solve(t_span=(0, t_end), t_eval=np.array([0, t_end]),
                          verbose=False, dt_poisson=None)
    avg = result['spatial_avg']
    return {
        'pH': result['pH_avg'],
        'NO3': avg.get('NO3-', 0) * 1e6,
        'NO2': avg.get('NO2-', 0) * 1e6,
        'H2O2': avg.get('H2O2', 0) * 1e6,
        'O3': avg.get('O3', 0) * 1e9,
        'OH': avg.get('OH', 0) * 1e12,
    }


def main():
    times, gas_conc = load_gas_data()

    print("=" * 70)
    print("HONO/NO₂ × O₃ MT scaling sweep")
    print(f"Fixed: HNO₃/N₂O₅={HNO3_RATIO}, H₂O₂/O₃={H2O2_RATIO}")
    print("=" * 70)

    # Run all cases
    results = {}
    n_total = len(HONO_RATIOS) * len(O3_SCALES)
    count = 0
    for hr in HONO_RATIOS:
        for o3s in O3_SCALES:
            count += 1
            label = f"HONO={hr}, O3×{o3s}"
            print(f"  [{count}/{n_total}] {label}...", end=' ', flush=True)
            t0 = time_mod.time()
            r = run_case(times, gas_conc, hr, o3s)
            w = time_mod.time() - t0
            results[(hr, o3s)] = r
            print(f"pH={r['pH']:.3f} NO3={r['NO3']:.1f} NO2-={r['NO2']:.4f} "
                  f"H2O2={r['H2O2']:.2f} O3={r['O3']:.1f}nM {w:.0f}s")

    # ---- Summary table ----
    print("\n" + "=" * 90)
    print(f"{'HONO/NO2':>10s} {'O3 scale':>9s} {'pH':>6s} {'NO3(µM)':>8s} "
          f"{'NO2-(µM)':>10s} {'H2O2(µM)':>9s} {'O3(nM)':>8s}")
    print("-" * 65)
    for hr in HONO_RATIOS:
        for o3s in O3_SCALES:
            r = results[(hr, o3s)]
            flag = ' ★' if abs(r['NO2'] - 3.0) < 1.5 else ''
            print(f"{hr:10.1f} {o3s:9.1f} {r['pH']:6.3f} {r['NO3']:8.1f} "
                  f"{r['NO2']:10.4f} {r['H2O2']:9.2f} {r['O3']:8.1f}{flag}")
        print()
    print(f"{'Exp':>10s} {'':>9s} {3.61:6.2f} {63.0:8.1f} {3.0:10.4f} {11.0:9.2f}")

    # ---- Heatmap plots ----
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 11, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    metrics = [
        ('pH', 'pH', EXP['pH']),
        ('NO₃⁻ (µM)', 'NO3', EXP['NO3-']),
        ('NO₂⁻ (µM)', 'NO2', EXP['NO2-']),
        ('H₂O₂ (µM)', 'H2O2', EXP['H2O2']),
        ('O₃ (nM)', 'O3', None),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for i, (title, key, exp_val) in enumerate(metrics):
        ax = axes[i]
        data = np.zeros((len(HONO_RATIOS), len(O3_SCALES)))
        for ri, hr in enumerate(HONO_RATIOS):
            for ci, o3s in enumerate(O3_SCALES):
                data[ri, ci] = results[(hr, o3s)][key]

        im = ax.imshow(data, aspect='auto', origin='lower',
                       cmap='viridis')
        ax.set_xticks(range(len(O3_SCALES)))
        ax.set_xticklabels([f'{s}' for s in O3_SCALES])
        ax.set_yticks(range(len(HONO_RATIOS)))
        ax.set_yticklabels([f'{r}' for r in HONO_RATIOS])
        ax.set_xlabel('O₃ MT scale')
        ax.set_ylabel('HONO/NO₂')
        ax.set_title(title)

        # Annotate cells
        for ri in range(len(HONO_RATIOS)):
            for ci in range(len(O3_SCALES)):
                v = data[ri, ci]
                fmt = f'{v:.2f}' if v < 10 else f'{v:.1f}'
                color = 'white' if v < (data.max() + data.min()) / 2 else 'black'
                ax.text(ci, ri, fmt, ha='center', va='center',
                        fontsize=7, color=color)

        # Mark experimental value in colorbar
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        if exp_val is not None:
            cb.ax.axhline(y=exp_val, color='red', lw=2, ls='--')

    fig.suptitle('HONO/NO₂ × O₃ MT scaling sweep\n'
                 f'(gas_alpha, δg=10mm, HNO₃/N₂O₅={HNO3_RATIO}, H₂O₂/O₃={H2O2_RATIO})',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig_hono_o3_sweep.png')
    fig.savefig(_script_dir / 'fig_hono_o3_sweep.pdf')
    print(f"\n  -> fig_hono_o3_sweep.png/pdf saved")
    plt.close(fig)


if __name__ == '__main__':
    main()
