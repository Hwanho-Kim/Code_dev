#!/usr/bin/env python3
"""
NO₂⁻ sensitivity test: R92 rate constant scaling.

R92: NO₃ + NO₂⁻ → NO₃⁻ + NO₂, k=1.2e9 M⁻¹s⁻¹
Also test R32: O₃ + NO₂⁻ → NO₃⁻, k=5e5

Cases:
  1. Baseline (all rates original)
  2. R92 k × 0.1
  3. R92 k × 0.01
  4. R92 OFF (k=0)
  5. R32 OFF (k=0)
  6. R92 + R32 both OFF

BC: gas_alpha, species-specific α_b, δ_gas=10mm
Gas: HONO/NO₂=0.33, HNO₃/N₂O₅=0.83, H₂O₂/O₃=0.03

Run:
    .venv/bin/python Figures/test/sweep/test_no2m_sensitivity.py
"""

import sys
import time as time_mod
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
HONO_RATIO = 0.33
HONO2_RATIO = 0.83
H2O2_RATIO = 0.03

EXP = {'pH': 3.61, 'NO3-': 63.0, 'NO2-': 3.0, 'H2O2': 11.0}

# (label, R92_scale, R32_scale)
CASES = [
    ('Baseline',        1.0,  1.0),
    ('R92 × 0.1',       0.1,  1.0),
    ('R92 × 0.01',      0.01, 1.0),
    ('R92 OFF',          0.0,  1.0),
    ('R32 OFF',          1.0,  0.0),
    ('R92+R32 OFF',      0.0,  0.0),
]


def load_gas_data():
    df = pd.read_csv(DEFAULT_CSV)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))
    if np.all(gas_conc.get('N2O4', np.array([0])) == 0):
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(math.log(N2O4_EQ.KP_298) +
                       (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)
    return times, gas_conc


def run_case(times, gas_conc, label, r92_scale, r32_scale):
    chem = AqueousChemistry1D(saline_mode=False)

    # Modify rate constants for R92 and R32
    for rxn_d in chem._rxn_data:
        lbl = rxn_d.get('label', '')
        if 'R92' in lbl and r92_scale != 1.0:
            rxn_d['k'] = rxn_d['k'] * r92_scale
            print(f"    {lbl}: k scaled to {rxn_d['k']:.2e}")
        if 'R32' in lbl and r32_scale != 1.0:
            rxn_d['k'] = rxn_d['k'] * r32_scale
            print(f"    {lbl}: k scaled to {rxn_d['k']:.2e}")

    # Update Numba rate constant arrays (used by compute_rates_batch)
    for i, rxn in enumerate(chem.reactions):
        lbl = rxn.get('label', '')
        if 'R92' in lbl and r92_scale != 1.0:
            chem._nb_rxn_k[i] *= r92_scale
        if 'R32' in lbl and r32_scale != 1.0:
            chem._nb_rxn_k[i] *= r32_scale

    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type='gas_alpha', alpha_b=None,
        delta_gas=DELTA_GAS,
    )

    hono_gas = gas_conc['NO2'] * HONO_RATIO
    hno3_gas = gas_conc['N2O5'] * HONO2_RATIO
    h2o2_gas = gas_conc['O3'] * H2O2_RATIO
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hno3_gas,
                        h2o2_gas=h2o2_gas)

    t_end = float(times[-1])
    t_eval = np.arange(2.0, t_end + 0.1, 2.0)
    t_eval = t_eval[t_eval <= t_end + 0.1]

    print(f"  [{label}] running...")
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    print(f"    pH={result['pH_avg']:.3f}, NO3-={avg.get('NO3-',0)*1e6:.1f}, "
          f"NO2-={avg.get('NO2-',0)*1e6:.4f}, H2O2={avg.get('H2O2',0)*1e6:.2f}, "
          f"O3={avg.get('O3',0)*1e9:.1f}nM, OH={avg.get('OH',0)*1e12:.2f}pM, "
          f"wall={wall:.0f}s")

    return result, solver


def main():
    times, gas_conc = load_gas_data()

    print("=" * 60)
    print("NO₂⁻ sensitivity: R92/R32 rate constant scaling")
    print("=" * 60)

    all_results = []
    for label, r92, r32 in CASES:
        result, solver = run_case(times, gas_conc, label, r92, r32)
        all_results.append((label, result, solver))

    # ---- RONS bar chart ----
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 11, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    labels = [c[0] for c in all_results]
    n = len(labels)
    metrics = [
        ('pH', [r['pH_avg'] for _, r, _ in all_results], EXP['pH']),
        ('NO₃⁻ (µM)', [r['spatial_avg'].get('NO3-',0)*1e6 for _, r, _ in all_results], EXP['NO3-']),
        ('NO₂⁻ (µM)', [r['spatial_avg'].get('NO2-',0)*1e6 for _, r, _ in all_results], EXP['NO2-']),
        ('H₂O₂ (µM)', [r['spatial_avg'].get('H2O2',0)*1e6 for _, r, _ in all_results], EXP['H2O2']),
        ('O₃ (nM)', [r['spatial_avg'].get('O3',0)*1e9 for _, r, _ in all_results], None),
        ('OH (pM)', [r['spatial_avg'].get('OH',0)*1e12 for _, r, _ in all_results], None),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.5 * len(metrics)),
                              sharex=True)
    x = np.arange(n)
    colors = ['#333333', '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']

    for i, (metric, vals, exp_v) in enumerate(metrics):
        ax = axes[i]
        ax.bar(x, vals, color=colors[:n], width=0.7)
        if exp_v is not None:
            ax.axhline(y=exp_v, color='red', ls='--', lw=1.5, label=f'Exp={exp_v}')
            ax.legend(fontsize=8)
        ax.set_ylabel(metric)
        for j, v in enumerate(vals):
            if v > 0.001:
                ax.text(j, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    fig.suptitle('NO₂⁻ sensitivity: R92/R32 rate constant scaling\n'
                 '(gas_alpha, δg=10mm, HONO/HNO₃/H₂O₂ ratio-based)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig_no2m_sensitivity.png')
    fig.savefig(_script_dir / 'fig_no2m_sensitivity.pdf')
    print(f"\n  -> fig_no2m_sensitivity.png/pdf saved")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Case':20s} {'pH':>6s} {'NO3(µM)':>8s} {'NO2(µM)':>9s} "
          f"{'H2O2(µM)':>9s} {'O3(nM)':>8s} {'OH(pM)':>8s}")
    print("-" * 75)
    for label, result, _ in all_results:
        avg = result['spatial_avg']
        print(f"{label:20s} {result['pH_avg']:6.3f} "
              f"{avg.get('NO3-',0)*1e6:8.1f} {avg.get('NO2-',0)*1e6:9.4f} "
              f"{avg.get('H2O2',0)*1e6:9.2f} {avg.get('O3',0)*1e9:8.1f} "
              f"{avg.get('OH',0)*1e12:8.2f}")
    print(f"{'Experiment':20s} {3.61:6.2f} {63.0:8.1f} {3.0:9.4f} "
          f"{11.0:9.2f} {'–':>8s} {'–':>8s}")


if __name__ == '__main__':
    main()
