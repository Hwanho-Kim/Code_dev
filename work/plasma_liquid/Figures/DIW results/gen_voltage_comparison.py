#!/usr/bin/env python3
"""
Voltage comparison: Model (Dry / Humid median) vs Experiment.

Generates grouped bar charts for pH, NO₃⁻, NO₂⁻, H₂O₂ across voltages.

Run:
    .venv/bin/python "Figures/OAS data/gen_voltage_comparison.py"
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent

# Experimental reference (DIW, OAS data)
EXP = {
    '2.6kV': {'pH': 5.09, 'NO3': 32.63, 'NO2': 0.0,   'H2O2': 4.76},
    '3.2kV': {'pH': 3.61, 'NO3': 62.74, 'NO2': 3.58,  'H2O2': 11.21},
    '3.6kV': {'pH': 3.25, 'NO3': 70.42, 'NO2': 20.74, 'H2O2': 16.25},
}

VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
CONDITIONS = [
    ('Dry', 'Henry_Dry_Dg_10mm', '#4878a8'),
    ('Humid fitting', 'Henry_Humid_fitting_Dg_10mm', '#9467bd'),
]


def load_model_results():
    results = {}
    for cond_label, cond_folder, _ in CONDITIONS:
        results[cond_label] = {}
        for v in VOLTAGES:
            cache_dir = _script_dir / f'{v}_{cond_folder}' / 'cache'
            npz_files = list(cache_dir.glob('gas_alpha*.npz'))
            if npz_files:
                d = dict(np.load(npz_files[0], allow_pickle=True))
                results[cond_label][v] = {
                    'pH': float(d['pH']),
                    'NO3': float(d['avg_NO3-']) * 1e6,
                    'NO2': float(d['avg_NO2-']) * 1e6,
                    'H2O2': float(d['avg_H2O2']) * 1e6,
                }
            else:
                results[cond_label][v] = {'pH': 0, 'NO3': 0, 'NO2': 0, 'H2O2': 0}
    return results


def main():
    results = load_model_results()

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
    n_groups = len(CONDITIONS) + 1  # +1 for experiment
    width = 0.8 / n_groups
    x = np.arange(n_v)

    for i, (label, key, unit) in enumerate(metrics):
        ax = axes[i]

        # Experiment bars
        exp_vals = [EXP[v][key] for v in VOLTAGES]
        bars_exp = ax.bar(x - width * (n_groups - 1) / 2, exp_vals, width,
                          color='#2ca02c', edgecolor='black', lw=0.8,
                          label='Experiment', alpha=0.85)

        # Model bars
        for ci, (cond_label, _, color) in enumerate(CONDITIONS):
            model_vals = [results[cond_label][v][key] for v in VOLTAGES]
            offset = width * (ci + 1 - (n_groups - 1) / 2)
            bars = ax.bar(x + offset, model_vals, width, color=color,
                          edgecolor='black', lw=0.8, label=cond_label, alpha=0.85)

            # Value labels
            for bar, val in zip(bars, model_vals):
                if val > 0.01:
                    fmt = f'{val:.1f}' if val >= 1 else f'{val:.2f}'
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            fmt, ha='center', va='bottom', fontsize=8)

        # Experiment value labels
        for bar, val in zip(bars_exp, exp_vals):
            if val > 0.01:
                fmt = f'{val:.1f}' if val >= 1 else f'{val:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        fmt, ha='center', va='bottom', fontsize=8, color='#2ca02c')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{v}pp' for v in VOLTAGES])
        ax.set_ylabel(f'{label}{unit}')
        ax.set_title(f'({chr(97+i)}) {label}', fontweight='bold')
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Model vs Experiment — Voltage dependence (DIW, 10 min, δg=10mm, Henry fix)',
                 fontsize=14, y=1.01)
    fig.tight_layout()

    out = _script_dir / 'fig_voltage_comparison.png'
    fig.savefig(out)
    fig.savefig(_script_dir / 'fig_voltage_comparison.pdf')
    print(f'  -> {out.name} saved')
    plt.close(fig)


if __name__ == '__main__':
    main()
