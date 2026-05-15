#!/usr/bin/env python3
"""Voltage comparison — Dry (raw OAS + ratios, no RH80) vs Humid_fitting vs Exp.

All sims share: HONO/NO2=0.10, HONO2/N2O5=0.83, H2O2/O3=0.003, three_film BC,
δ_gas=10mm, δ_liq=100µm.

Difference: 'Dry' uses raw OAS gas concentrations (no RH80 rescaling),
'Humid_fitting' applies RH80-extrapolated gas SS values.

Cache loaded from {V}_Dry_three_film/cache and {V}_Humid_fitting_three_film/cache
(produced by gen_all_figures.py).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_script_dir = Path(__file__).parent

VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
EXP = {
    '2.6kV': {'pH': 5.09, 'NO3': 32.63, 'NO2':  0.0,  'H2O2':  4.76},
    '3.2kV': {'pH': 3.61, 'NO3': 62.74, 'NO2':  3.58, 'H2O2': 11.21},
    '3.6kV': {'pH': 3.25, 'NO3': 70.42, 'NO2': 20.74, 'H2O2': 16.25},
}

CACHE_NAME = 'three_film_abspecies_dg0.0100.npz'


def _final(arr):
    a = np.asarray(arr)
    return float(a[-1]) if a.ndim > 0 else float(a)


def load_cache(voltage, condition):
    fp = _script_dir / f'{voltage}_{condition}_three_film' / 'cache' / CACHE_NAME
    d = dict(np.load(fp, allow_pickle=True))
    return {
        'pH':   _final(d['pH']),
        'NO3':  _final(d['avg_NO3-']) * 1e6,
        'NO2':  _final(d['avg_NO2-']) * 1e6,
        'H2O2': _final(d['avg_H2O2']) * 1e6,
    }


def main():
    dry   = {v: load_cache(v, 'Dry')           for v in VOLTAGES}
    humid = {v: load_cache(v, 'Humid_fitting') for v in VOLTAGES}

    print(f"{'V':<6} | {'metric':<6} | {'Dry':>10} | {'Humid':>10} | {'Exp':>10}")
    print('-' * 60)
    for v in VOLTAGES:
        for k in ['pH', 'NO3', 'NO2', 'H2O2']:
            print(f'{v:<6} | {k:<6} | {dry[v][k]:>10.3f} | '
                  f'{humid[v][k]:>10.3f} | {EXP[v][k]:>10.3f}')
        print()

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    metrics = [
        ('pH',    'pH',   ''),
        ('NO₃⁻',  'NO3',  ' (μM)'),
        ('NO₂⁻',  'NO2',  ' (μM)'),
        ('H₂O₂',  'H2O2', ' (μM)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    n_v = len(VOLTAGES)
    n_groups = 3
    width = 0.8 / n_groups
    x = np.arange(n_v)
    off = width * (n_groups - 1) / 2

    for i, (label, key, unit) in enumerate(metrics):
        ax = axes[i]
        dry_vals   = [dry[v][key]   for v in VOLTAGES]
        humid_vals = [humid[v][key] for v in VOLTAGES]
        exp_vals   = [EXP[v][key]   for v in VOLTAGES]

        b1 = ax.bar(x - off,           dry_vals,   width,
                    color='#4878a8', edgecolor='black', lw=0.8,
                    label='Sim (Dry, raw OAS + ratios)', alpha=0.85)
        b2 = ax.bar(x - off + width,   humid_vals, width,
                    color='#9467bd', edgecolor='black', lw=0.8,
                    label='Sim (Humid fitting, RH80 + ratios)', alpha=0.85)
        b3 = ax.bar(x - off + 2*width, exp_vals,   width,
                    color='#2ca02c', edgecolor='black', lw=0.8,
                    label='Experiment', alpha=0.85)

        for bars, vals, color in [(b1, dry_vals, '#2b4d7a'),
                                   (b2, humid_vals, '#6a3d9a'),
                                   (b3, exp_vals, '#1f7a1f')]:
            for bar, val in zip(bars, vals):
                if val > 0.01:
                    fmt = f'{val:.1f}' if val >= 1 else f'{val:.2f}'
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            fmt, ha='center', va='bottom', fontsize=8, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{v}pp' for v in VOLTAGES])
        ax.set_ylabel(f'{label}{unit}')
        ax.set_title(f'({chr(97+i)}) {label}', fontweight='bold')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(
        'Dry (raw OAS + HONO/NO₂=0.1, HONO₂/N₂O₅=0.83, H₂O₂/O₃=0.003) '
        'vs Humid fitting vs Experiment\n'
        '(DIW, 10 min, three_film, δg=10mm, δl=100µm)',
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    out_png = _script_dir / 'fig_voltage_comparison_Dry.png'
    out_pdf = _script_dir / 'fig_voltage_comparison_Dry.pdf'
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f'\n  -> {out_png.name} saved')
    print(f'  -> {out_pdf.name} saved')


if __name__ == '__main__':
    main()
