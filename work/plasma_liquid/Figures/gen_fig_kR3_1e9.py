#!/usr/bin/env python3
"""Simulation (k_R3 = 1e9) vs experimental hTPA across 3 voltages."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_script_dir = Path(__file__).parent
CACHE = _script_dir / 'cache' / 'tpa'
OUT_PNG = _script_dir / 'fig_kR3_1e9.png'
OUT_PDF = _script_dir / 'fig_kR3_1e9.pdf'

VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
EXPERIMENT = {'2.6kV': 12.66, '3.2kV': 57.72, '3.6kV': 43.26}
K_R3 = 1.0e9


def load_sim(voltage: str) -> float:
    path = CACHE / f"{voltage}_tpa2000uM_humidfitting_kR3-{K_R3:.0e}.npz"
    d = dict(np.load(path, allow_pickle=True))
    return float(d['hTPA_uM'])


def plot():
    sims = [load_sim(v) for v in VOLTAGES]
    exp_vals = [EXPERIMENT[v] for v in VOLTAGES]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.arange(len(VOLTAGES))
    width = 0.35

    ax.bar(x - width/2, sims, width,
           color='#e07856', edgecolor='k', linewidth=0.6,
           label='Simulation')
    ax.bar(x + width/2, exp_vals, width,
           color='#2a6a8b', edgecolor='k', linewidth=0.6,
           label='Experiment')

    ax.set_xticks(x)
    ax.set_xticklabels(VOLTAGES, fontsize=11)
    ax.set_ylabel('[hTPA] (µM)', fontsize=11)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=10, loc='upper left')

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f'Saved: {OUT_PNG}')
    print(f'Saved: {OUT_PDF}')
    for v, s, e in zip(VOLTAGES, sims, exp_vals):
        print(f'  {v}: sim={s:.2f}  exp={e:.2f}')


if __name__ == '__main__':
    plot()
