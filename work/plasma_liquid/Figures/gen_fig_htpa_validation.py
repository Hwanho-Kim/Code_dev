#!/usr/bin/env python3
"""Simulation hTPA vs experimental hTPA across 3 voltages."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_script_dir = Path(__file__).parent
CACHE = _script_dir / 'cache' / 'tpa'
OUT_PNG = _script_dir / 'fig_htpa_validation.png'
OUT_PDF = _script_dir / 'fig_htpa_validation.pdf'

VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
EXPERIMENT = {'2.6kV': 12.66, '3.2kV': 57.72, '3.6kV': 43.26}


def compute_sim(voltage: str) -> float:
    d = dict(np.load(CACHE / f"{voltage}_tpa2000uM_humidfitting.npz",
                     allow_pickle=True))
    keys = d['species_idx_keys']; vals = d['species_idx_vals']
    idx = {str(k): int(v_) for k, v_ in zip(keys, vals)}
    snap = d['snap_y']
    dz = d['dz_cells']; L = dz.sum()
    htpa_2d = snap[:, :, idx['hTPA']]
    return ((htpa_2d * dz[None, :]).sum(axis=1) / L)[-1] * 1e6  # µM


def plot():
    sims = [compute_sim(v) for v in VOLTAGES]
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
