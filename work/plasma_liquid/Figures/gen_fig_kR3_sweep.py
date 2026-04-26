#!/usr/bin/env python3
"""k_R3 sensitivity figure — loads from cache produced by run_kR3_sweep.py."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_script_dir = Path(__file__).parent
CACHE = _script_dir / 'cache' / 'tpa'
OUT_PNG = _script_dir / 'fig_kR3_sweep.png'
OUT_PDF = _script_dir / 'fig_kR3_sweep.pdf'

VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
EXPERIMENT = {'2.6kV': 12.66, '3.2kV': 57.72, '3.6kV': 43.26}

K_R3_CASES = [
    (0.0,    r'$k_{R3}=0$ (Tampieri practice)', '#4c72b0'),
    (1.0e9,  r'$k_{R3}=10^{9}$ (initial guess)', '#dd8452'),
    (6.3e9,  r'$k_{R3}=6.3\times10^{9}$ (Page 2010)', '#8172b2'),
]


def _cache_path(voltage: str, k_R3: float) -> Path:
    if k_R3 == 0:
        return CACHE / f"{voltage}_tpa2000uM_humidfitting.npz"
    return CACHE / f"{voltage}_tpa2000uM_humidfitting_kR3-{k_R3:.0e}.npz"


def load_hTPA(voltage: str, k_R3: float) -> float:
    d = dict(np.load(_cache_path(voltage, k_R3), allow_pickle=True))
    return float(d['hTPA_uM'])


def plot():
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    x = np.arange(len(VOLTAGES))
    n_bars = len(K_R3_CASES) + 1
    width = 0.8 / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * width

    for i, (k, lbl, color) in enumerate(K_R3_CASES):
        vals = [load_hTPA(v, k) for v in VOLTAGES]
        ax.bar(x + offsets[i], vals, width, color=color,
               edgecolor='k', linewidth=0.6, label=lbl)

    exp_vals = [EXPERIMENT[v] for v in VOLTAGES]
    ax.bar(x + offsets[-1], exp_vals, width, color='#c44e52',
           edgecolor='k', linewidth=0.6, label='Experiment (IF×2)')

    ax.set_xticks(x)
    ax.set_xticklabels(VOLTAGES, fontsize=11)
    ax.set_ylabel(r'$\langle [\mathrm{hTPA}]\rangle_z$ at 600 s  (µM)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, 68)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f'Saved: {OUT_PNG}')
    print(f'Saved: {OUT_PDF}')
    print()
    print(f"{'k_R3 [M⁻¹s⁻¹]':<18} {'V':<8} {'hTPA [µM]':>12} {'exp':>8}")
    for k, _, _ in K_R3_CASES:
        for v in VOLTAGES:
            print(f"{k:<18.3e} {v:<8} {load_hTPA(v, k):>12.2f} "
                  f"{EXPERIMENT[v]:>8.2f}")


if __name__ == '__main__':
    plot()
