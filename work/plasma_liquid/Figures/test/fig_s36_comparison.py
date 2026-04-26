#!/usr/bin/env python3
"""4-panel Saline sim vs exp (pH, H2O2, NO2-, NO3-) across 3 voltages,
using the three_film + H2O2/O3=0.003 + S36-disabled setup.

Data source: /tmp/smoke_s36off.log produced by smoke_saline_three_film.py
after S36 (Cl2- + H2O2 -> OH + OH- + Cl2) was commented out in
reactions_saline.yaml on 2026-04-23.

Output: fig_s36off_vs_exp.{png,pdf} in this directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_here = Path(__file__).parent

VOLTAGES = ['2.6 kV', '3.2 kV', '3.6 kV']
V_POS = np.arange(3)

# ===== Results (from /tmp/smoke_s36off.log SUMMARY, 2026-04-23) =====
# three_film default + H2O2/O3=0.003 + N2O4 fix + S36 disabled
SIM = {
    'DIW': {
        'pH':    [4.417, 4.228, 4.220],
        'H2O2':  [5.703, 16.646, 21.281],
        'NO2-':  [0.047, 0.054, 0.052],
        'NO3-':  [38.232, 59.084, 60.147],
    },
    'Saline': {
        'pH':    [4.446, 4.269, 4.262],
        'H2O2':  [5.349, 15.525, 19.559],
        'NO2-':  [0.011, 0.003, 0.003],
        'NO3-':  [38.278, 59.180, 60.242],
    },
}

EXP = {
    'DIW': {
        'pH':    [5.09, 3.61, 3.25],
        'H2O2':  [4.76, 11.21, 16.25],
        'NO2-':  [0.0, 3.58, 20.74],
        'NO3-':  [32.63, 62.74, 70.42],
    },
    'Saline': {
        'pH':    [5.15, 3.60, 3.43],
        'H2O2':  [2.00, 5.14, 7.73],
        'NO2-':  [0.0, 0.0, 0.0],
        'NO3-':  [32.44, 101.30, 112.77],
    },
}

METRICS = [
    ('pH',   'pH',                         ''),
    ('H2O2', 'H₂O₂ (µM)',  'µM'),
    ('NO2-', 'NO₂⁻ (µM)',  'µM'),
    ('NO3-', 'NO₃⁻ (µM)',  'µM'),
]

C_SAL_SIM    = '#d62728'   # red
C_SAL_EXP    = '#f5b0b0'   # light red


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()

    bar_w = 0.35
    offsets = np.array([-0.5, 0.5]) * bar_w

    for ax, (key, ylabel, _unit) in zip(axes, METRICS):
        s_sim = SIM['Saline'][key]
        s_exp = EXP['Saline'][key]

        ax.bar(V_POS + offsets[0], s_sim, bar_w,
               color=C_SAL_SIM, edgecolor='black', lw=0.6,
               label='Saline sim')
        ax.bar(V_POS + offsets[1], s_exp, bar_w,
               color=C_SAL_EXP, edgecolor='black', lw=0.6,
               label='Saline exp', hatch='//')

        ax.set_xticks(V_POS)
        ax.set_xticklabels(VOLTAGES)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

        if key == 'pH':
            ax.set_ylim(0, 6.5)
        elif key == 'NO2-':
            ax.set_ylim(0, max(max(s_exp) if max(s_exp) > 0 else 0.05,
                               max(s_sim), 0.05) * 1.2)

    # Title labels
    axes[0].set_title('(a) pH')
    axes[1].set_title('(b) H₂O₂')
    axes[2].set_title('(c) NO₂⁻')
    axes[3].set_title('(d) NO₃⁻')

    fig.suptitle(
        'Saline Sim vs Exp — three_film, H₂O₂/O₃=0.003, S36 disabled '
        '(2026-04-23)',
        fontsize=13, y=1.00,
    )
    fig.tight_layout()

    out_base = _here / 'fig_s36off_vs_exp'
    fig.savefig(str(out_base) + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(str(out_base) + '.pdf', bbox_inches='tight')
    print(f'saved: {out_base}.png and .pdf')


if __name__ == '__main__':
    main()
