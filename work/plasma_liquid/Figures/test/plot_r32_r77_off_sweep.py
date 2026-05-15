#!/usr/bin/env python3
"""Plot OH/O3 time evolution at 2.6 vs 3.6 kV for the R27/R32/R77 OFF sweep.

Top row: 2.6 kV, bottom row: 3.6 kV. Left column OH, right column O3.
Final bar chart shows OH/baseline ratios per voltage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))
from chemistry_1d import AqueousChemistry1D

OUT_DIR = Path(__file__).resolve().parent
PREFIX = 'r27_r32_r77_sweep'
VOLTS = ['2.6kV', '3.6kV']
CASES = ['baseline', 'R27_off', 'R32_off', 'R77_off', 'R32_R77_off']
LABELS = {
    'baseline':    'Baseline',
    'R27_off':     'R27 OFF (O$_3$+OH)',
    'R32_off':     'R32 OFF (O$_3$+NO$_2^-$)',
    'R77_off':     'R77 OFF (OH+NO$_2^-$)',
    'R32_R77_off': 'R32+R77 OFF',
}
COLORS = {
    'baseline':    '#222222',
    'R27_off':     '#7e57c2',
    'R32_off':     '#e07856',
    'R77_off':     '#2a6a8b',
    'R32_R77_off': '#c83737',
}


def load(volt, case):
    return dict(np.load(OUT_DIR / f'{PREFIX}_{volt}_{case}.npz',
                        allow_pickle=True))


def vol_avg_t(snap_y, dz, L, sp_idx):
    return (snap_y[:, :, sp_idx] * dz[None, :]).sum(axis=1) / L


def main():
    chem = AqueousChemistry1D(saline_mode=False)
    i_OH = chem.species_idx['OH']
    i_O3 = chem.species_idx['O3']

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.45,
                          wspace=0.25)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)]
                     for r in range(2)])
    ax_bar = fig.add_subplot(gs[2, :])

    for row, volt in enumerate(VOLTS):
        results = {c: load(volt, c) for c in CASES}
        for col, (sp_idx, name) in enumerate([(i_OH, 'OH'),
                                              (i_O3, 'O$_3$')]):
            ax = axes[row, col]
            all_vals = []
            for case in CASES:
                d = results[case]
                snap_t = d['snap_t']
                dz = d['dz_cells']
                L = float(d['L'])
                y_t = vol_avg_t(d['snap_y'], dz, L, sp_idx)
                ax.plot(snap_t / 60, np.maximum(y_t, 1e-40),
                        color=COLORS[case], lw=1.6, label=LABELS[case])
                mask = snap_t >= 10.0
                valid = y_t[mask]
                valid = valid[valid > 1e-30]
                if valid.size:
                    all_vals.append(valid)
            ax.set_yscale('log')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(f'{name} vol-avg [M]')
            ax.set_title(f'{volt} — {name}', fontweight='bold', loc='left')
            ax.set_xlim(0.1, 10)
            ax.grid(True, alpha=0.3, which='both')
            if all_vals:
                flat = np.concatenate(all_vals)
                lo = 10 ** np.floor(np.log10(flat.min()) - 0.2)
                hi = 10 ** np.ceil(np.log10(flat.max()) + 0.2)
                ax.set_ylim(lo, hi)

    axes[0, 0].legend(fontsize=8.5, loc='lower left',
                      bbox_to_anchor=(0.0, 1.04), ncol=3,
                      frameon=False)

    # bar chart of OH/baseline ratio per voltage
    n_cases = len(CASES) - 1  # exclude baseline
    x = np.arange(n_cases)
    width = 0.38
    case_labels = [LABELS[c] for c in CASES if c != 'baseline']
    OH_ratios = {volt: [] for volt in VOLTS}
    for volt in VOLTS:
        d_base = load(volt, 'baseline')
        OH_base = vol_avg_t(d_base['snap_y'], d_base['dz_cells'],
                            float(d_base['L']), i_OH)[-1]
        for case in CASES:
            if case == 'baseline':
                continue
            d = load(volt, case)
            OH_f = vol_avg_t(d['snap_y'], d['dz_cells'], float(d['L']),
                             i_OH)[-1]
            OH_ratios[volt].append(OH_f / OH_base)

    bar_colors = {'2.6kV': '#4a90e2', '3.6kV': '#e94e4e'}
    for i, volt in enumerate(VOLTS):
        offset = (i - 0.5) * width
        bars = ax_bar.bar(x + offset, OH_ratios[volt], width,
                          label=volt, color=bar_colors[volt],
                          edgecolor='black', linewidth=0.5)
        for b, val in zip(bars, OH_ratios[volt]):
            ax_bar.text(b.get_x() + b.get_width() / 2,
                        max(val, 1e-2) * 1.15,
                        f'{val:.2f}×' if val >= 0.1 else f'{val:.3f}×',
                        ha='center', va='bottom', fontsize=9)

    ax_bar.set_yscale('log')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(case_labels, fontsize=9)
    ax_bar.set_ylabel('[OH] / [OH]$_\\mathrm{baseline}$  (t=600s)')
    ax_bar.set_title('OH recovery factor per voltage (>1 = recovery, <1 = OH drops)',
                     fontweight='bold', loc='left')
    ax_bar.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax_bar.legend(loc='upper left', fontsize=10)
    ax_bar.grid(True, alpha=0.3, which='both', axis='y')

    fig.suptitle('R27/R32/R77 disable sweep — 2.6 vs 3.6 kV DIW Humid_fitting '
                 'three_film (vol-avg, N$_z$=49)', fontsize=11, y=0.995)
    fig.tight_layout()
    out_png = OUT_DIR / 'fig_r27_r32_r77_off_sweep.png'
    out_pdf = OUT_DIR / 'fig_r27_r32_r77_off_sweep.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'saved: {out_png}')
    print(f'saved: {out_pdf}')


if __name__ == '__main__':
    main()
