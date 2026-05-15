#!/usr/bin/env python3
"""Plot all RHS terms of dC_O3/dt vs depth for the v2 cache.

Reuses diag_o3_deep_source_v2.csv (must be run beforehand).

  dC_O3/dt = div_diff(z) + chem_net(z) + MT_in * delta_{j,0}

Two figures:

  Figure 1 (per-time overlay):  one panel per snapshot (1/2/4/6/8 min);
    each panel overlays chem_net(z), div_diff(z), dCdt(z), and the MT
    contribution as a single point at z=cell0.

  Figure 2 (per-quantity time evolution): one panel per RHS term, with
    the 5 snapshot lines coloured by time (viridis).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV = Path(__file__).resolve().parent / 'diag_o3_deep_source_v2.csv'
OUT_DIR = Path(__file__).resolve().parent
SNAP_TIMES_MIN = [1, 2, 4, 6, 8]


def main():
    df = pd.read_csv(CSV)
    df['t_min_round'] = df['t_min'].round(2)
    print(f'rows={len(df)}, times={sorted(df["t_min_round"].unique())}')

    # ---------------------------------------------------------------
    # Figure 1: per-time overlay of all RHS terms (incl. dCdt).
    # ---------------------------------------------------------------
    fig1, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
    color_chem = '#d62728'   # red
    color_diff = '#1f77b4'   # blue
    color_dcdt = 'black'
    color_mt   = '#2ca02c'   # green

    for ti, t_target in enumerate(SNAP_TIMES_MIN):
        ax = axes.flat[ti]
        sub = df[df['t_min_round'] == float(t_target)].sort_values('z_mm')
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        z = sub['z_mm'].values

        ax.plot(z, sub['chem_net'].values, color=color_chem, lw=1.4,
                label='chem_net')
        ax.plot(z, sub['div_diff'].values, color=color_diff, lw=1.4,
                label='div_diff')
        ax.plot(z, sub['dCdt_actual'].values, color=color_dcdt,
                lw=2.0, ls='--', label='dC/dt (actual)')

        # MT_in is concentrated at cell 0 -- single marker, since the
        # per-cell array has zero MT for j>0. residual at j=0 equals MT.
        surf = sub[sub['cell_idx'] == 0]
        if len(surf):
            mt_val = float(surf['residual'].iloc[0])
            ax.plot([float(surf['z_mm'].iloc[0])], [mt_val],
                    marker='^', color=color_mt, markersize=14,
                    markeredgecolor='black', linestyle='none',
                    label=f'MT (j=0) = {mt_val:+.2e}')

        ax.axhline(0, color='gray', lw=0.4)
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=1e-30)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('rate (M/s)')
        ax.set_title(f't = {t_target} min', fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8, loc='best')

    # Hide unused 6th panel.
    for k in range(len(SNAP_TIMES_MIN), len(axes.flat)):
        axes.flat[k].set_visible(False)

    fig1.suptitle(
        r'O$_3$ PDE RHS terms (chem + diff + MT) vs depth -- '
        '3.6 kV HONOvar v2 (N$_z$=188)',
        fontsize=13, y=1.01,
    )
    fig1.tight_layout()

    out_png = OUT_DIR / 'fig_o3_rhs_terms_v2.png'
    out_pdf = OUT_DIR / 'fig_o3_rhs_terms_v2.pdf'
    fig1.savefig(out_png, dpi=150, bbox_inches='tight')
    fig1.savefig(out_pdf, bbox_inches='tight')
    print(f'\nsaved: {out_png}')
    print(f'saved: {out_pdf}')

    # ---------------------------------------------------------------
    # Figure 2: per-quantity time evolution (5 panels including ratio).
    # ---------------------------------------------------------------
    # Pre-compute residual / dCdt_actual ratio. Guard against div-by-zero
    # for cells with dCdt near floating-point underflow.
    df['ratio'] = np.where(
        np.abs(df['dCdt_actual']) > 1e-30,
        df['residual'] / df['dCdt_actual'],
        np.nan,
    )

    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    cmap = plt.get_cmap('viridis')
    tcols = [cmap(i / max(len(SNAP_TIMES_MIN) - 1, 1))
             for i in range(len(SNAP_TIMES_MIN))]

    quantities = [
        ('chem_net',    r'chem_net  [M/s]', 'symlog', 1e-30),
        ('div_diff',    r'div_diff  [M/s]', 'symlog', 1e-30),
        ('dCdt_actual', r'dC/dt actual  [M/s]', 'symlog', 1e-30),
        ('residual',    r'residual = dCdt - chem - div  [M/s]', 'symlog', 1e-30),
        ('ratio',       r'residual / (dC/dt actual)  [-]', 'symlog', 1e-3),
    ]

    for ax, (col, ylab, scale, lin) in zip(axes2.flat, quantities):
        for ti, t_target in enumerate(SNAP_TIMES_MIN):
            sub = df[df['t_min_round'] == float(t_target)].sort_values('z_mm')
            if len(sub) == 0:
                continue
            ax.plot(sub['z_mm'].values, sub[col].values,
                    color=tcols[ti], lw=1.5,
                    label=f'{t_target} min')
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_xscale('log')
        if scale == 'symlog':
            ax.set_yscale('symlog', linthresh=lin)
        else:
            ax.set_yscale(scale)
        ax.set_xlabel('z (mm)')
        ax.set_ylabel(ylab)
        ax.set_title(ylab, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9, loc='best')

    # Hide unused 6th panel.
    axes2.flat[5].set_visible(False)

    fig2.suptitle(
        r'Per-quantity time evolution -- O$_3$ RHS in 3.6 kV HONOvar v2',
        fontsize=13, y=1.01,
    )
    fig2.tight_layout()

    out_png2 = OUT_DIR / 'fig_o3_rhs_byquantity_v2.png'
    out_pdf2 = OUT_DIR / 'fig_o3_rhs_byquantity_v2.pdf'
    fig2.savefig(out_png2, dpi=150, bbox_inches='tight')
    fig2.savefig(out_pdf2, bbox_inches='tight')
    print(f'saved: {out_png2}')
    print(f'saved: {out_pdf2}')

    # ---------------------------------------------------------------
    # Mass-balance summary
    # ---------------------------------------------------------------
    print('\n=== Mass-balance closure ===')
    print(f'{"t_min":>6s}  {"MT @ j=0":>14s}  {"max |resid|, j>0":>18s}  '
          f'{"max |dCdt|, j>0":>18s}')
    for t_target in SNAP_TIMES_MIN:
        sub = df[df['t_min_round'] == float(t_target)]
        if len(sub) == 0:
            continue
        surf = sub[sub['cell_idx'] == 0]
        mt_supply = float(surf['residual'].iloc[0]) if len(surf) else 0.0
        rest = sub[sub['cell_idx'] > 0]
        max_resid = float(rest['residual'].abs().max()) if len(rest) else 0.0
        max_dcdt = float(rest['dCdt_actual'].abs().max()) if len(rest) else 0.0
        print(f'{t_target:>6.1f}  {mt_supply:>+14.3e}  '
              f'{max_resid:>18.3e}  {max_dcdt:>18.3e}')


if __name__ == '__main__':
    main()
