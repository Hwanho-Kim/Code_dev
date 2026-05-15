#!/usr/bin/env python3
"""Plot c_O3 * c_NO2- (the R32 sink-rate driver) versus depth at
1, 2, 4, 6, 8 min from the v2 cache. R32 rate = k_R32 * c_O3 * c_NO2-,
so this product directly shows where R32 actually consumes O3.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from config_1d import ACID_BASE_PAIRS


CACHE = (_proj_root / 'Figures' / 'DIW results'
         / '3.6kV_Humid_fitting_three_film_HONOvar_v2' / 'cache'
         / 'three_film_abspecies_dg0.0100.npz')

OUT_DIR = Path(__file__).resolve().parent
SNAP_TIMES_MIN = [1, 2, 4, 6, 8]
K_R32 = 5.0e5


def main():
    print(f'loading cache: {CACHE}')
    data = np.load(CACHE, allow_pickle=True)
    snap_y = data['snap_y']
    snap_t = np.asarray(data['snap_t'], dtype=float)
    z_mm = np.asarray(data['z_centers'], dtype=float) * 1e3
    N_z = int(data['N_z'])
    print(f'  N_z={N_z}, snap_t in [{snap_t[0]:.1f}, {snap_t[-1]:.1f}] s, '
          f'z range = [{z_mm[0]:.4f}, {z_mm[-1]:.4f}] mm')

    chem = AqueousChemistry1D(saline_mode=False)
    o3_idx = chem.species_idx['O3']
    h_idx = chem.species_idx['H+']
    no2_total_idx = chem.species_idx['HONO_total']

    pKa = ACID_BASE_PAIRS['HONO_total'][2]
    Ka = 10.0 ** (-pKa)

    # Pick snapshot indices
    snap_idx = []
    for t_min in SNAP_TIMES_MIN:
        tgt = t_min * 60.0
        si = int(np.argmin(np.abs(snap_t - tgt)))
        snap_idx.append(si)
        print(f'  t={t_min} min  -> snap_idx={si}, t_actual={snap_t[si]:.2f} s')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(snap_idx) - 1, 1))
              for i in range(len(snap_idx))]

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Depth z (mm)')
        ax.set_xlim(z_mm[0], z_mm[-1])
        ax.grid(True, alpha=0.3, which='both')

    # Panel (a): c_O3 (sanity)
    ax = axes[0]
    for ci, si in enumerate(snap_idx):
        c_o3 = np.maximum(snap_y[si, :, o3_idx], 1e-40)
        ax.plot(z_mm, c_o3, color=colors[ci], lw=1.6,
                label=f'{snap_t[si]/60:.0f} min')
    ax.set_ylabel('[O$_3$] (M)')
    ax.set_title('(a) O$_3$ concentration', fontweight='bold', loc='left')
    ax.legend(fontsize=9)

    # Panel (b): c_NO2-
    ax = axes[1]
    for ci, si in enumerate(snap_idx):
        h = np.maximum(snap_y[si, :, h_idx], 1e-14)
        no2m = snap_y[si, :, no2_total_idx] * Ka / (h + Ka)
        no2m = np.maximum(no2m, 1e-40)
        ax.plot(z_mm, no2m, color=colors[ci], lw=1.6,
                label=f'{snap_t[si]/60:.0f} min')
    ax.set_ylabel('[NO$_2^-$] (M)')
    ax.set_title('(b) NO$_2^-$ concentration', fontweight='bold', loc='left')
    ax.legend(fontsize=9)

    # Panel (c): c_O3 * c_NO2-  (R32 driver)
    ax = axes[2]
    for ci, si in enumerate(snap_idx):
        c_o3 = snap_y[si, :, o3_idx]
        h = np.maximum(snap_y[si, :, h_idx], 1e-14)
        no2m = snap_y[si, :, no2_total_idx] * Ka / (h + Ka)
        product = np.abs(c_o3 * no2m)
        product = np.maximum(product, 1e-60)
        ax.plot(z_mm, product, color=colors[ci], lw=1.6,
                label=f'{snap_t[si]/60:.0f} min')
    ax.set_ylabel(r'[O$_3$] $\times$ [NO$_2^-$] (M$^2$)')
    ax.set_title(r'(c) R32 driver: $c_{O_3} \cdot c_{NO_2^-}$',
                 fontweight='bold', loc='left')
    ax.legend(fontsize=9)

    fig.suptitle(
        '3.6 kV HONOvar v2 — R32 sink driver across depth\n'
        f'(R32 rate = $k$ $\\cdot$ $c_{{O_3}}$ $\\cdot$ $c_{{NO_2^-}}$, '
        f'$k$ = {K_R32:.1e} M$^{{-1}}$ s$^{{-1}}$)',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_png = OUT_DIR / 'fig_o3_no2_product_v2.png'
    out_pdf = OUT_DIR / 'fig_o3_no2_product_v2.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'\nsaved: {out_png}')
    print(f'saved: {out_pdf}')

    # Quick sanity dump: position of max c_O3 * c_NO2- per snapshot
    print('\n=== Position of max R32 driver per time ===')
    print(f'{"t(min)":>6s}  {"z_max (mm)":>11s}  {"max product (M^2)":>20s}  '
          f'{"R32 rate (M/s)":>17s}')
    for ci, si in enumerate(snap_idx):
        c_o3 = snap_y[si, :, o3_idx]
        h = np.maximum(snap_y[si, :, h_idx], 1e-14)
        no2m = snap_y[si, :, no2_total_idx] * Ka / (h + Ka)
        product = c_o3 * no2m
        j_max = int(np.argmax(np.abs(product)))
        print(f'{snap_t[si]/60:>6.0f}  {z_mm[j_max]:>11.4f}  '
              f'{product[j_max]:>20.3e}  {K_R32 * product[j_max]:>17.3e}')


if __name__ == '__main__':
    main()
