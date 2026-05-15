#!/usr/bin/env python3
"""SG flux + mass conservation analysis for O3 in v2 cache.

Reanalysis only (no new simulation). For each (face j+1/2, t):

  J_{j+1/2}(t) = D_O3 × (c_j(t) - c_{j+1}(t)) / h_face_j

For each (z, t):
  - flux divergence div_diff
  - total mass in domain: integral c(z, t) dz (per cell, cumulative)
  - cumulative MT supply integrated over t (from cache surface c)

Output:
  - face-flux spatial profile vs time
  - mass-conservation closure
  - mass-flow direction at each face
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from config_1d import LIQUID_DIFFUSIVITY


CACHE = (_proj_root / 'Figures' / 'DIW results'
         / '3.6kV_Humid_fitting_three_film_HONOvar_v2'
         / 'cache' / 'three_film_abspecies_dg0.0100.npz')
OUT_DIR = Path(__file__).resolve().parent
SNAP_TIMES_MIN = [1, 2, 4, 6, 8]


def main():
    print(f'loading: {CACHE}', flush=True)
    data = np.load(CACHE, allow_pickle=True)
    snap_y = data['snap_y']
    snap_t = np.asarray(data['snap_t'], dtype=float)
    z_centers = np.asarray(data['z_centers'], dtype=float)
    dz_cells = np.asarray(data['dz_cells'], dtype=float)
    h_faces = z_centers[1:] - z_centers[:-1]
    N_z = int(data['N_z'])

    chem = AqueousChemistry1D(saline_mode=False)
    o3_idx = chem.species_idx['O3']
    no2_total_idx = chem.species_idx['HONO_total']
    h_idx = chem.species_idx['H+']

    D_O3 = LIQUID_DIFFUSIVITY['O3']
    D_HONO = LIQUID_DIFFUSIVITY['HONO']
    print(f'  N_z={N_z}, D_O3={D_O3:.3e}, D_HONO_total={D_HONO:.3e}', flush=True)

    # Snapshot indices
    snap_picks = [int(np.argmin(np.abs(snap_t - tm * 60.0)))
                  for tm in SNAP_TIMES_MIN]

    # 1) Total mass in domain at each cache snapshot
    print('\n=== Total O3 mass in domain (integral c(z) dz, mol/m²) ===', flush=True)
    total_mass = np.array([
        np.dot(snap_y[si, :, o3_idx], dz_cells)
        for si in range(len(snap_t))
    ])
    print(f'{"t (min)":>8s}  {"total mass":>14s}  {"surface c_O3":>14s}')
    for si in snap_picks:
        c_surf = float(snap_y[si, 0, o3_idx])
        print(f'{snap_t[si]/60:>8.1f}  {total_mass[si]:>14.3e}  '
              f'{c_surf:>14.3e}')

    # 2) SG face fluxes at each snapshot for O3
    # J_{j+1/2} = D × (c_j - c_{j+1}) / h_face_j
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(snap_picks) - 1, 1))
              for i in range(len(snap_picks))]

    z_face_mm = 0.5 * (z_centers[1:] + z_centers[:-1]) * 1e3

    print('\n=== Face J_{j+1/2} for O3 (M·m/s, +ve = downstream toward deep) ===', flush=True)
    for ti, si in enumerate(snap_picks):
        c = snap_y[si, :, o3_idx]
        J = D_O3 * (c[:-1] - c[1:]) / h_faces  # (N_z - 1,)
        # plot signed log magnitude
        ax = axes[0]
        ax.plot(z_face_mm, J, color=colors[ti], lw=1.4,
                label=f'{snap_t[si]/60:.0f} min')
        # cumulative mass through face (from t=0 to t_now)
        # we don't have continuous J, just snapshots: trapezoidal in time

    axes[0].set_xscale('log')
    axes[0].set_yscale('symlog', linthresh=1e-15)
    axes[0].axhline(0, color='gray', lw=0.4)
    axes[0].set_xlabel('z_face (mm)')
    axes[0].set_ylabel(r'$J_{j+\frac{1}{2}}$ for O$_3$  [M·m/s]')
    axes[0].set_title('(a) O$_3$ face flux (signed)', fontweight='bold', loc='left')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(fontsize=9)

    # 3) Same for NO2- (via HONO_total)
    for ti, si in enumerate(snap_picks):
        c_no2t = snap_y[si, :, no2_total_idx]
        J_no2t = D_HONO * (c_no2t[:-1] - c_no2t[1:]) / h_faces
        axes[1].plot(z_face_mm, J_no2t, color=colors[ti], lw=1.4,
                     label=f'{snap_t[si]/60:.0f} min')

    axes[1].set_xscale('log')
    axes[1].set_yscale('symlog', linthresh=1e-15)
    axes[1].axhline(0, color='gray', lw=0.4)
    axes[1].set_xlabel('z_face (mm)')
    axes[1].set_ylabel(r'$J_{j+\frac{1}{2}}$ for HONO_total  [M·m/s]')
    axes[1].set_title('(b) HONO_total (NO$_2^-$) face flux',
                      fontweight='bold', loc='left')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=9)

    fig.suptitle(
        '3.6 kV HONOvar v2 — face fluxes (positive = surface→deep)',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_png = OUT_DIR / 'fig_sg_face_flux_v2.png'
    out_pdf = OUT_DIR / 'fig_sg_face_flux_v2.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'\nsaved: {out_png}', flush=True)

    # 4) Mass conservation closure check.
    # Total domain mass at t=600s vs cumulative MT supply (trapezoidal).
    # MT supply per snapshot can be inferred from previous diagnostic
    # (residual at j=0). But here we re-compute MT_in directly from
    # the surface cell + chemistry at j=0:
    #   MT_in (per cell rate) = dCdt_actual - chem_net - div_diff
    # which is just the residual we already wrote.
    print('\n=== Mass conservation closure (cumulative) ===', flush=True)
    csv = OUT_DIR / 'diag_o3_deep_source_v2.csv'
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv)
        # Cumulative MT supply (trapezoidal in time over snapshot picks).
        df_surf = df[df['cell_idx'] == 0].sort_values('t_min')
        t_min = df_surf['t_min'].values
        mt_per_cell = df_surf['residual'].values  # M/s
        # MT contribution to total mass (per area):
        # dM/dt = MT_in × dz0 (since we only have it as concentration rate)
        dz0 = float(dz_cells[0])
        mt_mass_rate = mt_per_cell * dz0  # M·m/s = mol/m²/s
        # Trapezoidal integral over snap_t_picks (in seconds)
        t_s = t_min * 60.0
        from scipy.integrate import trapezoid
        cum_mt_mass = trapezoid(mt_mass_rate, t_s)
        # Compare to actual mass at t=8min minus mass at t=0.166min
        m_start = total_mass[snap_picks[0]]
        m_end = total_mass[snap_picks[-1]]
        delta_m = m_end - m_start
        print(f'  Cumulative MT supply (trapz):       {cum_mt_mass:.3e}  mol/m²')
        print(f'  Mass change in domain (t=8m - t=1m): {delta_m:.3e}  mol/m²')
        print(f'  Implied chemistry consumption:        {cum_mt_mass - delta_m:.3e}  mol/m²')
        if cum_mt_mass != 0:
            print(f'  Closure (consumed/supplied):         '
                  f'{(cum_mt_mass - delta_m)/cum_mt_mass:.3%}')
    else:
        print(f'  CSV not found: {csv}')


if __name__ == '__main__':
    main()
