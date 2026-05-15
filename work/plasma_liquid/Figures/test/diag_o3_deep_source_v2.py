#!/usr/bin/env python3
"""Identify mass source in deep cells (where R32 driver > mid dip).

For the v2 cache (N_z=188), tabulate at each (z, t):

  - c_O3, c_NO2-
  - chem_net = sum over all O3-touching reactions
  - per-reaction signed contribution (so we can see if R20/R26 etc are
    acting as internal sources)
  - div_diff: SG-equivalent diffusion divergence (FD with E=0)
  - dCdt: central finite-diff between adjacent snapshots
  - residual = dCdt - chem_net - div_diff (mass-balance error)

If mass appears in deep cells without diffusion-in or chemistry source,
that flags a numerical leak.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from config_1d import LIQUID_DIFFUSIVITY, ACID_BASE_PAIRS

CACHE = (_proj_root / 'Figures' / 'DIW results'
         / '3.6kV_Humid_fitting_three_film_HONOvar_v2'
         / 'cache' / 'three_film_abspecies_dg0.0100.npz')

OUT_DIR = Path(__file__).resolve().parent
SNAP_T_TARGET_S = [60, 120, 240, 360, 480]
DEEP_Z_RANGE_MM = (1.0, 6.0)


def per_cell_rates(chem, y_2d):
    N_z = y_2d.shape[0]
    n_rxn = len(chem.reactions)
    h_idx = chem.species_idx['H+']
    rates = np.zeros((n_rxn, N_z))
    for j in range(N_z):
        yc = np.clip(y_2d[j].copy(), chem.trace, 1.0)
        yc[h_idx] = max(yc[h_idx], 1e-14)
        spec = chem.speciate(yc)
        for ri, rxn_d in enumerate(chem._rxn_data):
            rates[ri, j] = chem._compute_single_rate(rxn_d, yc, spec)
    return rates


def find_o3_reactions(chem):
    out = []
    for ri, rxn in enumerate(chem.reactions):
        r_count = int(rxn.get('reactants', {}).get('O3', 0))
        p_count = int(rxn.get('products', {}).get('O3', 0))
        net = p_count - r_count
        if net != 0:
            out.append((ri, rxn.get('label', f'R{ri}'), net))
    return out


def fd_div(c_z, h_faces, dz_cells, D):
    """Per-cell -dJ/dz with no-flux BCs."""
    N_z = c_z.shape[0]
    J = D * (c_z[:-1] - c_z[1:]) / h_faces
    out = np.zeros(N_z)
    out[0] = -J[0] / dz_cells[0]
    if N_z > 2:
        out[1:-1] = -(J[1:] - J[:-1]) / dz_cells[1:-1]
    out[-1] = J[-1] / dz_cells[-1]
    return out


def main():
    print(f'loading: {CACHE}')
    data = np.load(CACHE, allow_pickle=True)
    snap_y = data['snap_y']
    snap_t = np.asarray(data['snap_t'], dtype=float)
    z_centers = np.asarray(data['z_centers'], dtype=float)
    dz_cells = np.asarray(data['dz_cells'], dtype=float)
    h_faces = z_centers[1:] - z_centers[:-1]
    N_z = int(data['N_z'])
    print(f'  N_z={N_z}, dz0={dz_cells[0]:.2e} m, '
          f'z_max={z_centers[-1]*1e3:.3f} mm')

    chem = AqueousChemistry1D(saline_mode=False)
    o3_idx = chem.species_idx['O3']
    h_idx = chem.species_idx['H+']
    no2_total_idx = chem.species_idx['HONO_total']
    D_O3 = LIQUID_DIFFUSIVITY['O3']
    pKa = ACID_BASE_PAIRS['HONO_total'][2]
    Ka = 10.0 ** (-pKa)

    o3_rxns = find_o3_reactions(chem)
    print(f'  O3-touching reactions: {len(o3_rxns)}')
    rxn_col_names = [f'rate_R{ri}' for ri, _, _ in o3_rxns]
    rxn_labels = {f'rate_R{ri}': lbl for ri, lbl, _ in o3_rxns}

    snap_picks = []
    for tv in SNAP_T_TARGET_S:
        if snap_t[0] <= tv <= snap_t[-1]:
            snap_picks.append(int(np.argmin(np.abs(snap_t - tv))))

    rows = []
    for si in snap_picks:
        t_now = float(snap_t[si])
        y_2d = snap_y[si]
        rates = per_cell_rates(chem, y_2d)
        c_o3 = y_2d[:, o3_idx].astype(float)
        div_diff = fd_div(c_o3, h_faces, dz_cells, D_O3)

        if 1 <= si <= len(snap_t) - 2:
            dt_c = float(snap_t[si + 1] - snap_t[si - 1])
            dCdt = (snap_y[si + 1, :, o3_idx]
                    - snap_y[si - 1, :, o3_idx]) / dt_c
        else:
            dCdt = np.zeros(N_z)

        h_arr = np.maximum(y_2d[:, h_idx], 1e-14)
        no2m = y_2d[:, no2_total_idx] * Ka / (h_arr + Ka)

        for j in range(N_z):
            row = {
                't_min': t_now / 60.0,
                'z_mm': z_centers[j] * 1e3,
                'cell_idx': j,
                'C_O3': float(c_o3[j]),
                'C_NO2-': float(no2m[j]),
                'div_diff': float(div_diff[j]),
                'dCdt_actual': float(dCdt[j]),
            }
            chem_net = 0.0
            for (ri, _, net), col in zip(o3_rxns, rxn_col_names):
                contrib = float(net) * float(rates[ri, j])
                row[col] = contrib
                chem_net += contrib
            row['chem_net'] = chem_net
            row['residual'] = (row['dCdt_actual']
                               - chem_net - row['div_diff'])
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'diag_o3_deep_source_v2.csv'
    df.to_csv(out_csv, index=False)
    print(f'wrote {len(df)} rows -> {out_csv}')

    # Filter to deep zone for tabulation.
    deep = df[(df['z_mm'] >= DEEP_Z_RANGE_MM[0])
              & (df['z_mm'] <= DEEP_Z_RANGE_MM[1])]

    print(f'\n=== deep zone (z in {DEEP_Z_RANGE_MM} mm) at each snapshot ===')
    # Subsample cells for readability
    cells_show = sorted(set(deep['cell_idx'].astype(int).tolist()))[::8]

    for tval in sorted(deep['t_min'].unique()):
        print(f'\n--- t = {tval:.1f} min ---')
        print(f'{"z(mm)":>7s}  {"C_O3":>10s}  {"NO2-":>10s}  '
              f'{"chem_net":>11s}  {"div_diff":>11s}  '
              f'{"dCdt":>11s}  {"resid":>11s}  '
              f'{"src/snk":>8s}')
        sub = deep[deep['t_min'] == tval]
        for j in cells_show:
            r = sub[sub['cell_idx'] == j]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            ratio = (r['div_diff'] / max(abs(r['chem_net']), 1e-40))
            print(f'{r["z_mm"]:>7.3f}  {r["C_O3"]:>10.2e}  '
                  f'{r["C_NO2-"]:>10.2e}  '
                  f'{r["chem_net"]:>+11.2e}  {r["div_diff"]:>+11.2e}  '
                  f'{r["dCdt_actual"]:>+11.2e}  {r["residual"]:>+11.2e}  '
                  f'{ratio:>+8.2e}')

    # Source identification: at each deep cell at t=8min, list top
    # contributions to chem_net.
    print('\n=== Top reaction contributions in deep cells at t=8 min ===')
    rxn_cols = [c for c in df.columns if c.startswith('rate_R')]
    t_target = 480.0 / 60.0
    sub = deep[deep['t_min'] == t_target]
    for j in cells_show:
        r = sub[sub['cell_idx'] == j]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        contribs = [(c, r[c]) for c in rxn_cols if abs(r[c]) > 1e-40]
        contribs.sort(key=lambda x: -abs(x[1]))
        top3 = contribs[:3]
        print(f'\nz={r["z_mm"]:.3f}mm  C_O3={r["C_O3"]:.2e}  '
              f'chem_net={r["chem_net"]:+.2e}  div_diff={r["div_diff"]:+.2e}')
        for col, val in top3:
            label = rxn_labels.get(col, col)[:55]
            sign = 'SOURCE' if val > 0 else 'sink'
            print(f'   {col:>8s}  {val:>+12.2e}  {sign}  {label}')

    # Identify whether deep cells are gaining mass overall (dCdt > 0)
    print('\n=== Net mass flow direction in deep cells ===')
    print(f'{"t_min":>6s}  ' + '  '.join(
        f'z={z_centers[j]*1e3:.2f}mm' for j in cells_show))
    for tval in sorted(deep['t_min'].unique()):
        sub = deep[deep['t_min'] == tval]
        signs = []
        for j in cells_show:
            r = sub[sub['cell_idx'] == j]
            if len(r) == 0:
                signs.append('     ')
                continue
            v = r.iloc[0]['dCdt_actual']
            if abs(v) < 1e-30:
                signs.append('  ~0 ')
            elif v > 0:
                signs.append(f'+{v:.1e}')
            else:
                signs.append(f'{v:.1e}')
        print(f'{tval:>6.1f}  ' + '  '.join(f'{s:>11s}' for s in signs))


if __name__ == '__main__':
    main()
