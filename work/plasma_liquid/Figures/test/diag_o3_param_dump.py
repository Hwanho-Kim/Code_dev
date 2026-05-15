#!/usr/bin/env python3
"""Per-cell, per-time dump of all parameters affecting O3(z, t).

Loads existing simulation cache for a given voltage / suffix and dumps a
CSV of every term that enters the O3 mass balance at each (cell, snapshot):

  - cell-local concentrations: O3, NO2- (via HONO_total speciation), OH,
    HO2_total, H+
  - per-reaction signed contribution to O3 mass pool (only reactions
    whose stoichiometry touches O3)
  - chem_net = sum of those contributions
  - div_diff: diffusion divergence of O3 from the SG flux scheme with
    E=0 (Poisson off matches our project default; reduces to Fickian
    central differences on the non-uniform grid)
  - dCdt_actual: central finite-difference of cached O3 between
    adjacent snapshots
  - residual = dCdt_actual - chem_net - div_diff (any non-zero residual
    in interior cells must come from the cell-zero MT BC propagating
    or from numerical leak)
  - timescales: tau_chem, tau_diff (per cell), lambda_react, Da

Mid-depth cells (z in [0.5, 2.0] mm) are summarised in the console
because the mid-dip in O3 spatial profile (4/6/8 min, see fig5_spatial)
is the diagnostic target.

Usage:
    python diag_o3_param_dump.py [VOLT] [SUFFIX]

Defaults: VOLT='3.6kV', SUFFIX='HONOvar' (49-cell, 5 um base grid).
SUFFIX='HONOvar_seedmin_dz1um' for the 63-cell, 1 um grid case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from config_1d import LIQUID_DIFFUSIVITY, D_LIQ_DEFAULT, ACID_BASE_PAIRS


SNAP_T_TARGET_S = [10, 30, 60, 120, 240, 360, 480, 600]
MID_Z_MM = (0.5, 2.0)
DEEP_Z_MM = (2.0, 6.0)
NO2_FRONT_THRESHOLD_M = 1e-6   # NO2- front = deepest z with conc above this


def cache_path(volt: str, suffix: str) -> Path:
    return (_project_root / 'Figures' / 'DIW results'
            / f'{volt}_Humid_fitting_three_film_{suffix}'
            / 'cache' / 'three_film_abspecies_dg0.0100.npz')


def load_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        'snap_y': data['snap_y'],
        'snap_t': np.asarray(data['snap_t'], dtype=float),
        'z_centers': np.asarray(data['z_centers'], dtype=float),
        'dz_cells': np.asarray(data['dz_cells'], dtype=float),
        'N_z': int(data['N_z']),
        'N_s': int(data['N_s']),
    }


def per_cell_rates(chem: AqueousChemistry1D,
                   y_2d: np.ndarray) -> np.ndarray:
    """Return shape (n_rxn, N_z) per-cell rate (M/s)."""
    N_z = y_2d.shape[0]
    n_rxn = len(chem.reactions)
    h_idx = chem.species_idx['H+']
    rates = np.zeros((n_rxn, N_z))
    for j in range(N_z):
        yc = np.clip(y_2d[j, :].copy(), chem.trace, 1.0)
        yc[h_idx] = max(yc[h_idx], 1e-14)
        spec = chem.speciate(yc)
        for ri, rxn_d in enumerate(chem._rxn_data):
            rates[ri, j] = chem._compute_single_rate(rxn_d, yc, spec)
    return rates


def find_o3_reactions(chem: AqueousChemistry1D) -> list[tuple[int, str, int]]:
    """Reactions with non-zero net O3 stoichiometry."""
    out = []
    for ri, rxn in enumerate(chem.reactions):
        r_count = int(rxn.get('reactants', {}).get('O3', 0))
        p_count = int(rxn.get('products', {}).get('O3', 0))
        net = p_count - r_count
        if net != 0:
            out.append((ri, rxn.get('label', f'R{ri}'), net))
    return out


def diffusion_divergence(c: np.ndarray, z_centers: np.ndarray,
                         dz_cells: np.ndarray, D: float) -> np.ndarray:
    """Per-cell -dJ/dz with no-flux BC (matches solver._compute_sg_transport
    when E_half=0, Z=0). c is shape (N_z,)."""
    N_z = c.shape[0]
    h_faces = z_centers[1:] - z_centers[:-1]   # center-to-center
    J = D * (c[:-1] - c[1:]) / h_faces          # shape (N_z-1,)
    div = np.zeros(N_z)
    if N_z >= 2:
        div[0] = -J[0] / dz_cells[0]
        if N_z >= 3:
            div[1:-1] = -(J[1:] - J[:-1]) / dz_cells[1:-1]
        div[-1] = J[-1] / dz_cells[-1]
    return div


def main():
    volt = sys.argv[1] if len(sys.argv) > 1 else '3.6kV'
    suffix = sys.argv[2] if len(sys.argv) > 2 else 'HONOvar'
    path = cache_path(volt, suffix)
    if not path.exists():
        print(f'CACHE NOT FOUND: {path}', file=sys.stderr)
        sys.exit(1)

    print(f'loading: {path}')
    cd = load_cache(path)
    print(f'  N_z={cd["N_z"]}, N_s={cd["N_s"]}, '
          f'n_snaps={len(cd["snap_t"])}, '
          f'dz0={cd["dz_cells"][0]:.2e} m, '
          f'z_max={cd["z_centers"][-1]*1e3:.3f} mm')

    chem = AqueousChemistry1D(saline_mode=False)
    o3_idx = chem.species_idx['O3']
    h_idx = chem.species_idx['H+']
    no2_total_idx = chem.species_idx.get('HONO_total')
    oh_idx = chem.species_idx.get('OH')
    ho2_total_idx = chem.species_idx.get('HO2_total')

    if no2_total_idx is None:
        raise RuntimeError('HONO_total species missing from chemistry')

    o3_rxns = find_o3_reactions(chem)
    print(f'O3-touching reactions: {len(o3_rxns)}')
    for ri, lbl, net in o3_rxns:
        print(f'  R{ri:3d}  net_O3={net:+d}  {lbl[:60]}')

    D_O3 = LIQUID_DIFFUSIVITY.get('O3', D_LIQ_DEFAULT)
    pKa_hono = ACID_BASE_PAIRS['HONO_total'][2]
    Ka_hono = 10.0 ** (-pKa_hono)

    snap_t = cd['snap_t']
    snap_y = cd['snap_y']
    z_centers = cd['z_centers']
    dz_cells = cd['dz_cells']
    N_z = cd['N_z']

    # Pick snapshot indices closest to target times (skip if out of range).
    snap_picks = []
    for t in SNAP_T_TARGET_S:
        if t < snap_t[0] or t > snap_t[-1]:
            continue
        snap_picks.append(int(np.argmin(np.abs(snap_t - t))))

    rxn_col_names = [f'rate_R{ri}' for ri, _, _ in o3_rxns]

    rows = []
    for si in snap_picks:
        t_now = float(snap_t[si])
        y_2d = snap_y[si]

        rates = per_cell_rates(chem, y_2d)   # (n_rxn, N_z)
        c_o3 = y_2d[:, o3_idx].astype(float)
        div_diff = diffusion_divergence(c_o3, z_centers, dz_cells, D_O3)

        # central-diff dCdt; edge snapshots fall back to forward/backward.
        if si == 0:
            dt_c = float(snap_t[si + 1] - snap_t[si])
            dCdt = (snap_y[si + 1, :, o3_idx] - snap_y[si, :, o3_idx]) / dt_c
        elif si == len(snap_t) - 1:
            dt_c = float(snap_t[si] - snap_t[si - 1])
            dCdt = (snap_y[si, :, o3_idx] - snap_y[si - 1, :, o3_idx]) / dt_c
        else:
            dt_c = float(snap_t[si + 1] - snap_t[si - 1])
            dCdt = (snap_y[si + 1, :, o3_idx]
                    - snap_y[si - 1, :, o3_idx]) / dt_c

        # NO2- via speciation.
        h_arr = np.maximum(y_2d[:, h_idx], 1e-14)
        no2m = y_2d[:, no2_total_idx] * Ka_hono / (h_arr + Ka_hono)

        for j in range(N_z):
            row = {
                't_min': t_now / 60.0,
                'z_mm': z_centers[j] * 1e3,
                'cell_idx': j,
                'C_O3': float(y_2d[j, o3_idx]),
                'C_NO2-': float(no2m[j]),
                'C_OH': float(y_2d[j, oh_idx]) if oh_idx is not None else 0.0,
                'C_HO2_total': float(y_2d[j, ho2_total_idx])
                                if ho2_total_idx is not None else 0.0,
                'pH': -np.log10(float(h_arr[j])),
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

            cnet_abs = abs(chem_net)
            row['tau_chem_s'] = (
                float(y_2d[j, o3_idx]) / cnet_abs if cnet_abs > 1e-30
                else float('inf')
            )
            row['tau_diff_s'] = float(dz_cells[j] ** 2 / D_O3)
            row['lambda_react_mm'] = (
                float(np.sqrt(D_O3 * row['tau_chem_s']) * 1e3)
                if row['tau_chem_s'] != float('inf') else float('inf')
            )
            row['Da_local'] = (row['tau_diff_s'] / row['tau_chem_s']
                               if row['tau_chem_s'] != float('inf')
                               and row['tau_chem_s'] > 0 else 0.0)
            rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = _script_dir / f'diag_o3_param_dump_{volt}_{suffix}.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nwrote {len(df)} rows -> {out_csv}')

    # Mid-depth summary
    mid = df[(df['z_mm'] >= MID_Z_MM[0]) & (df['z_mm'] <= MID_Z_MM[1])]
    print(f'\n=== Mid-depth (z in {MID_Z_MM} mm) closure summary ===')
    print(f'{"t_min":>6s}  {"|resid|/|dCdt|":>14s}  '
          f'{"max|residual|":>14s}  {"max|chem_net|":>14s}  '
          f'{"max|div_diff|":>14s}')
    for tval, grp in mid.groupby('t_min'):
        ratios = []
        for _, r in grp.iterrows():
            num = abs(r['residual'])
            den = max(abs(r['dCdt_actual']), 1e-30)
            ratios.append(num / den)
        print(f'{tval:6.1f}  {np.mean(ratios):14.3e}  '
              f'{grp["residual"].abs().max():14.3e}  '
              f'{grp["chem_net"].abs().max():14.3e}  '
              f'{grp["div_diff"].abs().max():14.3e}')

    # Surface O3 per time (cell 0) for erfc free-diffusion comparison.
    surf_O3 = {}
    for tval, grp in df.groupby('t_min'):
        srow = grp[grp['cell_idx'] == 0]
        if len(srow):
            surf_O3[tval] = float(srow.iloc[0]['C_O3'])

    # NO2- propagation front: deepest z with NO2- > threshold.
    print(f'\n=== NO2- propagation front (z where C_NO2- > '
          f'{NO2_FRONT_THRESHOLD_M:.0e} M) ===')
    print(f'{"t_min":>6s}  {"front (mm)":>10s}  {"surf NO2-":>11s}  '
          f'{"surf O3":>11s}')
    for tval, grp in df.groupby('t_min'):
        above = grp[grp['C_NO2-'] > NO2_FRONT_THRESHOLD_M]
        front_mm = above['z_mm'].max() if len(above) else 0.0
        surf_no2 = float(grp[grp['cell_idx'] == 0].iloc[0]['C_NO2-'])
        surf_o3 = surf_O3.get(tval, 0.0)
        print(f'{tval:6.1f}  {front_mm:10.3f}  '
              f'{surf_no2:11.3e}  {surf_o3:11.3e}')

    # Deep zone: O3 absolute level vs erfc free-diffusion estimate.
    # erfc estimate: c(z, t) = C_surface(t) * erfc(z / (2 sqrt(D t))).
    # If sim O3 >> erfc, deep level is internally seeded (chemistry source
    # or numerical leak); if sim O3 < erfc, sink is active even in deep.
    print(f'\n=== Deep-zone (z in {DEEP_Z_MM} mm) vs free-diffusion erfc ===')
    print(f'{"t_min":>6s}  {"z(mm)":>6s}  {"|O3| sim":>10s}  '
          f'{"NO2-":>10s}  {"erfc est":>12s}  '
          f'{"|sim|/erfc":>10s}  {"chem_net":>11s}')

    from scipy.special import erfc as _erfc
    deep = df[(df['z_mm'] >= DEEP_Z_MM[0])
              & (df['z_mm'] <= DEEP_Z_MM[1])].copy()
    for _, r in deep.iterrows():
        t_s = r['t_min'] * 60.0
        z_m = r['z_mm'] * 1e-3
        c_surf = surf_O3.get(r['t_min'], 0.0)
        eta = z_m / (2 * np.sqrt(D_O3 * t_s)) if t_s > 0 else float('inf')
        est = c_surf * float(_erfc(eta)) if eta < 25 else 0.0
        ratio = abs(r['C_O3']) / max(abs(est), 1e-100)
        print(f'{r["t_min"]:6.1f}  {r["z_mm"]:6.2f}  '
              f'{abs(r["C_O3"]):10.2e}  {r["C_NO2-"]:10.2e}  '
              f'{est:12.2e}  {ratio:10.2e}  {r["chem_net"]:+11.2e}')

    print('\n=== Mid-depth dominant terms (every ~10th row) ===')
    rxn_cols = [c for c in df.columns if c.startswith('rate_R')]
    step = max(1, len(mid) // 12)
    for i in range(0, len(mid), step):
        r = mid.iloc[i]
        contribs = [(c, r[c]) for c in rxn_cols if abs(r[c]) > 1e-30]
        contribs.sort(key=lambda x: -abs(x[1]))
        top = ', '.join(f'{c}={v:+.2e}' for c, v in contribs[:3])
        print(f"t={r['t_min']:5.1f}m  z={r['z_mm']:5.2f}mm  "
              f"O3={r['C_O3']:.2e}  NO2-={r['C_NO2-']:.2e}  "
              f"chem_net={r['chem_net']:+.2e}  "
              f"div_diff={r['div_diff']:+.2e}  "
              f"resid={r['residual']:+.2e}")
        if top:
            print(f"        top_rxn: {top}")


if __name__ == '__main__':
    main()
