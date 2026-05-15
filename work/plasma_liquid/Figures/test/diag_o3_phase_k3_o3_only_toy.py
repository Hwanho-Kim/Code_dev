#!/usr/bin/env python3
"""Phase K3: Single-species O3 toy with imposed NO2-(z, t).

Hypothesis: with multi-species coupling removed and only R32
(O3 + NO2- -> NO3-) as a sink, does V-shape (mid-dip + deep recovery)
still emerge? If yes, multi-species is not the cause; SG + R32 + BDF
suffice. If no, multi-species coupling is the source.

Setup:
- 49-cell production grid (dz_min=5 um, stretch=1.12).
- Single dynamic variable: O3(z, t).
- NO2-(z, t) is read from the HONOvar cache (pH-speciated from
  HONO_total) and interpolated linearly in (t, z) at every RHS call.
- Surface BC: Dirichlet, c(0, t) = cache's surface O3(t), enforced by
  a stiff relaxation term (k_relax=1e3 1/s).
- Bottom BC: no-flux (matches solver default).
- Sink: -k_R32 * NO2-(z, t) * c(z, t)  with k_R32 = 5e5.
- Time integration: scipy BDF, atol=1e-30, rtol=1e-8.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from config_1d import LIQUID_DIFFUSIVITY, ACID_BASE_PAIRS


CACHE = (_proj_root / 'Figures' / 'DIW results'
         / '3.6kV_Humid_fitting_three_film_HONOvar' / 'cache'
         / 'three_film_abspecies_dg0.0100.npz')

K_R32 = 5.0e5
K_RELAX = 1e3      # 1/s, surface Dirichlet stiff penalty
T_END = 600.0
SNAPSHOT_T = [60.0, 120.0, 240.0, 360.0, 480.0, 600.0]


def main():
    print('Phase K3: O3-only toy with imposed NO2-(z, t)')

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, fixed_cation_conc=0.0,
        bc_type='three_film', alpha_b=None, delta_gas=0.01,
    )
    n_t = 5
    times = np.linspace(0.0, T_END, n_t)
    zero_gas = {sp: np.zeros(n_t) for sp in
                ['O', 'O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']}
    solver.set_gas_data(times=times, gas_conc_molecules=zero_gas)

    o3_idx = chem.species_idx['O3']
    h_idx = chem.species_idx['H+']
    no2_total_idx = chem.species_idx['HONO_total']
    N_z = solver.N_z
    N_s = solver.N_s
    z = solver.z_centers

    print(f'  N_z={N_z}, dz0={solver.dz_cells[0]:.2e} m, '
          f'z_max={z[-1]*1e3:.3f} mm, k_R32={K_R32:.1e}')

    print(f'  loading NO2-(z, t) from {CACHE.name}')
    data = np.load(CACHE, allow_pickle=True)
    snap_y_full = data['snap_y']
    snap_t_full = np.asarray(data['snap_t'], dtype=float)
    if int(data['N_z']) != N_z:
        raise RuntimeError(
            f'cache N_z={int(data["N_z"])} != solver N_z={N_z}')

    pKa = ACID_BASE_PAIRS['HONO_total'][2]
    Ka_hono = 10.0 ** (-pKa)

    H_zt = np.maximum(snap_y_full[:, :, h_idx], 1e-14)
    NO2m_zt = snap_y_full[:, :, no2_total_idx] * Ka_hono / (H_zt + Ka_hono)
    surfO3_t = snap_y_full[:, 0, o3_idx].astype(float)
    print(f'  NO2- range: {float(NO2m_zt.min()):.2e} -> '
          f'{float(NO2m_zt.max()):.2e} M')
    print(f'  surf O3 range: {float(surfO3_t.min()):.2e} -> '
          f'{float(surfO3_t.max()):.2e} M')

    # Pre-compute spacing for fast interpolation
    dt_snap = float(snap_t_full[1] - snap_t_full[0])
    n_snap = len(snap_t_full)

    def interp_t(field_2d_or_1d, t_query):
        if t_query >= snap_t_full[-1]:
            return field_2d_or_1d[-1]
        if t_query <= snap_t_full[0]:
            return field_2d_or_1d[0]
        idx_f = (t_query - snap_t_full[0]) / dt_snap
        i0 = int(idx_f)
        if i0 >= n_snap - 1:
            return field_2d_or_1d[-1]
        frac = idx_f - i0
        return field_2d_or_1d[i0] * (1 - frac) + field_2d_or_1d[i0 + 1] * frac

    E_half = np.zeros(max(N_z - 1, 0))

    def rhs(t, y_o3):
        # SG transport using full y_2d but only O3 column matters; the
        # solver's transport is per-species so other columns at trace are
        # passive.
        y_2d = np.full((N_z, N_s), 1e-30)
        y_2d[:, o3_idx] = np.maximum(y_o3, 1e-30)
        transport = solver._compute_sg_transport(y_2d, E_half)
        diff_o3 = transport[:, o3_idx]

        no2m = interp_t(NO2m_zt, t)
        sink = -K_R32 * no2m * np.maximum(y_o3, 0.0)

        dydt = diff_o3 + sink

        # Stiff surface Dirichlet
        target = float(interp_t(surfO3_t, t))
        dydt[0] += K_RELAX * (target - y_o3[0])
        return dydt

    y0 = np.full(N_z, 1e-30)
    y0[0] = max(float(surfO3_t[0]), 1e-30)

    print(f'  integrating BDF, t in [0, {T_END}] s ...')
    sol = solve_ivp(
        rhs, (0.0, T_END), y0, method='BDF',
        t_eval=SNAPSHOT_T, atol=1e-30, rtol=1e-8, max_step=2.0,
    )
    print(f'  status: {sol.message}')
    print(f'  nfev={sol.nfev}, njev={sol.njev}')

    # Spatial profile output
    print('\nO3(z, t) [M]:')
    cells = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]
    print(f'{"t(s)":>8s}  ' + '  '.join(
        f'{"z="}{z[j]*1e3:5.2f}mm' for j in cells))
    for ti, tv in enumerate(SNAPSHOT_T):
        row = sol.y[:, ti]
        print(f'{tv:>8.1f}  ' + '  '.join(f'{row[j]:>10.2e}' for j in cells))

    # Detect non-monotonic transitions in final profile
    print('\nNon-monotonic transitions in O3(z) at t=600s:')
    cf = sol.y[:, -1]
    found = []
    for j in range(N_z - 1):
        if cf[j + 1] > cf[j] * 1.01 and cf[j + 1] > 1e-25:
            found.append((j, z[j] * 1e3, cf[j], cf[j + 1]))
    if not found:
        print('  none — profile is monotonic decreasing.')
    else:
        for j, zmm, cj, cj1 in found:
            print(f'  j={j}  z={zmm:.3f}mm  c[j]={cj:.2e} -> '
                  f'c[j+1]={cj1:.2e}  RISE x{cj1/max(cj,1e-100):.2e}')
        # Find min then deeper max (V-shape signature)
        below = cf > 1e-25
        if below.any():
            j_min = int(np.argmin(np.where(cf > 1e-25, cf, np.inf)))
            j_max_after = j_min + int(np.argmax(cf[j_min:])) if j_min < N_z - 1 else j_min
            if j_max_after > j_min and cf[j_max_after] > cf[j_min] * 10:
                print(f'\n  V-SHAPE detected: min at z={z[j_min]*1e3:.3f}mm '
                      f'(O3={cf[j_min]:.2e}), recovery max at '
                      f'z={z[j_max_after]*1e3:.3f}mm '
                      f'(O3={cf[j_max_after]:.2e}), '
                      f'recovery factor x{cf[j_max_after]/max(cf[j_min],1e-100):.2e}')

    # Compare cached full simulation vs toy at t=600s
    print('\nCompare toy O3(z, 600s) to cached full sim O3(z, 600s):')
    cached_final = snap_y_full[-1, :, o3_idx]
    print(f'{"j":>4s}  {"z(mm)":>8s}  {"toy O3":>12s}  '
          f'{"cached O3":>12s}  {"toy/cached":>10s}')
    for j in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]:
        ratio = abs(cf[j]) / max(abs(cached_final[j]), 1e-100)
        print(f'{j:>4d}  {z[j]*1e3:>8.3f}  {cf[j]:>12.3e}  '
              f'{cached_final[j]:>12.3e}  {ratio:>10.2e}')


if __name__ == '__main__':
    main()
