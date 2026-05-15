#!/usr/bin/env python3
"""Phase K5: K3 toy with SG vs simple central-difference FD.

Hypothesis: with E_half=0 SG should reduce to standard Fickian central-
difference and match FD. Any divergence between SG and FD outputs
indicates the SG-specific Bernoulli implementation introduces a
numerical floor leak.

Both schemes use the same grid, same BDF integrator, same surface BC,
same R32 sink. Only the transport scheme differs.
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
T_END = 600.0
SNAPSHOT_T = [60.0, 240.0, 480.0, 600.0]


def main():
    print('Phase K5: SG vs FD transport in the K3 toy')

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
    dz_cells = solver.dz_cells
    h_faces = z[1:] - z[:-1]    # center-to-center distances
    inv_dz = 1.0 / dz_cells

    D_O3 = LIQUID_DIFFUSIVITY['O3']

    data = np.load(CACHE, allow_pickle=True)
    snap_y_full = data['snap_y']
    snap_t_full = np.asarray(data['snap_t'], dtype=float)
    pKa = ACID_BASE_PAIRS['HONO_total'][2]
    Ka_hono = 10.0 ** (-pKa)
    H_zt = np.maximum(snap_y_full[:, :, h_idx], 1e-14)
    NO2m_zt = snap_y_full[:, :, no2_total_idx] * Ka_hono / (H_zt + Ka_hono)
    surfO3_t = snap_y_full[:, 0, o3_idx].astype(float)
    dt_snap = float(snap_t_full[1] - snap_t_full[0])
    n_snap = len(snap_t_full)

    def interp_t(field, t_q):
        if t_q >= snap_t_full[-1]:
            return field[-1]
        if t_q <= snap_t_full[0]:
            return field[0]
        idx_f = (t_q - snap_t_full[0]) / dt_snap
        i0 = int(idx_f)
        if i0 >= n_snap - 1:
            return field[-1]
        return field[i0] * (1 - (idx_f - i0)) + field[i0 + 1] * (idx_f - i0)

    E_half = np.zeros(max(N_z - 1, 0))

    def fd_transport_o3(c):
        """Simple central-difference 1D Fickian transport for O3.
        c shape (N_z,). Returns d(c)/dt|_diff with no-flux BCs."""
        # Face fluxes J_{j+1/2} = D * (c_j - c_{j+1}) / h_{j+1/2}
        J = D_O3 * (c[:-1] - c[1:]) / h_faces
        out = np.zeros(N_z)
        out[0] = -J[0] * inv_dz[0]
        if N_z > 2:
            out[1:-1] = -(J[1:] - J[:-1]) * inv_dz[1:-1]
        out[-1] = J[-1] * inv_dz[-1]
        return out

    def rhs_sg(t, y_o3):
        y_o3 = y_o3.copy()
        y_o3[0] = float(interp_t(surfO3_t, t))
        y_2d = np.full((N_z, N_s), 1e-30)
        y_2d[:, o3_idx] = np.maximum(y_o3, 1e-30)
        diff = solver._compute_sg_transport(y_2d, E_half)[:, o3_idx]
        no2m = interp_t(NO2m_zt, t)
        sink = -K_R32 * no2m * np.maximum(y_o3, 0.0)
        dydt = diff + sink
        dydt[0] = 0.0
        return dydt

    def rhs_fd(t, y_o3):
        y_o3 = y_o3.copy()
        y_o3[0] = float(interp_t(surfO3_t, t))
        diff = fd_transport_o3(np.maximum(y_o3, 1e-30))
        no2m = interp_t(NO2m_zt, t)
        sink = -K_R32 * no2m * np.maximum(y_o3, 0.0)
        dydt = diff + sink
        dydt[0] = 0.0
        return dydt

    y0 = np.full(N_z, 1e-30)
    y0[0] = max(float(surfO3_t[0]), 1e-30)

    print('\n--- SG transport (BDF) ---')
    sol_sg = solve_ivp(rhs_sg, (0.0, T_END), y0, method='BDF',
                       t_eval=SNAPSHOT_T, atol=1e-30, rtol=1e-8,
                       max_step=2.0)
    print(f'  status: {sol_sg.message}, nfev={sol_sg.nfev}')

    print('\n--- FD transport (BDF) ---')
    sol_fd = solve_ivp(rhs_fd, (0.0, T_END), y0, method='BDF',
                       t_eval=SNAPSHOT_T, atol=1e-30, rtol=1e-8,
                       max_step=2.0)
    print(f'  status: {sol_fd.message}, nfev={sol_fd.nfev}')

    print('\n=== O3(z, 600s) side-by-side ===')
    cells = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]
    print(f'{"z (mm)":>8s}  {"SG":>11s}  {"FD":>11s}  {"FD/SG":>10s}')
    sg = sol_sg.y[:, -1]
    fd = sol_fd.y[:, -1]
    for j in cells:
        ratio = abs(fd[j]) / max(abs(sg[j]), 1e-100)
        print(f'{z[j]*1e3:>8.3f}  {sg[j]:>11.2e}  {fd[j]:>11.2e}  '
              f'{ratio:>10.2e}')

    # V-shape detection
    print('\n=== V-shape detection at t=600s ===')
    for name, c in [('SG', sg), ('FD', fd)]:
        valid = c > 1e-25
        if not valid.any():
            print(f'  {name}: no significant cells')
            continue
        c_safe = np.where(c > 1e-25, c, np.inf)
        j_min = int(np.argmin(c_safe))
        j_max_after = j_min + int(np.argmax(c[j_min:]))
        recovery = c[j_max_after] / max(c[j_min], 1e-100)
        if recovery > 10:
            print(f'  {name}: V-SHAPE — min at z={z[j_min]*1e3:.3f}mm '
                  f'(c={c[j_min]:.2e}), max at z={z[j_max_after]*1e3:.3f}mm '
                  f'(c={c[j_max_after]:.2e}), recovery x{recovery:.2e}')
        else:
            print(f'  {name}: monotonic (recovery x{recovery:.2e} < 10)')


if __name__ == '__main__':
    main()
