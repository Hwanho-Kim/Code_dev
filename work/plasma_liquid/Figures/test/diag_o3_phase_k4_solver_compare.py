#!/usr/bin/env python3
"""Phase K4: K3 toy with multiple time-integration methods.

Hypothesis: if BDF is the offender, switching the integrator should kill
the V-shape. We try BDF, LSODA, Radau (all implicit), and RK45 (explicit
adaptive).  Surface BC is implemented as a hard reset at the start of
each RHS call (no stiff penalty, so methods are not penalised on stiffness
of the BC term itself).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from config_1d import ACID_BASE_PAIRS


CACHE = (_proj_root / 'Figures' / 'DIW results'
         / '3.6kV_Humid_fitting_three_film_HONOvar' / 'cache'
         / 'three_film_abspecies_dg0.0100.npz')

K_R32 = 5.0e5
T_END = 600.0
SNAPSHOT_T = [60.0, 240.0, 480.0, 600.0]


def setup_toy():
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
    return chem, solver


def main():
    print('Phase K4: K3 toy with multiple time-integration methods')

    chem, solver = setup_toy()
    o3_idx = chem.species_idx['O3']
    h_idx = chem.species_idx['H+']
    no2_total_idx = chem.species_idx['HONO_total']
    N_z = solver.N_z
    N_s = solver.N_s
    z = solver.z_centers

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

    def rhs(t, y_o3):
        y_o3 = y_o3.copy()
        y_o3[0] = float(interp_t(surfO3_t, t))   # hard surface BC
        y_2d = np.full((N_z, N_s), 1e-30)
        y_2d[:, o3_idx] = np.maximum(y_o3, 1e-30)
        transport = solver._compute_sg_transport(y_2d, E_half)
        diff_o3 = transport[:, o3_idx]
        no2m = interp_t(NO2m_zt, t)
        sink = -K_R32 * no2m * np.maximum(y_o3, 0.0)
        dydt = diff_o3 + sink
        dydt[0] = 0.0   # cell 0 frozen at target
        return dydt

    y0 = np.full(N_z, 1e-30)
    y0[0] = max(float(surfO3_t[0]), 1e-30)

    methods = [
        ('BDF',   {'atol': 1e-30, 'rtol': 1e-8, 'max_step': 2.0}),
        ('Radau', {'atol': 1e-30, 'rtol': 1e-8, 'max_step': 2.0}),
        ('LSODA', {'atol': 1e-30, 'rtol': 1e-8, 'max_step': 2.0}),
        ('RK45',  {'atol': 1e-30, 'rtol': 1e-6, 'max_step': 0.05}),
    ]

    results = {}
    for name, opts in methods:
        print(f'\n--- {name} ---')
        try:
            sol = solve_ivp(rhs, (0.0, T_END), y0, method=name,
                            t_eval=SNAPSHOT_T, **opts)
            print(f'  status: {sol.message}, nfev={sol.nfev}, '
                  f'njev={getattr(sol, "njev", 0)}')
            results[name] = sol
        except Exception as e:
            print(f'  FAILED: {e}')
            results[name] = None

    # Final-time spatial profiles side by side
    print('\n=== Final O3(z, 600s) profiles ===')
    cells = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]
    head = '  '.join(f'{name:>11s}' for name, _ in methods)
    print(f'{"z(mm)":>8s}  {head}')
    for j in cells:
        row = []
        for name, _ in methods:
            sol = results[name]
            if sol is None:
                row.append('FAILED')
            else:
                row.append(f'{sol.y[j, -1]:>11.2e}')
        print(f'{z[j]*1e3:>8.3f}  ' + '  '.join(row))

    # V-shape detection per method
    print('\n=== V-shape detection at t=600s ===')
    for name, _ in methods:
        sol = results[name]
        if sol is None:
            print(f'  {name}: solver failed')
            continue
        cf = sol.y[:, -1]
        rises = []
        for j in range(N_z - 1):
            if cf[j + 1] > cf[j] * 1.01 and cf[j + 1] > 1e-25:
                rises.append((j, cf[j], cf[j + 1]))
        if not rises:
            print(f'  {name}: monotonic (no V-shape)')
            continue
        # find min and post-min max
        valid = cf > 1e-25
        if not valid.any():
            print(f'  {name}: no significant cells')
            continue
        cf_safe = np.where(cf > 1e-25, cf, np.inf)
        j_min = int(np.argmin(cf_safe))
        j_max_after = j_min + int(np.argmax(cf[j_min:]))
        recovery = cf[j_max_after] / max(cf[j_min], 1e-100)
        print(f'  {name}: V-shape — min at z={z[j_min]*1e3:.3f}mm '
              f'(O3={cf[j_min]:.2e}), max at z={z[j_max_after]*1e3:.3f}mm '
              f'(O3={cf[j_max_after]:.2e}), recovery x{recovery:.2e}')


if __name__ == '__main__':
    main()
