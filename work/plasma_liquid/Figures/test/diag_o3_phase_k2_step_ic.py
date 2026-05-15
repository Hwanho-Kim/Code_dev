#!/usr/bin/env python3
"""Phase K2: Pure diffusion with step IC + extreme gradient.

Hypothesis: with chemistry OFF and only SG transport, an initial step
function should evolve into the erfc-like penetration profile of pure
diffusion. If a deep peak appears, SG/BDF coupling injects spurious mass.

IC: c(z<0.1mm) = 1e-6, c(z>=0.1mm) = 1e-30 (15-order step).
RHS: SG transport only (chemistry suppressed by zeroing all reactions).
Time integration: scipy BDF, atol=1e-30, rtol=1e-8, t_end=600 s.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc as scipy_erfc

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from config_1d import LIQUID_DIFFUSIVITY


D_O3 = LIQUID_DIFFUSIVITY['O3']
C0 = 1e-6
TRACE = 1e-30
STEP_Z = 1e-4   # step at 100 um (cell ~20 in default grid)
T_END = 600.0
SNAPSHOT_T = [10.0, 60.0, 120.0, 240.0, 360.0, 480.0, 600.0]


def main():
    print('Phase K2: Pure diffusion step IC, chemistry OFF, SG transport only')

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
    z = solver.z_centers
    N_z = solver.N_z
    N_s = solver.N_s

    # Step IC. All other species at trace.
    y0_2d = np.full((N_z, N_s), TRACE)
    y0_2d[:, o3_idx] = np.where(z < STEP_Z, C0, TRACE)
    y0 = y0_2d.ravel()

    # RHS: only SG transport for the O3 column. Chemistry zeroed.
    E_half = np.zeros(max(N_z - 1, 0))
    o3_offset = o3_idx

    def rhs(t, y):
        y_2d = y.reshape((N_z, N_s))
        transport = solver._compute_sg_transport(y_2d, E_half)
        # Apply only to O3, leave other species frozen.
        dydt_2d = np.zeros_like(y_2d)
        dydt_2d[:, o3_idx] = transport[:, o3_idx]
        return dydt_2d.ravel()

    sol = solve_ivp(
        rhs, (0.0, T_END), y0, method='BDF',
        t_eval=SNAPSHOT_T, atol=1e-30, rtol=1e-8, max_step=10.0,
    )
    print(f'  solver status: {sol.message}, nfev={sol.nfev}, njev={sol.njev}')

    print(f'\n{"t (s)":>8s}  ' + '  '.join(
        f'{z[j]*1e3:>10.2f}' for j in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]))
    print(f'{"":>8s}  ' + '  '.join(
        f'{"z(mm)":>10s}' for _ in range(11)))
    for ti, t_now in enumerate(SNAPSHOT_T):
        c_t = sol.y[:, ti].reshape((N_z, N_s))[:, o3_idx]
        cells = [c_t[j] for j in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]]
        cell_str = '  '.join(f'{c:>10.2e}' for c in cells)
        print(f'{t_now:>8.1f}  {cell_str}')

    # Compare to erfc free-diffusion estimate. Surface BC here is no-flux,
    # so the erfc form does not strictly apply (closed system); instead the
    # inner block diffuses both ways. We compare to a Gaussian/erfc combo:
    # for a step c0 on (-inf, STEP_Z], 0 on (STEP_Z, inf), the solution is
    # c(z, t) = (c0/2) * erfc((z - STEP_Z)/(2 sqrt(D t))).
    print('\nComparison to free-diffusion (z > step) at t=600s:')
    print(f'{"z (mm)":>8s}  {"sim O3":>12s}  {"erfc est":>12s}  '
          f'{"sim/erfc":>10s}  {"monotonic?":>10s}')
    c_final = sol.y[:, -1].reshape((N_z, N_s))[:, o3_idx]
    for j in range(N_z):
        if z[j] <= STEP_Z:
            continue
        eta = (z[j] - STEP_Z) / (2 * np.sqrt(D_O3 * T_END))
        est = (C0 / 2) * float(scipy_erfc(eta)) if eta < 25 else 0.0
        ratio = abs(c_final[j]) / max(abs(est), 1e-100)
        # Is c monotonically decreasing in z (after step)?
        if j + 1 < N_z and z[j + 1] > STEP_Z:
            mono = '✓' if c_final[j] >= c_final[j + 1] else '✗ NON-MONO'
        else:
            mono = '-'
        print(f'{z[j]*1e3:>8.2f}  {c_final[j]:>12.3e}  '
              f'{est:>12.3e}  {ratio:>10.2e}  {mono:>10s}')

    # Look for non-monotonicity (mid-dip / deep recovery in pure diffusion).
    print('\nNon-monotonic cells in final profile (any z where c[j+1] > c[j]):')
    found = 0
    for j in range(N_z - 1):
        if c_final[j + 1] > c_final[j] * 1.001:  # 0.1% tolerance
            print(f'  j={j} (z={z[j]*1e3:.3f}mm): c[j]={c_final[j]:.3e} -> '
                  f'c[j+1]={c_final[j+1]:.3e}  RISE')
            found += 1
    if not found:
        print('  none — profile is monotonic decreasing.')
    else:
        print(f'  {found} non-monotonic transitions detected.')


if __name__ == '__main__':
    main()
