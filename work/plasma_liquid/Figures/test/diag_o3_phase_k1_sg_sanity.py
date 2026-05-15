#!/usr/bin/env python3
"""Phase K1: SG flux scheme sanity test on a Gaussian.

Hypothesis: With Poisson off (E_half=0) the SG transport should match the
analytical Fickian divergence D * d2c/dz2 to truncation order. If a known
function diverges from the analytical answer beyond grid-truncation error,
SG itself is suspect.

Input: c(z) = exp(-(z-z0)^2 / 2 sigma^2), z0=1mm, sigma=0.5mm. Computed
on the same stretched grid the project uses (dz_min in {1, 5, 20} um,
stretch=1.12). Verifies 2nd-order convergence.
"""
import sys
from pathlib import Path

import numpy as np

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from config_1d import LIQUID_DIFFUSIVITY


D_O3 = LIQUID_DIFFUSIVITY['O3']
Z0 = 1e-3        # 1 mm
SIGMA = 0.5e-3   # 0.5 mm


def gaussian_test(dz_min: float, stretch: float = 1.12) -> dict:
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=dz_min, stretch_ratio=stretch,
        saline_mode=False, fixed_cation_conc=0.0,
        bc_type='three_film', alpha_b=None, delta_gas=0.01,
    )
    n_t = 5
    times = np.linspace(0.0, 600.0, n_t)
    zero_gas = {sp: np.zeros(n_t) for sp in
                ['O', 'O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']}
    solver.set_gas_data(times=times, gas_conc_molecules=zero_gas)

    o3_idx = chem.species_idx['O3']
    z = solver.z_centers
    c = np.exp(-((z - Z0) ** 2) / (2 * SIGMA ** 2))

    y_2d = np.zeros((solver.N_z, solver.N_s))
    y_2d[:, o3_idx] = c

    E_half = np.zeros(max(solver.N_z - 1, 0))
    transport = solver._compute_sg_transport(y_2d, E_half)
    sg_div = transport[:, o3_idx]

    # Analytical: c'' = c * ((z-z0)^2 / sigma^4 - 1/sigma^2)
    d2c = c * ((z - Z0) ** 2 / SIGMA ** 4 - 1.0 / SIGMA ** 2)
    analytical = D_O3 * d2c

    abs_err = sg_div - analytical
    scale = np.max(np.abs(analytical))
    norm_err = abs_err / scale if scale > 0 else abs_err
    return {
        'dz_min': dz_min,
        'N_z': solver.N_z,
        'z': z,
        'c': c,
        'sg_div': sg_div,
        'analytical': analytical,
        'abs_err': abs_err,
        'norm_err': norm_err,
        'L_inf_norm': float(np.max(np.abs(norm_err))),
        'L2_norm': float(np.sqrt(np.mean(norm_err ** 2))),
    }


def main():
    print('Phase K1: SG flux Gaussian manufactured-solution test')
    print(f'  Gaussian: z0={Z0*1e3:.2f} mm, sigma={SIGMA*1e3:.2f} mm')
    print(f'  D_O3 = {D_O3:.3e} m^2/s')
    print()
    print(f'{"dz_min":>8s}  {"N_z":>4s}  '
          f'{"L_inf rel":>10s}  {"L2 rel":>10s}  {"order":>8s}')

    prev = None
    for dz in [20e-6, 5e-6, 1e-6]:
        res = gaussian_test(dz)
        order = '-'
        if prev is not None:
            ratio = prev['L2_norm'] / max(res['L2_norm'], 1e-30)
            scale = prev['dz_min'] / dz
            order = f'{np.log(ratio)/np.log(scale):.2f}'
        print(f'  {dz*1e6:>5.0f}µm  {res["N_z"]:>4d}  '
              f'{res["L_inf_norm"]:>10.3e}  '
              f'{res["L2_norm"]:>10.3e}  {order:>8s}')
        prev = res

    # Detailed cell-by-cell for the production grid (5 µm).
    print('\nProduction grid (dz_min=5 µm, stretch=1.12) cell errors:')
    res5 = gaussian_test(5e-6)
    z = res5['z']
    print(f'{"j":>4s}  {"z (mm)":>8s}  {"c":>10s}  '
          f'{"SG (M/s)":>12s}  {"Analytic":>12s}  {"|rel|":>10s}')
    for j in range(0, res5['N_z'], max(1, res5['N_z'] // 12)):
        a = res5['analytical'][j]
        s = res5['sg_div'][j]
        rel = abs((s - a) / a) if abs(a) > 1e-15 else 0.0
        print(f'{j:>4d}  {z[j]*1e3:>8.3f}  {res5["c"][j]:>10.3e}  '
              f'{s:>12.3e}  {a:>12.3e}  {rel:>10.3e}')


if __name__ == '__main__':
    main()
