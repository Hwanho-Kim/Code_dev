#!/usr/bin/env python3
"""Poisson ON + electroneutrality enforcement OFF.

Purpose: verify that the Poisson-OFF baseline V-shape vs Poisson-ON V-shape
difference is not masked by the algebraic H+ enforcement at macro-step
boundaries. With enforcement off, charge balance follows the Poisson PDE
solely, allowing real ionic ambipolar coupling.

Setup is identical to diag_poisson_on.py except _enforce_electroneutrality
is monkey-patched to no-op.
"""
from __future__ import annotations

import sys
import time as time_mod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_proj_root / 'Ver4_1D'))
sys.path.insert(0, str(_proj_root / 'Figures'))

from config_1d import POISSON
object.__setattr__(POISSON, 'enabled', True)

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

import gen_all_figures as gaf


VOLT = '3.6kV'
T_END = 600.0
DT_SNAP = 2.0
SNAP_TIMES_MIN = [1, 2, 4, 6, 8]
OUT_DIR = Path(__file__).resolve().parent
NEW_CACHE = OUT_DIR / 'poisson_no_enforce.npz'
BASELINE_CACHE = (
    _proj_root / 'Figures' / 'DIW results'
    / f'{VOLT}_Humid_fitting_three_film_HONOvar' / 'cache'
    / 'three_film_abspecies_dg0.0100.npz'
)
POISSON_ON_CACHE = OUT_DIR / 'poisson_on_baseline.npz'


def main():
    print(f'POISSON.enabled = {POISSON.enabled}', flush=True)

    if NEW_CACHE.exists():
        print(f'[no-enforce] cache hit', flush=True)
        sim = dict(np.load(NEW_CACHE, allow_pickle=True))
    else:
        print(f'[no-enforce] preparing simulation...', flush=True)
        gaf.DEFAULT_GAS_SHEET = VOLT
        times, gas_conc = gaf.load_gas_data()
        rh80 = gaf.RH80_RATIOS.get(VOLT, {})
        no2 = gas_conc.get('NO2', np.zeros_like(times))
        n2o5 = gas_conc.get('N2O5', np.zeros_like(times))
        o3 = gas_conc.get('O3', np.zeros_like(times))
        hono_gas = no2 * rh80.get('HONO_NO2', 0.097)
        hono2_gas = n2o5 * rh80.get('HONO2_N2O5', 0.83)
        h2o2_gas = o3 * rh80.get('H2O2_O3', 0.003)

        chem = AqueousChemistry1D(saline_mode=False)
        solver = PDESolver1D(
            chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
            mass_transfer_eta=1.0, saline_mode=False, fixed_cation_conc=0.0,
            bc_type='three_film', alpha_b=None, delta_gas=0.01,
        )
        print(f'  solver._poisson_enabled = {solver._poisson_enabled}',
              flush=True)
        # Disable electroneutrality enforcement
        solver._enforce_electroneutrality = lambda y: None
        print('  electroneutrality enforcement DISABLED (monkey-patched)',
              flush=True)
        solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                            hono_gas=hono_gas, hono2_gas=hono2_gas,
                            h2o2_gas=h2o2_gas)
        print(f'  N_z={solver.N_z}', flush=True)

        t_eval = np.arange(DT_SNAP, T_END + 0.1, DT_SNAP)
        y0 = solver.build_initial_condition(initial_pH=7.0)
        t0 = time_mod.time()
        result = solver.solve(t_span=(0, T_END), t_eval=t_eval, y0=y0,
                              verbose=True, dt_poisson=None)
        dt = time_mod.time() - t0
        print(f'  done {dt:.1f}s, success={result["success"]}, '
              f'nfev={result.get("nfev", 0)}', flush=True)

        snap_t = np.array([0.0] + [float(tv) for tv in result['t_eval']])
        N_z, N_s = solver.N_z, solver.N_s
        snap_y = [y0.reshape(N_z, N_s).copy()]
        for yv in result['y_eval']:
            snap_y.append(np.array(yv).reshape(N_z, N_s))
        snap_y = np.array(snap_y)
        sim = {
            'snap_t': snap_t, 'snap_y': snap_y,
            'z_centers': solver.z_centers, 'dz_cells': solver.dz_cells,
            'N_z': np.int64(N_z), 'N_s': np.int64(N_s),
            'L': np.float64(solver.L),
        }
        np.savez_compressed(NEW_CACHE, **sim)
        print(f'  cached: {NEW_CACHE}', flush=True)

    base = dict(np.load(BASELINE_CACHE, allow_pickle=True))
    poisson_on = (dict(np.load(POISSON_ON_CACHE, allow_pickle=True))
                  if POISSON_ON_CACHE.exists() else None)

    chem = AqueousChemistry1D(saline_mode=False)
    o3_idx = chem.species_idx['O3']
    z_mm = base['z_centers'] * 1e3

    n_panels = 3 if poisson_on is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6),
                             sharex=True, sharey=True)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(SNAP_TIMES_MIN) - 1, 1))
              for i in range(len(SNAP_TIMES_MIN))]

    panels = [
        (axes[0], base, 'Poisson OFF (enforce ON)'),
        (axes[-1], sim, 'Poisson ON + enforce OFF'),
    ]
    if poisson_on is not None:
        panels.insert(1, (axes[1], poisson_on, 'Poisson ON + enforce ON'))

    for ax, simd, title in panels:
        snap_t = simd['snap_t']
        idxs = [int(np.argmin(np.abs(snap_t - tm * 60.0)))
                for tm in SNAP_TIMES_MIN]
        for ci, si in enumerate(idxs):
            c = simd['snap_y'][si, :, o3_idx]
            c = np.maximum(c, 1e-40)
            ax.plot(z_mm, c, color=colors[ci], lw=1.6,
                    label=f'{SNAP_TIMES_MIN[ci]} min')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('O$_3$ (M)')
        ax.set_title(title, fontweight='bold', loc='left')
        ax.set_xlim(z_mm[0], z_mm[-1])
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8)

    fig.suptitle('Poisson + enforcement combinations on V-shape',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = OUT_DIR / 'fig_poisson_no_enforce.png'
    out_pdf = OUT_DIR / 'fig_poisson_no_enforce.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'\nsaved: {out_png}', flush=True)

    print('\n=== V-shape detection at t=8 min ===', flush=True)
    cases = [('Poisson OFF + enforce ON (baseline)', base)]
    if poisson_on is not None:
        cases.append(('Poisson ON  + enforce ON', poisson_on))
    cases.append(('Poisson ON  + enforce OFF', sim))
    for name, simd in cases:
        snap_t = simd['snap_t']
        si = int(np.argmin(np.abs(snap_t - 480.0)))
        c = simd['snap_y'][si, :, o3_idx]
        if (c > 1e-25).any():
            c_safe = np.where(c > 1e-25, c, np.inf)
            j_min = int(np.argmin(c_safe))
            j_max_after = j_min + int(np.argmax(c[j_min:]))
            recovery = c[j_max_after] / max(c[j_min], 1e-100)
            print(f'  {name:>40s}: min z={z_mm[j_min]:6.3f}mm '
                  f'(c={c[j_min]:.2e}), peak z={z_mm[j_max_after]:6.3f}mm '
                  f'(c={c[j_max_after]:.2e}), recovery x{recovery:.2e}',
                  flush=True)
        else:
            print(f'  {name}: monotonic', flush=True)


if __name__ == '__main__':
    main()
