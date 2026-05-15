#!/usr/bin/env python3
"""Run full multi-species simulation with R32 disabled to test whether
V-shape in O3(z) profile still emerges.

Uses solver.solve() (same path as gen_all_figures.run_case) so that
charge balance, species-tight atol, and Poisson handling are identical
to the baseline simulation. Baseline result is loaded from the existing
HONOvar cache (R32 ON).
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

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from config_1d import ACID_BASE_PAIRS

import gen_all_figures as gaf


VOLT = '3.6kV'
T_END = 600.0
DT_SNAP = 2.0
SNAP_TIMES_MIN = [1, 2, 4, 6, 8]
OUT_DIR = Path(__file__).resolve().parent
BASELINE_CACHE = (
    _proj_root / 'Figures' / 'DIW results'
    / f'{VOLT}_Humid_fitting_three_film_HONOvar' / 'cache'
    / 'three_film_abspecies_dg0.0100.npz'
)
NOR32_CACHE = OUT_DIR / 'r32_off_v2.npz'


def find_R32(chem):
    for ri, rxn in enumerate(chem.reactions):
        if 'R32:' in rxn.get('label', ''):
            return ri, rxn['label']
    raise RuntimeError('R32 not found')


def run_no_R32(rerun: bool):
    if NOR32_CACHE.exists() and not rerun:
        print(f'[no-R32] loading cache {NOR32_CACHE}', flush=True)
        return dict(np.load(NOR32_CACHE, allow_pickle=True))

    print('[no-R32] preparing simulation...', flush=True)
    gaf.DEFAULT_GAS_SHEET = VOLT
    times, gas_conc = gaf.load_gas_data()
    rh80 = gaf.RH80_RATIOS.get(VOLT, {})
    h2o2_ratio = rh80.get('H2O2_O3', 0.003)
    hono2_ratio = rh80.get('HONO2_N2O5', 0.83)
    hono_ratio = rh80.get('HONO_NO2', 0.097)
    no2_arr = gas_conc.get('NO2', np.zeros_like(times))
    n2o5_arr = gas_conc.get('N2O5', np.zeros_like(times))
    o3_arr = gas_conc.get('O3', np.zeros_like(times))
    hono_gas = no2_arr * hono_ratio
    hono2_gas = n2o5_arr * hono2_ratio
    h2o2_gas = o3_arr * h2o2_ratio
    print(f'  HONO/NO2={hono_ratio}, HONO2/N2O5={hono2_ratio}, '
          f'H2O2/O3={h2o2_ratio}', flush=True)

    chem = AqueousChemistry1D(saline_mode=False)
    ri, label = find_R32(chem)
    print(f'  Disabling R{ri}: {label}', flush=True)
    chem.reactions[ri]['k'] = 0.0
    chem._rxn_data[ri]['k'] = 0.0
    chem._precompute_numba_arrays()
    print(f'  After disable: k_R32 = {chem._rxn_data[ri]["k"]}', flush=True)

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        fixed_cation_conc=0.0,
        bc_type='three_film',
        alpha_b=None,
        delta_gas=0.01,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hono2_gas,
                        h2o2_gas=h2o2_gas)
    print(f'  N_z={solver.N_z}, N_s={solver.N_s}', flush=True)

    t_eval = np.arange(DT_SNAP, T_END + 0.1, DT_SNAP)
    y0 = solver.build_initial_condition(initial_pH=7.0)

    print(f'  starting solver.solve(t_end={T_END}s) ...', flush=True)
    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, T_END), t_eval=t_eval, y0=y0,
        verbose=True, dt_poisson=None,
    )
    wall = time_mod.time() - t0
    print(f'  done in {wall:.1f}s, success={result["success"]}, '
          f'nfev={result.get("nfev", 0)}', flush=True)

    snap_t = np.array([0.0] + [float(tv) for tv in result['t_eval']])
    N_z, N_s = solver.N_z, solver.N_s
    snap_y = [y0.reshape(N_z, N_s).copy()]
    for yv in result['y_eval']:
        snap_y.append(np.array(yv).reshape(N_z, N_s))
    snap_y = np.array(snap_y)

    out = {
        'snap_t': snap_t, 'snap_y': snap_y,
        'z_centers': solver.z_centers, 'dz_cells': solver.dz_cells,
        'N_z': np.int64(N_z), 'N_s': np.int64(N_s), 'L': np.float64(solver.L),
        'wall_s': np.float64(wall),
    }
    np.savez_compressed(NOR32_CACHE, **out)
    print(f'  cached: {NOR32_CACHE}', flush=True)
    return out


def main():
    print('Loading baseline cache (R32 ON)...', flush=True)
    base = dict(np.load(BASELINE_CACHE, allow_pickle=True))
    print(f'  baseline N_z={int(base["N_z"])}, '
          f'snap_t in [{base["snap_t"][0]}, {base["snap_t"][-1]}]', flush=True)

    nor32 = run_no_R32(rerun=False)

    chem = AqueousChemistry1D(saline_mode=False)
    o3_idx = chem.species_idx['O3']

    z_mm = base['z_centers'] * 1e3
    snap_t = base['snap_t']
    snap_idx = [int(np.argmin(np.abs(snap_t - tm * 60.0)))
                for tm in SNAP_TIMES_MIN]
    snap_t_no = nor32['snap_t']
    snap_idx_no = [int(np.argmin(np.abs(snap_t_no - tm * 60.0)))
                   for tm in SNAP_TIMES_MIN]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(SNAP_TIMES_MIN) - 1, 1))
              for i in range(len(SNAP_TIMES_MIN))]

    for ax, sim, idxs, title in [
        (axes[0], base, snap_idx, 'Baseline (R32 ON)'),
        (axes[1], nor32, snap_idx_no, 'R32 disabled'),
    ]:
        for ci, si in enumerate(idxs):
            c = sim['snap_y'][si, :, o3_idx]
            c = np.maximum(c, 1e-40)
            ax.plot(z_mm, c, color=colors[ci], lw=1.6,
                    label=f'{snap_t[idxs[ci]] / 60:.0f} min'
                          if sim is base else f'{snap_t_no[si]/60:.0f} min')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Depth z (mm)')
        ax.set_ylabel(r'O$_3$ (M)')
        ax.set_title(title, fontweight='bold', loc='left')
        ax.set_xlim(z_mm[0], z_mm[-1])
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)

    fig.suptitle(
        f'V-shape test: R32 disabled vs baseline (3.6 kV HONOvar, three_film)',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_png = OUT_DIR / 'fig_r32_off_v2.png'
    out_pdf = OUT_DIR / 'fig_r32_off_v2.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'\nsaved: {out_png}', flush=True)
    print(f'saved: {out_pdf}', flush=True)

    print('\n=== V-shape detection at t=8 min ===', flush=True)
    si8 = snap_idx[-1]
    si8_no = snap_idx_no[-1]
    for name, sim, si in [('baseline', base, si8),
                          ('R32 OFF', nor32, si8_no)]:
        c = sim['snap_y'][si, :, o3_idx]
        if (c > 1e-25).any():
            c_safe = np.where(c > 1e-25, c, np.inf)
            j_min = int(np.argmin(c_safe))
            j_max_after = j_min + int(np.argmax(c[j_min:]))
            recovery = c[j_max_after] / max(c[j_min], 1e-100)
            print(f'  {name}: min at z={z_mm[j_min]:.3f}mm '
                  f'(c={c[j_min]:.2e}), peak at z={z_mm[j_max_after]:.3f}mm '
                  f'(c={c[j_max_after]:.2e}), recovery x{recovery:.2e}',
                  flush=True)
        else:
            print(f'  {name}: all c < 1e-25', flush=True)

    print('\n=== O3(z, 8min) baseline vs R32 OFF ===', flush=True)
    print(f'{"z (mm)":>8s}  {"baseline":>11s}  {"R32 OFF":>11s}  '
          f'{"OFF/base":>10s}', flush=True)
    cells = list(range(0, int(base['N_z']), max(1, int(base['N_z']) // 15)))
    for j in cells:
        cb = base['snap_y'][si8, j, o3_idx]
        cn = nor32['snap_y'][si8_no, j, o3_idx]
        ratio = cn / max(abs(cb), 1e-100)
        print(f'{z_mm[j]:>8.3f}  {cb:>11.2e}  {cn:>11.2e}  '
              f'{ratio:>10.2e}', flush=True)


if __name__ == '__main__':
    main()
