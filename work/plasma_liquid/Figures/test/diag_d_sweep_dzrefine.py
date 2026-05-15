#!/usr/bin/env python3
"""D_O3 sweep + dz refinement to test V-shape sensitivity.

Five simulations (49-cell baseline grid except case 5):
  1) D_O3 × 0.1
  2) D_O3 × 1   (= baseline)
  3) D_O3 × 10
  4) D_O3 × 100
  5) dz_min=0.5µm + stretch=1.20 (~50 cells, surface refined)

For each, the rest of chemistry is unchanged. After 600 s integration,
compare O3(z, 8 min) spatial profile and V-shape recovery factor.
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

import gen_all_figures as gaf

VOLT = '3.6kV'
T_END = 600.0
DT_SNAP = 2.0
OUT_DIR = Path(__file__).resolve().parent
SNAP_TIMES_MIN = [1, 2, 4, 6, 8]


def run_case(label: str, d_factor: float = 1.0,
             dz_min: float = 5e-6, stretch: float = 1.12) -> dict:
    cache = OUT_DIR / f'd_sweep_{label}.npz'
    if cache.exists():
        print(f'[{label}] cache hit', flush=True)
        return dict(np.load(cache, allow_pickle=True))

    print(f'\n[{label}] dz_min={dz_min*1e6:.2f}µm, stretch={stretch}, '
          f'D_O3 ×{d_factor}', flush=True)

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
        chemistry=chem, dz_min=dz_min, stretch_ratio=stretch,
        mass_transfer_eta=1.0, saline_mode=False, fixed_cation_conc=0.0,
        bc_type='three_film', alpha_b=None, delta_gas=0.01,
    )
    if d_factor != 1.0:
        o3_idx = chem.species_idx['O3']
        original = float(solver.D_species[o3_idx])
        solver.D_species[o3_idx] = original * d_factor
        print(f'  D_O3: {original:.3e} → {solver.D_species[o3_idx]:.3e}',
              flush=True)
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hono2_gas,
                        h2o2_gas=h2o2_gas)
    print(f'  N_z={solver.N_z}', flush=True)
    t_eval = np.arange(DT_SNAP, T_END + 0.1, DT_SNAP)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, T_END), t_eval=t_eval, y0=y0,
                          verbose=False, dt_poisson=None)
    dt = time_mod.time() - t0
    print(f'  done {dt:.1f}s, success={result["success"]}, '
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
        'N_z': np.int64(N_z), 'N_s': np.int64(N_s),
        'L': np.float64(solver.L),
    }
    np.savez_compressed(cache, **out)
    return out


def main():
    cases = [
        ('D_x0.1',   {'d_factor': 0.1}),
        ('D_x1',     {'d_factor': 1.0}),
        ('D_x10',    {'d_factor': 10.0}),
        ('D_x100',   {'d_factor': 100.0}),
        ('dz_0.5um', {'dz_min': 5e-7, 'stretch': 1.20}),
    ]
    results = {}
    for label, kw in cases:
        results[label] = run_case(label, **kw)

    chem = AqueousChemistry1D(saline_mode=False)
    o3_idx = chem.species_idx['O3']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.get_cmap('plasma')
    case_colors = {lab: cmap(i / max(len(cases) - 1, 1))
                   for i, (lab, _) in enumerate(cases)}

    print('\n=== V-shape detection at t=8 min ===', flush=True)
    for label, kw in cases:
        sim = results[label]
        snap_t = sim['snap_t']
        si = int(np.argmin(np.abs(snap_t - 480.0)))
        z_mm = sim['z_centers'] * 1e3
        c = sim['snap_y'][si, :, o3_idx]
        c_plot = np.maximum(c, 1e-40)
        axes[0].plot(z_mm, c_plot, color=case_colors[label], lw=1.6,
                     label=label)
        if (c > 1e-25).any():
            c_safe = np.where(c > 1e-25, c, np.inf)
            j_min = int(np.argmin(c_safe))
            j_max_after = j_min + int(np.argmax(c[j_min:]))
            recovery = c[j_max_after] / max(c[j_min], 1e-100)
            print(f'  {label:>10s}: min z={z_mm[j_min]:6.3f}mm '
                  f'(c={c[j_min]:.2e}), peak z={z_mm[j_max_after]:6.3f}mm '
                  f'(c={c[j_max_after]:.2e}), recovery x{recovery:.2e}',
                  flush=True)
        else:
            print(f'  {label}: monotonic (no recovery)', flush=True)

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('z (mm)')
    axes[0].set_ylabel('O$_3$ (M)')
    axes[0].set_title('(a) O$_3$(z, 8 min)', fontweight='bold', loc='left')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(fontsize=9)

    # Panel b: recovery factor vs case
    labels = [lab for lab, _ in cases]
    recoveries = []
    for label, _ in cases:
        sim = results[label]
        snap_t = sim['snap_t']
        si = int(np.argmin(np.abs(snap_t - 480.0)))
        c = sim['snap_y'][si, :, o3_idx]
        if (c > 1e-25).any():
            c_safe = np.where(c > 1e-25, c, np.inf)
            j_min = int(np.argmin(c_safe))
            j_max_after = j_min + int(np.argmax(c[j_min:]))
            recovery = c[j_max_after] / max(c[j_min], 1e-100)
        else:
            recovery = 1.0
        recoveries.append(recovery)
    axes[1].bar(labels, recoveries,
                color=[case_colors[l] for l in labels])
    axes[1].set_yscale('log')
    axes[1].set_ylabel('V-shape recovery factor (peak/valley)')
    axes[1].set_title('(b) V-shape strength per case',
                      fontweight='bold', loc='left')
    axes[1].grid(True, alpha=0.3, which='major')
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha='right')

    fig.suptitle(
        'D_O3 sweep + dz refinement — V-shape sensitivity',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_png = OUT_DIR / 'fig_d_sweep_dzrefine.png'
    out_pdf = OUT_DIR / 'fig_d_sweep_dzrefine.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'\nsaved: {out_png}', flush=True)


if __name__ == '__main__':
    main()
