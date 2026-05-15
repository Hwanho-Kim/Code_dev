#!/usr/bin/env python3
"""D_HONO_total (NO2- pool) sweep to test whether the V-shape recovery
factor depends on the D_O3 / D_HONO_total ratio.

Hypothesis (to verify): Mid sink wall propagates with NO2-. If D_HONO_total
is reduced, NO2- cannot reach deep cells -> wall stays near surface -> V
weakens. If D_HONO_total is increased, NO2- floods all cells -> sink is
uniform -> V also weakens (no protected deep zone).

Sweep:
  D_HONO_total × {0.1, 1, 10, 100}
  D_O3 unchanged at baseline (1.75e-9 m^2/s).

Comparison: combine with existing D_O3 sweep cache and produce a
ratio-vs-recovery figure.
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


def run_case(label: str, d_hono_factor: float = 1.0) -> dict:
    cache = OUT_DIR / f'd_hono_sweep_{label}.npz'
    if cache.exists():
        print(f'[{label}] cache hit', flush=True)
        return dict(np.load(cache, allow_pickle=True))

    print(f'\n[{label}] D_HONO_total ×{d_hono_factor}', flush=True)
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
    if d_hono_factor != 1.0:
        hono_idx = chem.species_idx['HONO_total']
        original = float(solver.D_species[hono_idx])
        solver.D_species[hono_idx] = original * d_hono_factor
        print(f'  D_HONO_total: {original:.3e} -> '
              f'{solver.D_species[hono_idx]:.3e}', flush=True)
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


def detect_v_shape(snap_y, snap_t, z_mm, o3_idx):
    si = int(np.argmin(np.abs(snap_t - 480.0)))
    c = snap_y[si, :, o3_idx]
    if not (c > 1e-25).any():
        return None, None, None, 1.0
    c_safe = np.where(c > 1e-25, c, np.inf)
    j_min = int(np.argmin(c_safe))
    j_max_after = j_min + int(np.argmax(c[j_min:]))
    recovery = c[j_max_after] / max(c[j_min], 1e-100)
    return c, j_min, j_max_after, recovery


def main():
    chem = AqueousChemistry1D(saline_mode=False)
    o3_idx = chem.species_idx['O3']

    cases_hono = [
        ('D_HONO_x0.1',  0.1),
        ('D_HONO_x1',    1.0),
        ('D_HONO_x10',   10.0),
        ('D_HONO_x100',  100.0),
    ]
    results_hono = {lab: run_case(lab, fac) for lab, fac in cases_hono}

    cases_o3 = [
        ('D_x0.1',   0.1),
        ('D_x1',     1.0),
        ('D_x10',    10.0),
        ('D_x100',   100.0),
    ]
    results_o3 = {}
    for lab, _ in cases_o3:
        cache = OUT_DIR / f'd_sweep_{lab}.npz'
        if cache.exists():
            results_o3[lab] = dict(np.load(cache, allow_pickle=True))
        else:
            print(f'WARN: D_O3 sweep cache missing for {lab}')

    # ----------------------------------------------------------------
    # 3-panel figure: (a) D_O3 sweep profile, (b) D_HONO sweep profile,
    # (c) recovery vs D_O3/D_HONO ratio
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    cmap_p = plt.get_cmap('plasma')
    cmap_v = plt.get_cmap('viridis')

    print('\n=== D_HONO sweep V-shape detection at t=8 min ===', flush=True)
    z_mm_all = {}
    rec_hono = {}
    for li, (lab, fac) in enumerate(cases_hono):
        sim = results_hono[lab]
        z_mm = sim['z_centers'] * 1e3
        z_mm_all[lab] = z_mm
        c, j_min, j_max, rec = detect_v_shape(
            sim['snap_y'], sim['snap_t'], z_mm, o3_idx)
        rec_hono[lab] = rec
        if c is None:
            print(f'  {lab}: monotonic')
            continue
        print(f'  {lab:>14s} (×{fac}): min z={z_mm[j_min]:6.3f}mm '
              f'(c={c[j_min]:.2e}), peak z={z_mm[j_max]:6.3f}mm '
              f'(c={c[j_max]:.2e}), recovery x{rec:.2e}', flush=True)
        color = cmap_v(li / max(len(cases_hono) - 1, 1))
        axes[1].plot(z_mm, np.maximum(c, 1e-40), color=color, lw=1.6,
                     label=lab)

    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('z (mm)'); axes[1].set_ylabel('O$_3$ (M)')
    axes[1].set_title('(b) D_HONO_total sweep at t=8 min',
                      fontweight='bold', loc='left')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=9)

    rec_o3 = {}
    for li, (lab, fac) in enumerate(cases_o3):
        if lab not in results_o3:
            continue
        sim = results_o3[lab]
        z_mm = sim['z_centers'] * 1e3
        c, j_min, j_max, rec = detect_v_shape(
            sim['snap_y'], sim['snap_t'], z_mm, o3_idx)
        rec_o3[lab] = rec
        if c is None:
            continue
        color = cmap_p(li / max(len(cases_o3) - 1, 1))
        axes[0].plot(z_mm, np.maximum(c, 1e-40), color=color, lw=1.6,
                     label=lab)

    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('z (mm)'); axes[0].set_ylabel('O$_3$ (M)')
    axes[0].set_title('(a) D_O3 sweep at t=8 min',
                      fontweight='bold', loc='left')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend(fontsize=9)

    # Panel c: recovery vs ratio.
    # D_O3 sweep: D_O3/D_HONO = (1.75e-9 × factor) / 1.85e-9
    # D_HONO sweep: D_O3/D_HONO = 1.75e-9 / (1.85e-9 × factor)
    ratios = []
    recs = []
    labels = []
    D_O3_base = 1.75e-9
    D_HONO_base = 1.85e-9
    for lab, fac in cases_o3:
        if lab in rec_o3:
            r = (D_O3_base * fac) / D_HONO_base
            ratios.append(r)
            recs.append(rec_o3[lab])
            labels.append(f'D_O3 ×{fac}')
    for lab, fac in cases_hono:
        if lab in rec_hono:
            r = D_O3_base / (D_HONO_base * fac)
            ratios.append(r)
            recs.append(rec_hono[lab])
            labels.append(f'D_HONO ×{fac}')

    axes[2].scatter(ratios, recs, s=80, c=range(len(ratios)),
                    cmap='tab10', edgecolor='black', zorder=3)
    for r, rec, lab in zip(ratios, recs, labels):
        axes[2].annotate(lab, (r, rec), xytext=(5, 5),
                         textcoords='offset points', fontsize=8)
    axes[2].set_xscale('log'); axes[2].set_yscale('log')
    axes[2].set_xlabel(r'$D_{O_3} / D_{HONO\_total}$')
    axes[2].set_ylabel('V-shape recovery factor')
    axes[2].set_title('(c) Recovery vs diffusivity ratio',
                      fontweight='bold', loc='left')
    axes[2].grid(True, alpha=0.3, which='both')
    axes[2].axvline(1.0, color='gray', ls='--', alpha=0.5,
                    label='ratio=1 (baseline)')
    axes[2].legend(fontsize=9)

    fig.suptitle(
        r'D_O3 vs D_HONO_total ratio — V-shape sensitivity',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_png = OUT_DIR / 'fig_d_ratio_sweep.png'
    out_pdf = OUT_DIR / 'fig_d_ratio_sweep.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'\nsaved: {out_png}', flush=True)


if __name__ == '__main__':
    main()
