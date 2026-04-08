#!/usr/bin/env python3
"""
Figure 1-2: Mass transfer flux time series by BC type.

For each BC model, plots:
  - Row 1: Instantaneous MT flux (mol/m²/s → µM/s volume-avg)
  - Row 2: Cumulative MT (integrated flux, µM)
  for key species: N₂O₅, O₃, NO₂, NO₃

Columns = species, line color = BC type.
"""

import sys
import os
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ, GAS_TO_AQUEOUS_MAP, LIQUID_DIFFUSIVITY, D_LIQ_DEFAULT
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

# Species to track MT flux (gas_name → display label)
MT_SPECIES = [
    ('N2O5', 'N₂O₅'),
    ('O3',   'O₃'),
    ('NO2',  'NO₂'),
    ('NO3',  'NO₃'),
]

BC_CASES = [
    ('Dirichlet',      'dirichlet',  1.0),
    ('Two-film',       'two_film',   1.0),
    ('Film (αb=1)',    'film',       1.0),
    ('Film+αb=0.05',  'film_alpha', 0.05),
    ('Film+αb=0.03',  'film_alpha', 0.03),
    ('Film+αb=0.01',  'film_alpha', 0.01),
]

DT_SNAPSHOT = 10.0  # seconds


def load_gas_data():
    df = pd.read_csv(DEFAULT_CSV)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))
    if 'N2O4' not in df.columns or np.all(gas_conc['N2O4'] == 0):
        import math
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298)
            + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)
    return times, gas_conc


def run_and_collect_mt(times, gas_conc, bc_type, alpha_b, label):
    """Run simulation, extract MT flux at each snapshot."""
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type=bc_type,
        alpha_b=alpha_b,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=0, hono2_gas=0, h2o2_gas=0)
    t_end = float(times[-1])
    t_eval = np.arange(0, t_end + 0.1, DT_SNAPSHOT)
    t_eval = t_eval[t_eval <= t_end]

    print(f"\n{'='*60}")
    print(f"Running: {label} (bc={bc_type}, α_b={alpha_b})")
    print(f"{'='*60}")
    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval,
        verbose=True, dt_poisson=None,
    )
    wall = time_mod.time() - t0
    print(f"  Wall time: {wall:.0f}s, success={result['success']}")

    if not result['success']:
        print(f"  WARNING: solver failed for {label}")
        return None

    N_z, N_s = solver.N_z, solver.N_s
    L = solver.L

    # Build gas_name → (aq_idx, k_mt) mapping
    idx_to_name = {v: k for k, v in solver.species_idx.items()}
    mt_map = {}
    for aq_idx, k_mt, gas_sp, _, Ka in solver._interface_species:
        aq_name = idx_to_name[aq_idx]
        # Find gas name from aq name
        for g, a in GAS_TO_AQUEOUS_MAP.items():
            if a == aq_name:
                mt_map[g] = (aq_idx, k_mt, gas_sp, Ka)
                break

    # Extract MT flux at each snapshot
    snap_times = result['t_eval']
    n_snaps = len(snap_times)

    mt_data = {}  # gas_name → flux array (M/s, volume-averaged)
    for gas_name, _ in MT_SPECIES:
        mt_data[gas_name] = np.zeros(n_snaps)

    dz0 = solver.dz_cells[0]
    is_dirichlet = (bc_type == 'dirichlet')

    for si, tv in enumerate(snap_times):
        y2d = result['y_eval'][si].reshape(N_z, N_s)
        t_idx = max(0, min(int(tv / solver._dt_gas), solver._n_times - 1))

        for gas_name, _ in MT_SPECIES:
            if gas_name not in mt_map:
                continue
            aq_idx, k_mt, gas_sp, Ka = mt_map[gas_name]
            if is_dirichlet:
                # Dirichlet: flux from surface gradient D·(C[0]-C[1])/dz
                D_l = LIQUID_DIFFUSIVITY.get(gas_name, D_LIQ_DEFAULT)
                flux_surf = D_l * (y2d[0, aq_idx] - y2d[1, aq_idx]) / dz0
                mt_data[gas_name][si] = flux_surf / L
            else:
                C_eq = solver._get_C_eq_fast(gas_sp, t_idx)
                C_surface = y2d[0, aq_idx]
                # For _total species: only molecular fraction transfers
                hp_idx = solver._h_plus_idx
                if Ka is not None and hp_idx >= 0:
                    h_s = max(y2d[0, hp_idx], 1e-14)
                    C_surface = C_surface * h_s / (h_s + Ka)
                mt_data[gas_name][si] = k_mt * (C_eq - C_surface) / L

    return {
        'label': label,
        'snap_times': snap_times,
        'mt_data': mt_data,
        'wall': wall,
        'pH': result['pH_avg'],
    }


def plot_mt_figure(all_results):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 8, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
    })

    n_species = len(MT_SPECIES)
    fig, axes = plt.subplots(2, n_species, figsize=(4.5 * n_species, 8),
                             sharex=True)

    # Color cycle for BC types
    bc_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
    bc_styles = ['-', '--', '-.', '-', '-', '-']

    for col, (gas_name, sp_label) in enumerate(MT_SPECIES):
        ax_inst = axes[0, col]
        ax_cum = axes[1, col]

        for ri, res in enumerate(all_results):
            if res is None:
                continue
            t_min = res['snap_times'] / 60.0
            flux = res['mt_data'][gas_name]
            # Cumulative: trapezoidal integration → µM
            cumul = np.zeros_like(flux)
            dt = np.diff(res['snap_times'])
            for i in range(1, len(flux)):
                cumul[i] = cumul[i-1] + 0.5 * (flux[i-1] + flux[i]) * dt[i-1]
            cumul_uM = cumul * 1e6  # M → µM

            color = bc_colors[ri % len(bc_colors)]
            ls = bc_styles[ri % len(bc_styles)]

            # Instantaneous (M/s)
            ax_inst.plot(t_min, flux, color=color, ls=ls, lw=1.5,
                         label=res['label'])
            # Cumulative (µM)
            ax_cum.plot(t_min, cumul_uM, color=color, ls=ls, lw=1.5,
                        label=res['label'])

        ax_inst.set_title(f'{sp_label}', fontweight='bold')
        ax_inst.yaxis.set_major_formatter(
            mticker.ScalarFormatter(useMathText=True))
        ax_inst.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_cum.set_xlabel('Time (min)')

        if col == 0:
            ax_inst.set_ylabel('MT flux (M/s)')
            ax_cum.set_ylabel('Cumulative MT (µM)')
            ax_inst.legend(loc='best', fontsize=7)

    # Row labels
    axes[0, 0].annotate('Instantaneous', xy=(0, 0.5),
                         xytext=(-0.35, 0.5),
                         xycoords='axes fraction',
                         textcoords='axes fraction',
                         fontsize=13, fontweight='bold',
                         rotation=90, va='center', ha='center')
    axes[1, 0].annotate('Cumulative', xy=(0, 0.5),
                         xytext=(-0.35, 0.5),
                         xycoords='axes fraction',
                         textcoords='axes fraction',
                         fontsize=13, fontweight='bold',
                         rotation=90, va='center', ha='center')

    fig.suptitle('Mass transfer flux by BC model (DIW, 12 min)',
                 fontsize=14, y=1.01)
    fig.tight_layout()

    out_png = _script_dir / 'fig1b_mt_flux.png'
    out_pdf = _script_dir / 'fig1b_mt_flux.pdf'
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"\n  → {out_png.name} / {out_pdf.name} saved")
    plt.close(fig)


CACHE_FILE = _script_dir / 'fig1b_mt_cache.npz'


def save_cache(all_results):
    """Save simulation results to npz cache."""
    data = {}
    for i, res in enumerate(all_results):
        if res is None:
            continue
        data[f'label_{i}'] = res['label']
        data[f'snap_times_{i}'] = res['snap_times']
        data[f'wall_{i}'] = res['wall']
        data[f'pH_{i}'] = res['pH']
        for gas_name, _ in MT_SPECIES:
            data[f'mt_{i}_{gas_name}'] = res['mt_data'][gas_name]
    data['n_results'] = len(all_results)
    np.savez(CACHE_FILE, **data)
    print(f"  Cache saved: {CACHE_FILE.name}")


def load_cache():
    """Load cached results. Returns list or None if cache missing."""
    if not CACHE_FILE.exists():
        return None
    d = np.load(CACHE_FILE, allow_pickle=True)
    n = int(d['n_results'])
    all_results = []
    for i in range(n):
        k = f'label_{i}'
        if k not in d:
            all_results.append(None)
            continue
        mt_data = {}
        for gas_name, _ in MT_SPECIES:
            mt_data[gas_name] = d[f'mt_{i}_{gas_name}']
        all_results.append({
            'label': str(d[f'label_{i}']),
            'snap_times': d[f'snap_times_{i}'],
            'mt_data': mt_data,
            'wall': float(d[f'wall_{i}']),
            'pH': float(d[f'pH_{i}']),
        })
    print(f"  Loaded cache: {CACHE_FILE.name}")
    return all_results


def main():
    os.chdir(_project_root)
    times, gas_conc = load_gas_data()

    print("=" * 60)
    print("Figure 1-2: MT flux time series by BC type")
    print("=" * 60)

    all_results = load_cache()
    if all_results is None:
        all_results = []
        for label, bc_type, ab in BC_CASES:
            res = run_and_collect_mt(times, gas_conc, bc_type, ab, label)
            all_results.append(res)
        save_cache(all_results)

    plot_mt_figure(all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"{'BC':20s} {'pH':>6s} {'wall(s)':>8s}  "
          + "  ".join(f'{sp[1]:>10s}' for sp in MT_SPECIES))
    for res in all_results:
        if res is None:
            continue
        cumuls = []
        for gas_name, _ in MT_SPECIES:
            flux = res['mt_data'][gas_name]
            dt = np.diff(res['snap_times'])
            c = sum(0.5 * (flux[i] + flux[i+1]) * dt[i]
                    for i in range(len(dt))) * 1e6
            cumuls.append(f'{c:10.1f}')
        print(f"{res['label']:20s} {res['pH']:6.3f} {res['wall']:8.0f}  "
              + "  ".join(cumuls))

    print("\nDone!")


if __name__ == '__main__':
    main()
