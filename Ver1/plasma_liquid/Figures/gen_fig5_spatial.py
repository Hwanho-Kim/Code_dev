#!/usr/bin/env python3
"""
Figure 5: Spatial concentration profiles at multiple time snapshots.

Runs DIW Film+α_b simulation (monolithic BDF), extracts depth profiles
at selected time points, and generates multi-panel figure.
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

from config_1d import PHYSICAL, N2O4_EQ
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

# ── Configuration ──
ALPHA_B = 0.03
SNAP_TIMES_MIN = [1, 2, 4, 6, 8, 12]   # time snapshots (minutes)
DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

# Species to plot: (internal_name, display_label, is_total_var, unit, scale)
# scale: multiply concentration by this to get the displayed unit
SPECIES_PANELS = [
    ('HONO2_total', 'NO₃⁻',  True,  'µM',  1e6),
    ('O3',          'O₃',     False, 'µM',  1e6),
    ('H2O2_total',  'H₂O₂',  True,  'nM',  1e9),
    ('OH',          'OH',     False, 'pM',  1e12),
    ('HO2_total',   'HO₂',   True,  'pM',  1e12),
]

# pH panel handled separately (from H+)


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


def run_simulation():
    """Run DIW simulation, return snapshots at requested times."""
    times, gas_conc = load_gas_data()
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='film_alpha',
        alpha_b=ALPHA_B,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=0, hono2_gas=0, h2o2_gas=0,
    )
    t_end = float(times[-1])

    # t_eval at requested snapshot times
    t_eval = np.array([t * 60.0 for t in SNAP_TIMES_MIN])
    t_eval = t_eval[t_eval <= t_end]

    print(f"Running DIW (monolithic BDF): α_b={ALPHA_B}, "
          f"dt_enforce=None, t_end={t_end}s")
    print(f"Snapshots at: {[f'{t/60:.0f}min' for t in t_eval]}")
    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval,
        verbose=True, dt_poisson=None,
    )
    print(f"Simulation: {time_mod.time()-t0:.1f}s")

    return result, solver, t_eval


def plot_fig5(result, solver, t_eval):
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

    chem = solver.chem
    N_z, N_s = solver.N_z, solver.N_s

    # Cell center depths (mm)
    z_mm = solver.z_centers * 1e3  # m → mm

    # Color map for time snapshots
    n_snaps = len(t_eval)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_snaps - 1, 1)) for i in range(n_snaps)]

    n_panels = len(SPECIES_PANELS) + 1  # +1 for pH
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows),
                             sharex=True)

    # ── pH panel ──
    ax_pH = axes.flat[0]
    h_idx = chem.species_idx['H+']
    for si, tv in enumerate(t_eval):
        y2d = result['y_eval'][si].reshape(N_z, N_s)
        h_conc = np.clip(y2d[:, h_idx], 1e-14, None)
        pH_profile = -np.log10(h_conc)
        ax_pH.plot(z_mm, pH_profile, color=colors[si], lw=1.5,
                   label=f'{tv/60:.0f} min')
    ax_pH.set_ylabel('pH')
    ax_pH.set_title('(a) pH', fontweight='bold', loc='left')
    ax_pH.legend(loc='best', fontsize=7)

    # ── Species panels (log scale) ──
    panel_labels = 'bcdefghij'
    for pi, (sp_name, sp_label, is_total, unit, scale) in enumerate(SPECIES_PANELS):
        ax = axes.flat[pi + 1]
        idx = chem.species_idx.get(sp_name)
        if idx is None:
            ax.set_visible(False)
            continue

        for si, tv in enumerate(t_eval):
            y2d = result['y_eval'][si].reshape(N_z, N_s)
            profile = np.clip(y2d[:, idx], 1e-30, None) * scale
            ax.plot(z_mm, profile, color=colors[si], lw=1.5,
                    label=f'{tv/60:.0f} min')

        ax.set_yscale('log')
        ax.set_ylabel(f'{sp_label} ({unit})')
        ax.set_title(f'({panel_labels[pi]}) {sp_label}', fontweight='bold',
                     loc='left')
        if pi == 0:
            ax.legend(loc='best', fontsize=7)

    # Hide unused axes
    for i in range(n_panels, len(axes.flat)):
        axes.flat[i].set_visible(False)

    # x-axis label on bottom row
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel('Depth (mm)')
            ax.set_xlim(0, z_mm[-1])

    fig.suptitle(
        f'Spatial concentration profiles (DIW, Film+αb, αb={ALPHA_B})',
        fontsize=14, y=1.01)
    fig.tight_layout()

    out_png = _script_dir / 'fig5_spatial.png'
    out_pdf = _script_dir / 'fig5_spatial.pdf'
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  → {out_png.name} / {out_pdf.name} saved")
    plt.close(fig)


def main():
    os.chdir(_project_root)
    print("=" * 60)
    print(f"Figure 5: Spatial profiles (α_b={ALPHA_B})")
    print("=" * 60)

    result, solver, t_eval = run_simulation()
    plot_fig5(result, solver, t_eval)
    print("\nDone!")


if __name__ == '__main__':
    main()
