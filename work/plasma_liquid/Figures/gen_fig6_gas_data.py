#!/usr/bin/env python3
"""
Figure 6: Gas-phase species input data.

Row 1: Raw data (from CSV, no filtering)
Row 2: After onset filter + linear interpolation (measured species)
Row 3: Unmeasured species (HONO, HNO₃, H₂O₂) — constant values

Species (measured): O₃, NO₂, NO₃, N₂O₅
Species (unmeasured): HONO, HNO₃ (HONO2), H₂O₂
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import PHYSICAL
from pde_solver import _filter_onset

DEFAULT_GAS_XLSX = (
    _project_root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
)
DEFAULT_GAS_SHEET = '3.2kV'

MEASURED_SPECIES = [
    ('O3',   'O\u2083',          '#1f77b4'),
    ('NO2',  'NO\u2082',         '#ff7f0e'),
    ('NO3',  'NO\u2083',         '#2ca02c'),
    ('N2O5', 'N\u2082O\u2085',   '#d62728'),
]

# Unmeasured species: ratio-based (see notes/unmeasured_gas_species.md)
HONO_RATIO = 0.33       # HONO/NO₂
HONO2_RATIO = 0.83      # HNO₃/N₂O₅
H2O2_RATIO = 0.03       # H₂O₂/O₃

UNMEASURED_SPECS = [
    ('HONO',  'HONO',          'NO2',  HONO_RATIO,  '#9467bd'),
    ('HONO2', 'HNO\u2083',    'N2O5', HONO2_RATIO, '#8c564b'),
    ('H2O2',  'H\u2082O\u2082', 'O3', H2O2_RATIO,  '#e377c2'),
]


def main():
    df = pd.read_excel(DEFAULT_GAS_XLSX, sheet_name=DEFAULT_GAS_SHEET)
    times = df.iloc[:, 0].values.astype(float)
    t_min = times / 60.0
    conv = 1000.0 / PHYSICAL.AVOGADRO

    # Raw data (molecules/cm³)
    raw = {}
    for col, _, _ in MEASURED_SPECIES:
        if col in df.columns:
            raw[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            raw[col] = np.zeros(len(df))

    # Filtered data (onset filter, mol/L)
    filtered = {}
    for col, _, _ in MEASURED_SPECIES:
        filtered[col] = _filter_onset(raw[col] * conv)

    # Dense time for interpolation
    t_dense = np.linspace(0, times[-1], 2000)
    t_dense_min = t_dense / 60.0
    dt_gas = float(times[1] - times[0])
    n_times = len(times)

    def interp_at(arr, t):
        t_frac = t / dt_gas
        i0 = int(t_frac)
        if i0 >= n_times - 1:
            return arr[n_times - 1]
        if i0 < 0:
            return arr[0]
        frac = t_frac - i0
        return arr[i0] * (1.0 - frac) + arr[i0 + 1] * frac

    interped = {}
    for col, _, _ in MEASURED_SPECIES:
        interped[col] = np.array([interp_at(filtered[col], t) for t in t_dense])

    # Unmeasured: ratio-based time-varying arrays
    unmeas_raw = {}
    unmeas_interped = {}
    for col, _, ref_col, ratio, _ in UNMEASURED_SPECS:
        ref_raw = raw.get(ref_col, np.zeros(len(df)))
        unmeas_raw[col] = ref_raw * ratio  # molecules/cm³
        arr_molar = _filter_onset(ref_raw * ratio * conv)
        unmeas_interped[col] = np.array([interp_at(arr_molar, t) for t in t_dense])

    # Plot
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Row 1: Raw (molecules/cm³)
    ax = axes[0]
    for col, label, color in MEASURED_SPECIES:
        vals = raw[col]
        vals_plot = np.where(vals > 0, vals, np.nan)
        ax.plot(t_min, vals_plot, color=color, lw=1.2, label=label)
    ax.set_yscale('log')
    ax.set_ylabel('Concentration (cm\u207b\u00b3)')
    ax.set_title('(a) Raw measured data', fontweight='bold', loc='left')
    ax.legend(loc='right')
    ax.set_ylim(bottom=1e8)

    # Row 2: Linear interpolation (mol/L)
    ax = axes[1]
    for col, label, color in MEASURED_SPECIES:
        vals = interped[col]
        vals_plot = np.where(vals > 0, vals, np.nan)
        ax.plot(t_dense_min, vals_plot, color=color, lw=1.2, label=label)
    ax.set_yscale('log')
    ax.set_ylabel('Concentration (mol/L)')
    ax.set_title('(b) Measured species (onset-filtered + linear interpolation)',
                 fontweight='bold', loc='left')
    ax.legend(loc='right')
    ax.set_ylim(bottom=1e-12)

    # Row 3: Unmeasured (ratio-based, mol/L)
    ax = axes[2]
    for col, label, ref_col, ratio, color in UNMEASURED_SPECS:
        vals = unmeas_interped[col]
        vals_plot = np.where(vals > 0, vals, np.nan)
        ax.plot(t_dense_min, vals_plot, color=color, lw=1.2,
                label=f'{label} = {ref_col} \u00d7 {ratio}')
    ax.set_yscale('log')
    ax.set_ylabel('Concentration (mol/L)')
    ax.set_xlabel('Time (min)')
    ax.set_title('(c) Unmeasured species (ratio-based estimate)',
                 fontweight='bold', loc='left')
    ax.legend(loc='right', fontsize=8)
    ax.set_ylim(bottom=1e-12)

    fig.suptitle('Gas-phase input data (DIW, 3.2 kVpp, 12 min)',
                 fontsize=14, y=1.01)
    fig.tight_layout()

    out_png = _script_dir / 'fig6_gas_data.png'
    out_pdf = _script_dir / 'fig6_gas_data.pdf'
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  -> {out_png.name} / {out_pdf.name} saved")
    plt.close(fig)


if __name__ == '__main__':
    main()
