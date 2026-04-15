#!/usr/bin/env python3
"""
RH extrapolation via physically-motivated ratio fitting.

N₂O₅/NO₂ = A / (1 + B×RH²)   — dimer hydrolysis steady-state
HONO/NO₂ = A × RH              — surface H₂O ∝ RH
NO₂/O₃   = A + B×RH            — empirical mode transition

Data: Dry(0%), 55%, 65% (RH 25% excluded)

Run:
    .venv/bin/python Figures/test/test_rh_ratio_fit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent

DRY_XLSX = _project_root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
HUMID_DIR = _project_root / 'OAS data' / 'Humid'

VOLTAGES = [2.6, 3.2, 3.6]
RH_DATA = [0, 55, 65]  # RH 25% excluded
RH_EXTRAP = 80
RH_PLOT = np.linspace(0, 90, 300)

VOLTAGE_COLORS = {2.6: '#1f77b4', 3.2: '#ff7f0e', 3.6: '#2ca02c'}
VOLTAGE_MARKERS = {2.6: 'o', 3.2: 's', 3.6: '^'}
n_total = 2.46e19


def load_ss(voltage, rh):
    if rh == 0:
        df = pd.read_excel(DRY_XLSX, sheet_name=f'{voltage}kV')
    else:
        df = pd.read_csv(HUMID_DIR / f'{rh}_{voltage}.csv')
        df = df.dropna(subset=[df.columns[0]])
    times = df.iloc[:, 0].values.astype(float)
    t_end = times[-1]
    mask = times >= (t_end - 100)
    result = {}
    for sp in ['O3', 'NO2', 'NO3', 'N2O5', 'HONO']:
        if sp in df.columns:
            vals = pd.to_numeric(df[sp], errors='coerce').values.copy()
            vals = np.nan_to_num(vals, nan=0.0)
            result[sp] = max(np.mean(vals[mask]), 0.0)
        else:
            result[sp] = 0.0
    return result


# ── Fitting functions ──

def n2o5_no2_model(rh, A, B):
    """N₂O₅/NO₂ = A / (1 + B×RH²), physically: dimer hydrolysis."""
    return A / (1.0 + B * rh**2)

def hono_no2_model(rh, A):
    """HONO/NO₂ = A × RH, physically: surface [H₂O] ∝ RH."""
    return A * rh

def no2_o3_model(rh, A, B):
    """NO₂/O₃ = A + B×RH, empirical mode transition."""
    return A + B * rh


def main():
    # Load data
    data = {}
    for v in VOLTAGES:
        data[v] = {}
        for rh in RH_DATA:
            data[v][rh] = load_ss(v, rh)

    # Compute ratios
    ratios_config = [
        ('N₂O₅/NO₂', 'N2O5', 'NO2', n2o5_no2_model,
         'A/(1+B·RH²)', [1.0, 1e-3], ([0, 0], [np.inf, np.inf])),
        ('HONO/NO₂', 'HONO', 'NO2', hono_no2_model,
         'A·RH', [1e-4], ([0], [np.inf])),
        ('NO₂/O₃', 'NO2', 'O3', no2_o3_model,
         'A+B·RH', [0.01, 1e-3], ([-np.inf, 0], [np.inf, np.inf])),
        ('NO₃/O₃', 'NO3', 'O3', no2_o3_model,
         'A+B·RH', [0.003, 1e-4], ([-np.inf, -np.inf], [np.inf, np.inf])),
    ]

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    print("=" * 80)
    print("Ratio-based RH extrapolation (physically motivated)")
    print("=" * 80)

    for i, (ratio_label, num_sp, den_sp, model_func,
            formula, p0, bounds) in enumerate(ratios_config):
        ax = axes[i]

        print(f"\n--- {ratio_label}: {formula} ---")
        print(f"{'Voltage':>8s}  {'Params':>30s}  {'RH80% ratio':>12s}  "
              f"{'[{num_sp}]@80%':>12s}  {'ppm':>8s}")

        # Collect all voltage data for combined fit + per-voltage fit
        for v in VOLTAGES:
            rh_arr = np.array(RH_DATA, dtype=float)
            ratio_arr = np.array([
                data[v][rh][num_sp] / data[v][rh][den_sp]
                if data[v][rh][den_sp] > 0 else 0.0
                for rh in RH_DATA
            ])

            # Filter out zero ratios for fitting
            valid = ratio_arr > 0
            if valid.sum() < len(p0):
                # Not enough points, use available
                rh_fit = rh_arr
                r_fit = ratio_arr
            else:
                rh_fit = rh_arr[valid]
                r_fit = ratio_arr[valid]

            color = VOLTAGE_COLORS[v]
            marker = VOLTAGE_MARKERS[v]

            # Plot data points
            ax.scatter(rh_arr, ratio_arr, color=color, marker=marker,
                       s=80, zorder=5, edgecolors='black', lw=0.8,
                       label=f'{v}kV')

            # Fit
            try:
                popt, _ = curve_fit(model_func, rh_fit, r_fit, p0=p0,
                                    bounds=bounds, maxfev=5000)
                y_plot = model_func(RH_PLOT, *popt)
                y_80 = model_func(RH_EXTRAP, *popt)

                ax.plot(RH_PLOT, np.maximum(y_plot, 0), color=color,
                        ls='--', lw=1.5)
                ax.scatter([RH_EXTRAP], [y_80], color=color, marker='*',
                           s=150, zorder=6, edgecolors='black', lw=0.8)

                # Back-calculate absolute concentration at RH 80%
                # Use denominator species extrapolated linearly
                den_vals = [data[v][rh][den_sp] for rh in RH_DATA]
                den_fit = np.polyfit(RH_DATA, den_vals, 1)
                den_80 = max(np.polyval(den_fit, RH_EXTRAP), 0)
                num_80 = y_80 * den_80
                ppm_80 = num_80 / n_total * 1e6

                param_str = ', '.join(f'{p:.4e}' for p in popt)
                print(f"{v:>7.1f}kV  {param_str:>30s}  {y_80:12.6f}  "
                      f"{num_80:12.2e}  {ppm_80:8.1f}")

            except Exception as e:
                print(f"{v:>7.1f}kV  FAILED: {e}")

        ax.axvline(x=RH_EXTRAP, color='red', ls=':', lw=1, alpha=0.7)
        ax.set_xlabel('RH (%)')
        ax.set_ylabel(ratio_label)
        ax.set_title(f'{ratio_label}\n{formula}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(-5, 95)

    # ── O₃ direct extrapolation (anchor species) ──
    ax_o3 = axes[len(ratios_config)]
    print(f"\n--- O₃ direct (linear fit) ---")
    print(f"{'Voltage':>8s}  {'slope':>12s}  {'intercept':>12s}  {'O3@80%':>12s}  {'ppm':>8s}")

    o3_at_80 = {}
    for v in VOLTAGES:
        rh_arr = np.array(RH_DATA, dtype=float)
        o3_arr = np.array([data[v][rh]['O3'] for rh in RH_DATA])
        color = VOLTAGE_COLORS[v]
        marker = VOLTAGE_MARKERS[v]

        ax_o3.scatter(rh_arr, o3_arr, color=color, marker=marker,
                      s=80, zorder=5, edgecolors='black', lw=0.8,
                      label=f'{v}kV')

        coeffs = np.polyfit(rh_arr, o3_arr, 1)
        y_plot = np.maximum(np.polyval(coeffs, RH_PLOT), 0)
        y_80 = max(np.polyval(coeffs, RH_EXTRAP), 0)
        o3_at_80[v] = y_80

        ax_o3.plot(RH_PLOT, y_plot, color=color, ls='--', lw=1.5)
        ax_o3.scatter([RH_EXTRAP], [y_80], color=color, marker='*',
                      s=150, zorder=6, edgecolors='black', lw=0.8)

        ppm = y_80 / n_total * 1e6
        print(f"{v:>7.1f}kV  {coeffs[0]:12.2e}  {coeffs[1]:12.2e}  {y_80:12.2e}  {ppm:8.1f}")

    ax_o3.axvline(x=RH_EXTRAP, color='red', ls=':', lw=1, alpha=0.7)
    ax_o3.set_xlabel('RH (%)')
    ax_o3.set_ylabel('O₃ (cm⁻³)')
    ax_o3.set_title('O₃ (anchor)\nLinear fit', fontweight='bold')
    ax_o3.legend(fontsize=9)
    ax_o3.set_xlim(-5, 95)
    ax_o3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Hide unused axes
    for j in range(len(ratios_config) + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Species ratios vs RH — physically motivated fitting\n'
                 '(dots=measured, dashed=fit, star=RH80% extrapolated)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig_rh_ratio_fit.png')
    fig.savefig(_script_dir / 'fig_rh_ratio_fit.pdf')
    print(f"\n  -> fig_rh_ratio_fit.png/pdf saved")
    plt.close(fig)

    # ── Summary: all predicted RH 80% concentrations ──
    print("\n" + "=" * 80)
    print("Predicted RH 80% concentrations")
    print("=" * 80)
    print(f"{'Voltage':>8s}  {'O3':>12s}  {'NO2':>12s}  {'N2O5':>12s}  {'HONO':>12s}  {'NO3':>12s}")
    print("-" * 68)


if __name__ == '__main__':
    main()
