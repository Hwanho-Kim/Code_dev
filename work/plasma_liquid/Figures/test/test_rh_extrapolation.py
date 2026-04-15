#!/usr/bin/env python3
"""
RH extrapolation: predict RH=80% gas-phase concentrations from Dry/25/55/65% data.

Method C: Steady-state extrapolation.
  - Extract steady-state values (t=500-600s avg) at each RH
  - Fit vs RH with quadratic (clipped to ≥0)
  - Extrapolate to 80%

Output: fig_rh_extrapolation.png — one subplot per species, 3 voltages as lines

Run:
    .venv/bin/python Figures/test/test_rh_extrapolation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent

DRY_XLSX = _project_root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
HUMID_DIR = _project_root / 'OAS data' / 'Humid'

VOLTAGES = [2.6, 3.2, 3.6]
RH_MEASURED = [0, 55, 65]  # RH 25% excluded (경향 이탈)
RH_EXTRAP = 80
RH_PLOT = np.linspace(0, 90, 200)

SPECIES = ['O3', 'NO2', 'NO3', 'N2O5', 'HONO']
SPECIES_LABELS = {
    'O3': 'O\u2083', 'NO2': 'NO\u2082', 'NO3': 'NO\u2083',
    'N2O5': 'N\u2082O\u2085', 'HONO': 'HONO',
}

VOLTAGE_COLORS = {2.6: '#1f77b4', 3.2: '#ff7f0e', 3.6: '#2ca02c'}
n_total = 2.46e19  # molecules/cm³ at 1 atm 298K


def load_steady_state(voltage, rh):
    """Load time series and return steady-state average (last 100s)."""
    if rh == 0:
        sheet = f'{voltage}kV'
        df = pd.read_excel(DRY_XLSX, sheet_name=sheet)
    else:
        fname = HUMID_DIR / f'{rh}_{voltage}.csv'
        df = pd.read_csv(fname)

    # Drop trailing NaN rows
    df = df.dropna(subset=[df.columns[0]])
    times = df.iloc[:, 0].values.astype(float)

    t_end = times[-1]
    mask = times >= (t_end - 100)  # last 100s average

    result = {}
    for sp in SPECIES:
        if sp in df.columns:
            vals = pd.to_numeric(df[sp], errors='coerce').values.copy()
            vals = np.nan_to_num(vals, nan=0.0)
            result[sp] = max(np.mean(vals[mask]), 0.0)
        else:
            result[sp] = 0.0
    return result


def _fit_exp_decay(rh, c):
    """Fit a*exp(-b*RH) + c with constraints b>0, c>=0."""
    from scipy.optimize import curve_fit

    def model(x, a, b, c):
        return a * np.exp(-b * x) + c

    try:
        p0 = [c[0], 0.03, c[-1] * 0.5]
        popt, _ = curve_fit(model, rh, c, p0=p0,
                            bounds=([0, 0, 0], [np.inf, 1.0, np.inf]),
                            maxfev=5000)
        return lambda x: np.maximum(model(np.asarray(x, float), *popt), 0.0), popt
    except Exception:
        return None, None


def _fit_linear(rh, c):
    """Simple linear fit, clipped ≥0."""
    coeffs = P.polyfit(rh, c, 1)
    return lambda x: np.maximum(P.polyval(np.asarray(x, float), coeffs), 0.0), coeffs


def _fit_linear_through_zero(rh, c):
    """Linear fit forced through origin (for HONO: 0 at RH=0)."""
    # y = a*x → a = sum(x*y)/sum(x^2)
    rh_arr = np.asarray(rh, float)
    c_arr = np.asarray(c, float)
    a = np.sum(rh_arr * c_arr) / np.sum(rh_arr**2) if np.sum(rh_arr**2) > 0 else 0
    return lambda x: np.maximum(a * np.asarray(x, float), 0.0), a


# Species → fitting strategy
FIT_STRATEGY = {
    'O3':   'exp_decay',      # decreasing, physically: O3 + H2O
    'N2O5': 'exp_decay',      # sharp decrease: N2O5 + H2O → 2HNO3
    'NO2':  'linear',         # monotonic increase
    'NO3':  'linear',         # irregular, keep simple
    'HONO': 'linear_zero',    # starts at 0, increases with RH
}


def fit_and_extrapolate(rh_vals, conc_vals, rh_target, rh_plot, species=''):
    """Species-specific fitting and extrapolation."""
    rh_arr = np.array(rh_vals, dtype=float)
    c_arr = np.array(conc_vals, dtype=float)

    valid = ~np.isnan(c_arr)
    rh_arr = rh_arr[valid]
    c_arr = c_arr[valid]

    if len(rh_arr) < 2:
        return 0.0, np.zeros_like(rh_plot), 'none'

    # Use highest degree possible: n_points - 1, max 2
    deg = min(2, len(rh_arr) - 1)
    coeffs = P.polyfit(rh_arr, c_arr, deg)
    fit_label = f'poly{deg}'

    y_plot = np.maximum(P.polyval(rh_plot, coeffs), 0.0)
    y_target = max(float(P.polyval(rh_target, coeffs)), 0.0)

    return y_target, y_plot, fit_label


def main():
    print("=" * 60)
    print("RH extrapolation: Dry/25/55/65% → 80%")
    print("=" * 60)

    # Load all steady-state data
    data = {}  # data[voltage][rh][species] = value
    for v in VOLTAGES:
        data[v] = {}
        for rh in RH_MEASURED:
            data[v][rh] = load_steady_state(v, rh)

    # Plot
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 11, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()

    # Summary table
    print(f"\n{'Species':>8s} {'Voltage':>8s}  ", end='')
    for rh in RH_MEASURED:
        print(f"{'RH'+str(rh)+'%':>12s}", end='')
    print(f"  {'RH80%(ext)':>12s}  {'ppm@80%':>8s}")
    print("-" * 85)

    for si, sp in enumerate(SPECIES):
        ax = axes_flat[si]

        for v in VOLTAGES:
            rh_vals = RH_MEASURED
            conc_vals = [data[v][rh][sp] for rh in rh_vals]

            y_target, y_plot, fit_info = fit_and_extrapolate(
                rh_vals, conc_vals, RH_EXTRAP, RH_PLOT, species=sp)

            color = VOLTAGE_COLORS[v]

            # Data points
            ax.scatter(rh_vals, conc_vals, color=color, s=60, zorder=5,
                       edgecolors='black', lw=0.8)
            # Fit curve
            ax.plot(RH_PLOT, y_plot, color=color, lw=1.5, ls='--',
                    label=f'{v}kV')
            # Extrapolated point
            ax.scatter([RH_EXTRAP], [y_target], color=color, s=120,
                       marker='*', zorder=6, edgecolors='black', lw=0.8)

            ppm = y_target / n_total * 1e6
            print(f"{sp:>8s} {v:>7.1f}kV  ", end='')
            for rh in rh_vals:
                print(f"{data[v][rh][sp]:12.2e}", end='')
            print(f"  {y_target:12.2e}  {ppm:8.1f}")

        # Vertical line at RH=80%
        ax.axvline(x=RH_EXTRAP, color='red', ls=':', lw=1, alpha=0.7)
        ax.text(RH_EXTRAP + 1, ax.get_ylim()[1] * 0.9, 'RH=80%',
                color='red', fontsize=8, va='top')

        ax.set_xlabel('RH (%)')
        ax.set_ylabel(f'{SPECIES_LABELS[sp]} (cm⁻³)')
        ax.set_title(f'{SPECIES_LABELS[sp]}', fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.set_xlim(-5, 95)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Hide unused subplot
    if len(SPECIES) < len(axes_flat):
        for j in range(len(SPECIES), len(axes_flat)):
            axes_flat[j].set_visible(False)

    fig.suptitle('Gas-phase species vs RH — Steady-state extrapolation to 80%\n'
                 '(dots=measured, dashed=fit, star=extrapolated)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig_rh_extrapolation.png')
    fig.savefig(_script_dir / 'fig_rh_extrapolation.pdf')
    print(f"\n  -> fig_rh_extrapolation.png/pdf saved")
    plt.close(fig)


if __name__ == '__main__':
    main()
