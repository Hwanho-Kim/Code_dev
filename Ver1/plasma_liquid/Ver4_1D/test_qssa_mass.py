#!/usr/bin/env python
"""Diagnostic: verify QSSA mass conservation for Cl atoms in a single cell.

Tests:
1. Single RHS call: total Cl atom rate should be ~0
2. BDF integration (1s, 10s, 60s): Cl⁻ should remain stable
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp

from config_1d import DEFAULTS, ODE_CONFIG, AQUEOUS_SPECIES, SALINE_SPECIES
from chemistry_1d import AqueousChemistry1D

# Build chemistry
chem = AqueousChemistry1D(saline_mode=True)
idx = chem.species_idx
N_s = chem.n_species
trace = DEFAULTS.trace_concentration

# --- Initial condition: saline pH 7, [Cl⁻]=0.154 M ---
y0 = np.full(N_s, trace)
y0[idx['H+']] = 1e-7
y0[idx['Cl-']] = 0.154
if 'Na+' in idx:
    y0[idx['Na+']] = 0.154
# Small radical concentrations (simulating interface cell)
y0[idx['OH']] = 1e-12
y0[idx['O3']] = 1e-8
if 'Cl' in idx:
    y0[idx['Cl']] = 1e-15
if 'H' in idx:
    y0[idx['H']] = 1e-15

# Apply initial QSSA
chem.apply_qssa(y0)
print(f"Initial QSSA: HOCl⁻ = {y0[idx['HOCl-']]:.3e} M, Cl₂⁻ = {y0[idx['Cl2-']]:.3e} M")

# --- Cl atom count function ---
# Species with Cl atoms: Cl⁻(1), HOCl⁻(1), Cl₂⁻(2), Cl(1), Cl₂(2),
#   Cl₃⁻(3), HClO_total(1), HClO2_total(1), ClO(1), ClO₂(1), etc.
cl_species = {
    'Cl-': 1, 'HOCl-': 1, 'Cl2-': 2, 'Cl': 1, 'Cl2': 2,
    'Cl3-': 3, 'HClO_total': 1, 'HClO2_total': 1,
    'ClO': 1, 'ClO2': 1, 'ClO3': 1,
    'Cl2O': 2, 'Cl2O2': 2, 'Cl2O3': 2, 'Cl2O4': 2, 'Cl2O5': 2, 'Cl2O6': 2,
    'ClNO2': 1, 'ClO3-': 1, 'ClO4-': 1, 'HOClH': 1,
    'HCl': 1,
}

def total_cl(y):
    """Total Cl atoms [M]"""
    total = 0.0
    for sp, n_cl in cl_species.items():
        if sp in idx:
            total += n_cl * y[idx[sp]]
    return total

def cl_rate(dydt):
    """Total Cl atom rate [M/s]"""
    rate = 0.0
    for sp, n_cl in cl_species.items():
        if sp in idx:
            rate += n_cl * dydt[idx[sp]]
    return rate

# === Test 1: Single RHS call ===
print("\n=== Test 1: Single RHS call ===")
dydt = chem.compute_rates_numba(y0.copy())
total_cl_rate = cl_rate(dydt)
print(f"Total Cl atom rate: {total_cl_rate:.6e} M/s")
print(f"dydt[HOCl⁻] = {dydt[idx['HOCl-']]:.6e}")
print(f"dydt[Cl₂⁻]  = {dydt[idx['Cl2-']]:.6e}")
print(f"dydt[Cl⁻]   = {dydt[idx['Cl-']]:.6e}")

# Check a few key rates
for sp in ['OH', 'Cl', 'O3', 'HClO_total', 'Cl2', 'Cl3-', 'HOClH']:
    if sp in idx:
        print(f"  dydt[{sp:12s}] = {dydt[idx[sp]]:.6e}")

# === Test 2: BDF integration ===
print("\n=== Test 2: BDF integration (mass conservation) ===")

def rhs(t, y):
    yc = np.maximum(y, trace)
    return chem.compute_rates_numba(yc)

cl0 = total_cl(y0)
print(f"Initial total Cl: {cl0*1e6:.3f} µM")

for dt in [1.0, 10.0, 60.0]:
    y_init = y0.copy()
    sol = solve_ivp(rhs, [0, dt], y_init, method='BDF',
                    rtol=1e-4, atol=1e-8, max_step=dt)
    if sol.success:
        yf = sol.y[:, -1]
        cl_f = total_cl(yf)
        cl_change = (cl_f - cl0) * 1e6  # µM
        clm_change = (yf[idx['Cl-']] - y0[idx['Cl-']]) * 1e6
        print(f"  dt={dt:5.1f}s: nfev={sol.nfev:5d}, "
              f"Cl⁻ change={clm_change:+.6f} µM, "
              f"total Cl change={cl_change:+.6f} µM, "
              f"pH={-np.log10(max(yf[idx['H+']],1e-14)):.4f}")
    else:
        print(f"  dt={dt:5.1f}s: FAILED — {sol.message}")

# === Test 3: Elevated radical scenario ===
print("\n=== Test 3: Elevated radicals (interface-like) ===")
y_high = y0.copy()
y_high[idx['OH']] = 1e-9   # higher OH
y_high[idx['O3']] = 1e-6
if 'Cl' in idx:
    y_high[idx['Cl']] = 1e-12
chem.apply_qssa(y_high)
print(f"QSSA: HOCl⁻ = {y_high[idx['HOCl-']]:.3e}, Cl₂⁻ = {y_high[idx['Cl2-']]:.3e}")

dydt_h = chem.compute_rates_numba(y_high.copy())
print(f"Cl atom rate: {cl_rate(dydt_h):.6e} M/s")
print(f"dydt[HOCl⁻] = {dydt_h[idx['HOCl-']]:.6e}")
print(f"dydt[Cl₂⁻]  = {dydt_h[idx['Cl2-']]:.6e}")
print(f"dydt[Cl⁻]   = {dydt_h[idx['Cl-']]:.6e}")

cl0_h = total_cl(y_high)
for dt in [1.0, 10.0, 60.0]:
    y_init = y_high.copy()
    sol = solve_ivp(rhs, [0, dt], y_init, method='BDF',
                    rtol=1e-4, atol=1e-8, max_step=dt)
    if sol.success:
        yf = sol.y[:, -1]
        cl_f = total_cl(yf)
        cl_change = (cl_f - cl0_h) * 1e6
        clm_change = (yf[idx['Cl-']] - y_high[idx['Cl-']]) * 1e6
        print(f"  dt={dt:5.1f}s: nfev={sol.nfev:5d}, "
              f"Cl⁻ change={clm_change:+.6f} µM, "
              f"total Cl change={cl_change:+.6f} µM, "
              f"pH={-np.log10(max(yf[idx['H+']],1e-14)):.4f}")
    else:
        print(f"  dt={dt:5.1f}s: FAILED — {sol.message}")

print("\nDone.")
