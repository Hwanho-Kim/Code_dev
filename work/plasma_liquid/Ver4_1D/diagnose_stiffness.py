#!/usr/bin/env python
"""
Stiffness diagnosis: Jacobian eigenvalue analysis for Saline vs DI water.

Computes the numerical Jacobian of the chemistry RHS at representative
cell concentrations, then analyzes eigenvalues to identify:
  1. Which eigenvalues create stiffness
  2. Which species/reactions contribute to the fastest modes
  3. Why DI water doesn't have the same problem
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from config_1d import DEFAULTS, ODE_CONFIG, AQUEOUS_SPECIES, SALINE_SPECIES
from chemistry_1d import AqueousChemistry1D

np.set_printoptions(precision=3, linewidth=120)


def numerical_jacobian(rhs_func, y0, eps_rel=1e-6, eps_abs=1e-14):
    """Compute dense numerical Jacobian by column perturbation."""
    N = len(y0)
    f0 = rhs_func(y0)
    J = np.zeros((N, N))
    for j in range(N):
        h = max(eps_rel * abs(y0[j]), eps_abs)
        y_pert = y0.copy()
        y_pert[j] += h
        f_pert = rhs_func(y_pert)
        J[:, j] = (f_pert - f0) / h
    return J


def analyze_eigenvalues(J, species_names, label, top_n=10):
    """Eigenvalue analysis of Jacobian."""
    eigvals, eigvecs = np.linalg.eig(J)

    # Sort by real part (most negative = fastest decay)
    order = np.argsort(np.real(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    real_parts = np.real(eigvals)
    nonzero = np.abs(real_parts) > 1e-20
    if np.any(nonzero):
        lambda_min = real_parts[nonzero][0]    # most negative
        lambda_max = real_parts[nonzero][-1]   # least negative (or most positive)
        stiffness_ratio = abs(lambda_min / lambda_max) if abs(lambda_max) > 1e-20 else float('inf')
    else:
        lambda_min = lambda_max = 0
        stiffness_ratio = 1

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Eigenvalue range: [{lambda_min:.3e}, {lambda_max:.3e}]")
    print(f"  Fastest timescale: τ_min = {abs(1/lambda_min):.3e} s" if abs(lambda_min) > 1e-20 else "  No fast mode")
    print(f"  Slowest timescale: τ_max = {abs(1/lambda_max):.3e} s" if abs(lambda_max) > 1e-20 else "  No slow mode")
    print(f"  Stiffness ratio: {stiffness_ratio:.3e}")

    # Top N fastest modes
    print(f"\n  --- Top {top_n} fastest modes (most negative eigenvalues) ---")
    for i in range(min(top_n, len(eigvals))):
        ev = eigvals[i]
        vec = eigvecs[:, i]
        # Find dominant species in this eigenvector
        mag = np.abs(vec)
        top_idx = np.argsort(mag)[::-1][:5]
        tau = abs(1.0 / np.real(ev)) if abs(np.real(ev)) > 1e-20 else float('inf')
        contributors = [(species_names[k], mag[k]) for k in top_idx if mag[k] > 0.01]
        contrib_str = ", ".join(f"{name}({w:.2f})" for name, w in contributors[:4])
        print(f"  λ={np.real(ev):+12.3e} {'+ ' + str(np.imag(ev)) + 'j' if abs(np.imag(ev)) > 1e-10 else '':>20s}"
              f"  τ={tau:.2e}s  [{contrib_str}]")

    # Positive eigenvalues (unstable modes)
    pos = real_parts > 1e-10
    if np.any(pos):
        print(f"\n  ⚠ {np.sum(pos)} POSITIVE eigenvalues (unstable!):")
        for i in np.where(pos)[0]:
            ev = eigvals[i]
            vec = eigvecs[:, i]
            mag = np.abs(vec)
            top_idx = np.argsort(mag)[::-1][:3]
            contrib_str = ", ".join(f"{species_names[k]}({mag[k]:.2f})" for k in top_idx)
            print(f"    λ={np.real(ev):+.3e}  [{contrib_str}]")

    return eigvals, eigvecs, stiffness_ratio


def per_reaction_stiffness(chem, y0, species_names, label):
    """Identify which REACTIONS contribute most to stiffness.

    For each reaction, compute its individual Jacobian contribution
    and find the maximum |eigenvalue|.
    """
    N = chem.n_species
    trace = chem.trace

    y_clean = np.maximum(y0.copy(), trace)
    H_idx = chem.species_idx['H+']
    y_clean[H_idx] = max(y_clean[H_idx], 1e-14)

    speciated = chem.speciate(y_clean)

    print(f"\n  --- Per-reaction stiffness analysis ({label}) ---")
    print(f"  {'Reaction':>10s}  {'max|λ|':>12s}  {'τ_min':>10s}  {'rate':>12s}  Species involved")

    results = []
    for ri, rxn in enumerate(chem.reactions):
        rxn_d = chem._rxn_data[ri]

        # Compute this reaction's rate
        rate = chem._compute_single_rate(rxn_d, y_clean, speciated)

        if abs(rate) < 1e-30:
            continue

        # Compute this reaction's Jacobian contribution by finite diff
        def rhs_single_rxn(y):
            yc = np.maximum(y, trace)
            yc[H_idx] = max(yc[H_idx], 1e-14)
            sp = chem.speciate(yc)
            r = chem._compute_single_rate(rxn_d, yc, sp)
            dydt = np.zeros(N)
            if abs(r) > 1e-30:
                chem._apply_rate(rxn_d, r, dydt)
            return dydt

        J_rxn = numerical_jacobian(rhs_single_rxn, y_clean)
        eigvals_rxn = np.linalg.eigvals(J_rxn)
        max_abs_eig = np.max(np.abs(eigvals_rxn))

        if max_abs_eig > 1e-5:
            tau = 1.0 / max_abs_eig if max_abs_eig > 0 else float('inf')
            # Which species are involved?
            reactants = [sp for sp, _, _ in rxn_d.get('reactants', [])]
            products = [sp for sp, _, _ in rxn_d.get('products', [])]
            species_str = " + ".join(reactants) + " → " + " + ".join(products)
            label_str = rxn.get('label', f'R{ri}')[:10]
            results.append((max_abs_eig, tau, abs(rate), label_str, species_str))

    results.sort(key=lambda x: -x[0])
    for max_eig, tau, rate, lbl, sp_str in results[:20]:
        print(f"  {lbl:>10s}  {max_eig:12.3e}  {tau:10.2e}  {rate:12.3e}  {sp_str[:60]}")


# =============================================================================
# Setup
# =============================================================================
print("Loading chemistry systems...")
chem_sal = AqueousChemistry1D(saline_mode=True)
chem_diw = AqueousChemistry1D(saline_mode=False)

sal_idx = chem_sal.species_idx
diw_idx = chem_diw.species_idx
sal_names = chem_sal.aqueous_species
diw_names = chem_diw.aqueous_species

trace = DEFAULTS.trace_concentration

# =============================================================================
# Scenario 1: Initial state (minimal radicals)
# =============================================================================
print("\n" + "#"*70)
print("# SCENARIO 1: Initial state (minimal radicals, pH 7)")
print("#"*70)

# DI water
y_diw = np.full(chem_diw.n_species, trace)
y_diw[diw_idx['H+']] = 1e-7  # pH 7

# Saline
y_sal = np.full(chem_sal.n_species, trace)
y_sal[sal_idx['H+']] = 1e-7
y_sal[sal_idx['Cl-']] = 0.154

def rhs_diw(y):
    return chem_diw.compute_rates_numba(np.maximum(y, trace))
def rhs_sal(y):
    return chem_sal.compute_rates_numba(np.maximum(y, trace))

J_diw = numerical_jacobian(rhs_diw, y_diw)
J_sal = numerical_jacobian(rhs_sal, y_sal)

analyze_eigenvalues(J_diw, diw_names, "DI Water — Initial (pH 7, trace radicals)")
analyze_eigenvalues(J_sal, sal_names, "Saline — Initial (pH 7, Cl⁻=0.154M, trace radicals)")

# =============================================================================
# Scenario 2: Interface cell (dissolved O3 + moderate radicals)
# =============================================================================
print("\n" + "#"*70)
print("# SCENARIO 2: Interface-like (O3=1µM, OH=1nM, moderate radicals)")
print("#"*70)

# DI water
y_diw2 = np.full(chem_diw.n_species, trace)
y_diw2[diw_idx['H+']] = 1e-4     # pH 4
y_diw2[diw_idx['O3']] = 1e-6     # 1 µM O3
y_diw2[diw_idx['OH']] = 1e-9     # 1 nM OH
y_diw2[diw_idx['H']] = 1e-12
y_diw2[diw_idx['O2']] = 2.5e-4   # dissolved O2
y_diw2[diw_idx['H2O2_total']] = 1e-5  # 10 µM H2O2
y_diw2[diw_idx['HONO2_total']] = 5e-5  # 50 µM HNO3
y_diw2[diw_idx['HONO_total']] = 3e-6  # 3 µM HONO
y_diw2[diw_idx['HO2_total']] = 1e-9

# Saline
y_sal2 = np.full(chem_sal.n_species, trace)
y_sal2[sal_idx['H+']] = 1e-4
y_sal2[sal_idx['Cl-']] = 0.154
y_sal2[sal_idx['O3']] = 1e-6
y_sal2[sal_idx['OH']] = 1e-9
y_sal2[sal_idx['H']] = 1e-12
y_sal2[sal_idx['O2']] = 2.5e-4
y_sal2[sal_idx['H2O2_total']] = 1e-5
y_sal2[sal_idx['HONO2_total']] = 5e-5
y_sal2[sal_idx['HONO_total']] = 3e-6
y_sal2[sal_idx['HO2_total']] = 1e-9
if 'Cl' in sal_idx:
    y_sal2[sal_idx['Cl']] = 1e-12   # trace Cl radical

# Apply QSSA to saline
chem_sal.apply_qssa(y_sal2)
print(f"  QSSA: HOCl⁻={y_sal2[sal_idx['HOCl-']]:.3e}, Cl₂⁻={y_sal2[sal_idx['Cl2-']]:.3e}")

J_diw2 = numerical_jacobian(rhs_diw, y_diw2)
J_sal2 = numerical_jacobian(rhs_sal, y_sal2)

ev_diw2, _, sr_diw2 = analyze_eigenvalues(J_diw2, diw_names, "DI Water — Interface (O3=1µM, OH=1nM)")
ev_sal2, _, sr_sal2 = analyze_eigenvalues(J_sal2, sal_names, "Saline — Interface (O3=1µM, OH=1nM, Cl⁻=0.154M)")

# =============================================================================
# Scenario 3: Diffusion-front cell (very low dissolved gas)
# =============================================================================
print("\n" + "#"*70)
print("# SCENARIO 3: Diffusion front (O3=1nM, OH=1fM) — the outlier cells")
print("#"*70)

# DI water
y_diw3 = np.full(chem_diw.n_species, trace)
y_diw3[diw_idx['H+']] = 1e-5     # pH 5
y_diw3[diw_idx['O3']] = 1e-9     # 1 nM
y_diw3[diw_idx['OH']] = 1e-15
y_diw3[diw_idx['O2']] = 1e-6
y_diw3[diw_idx['HONO2_total']] = 1e-7

# Saline
y_sal3 = np.full(chem_sal.n_species, trace)
y_sal3[sal_idx['H+']] = 1e-5
y_sal3[sal_idx['Cl-']] = 0.154
y_sal3[sal_idx['O3']] = 1e-9
y_sal3[sal_idx['OH']] = 1e-15
y_sal3[sal_idx['O2']] = 1e-6
y_sal3[sal_idx['HONO2_total']] = 1e-7
if 'Cl' in sal_idx:
    y_sal3[sal_idx['Cl']] = 1e-18
chem_sal.apply_qssa(y_sal3)

J_diw3 = numerical_jacobian(rhs_diw, y_diw3)
J_sal3 = numerical_jacobian(rhs_sal, y_sal3)

analyze_eigenvalues(J_diw3, diw_names, "DI Water — Diffusion front (O3=1nM)")
analyze_eigenvalues(J_sal3, sal_names, "Saline — Diffusion front (O3=1nM, Cl⁻=0.154M)")

# =============================================================================
# Per-reaction analysis (most stiff scenario)
# =============================================================================
print("\n" + "#"*70)
print("# PER-REACTION STIFFNESS (Interface scenario)")
print("#"*70)
per_reaction_stiffness(chem_diw, y_diw2, diw_names, "DI Water")
per_reaction_stiffness(chem_sal, y_sal2, sal_names, "Saline")

# =============================================================================
# Compare: which SALINE reactions add the most stiffness?
# =============================================================================
print("\n" + "#"*70)
print("# SALINE-ONLY REACTIONS: eigenvalue contribution")
print("#"*70)

# Compute Jacobian with only base reactions vs base+saline
chem_base_only = AqueousChemistry1D(saline_mode=False)
# Can't directly test saline species with base-only chem,
# so we compare eigenvalue spectra instead.

# Eigenvalue histogram comparison
print("\n  Eigenvalue magnitude distribution:")
for label, ev in [("DI Water Interface", ev_diw2), ("Saline Interface", ev_sal2)]:
    mags = np.abs(np.real(ev))
    mags = mags[mags > 1e-20]
    bins = [1e-10, 1e-5, 1e0, 1e3, 1e6, 1e9, 1e12, 1e15]
    counts = np.histogram(np.log10(mags), bins=np.log10(bins))[0]
    print(f"\n  {label}:")
    for i, c in enumerate(counts):
        if c > 0:
            print(f"    |λ| ∈ [{bins[i]:.0e}, {bins[i+1]:.0e}): {c} modes")

print("\nDone.")
