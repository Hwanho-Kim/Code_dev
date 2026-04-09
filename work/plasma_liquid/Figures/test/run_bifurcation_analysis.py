#!/usr/bin/env python3
"""
Radical bifurcation analysis: Step 1~3.

Step 1: Slow driver identification — what changes gradually before t≈2min
Step 2: Jacobian eigenvalue analysis — pre/post transition
Step 3: Perturbation tests — which variable triggers the transition
"""
import numpy as np, sys, time as time_mod, math
from pathlib import Path

_dir = Path(__file__).parent
_proj = _dir.parent.parent
sys.path.insert(0, str(_proj / 'Ver4_1D'))
sys.path.insert(0, str(_dir.parent))

from config_1d import PHYSICAL, N2O4_EQ, ODE_CONFIG, ACID_BASE_PAIRS
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from gen_all_figures import load_gas_data

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Load simulation data
times, gas_conc = load_gas_data()
ref = dict(np.load(_dir.parent / 'cache' / 'film_alpha_ab0.0300.npz', allow_pickle=True))
snap_t = ref['snap_t']
snap_y = ref['snap_y']
dz = ref['dz_cells']
L = float(ref['L'])

chem = AqueousChemistry1D(saline_mode=False)
idx = chem.species_idx
N_z = int(ref['N_z'])
N_s = int(ref['N_s'])

Ka_ho2 = 10**(-4.8)
Ka_onooh = 10**(-6.6)

def vol_avg(snap_idx, sp):
    return np.dot(snap_y[snap_idx][:, idx[sp]], dz) / L

def surf(snap_idx, sp):
    return snap_y[snap_idx][0, idx[sp]]

nt = len(snap_t)
t_min = snap_t / 60.0


# ═══════════════════════════════════════════════════════════════
# STEP 1: Slow Driver Identification
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Slow Driver Identification")
print("=" * 60)

# Track all slowly-changing variables (surface cell, 0-3 min)
slow_vars = {
    'O3': ('O3', 1e6, 'uM'),
    'pH': (None, None, None),  # special
    'NO2(aq)': ('NO2', 1e9, 'nM'),
    'NO3(aq)': ('NO3', 1e12, 'pM'),
    'HONO2_t': ('HONO2_total', 1e6, 'uM'),
    'HONO_t': ('HONO_total', 1e9, 'nM'),
    'N2O5': ('N2O5', 1e15, 'fM'),
    'O2': ('O2', 1e4, 'x1e-4 M'),
}

fig1, axes1 = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
mask = t_min <= 4.0

pi = 0
for label, (sp, scale, unit) in slow_vars.items():
    ax = axes1.flat[pi]
    if sp is None:  # pH
        h_arr = np.array([vol_avg(i, 'H+') for i in range(nt)])
        pH_arr = -np.log10(np.clip(h_arr, 1e-14, None))
        ax.plot(t_min[mask], pH_arr[mask], 'k-', lw=1.0)
        ax.set_ylabel('pH')

        h_surf = np.array([surf(i, 'H+') for i in range(nt)])
        pH_surf = -np.log10(np.clip(h_surf, 1e-14, None))
        ax.plot(t_min[mask], pH_surf[mask], 'r--', lw=0.8, label='surface')
        ax.legend()
    else:
        v_avg = np.array([vol_avg(i, sp) for i in range(nt)]) * scale
        v_surf = np.array([surf(i, sp) for i in range(nt)]) * scale
        ax.plot(t_min[mask], v_avg[mask], 'b-', lw=1.0, label='vol avg')
        ax.plot(t_min[mask], v_surf[mask], 'r--', lw=0.8, label='surface')
        ax.set_ylabel(f'{label} ({unit})')
        ax.legend()
    ax.set_title(label, fontweight='bold', loc='left')
    ax.axvline(2.1, color='gray', ls=':', lw=0.5)
    ax.grid(True, alpha=0.2)
    pi += 1

# Last panel: gas-phase NO2 and NO3 C_eq
ax = axes1.flat[pi]
H_no2, H_no3 = 0.978, 44.0
conv = 1000.0 / PHYSICAL.AVOGADRO
no2_ceq = H_no2 * gas_conc['NO2'] * conv * 1e9
no3_ceq = H_no3 * gas_conc['NO3'] * conv * 1e9
gas_t_min = times / 60.0
gas_mask = gas_t_min <= 4.0
ax.plot(gas_t_min[gas_mask], no2_ceq[gas_mask], label='NO2 Ceq (nM)')
ax.plot(gas_t_min[gas_mask], no3_ceq[gas_mask], label='NO3 Ceq (nM)')
ax.set_ylabel('C_eq (nM)')
ax.set_title('Gas-phase C_eq', fontweight='bold', loc='left')
ax.legend()
ax.axvline(2.1, color='gray', ls=':', lw=0.5)
ax.grid(True, alpha=0.2)

for ax in axes1[-1]:
    ax.set_xlabel('Time (min)')
fig1.suptitle('Step 1: Slow drivers (0-4 min, gray line = transition at 2.1 min)',
              fontsize=14, y=1.01)
fig1.tight_layout()
fig1.savefig(_dir / 'fig_step1_slow_drivers.png')
fig1.savefig(_dir / 'fig_step1_slow_drivers.pdf')
plt.close(fig1)
print("saved: fig_step1_slow_drivers")


# ═══════════════════════════════════════════════════════════════
# STEP 2: Jacobian Eigenvalue Analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Jacobian Eigenvalue Analysis")
print("=" * 60)

# Build solver for Jacobian computation
solver = PDESolver1D(chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
                     saline_mode=False, bc_type='film_alpha', alpha_b=0.03)
solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                    hono_gas=0, hono2_gas=0, h2o2_gas=0)

# Compute chemistry Jacobian for SURFACE CELL only (where transition happens)
# J_ij = d(dydt_i)/d(C_j) from chemistry only (no diffusion/MT)
def chemistry_jacobian_cell(y_cell):
    """Finite-difference chemistry Jacobian for one cell."""
    h_idx_local = idx['H+']
    yc = np.clip(y_cell.copy(), 1e-30, 1.0)
    yc[h_idx_local] = max(yc[h_idx_local], 1e-14)

    dydt0 = chem.compute_rates(yc.copy())
    eps = 1e-8
    J = np.zeros((N_s, N_s))
    for j in range(N_s):
        h = max(abs(yc[j]) * eps, 1e-20)
        yp = yc.copy()
        yp[j] += h
        dydt_p = chem.compute_rates(yp)
        J[:, j] = (dydt_p - dydt0) / h
    return J

# Radical species indices for sub-Jacobian
radical_names = ['OH', 'O3', 'O-', 'O3-', 'HO3', 'HO2_total', 'ONOOH_total', 'NO2', 'NO3']
radical_idx = [idx[sp] for sp in radical_names]

time_points = [60, 100, 110, 116, 120, 124, 126, 130, 150, 300]

print(f"\n{'t(s)':>5} {'t(min)':>6} {'max_Re(eig)':>12} {'max_Re radical':>15} {'top eigenvalues (real part)':>40}")

eig_data = []
for t_target in time_points:
    i = np.argmin(np.abs(snap_t - t_target))
    y_surf = snap_y[i][0, :]

    J_full = chemistry_jacobian_cell(y_surf)

    # Full eigenvalues
    eigs_full = np.linalg.eigvals(J_full)
    max_re_full = np.max(eigs_full.real)

    # Radical sub-Jacobian
    J_rad = J_full[np.ix_(radical_idx, radical_idx)]
    eigs_rad = np.linalg.eigvals(J_rad)
    max_re_rad = np.max(eigs_rad.real)

    # Top 5 eigenvalues by real part
    top5 = sorted(eigs_full.real, reverse=True)[:5]
    top5_str = ', '.join(f'{v:.2e}' for v in top5)

    print(f'{snap_t[i]:5.0f} {snap_t[i]/60:6.2f} {max_re_full:12.3e} {max_re_rad:15.3e} {top5_str}')

    eig_data.append((snap_t[i], eigs_full, eigs_rad))

# Plot eigenvalue spectrum
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# (a) Max real eigenvalue vs time
ax = axes2[0, 0]
ts = [d[0] for d in eig_data]
max_re = [np.max(d[1].real) for d in eig_data]
max_re_rad = [np.max(d[2].real) for d in eig_data]
ax.plot(np.array(ts)/60, max_re, 'bo-', label='Full system')
ax.plot(np.array(ts)/60, max_re_rad, 'rs-', label='Radical subsystem')
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.axvline(2.1, color='gray', ls=':', lw=0.5)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Max Re(eigenvalue) (1/s)')
ax.set_title('(a) Maximum eigenvalue real part', fontweight='bold', loc='left')
ax.legend()
ax.grid(True, alpha=0.2)

# (b) Eigenvalue spectrum at t=100s (pre)
ax = axes2[0, 1]
i_pre = np.argmin(np.abs(snap_t - 100))
J_pre = chemistry_jacobian_cell(snap_y[i_pre][0, :])
eigs_pre = np.linalg.eigvals(J_pre)
ax.scatter(eigs_pre.real, eigs_pre.imag, s=15, alpha=0.7)
ax.axvline(0, color='gray', ls='-', lw=0.5)
ax.set_xlabel('Re(eigenvalue)')
ax.set_ylabel('Im(eigenvalue)')
ax.set_title(f'(b) Spectrum at t=100s (pre-transition)', fontweight='bold', loc='left')
ax.grid(True, alpha=0.2)

# (c) Eigenvalue spectrum at t=126s (transition)
ax = axes2[1, 0]
i_trans = np.argmin(np.abs(snap_t - 126))
J_trans = chemistry_jacobian_cell(snap_y[i_trans][0, :])
eigs_trans = np.linalg.eigvals(J_trans)
ax.scatter(eigs_trans.real, eigs_trans.imag, s=15, alpha=0.7, color='red')
ax.axvline(0, color='gray', ls='-', lw=0.5)
ax.set_xlabel('Re(eigenvalue)')
ax.set_ylabel('Im(eigenvalue)')
ax.set_title(f'(c) Spectrum at t=126s (transition)', fontweight='bold', loc='left')
ax.grid(True, alpha=0.2)

# (d) Eigenvalue spectrum at t=300s (post)
ax = axes2[1, 1]
i_post = np.argmin(np.abs(snap_t - 300))
J_post = chemistry_jacobian_cell(snap_y[i_post][0, :])
eigs_post = np.linalg.eigvals(J_post)
ax.scatter(eigs_post.real, eigs_post.imag, s=15, alpha=0.7, color='green')
ax.axvline(0, color='gray', ls='-', lw=0.5)
ax.set_xlabel('Re(eigenvalue)')
ax.set_ylabel('Im(eigenvalue)')
ax.set_title(f'(d) Spectrum at t=300s (post-transition)', fontweight='bold', loc='left')
ax.grid(True, alpha=0.2)

fig2.suptitle('Step 2: Chemistry Jacobian eigenvalues (surface cell)', fontsize=14, y=1.01)
fig2.tight_layout()
fig2.savefig(_dir / 'fig_step2_eigenvalues.png')
fig2.savefig(_dir / 'fig_step2_eigenvalues.pdf')
plt.close(fig2)
print("saved: fig_step2_eigenvalues")


# ═══════════════════════════════════════════════════════════════
# STEP 3: Perturbation Tests
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Perturbation Tests")
print("=" * 60)

# Run from t=0 to t=90s (baseline), then branch with perturbations
# Each perturbation: modify y or gas data at t=90s, run to t=300s

# Baseline: run to t=90s
solver_base = PDESolver1D(chemistry=AqueousChemistry1D(saline_mode=False),
                          dz_min=5e-6, stretch_ratio=1.12,
                          saline_mode=False, bc_type='film_alpha', alpha_b=0.03)
solver_base.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                         hono_gas=0, hono2_gas=0, h2o2_gas=0)
y0 = solver_base.build_initial_condition(initial_pH=7.0)

print("\nPhase 1: Running baseline to t=90s...")
res_pre = solver_base.solve(t_span=(0, 90), t_eval=np.arange(2, 92, 2.0),
                            y0=y0, verbose=False, dt_poisson=None)
y90_base = res_pre['y_final'].ravel()

# Define perturbation experiments
def run_perturbed(label, y90_mod, gas_conc_mod=None, t_end=300):
    """Run from t=90 to t_end with modified initial state or gas data."""
    chem_p = AqueousChemistry1D(saline_mode=False)
    solver_p = PDESolver1D(chemistry=chem_p, dz_min=5e-6, stretch_ratio=1.12,
                           saline_mode=False, bc_type='film_alpha', alpha_b=0.03)
    gc = gas_conc_mod if gas_conc_mod is not None else gas_conc
    solver_p.set_gas_data(times=times, gas_conc_molecules=gc,
                          hono_gas=0, hono2_gas=0, h2o2_gas=0)

    t_eval_p = np.arange(92, t_end + 1, 2.0)
    t0 = time_mod.time()
    res = solver_p.solve(t_span=(90, t_end), t_eval=t_eval_p,
                         y0=y90_mod, verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0

    # Extract OH surface time series
    ohi = chem_p.species_idx['OH']
    o3i = chem_p.species_idx['O3']
    st = np.array([90.0] + [float(tv) for tv in res['t_eval']])
    y90_2d = y90_mod.reshape(solver_p.N_z, solver_p.N_s)
    sy = [y90_2d.copy()] + [np.array(yv).reshape(solver_p.N_z, solver_p.N_s)
                             for yv in res['y_eval']]
    oh_s = np.array([sy[k][0, ohi] * 1e12 for k in range(len(st))])
    o3_s = np.array([sy[k][0, o3i] * 1e6 for k in range(len(st))])

    print(f"  {label:40s}: wall={wall:.1f}s, success={res['success']}")
    return st, oh_s, o3_s

# Baseline
print("\nRunning perturbation experiments (t=90→300s)...")
st_base, oh_base, o3_base = run_perturbed("Baseline", y90_base.copy())

# P1: O3 x2 at t=90s
y90_o3x2 = y90_base.copy().reshape(N_z, N_s)
y90_o3x2[:, idx['O3']] *= 2.0
st_o3x2, oh_o3x2, o3_o3x2 = run_perturbed("O3 x2", y90_o3x2.ravel())

# P2: O3 x0.5 at t=90s
y90_o3h = y90_base.copy().reshape(N_z, N_s)
y90_o3h[:, idx['O3']] *= 0.5
st_o3h, oh_o3h, o3_o3h = run_perturbed("O3 x0.5", y90_o3h.ravel())

# P3: O3 x0 (remove all O3)
y90_o3z = y90_base.copy().reshape(N_z, N_s)
y90_o3z[:, idx['O3']] = 1e-30
st_o3z, oh_o3z, o3_o3z = run_perturbed("O3 = 0", y90_o3z.ravel())

# P4: NO2 gas MT off (set gas NO2 to 0)
gc_no_no2 = {k: v.copy() for k, v in gas_conc.items()}
gc_no_no2['NO2'] = np.zeros_like(gc_no_no2['NO2'])
st_nno2, oh_nno2, o3_nno2 = run_perturbed("NO2 gas OFF", y90_base.copy(), gc_no_no2)

# P5: NO3 gas MT off
gc_no_no3 = {k: v.copy() for k, v in gas_conc.items()}
gc_no_no3['NO3'] = np.zeros_like(gc_no_no3['NO3'])
st_nno3, oh_nno3, o3_nno3 = run_perturbed("NO3 gas OFF", y90_base.copy(), gc_no_no3)

# P6: O3 gas MT off (no new O3 input, only existing O3)
gc_no_o3 = {k: v.copy() for k, v in gas_conc.items()}
gc_no_o3['O3'] = np.zeros_like(gc_no_o3['O3'])
st_no3g, oh_no3g, o3_no3g = run_perturbed("O3 gas OFF", y90_base.copy(), gc_no_o3)

# P7: pH fixed (H+ frozen at t=90 value)
# Can't easily fix pH in solver, instead double OH- to shift equilibrium
# Skip this — too complex without solver modification

# Plot
fig3, axes3 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

cases = [
    ('Baseline', st_base, oh_base, o3_base, 'k', '-', 2.0),
    ('O3 x2', st_o3x2, oh_o3x2, o3_o3x2, '#d62728', '-', 1.2),
    ('O3 x0.5', st_o3h, oh_o3h, o3_o3h, '#2ca02c', '-', 1.2),
    ('O3 = 0', st_o3z, oh_o3z, o3_o3z, '#9467bd', '--', 1.2),
    ('NO2 gas OFF', st_nno2, oh_nno2, o3_nno2, '#ff7f0e', '-.', 1.2),
    ('NO3 gas OFF', st_nno3, oh_nno3, o3_nno3, '#8c564b', ':', 1.2),
    ('O3 gas OFF', st_no3g, oh_no3g, o3_no3g, '#1f77b4', '--', 1.2),
]

for label, st, oh, o3, color, ls, lw in cases:
    axes3[0].plot(st / 60, oh, color=color, ls=ls, lw=lw, label=label)
    axes3[1].plot(st / 60, o3, color=color, ls=ls, lw=lw, label=label)

axes3[0].set_ylabel('OH surface (pM)')
axes3[0].set_title('(a) OH surface — perturbation comparison', fontweight='bold', loc='left')
axes3[0].legend(fontsize=8, loc='best')
axes3[0].axvline(2.1, color='gray', ls=':', lw=0.5)
axes3[0].grid(True, alpha=0.2)

axes3[1].set_ylabel('O3 surface (uM)')
axes3[1].set_xlabel('Time (min)')
axes3[1].set_title('(b) O3 surface — perturbation comparison', fontweight='bold', loc='left')
axes3[1].legend(fontsize=8, loc='best')
axes3[1].axvline(2.1, color='gray', ls=':', lw=0.5)
axes3[1].grid(True, alpha=0.2)

fig3.suptitle('Step 3: Perturbation tests (branching from t=90s)',
              fontsize=14, y=1.01)
fig3.tight_layout()
fig3.savefig(_dir / 'fig_step3_perturbation.png')
fig3.savefig(_dir / 'fig_step3_perturbation.pdf')
plt.close(fig3)
print("saved: fig_step3_perturbation")

print("\n=== ALL STEPS COMPLETE ===")
