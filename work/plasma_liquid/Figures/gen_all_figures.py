#!/usr/bin/env python3
"""
Generate all figures (Fig 1 ~ 5) from fresh simulations.

Runs needed simulations, caches results as .npz, then plots.
Re-running with cache skips simulation (plot-only).

Usage:
    python gen_all_figures.py              # all figures
    python gen_all_figures.py --rerun      # force re-simulation
    python gen_all_figures.py --fig 1 3 5  # selected figures only
"""

import sys
import os
import math
import time as time_mod
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ, ACID_BASE_PAIRS, GAS_TO_AQUEOUS_MAP
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

CACHE_DIR = _script_dir / 'cache'
DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

# Grid
DZ_MIN = 5e-6
STRETCH = 1.12

# Reference case
REF_ALPHA = 0.03
DT_SNAPSHOT = 2.0     # seconds, snapshot interval for all simulations
MIN_STABLE_RUN = 5    # consecutive nonzero points to define stable detection

# BC comparison cases (Fig 1)
BC_CASES = [
    ('Two-film',      'two_film',   1.0),
    ('Dirichlet',     'dirichlet',  1.0),
    ('Film (ab=1)',   'film',       1.0),
    ('Film+ab=0.05',  'film_alpha', 0.05),
    ('Film+ab=0.01',  'film_alpha', 0.01),
]

# MT flux cases (Fig 1b) — subset of BCs with dense snapshots
MT_BC_CASES = [
    ('Two-film',       'two_film',   1.0),
    ('Film (ab=1)',    'film',       1.0),
    ('Film+ab=0.03',   'film_alpha', 0.03),
    ('Film+ab=0.01',   'film_alpha', 0.01),
]

# Species to track MT flux
MT_SPECIES = [
    ('N2O5', 'N\u2082O\u2085'),
    ('O3',   'O\u2083'),
    ('NO2',  'NO\u2082'),
    ('NO3',  'NO\u2083'),
]

# alpha_b sweep (Fig 3)
ALPHA_CASES = [0.01, 0.03, 0.05]

# Experimental targets
EXP = {'pH': 3.61, 'NO3': 63.0, 'NO2': 3.0, 'H2O2': 11.0}

# Species for Fig 2 rate evolution
TARGET_SPECIES = ['NO3-', 'O3', 'NO2-', 'H2O2']
SPEC_TO_TOTAL = {
    'HONO': 'HONO_total', 'NO2-': 'HONO_total',
    'HONO2': 'HONO2_total', 'NO3-': 'HONO2_total',
    'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
    'HO2': 'HO2_total', 'O2-': 'HO2_total',
    'ONOOH': 'ONOOH_total', 'ONOO-': 'ONOOH_total',
    'O2NOOH': 'O2NOOH_total', 'O2NOO-': 'O2NOOH_total',
}

# Fig 5 snapshots and species
SNAP_TIMES_MIN = [1, 2, 4, 6, 8, 12]
SPATIAL_SPECIES = [
    ('HONO2_total', 'NO3-',  'uM',  1e6),
    ('O3',          'O3',    'uM',  1e6),
    ('H2O2_total',  'H2O2',  'nM',  1e9),
    ('OH',          'OH',    'pM',  1e12),
    ('HO2_total',   'HO2',   'pM',  1e12),
]

# Unicode labels
_UNI = {
    'NO3-': 'NO\u2083\u207b', 'O3': 'O\u2083', 'NO2-': 'NO\u2082\u207b',
    'H2O2': 'H\u2082O\u2082', 'OH': 'OH', 'HO2': 'HO\u2082',
    'N2O5': 'N\u2082O\u2085', 'NO2': 'NO\u2082', 'NO3': 'NO\u2083',
    'N2O4': 'N\u2082O\u2084', 'O2': 'O\u2082', 'H2O': 'H\u2082O',
    'H+': 'H\u207a', 'OH-': 'OH\u207b', 'HO2-': 'HO\u2082\u207b',
    'O2-': 'O\u2082\u207b', 'ONOO-': 'ONOO\u207b', 'O3-': 'O\u2083\u207b',
    'HONO_total': 'HONO_t', 'HONO2_total': 'HNO\u2083_t',
    'H2O2_total': 'H\u2082O\u2082_t', 'HO2_total': 'HO\u2082_t',
    'ONOOH_total': 'ONOOH_t', 'O2NOOH_total': 'O\u2082NOOH_t',
    'N2O3': 'N\u2082O\u2083', 'HO3': 'HO\u2083',
    'O2NOO-': 'O\u2082NOO\u207b', 'O2NOOH': 'O\u2082NOOH',
}


def _uni(sp):
    return _UNI.get(sp, sp)


# ═══════════════════════════════════════════════════════════════════════
# Gas Data
# ═══════════════════════════════════════════════════════════════════════

def _preprocess_below_lod(vals):
    """Linear interpolation for below-LOD (zero) data points.

    1. Find stable detection start (MIN_STABLE_RUN consecutive nonzero).
    2. Before stable start: linear ramp from 0 to first stable value.
    3. After stable start: linear interp between nonzero points.
    """
    out = vals.copy()
    n = len(vals)

    # Find stable start
    run_start, run_len = -1, 0
    stable_start = n
    for i in range(n):
        if vals[i] > 0:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= MIN_STABLE_RUN:
                stable_start = run_start
                break
        else:
            run_len = 0

    if stable_start >= n:
        return np.maximum(out, 0.0)

    # Before stable start: linear ramp
    first_val = vals[stable_start]
    for i in range(stable_start):
        out[i] = first_val * (i / max(stable_start, 1))

    # After stable start: fill intermittent zeros by linear interp
    nz_after = [(i, vals[i]) for i in range(stable_start, n) if vals[i] > 0]
    if len(nz_after) >= 2:
        nz_idx = np.array([x[0] for x in nz_after])
        nz_vals = np.array([x[1] for x in nz_after])
        for i in range(stable_start, n):
            if out[i] <= 0:
                out[i] = np.interp(i, nz_idx, nz_vals)

    return np.maximum(out, 0.0)


def load_gas_data():
    df = pd.read_csv(DEFAULT_CSV)
    times = np.arange(len(df), dtype=float) * 2.0  # OAS 2-sec intervals
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            raw = np.maximum(df[col].values.astype(float), 0.0)
            gas_conc[col] = _preprocess_below_lod(raw)
        else:
            gas_conc[col] = np.zeros(len(df))
    # Estimate N2O4 from preprocessed NO2
    if 'N2O4' not in df.columns or np.all(df.get('N2O4', pd.Series([0])).values == 0):
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298)
            + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)
    return times, gas_conc


# ═══════════════════════════════════════════════════════════════════════
# Simulation Runner with Cache
# ═══════════════════════════════════════════════════════════════════════

def run_case(times, gas_conc, bc_type, alpha_b, label, rerun=False):
    """Run one DIW case with 2s snapshots. Returns cached data dict.

    All simulations use the same t_eval (2s intervals).
    Each unique (bc_type, alpha_b) runs once; figures extract what they need.
    """
    key = f"{bc_type}_ab{alpha_b:.4f}"
    cache_file = CACHE_DIR / f"{key}.npz"

    if cache_file.exists() and not rerun:
        print(f"  [{label}] loading from cache")
        return dict(np.load(cache_file, allow_pickle=True))

    print(f"  [{label}] running simulation...")
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=DZ_MIN,
        stretch_ratio=STRETCH,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type=bc_type,
        alpha_b=alpha_b,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=0, hono2_gas=0, h2o2_gas=0,
    )
    t_end = float(times[-1])
    t_eval = np.arange(DT_SNAPSHOT, t_end + 0.1, DT_SNAPSHOT)
    t_eval = t_eval[t_eval <= t_end + 0.1]

    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval, y0=y0,
        verbose=True, dt_poisson=None,
    )
    wall = time_mod.time() - t0
    print(f"    done in {wall:.1f}s")

    # Extract results
    avg = result['spatial_avg']
    N_z, N_s = solver.N_z, solver.N_s
    data = {
        'pH': np.float64(result['pH_avg']),
        'pH_surface': np.float64(result.get('pH_surface', 0)),
        'wall_s': np.float64(wall),
        'success': np.bool_(result['success']),
        'nfev': np.int64(result.get('nfev', 0)),
        'N_z': np.int64(N_z),
        'N_s': np.int64(N_s),
        'z_centers': solver.z_centers,
        'dz_cells': solver.dz_cells,
        'L': np.float64(solver.L),
    }

    for sp in ['NO2-', 'NO3-', 'H2O2', 'O3', 'OH', 'HO2',
               'ONOOH', 'O2NOOH', 'ONOO-', 'O2NOO-',
               'NO2', 'O3-', 'O2-', 'N2O5', 'HO3']:
        data[f'avg_{sp}'] = np.float64(avg.get(sp, 0.0))

    y_final = result['y_final']
    if y_final.ndim == 1:
        y_final = y_final.reshape(N_z, N_s)
    data['y_final'] = y_final
    data['y0'] = y0.reshape(N_z, N_s)

    # Snapshots (always collected at DT_SNAPSHOT intervals)
    snap_t = np.array([0.0] + [float(tv) for tv in result['t_eval']])
    snap_y = [y0.reshape(N_z, N_s).copy()]
    for yv in result['y_eval']:
        snap_y.append(np.array(yv).reshape(N_z, N_s))
    data['snap_t'] = snap_t
    data['snap_y'] = np.array(snap_y)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **data)
    print(f"    cached → {cache_file.name}")

    return data


# ═══════════════════════════════════════════════════════════════════════
# Rate Computation Utilities (for Fig 2 & 4)
# ═══════════════════════════════════════════════════════════════════════

def _get_solver(times, gas_conc, alpha_b=REF_ALPHA):
    """Create a solver instance (for rate evaluation, not simulation)."""
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type='film_alpha', alpha_b=alpha_b,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=0, hono2_gas=0, h2o2_gas=0)
    return solver


def compute_rates_snapshot(solver, y_2d, t):
    """Per-reaction volume-averaged rates + MT flux at one snapshot."""
    chem = solver.chem
    N_z, dz, L = solver.N_z, solver.dz_cells, solver.L
    n_rxn = len(chem.reactions)
    h_idx = chem.species_idx['H+']

    rates_2d = np.zeros((n_rxn, N_z))
    for j in range(N_z):
        yc = np.clip(y_2d[j, :].copy(), chem.trace, 1.0)
        yc[h_idx] = max(yc[h_idx], 1e-14)
        spec = chem.speciate(yc)
        for ri, rxn_d in enumerate(chem._rxn_data):
            rates_2d[ri, j] = chem._compute_single_rate(rxn_d, yc, spec)

    rate_avg = np.dot(rates_2d, dz) / L

    rxn_rates = []
    for ri in range(n_rxn):
        rxn = chem.reactions[ri]
        rxn_rates.append({
            'label': rxn.get('label', f'R{ri}'),
            'rate': rate_avg[ri],
            'reactants': rxn['reactants'],
            'products': rxn.get('products', {}),
        })

    mt_flux = {}
    hp_idx = solver._h_plus_idx
    h_s = max(y_2d[0, hp_idx], 1e-14) if hp_idx >= 0 else 1e-7
    idx_to_name = {v: k for k, v in solver.species_idx.items()}
    for aq_idx, k_mt, gas_sp, _, Ka in solver._interface_species:
        C_eq = solver._get_C_eq_fast(gas_sp, t)
        C_0 = y_2d[0, aq_idx]
        c_eff = C_0 * h_s / (h_s + Ka) if Ka is not None else C_0
        mt_flux[idx_to_name[aq_idx]] = k_mt * (C_eq - c_eff) / L

    return rxn_rates, mt_flux


def _total_match_names(sp):
    names = {sp}
    total = SPEC_TO_TOTAL.get(sp)
    if total:
        names.add(total)
        for s, t in SPEC_TO_TOTAL.items():
            if t == total:
                names.add(s)
    return names


def species_contribution(rxn_rates, species_name, mt_flux):
    """Net rate contribution of each reaction to one species."""
    match = _total_match_names(species_name)
    contribs = []
    for r in rxn_rates:
        in_r = set(r['reactants'].keys()) & match
        in_p = set(r['products'].keys()) & match
        if not in_r and not in_p:
            continue
        net = 0.0
        for sp in in_p:
            net += int(r['products'][sp]) * r['rate']
        for sp in in_r:
            net -= int(r['reactants'][sp]) * r['rate']
        if abs(net) > 1e-30:
            contribs.append((r['label'], net))
    mt_val = sum(mt_flux.get(n, 0.0) for n in match)
    if abs(mt_val) > 1e-30:
        contribs.append(('MT', mt_val))
    return contribs


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Run All Simulations
# ═══════════════════════════════════════════════════════════════════════

def run_all_simulations(rerun=False):
    """Run unique (bc_type, alpha_b) simulations, return collected data dict.

    6 unique runs: 5 BC cases + alpha_b=0.03 (if not already in BC_CASES).
    All with 2s t_eval — each figure extracts what it needs.
    """
    times, gas_conc = load_gas_data()

    # Collect all unique (bc_type, alpha_b) combinations
    all_cases = {}  # (bc_type, alpha_b) → label
    for label, bc_type, ab in BC_CASES:
        all_cases[(bc_type, ab)] = label
    for ab in ALPHA_CASES:
        key = ('film_alpha', ab)
        if key not in all_cases:
            all_cases[key] = f'Film+ab={ab}'
    for label, bc_type, ab in MT_BC_CASES:
        key = (bc_type, ab)
        if key not in all_cases:
            all_cases[key] = label

    print(f"\n=== Running {len(all_cases)} unique simulations ===")
    cache = {}
    for (bc_type, ab), label in all_cases.items():
        cache[(bc_type, ab)] = run_case(
            times, gas_conc, bc_type, ab, label, rerun=rerun)

    # Build data dict with views for each figure
    data = {
        'bc': [(label, cache[(bc_type, ab)])
               for label, bc_type, ab in BC_CASES],
        'alpha': [(ab, cache[('film_alpha', ab)])
                  for ab in ALPHA_CASES],
        'ref': cache[('film_alpha', REF_ALPHA)],
        'mt': [(label, bc_type, ab, cache[(bc_type, ab)])
               for label, bc_type, ab in MT_BC_CASES],
        'times': times,
        'gas_conc': gas_conc,
    }
    return data


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: BC Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════

def gen_fig1(data):
    import matplotlib.pyplot as plt
    print("\n--- Fig 1: BC comparison ---")

    labels = [lab for lab, _ in data['bc']]
    pH_vals = [r['pH'].item() for _, r in data['bc']]
    no3_vals = [r['avg_NO3-'].item() * 1e6 for _, r in data['bc']]
    no2_vals = [r['avg_NO2-'].item() * 1e6 for _, r in data['bc']]
    h2o2_vals = [r['avg_H2O2'].item() * 1e6 for _, r in data['bc']]

    x = np.arange(len(labels))
    w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    panels = [
        ('pH', pH_vals, EXP['pH'], False, (1.5, 5.0)),
        ('NO\u2082\u207b (\u00b5M)', no2_vals, EXP['NO2'], False, None),
        ('NO\u2083\u207b (\u00b5M)', no3_vals, EXP['NO3'], True, (20, 15000)),
        ('H\u2082O\u2082 (\u00b5M)', h2o2_vals, EXP['H2O2'], False, None),
    ]

    for i, (ylabel, vals, exp_val, use_log, ylim) in enumerate(panels):
        ax = axes.flat[i]
        ax.bar(x, vals, w, color='#4878a8', edgecolor='black', lw=0.8, alpha=0.85)
        ax.axhline(exp_val, color='k', ls='--', lw=1.2)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
        ax.set_title(f'({"abcd"[i]}) {ylabel}')
        if use_log:
            ax.set_yscale('log')
        if ylim:
            ax.set_ylim(ylim)
        for bar, val in zip(ax.patches, vals):
            ypos = val * 1.15 if use_log and val > 0 else val + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            txt = f'{val:.1f}' if val >= 0.1 else f'{val*1e3:.1f} nM'
            ax.text(bar.get_x() + bar.get_width() / 2, max(ypos, 0),
                    txt, ha='center', va='bottom', fontsize=8)

    fig.suptitle('Effect of gas-liquid interface BC model (DIW, 3.2 kVpp, 12 min)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig1_bc_comparison')


# ═══════════════════════════════════════════════════════════════════════
# Figure 1b: MT Flux Time Series by BC Type
# ═══════════════════════════════════════════════════════════════════════

def gen_fig1b(data):
    import matplotlib.pyplot as plt
    print("\n--- Fig 1b: MT flux time series ---")

    times, gas_conc = data['times'], data['gas_conc']
    mt_results = data['mt']

    bc_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']
    n_sp = len(MT_SPECIES)
    fig, axes = plt.subplots(2, n_sp, figsize=(4.5 * n_sp, 8), sharex=True)

    for col, (gas_name, sp_label) in enumerate(MT_SPECIES):
        ax_inst = axes[0, col]
        ax_cum = axes[1, col]

        for ri, (label, bc_type, ab, rdata) in enumerate(mt_results):
            snap_t = rdata['snap_t']
            snap_y = rdata['snap_y']
            N_z = int(rdata['N_z'])
            N_s = int(rdata['N_s'])

            solver = _get_solver(times, gas_conc, alpha_b=ab)
            # Override bc_type for this solver
            solver_bc = PDESolver1D(
                chemistry=AqueousChemistry1D(saline_mode=False),
                dz_min=DZ_MIN, stretch_ratio=STRETCH,
                saline_mode=False, bc_type=bc_type, alpha_b=ab,
            )
            solver_bc.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                                   hono_gas=0, hono2_gas=0, h2o2_gas=0)

            # Find MT mapping for this gas species
            idx_to_name = {v: k for k, v in solver_bc.species_idx.items()}
            mt_info = None
            for aq_idx, k_mt, gas_sp, _, Ka in solver_bc._interface_species:
                for g, a in GAS_TO_AQUEOUS_MAP.items():
                    if g == gas_name and a == idx_to_name[aq_idx]:
                        mt_info = (aq_idx, k_mt, gas_sp, Ka)
                        break

            if mt_info is None:
                continue

            aq_idx, k_mt, gas_sp, Ka = mt_info
            L = solver_bc.L
            hp_idx = solver_bc._h_plus_idx
            n_snaps = len(snap_t)
            flux = np.zeros(n_snaps)

            for si in range(n_snaps):
                y2d = snap_y[si]
                C_eq = solver_bc._get_C_eq_fast(gas_sp, snap_t[si])
                C_s = y2d[0, aq_idx]
                if Ka is not None and hp_idx >= 0:
                    h_s = max(y2d[0, hp_idx], 1e-14)
                    C_s = C_s * h_s / (h_s + Ka)
                flux[si] = k_mt * (C_eq - C_s) / L

            t_min = snap_t / 60.0
            cum = np.cumsum(flux[:-1] * np.diff(snap_t)) * 1e6  # µM
            cum = np.concatenate(([0.0], cum))

            color = bc_colors[ri % len(bc_colors)]
            ax_inst.plot(t_min, flux, color=color, lw=1.2, label=label)
            ax_cum.plot(t_min, cum, color=color, lw=1.2, label=label)

        ax_inst.set_title(f'{sp_label}', fontweight='bold')
        ax_inst.set_ylabel('Flux (M/s)')
        ax_inst.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax_cum.set_ylabel('Cumulative (\u00b5M)')
        ax_cum.set_xlabel('Time (min)')
        if col == 0:
            ax_inst.legend(fontsize=7, loc='best')

    axes[0, 0].set_ylabel('Instantaneous flux (M/s)')
    axes[1, 0].set_ylabel('Cumulative (\u00b5M)')

    fig.suptitle('Mass transfer flux by BC type (DIW, 720 s)', fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig1b_mt_flux')


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Rate Evolution (time-resolved per-reaction rates)
# ═══════════════════════════════════════════════════════════════════════

def gen_fig2(data):
    import matplotlib.pyplot as plt
    print("\n--- Fig 2: Rate evolution ---")

    ref = data['ref']
    snap_t = ref['snap_t']
    snap_y = ref['snap_y']
    times, gas_conc = data['times'], data['gas_conc']

    solver = _get_solver(times, gas_conc)
    nt = len(snap_t)

    # Compute per-reaction rates at each snapshot
    print("  computing per-reaction rates...")
    all_rxn, all_mt = [], []
    for i in range(nt):
        rr, mt = compute_rates_snapshot(solver, snap_y[i], snap_t[i])
        all_rxn.append(rr)
        all_mt.append(mt)

    # Volume-averaged concentration time series
    dz, L = solver.dz_cells, solver.L
    conc = {}
    for sp in TARGET_SPECIES:
        total = SPEC_TO_TOTAL.get(sp, sp)
        idx = solver.chem.species_idx.get(total, solver.chem.species_idx.get(sp))
        if idx is not None:
            conc[sp] = np.array([np.dot(snap_y[i][:, idx], dz) / L for i in range(nt)])

    # Build time series per species
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    t_min = snap_t / 60.0

    for pi, sp in enumerate(TARGET_SPECIES):
        ax = axes.flat[pi]
        by_label = defaultdict(lambda: np.zeros(nt))
        for i in range(nt):
            for label, rate in species_contribution(all_rxn[i], sp, all_mt[i]):
                by_label[label][i] = rate

        # Filter: keep >=1% contribution at any time
        max_total = max(sum(abs(r[i]) for r in by_label.values()) for i in range(nt)) if nt else 1
        sig_labels = []
        for label, rates in by_label.items():
            peak_frac = np.max(np.abs(rates)) / max(max_total, 1e-30)
            if peak_frac >= 0.01:
                sig_labels.append(label)

        # Despike only: median filter removes BDF dense output spikes.
        # No moving average — preserves real transients (e.g. radical ignition).
        from scipy.ndimage import median_filter
        med_win = max(int(10 / DT_SNAPSHOT), 5)  # 10s median window

        for label in sig_labels:
            r = by_label[label]
            ax.plot(t_min, median_filter(r, size=med_win), lw=1.2,
                    label=label[:40])

        # net dC/dt: finite-difference from actual concentration (ground truth)
        if sp in conc:
            c_arr = conc[sp]
            net_fd = np.zeros(nt)
            dt_snap = np.diff(snap_t)
            dc = np.diff(c_arr)
            dcdt = dc / dt_snap
            # Central diff (interior), forward/backward at edges
            net_fd[0] = dcdt[0] if len(dcdt) > 0 else 0
            net_fd[-1] = dcdt[-1] if len(dcdt) > 0 else 0
            if nt > 2:
                net_fd[1:-1] = 0.5 * (dcdt[:-1] + dcdt[1:])
            ax.plot(t_min, median_filter(net_fd, size=med_win),
                    'k--', lw=2, label=r'$\Delta C / \Delta t$')

        ax.set_ylabel(f'd[{_uni(sp)}]/dt (M/s)')
        ax.set_title(f'({"abcd"[pi]}) {_uni(sp)}', fontweight='bold', loc='left')
        ax.axhline(0, color='gray', lw=0.5)
        ax.legend(fontsize=7, loc='best')

    for ax in axes[1]:
        ax.set_xlabel('Time (min)')

    fig.suptitle(f'Rate evolution (DIW, Film+\u03b1b, \u03b1b={REF_ALPHA})',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig2_rate_evolution')


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Radical Concentrations Table
# ═══════════════════════════════════════════════════════════════════════

def gen_fig3(data):
    import matplotlib.pyplot as plt
    print("\n--- Fig 3: Radical concentrations ---")

    rad_species = [
        ('O3',      'O\u2083'),
        ('O2NOOH',  'O\u2082NOOH'),
        ('NO2',     'NO\u2082'),
        ('ONOOH',   'ONOOH'),
        ('O2NOO-',  'O\u2082NOO\u207b'),
        ('HO2',     'HO\u2082'),
        ('O2-',     'O\u2082\u207b'),
        ('OH',      'OH'),
        ('ONOO-',   'ONOO\u207b'),
        ('O3-',     'O\u2083\u207b'),
        ('N2O5',    'N\u2082O\u2085'),
    ]

    # Gather data
    rows = []
    for sp_key, sp_label in rad_species:
        vals = []
        for ab, r in data['alpha']:
            v = r.get(f'avg_{sp_key}', np.float64(0)).item()
            vals.append(v)
        if max(vals) > 0:
            exp_order = int(math.floor(math.log10(max(vals))))
            scale = 10 ** (-exp_order)
            rows.append((sp_label, exp_order,
                         [v * scale for v in vals]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    ab_labels = [f'\u03b1b = {ab}' for ab in ALPHA_CASES]
    col_labels = ['Species', 'Order (M)'] + ab_labels
    cell_text = []
    for sp_label, exp_order, scaled_vals in rows:
        order_str = f'1e{exp_order}'
        cell_text.append(
            [sp_label, order_str] + [f'{v:.2f}' for v in scaled_vals]
        )

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c4a6e')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(cell_text)):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)
        table[i + 1, 0].set_text_props(fontweight='bold')

    ax.set_title('Radical and intermediate species (DIW, Film + \u03b1b BC, 720 s)',
                 fontsize=12, pad=20)
    fig.tight_layout()
    _save(fig, 'fig3_radicals')


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Mass Balance Breakdown
# ═══════════════════════════════════════════════════════════════════════

def gen_fig4(data):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    import matplotlib.patches as mpatches
    print("\n--- Fig 4: Mass balance ---")

    ref = data['ref']
    times, gas_conc = data['times'], data['gas_conc']
    solver = _get_solver(times, gas_conc)

    y_final = ref['y_final']
    t_end = float(times[-1])
    rxn_rates, mt_flux = compute_rates_snapshot(solver, y_final, t_end)

    src_color, snk_color = '#2166ac', '#c0392b'
    species_list = TARGET_SPECIES

    panels_data = []
    for sp in species_list:
        contribs = species_contribution(rxn_rates, sp, mt_flux)
        sources = [(lab, val) for lab, val in contribs if val > 0]
        sinks = [(lab, -val) for lab, val in contribs if val < 0]

        total_src = sum(v for _, v in sources) if sources else 1e-30
        total_snk = sum(v for _, v in sinks) if sinks else 1e-30

        src_pct = [(lab, val / total_src * 100) for lab, val in
                   sorted(sources, key=lambda x: -x[1])[:5]]
        snk_pct = [(lab, val / total_snk * 100) for lab, val in
                   sorted(sinks, key=lambda x: -x[1])[:5]]

        panels_data.append({
            'title': _uni(sp),
            'src_rate': total_src,
            'sources': src_pct,
            'sinks': snk_pct,
        })

    n_panels = len(panels_data)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels))
    if n_panels == 1:
        axes = [axes]

    bar_h = 0.6
    for pi, (ax, pd_) in enumerate(zip(axes, panels_data)):
        labels, values, colors, ypos = [], [], [], []
        y = 0
        for lab, pct in pd_['sources']:
            labels.append(lab); values.append(pct)
            colors.append(src_color); ypos.append(y); y += 1
        if not pd_['sources']:
            labels.append('(none)'); values.append(0)
            colors.append('#ccc'); ypos.append(y); y += 1
        y += 0.3
        for lab, pct in pd_['sinks']:
            labels.append(lab); values.append(-pct)
            colors.append(snk_color); ypos.append(y); y += 1
        if not pd_['sinks']:
            labels.append('(accumulating)'); values.append(0)
            colors.append('#ccc'); ypos.append(y); y += 1

        ax.barh(ypos, values, height=bar_h, color=colors,
                edgecolor='black', lw=0.7)
        ax.set_yticks([])
        ax.set_ylim(max(ypos) + 0.5, min(ypos) - 0.5)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlim(-105, 105)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.0f}'))
        ax.set_title(f'({"abcd"[pi]}) {pd_["title"]}  '
                     f'(\u03a3src = {pd_["src_rate"]:.2e} M/s)',
                     fontweight='bold', loc='left')

        for yp, val, lab in zip(ypos, values, labels):
            if val >= 0:
                ax.text(-2, yp, lab, va='center', ha='right', fontsize=8)
            else:
                ax.text(2, yp, lab, va='center', ha='left', fontsize=8)
            av = abs(val)
            if av < 0.5:
                continue
            if av > 40:
                ax.text(val - np.sign(val) * 3, yp, f'{av:.1f}%',
                        va='center', ha='right' if val > 0 else 'left',
                        fontsize=9, fontweight='bold', color='white')
            else:
                ax.text(val + np.sign(val) * 1.5, yp, f'{av:.1f}%',
                        va='center', ha='left' if val > 0 else 'right',
                        fontsize=9, fontweight='bold')

    axes[-1].set_xlabel('% of turnover')

    src_p = mpatches.Patch(facecolor=src_color, edgecolor='black', lw=0.7, label='Source')
    snk_p = mpatches.Patch(facecolor=snk_color, edgecolor='black', lw=0.7, label='Sink')
    fig.legend(handles=[src_p, snk_p], loc='upper right', fontsize=11)

    fig.suptitle(f'Mass balance (DIW, \u03b1b = {REF_ALPHA}, 720 s)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig4_mass_balance')


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Spatial Profiles
# ═══════════════════════════════════════════════════════════════════════

def gen_fig5(data):
    import matplotlib.pyplot as plt
    print("\n--- Fig 5: Spatial profiles ---")

    ref = data['ref']
    snap_t = ref['snap_t']
    snap_y = ref['snap_y']
    z_mm = ref['z_centers'] * 1e3
    N_z = int(ref['N_z'])
    N_s = int(ref['N_s'])

    times, gas_conc = data['times'], data['gas_conc']
    solver = _get_solver(times, gas_conc)
    chem = solver.chem

    # Find snapshot indices closest to requested times
    snap_targets = [t * 60.0 for t in SNAP_TIMES_MIN]
    snap_idx = []
    for tgt in snap_targets:
        if tgt <= snap_t[-1]:
            snap_idx.append(np.argmin(np.abs(snap_t - tgt)))

    n_panels = len(SPATIAL_SPECIES) + 1
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), sharex=True)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(snap_idx) - 1, 1)) for i in range(len(snap_idx))]

    # pH panel
    ax = axes.flat[0]
    h_idx = chem.species_idx['H+']
    for ci, si in enumerate(snap_idx):
        y2d = snap_y[si]
        pH_prof = -np.log10(np.clip(y2d[:, h_idx], 1e-14, None))
        ax.plot(z_mm, pH_prof, color=colors[ci], lw=1.5,
                label=f'{snap_t[si]/60:.0f} min')
    ax.set_ylabel('pH')
    ax.set_title('(a) pH', fontweight='bold', loc='left')
    ax.legend(fontsize=7)

    # Species panels
    for pi, (sp_name, sp_label, unit, scale) in enumerate(SPATIAL_SPECIES):
        ax = axes.flat[pi + 1]
        idx = chem.species_idx.get(sp_name)
        if idx is None:
            ax.set_visible(False)
            continue
        for ci, si in enumerate(snap_idx):
            prof = np.clip(snap_y[si][:, idx], 1e-30, None) * scale
            ax.plot(z_mm, prof, color=colors[ci], lw=1.5,
                    label=f'{snap_t[si]/60:.0f} min')
        ax.set_yscale('log')
        ax.set_ylabel(f'{_uni(sp_label)} ({unit})')
        ax.set_title(f'({"bcdefgh"[pi]}) {_uni(sp_label)}',
                     fontweight='bold', loc='left')
        if pi == 0:
            ax.legend(fontsize=7)

    for i in range(n_panels, len(axes.flat)):
        axes.flat[i].set_visible(False)
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel('Depth (mm)')
            ax.set_xlim(0, z_mm[-1])

    fig.suptitle(f'Spatial profiles (DIW, Film+\u03b1b, \u03b1b={REF_ALPHA})',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig5_spatial')


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def _save(fig, name):
    for ext in ('png', 'pdf'):
        path = _script_dir / f'{name}.{ext}'
        fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  -> {name}.png/pdf saved")


def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8, 'lines.linewidth': 1.5,
    })


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

FIGURE_MAP = {
    '1':  gen_fig1,
    '1b': gen_fig1b,
    '2':  gen_fig2,
    '3':  gen_fig3,
    '4':  gen_fig4,
    '5':  gen_fig5,
}


def main():
    parser = argparse.ArgumentParser(description='Generate all figures')
    parser.add_argument('--rerun', action='store_true',
                        help='Force re-simulation (ignore cache)')
    parser.add_argument('--fig', nargs='+', default=list(FIGURE_MAP.keys()),
                        help='Which figures to generate (default: all)')
    args = parser.parse_args()

    os.chdir(_project_root)
    _setup_mpl()

    print("=" * 60)
    print("  Plasma-Liquid Figure Generator")
    print(f"  Figures: {', '.join(args.fig)}")
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Rerun: {args.rerun}")
    print("=" * 60)

    data = run_all_simulations(rerun=args.rerun)

    for fig_id in args.fig:
        if fig_id in FIGURE_MAP:
            FIGURE_MAP[fig_id](data)
        else:
            print(f"  [WARN] Unknown figure: {fig_id}")

    print("\n" + "=" * 60)
    print("  All done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
