#!/usr/bin/env python3
"""3 tests: Fig 2 rate evolution under different conditions."""
import numpy as np, sys, time as time_mod, math
from collections import defaultdict
from pathlib import Path

_dir = Path(__file__).parent
sys.path.insert(0, str(_dir.parent / 'Ver4_1D'))
sys.path.insert(0, str(_dir))

from config_1d import PHYSICAL, N2O4_EQ, ODE_CONFIG
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from gen_all_figures import compute_rates_snapshot, species_contribution, _uni
from gen_fig_gas_interpolation import load_raw, method_linear_interp
from scipy.ndimage import median_filter

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 7, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

TARGET_SPECIES = ['NO3-', 'O3', 'NO2-', 'H2O2']
SPEC_TO_TOTAL = {
    'NO3-': 'HONO2_total', 'NO2-': 'HONO_total',
    'H2O2': 'H2O2_total', 'HO2': 'HO2_total',
}

# Load preprocessed gas data
times, raw_data = load_raw()
gas_conc = {}
for sp in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
    gas_conc[sp] = method_linear_interp(raw_data.get(sp, np.zeros(len(times))))
no2 = gas_conc['NO2']; T = 298.15
Kp = math.exp(math.log(N2O4_EQ.KP_298) + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T))
gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)


def run_sim(label, max_step, dt_snap, dz_min=5e-6, stretch=1.12, nz=None, t_span_override=None, y0_override=None):
    old_ms = ODE_CONFIG.max_step
    ODE_CONFIG.max_step = max_step

    chem = AqueousChemistry1D(saline_mode=False)
    if nz is not None:
        solver = PDESolver1D(chemistry=chem, N_z=nz,
                             saline_mode=False, bc_type='film_alpha', alpha_b=0.03)
    else:
        solver = PDESolver1D(chemistry=chem, dz_min=dz_min, stretch_ratio=stretch,
                             saline_mode=False, bc_type='film_alpha', alpha_b=0.03)
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=0, hono2_gas=0, h2o2_gas=0)

    t_start, t_end = t_span_override or (0, float(times[-1]))
    t_eval = np.arange(t_start + dt_snap, t_end + 0.01, dt_snap)
    t_eval = t_eval[t_eval <= t_end + 0.01]

    if y0_override is not None:
        y0 = y0_override
    else:
        y0 = solver.build_initial_condition(initial_pH=7.0)

    print(f'\n=== {label} ===')
    t0 = time_mod.time()
    result = solver.solve(t_span=(t_start, t_end), t_eval=t_eval, y0=y0,
                          verbose=True, dt_poisson=None)
    print(f'Done: {time_mod.time() - t0:.1f}s, nfev={result.get("nfev", 0)}')
    ODE_CONFIG.max_step = old_ms

    N_z, N_s = solver.N_z, solver.N_s
    snap_t = np.array([t_start] + [float(tv) for tv in result['t_eval']])
    y0_2d = y0.reshape(N_z, N_s) if y0_override is not None else y0.reshape(N_z, N_s)
    snap_y = [y0_2d.copy()] + [np.array(yv).reshape(N_z, N_s) for yv in result['y_eval']]
    return snap_t, np.array(snap_y), solver, result


def plot_fig2(snap_t, snap_y, solver, dt_snap, title, filename):
    chem = solver.chem
    dz, L = solver.dz_cells, solver.L
    nt = len(snap_t)
    med_win = max(int(10 / dt_snap), 5)
    t_min = snap_t / 60.0

    print(f'  Computing rates ({nt} snapshots)...')
    all_rxn, all_mt = [], []
    for i in range(nt):
        rr, mt = compute_rates_snapshot(solver, snap_y[i], snap_t[i])
        all_rxn.append(rr)
        all_mt.append(mt)

    conc = {}
    for sp in TARGET_SPECIES:
        total = SPEC_TO_TOTAL.get(sp, sp)
        idx = chem.species_idx.get(total, chem.species_idx.get(sp))
        if idx is not None:
            conc[sp] = np.array([np.dot(snap_y[i][:, idx], dz) / L for i in range(nt)])

    fig, axes = plt.subplots(2, 2, figsize=(11.25, 7.5), sharex=True)
    for pi, sp in enumerate(TARGET_SPECIES):
        ax = axes.flat[pi]
        by_label = defaultdict(lambda: np.zeros(nt))
        for i in range(nt):
            for lab, rate in species_contribution(all_rxn[i], sp, all_mt[i]):
                by_label[lab][i] = rate

        max_total = max((sum(abs(r[i]) for r in by_label.values()) for i in range(nt)), default=1)
        sig_labels = [lab for lab, rates in by_label.items()
                      if np.max(np.abs(rates)) / max(max_total, 1e-30) >= 0.01]

        for lab in sig_labels:
            ax.plot(t_min, median_filter(by_label[lab], size=med_win), lw=1.2, label=lab[:40])

        if sp in conc:
            c = conc[sp]
            dcdt = np.diff(c) / np.diff(snap_t)
            net = np.zeros(nt)
            net[0] = dcdt[0] if len(dcdt) > 0 else 0
            net[-1] = dcdt[-1] if len(dcdt) > 0 else 0
            if nt > 2:
                net[1:-1] = 0.5 * (dcdt[:-1] + dcdt[1:])
            ax.plot(t_min, median_filter(net, size=med_win), 'k--', lw=2, label='dC/dt')

        ax.set_ylabel(f'd[{_uni(sp)}]/dt (M/s)')
        ax.set_title(f'({"abcd"[pi]}) {_uni(sp)}', fontweight='bold', loc='left')
        ax.axhline(0, color='gray', lw=0.5)
        ax.legend(fontsize=6, loc='best')

    for ax in axes[1]:
        ax.set_xlabel('Time (min)')
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(_dir / f'{filename}.png')
    fig.savefig(_dir / f'{filename}.pdf')
    plt.close(fig)
    print(f'  saved: {filename}')


# ═══════════════════════════════════════════════════════════════
# Test 1: max_step=0.01s (force fine stepping)
# ═══════════════════════════════════════════════════════════════
st1, sy1, sol1, _ = run_sim('Test 1: max_step=0.01s', max_step=0.01, dt_snap=2.0)
plot_fig2(st1, sy1, sol1, 2.0,
          'Test 1: max_step=0.01s — rate evolution',
          'fig_test1_maxstep001')

# ═══════════════════════════════════════════════════════════════
# Test 2: 0D (N_z=2, minimal spatial coupling)
# ═══════════════════════════════════════════════════════════════
st2, sy2, sol2, _ = run_sim('Test 2: 0D (N_z=2)', max_step=1.0, dt_snap=0.5, nz=2)
plot_fig2(st2, sy2, sol2, 0.5,
          'Test 2: 0D (2 cells) — rate evolution',
          'fig_test2_0d_cell')

# ═══════════════════════════════════════════════════════════════
# Test 3: dt_snap=0.1s (aliasing check, t=90-200s)
# ═══════════════════════════════════════════════════════════════
# Phase 1: coarse run to t=90s
_, _, sol3pre, res3pre = run_sim('Test 3 phase1: to t=90s',
                                 max_step=1.0, dt_snap=2.0,
                                 t_span_override=(0, 90))
y90 = res3pre['y_final'].ravel()

# Phase 2: fine dt=0.1s from t=90 to t=200
st3, sy3, sol3, _ = run_sim('Test 3 phase2: dt=0.1s (90-200s)',
                             max_step=1.0, dt_snap=0.1,
                             t_span_override=(90, 200),
                             y0_override=y90)
plot_fig2(st3, sy3, sol3, 0.1,
          'Test 3: dt_snap=0.1s (90-200s) — aliasing check',
          'fig_test3_aliasing')

print('\n=== ALL TESTS COMPLETE ===')
