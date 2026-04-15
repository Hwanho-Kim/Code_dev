#!/usr/bin/env python3
"""
Sweep unmeasured gas-phase species one at a time (others = 0).

13 cases + 1 baseline (all zero):
  HONO/NO₂  = 0.01, 0.1, 0.33, 1.0        (4 cases, HNO₃=0, H₂O₂=0)
  HNO₃/N₂O₅ = 0.02, 0.1, 0.5, 0.83, 2.0   (5 cases, HONO=0, H₂O₂=0)
  H₂O₂/O₃  = 0.001, 0.01, 0.03, 0.1       (4 cases, HONO=0, HNO₃=0)

BC: gas_alpha, species-specific α_b, δ_gas=10mm

Output:
  fig_sweep_rons.png  — RONS concentration summary (single chart)
  fig_sweep_fig2_*.png — Rate evolution per sweep group

Run:
    .venv/bin/python Figures/test/test_gas_sweep.py
"""

import sys
import time as time_mod
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ, GAS_TO_AQUEOUS_MAP
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

DZ_MIN = 5e-6
STRETCH = 1.12
DELTA_GAS = 0.01
DT_SNAPSHOT = 2.0

# Experimental targets
EXP = {'pH': 3.61, 'NO3-': 63.0, 'NO2-': 3.0, 'H2O2': 11.0}

# Sweep definitions: (label, hono_ratio, hno3_ratio, h2o2_ratio)
CASES = [
    # Baseline
    ('Baseline (dry)',     0,    0,    0),
    # HONO sweep
    ('HONO/NO₂=0.01',     0.01, 0,    0),
    ('HONO/NO₂=0.1',      0.1,  0,    0),
    ('HONO/NO₂=0.33',     0.33, 0,    0),
    ('HONO/NO₂=1.0',      1.0,  0,    0),
    # HNO3 sweep
    ('HNO₃/N₂O₅=0.02',   0,    0.02, 0),
    ('HNO₃/N₂O₅=0.1',    0,    0.1,  0),
    ('HNO₃/N₂O₅=0.5',    0,    0.5,  0),
    ('HNO₃/N₂O₅=0.83',   0,    0.83, 0),
    ('HNO₃/N₂O₅=2.0',    0,    2.0,  0),
    # H2O2 sweep
    ('H₂O₂/O₃=0.001',    0,    0,    0.001),
    ('H₂O₂/O₃=0.01',     0,    0,    0.01),
    ('H₂O₂/O₃=0.03',     0,    0,    0.03),
    ('H₂O₂/O₃=0.1',      0,    0,    0.1),
]

SPEC_TO_TOTAL = {
    'HONO': 'HONO_total', 'NO2-': 'HONO_total',
    'HONO2': 'HONO2_total', 'NO3-': 'HONO2_total',
    'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
    'HO2': 'HO2_total', 'O2-': 'HO2_total',
    'ONOOH': 'ONOOH_total', 'ONOO-': 'ONOOH_total',
    'O2NOOH': 'O2NOOH_total', 'O2NOO-': 'O2NOOH_total',
}
TARGET_SPECIES = ['NO3-', 'O3', 'NO2-', 'H2O2']


def load_gas_data():
    df = pd.read_csv(DEFAULT_CSV)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))
    if np.all(gas_conc.get('N2O4', np.array([0])) == 0):
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(math.log(N2O4_EQ.KP_298) +
                       (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)
    return times, gas_conc


def run_case(times, gas_conc, label, hono_r, hno3_r, h2o2_r):
    hono_gas = gas_conc['NO2'] * hono_r if hono_r > 0 else 0.0
    hno3_gas = gas_conc['N2O5'] * hno3_r if hno3_r > 0 else 0.0
    h2o2_gas = gas_conc['O3'] * h2o2_r if h2o2_r > 0 else 0.0

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type='gas_alpha', alpha_b=None,
        delta_gas=DELTA_GAS,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=hono_gas, hono2_gas=hno3_gas,
                        h2o2_gas=h2o2_gas)

    t_end = float(times[-1])
    t_eval = np.arange(DT_SNAPSHOT, t_end + 0.1, DT_SNAPSHOT)
    t_eval = t_eval[t_eval <= t_end + 0.1]

    print(f"  [{label}] running...")
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    print(f"    pH={result['pH_avg']:.3f}, NO3-={avg.get('NO3-',0)*1e6:.1f}, "
          f"NO2-={avg.get('NO2-',0)*1e6:.3f}, H2O2={avg.get('H2O2',0)*1e6:.2f}, "
          f"O3={avg.get('O3',0)*1e9:.1f}nM, wall={wall:.0f}s")

    return result, solver


def compute_rates_snapshot(solver, y_2d, t):
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


def main():
    times, gas_conc = load_gas_data()

    print("=" * 60)
    print("Gas-phase sweep: HONO / HNO₃ / H₂O₂ (one at a time)")
    print("=" * 60)

    all_results = []
    for label, hr, hn, hp in CASES:
        result, solver = run_case(times, gas_conc, label, hr, hn, hp)
        all_results.append((label, hr, hn, hp, result, solver))

    # ================================================================
    # RONS concentration summary — single figure
    # ================================================================
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.labelsize': 10, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    labels = [c[0] for c in all_results]
    n = len(labels)
    metrics = ['pH', 'NO₃⁻ (µM)', 'NO₂⁻ (µM)', 'H₂O₂ (µM)', 'O₃ (nM)']
    exp_vals = [EXP['pH'], EXP['NO3-'], EXP['NO2-'], EXP['H2O2'], None]

    vals = {m: [] for m in metrics}
    for _, _, _, _, result, _ in all_results:
        avg = result['spatial_avg']
        vals['pH'].append(result['pH_avg'])
        vals['NO₃⁻ (µM)'].append(avg.get('NO3-', 0) * 1e6)
        vals['NO₂⁻ (µM)'].append(avg.get('NO2-', 0) * 1e6)
        vals['H₂O₂ (µM)'].append(avg.get('H2O2', 0) * 1e6)
        vals['O₃ (nM)'].append(avg.get('O3', 0) * 1e9)

    # Color by group
    colors = []
    for _, hr, hn, hp, _, _ in all_results:
        if hr == 0 and hn == 0 and hp == 0:
            colors.append('#333333')
        elif hr > 0:
            colors.append('#9467bd')  # HONO purple
        elif hn > 0:
            colors.append('#8c564b')  # HNO3 brown
        else:
            colors.append('#e377c2')  # H2O2 pink

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 2.5 * len(metrics)),
                              sharex=True)
    x = np.arange(n)

    for i, (metric, exp_v) in enumerate(zip(metrics, exp_vals)):
        ax = axes[i]
        ax.bar(x, vals[metric], color=colors, width=0.7)
        if exp_v is not None:
            ax.axhline(y=exp_v, color='red', ls='--', lw=1.5, label=f'Exp={exp_v}')
            ax.legend(fontsize=8)
        ax.set_ylabel(metric)
        # Value labels
        for j, v in enumerate(vals[metric]):
            if v > 0.01:
                ax.text(j, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    fig.suptitle('Gas-phase sweep: RONS concentration (gas_alpha, δg=10mm)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig_sweep_rons.png')
    fig.savefig(_script_dir / 'fig_sweep_rons.pdf')
    print(f"\n  -> fig_sweep_rons.png/pdf saved")
    plt.close(fig)

    # ================================================================
    # Rate evolution (Fig 2 style) — one figure per sweep group
    # ================================================================
    groups = [
        ('HONO', [0, 1, 2, 3, 4]),
        ('HNO3', [0, 5, 6, 7, 8, 9]),
        ('H2O2', [0, 10, 11, 12, 13]),
    ]
    group_colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))

    for gname, indices in groups:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.ravel()

        for si, sp in enumerate(TARGET_SPECIES):
            ax = axes_flat[si]
            for ci, idx in enumerate(indices):
                label, _, _, _, result, solver = all_results[idx]
                snap_t = np.array([0.0] + [float(tv) for tv in result['t_eval']])
                snap_y_list = [solver.build_initial_condition(initial_pH=7.0).reshape(
                    solver.N_z, solver.N_s)]
                for yv in result['y_eval']:
                    snap_y_list.append(np.array(yv).reshape(solver.N_z, solver.N_s))

                net_rates = []
                for k in range(len(snap_t)):
                    rr, mt = compute_rates_snapshot(solver, snap_y_list[k], snap_t[k])
                    contribs = species_contribution(rr, sp, mt)
                    net_rates.append(sum(v for _, v in contribs))

                lw = 2.0 if idx == 0 else 1.2
                ls = '--' if idx == 0 else '-'
                ax.plot(np.array(snap_t) / 60, net_rates,
                        color=group_colors[ci], lw=lw, ls=ls, label=label)

            ax.set_title(f'{sp}', fontweight='bold')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(f'd[{sp}]/dt (M/s)')
            ax.legend(fontsize=7, loc='best')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        fig.suptitle(f'Rate evolution — {gname} sweep (gas_alpha, δg=10mm)',
                     fontsize=13)
        fig.tight_layout()
        fname = f'fig_sweep_fig2_{gname.lower()}'
        fig.savefig(_script_dir / f'{fname}.png')
        fig.savefig(_script_dir / f'{fname}.pdf')
        print(f"  -> {fname}.png/pdf saved")
        plt.close(fig)

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Case':25s} {'pH':>6s} {'NO3(µM)':>8s} {'NO2(µM)':>8s} "
          f"{'H2O2(µM)':>9s} {'O3(nM)':>8s}")
    print("-" * 70)
    for label, _, _, _, result, _ in all_results:
        avg = result['spatial_avg']
        print(f"{label:25s} {result['pH_avg']:6.3f} "
              f"{avg.get('NO3-',0)*1e6:8.1f} {avg.get('NO2-',0)*1e6:8.4f} "
              f"{avg.get('H2O2',0)*1e6:9.4f} {avg.get('O3',0)*1e9:8.1f}")
    print(f"{'Experiment':25s} {3.61:6.2f} {63.0:8.1f} {3.0:8.4f} {11.0:9.4f} {'–':>8s}")


if __name__ == '__main__':
    main()
