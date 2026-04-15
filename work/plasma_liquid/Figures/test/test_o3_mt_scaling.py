#!/usr/bin/env python3
"""
Test: O3 mass transfer scaling.

Run gas_alpha BC with O3 k_mt scaled by factors [1.0, 0.5, 0.1, 0.01]
to see effect on rate evolution (Fig 2) and mass balance (Fig 4).

Run:
    .venv/bin/python Figures/test/test_o3_mt_scaling.py
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

from config_1d import (
    PHYSICAL, N2O4_EQ, GAS_TO_AQUEOUS_MAP, HENRY_CONSTANTS,
    LIQUID_DIFFUSIVITY, D_LIQ_DEFAULT,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D, _filter_onset

DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

# Config matching gen_all_figures.py
DZ_MIN = 5e-6
STRETCH = 1.12
DELTA_GAS = 0.01       # 10 mm
DT_SNAPSHOT = 2.0

# Ratio-based unmeasured species
HONO_RATIO = 0.33
HONO2_RATIO = 0.83
H2O2_RATIO = 0.03

# O3 MT scaling factors to test
O3_SCALES = [1.0, 0.5, 0.1, 0.01]

# Species for rate evolution
TARGET_SPECIES = ['NO3-', 'O3', 'NO2-', 'H2O2']
SPEC_TO_TOTAL = {
    'HONO': 'HONO_total', 'NO2-': 'HONO_total',
    'HONO2': 'HONO2_total', 'NO3-': 'HONO2_total',
    'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
    'HO2': 'HO2_total', 'O2-': 'HO2_total',
    'ONOOH': 'ONOOH_total', 'ONOO-': 'ONOOH_total',
    'O2NOOH': 'O2NOOH_total', 'O2NOO-': 'O2NOOH_total',
}
EXP = {'pH': 3.61, 'NO3': 63.0, 'NO2': 3.0, 'H2O2': 11.0}


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

    hono_gas = gas_conc['NO2'] * HONO_RATIO
    hono2_gas = gas_conc['N2O5'] * HONO2_RATIO
    h2o2_gas = gas_conc['O3'] * H2O2_RATIO
    return times, gas_conc, hono_gas, hono2_gas, h2o2_gas


def run_case(times, gas_conc, hono_gas, hono2_gas, h2o2_gas, o3_scale, label):
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=DZ_MIN, stretch_ratio=STRETCH,
        saline_mode=False, bc_type='gas_alpha', alpha_b=None,
        delta_gas=DELTA_GAS,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=hono_gas, hono2_gas=hono2_gas, h2o2_gas=h2o2_gas,
    )

    # Scale O3 k_mt
    if o3_scale != 1.0:
        new_iface = []
        for entry in solver._interface_species:
            aq_idx, k_mt, gas_sp, H, Ka = entry
            if gas_sp == 'O3':
                new_iface.append((aq_idx, k_mt * o3_scale, gas_sp, H, Ka))
            else:
                new_iface.append(entry)
        solver._interface_species = new_iface

    t_end = float(times[-1])
    t_eval = np.arange(DT_SNAPSHOT, t_end + 0.1, DT_SNAPSHOT)
    t_eval = t_eval[t_eval <= t_end + 0.1]

    print(f"  [{label}] running...")
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval,
                          verbose=True, dt_poisson=None)
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    print(f"    pH={result['pH_avg']:.3f}, NO3-={avg.get('NO3-',0)*1e6:.1f}µM, "
          f"O3={avg.get('O3',0)*1e9:.1f}nM, H2O2={avg.get('H2O2',0)*1e6:.2f}µM, "
          f"wall={wall:.0f}s")

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
    times, gas_conc, hono_gas, hono2_gas, h2o2_gas = load_gas_data()

    print("=" * 60)
    print("O3 MT scaling test")
    print("=" * 60)

    all_results = {}
    for scale in O3_SCALES:
        label = f"O3×{scale}"
        result, solver = run_case(times, gas_conc, hono_gas, hono2_gas,
                                   h2o2_gas, scale, label)
        all_results[scale] = (result, solver)

    # ---- Fig 2: Rate evolution per scale ----
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 11, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    colors_scale = {1.0: '#1f77b4', 0.5: '#ff7f0e', 0.1: '#2ca02c', 0.01: '#d62728'}

    # Fig 2-style: one subplot per target species, lines = O3 scale
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for si, sp in enumerate(TARGET_SPECIES):
        ax = axes[si]
        for scale in O3_SCALES:
            result, solver = all_results[scale]
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

            ax.plot(np.array(snap_t) / 60, net_rates,
                    color=colors_scale[scale], lw=1.5,
                    label=f'O3×{scale}')

        ax.set_title(f'({chr(97+si)}) {sp}', fontweight='bold')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(f'd[{sp}]/dt (M/s)')
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    fig.suptitle(f'Net rate evolution — O₃ MT scaling (gas_alpha, δg={DELTA_GAS*1e3:.0f}mm)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig_o3_scaling_rates.png')
    fig.savefig(_script_dir / 'fig_o3_scaling_rates.pdf')
    print(f"\n  -> fig_o3_scaling_rates.png/pdf saved")
    plt.close(fig)

    # ---- Fig 4-style: mass balance bar chart per scale ----
    fig, axes = plt.subplots(len(O3_SCALES), len(TARGET_SPECIES),
                              figsize=(5 * len(TARGET_SPECIES), 4 * len(O3_SCALES)))
    if len(O3_SCALES) == 1:
        axes = axes[np.newaxis, :]

    for row, scale in enumerate(O3_SCALES):
        result, solver = all_results[scale]
        snap_t = np.array([0.0] + [float(tv) for tv in result['t_eval']])
        snap_y_list = [solver.build_initial_condition(initial_pH=7.0).reshape(
            solver.N_z, solver.N_s)]
        for yv in result['y_eval']:
            snap_y_list.append(np.array(yv).reshape(solver.N_z, solver.N_s))

        # Use last snapshot for mass balance
        rr, mt = compute_rates_snapshot(solver, snap_y_list[-1], snap_t[-1])

        for col, sp in enumerate(TARGET_SPECIES):
            ax = axes[row, col]
            contribs = species_contribution(rr, sp, mt)
            contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            top = contribs[:6]

            labels_bar = [c[0][:30] for c in top]
            vals = [c[1] for c in top]
            colors_bar = ['#d62728' if v < 0 else '#1f77b4' for v in vals]

            y_pos = range(len(top))
            ax.barh(y_pos, vals, color=colors_bar, height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_bar, fontsize=7)
            ax.invert_yaxis()
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
            if row == 0:
                ax.set_title(sp, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'O3×{scale}', fontweight='bold')

    fig.suptitle(f'Mass balance — O₃ MT scaling (gas_alpha, δg={DELTA_GAS*1e3:.0f}mm)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig_o3_scaling_balance.png')
    fig.savefig(_script_dir / 'fig_o3_scaling_balance.pdf')
    print(f"  -> fig_o3_scaling_balance.png/pdf saved")
    plt.close(fig)

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Scale':>8s}  {'pH':>6s}  {'NO3(µM)':>8s}  {'O3(nM)':>8s}  {'H2O2(µM)':>9s}  {'OH(pM)':>7s}")
    print("-" * 55)
    for scale in O3_SCALES:
        result, _ = all_results[scale]
        avg = result['spatial_avg']
        print(f"O3×{scale:<4.2f}  {result['pH_avg']:6.3f}  "
              f"{avg.get('NO3-',0)*1e6:8.1f}  {avg.get('O3',0)*1e9:8.1f}  "
              f"{avg.get('H2O2',0)*1e6:9.2f}  {avg.get('OH',0)*1e12:7.2f}")
    print(f"{'Exp':>8s}  {3.61:6.2f}  {63.0:8.1f}  {'–':>8s}  {11.0:9.2f}  {'–':>7s}")


if __name__ == '__main__':
    main()
