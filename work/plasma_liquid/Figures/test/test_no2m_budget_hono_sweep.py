#!/usr/bin/env python3
"""NO2- rate budget analysis for HONO/NO2 sweep at 3.2 kV.

HONO/NO2 ∈ {0.007, 0.03, 0.07, 0.1} — 4 sims.
각 sim에서 t=600s 시점의 NO2- 관련 반응 rate 추출.
  - 생성: R19, R82, R94, R95 + HONO_total mass transfer × (Ka/(H++Ka))
  - 소비: R32, R92
  - Net:  d[NO2-]/dt total

출력:
  1) 반응별 bulk-avg rate [M/s] 표
  2) Bar chart: HONO ratio별 source vs sink
"""
import sys, functools, time as time_mod
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))

from config_1d import PHYSICAL, N2O4_EQ, ACID_BASE_PAIRS
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

print = functools.partial(print, flush=True)

GAS_XLSX = _root / 'OAS data' / 'Dry' / '(P-L) 가스활성종 농도.xlsx'
RH80 = {'O3_scale': 0.647, 'NO2_O3': 0.091, 'N2O5_NO2': 0.054, 'NO3_O3': 0.00442}
VOLTAGE = '3.2kV'
HONO_RATIOS = [0.007, 0.030, 0.070, 0.100]
HONO2_RATIO_FIXED = 0.83
H2O2_RATIO_FIXED = 0.003
KA_HONO = 10 ** (-ACID_BASE_PAIRS['HONO_total'][2])   # 10^-3.4

# NO2- 관련 반응 classification
NO2M_SOURCE_LABELS = {
    'R19': '2NO2 → NO2⁻ + NO3⁻ + 2H⁺',
    'R82': 'N2O3 + ONOO⁻ → 2NO2 + NO2⁻',
    'R94': 'N2O3 + H2O → 2NO2⁻ + H⁺',
    'R95': 'N2O4 + H2O → NO2⁻ + NO3⁻ + 2H⁺',
    'MT_HONO': 'HONO(g) uptake × f_ion',
}
NO2M_SINK_LABELS = {
    'R32': 'O3 + NO2⁻ → O2 + NO3⁻',
    'R92': 'NO3 + NO2⁻ → NO3⁻ + NO2',
}


def load_gas(voltage):
    df = pd.read_excel(GAS_XLSX, sheet_name=voltage)
    times = df.iloc[:, 0].values.astype(float)
    gas = {}
    for sp in ['O3', 'NO2', 'NO3', 'N2O5']:
        for c in df.columns:
            if sp in str(c):
                arr = df[c].values.astype(float)
                arr[arr < 0] = 0
                # onset filter
                cnt, si = 0, len(arr)
                for i in range(len(arr)):
                    if arr[i] > 0:
                        cnt += 1
                        if cnt >= 5:
                            si = i - 4
                            break
                    else:
                        cnt = 0
                arr[:si] = 0
                if si < len(arr):
                    nz = np.nonzero(arr)[0]
                    if len(nz) > 1:
                        arr = np.interp(np.arange(len(arr)), nz, arr[nz])
                        arr[:si] = np.linspace(0, arr[si], si + 1)[:-1]
                gas[sp] = arr
                break
    return times, gas


def apply_rh80(gas_dry, times, hono_ratio):
    r = RH80
    mask = times >= (times[-1] - 100)
    ss = lambda a: max(np.mean(a[mask]), 1e-30)
    o3d, no2d = ss(gas_dry['O3']), ss(gas_dry['NO2'])
    n2o5d, no3d = ss(gas_dry['N2O5']), ss(gas_dry['NO3'])
    o3_80 = o3d * r['O3_scale']
    no2_80 = o3_80 * r['NO2_O3']
    n2o5_80 = no2_80 * r['N2O5_NO2']
    no3_80 = o3_80 * r['NO3_O3']
    g = {
        'O3':   gas_dry['O3']   * (o3_80 / o3d),
        'NO2':  gas_dry['NO2']  * (no2_80 / no2d),
        'N2O5': gas_dry['N2O5'] * (n2o5_80 / n2o5d),
        'NO3':  gas_dry['NO3']  * (no3_80 / no3d),
    }
    T = N2O4_EQ.REF_TEMP
    Kp = np.exp(np.log(N2O4_EQ.KP_298)
                + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / T - 1 / T))
    g['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (g['NO2'] ** 2)
    hono = g['NO2'] * hono_ratio
    hno3 = g['N2O5'] * HONO2_RATIO_FIXED
    h2o2 = g['O3'] * H2O2_RATIO_FIXED
    return g, hono, hno3, h2o2


def run_and_analyze(hono_ratio):
    """Run 3.2kV sim at given HONO ratio and extract NO2- rate budget."""
    times, gas_dry = load_gas(VOLTAGE)
    gas, hono, hno3, h2o2 = apply_rh80(gas_dry, times, hono_ratio)

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type='three_film', alpha_b=None,
        delta_gas=0.01, delta_liq=1e-4,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas,
                        hono_gas=hono, hono2_gas=hno3, h2o2_gas=h2o2)
    t_end = float(times[-1])
    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0

    # Final snapshot = y at last time
    y_final = result['y_final']          # (N_z, N_s)
    N_z = solver.N_z
    dz = solver.dz_cells
    L = solver.L

    # Per-cell reaction rates (full reaction list) at t_end
    rates_per_cell = np.zeros((len(chem.reactions), N_z))
    for j in range(N_z):
        yc = np.clip(y_final[j, :].copy(), chem.trace, 1.0)
        hp_idx = solver._h_plus_idx
        if hp_idx >= 0:
            yc[hp_idx] = max(yc[hp_idx], 1e-14)
        spec = chem.speciate(yc)
        for ri, rxn_d in enumerate(chem._rxn_data):
            rates_per_cell[ri, j] = chem._compute_single_rate(rxn_d, yc, spec)

    # Bulk volume-averaged rates [M/s]
    rate_avg = np.dot(rates_per_cell, dz) / L

    # Identify target reactions by label
    rxn_labels = [rxn.get('label', f'R{ri}')
                  for ri, rxn in enumerate(chem.reactions)]

    def find_rxn(tag):
        for ri, lab in enumerate(rxn_labels):
            if tag in lab:
                return ri
        return -1

    budget = {}
    # Sources (stoichiometry of NO2- in products)
    # R19: 2NO2 → NO2-+NO3-+2H+ (stoich 1)
    # R82: N2O3+ONOO- → 2NO2+NO2- (stoich 1)
    # R94: N2O3+H2O → 2NO2-+H+ (stoich 2)
    # R95: N2O4+H2O → NO2-+NO3-+2H+ (stoich 1)
    for tag, stoich in [('R19', 1), ('R82', 1), ('R94', 2), ('R95', 1)]:
        ri = find_rxn(tag + ':')
        if ri < 0:
            ri = find_rxn(tag)
        budget[tag] = stoich * rate_avg[ri] if ri >= 0 else 0.0

    # Sinks (stoichiometry of NO2- in reactants; rate is positive if forward)
    for tag, stoich in [('R32', 1), ('R92', 1)]:
        ri = find_rxn(tag + ':')
        if ri < 0:
            ri = find_rxn(tag)
        budget[tag] = -stoich * rate_avg[ri] if ri >= 0 else 0.0

    # Mass-transfer HONO contribution
    # J_HONO into HONO_total, then × f_ion = Ka/(H++Ka) at bulk pH
    # For simplicity use bulk-avg H+ and bulk-avg HONO_total, flux at z=0
    hp_idx = solver._h_plus_idx
    hono_tot_idx = chem.species_idx.get('HONO_total', -1)
    # Bulk averages
    hp_bulk = np.dot(y_final[:, hp_idx], dz) / L if hp_idx >= 0 else 1e-7
    # Surface flux at z=0
    for aq_idx, k_mt, gas_sp, H, Ka in solver._interface_species:
        if gas_sp == 'HONO':
            C_eq = solver._get_C_eq_fast(gas_sp, t_end)
            c0 = y_final[0, aq_idx]
            h_surf = y_final[0, hp_idx] if hp_idx >= 0 else 1e-7
            c_eff = c0 * h_surf / (h_surf + Ka) if Ka is not None else c0
            flux_hono = k_mt * (C_eq - c_eff)  # [M · m/s]
            # bulk production rate (M/s) = flux / L
            flux_per_L = flux_hono / L
            # f_ion at bulk pH
            f_ion = KA_HONO / (hp_bulk + KA_HONO)
            budget['MT_HONO'] = flux_per_L * f_ion
            break

    # Speciate NO2- from HONO_total at bulk
    hono_tot_bulk = np.dot(y_final[:, hono_tot_idx], dz) / L
    no2m_bulk = hono_tot_bulk * KA_HONO / (hp_bulk + KA_HONO)
    pH_bulk = -np.log10(max(hp_bulk, 1e-14))

    print(f'[HONO={hono_ratio:.3f}] pH={pH_bulk:.3f}, '
          f'NO2-={no2m_bulk*1e6:.3f} µM, H+={hp_bulk:.3e}, '
          f'HONO_tot={hono_tot_bulk*1e6:.3f} µM, wall={wall:.0f}s')

    return {
        'hono_ratio': hono_ratio,
        'pH': pH_bulk,
        'NO2-': no2m_bulk * 1e6,        # µM
        'HONO_total': hono_tot_bulk * 1e6,
        'budget': budget,                # [M/s] per reaction (signed)
    }


def print_budget_table(results):
    print('\n' + '=' * 110)
    print(f'NO₂⁻ rate budget at t=600s — 3.2 kV, three_film, varying HONO/NO₂')
    print('=' * 110)

    rxns_src = ['R19', 'R82', 'R94', 'R95', 'MT_HONO']
    rxns_sink = ['R32', 'R92']

    # Header
    hdr = f"{'Reaction':<10} | {'Description':<40} |"
    for r in results:
        tag_str = f"HONO={r['hono_ratio']:.3f}"
        hdr += f" {tag_str:>12} |"
    print(hdr)
    print('-' * 110)

    print('SOURCES (µM/s, positive = production):')
    for tag in rxns_src:
        label = NO2M_SOURCE_LABELS.get(tag, '')
        row = f"{tag:<10} | {label:<40} |"
        for r in results:
            val = r['budget'].get(tag, 0.0) * 1e6   # M/s → µM/s
            row += f" {val:12.4f} |"
        print(row)

    print('\nSINKS (µM/s, positive = consumption):')
    for tag in rxns_sink:
        label = NO2M_SINK_LABELS.get(tag, '')
        row = f"{tag:<10} | {label:<40} |"
        for r in results:
            val = -r['budget'].get(tag, 0.0) * 1e6  # negate for positive display
            row += f" {val:12.4f} |"
        print(row)

    print('\nTotals:')
    for label, rxns, sign in [('Total source', rxns_src, 1),
                                ('Total sink', rxns_sink, -1)]:
        row = f"{label:<52}  |"
        for r in results:
            s = sign * sum(r['budget'].get(tag, 0.0) for tag in rxns) * 1e6
            row += f" {s:12.4f} |"
        print(row)
    row = f"{'Net (source − sink)':<52}  |"
    for r in results:
        net = sum(r['budget'].values()) * 1e6
        row += f" {net:12.4f} |"
    print(row)

    print('\nFinal [NO₂⁻] (µM):')
    row = ' ' * 52 + '  |'
    for r in results:
        row += f" {r['NO2-']:12.3f} |"
    print(row)
    print('Final pH:')
    row = ' ' * 52 + '  |'
    for r in results:
        row += f" {r['pH']:12.3f} |"
    print(row)


def plot_budget(results):
    """Stacked bar: source contributions + sink contributions per HONO ratio."""
    rxns_src = ['R19', 'R82', 'R94', 'R95', 'MT_HONO']
    rxns_sink = ['R32', 'R92']
    colors_src = {'R19': '#d95f0e', 'R82': '#fdae6b', 'R94': '#e6550d',
                  'R95': '#fdd0a2', 'MT_HONO': '#fe9929'}
    colors_sink = {'R32': '#3182bd', 'R92': '#6baed6'}

    n = len(results)
    xs = np.arange(n)
    x_labels = [f'{r["hono_ratio"]:.3f}' for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: stacked bars for source and sink (µM/s)
    ax = axes[0]
    # sources (positive)
    bottoms = np.zeros(n)
    for tag in rxns_src:
        vals = np.array([r['budget'].get(tag, 0.0) * 1e6 for r in results])
        vals_pos = np.maximum(vals, 0)
        ax.bar(xs - 0.2, vals_pos, 0.4, bottom=bottoms,
               label=f'{tag} (src)', color=colors_src[tag],
               edgecolor='black', lw=0.5)
        bottoms += vals_pos

    # sinks (positive for display)
    bottoms = np.zeros(n)
    for tag in rxns_sink:
        vals = np.array([-r['budget'].get(tag, 0.0) * 1e6 for r in results])
        vals_pos = np.maximum(vals, 0)
        ax.bar(xs + 0.2, vals_pos, 0.4, bottom=bottoms,
               label=f'{tag} (sink)', color=colors_sink[tag],
               edgecolor='black', lw=0.5)
        bottoms += vals_pos

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('HONO/NO₂ ratio')
    ax.set_ylabel('Rate (µM/s)')
    ax.set_title('NO₂⁻ source & sink rates (bulk-avg at t=600s)')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)

    # Right: NO2- concentration vs HONO ratio (log-log to show power law)
    ax = axes[1]
    no2m = np.array([r['NO2-'] for r in results])
    ratios_num = np.array([r['hono_ratio'] for r in results])
    ax.plot(ratios_num, no2m, 'o-', color='#1f77b4', lw=2, ms=8,
            label='[NO₂⁻] bulk (sim)')
    ax.axhline(3.58, color='k', ls='--', lw=1.5, label='Exp = 3.58 µM')
    ax.set_xlabel('HONO/NO₂ ratio')
    ax.set_ylabel('[NO₂⁻] (µM)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('[NO₂⁻] bulk vs HONO ratio (log-log)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    # Compute log-log slope
    log_ratio = np.log10(ratios_num)
    log_no2 = np.log10(no2m)
    slope, intercept = np.polyfit(log_ratio, log_no2, 1)
    ax.text(0.05, 0.95,
            f'power-law slope ≈ {slope:.2f}\n(linear=1)',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    fig.suptitle(
        'NO₂⁻ rate budget analysis — 3.2 kV Humid, three_film, HONO sweep',
        fontsize=13, weight='bold', y=1.02,
    )
    fig.tight_layout()

    out_dir = Path(__file__).parent
    for ext in ('png', 'pdf'):
        p = out_dir / f'no2m_rate_budget_hono_sweep.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {p}')


if __name__ == '__main__':
    results = []
    for hono in HONO_RATIOS:
        print(f'\nRunning HONO={hono:.3f}...')
        r = run_and_analyze(hono)
        results.append(r)

    print_budget_table(results)
    plot_budget(results)
