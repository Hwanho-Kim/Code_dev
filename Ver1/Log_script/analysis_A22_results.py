"""A22 fix result analysis: conversion, reaction contributions, plots."""
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, T_STP, P_STP, total_concentration

EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
INLET = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}
P_W = 5.0
Q_SLM = 0.4
V_EFF_CM3 = 1.6
V_REACTOR_CM3 = 250.0

temps_C = []
conv_sim = []
conv_exp = []
ne_list = []
Te_list = []
Tg_list = []
eps_list = []

all_contributions = {}  # {T_K: {label: fraction}}
rxn_labels_global = []

for T_K in EXP_DATA:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))
    cfg['V_eff'] = V_EFF_CM3 * 1e-6
    cfg['reactor']['volume'] = V_REACTOR_CM3 * 1e-6
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_SLM
    cfg['T_wall'] = T_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_K
    cfg['inlet_composition'] = INLET

    Q_actual = Q_SLM * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau = V_REACTOR_CM3 * 1e-6 / Q_actual
    t_end = max(10.0, 5.0 * tau)
    cfg['solver'] = {'t_end': t_end, 'n_points': 150, 'method': 'BDF',
                     'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False}

    with contextlib.redirect_stdout(io.StringIO()):
        solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)
        scfg = cfg_out['solver']
        result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                              rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    sm = solver.sm
    rxn_set = solver.rxn
    n_sp = sm.n_species
    ch4_idx = sm.index('CH4')
    stoich = rxn_set.stoich_matrix

    # Steady-state values
    y_ss = result.y[:, -1]
    c = np.maximum(y_ss[:n_sp], 1e-30)
    ne_eps = y_ss[sm.idx_energy]
    T_gas = y_ss[sm.idx_Tgas]
    c_e = c[0]
    n_e = c_e * NA
    eps_thermal = 1.5 * KB * max(T_gas, 200) / QE
    eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0) if n_e > 1 else max(1.0, eps_thermal)
    Te_eV = (2.0 / 3.0) * eps_mean

    # Conversion
    c0 = result.concentrations[ch4_idx, 0]
    cf = result.concentrations[ch4_idx, -1]
    conv = (c0 - cf) / c0 * 100 if c0 > 0 else 0

    temps_C.append(T_K - 273)
    conv_sim.append(conv)
    conv_exp.append(EXP_DATA[T_K])
    ne_list.append(n_e)
    Te_list.append(Te_eV)
    Tg_list.append(T_gas)
    eps_list.append(eps_mean)

    # Reaction rates at steady state
    c_total = total_concentration(101325.0, T_gas)
    k_ei_conc = None
    if solver.lut is not None and eps_mean >= solver._eps_min_lut:
        k_ei_conc, _ = solver.lut.get_rate_coefficients_conc(eps_mean)
    rates = rxn_set.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                           Te_eV=Te_eV, P_gas=101325.0)

    # CH4 consumption by each reaction: -stoich[ch4, j] * rate[j]
    ch4_consumers = {}
    for j, r in enumerate(rxn_set.reactions):
        nu = stoich[ch4_idx, j]
        if nu < 0:
            consumption = -nu * rates[j]  # mol/(m3·s), positive = consumption
            label = f"R{r.id}" if hasattr(r, 'id') else f"R{j+1}"
            ch4_consumers[label] = consumption

    total_consumption = sum(ch4_consumers.values())

    contributions = {}
    for label, cons in ch4_consumers.items():
        frac = cons / total_consumption * 100 if total_consumption > 0 else 0
        contributions[label] = frac

    all_contributions[T_K] = contributions

    if not rxn_labels_global:
        for j, r in enumerate(rxn_set.reactions):
            nu = stoich[ch4_idx, j]
            if nu < 0:
                lbl = f"R{r.id}" if hasattr(r, 'id') else f"R{j+1}"
                formula = r.formula if hasattr(r, 'formula') else ''
                rxn_labels_global.append((lbl, formula))

# --- Write report ---
report_path = '/home/hawn/work/A22_fix_results_0312.txt'
with open(report_path, 'w') as f:
    f.write('='*80 + '\n')
    f.write('A22 Fix Results — CH4 Conversion Comparison\n')
    f.write(f'Date: 2025-03-12\n')
    f.write(f'Conditions: P={P_W}W, Q={Q_SLM}slm, V_eff={V_EFF_CM3}cm3, V_reactor={V_REACTOR_CM3}cm3\n')
    f.write(f'Inlet: N2={INLET["N2"]}, O2={INLET["O2"]}, CO2={INLET["CO2"]}, CH4={INLET["CH4"]}\n')
    f.write('='*80 + '\n\n')

    f.write('--- CH4 Conversion ---\n')
    f.write(f'{"T(C)":>6} {"Exp(%)":>8} {"Sim(%)":>8} {"D(%p)":>8}\n')
    f.write('-'*35 + '\n')
    for i, T_K in enumerate(EXP_DATA):
        delta = conv_sim[i] - conv_exp[i]
        f.write(f'{temps_C[i]:6d} {conv_exp[i]:8.2f} {conv_sim[i]:8.2f} {delta:+8.2f}\n')

    f.write('\n--- Plasma Parameters (steady state) ---\n')
    f.write(f'{"T(C)":>6} {"n_e(m-3)":>12} {"Te(eV)":>8} {"eps(eV)":>8} {"Tg(K)":>8}\n')
    f.write('-'*47 + '\n')
    for i, T_K in enumerate(EXP_DATA):
        f.write(f'{temps_C[i]:6d} {ne_list[i]:12.3e} {Te_list[i]:8.3f} {eps_list[i]:8.3f} {Tg_list[i]:8.1f}\n')

    f.write('\n--- CH4 Consumption Reaction Contributions (%) ---\n')
    header = f'{"Rxn":<6} {"Formula":<40}'
    for tc in temps_C:
        header += f' {tc:>5d}C'
    f.write(header + '\n')
    f.write('-' * (46 + 7 * len(temps_C)) + '\n')
    for lbl, formula in rxn_labels_global:
        line = f'{lbl:<6} {formula:<40}'
        for T_K in EXP_DATA:
            val = all_contributions[T_K].get(lbl, 0.0)
            line += f' {val:6.1f}'
        f.write(line + '\n')

print(f'Report saved: {report_path}')

# --- Plot 1: Exp vs Sim conversion ---
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(temps_C, conv_exp, 'ko-', linewidth=2, markersize=8, label='Experiment')
ax1.plot(temps_C, conv_sim, 'rs--', linewidth=2, markersize=8, label='Simulation')
ax1.set_xlabel('Temperature (°C)', fontsize=13)
ax1.set_ylabel('CH$_4$ Conversion (%)', fontsize=13)
ax1.set_title('CH$_4$ Conversion: Experiment vs Simulation', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(20, 260)
ax1.set_ylim(0, max(max(conv_exp), max(conv_sim)) * 1.15)
fig1.tight_layout()
plot1_path = '/home/hawn/work/plot_conversion_exp_vs_sim.png'
fig1.savefig(plot1_path, dpi=200)
print(f'Plot 1 saved: {plot1_path}')

# --- Plot 2: Reaction contributions ---
# Filter out reactions with <1% contribution at all temperatures
significant = []
for lbl, formula in rxn_labels_global:
    max_frac = max(all_contributions[T_K].get(lbl, 0.0) for T_K in EXP_DATA)
    if max_frac >= 1.0:
        significant.append((lbl, formula))

fig2, ax2 = plt.subplots(figsize=(9, 6))
colors = plt.cm.tab20(np.linspace(0, 1, max(len(significant), 1)))
for i, (lbl, formula) in enumerate(significant):
    fracs = [all_contributions[T_K].get(lbl, 0.0) for T_K in EXP_DATA]
    ax2.plot(temps_C, fracs, 'o-', color=colors[i], linewidth=1.8, markersize=6, label=lbl)

ax2.set_xlabel('Temperature (°C)', fontsize=13)
ax2.set_ylabel('Contribution to CH$_4$ Consumption (%)', fontsize=13)
ax2.set_title('Reaction Contributions to CH$_4$ Conversion', fontsize=14)
ax2.legend(fontsize=9, ncol=2, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(20, 260)
ax2.set_ylim(0, 105)
fig2.tight_layout()
plot2_path = '/home/hawn/work/plot_reaction_contributions.png'
fig2.savefig(plot2_path, dpi=200)
print(f'Plot 2 saved: {plot2_path}')
