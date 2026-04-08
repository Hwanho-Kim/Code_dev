"""Electron energy & density budget diagnostic at each T_gas.

Breaks down:
  (A) Electron ENERGY equation: d(ne*eps)/dt = P_dep - P_el - P_inel - P_diff - P_flow - P_eloss
  (B) Electron DENSITY equation: dc_e/dt = S_ioniz - S_diff - S_DR_AT - S_flow

Identifies which terms dominate and why Te drops with T_gas.
"""
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP, ME

# === Fixed conditions (same as sweep_tgas.py) ===
V_eff_cm3 = 1.6
V_reactor_cm3 = 100
P_W = 5.0
Q_slm = 0.4

T_gas_list_K = [300, 350, 400, 450, 500, 523]


def diagnose(T_target_K):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    V_eff_m3 = V_eff_cm3 * 1e-6
    V_reactor_m3 = V_reactor_cm3 * 1e-6

    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm
    cfg['T_wall'] = T_target_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_target_K

    Q_actual = Q_slm * (T_target_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau_est = V_reactor_m3 / Q_actual
    t_end = min(max(3.0, 1.5 * tau_est), 15.0)

    cfg['solver'] = {
        't_end': t_end, 'n_points': 100, 'method': 'BDF',
        'rtol': 1e-5, 'atol': 1e-10, 'max_step': 5e-4, 'constrained': False
    }

    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
    scfg = cfg['solver']
    result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                          rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    # --- Extract final state ---
    sm = solver.sm; rxn = solver.rxn; n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    T_gas = y[sm.idx_Tgas]
    ne_eps = y[sm.idx_energy]
    c_e = c[0]; n_e = c_e * NA
    T_gs = max(T_gas, 200.0)
    eps_thermal = 1.5 * KB * T_gs / QE
    eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0) if n_e > 1 else max(1.0, eps_thermal)
    Te_eV = (2.0 / 3.0) * eps_mean
    N_gas = 101325.0 / (KB * T_gs)
    c_total = 101325.0 / (R_GAS * T_gs)
    tau = solver.flow.get_residence_time(T_gas)

    # --- (A) Electron ENERGY budget ---
    # P_dep
    P_dep_Wm3 = P_W / V_eff_m3  # constant power mode
    P_dep_eV = P_dep_Wm3 / QE

    # LUT query
    k_ei_conc = None
    P_el_eVm3s = 0.0
    if solver.lut and eps_mean >= solver.lut.eps_range[0]:
        k_ei_conc, Te_eV_lut = solver.lut.get_rate_coefficients_conc(eps_mean)
        transport = solver.lut.get_transport(eps_mean)
        A21 = transport.elastic_power_N
        P_el_eVm3s = n_e * N_gas * A21

    # Reaction rates
    rates = rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                        Te_eV=Te_eV, P_gas=101325.0)

    # P_inel
    P_inel_Wm3 = rxn.compute_electron_energy_loss(rates)
    P_inel_eV = P_inel_Wm3 / QE

    # P_diff = ne_eps * D_a / Lambda^2
    Lambda = solver.ekin.Lambda
    mu_i_N = 2.8e22
    mu_i = mu_i_N / N_gas
    D_a = mu_i * Te_eV
    P_diff_eV = ne_eps * D_a / (Lambda * Lambda)

    # P_flow = ne_eps / tau
    P_flow_eV = ne_eps / tau if tau > 0 else 0.0

    # P_e_loss = eps_mean * S_e_loss * NA
    S_e_loss = rxn.compute_electron_loss_rate(rates)
    P_eloss_eV = eps_mean * S_e_loss * NA

    # Total RHS
    energy_rhs = P_dep_eV - P_el_eVm3s - P_inel_eV - P_diff_eV - P_flow_eV - P_eloss_eV

    # --- (B) Electron DENSITY budget ---
    S_electron, S_thermal = rxn.compute_source_terms_split(rates)
    S_flow = solver.flow.compute_flow_source(c, T_gas)

    # Electron source from EI reactions (ionization) 
    dc_e_EI = S_electron[0]  # net electron production from EI reactions [mol/(m³·s)]
    dc_e_thermal = S_thermal[0]  # from Arrhenius (should be ~0 for electrons)
    dc_e_flow = S_flow[0]  # flow loss (= -c_e/tau since no inlet electrons)
    
    # Diffusion loss is NOT in S_electron - it's implicit in the ambipolar term
    # Actually, let me check: is D_a/Lambda^2 applied to n_e density equation?
    # Looking at solver.py line 440: dydt[:n_sp] = S_electron * f + S_thermal + S_flow
    # Diffusion loss for electrons is NOT explicitly in the species equation!
    # It's only in the ENERGY equation via P_diff.
    # The electron density is governed purely by chemistry + flow.
    
    dc_e_total = dc_e_EI + dc_e_thermal + dc_e_flow

    # Break down EI electron production by reaction
    ei_details = []
    for rxn_obj in rxn.ei_reactions:
        j = rxn_obj._global_index
        rate_j = rates[j]
        stoich_e = rxn.stoich_matrix[0, j]  # net electron stoich (+1 for ioniz, -1 for attach)
        contrib = stoich_e * rate_j
        if abs(contrib) > 1e-20:
            ei_details.append((rxn_obj.formula, stoich_e, rate_j, contrib))
    
    # Sort by absolute contribution
    ei_details.sort(key=lambda x: -abs(x[3]))

    # Per-electron power = P_dep / n_e
    P_per_e = P_dep_Wm3 / n_e if n_e > 0 else 0  # W per electron
    P_per_e_eV = P_dep_eV / n_e if n_e > 0 else 0  # eV/s per electron

    return {
        'T_target': T_target_K, 'T_gas': T_gas, 'Te': Te_eV, 'eps': eps_mean,
        'n_e': n_e, 'c_e': c_e, 'N_gas': N_gas, 'tau': tau,
        'ne_eps': ne_eps, 'D_a': D_a,
        # Energy budget [eV/(m³·s)]
        'P_dep': P_dep_eV, 'P_elastic': P_el_eVm3s, 'P_inel': P_inel_eV,
        'P_diff': P_diff_eV, 'P_flow': P_flow_eV, 'P_eloss': P_eloss_eV,
        'E_rhs': energy_rhs,
        # Per-electron
        'P_per_e_eVs': P_per_e_eV,
        # Density budget [mol/(m³·s)]
        'dc_e_EI': dc_e_EI, 'dc_e_thermal': dc_e_thermal,
        'dc_e_flow': dc_e_flow, 'dc_e_total': dc_e_total,
        # Top EI reactions
        'ei_top': ei_details[:10],
    }


if __name__ == '__main__':
    print(f'ELECTRON ENERGY & DENSITY DIAGNOSTIC')
    print(f'V_eff={V_eff_cm3}cm³, V_reactor={V_reactor_cm3}cm³, P={P_W}W, Q={Q_slm}slm')
    print(f'T_gas = {T_gas_list_K} K\n')

    results = []
    for i, T_K in enumerate(T_gas_list_K):
        print(f'  Running T={T_K}K ...', end='', flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            r = diagnose(T_K)
        results.append(r)
        print(f' Te={r["Te"]:.3f}eV, n_e={r["n_e"]:.2e}')

    # =============================================
    # TABLE 1: Energy budget overview
    # =============================================
    print(f'\n{"="*140}')
    print(f'[1] ELECTRON ENERGY BUDGET  [eV/(m³·s)]  —  d(ne·ε̄)/dt = P_dep - P_el - P_inel - P_diff - P_flow - P_eloss')
    print(f'{"="*140}')
    print(f'{"T(K)":>5} {"Te(eV)":>7} {"n_e":>10} {"P_dep":>12} {"P_elastic":>12} {"P_inel":>12} '
          f'{"P_diff":>12} {"P_flow":>12} {"P_eloss":>12} {"RHS":>12}')
    print('-' * 140)
    for r in results:
        print(f'{r["T_target"]:5.0f} {r["Te"]:7.3f} {r["n_e"]:10.2e} '
              f'{r["P_dep"]:+12.3e} {r["P_elastic"]:+12.3e} {r["P_inel"]:+12.3e} '
              f'{r["P_diff"]:+12.3e} {r["P_flow"]:+12.3e} {r["P_eloss"]:+12.3e} '
              f'{r["E_rhs"]:+12.3e}')

    # =============================================
    # TABLE 2: Energy budget as % of P_dep
    # =============================================
    print(f'\n{"="*100}')
    print(f'[2] ENERGY BUDGET AS % OF P_dep')
    print(f'{"="*100}')
    print(f'{"T(K)":>5} {"Te(eV)":>7} {"P_el%":>8} {"P_inel%":>8} {"P_diff%":>8} {"P_flow%":>8} {"P_eloss%":>8} {"RHS%":>8}')
    print('-' * 100)
    for r in results:
        P = r['P_dep']
        print(f'{r["T_target"]:5.0f} {r["Te"]:7.3f} '
              f'{r["P_elastic"]/P*100:8.1f} {r["P_inel"]/P*100:8.1f} {r["P_diff"]/P*100:8.1f} '
              f'{r["P_flow"]/P*100:8.1f} {r["P_eloss"]/P*100:8.1f} {r["E_rhs"]/P*100:8.1f}')

    # =============================================
    # TABLE 3: Per-electron energy input
    # =============================================
    print(f'\n{"="*100}')
    print(f'[3] PER-ELECTRON DIAGNOSTICS')
    print(f'{"="*100}')
    print(f'{"T(K)":>5} {"Te(eV)":>7} {"n_e":>10} {"N_gas":>10} {"n_e/N":>10} '
          f'{"P/n_e(eV/s)":>13} {"D_a(m²/s)":>11} {"τ(s)":>8}')
    print('-' * 100)
    for r in results:
        print(f'{r["T_target"]:5.0f} {r["Te"]:7.3f} {r["n_e"]:10.2e} {r["N_gas"]:10.2e} '
              f'{r["n_e"]/r["N_gas"]:10.2e} {r["P_per_e_eVs"]:13.2e} '
              f'{r["D_a"]:11.3e} {r["tau"]:8.1f}')

    # =============================================
    # TABLE 4: Electron density budget
    # =============================================
    print(f'\n{"="*100}')
    print(f'[4] ELECTRON DENSITY BUDGET [mol/(m³·s)]  —  dc_e/dt = S_EI + S_thermal + S_flow')
    print(f'{"="*100}')
    print(f'{"T(K)":>5} {"Te(eV)":>7} {"n_e":>10} {"S_EI":>12} {"S_thermal":>12} {"S_flow":>12} '
          f'{"Total":>12} {"EI%":>7} {"Flow%":>7}')
    print('-' * 100)
    for r in results:
        mag = abs(r['dc_e_EI']) + abs(r['dc_e_thermal']) + abs(r['dc_e_flow'])
        ei_pct = r['dc_e_EI'] / mag * 100 if mag > 0 else 0
        fl_pct = r['dc_e_flow'] / mag * 100 if mag > 0 else 0
        print(f'{r["T_target"]:5.0f} {r["Te"]:7.3f} {r["n_e"]:10.2e} '
              f'{r["dc_e_EI"]:+12.3e} {r["dc_e_thermal"]:+12.3e} {r["dc_e_flow"]:+12.3e} '
              f'{r["dc_e_total"]:+12.3e} {ei_pct:+7.1f} {fl_pct:+7.1f}')

    # =============================================
    # TABLE 5: Top ionization reactions (300K vs 523K)
    # =============================================
    for r in [results[0], results[-1]]:
        print(f'\n{"="*100}')
        print(f'[5] TOP EI REACTIONS AT T={r["T_target"]:.0f}K (Te={r["Te"]:.3f}eV)')
        print(f'{"="*100}')
        print(f'{"Reaction":>50} {"ν_e":>4} {"Rate":>12} {"Contrib":>12} {"% of |S_EI|":>12}')
        print('-' * 100)
        s_ei_abs = abs(r['dc_e_EI'])
        for formula, nu_e, rate, contrib in r['ei_top']:
            pct = contrib / s_ei_abs * 100 if s_ei_abs > 0 else 0
            print(f'{formula:>50} {nu_e:+4.0f} {rate:12.3e} {contrib:+12.3e} {pct:+12.1f}')
