"""V_eff parametric sweep with RHS budget analysis.
V_reactor=20cm³ fixed, Q=0.4slm, P=5W, t=3s.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS


def run_one(V_eff_m3, V_reactor_m3, P_W, Q_slm, t_end_s):
    cfg = load_config(os.path.join(os.path.dirname(__file__), 'plasma0d_v2', 'config.yaml'))
    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm
    cfg['solver'] = {'t_end': t_end_s, 'n_points': 2000, 'method': 'BDF',
                     'rtol': 1e-6, 'atol': 1e-10, 'max_step': 1e-4, 'constrained': False}
    solver, y0, t_span, cfg = setup_simulation(cfg, os.path.join(os.path.dirname(__file__), 'plasma0d_v2'))
    scfg = cfg['solver']
    result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                          rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])
    return solver, result


def analyze(solver, result):
    sm = solver.sm
    rxn = solver.rxn
    n_sp = sm.n_species
    f = solver._vol_ratio

    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    T_gas = y[sm.idx_Tgas]
    ne_eps = y[sm.idx_energy]
    c_e = c[0]; n_e = c_e * NA
    T_gs = max(T_gas, 200.0)
    eps_th = 1.5 * KB * T_gs / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean
    c_total = solver.power.P_gas / (R_GAS * T_gas)
    N_gas = solver.power.P_gas / (KB * T_gs)
    tau = solver.flow.get_residence_time(T_gas)

    k_ei_conc = None
    P_el_eVm3s = 0.0
    Q_elastic_Wm3 = 0.0
    if solver.lut and eps_mean >= solver.lut.eps_range[0]:
        k_ei_conc, Te_eV = solver.lut.get_rate_coefficients_conc(eps_mean)
        transport = solver.lut.get_transport(eps_mean)
        P_el_eVm3s = n_e * N_gas * transport.elastic_power_N
        Q_elastic_Wm3 = P_el_eVm3s * QE

    rates = rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                        Te_eV=Te_eV, P_gas=solver.power.P_gas)
    S_electron, S_thermal = rxn.compute_source_terms_split(rates)
    S_flow = solver.flow.compute_flow_source(c, T_gas)
    f_sp = solver._f_species

    # --- Electron energy budget ---
    P_dep_Wm3 = solver.power.get_power_density(result.t[-1])
    P_inel_Wm3 = rxn.compute_electron_energy_loss(rates)
    S_e_loss = rxn.compute_electron_loss_rate(rates)
    P_e_loss_eVm3s = eps_mean * S_e_loss * NA
    mu_i = 2.8e22 / N_gas
    D_a = mu_i * Te_eV
    P_diff = ne_eps * D_a / solver.ekin.Lambda_sq
    P_flow_e = ne_eps / tau if tau > 0 else 0.0

    # --- Gas temp budget ---
    Q_rxn_te, Q_rxn_arr = rxn.compute_gas_heating_split(rates)
    Q_e_loss_Wm3 = P_e_loss_eVm3s * QE
    rho_cp = (solver.power.P_gas * solver.gth.M_avg / (R_GAS * T_gas)) * solver.gth.cp_avg
    Q_wall = rho_cp * solver.gth.wall_loss_freq * (T_gas - solver.gth.T_wall)
    Q_flow_gas = rho_cp * (T_gas - 300.0) / tau if tau > 0 else 0.0

    # CH4 budget
    ch4 = sm.index('CH4')
    co2 = sm.index('CO2')
    c0_ch4 = result.concentrations[ch4, 0]
    cf_ch4 = result.concentrations[ch4, -1]
    c0_co2 = result.concentrations[co2, 0]
    cf_co2 = result.concentrations[co2, -1]

    return {
        'V_eff_cm3': solver._V_eff * 1e6,
        'f': f,
        'tau_ms': tau * 1e3,
        'n_e': n_e, 'Te': Te_eV, 'eps': eps_mean, 'Tgas': T_gas,
        # species: CH4
        'ch4_c': c[ch4],
        'ch4_sei_f': S_electron[ch4] * f_sp[ch4],
        'ch4_sarr': S_thermal[ch4],
        'ch4_sflow': S_flow[ch4],
        'ch4_conv': (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0,
        # species: CO2
        'co2_conv': (c0_co2 - cf_co2) / c0_co2 * 100 if c0_co2 > 0 else 0,
        # electron energy
        'P_dep': P_dep_Wm3 / QE,
        'P_el': P_el_eVm3s,
        'P_inel': P_inel_Wm3 / QE,
        'P_diff': P_diff,
        'P_flow_e': P_flow_e,
        'P_eloss': P_e_loss_eVm3s,
        # gas temp
        'Q_el_f': Q_elastic_Wm3 * f,
        'Q_te_f': Q_rxn_te * f,
        'Q_arr': Q_rxn_arr,
        'Q_eloss_f': Q_e_loss_Wm3 * f,
        'Q_wall': Q_wall,
        'Q_flow_g': Q_flow_gas,
        'wall_time': result.wall_time,
    }


if __name__ == '__main__':
    V_reactor = 20e-6  # 20 cm³ fixed
    P_W = 5.0
    Q_slm = 0.4
    t_end = 3.0  # ~1.1τ at V_reactor=20cm³

    V_eff_list_cm3 = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5.0]
    results = []

    for ve_cm3 in V_eff_list_cm3:
        ve_m3 = ve_cm3 * 1e-6
        print(f'\n{"="*60}')
        print(f'  V_eff = {ve_cm3} cm³, f = {ve_m3/V_reactor:.4f}')
        print(f'{"="*60}')
        solver, result = run_one(ve_m3, V_reactor, P_W, Q_slm, t_end)
        r = analyze(solver, result)
        results.append(r)

    # === SUMMARY TABLES ===
    print('\n' + '=' * 130)
    print(f'  V_eff SWEEP: V_reactor=20cm³, P={P_W}W, Q={Q_slm}slm, t={t_end}s')
    print('=' * 130)

    # (1) Overview
    print(f'\n[1] OVERVIEW')
    print(f'{"V_eff":>6} {"f":>7} {"τ(ms)":>7} {"n_e":>11} {"Te(eV)":>7} {"Tg(K)":>7} {"CH4%":>7} {"CO2%":>7} {"time":>6}')
    print('-' * 80)
    for r in results:
        print(f'{r["V_eff_cm3"]:6.1f} {r["f"]:7.4f} {r["tau_ms"]:7.0f} {r["n_e"]:11.2e} '
              f'{r["Te"]:7.2f} {r["Tgas"]:7.1f} {r["ch4_conv"]:+7.2f} {r["co2_conv"]:+7.2f} {r["wall_time"]:6.1f}')

    # (2) CH4 species budget
    print(f'\n[2] CH4 SPECIES BUDGET [mol/(m³·s)]')
    print(f'{"V_eff":>6} {"S_EI/TE×f":>12} {"S_Arr":>12} {"S_flow":>12} {"Total":>12}   {"EI%":>6} {"Arr%":>6} {"Flow%":>6}')
    print('-' * 90)
    for r in results:
        mag = abs(r['ch4_sei_f']) + abs(r['ch4_sarr']) + abs(r['ch4_sflow'])
        if mag > 0:
            pe = r['ch4_sei_f']/mag*100; pt = r['ch4_sarr']/mag*100; pf = r['ch4_sflow']/mag*100
        else:
            pe = pt = pf = 0
        total = r['ch4_sei_f'] + r['ch4_sarr'] + r['ch4_sflow']
        print(f'{r["V_eff_cm3"]:6.1f} {r["ch4_sei_f"]:+12.3e} {r["ch4_sarr"]:+12.3e} '
              f'{r["ch4_sflow"]:+12.3e} {total:+12.3e}   {pe:+6.1f} {pt:+6.1f} {pf:+6.1f}')

    # (3) Electron energy budget
    print(f'\n[3] ELECTRON ENERGY BUDGET [eV/(m³·s)]')
    print(f'{"V_eff":>6} {"P_dep":>11} {"P_inel":>11} {"P_el":>11} {"P_eloss":>11} {"P_diff":>11} {"P_flow":>11}  {"inel%":>6} {"el%":>6} {"eloss%":>6}')
    print('-' * 120)
    for r in results:
        total_loss = r['P_el'] + r['P_inel'] + r['P_eloss'] + r['P_diff'] + r['P_flow_e']
        pd = r['P_dep']
        pi_pct = r['P_inel']/pd*100 if pd>0 else 0
        pe_pct = r['P_el']/pd*100 if pd>0 else 0
        pl_pct = r['P_eloss']/pd*100 if pd>0 else 0
        print(f'{r["V_eff_cm3"]:6.1f} {r["P_dep"]:11.2e} {r["P_inel"]:11.2e} {r["P_el"]:11.2e} '
              f'{r["P_eloss"]:11.2e} {r["P_diff"]:11.2e} {r["P_flow_e"]:11.2e}  '
              f'{pi_pct:6.1f} {pe_pct:6.1f} {pl_pct:6.1f}')

    # (4) Gas temperature budget
    print(f'\n[4] GAS TEMPERATURE BUDGET [W/m³]')
    print(f'{"V_eff":>6} {"Q_el×f":>11} {"Q_te×f":>11} {"Q_arr":>11} {"Q_eloss×f":>11} {"Q_wall":>11} {"Q_flow":>11}  {"arr%h":>6} {"wall%c":>6}')
    print('-' * 110)
    for r in results:
        th = abs(r['Q_el_f']) + abs(r['Q_arr']) + abs(r['Q_eloss_f']) + max(r['Q_te_f'],0)
        arr_pct = r['Q_arr']/th*100 if th>0 else 0
        tc = r['Q_wall'] + r['Q_flow_g']
        wall_pct = r['Q_wall']/tc*100 if tc>0 else 0
        print(f'{r["V_eff_cm3"]:6.1f} {r["Q_el_f"]:+11.3e} {r["Q_te_f"]:+11.3e} {r["Q_arr"]:+11.3e} '
              f'{r["Q_eloss_f"]:+11.3e} {r["Q_wall"]:+11.3e} {r["Q_flow_g"]:+11.3e}  '
              f'{arr_pct:6.1f} {wall_pct:6.1f}')
