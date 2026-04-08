"""Gas temperature parametric sweep.
V_eff=1.6cm³, V_reactor=100cm³ fixed.
T_gas from 300K to 523K (250°C), clamped via T_wall + high wlf.
P=5W, Q=0.4slm.
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

# === Fixed conditions ===
V_eff_cm3 = 1.6
V_reactor_cm3 = 100
P_W = 5.0
Q_slm = 0.4

# === Gas temperature sweep ===
T_gas_list_K = [300, 325, 350, 375, 400, 425, 450, 475, 500, 523]


def run_and_analyze(T_target_K):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    V_eff_m3 = V_eff_cm3 * 1e-6
    V_reactor_m3 = V_reactor_cm3 * 1e-6

    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm

    # Clamp gas temperature via T_wall + very high wall_loss_freq
    cfg['T_wall'] = T_target_K
    cfg['wall_loss_freq'] = 10000.0  # strong thermal clamp
    cfg['initial']['T_gas'] = T_target_K

    # Adaptive t_end: tau depends on T_gas (higher T → faster flow → shorter tau)
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

    # Analyze final state
    sm = solver.sm; rxn = solver.rxn; n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    T_gas = y[sm.idx_Tgas]; ne_eps = y[sm.idx_energy]
    c_e = c[0]; n_e = c_e * NA
    T_gs = max(T_gas, 200.0)
    eps_th = 1.5 * KB * T_gs / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean
    c_total = solver.power.P_gas / (R_GAS * T_gas)
    tau = solver.flow.get_residence_time(T_gas)

    # LUT query for rate coefficients
    k_ei_conc = None
    if solver.lut and eps_mean >= solver.lut.eps_range[0]:
        k_ei_conc, Te_eV_lut = solver.lut.get_rate_coefficients_conc(eps_mean)

    rates = rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                        Te_eV=Te_eV, P_gas=solver.power.P_gas)
    S_electron, S_thermal = rxn.compute_source_terms_split(rates)
    S_flow = solver.flow.compute_flow_source(c, T_gas)
    f_sp = solver._f_species

    # CH4 and CO2 conversion
    ch4 = sm.index('CH4'); co2 = sm.index('CO2')
    c0_ch4 = result.concentrations[ch4, 0]; cf_ch4 = result.concentrations[ch4, -1]
    c0_co2 = result.concentrations[co2, 0]; cf_co2 = result.concentrations[co2, -1]

    # CH4 budget
    ch4_sei_f = S_electron[ch4] * f_sp[ch4]
    ch4_sarr = S_thermal[ch4]
    ch4_sflow = S_flow[ch4]
    mag = abs(ch4_sei_f) + abs(ch4_sarr) + abs(ch4_sflow)
    ei_pct = ch4_sei_f / mag * 100 if mag > 0 else 0
    arr_pct = ch4_sarr / mag * 100 if mag > 0 else 0
    flow_pct = ch4_sflow / mag * 100 if mag > 0 else 0

    # Product species
    h2_idx = sm.index('H2'); co_idx = sm.index('CO')
    c2h6_idx = sm.index('C2H6'); c2h2_idx = sm.index('C2H2'); c2h4_idx = sm.index('C2H4')
    ch3oh_idx = sm.index('CH3OH'); ch2o_idx = sm.index('CH2O')

    return {
        'T_target': T_target_K,
        'T_gas_actual': T_gas,
        'Te': Te_eV, 'n_e': n_e, 'eps': eps_mean,
        'tau_ms': tau * 1e3,
        'ch4_conv': (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0,
        'co2_conv': (c0_co2 - cf_co2) / c0_co2 * 100 if c0_co2 > 0 else 0,
        'ch4_sei_f': ch4_sei_f, 'ch4_sarr': ch4_sarr, 'ch4_sflow': ch4_sflow,
        'ei_pct': ei_pct, 'arr_pct': arr_pct, 'flow_pct': flow_pct,
        # Products [mol/m³]
        'H2': c[h2_idx], 'CO': c[co_idx],
        'C2H6': c[c2h6_idx], 'C2H2': c[c2h2_idx], 'C2H4': c[c2h4_idx],
        'CH3OH': c[ch3oh_idx], 'CH2O': c[ch2o_idx],
        'wall_time': result.wall_time,
    }


if __name__ == '__main__':
    print(f'Gas Temperature Sweep: {len(T_gas_list_K)} cases')
    print(f'V_eff={V_eff_cm3}cm³, V_reactor={V_reactor_cm3}cm³, P={P_W}W, Q={Q_slm}slm')
    print(f'T_gas = {T_gas_list_K} K')
    print(f'       = {[T-273 for T in T_gas_list_K]} °C')

    results = []
    t_start = time.time()

    for i, T_K in enumerate(T_gas_list_K):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                r = run_and_analyze(T_K)
            except Exception as e:
                r = {'T_target': T_K, 'T_gas_actual': 0, 'Te': -1, 'n_e': 0,
                     'ch4_conv': -999, 'co2_conv': -999, 'error': str(e),
                     'tau_ms': 0, 'eps': 0, 'wall_time': 0,
                     'ch4_sei_f': 0, 'ch4_sarr': 0, 'ch4_sflow': 0,
                     'ei_pct': 0, 'arr_pct': 0, 'flow_pct': 0,
                     'H2': 0, 'CO': 0, 'C2H6': 0, 'C2H2': 0, 'C2H4': 0,
                     'CH3OH': 0, 'CH2O': 0}
        results.append(r)
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(T_gas_list_K) - i - 1)
        err = r.get('error', '')
        print(f'  [{i+1:2d}/{len(T_gas_list_K)}] T={T_K}K ({T_K-273}°C) |'
              f' Te={r["Te"]:5.2f}eV Tg={r["T_gas_actual"]:5.0f}K'
              f' CH4={r["ch4_conv"]:+6.1f}% CO2={r["co2_conv"]:+6.1f}% |'
              f' {r["wall_time"]:4.0f}s {f"ERR:{err}" if err else ""}', flush=True)

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time/60:.1f} min')

    # =================================================================
    # SUMMARY TABLE
    # =================================================================
    print(f'\n{"="*130}')
    print(f'  GAS TEMPERATURE SWEEP: V_eff={V_eff_cm3}cm³, V_reactor={V_reactor_cm3}cm³, P={P_W}W, Q={Q_slm}slm')
    print(f'{"="*130}')

    print(f'\n[1] OVERVIEW')
    print(f'{"T(K)":>6} {"T(°C)":>6} {"Tg_act":>6} {"τ(ms)":>7} {"n_e":>11} {"Te(eV)":>7} {"CH4%":>7} {"CO2%":>7}')
    print('-' * 75)
    for r in results:
        print(f'{r["T_target"]:6.0f} {r["T_target"]-273:6.0f} {r["T_gas_actual"]:6.0f} '
              f'{r["tau_ms"]:7.0f} {r["n_e"]:11.2e} {r["Te"]:7.2f} '
              f'{r["ch4_conv"]:+7.2f} {r["co2_conv"]:+7.2f}')

    print(f'\n[2] CH4 SPECIES BUDGET [mol/(m³·s)]')
    print(f'{"T(K)":>6} {"S_EI×f":>12} {"S_Arr":>12} {"S_flow":>12} {"EI%":>7} {"Arr%":>7} {"Flow%":>7}')
    print('-' * 75)
    for r in results:
        print(f'{r["T_target"]:6.0f} {r["ch4_sei_f"]:+12.3e} {r["ch4_sarr"]:+12.3e} '
              f'{r["ch4_sflow"]:+12.3e} {r["ei_pct"]:+7.1f} {r["arr_pct"]:+7.1f} {r["flow_pct"]:+7.1f}')

    print(f'\n[3] PRODUCT CONCENTRATIONS [mol/m³]')
    print(f'{"T(K)":>6} {"H2":>10} {"CO":>10} {"C2H6":>10} {"C2H4":>10} {"C2H2":>10} {"CH3OH":>10} {"CH2O":>10}')
    print('-' * 85)
    for r in results:
        print(f'{r["T_target"]:6.0f} {r["H2"]:10.3e} {r["CO"]:10.3e} {r["C2H6"]:10.3e} '
              f'{r["C2H4"]:10.3e} {r["C2H2"]:10.3e} {r["CH3OH"]:10.3e} {r["CH2O"]:10.3e}')
