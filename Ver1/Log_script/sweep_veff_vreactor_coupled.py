"""V_eff & V_reactor coupled sweep.
Keep f=V_eff/V_reactor constant (=0.02) and also try other combos.
P=5W, Q=0.4slm, t=3s.
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
    sm = solver.sm; rxn = solver.rxn; n_sp = sm.n_species
    f = solver._vol_ratio
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

    k_ei_conc = None
    if solver.lut and eps_mean >= solver.lut.eps_range[0]:
        k_ei_conc, Te_eV = solver.lut.get_rate_coefficients_conc(eps_mean)

    rates = rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                        Te_eV=Te_eV, P_gas=solver.power.P_gas)
    S_electron, S_thermal = rxn.compute_source_terms_split(rates)
    S_flow = solver.flow.compute_flow_source(c, T_gas)
    f_sp = solver._f_species

    ch4 = sm.index('CH4'); co2 = sm.index('CO2')
    c0_ch4 = result.concentrations[ch4, 0]; cf_ch4 = result.concentrations[ch4, -1]
    c0_co2 = result.concentrations[co2, 0]; cf_co2 = result.concentrations[co2, -1]

    mag = abs(S_electron[ch4]*f_sp[ch4]) + abs(S_thermal[ch4]) + abs(S_flow[ch4])
    ei_pct = S_electron[ch4]*f_sp[ch4]/mag*100 if mag>0 else 0
    arr_pct = S_thermal[ch4]/mag*100 if mag>0 else 0
    flow_pct = S_flow[ch4]/mag*100 if mag>0 else 0

    return {
        'V_eff': solver._V_eff*1e6, 'V_reactor': solver._V_reactor*1e6,
        'f': f, 'tau_ms': tau*1e3,
        'n_e': n_e, 'Te': Te_eV, 'Tgas': T_gas,
        'ch4_conv': (c0_ch4-cf_ch4)/c0_ch4*100 if c0_ch4>0 else 0,
        'co2_conv': (c0_co2-cf_co2)/c0_co2*100 if c0_co2>0 else 0,
        'P_dep_Wm3': solver.power.get_power_density(result.t[-1]),
        'ei_pct': ei_pct, 'arr_pct': arr_pct, 'flow_pct': flow_pct,
        'wall_time': result.wall_time,
    }


if __name__ == '__main__':
    P_W = 5.0; Q_slm = 0.4; t_end = 3.0

    # Test cases: (V_eff_cm3, V_reactor_cm3)
    cases = [
        # Baseline
        (0.4, 20),
        # f=0.02 constant, scale both up
        (1.0, 50),
        (2.0, 100),
        (4.0, 200),
        # f=0.02 constant, even bigger
        (8.0, 400),
        # Larger V_eff, moderate V_reactor (higher f)
        (1.0, 20),
        (2.0, 20),
        (2.0, 50),
    ]

    results = []
    for ve, vr in cases:
        print(f'\n{"="*60}')
        print(f'  V_eff={ve}cm³, V_reactor={vr}cm³, f={ve/vr:.4f}')
        print(f'{"="*60}')
        solver, result = run_one(ve*1e-6, vr*1e-6, P_W, Q_slm, t_end)
        r = analyze(solver, result)
        results.append(r)

    # Summary
    print('\n' + '='*140)
    print(f'  V_eff & V_reactor COUPLED SWEEP: P={P_W}W, Q={Q_slm}slm, t={t_end}s')
    print('='*140)
    print(f'{"V_eff":>6} {"V_rct":>6} {"f":>7} {"τ(ms)":>7} {"P_dep":>11} {"n_e":>11} {"Te(eV)":>7} {"Tg(K)":>7} {"CH4%":>7} {"CO2%":>7}  {"EI%":>6} {"Arr%":>6} {"Flw%":>6} {"time":>5}')
    print('-'*130)
    for r in results:
        print(f'{r["V_eff"]:6.1f} {r["V_reactor"]:6.0f} {r["f"]:7.4f} {r["tau_ms"]:7.0f} '
              f'{r["P_dep_Wm3"]:11.2e} {r["n_e"]:11.2e} {r["Te"]:7.2f} {r["Tgas"]:7.1f} '
              f'{r["ch4_conv"]:+7.2f} {r["co2_conv"]:+7.2f}  '
              f'{r["ei_pct"]:+6.1f} {r["arr_pct"]:+6.1f} {r["flow_pct"]:+6.1f} {r["wall_time"]:5.0f}')
