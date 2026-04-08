"""Gas temperature sweep with A20-based self-consistent power deposition.

Tests the hypothesis: T_gas↑ → N↓ → ε̄↑ (to compensate) → Te↑

Uses energy_source='A20': P_abs = n_e * N_gas * A20(ε̄) * QE
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

# === Fixed conditions ===
V_eff_cm3 = 1.6
V_reactor_cm3 = 100
P_W = 5.0           # fallback power (used when eps < eps_min_lut)
Q_slm = 0.4

# === Available LUTs ===
LUT_MAP = {
    300: {'bolsig': 'BOLSIG_parameter/Condition1_300K.txt',
          'eedf':   'BOLSIG_EEDF/EEDF_300K.dat'},
    523: {'bolsig': 'BOLSIG_parameter/Condition1_523K.txt',
          'eedf':   'BOLSIG_EEDF/EEDF_523K.dat'},
}
LUT_TEMPS = sorted(LUT_MAP.keys())

def nearest_lut(T_gas_K):
    return min(LUT_TEMPS, key=lambda t: abs(t - T_gas_K))

# === Gas temperature sweep ===
T_gas_list_K = [300, 350, 400, 450, 500, 523]


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
    cfg['T_wall'] = T_target_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_target_K

    # *** KEY: A20 self-consistent power ***
    cfg['energy_source'] = 'A20'

    # Select T_gas-matched LUT
    lut_T = nearest_lut(T_target_K)
    lut_files = LUT_MAP[lut_T]
    cfg['bolsig_files'] = [lut_files['bolsig']]
    cfg['eedf_files'] = [lut_files['eedf']]

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
    c_total = 101325.0 / (R_GAS * T_gas)
    tau = solver.flow.get_residence_time(T_gas)
    N_gas = 101325.0 / (KB * T_gs)

    # Compute P_abs = n_e * N_gas * A20(ε̄) * QE at steady state
    P_abs_Wm3 = 0.0
    A20_val = 0.0
    if solver.lut and eps_mean >= solver.lut.eps_range[0]:
        transport = solver.lut.get_transport(eps_mean)
        A20_val = transport.power_N
        P_abs_Wm3 = n_e * N_gas * A20_val * QE
    P_abs_W = P_abs_Wm3 * V_eff_cm3 * 1e-6  # total power [W]

    k_ei_conc = None
    if solver.lut and eps_mean >= solver.lut.eps_range[0]:
        k_ei_conc, Te_eV_lut = solver.lut.get_rate_coefficients_conc(eps_mean)

    rates = rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                        Te_eV=Te_eV, P_gas=101325.0)
    S_electron, S_thermal = rxn.compute_source_terms_split(rates)
    S_flow = solver.flow.compute_flow_source(c, T_gas)

    ch4 = sm.index('CH4'); co2 = sm.index('CO2')
    c0_ch4 = result.concentrations[ch4, 0]; cf_ch4 = result.concentrations[ch4, -1]
    c0_co2 = result.concentrations[co2, 0]; cf_co2 = result.concentrations[co2, -1]

    h2_idx = sm.index('H2'); co_idx = sm.index('CO')

    return {
        'T_target': T_target_K,
        'T_gas_actual': T_gas,
        'lut_T': lut_T,
        'Te': Te_eV, 'n_e': n_e, 'eps': eps_mean,
        'N_gas': N_gas,
        'A20': A20_val,
        'P_abs_Wm3': P_abs_Wm3,
        'P_abs_W': P_abs_W,
        'tau_ms': tau * 1e3,
        'ch4_conv': (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0,
        'co2_conv': (c0_co2 - cf_co2) / c0_co2 * 100 if c0_co2 > 0 else 0,
        'H2': c[h2_idx], 'CO': c[co_idx],
        'wall_time': result.wall_time,
    }


if __name__ == '__main__':
    print(f'A20 Self-Consistent Power: Gas Temperature Sweep')
    print(f'V_eff={V_eff_cm3}cm³, V_reactor={V_reactor_cm3}cm³, Q={Q_slm}slm')
    print(f'Fallback P={P_W}W (used only when ε̄ < ε̄_min_LUT)')
    print(f'Available LUTs: {LUT_TEMPS} K')
    print(f'T_gas = {T_gas_list_K} K')
    print(f'LUT selection: {[nearest_lut(T) for T in T_gas_list_K]}\n')

    results = []
    t_start = time.time()

    for i, T_K in enumerate(T_gas_list_K):
        lut_T = nearest_lut(T_K)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                r = run_and_analyze(T_K)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                r = {'T_target': T_K, 'T_gas_actual': 0, 'lut_T': lut_T,
                     'Te': -1, 'n_e': 0, 'N_gas': 0, 'A20': 0,
                     'P_abs_Wm3': 0, 'P_abs_W': 0,
                     'ch4_conv': -999, 'co2_conv': -999, 'error': str(e),
                     'traceback': tb,
                     'tau_ms': 0, 'eps': 0, 'wall_time': 0,
                     'H2': 0, 'CO': 0}
        results.append(r)
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(T_gas_list_K) - i - 1)
        err = r.get('error', '')
        print(f'  [{i+1:2d}/{len(T_gas_list_K)}] T={T_K}K LUT={r["lut_T"]}K |'
              f' Te={r["Te"]:5.2f}eV n_e={r["n_e"]:.2e} P_abs={r["P_abs_W"]:.3f}W'
              f' CH4={r["ch4_conv"]:+6.1f}% |'
              f' {r["wall_time"]:4.0f}s {f"ERR:{err}" if err else ""}', flush=True)

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time/60:.1f} min')

    # =================================================================
    # SUMMARY TABLE
    # =================================================================
    print(f'\n{"="*140}')
    print(f'  A20 SELF-CONSISTENT POWER: T_GAS SWEEP')
    print(f'  V_eff={V_eff_cm3}cm³, V_reactor={V_reactor_cm3}cm³, Q={Q_slm}slm')
    print(f'  P_dep = n_e · N_gas · A20(ε̄) · QE')
    print(f'{"="*140}')

    print(f'\n[1] OVERVIEW')
    print(f'{"T(K)":>6} {"LUT":>5} {"Tg_act":>6} {"τ(ms)":>7} {"N_gas":>10} {"n_e":>10} '
          f'{"ε̄(eV)":>8} {"Te(eV)":>7} {"A20":>10} {"P_abs(W)":>9} {"CH4%":>7} {"CO2%":>7}')
    print('-' * 110)
    for r in results:
        print(f'{r["T_target"]:6.0f} {r["lut_T"]:5.0f} {r["T_gas_actual"]:6.0f} '
              f'{r["tau_ms"]:7.0f} {r["N_gas"]:10.2e} {r["n_e"]:10.2e} {r["eps"]:8.4f} '
              f'{r["Te"]:7.3f} {r["A20"]:10.3e} {r["P_abs_W"]:9.4f} '
              f'{r["ch4_conv"]:+7.2f} {r["co2_conv"]:+7.2f}')

    # Comparison: constant power vs A20
    print(f'\n[2] COMPARISON: Constant Power (old) vs A20 (new)')
    print(f'{"T(K)":>6} {"mode":>10} {"Te(eV)":>7} {"n_e":>10} {"P_abs(W)":>9} {"CH4%":>7}')
    print('-' * 60)
    prev = {300: (2.02, 5.37e14, 5.0, 12.8), 523: (1.07, 7.99e15, 5.0, 9.4)}
    for r in results:
        T = r['T_target']
        if T in prev:
            print(f'{T:6.0f} {"const_P":>10} {prev[T][0]:7.3f} {prev[T][1]:10.2e} '
                  f'{prev[T][2]:9.4f} {prev[T][3]:+7.2f}')
        print(f'{r["T_target"]:6.0f} {"A20":>10} {r["Te"]:7.3f} {r["n_e"]:10.2e} '
              f'{r["P_abs_W"]:9.4f} {r["ch4_conv"]:+7.2f}')

    # Key physics check
    print(f'\n[3] PHYSICS CHECK: Te vs T_gas trend')
    Te_vals = [r['Te'] for r in results if r['Te'] > 0]
    if len(Te_vals) >= 2:
        if Te_vals[-1] > Te_vals[0]:
            print(f'  ✅ Te INCREASES with T_gas: {Te_vals[0]:.3f} → {Te_vals[-1]:.3f} eV')
        elif abs(Te_vals[-1] - Te_vals[0]) < 0.01:
            print(f'  ⚠️  Te roughly constant: {Te_vals[0]:.3f} → {Te_vals[-1]:.3f} eV')
        else:
            print(f'  ❌ Te still DECREASES: {Te_vals[0]:.3f} → {Te_vals[-1]:.3f} eV')

    # Print tracebacks for any errors
    for r in results:
        if 'traceback' in r:
            print(f'\n*** ERROR at T={r["T_target"]}K ***')
            print(r['traceback'])
