"""Gas temperature sweep with A20_power_balance mode.

ε̄ is determined by power constraint: n_e·N·A20(ε̄) = P_input/V_eff
P_dep = P_input/V_eff (constant), ne_eps relaxes to constrained value.
"""
import sys, os, io, contextlib, time, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

V_eff_cm3 = 1.6
V_reactor_cm3 = 100
P_W = 5.0
Q_slm = 0.4

LUT_MAP = {
    300: {'bolsig': 'BOLSIG_parameter/Condition1_300K.txt',
          'eedf':   'BOLSIG_EEDF/EEDF_300K.dat'},
    523: {'bolsig': 'BOLSIG_parameter/Condition1_523K.txt',
          'eedf':   'BOLSIG_EEDF/EEDF_523K.dat'},
}
LUT_TEMPS = sorted(LUT_MAP.keys())

def nearest_lut(T_gas_K):
    return min(LUT_TEMPS, key=lambda t: abs(t - T_gas_K))

T_gas_list_K = [300, 350, 400, 450, 500, 523]


def run_case(T_target_K):
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
    cfg['energy_source'] = 'A20_power_balance'

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

    sm = solver.sm; n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    T_gas = y[sm.idx_Tgas]
    c_e = c[0]; n_e = c_e * NA
    N_gas = 101325.0 / (KB * max(T_gas, 200.0))

    # Constrained eps
    P_target = P_W / V_eff_m3
    A20_target = P_target / (QE * max(n_e, 1.0) * max(N_gas, 1e20))
    eps_c = solver.lut.invert_A20(A20_target)
    Te = (2.0/3.0) * eps_c

    # Verify power balance
    transport = solver.lut.get_transport(eps_c)
    P_abs_check = n_e * N_gas * transport.power_N * QE
    P_abs_W = P_abs_check * V_eff_m3

    tau = solver.flow.get_residence_time(T_gas)
    diff_freq = solver.ekin.compute_diffusion_rate(T_gas, Te)

    ch4 = sm.index('CH4'); co2 = sm.index('CO2')
    c0_ch4 = result.y[ch4, 0]; cf_ch4 = result.y[ch4, -1]
    c0_co2 = result.y[co2, 0]; cf_co2 = result.y[co2, -1]

    return {
        'T_target': T_target_K, 'T_gas_actual': T_gas,
        'lut_T': lut_T,
        'Te': Te, 'n_e': n_e, 'eps': eps_c,
        'N_gas': N_gas, 'P_abs_W': P_abs_W,
        'tau_ms': tau * 1e3,
        'diff_freq': diff_freq,
        'ch4_conv': (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0,
        'co2_conv': (c0_co2 - cf_co2) / c0_co2 * 100 if c0_co2 > 0 else 0,
        'wall_time': result.wall_time,
    }


if __name__ == '__main__':
    print(f'A20 Power Balance: Gas Temperature Sweep', flush=True)
    print(f'V_eff={V_eff_cm3}cm³, V_reactor={V_reactor_cm3}cm³, P={P_W}W, Q={Q_slm}slm', flush=True)
    print(f'Mode: ε̄ from n_e·N·A20(ε̄)=P_input/V_eff, P_dep=P_input/V_eff', flush=True)
    print(f'T_gas = {T_gas_list_K} K', flush=True)

    results = []
    t_start = time.time()

    for i, T_K in enumerate(T_gas_list_K):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                r = run_case(T_K)
            except Exception as e:
                import traceback
                r = {'T_target': T_K, 'T_gas_actual': 0, 'lut_T': nearest_lut(T_K),
                     'Te': -1, 'n_e': 0, 'N_gas': 0, 'P_abs_W': 0,
                     'ch4_conv': -999, 'co2_conv': -999, 'error': str(e),
                     'traceback': traceback.format_exc(),
                     'tau_ms': 0, 'eps': 0, 'wall_time': 0, 'diff_freq': 0}
        results.append(r)
        err = r.get('error', '')
        print(f'  [{i+1}/{len(T_gas_list_K)}] T={T_K}K LUT={r["lut_T"]}K |'
              f' Te={r["Te"]:5.3f}eV n_e={r["n_e"]:.2e}'
              f' CH4={r["ch4_conv"]:+6.2f}% P={r["P_abs_W"]:.3f}W |'
              f' {r["wall_time"]:4.0f}s {f"ERR:{err}" if err else ""}', flush=True)

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time/60:.1f} min', flush=True)

    print(f'\n{"="*130}', flush=True)
    print(f'  A20 POWER BALANCE: T_GAS SWEEP', flush=True)
    print(f'  ε̄ = A20_inv(P_input / (V_eff·QE·n_e·N))', flush=True)
    print(f'{"="*130}', flush=True)

    print(f'\n[1] OVERVIEW', flush=True)
    print(f'{"T(K)":>6} {"LUT":>5} {"N_gas":>10} {"n_e":>10} '
          f'{"ε̄(eV)":>8} {"Te(eV)":>7} {"P_abs(W)":>9} {"D_a/L2":>8} '
          f'{"CH4%":>7} {"CO2%":>7}', flush=True)
    print('-' * 100, flush=True)
    for r in results:
        print(f'{r["T_target"]:6.0f} {r["lut_T"]:5.0f} {r["N_gas"]:10.2e} {r["n_e"]:10.2e} '
              f'{r["eps"]:8.4f} {r["Te"]:7.4f} {r["P_abs_W"]:9.4f} {r["diff_freq"]:8.1f} '
              f'{r["ch4_conv"]:+7.2f} {r["co2_conv"]:+7.2f}', flush=True)

    print(f'\n[2] COMPARISON: All three modes', flush=True)
    print(f'{"T(K)":>6} {"mode":>15} {"Te(eV)":>7} {"n_e":>10} {"CH4%":>7}', flush=True)
    print('-' * 55, flush=True)
    prev_const = {300: (2.02, 5.37e14, 12.8), 523: (1.07, 7.99e15, 9.4)}
    prev_const_da = {300: (2.17, 4.60e14, 6.12), 523: (1.80, 1.40e15, 19.9)}
    for r in results:
        T = r['T_target']
        if T in prev_const:
            print(f'{T:6.0f} {"const(old)":>15} {prev_const[T][0]:7.3f} {prev_const[T][1]:10.2e} {prev_const[T][2]:+7.2f}', flush=True)
            print(f'{T:6.0f} {"const+Da":>15} {prev_const_da[T][0]:7.3f} {prev_const_da[T][1]:10.2e} {prev_const_da[T][2]:+7.2f}', flush=True)
        print(f'{T:6.0f} {"A20_PB":>15} {r["Te"]:7.4f} {r["n_e"]:10.2e} {r["ch4_conv"]:+7.2f}', flush=True)

    print(f'\n[3] PHYSICS CHECK', flush=True)
    Te_vals = [r['Te'] for r in results if r['Te'] > 0]
    if len(Te_vals) >= 2:
        delta = Te_vals[-1] - Te_vals[0]
        pct = delta / Te_vals[0] * 100
        if delta > 0.01:
            print(f'  ✅ Te INCREASES: {Te_vals[0]:.4f} → {Te_vals[-1]:.4f} eV ({pct:+.1f}%)', flush=True)
        elif abs(delta) < 0.01:
            print(f'  ⚠️  Te roughly FLAT: {Te_vals[0]:.4f} → {Te_vals[-1]:.4f} eV ({pct:+.1f}%)', flush=True)
        else:
            print(f'  ❌ Te still decreases: {Te_vals[0]:.4f} → {Te_vals[-1]:.4f} eV ({pct:+.1f}%)', flush=True)
        print(f'  Original drop (const, no Da): 2.02 → 1.07 eV (-47%)', flush=True)
        print(f'  With Da species loss:         2.17 → 1.80 eV (-17%)', flush=True)
        print(f'  A20 power balance:            {Te_vals[0]:.2f} → {Te_vals[-1]:.2f} eV ({pct:+.1f}%)', flush=True)

    for r in results:
        if 'traceback' in r:
            print(f'\n*** ERROR at T={r["T_target"]}K ***\n{r["traceback"]}', flush=True)
