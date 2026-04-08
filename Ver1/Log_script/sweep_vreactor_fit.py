"""V_reactor parameter sweep to fit experimental CH4 conversion.

Experimental data (T_gas °C → CH4 conversion %):
  30°C  →  5.26%
  100°C →  8.05%
  180°C → 14.36%
  250°C → 20.02%

Sweep V_reactor, fixed V_eff=1.6cm³, P=5W, Q=0.4slm, constant mode.
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

# === Experimental data ===
EXP_DATA = {
    303: 5.26,    # 30°C
    373: 8.05,    # 100°C
    453: 14.36,   # 180°C
    523: 20.02,   # 250°C
}
T_gas_list_K = list(EXP_DATA.keys())

# === Fixed conditions ===
V_eff_cm3 = 1.6
P_W = 5.0
Q_slm = 0.4

# === V_reactor sweep (cm³) ===
V_reactor_list_cm3 = [100, 150, 200, 250, 300, 400, 500]


def run_single(T_target_K, V_reactor_cm3):
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

    sm = solver.sm
    ch4_idx = sm.index('CH4')
    co2_idx = sm.index('CO2')
    c0_ch4 = result.concentrations[ch4_idx, 0]
    cf_ch4 = result.concentrations[ch4_idx, -1]
    conv_ch4 = (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0
    c0_co2 = result.concentrations[co2_idx, 0]
    cf_co2 = result.concentrations[co2_idx, -1]
    conv_co2 = (c0_co2 - cf_co2) / c0_co2 * 100 if c0_co2 > 0 else 0

    n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    ne_eps = y[sm.idx_energy]
    n_e = c[0] * NA
    T_gs = max(y[sm.idx_Tgas], 200.0)
    eps_th = 1.5 * KB * T_gs / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean
    tau = solver.flow.get_residence_time(y[sm.idx_Tgas])

    return conv_ch4, conv_co2, Te_eV, n_e, tau, result.wall_time


if __name__ == '__main__':
    print(f'V_reactor Parameter Sweep for Experimental Fit')
    print(f'V_eff={V_eff_cm3}cm³, P={P_W}W, Q={Q_slm}slm')
    print(f'T_gas = {T_gas_list_K} K  ({[T-273 for T in T_gas_list_K]} °C)')
    print(f'V_reactor = {V_reactor_list_cm3} cm³')
    n_total = len(V_reactor_list_cm3) * len(T_gas_list_K)
    print(f'Total runs: {len(V_reactor_list_cm3)} × {len(T_gas_list_K)} = {n_total}')
    print()

    results = {}
    t_start = time.time()
    run_count = 0

    for v_r in V_reactor_list_cm3:
        results[v_r] = {}
        for T_K in T_gas_list_K:
            run_count += 1
            print(f'  [{run_count:2d}/{n_total}] V_r={v_r}cm³, T={T_K}K ({T_K-273}°C) ...',
                  end='', flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    conv_ch4, conv_co2, Te, n_e, tau, wt = run_single(T_K, v_r)
                except Exception as e:
                    print(f' ERROR: {e}')
                    conv_ch4, conv_co2, Te, n_e, tau, wt = -999, -999, -1, 0, 0, 0
            results[v_r][T_K] = {
                'ch4': conv_ch4, 'co2': conv_co2, 'Te': Te,
                'n_e': n_e, 'tau': tau, 'wt': wt
            }
            f_val = V_eff_cm3 / v_r
            print(f' CH4={conv_ch4:+6.1f}% Te={Te:.3f}eV τ={tau*1e3:.0f}ms f={f_val:.4f} ({wt:.0f}s)',
                  flush=True)

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time/60:.1f} min')

    # =================================================================
    # CH4 CONVERSION TABLE
    # =================================================================
    print(f'\n{"="*120}')
    print(f'  V_reactor SWEEP: CH4 Conversion (%)')
    print(f'{"="*120}')

    header = f'  {"V_r(cm³)":>9} {"f":>7} |'
    for T_K in T_gas_list_K:
        header += f' {T_K-273:>6}°C'
    header += f' |  {"RMSE":>6}  {"MAE":>6}  {"Max|Δ|":>6}'
    print(header)
    print(f'  {"-"*100}')

    line_exp = f'  {"EXP":>9} {"":>7} |'
    for T_K in T_gas_list_K:
        line_exp += f' {EXP_DATA[T_K]:7.2f}'
    line_exp += f' |  {"---":>6}  {"---":>6}  {"---":>6}'
    print(line_exp)
    print(f'  {"-"*100}')

    best_rmse = 1e10
    best_vr = None

    for v_r in V_reactor_list_cm3:
        f_val = V_eff_cm3 / v_r
        line = f'  {v_r:>9.0f} {f_val:7.4f} |'
        errors = []
        for T_K in T_gas_list_K:
            conv = results[v_r][T_K]['ch4']
            line += f' {conv:7.2f}'
            errors.append(conv - EXP_DATA[T_K])

        errors_arr = np.array(errors)
        rmse = np.sqrt(np.mean(errors_arr**2))
        mae = np.mean(np.abs(errors_arr))
        max_err = np.max(np.abs(errors_arr))
        line += f' |  {rmse:6.2f}  {mae:6.2f}  {max_err:6.2f}'

        if rmse < best_rmse:
            best_rmse = rmse
            best_vr = v_r

        if v_r == best_vr:
            line += '  ←'
        print(line)

    # =================================================================
    # BEST FIT DETAIL
    # =================================================================
    print(f'\n{"="*90}')
    print(f'  BEST FIT: V_reactor = {best_vr} cm³  (f = {V_eff_cm3/best_vr:.4f}, RMSE = {best_rmse:.2f}%)')
    print(f'{"="*90}')

    print(f'\n  {"T(°C)":>6} {"T(K)":>6} {"Exp(%)":>8} {"Sim(%)":>8} {"Δ(%)":>8} {"sim/exp":>8} '
          f'{"Te(eV)":>8} {"n_e(m⁻³)":>12} {"τ(ms)":>8}')
    print(f'  {"-"*85}')
    for T_K in T_gas_list_K:
        r = results[best_vr][T_K]
        exp_val = EXP_DATA[T_K]
        delta = r['ch4'] - exp_val
        ratio = r['ch4'] / exp_val if exp_val > 0 else 0
        print(f'  {T_K-273:6d} {T_K:6d} {exp_val:8.2f} {r["ch4"]:8.2f} {delta:+8.2f} {ratio:8.3f} '
              f'{r["Te"]:8.3f} {r["n_e"]:12.2e} {r["tau"]*1e3:8.1f}')

    # =================================================================
    # Te / n_e TABLE (all V_reactor)
    # =================================================================
    print(f'\n{"="*120}')
    print(f'  PLASMA PARAMETERS: Te (eV) at each V_reactor')
    print(f'{"="*120}')
    header2 = f'  {"V_r(cm³)":>9} |'
    for T_K in T_gas_list_K:
        header2 += f'   Te@{T_K-273}°C  n_e@{T_K-273}°C'
    print(header2)
    print(f'  {"-"*100}')
    for v_r in V_reactor_list_cm3:
        line = f'  {v_r:>9.0f} |'
        for T_K in T_gas_list_K:
            r = results[v_r][T_K]
            line += f'   {r["Te"]:.3f}  {r["n_e"]:.2e}'
        print(line)
