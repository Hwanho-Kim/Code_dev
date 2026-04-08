"""V_eff parameter sweep to fit experimental CH4 conversion.

Experimental data (T_gas °C → CH4 conversion %):
  30°C  →  5.26%
  100°C →  8.05%
  180°C → 14.36%
  250°C → 20.02%

Sweep V_eff from 0.4 to 1.6 cm³, find best fit (min RMSE).
V_reactor=100cm³, P=5W, Q=0.4slm, constant mode.
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
V_reactor_cm3 = 100
P_W = 5.0
Q_slm = 0.4

# === V_eff sweep ===
V_eff_list_cm3 = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]


def run_single(T_target_K, V_eff_cm3):
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
    c0 = result.concentrations[ch4_idx, 0]
    cf = result.concentrations[ch4_idx, -1]
    conv = (c0 - cf) / c0 * 100 if c0 > 0 else 0

    # Extract Te and n_e
    n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    ne_eps = y[sm.idx_energy]
    n_e = c[0] * NA
    T_gs = max(y[sm.idx_Tgas], 200.0)
    eps_th = 1.5 * KB * T_gs / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean

    return conv, Te_eV, n_e, result.wall_time


if __name__ == '__main__':
    print(f'V_eff Parameter Sweep for Experimental Fit')
    print(f'V_reactor={V_reactor_cm3}cm³, P={P_W}W, Q={Q_slm}slm')
    print(f'T_gas = {T_gas_list_K} K  ({[T-273 for T in T_gas_list_K]} °C)')
    print(f'V_eff = {V_eff_list_cm3} cm³')
    print(f'Total runs: {len(V_eff_list_cm3)} × {len(T_gas_list_K)} = {len(V_eff_list_cm3)*len(T_gas_list_K)}')
    print()

    # results[v_eff_cm3][T_K] = (conv, Te, n_e, wall_time)
    results = {}
    t_start = time.time()
    total_runs = len(V_eff_list_cm3) * len(T_gas_list_K)
    run_count = 0

    for v_eff in V_eff_list_cm3:
        results[v_eff] = {}
        for T_K in T_gas_list_K:
            run_count += 1
            print(f'  [{run_count:2d}/{total_runs}] V_eff={v_eff:.1f}cm³, T={T_K}K ({T_K-273}°C) ...',
                  end='', flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    conv, Te, n_e, wt = run_single(T_K, v_eff)
                except Exception as e:
                    print(f' ERROR: {e}')
                    conv, Te, n_e, wt = -999, -1, 0, 0
            results[v_eff][T_K] = (conv, Te, n_e, wt)
            print(f' CH4={conv:+6.1f}% Te={Te:.3f}eV ({wt:.0f}s)', flush=True)

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time/60:.1f} min')

    # =================================================================
    # RESULTS TABLE
    # =================================================================
    print(f'\n{"="*120}')
    print(f'  V_eff SWEEP RESULTS: CH4 Conversion (%)')
    print(f'{"="*120}')

    # Header
    header = f'  {"V_eff(cm³)":>10} |'
    for T_K in T_gas_list_K:
        header += f' {T_K-273:>5}°C'
    header += f' |  {"RMSE":>6}  {"MAE":>6}  {"Max|Δ|":>6}'
    print(header)
    print(f'  {"-"*len(header)}')

    # Experimental row
    line_exp = f'  {"EXP":>10} |'
    for T_K in T_gas_list_K:
        line_exp += f' {EXP_DATA[T_K]:6.2f}'
    line_exp += f' |  {"---":>6}  {"---":>6}  {"---":>6}'
    print(line_exp)
    print(f'  {"-"*len(header)}')

    # Simulation rows + compute errors
    best_rmse = 1e10
    best_veff = None
    error_table = {}

    for v_eff in V_eff_list_cm3:
        line = f'  {v_eff:>10.1f} |'
        errors = []
        for T_K in T_gas_list_K:
            conv = results[v_eff][T_K][0]
            exp_val = EXP_DATA[T_K]
            line += f' {conv:6.2f}'
            errors.append(conv - exp_val)

        errors_arr = np.array(errors)
        rmse = np.sqrt(np.mean(errors_arr**2))
        mae = np.mean(np.abs(errors_arr))
        max_err = np.max(np.abs(errors_arr))
        line += f' |  {rmse:6.2f}  {mae:6.2f}  {max_err:6.2f}'

        error_table[v_eff] = {'rmse': rmse, 'mae': mae, 'max_err': max_err, 'errors': errors}

        if rmse < best_rmse:
            best_rmse = rmse
            best_veff = v_eff

        # Mark best
        if v_eff == best_veff:
            line += '  ← best'
        print(line)

    # =================================================================
    # DETAILED COMPARISON FOR BEST V_eff
    # =================================================================
    print(f'\n{"="*90}')
    print(f'  BEST FIT: V_eff = {best_veff} cm³  (RMSE = {best_rmse:.2f}%)')
    print(f'{"="*90}')

    print(f'\n  {"T(°C)":>6} {"T(K)":>6} {"Exp(%)":>8} {"Sim(%)":>8} {"Δ(%)":>8} {"sim/exp":>8}')
    print(f'  {"-"*50}')
    for T_K in T_gas_list_K:
        conv = results[best_veff][T_K][0]
        exp_val = EXP_DATA[T_K]
        delta = conv - exp_val
        ratio = conv / exp_val if exp_val > 0 else 0
        print(f'  {T_K-273:6d} {T_K:6d} {exp_val:8.2f} {conv:8.2f} {delta:+8.2f} {ratio:8.3f}')

    # =================================================================
    # Te and n_e at best V_eff
    # =================================================================
    print(f'\n  Plasma parameters at V_eff = {best_veff} cm³:')
    print(f'  {"T(°C)":>6} {"Te(eV)":>8} {"n_e(m⁻³)":>12} {"f=V/Vr":>8}')
    print(f'  {"-"*40}')
    f_best = best_veff * 1e-6 / (V_reactor_cm3 * 1e-6)
    for T_K in T_gas_list_K:
        _, Te, n_e, _ = results[best_veff][T_K]
        print(f'  {T_K-273:6d} {Te:8.3f} {n_e:12.2e} {f_best:8.4f}')
