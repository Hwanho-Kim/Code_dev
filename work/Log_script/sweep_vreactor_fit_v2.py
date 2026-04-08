"""V_reactor re-fit with proper steady-state convergence (t_end = 5*tau).

Previous fit (RMSE=1.14%) used t_end capped at 15s, which was < 1*tau
for V_reactor=250cm³. This sweep uses 5*tau to ensure true steady state.
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

# === Experimental data ===
EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
T_GAS_LIST = list(EXP_DATA.keys())

# === Fixed conditions ===
V_EFF_CM3 = 1.6
P_W = 5.0
Q_SLM = 0.4

# === V_reactor sweep ===
V_REACTOR_LIST = [20, 30, 40, 50, 60, 70, 80, 90, 100,
                  120, 150, 180, 210, 250, 300]


def run_single(T_K, V_reactor_cm3):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    V_reactor_m3 = V_reactor_cm3 * 1e-6
    cfg['V_eff'] = V_EFF_CM3 * 1e-6
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_SLM
    cfg['T_wall'] = T_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_K

    Q_actual = Q_SLM * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau = V_reactor_m3 / Q_actual
    t_end = max(10.0, 5.0 * tau)

    cfg['solver'] = {
        't_end': t_end, 'n_points': 150, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
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

    n_sp = sm.n_species
    y = result.y[:, -1]
    c_e = max(y[0], 1e-30)
    n_e = c_e * NA
    ne_eps = y[sm.idx_energy]
    T_gas = y[sm.idx_Tgas]
    eps_th = 1.5 * KB * max(T_gas, 200) / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean

    return conv, Te_eV, n_e, tau, result.wall_time


if __name__ == '__main__':
    print(f'V_reactor Re-Fit (steady-state converged, t_end=5*tau)')
    print(f'V_eff={V_EFF_CM3}cm³, P={P_W}W, Q={Q_SLM}slm')
    print(f'V_reactor = {V_REACTOR_LIST} cm³')
    n_total = len(V_REACTOR_LIST) * len(T_GAS_LIST)
    print(f'Total runs: {len(V_REACTOR_LIST)} x {len(T_GAS_LIST)} = {n_total}')
    print()

    results = {}
    t_start = time.time()
    run_count = 0

    for v_r in V_REACTOR_LIST:
        results[v_r] = {}
        for T_K in T_GAS_LIST:
            run_count += 1
            print(f'  [{run_count:2d}/{n_total}] V_r={v_r:3d}cm³ T={T_K}K ...',
                  end='', flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    conv, Te, n_e, tau, wt = run_single(T_K, v_r)
                except Exception as e:
                    print(f' ERROR: {e}')
                    conv, Te, n_e, tau, wt = -999, -1, 0, 0, 0
            results[v_r][T_K] = {'conv': conv, 'Te': Te, 'n_e': n_e, 'tau': tau}
            print(f' CH4={conv:5.1f}% Te={Te:.3f}eV τ={tau:.1f}s ({wt:.0f}s)')

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time/60:.1f} min')

    # ============================================================
    # TABLE 1: CH4 Conversion
    # ============================================================
    print(f'\n{"="*120}')
    print(f'  CH4 CONVERSION (%) — Steady State')
    print(f'{"="*120}')

    header = f'  {"V_r":>6} {"f":>7} {"tau30":>6} |'
    for T_K in T_GAS_LIST:
        header += f' {T_K-273:>6}°C'
    header += f' | {"RMSE":>6} {"MAE":>6} {"MaxΔ":>6}'
    print(header)
    print(f'  {"-"*100}')

    # Exp row
    line = f'  {"EXP":>6} {"":>7} {"":>6} |'
    for T_K in T_GAS_LIST:
        line += f' {EXP_DATA[T_K]:7.2f}'
    print(line)
    print(f'  {"-"*100}')

    best_rmse = 1e10
    best_vr = None
    all_rmse = []

    for v_r in V_REACTOR_LIST:
        f_val = V_EFF_CM3 / v_r
        tau30 = results[v_r][303]['tau']
        line = f'  {v_r:6d} {f_val:7.4f} {tau30:6.1f} |'
        errors = []
        for T_K in T_GAS_LIST:
            conv = results[v_r][T_K]['conv']
            line += f' {conv:7.2f}'
            errors.append(conv - EXP_DATA[T_K])

        err = np.array(errors)
        rmse = np.sqrt(np.mean(err**2))
        mae = np.mean(np.abs(err))
        max_err = np.max(np.abs(err))
        all_rmse.append((v_r, rmse))
        line += f' | {rmse:6.2f} {mae:6.2f} {max_err:6.2f}'

        if rmse < best_rmse:
            best_rmse = rmse
            best_vr = v_r
            line += '  <--'
        print(line)

    # ============================================================
    # BEST FIT DETAIL
    # ============================================================
    print(f'\n{"="*90}')
    print(f'  BEST FIT: V_reactor = {best_vr} cm³ (f={V_EFF_CM3/best_vr:.4f}, RMSE={best_rmse:.2f}%)')
    print(f'{"="*90}')
    print(f'  {"T(°C)":>6} {"T(K)":>6} {"Exp%":>8} {"Sim%":>8} {"Δ(%p)":>8} '
          f'{"Te(eV)":>8} {"n_e(m⁻³)":>12} {"τ(s)":>8}')
    print(f'  {"-"*75}')
    for T_K in T_GAS_LIST:
        r = results[best_vr][T_K]
        exp = EXP_DATA[T_K]
        delta = r['conv'] - exp
        print(f'  {T_K-273:6d} {T_K:6d} {exp:8.2f} {r["conv"]:8.2f} {delta:+8.2f} '
              f'{r["Te"]:8.3f} {r["n_e"]:12.3e} {r["tau"]:8.2f}')

    # ============================================================
    # RMSE vs V_reactor
    # ============================================================
    print(f'\n{"="*60}')
    print(f'  RMSE vs V_reactor')
    print(f'{"="*60}')
    for v_r, rmse in all_rmse:
        bar = '#' * int(rmse / 0.5)
        mark = ' <-- BEST' if v_r == best_vr else ''
        print(f'  {v_r:4d} cm³ | RMSE={rmse:6.2f} | {bar}{mark}')
