"""Λ steady-state convergence test.

Compare Λ=1mm vs Λ=10mm at T_gas=303K with t_end=120s (≈3.5τ).
V_reactor=250cm³, V_eff=1.6cm³, P=5W, Q=0.4slm, constant mode.

Goal: Determine if the Λ-dependent Te/n_e difference persists at true steady state,
or if it was an artifact of insufficient simulation time (t_end=15s < τ=33.8s).
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP


# === Fixed conditions ===
T_GAS_K = 303.0       # 30°C — worst case (longest τ)
V_EFF_CM3 = 1.6
V_REACTOR_CM3 = 250.0
P_W = 5.0
Q_SLM = 0.4
T_END = 120.0          # seconds — ≈3.5τ at 303K

# === Λ cases ===
LAMBDA_CASES = {
    '1mm':  1.0e-3,
    '10mm': 1.0e-2,
}


def run_case(lambda_m, label):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    cfg['V_eff'] = V_EFF_CM3 * 1e-6
    cfg['reactor']['volume'] = V_REACTOR_CM3 * 1e-6
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_SLM
    cfg['Lambda'] = lambda_m

    cfg['T_wall'] = T_GAS_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_GAS_K

    # Solver: long run, coarser output for speed
    cfg['solver'] = {
        't_end': T_END, 'n_points': 500, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
    }

    print(f'\n  [{label}] Λ={lambda_m*1e3:.1f}mm, t_end={T_END}s', flush=True)
    t0 = time.time()

    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
    scfg = cfg['solver']

    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                              rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    wall_time = time.time() - t0
    print(f'  [{label}] Done in {wall_time:.0f}s', flush=True)

    # Extract time series
    sm = solver.sm
    n_sp = sm.n_species
    ch4_idx = sm.index('CH4')

    t_arr = result.t
    n_pts = len(t_arr)

    Te_arr = np.zeros(n_pts)
    ne_arr = np.zeros(n_pts)
    ch4_conv_arr = np.zeros(n_pts)
    c0_ch4 = result.concentrations[ch4_idx, 0]

    for i in range(n_pts):
        y = result.y[:, i]
        c_e = max(y[0], 1e-30)
        n_e = c_e * NA
        ne_eps = y[sm.idx_energy]
        T_g = max(y[sm.idx_Tgas], 200.0)
        eps_th = 1.5 * KB * T_g / QE
        eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)

        Te_arr[i] = (2.0 / 3.0) * eps_mean
        ne_arr[i] = n_e
        ch4_conv_arr[i] = (c0_ch4 - result.concentrations[ch4_idx, i]) / c0_ch4 * 100 if c0_ch4 > 0 else 0

    return {
        't': t_arr, 'Te': Te_arr, 'ne': ne_arr, 'ch4_conv': ch4_conv_arr,
        'wall_time': wall_time
    }


def check_convergence(data, label, window_frac=0.1):
    """Check if last window_frac of data is converged (< 1% change)."""
    n = len(data['t'])
    i_start = int(n * (1 - window_frac))

    Te_last = data['Te'][i_start:]
    ne_last = data['ne'][i_start:]
    ch4_last = data['ch4_conv'][i_start:]

    Te_var = (Te_last.max() - Te_last.min()) / max(abs(Te_last.mean()), 1e-30) * 100
    ne_var = (ne_last.max() - ne_last.min()) / max(abs(ne_last.mean()), 1e-30) * 100
    ch4_var = (ch4_last.max() - ch4_last.min()) / max(abs(ch4_last.mean()), 1e-3) * 100

    converged = Te_var < 1.0 and ne_var < 1.0 and ch4_var < 1.0
    return converged, Te_var, ne_var, ch4_var


if __name__ == '__main__':
    Q_actual = Q_SLM * (T_GAS_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau = (V_REACTOR_CM3 * 1e-6) / Q_actual

    print(f'=== Λ Steady-State Convergence Test ===')
    print(f'T_gas={T_GAS_K}K ({T_GAS_K-273:.0f}°C), V_eff={V_EFF_CM3}cm³, V_reactor={V_REACTOR_CM3}cm³')
    print(f'P={P_W}W, Q={Q_SLM}slm, constant mode')
    print(f'τ = {tau:.1f}s, t_end = {T_END}s ({T_END/tau:.1f}τ)')
    print(f'Λ cases: {list(LAMBDA_CASES.keys())}')

    results = {}
    for label, lam in LAMBDA_CASES.items():
        results[label] = run_case(lam, label)

    # === Convergence check ===
    print(f'\n{"="*80}')
    print(f'  CONVERGENCE CHECK (last 10% of time series)')
    print(f'{"="*80}')
    print(f'  {"Case":>6} {"Te var%":>8} {"n_e var%":>9} {"CH4 var%":>9} {"Converged?":>11}')
    print(f'  {"-"*50}')
    for label in LAMBDA_CASES:
        conv, Te_v, ne_v, ch4_v = check_convergence(results[label], label)
        status = "YES ✓" if conv else "NO ✗"
        print(f'  {label:>6} {Te_v:8.3f} {ne_v:9.3f} {ch4_v:9.3f} {status:>11}')

    # === Snapshot at multiple times ===
    print(f'\n{"="*80}')
    print(f'  TIME EVOLUTION SNAPSHOTS')
    print(f'{"="*80}')

    snap_times = [1, 5, 10, 15, 30, 50, 80, 100, 120]

    for label in LAMBDA_CASES:
        d = results[label]
        print(f'\n  [{label}] Λ={LAMBDA_CASES[label]*1e3:.0f}mm:')
        print(f'  {"t(s)":>7} {"t/τ":>6} {"Te(eV)":>8} {"n_e(m⁻³)":>12} {"CH4 conv%":>10}')
        print(f'  {"-"*50}')
        for ts in snap_times:
            if ts > d['t'][-1]:
                break
            idx = np.argmin(np.abs(d['t'] - ts))
            ratio = ts / tau
            print(f'  {ts:7.0f} {ratio:6.2f} {d["Te"][idx]:8.4f} {d["ne"][idx]:12.3e} {d["ch4_conv"][idx]:10.3f}')

    # === Final comparison ===
    print(f'\n{"="*80}')
    print(f'  FINAL STATE COMPARISON (t = {T_END}s ≈ {T_END/tau:.1f}τ)')
    print(f'{"="*80}')
    print(f'  {"":>6} {"Te(eV)":>8} {"n_e(m⁻³)":>12} {"CH4 conv%":>10}')
    print(f'  {"-"*45}')
    finals = {}
    for label in LAMBDA_CASES:
        d = results[label]
        finals[label] = {'Te': d['Te'][-1], 'ne': d['ne'][-1], 'ch4': d['ch4_conv'][-1]}
        print(f'  {label:>6} {d["Te"][-1]:8.4f} {d["ne"][-1]:12.3e} {d["ch4_conv"][-1]:10.3f}')

    # Relative differences
    f1 = finals['1mm']
    f10 = finals['10mm']
    dTe = (f10['Te'] - f1['Te']) / f1['Te'] * 100
    dne = (f10['ne'] - f1['ne']) / f1['ne'] * 100
    dch4 = f10['ch4'] - f1['ch4']
    print(f'\n  Δ(10mm vs 1mm):')
    print(f'    Te:   {dTe:+.2f}%')
    print(f'    n_e:  {dne:+.2f}%')
    print(f'    CH4:  {dch4:+.3f}%p')

    # === D_a/Λ² analysis at final state ===
    print(f'\n{"="*80}')
    print(f'  D_a/Λ² AT FINAL STATE')
    print(f'{"="*80}')
    N_gas = 101325.0 / (KB * T_GAS_K)
    mu_i_N = 2.8e22
    mu_i = mu_i_N / N_gas

    for label, lam in LAMBDA_CASES.items():
        Te = finals[label]['Te']
        D_a = mu_i * Te
        diff_rate = D_a / (lam**2)
        print(f'  [{label}] Λ={lam*1e3:.0f}mm: D_a={D_a:.4f} m²/s, D_a/Λ²={diff_rate:.1f} s⁻¹')

    print(f'\n  Total wall time: {sum(r["wall_time"] for r in results.values())/60:.1f} min')
    print(f'  Done.')
