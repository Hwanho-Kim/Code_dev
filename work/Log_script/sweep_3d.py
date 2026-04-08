"""3D parametric sweep: V_eff × V_reactor × wall_loss_freq
5×5×5 = 125 cases. P=5W, Q=0.4slm.
Find conditions giving ~5% CH4 conversion with physically reasonable Te.
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

# === Fixed experimental conditions ===
P_W = 5.0
Q_slm = 0.4

# === Sweep variables (5 levels each) ===
V_eff_cm3_list = [0.2, 0.4, 0.8, 1.6, 3.2]
V_reactor_cm3_list = [5, 10, 20, 50, 100]
wlf_list = [10, 30, 100, 300, 1000]


def estimate_tau(V_reactor_m3):
    Q_actual = Q_slm * (300.0 / T_STP) * (P_STP / 101325.0) / 60000.0
    return V_reactor_m3 / Q_actual


def run_and_analyze(V_eff_m3, V_reactor_m3, wlf):
    tau_est = estimate_tau(V_reactor_m3)
    t_end = min(max(3.0, 1.5 * tau_est), 15.0)

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))
    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm
    cfg['wall_loss_freq'] = wlf
    cfg['solver'] = {
        't_end': t_end, 'n_points': 100, 'method': 'BDF',
        'rtol': 1e-5, 'atol': 1e-10, 'max_step': 5e-4, 'constrained': False
    }

    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
    scfg = cfg['solver']
    result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                          rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    # Analyze final state
    sm = solver.sm; n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    T_gas = y[sm.idx_Tgas]; ne_eps = y[sm.idx_energy]
    c_e = c[0]; n_e = c_e * NA
    T_gs = max(T_gas, 200.0)
    eps_th = 1.5 * KB * T_gs / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean

    ch4 = sm.index('CH4')
    c0 = result.concentrations[ch4, 0]; cf = result.concentrations[ch4, -1]
    ch4_conv = (c0 - cf) / c0 * 100 if c0 > 0 else 0

    return {
        'Te': Te_eV, 'n_e': n_e, 'Tgas': T_gas,
        'ch4_conv': ch4_conv, 'wall_time': result.wall_time,
        't_end': t_end, 'tau_est': tau_est,
    }


if __name__ == '__main__':
    total = len(V_eff_cm3_list) * len(V_reactor_cm3_list) * len(wlf_list)
    print(f'3D Sweep: {len(V_eff_cm3_list)}x{len(V_reactor_cm3_list)}x{len(wlf_list)} = {total} cases')
    print(f'P={P_W}W, Q={Q_slm}slm')
    print(f'V_eff  = {V_eff_cm3_list} cm3')
    print(f'V_rct  = {V_reactor_cm3_list} cm3')
    print(f'wlf    = {wlf_list} s-1')

    results = {}
    count = 0
    t_start = time.time()

    for wlf in wlf_list:
        for vr_cm3 in V_reactor_cm3_list:
            for ve_cm3 in V_eff_cm3_list:
                count += 1
                ve_m3 = ve_cm3 * 1e-6
                vr_m3 = vr_cm3 * 1e-6

                # Suppress verbose output
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        r = run_and_analyze(ve_m3, vr_m3, wlf)
                    except Exception as e:
                        r = {'Te': -1, 'n_e': 0, 'Tgas': 0,
                             'ch4_conv': -999, 'wall_time': 0,
                             't_end': 0, 'tau_est': 0, 'error': str(e)}

                results[(ve_cm3, vr_cm3, wlf)] = r

                elapsed = time.time() - t_start
                eta = elapsed / count * (total - count)
                marker = '*' if 4.0 <= r['ch4_conv'] <= 6.5 else ' '
                print(f'  [{count:3d}/{total}] Ve={ve_cm3:4.1f} Vr={vr_cm3:4.0f} wlf={wlf:4.0f} |'
                      f' Te={r["Te"]:5.2f}eV Tg={r["Tgas"]:5.0f}K CH4={r["ch4_conv"]:+6.1f}%{marker}|'
                      f' {r["wall_time"]:4.0f}s (ETA {eta/60:.0f}min)', flush=True)

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time/60:.1f} min')

    # =================================================================
    # SUMMARY: 2D tables (V_eff × V_reactor) for each wall_loss_freq
    # =================================================================
    print(f'\n{"="*130}')
    print(f'  3D SWEEP RESULTS: P={P_W}W, Q={Q_slm}slm')
    print(f'  CH4 conversion [%] (Te[eV] / Tgas[K])')
    print(f'{"="*130}')

    for wlf in wlf_list:
        print(f'\n  === wall_loss_freq = {wlf} s⁻¹ ===')
        # Header
        print(f'  {"Ve\\Vr":>8}', end='')
        for vr in V_reactor_cm3_list:
            print(f' | {"Vr="+str(vr):^18}', end='')
        print(' |')
        print(f'  {"(cm3)":>8}', end='')
        for vr in V_reactor_cm3_list:
            print(f' | {"CH4%":>6} {"Te":>5} {"Tg":>5}', end='')
        print(' |')
        print('  ' + '-' * (9 + 20 * len(V_reactor_cm3_list)))

        for ve in V_eff_cm3_list:
            print(f'  {ve:8.1f}', end='')
            for vr in V_reactor_cm3_list:
                r = results.get((ve, vr, wlf), {})
                ch4 = r.get('ch4_conv', -999)
                te = r.get('Te', -1)
                tg = r.get('Tgas', 0)
                if ch4 == -999:
                    print(f' |  FAIL  ----  ----', end='')
                else:
                    m = '*' if 4.0 <= ch4 <= 6.5 else ' '
                    print(f' | {ch4:+5.1f}{m} {te:5.2f} {tg:5.0f}', end='')
            print(' |')

    # =================================================================
    # BEST CANDIDATES
    # =================================================================
    print(f'\n{"="*130}')
    print(f'  BEST CANDIDATES')
    print(f'{"="*130}')

    # Tier 1: CH4 ~ 5% AND Te <= 2.0
    print(f'\n  [Tier 1] 4% <= CH4 <= 7% AND Te <= 2.0 eV:')
    print(f'  {"Ve":>5} {"Vr":>5} {"wlf":>5} {"f":>7} {"Te":>6} {"n_e":>11} {"Tg":>5} {"CH4%":>7}')
    print('  ' + '-' * 65)
    t1 = [(ve, vr, w, r) for (ve, vr, w), r in results.items()
          if 4.0 <= r.get('ch4_conv', 0) <= 7.0 and r.get('Te', 99) <= 2.0]
    t1.sort(key=lambda x: abs(x[3]['ch4_conv'] - 5.0))
    for ve, vr, w, r in t1[:15]:
        print(f'  {ve:5.1f} {vr:5.0f} {w:5.0f} {ve/vr:7.4f} {r["Te"]:6.2f} '
              f'{r["n_e"]:11.2e} {r["Tgas"]:5.0f} {r["ch4_conv"]:+7.2f}')
    if not t1:
        print('  (none)')

    # Tier 2: CH4 ~ 5% (any Te)
    print(f'\n  [Tier 2] 4% <= CH4 <= 7% (any Te):')
    print(f'  {"Ve":>5} {"Vr":>5} {"wlf":>5} {"f":>7} {"Te":>6} {"n_e":>11} {"Tg":>5} {"CH4%":>7}')
    print('  ' + '-' * 65)
    t2 = [(ve, vr, w, r) for (ve, vr, w), r in results.items()
          if 4.0 <= r.get('ch4_conv', 0) <= 7.0]
    t2.sort(key=lambda x: abs(x[3]['ch4_conv'] - 5.0))
    for ve, vr, w, r in t2[:15]:
        print(f'  {ve:5.1f} {vr:5.0f} {w:5.0f} {ve/vr:7.4f} {r["Te"]:6.2f} '
              f'{r["n_e"]:11.2e} {r["Tgas"]:5.0f} {r["ch4_conv"]:+7.2f}')
    if not t2:
        print('  (none)')

    # Tier 3: closest to 5% regardless
    print(f'\n  [Tier 3] Top 15 closest to CH4=5%:')
    print(f'  {"Ve":>5} {"Vr":>5} {"wlf":>5} {"f":>7} {"Te":>6} {"n_e":>11} {"Tg":>5} {"CH4%":>7}')
    print('  ' + '-' * 65)
    all_cases = [(ve, vr, w, r) for (ve, vr, w), r in results.items()
                 if r.get('ch4_conv', -999) > -900]
    all_cases.sort(key=lambda x: abs(x[3]['ch4_conv'] - 5.0))
    for ve, vr, w, r in all_cases[:15]:
        print(f'  {ve:5.1f} {vr:5.0f} {w:5.0f} {ve/vr:7.4f} {r["Te"]:6.2f} '
              f'{r["n_e"]:11.2e} {r["Tgas"]:5.0f} {r["ch4_conv"]:+7.2f}')

    # =================================================================
    # TREND ANALYSIS
    # =================================================================
    print(f'\n{"="*130}')
    print(f'  TREND ANALYSIS (sensitivity)')
    print(f'{"="*130}')

    # Effect of each variable (holding others at middle value)
    ve_mid = 0.8; vr_mid = 20; wlf_mid = 100

    print(f'\n  V_eff sensitivity (Vr={vr_mid}, wlf={wlf_mid}):')
    for ve in V_eff_cm3_list:
        r = results.get((ve, vr_mid, wlf_mid), {})
        print(f'    Ve={ve:4.1f}cm³: CH4={r.get("ch4_conv",0):+5.1f}%, '
              f'Te={r.get("Te",0):.2f}eV, Tg={r.get("Tgas",0):.0f}K')

    print(f'\n  V_reactor sensitivity (Ve={ve_mid}, wlf={wlf_mid}):')
    for vr in V_reactor_cm3_list:
        r = results.get((ve_mid, vr, wlf_mid), {})
        print(f'    Vr={vr:4.0f}cm³: CH4={r.get("ch4_conv",0):+5.1f}%, '
              f'Te={r.get("Te",0):.2f}eV, Tg={r.get("Tgas",0):.0f}K')

    print(f'\n  wall_loss_freq sensitivity (Ve={ve_mid}, Vr={vr_mid}):')
    for w in wlf_list:
        r = results.get((ve_mid, vr_mid, w), {})
        print(f'    wlf={w:5.0f}s⁻¹: CH4={r.get("ch4_conv",0):+5.1f}%, '
              f'Te={r.get("Te",0):.2f}eV, Tg={r.get("Tgas",0):.0f}K')
