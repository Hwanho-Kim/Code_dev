"""V_eff sweep: PFR 1τ, constant P=5W, Q=0.4slm.

sDBD area = 70mm × 70mm = 49 cm².
V_eff = area × gap.
Sweep from 0.49 cm³ (gap 0.1mm) to 250 cm³ (V_reactor).

PFR mode: no flow source/sink in ODE, t_end = V_eff / Q_actual = 1τ.
P_dep = P / V_eff [W/m³].
SEI = P/Q = const for all V_eff → 1st-order reactions identical.
Differences arise from nonlinear reactions (radical recombination, ion-ion).

2026-03-27
"""
import sys, os, io, contextlib, time, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, T_STP, P_STP

# === Experimental data ===
EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
T_LIST = sorted(EXP_DATA.keys())

# === Fixed conditions ===
P_W = 5.0
Q_SLM = 0.4
INLET = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

# === V_eff sweep (cm³) ===
# gap(mm): 0.1   0.2   0.5   1.0   2.0   5.0   10    20    ~51
V_EFF_LIST = [0.49, 0.98, 2.45, 4.90, 9.80, 24.5, 49.0, 98.0, 250.0]


def run_single(T_K, V_eff_cm3):
    """Run PFR 1τ simulation at given T and V_eff."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    V_eff_m3 = V_eff_cm3 * 1e-6

    # Override config
    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = max(V_eff_m3, 250e-6)
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_SLM
    cfg['flow']['model'] = 'PFR'
    cfg['inlet_composition'] = INLET

    cfg['T_wall'] = T_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_K

    # PFR 1τ
    Q_actual = Q_SLM * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau = V_eff_m3 / Q_actual
    t_end = tau

    cfg['solver'] = {
        't_end': t_end,
        'n_points': max(200, int(t_end / 0.01)),
        'method': 'BDF',
        'rtol': 1e-6,
        'atol': 1e-12,
        'max_step': min(0.1, tau / 50),
        'constrained': False,
    }

    solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)
    scfg = cfg_out['solver']
    result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                          rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    sm = solver.sm
    ch4_idx = sm.index('CH4')
    c0 = result.concentrations[ch4_idx, 0]
    cf = result.concentrations[ch4_idx, -1]
    conv = (c0 - cf) / c0 * 100 if c0 > 0 else 0

    # Plasma parameters
    n_sp = sm.n_species
    y = result.y[:, -1]
    n_e = max(y[0], 1e-30) * NA
    ne_eps = y[sm.idx_energy]
    T_gas = max(y[sm.idx_Tgas], 200.0)
    eps_th = 1.5 * KB * T_gas / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean

    return conv, Te_eV, n_e, tau, result.wall_time


if __name__ == '__main__':
    total = len(V_EFF_LIST) * len(T_LIST)
    print(f'V_eff Sweep (PFR 1τ, constant P={P_W}W, Q={Q_SLM}slm)')
    print(f'V_eff = {V_EFF_LIST} cm³')
    print(f'Temps = {T_LIST} K')
    print(f'Total: {len(V_EFF_LIST)} × {len(T_LIST)} = {total} runs\n')

    results = {}
    t0 = time.time()
    n = 0

    for v in V_EFF_LIST:
        results[v] = {}
        for T in T_LIST:
            n += 1
            gap_mm = v / 4.9
            print(f'  [{n:2d}/{total}] V={v:>6.1f}cm³ (gap={gap_mm:.2f}mm) T={T}K ...',
                  end='', flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    conv, Te, ne, tau, wt = run_single(T, v)
                except Exception as e:
                    print(f' ERROR: {e}')
                    import traceback; traceback.print_exc()
                    conv, Te, ne, tau, wt = -999, -1, 0, 0, 0
            results[v][T] = (conv, Te, ne, tau, wt)
            status = f' CH4={conv:6.2f}%  τ={tau:.4f}s  ({wt:.1f}s)'
            print(status, flush=True)

    elapsed = time.time() - t0
    print(f'\nTotal elapsed: {elapsed/60:.1f} min')

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print(f'\n{"="*110}')
    print(f'  V_eff SWEEP (PFR 1τ): CH4 Conversion (%)')
    print(f'  P={P_W}W, Q={Q_SLM}slm, constant mode')
    print(f'{"="*110}')

    hdr = f'  {"V_eff":>8} {"gap":>6} |'
    for T in T_LIST:
        hdr += f'  {T-273:>4}°C'
    hdr += f'  | {"RMSE":>6} {"MAE":>6} {"MaxΔ":>6}'
    print(hdr)
    print(f'  {"(cm³)":>8} {"(mm)":>6} |' + ' ' * (7 * len(T_LIST)) + f'  |')
    print(f'  {"-" * (len(hdr) + 5)}')

    # Experimental row
    line = f'  {"EXP":>8} {"":>6} |'
    for T in T_LIST:
        line += f'  {EXP_DATA[T]:5.2f}'
    line += f'  |  {"--":>5} {"--":>5} {"--":>5}'
    print(line)
    print(f'  {"-" * (len(hdr) + 5)}')

    best_rmse = 1e10
    best_v = None

    for v in V_EFF_LIST:
        gap_mm = v / 4.9
        line = f'  {v:>8.2f} {gap_mm:>6.2f} |'
        errs = []
        for T in T_LIST:
            c = results[v][T][0]
            line += f'  {c:5.2f}'
            errs.append(c - EXP_DATA[T])

        ea = np.array(errs)
        rmse = np.sqrt(np.mean(ea**2))
        mae = np.mean(np.abs(ea))
        maxd = np.max(np.abs(ea))
        line += f'  | {rmse:6.2f} {mae:6.2f} {maxd:6.2f}'

        if rmse < best_rmse:
            best_rmse = rmse
            best_v = v

        if v == best_v:
            line += '  *'
        print(line)

    print(f'  {"-" * (len(hdr) + 5)}')
    print(f'  Best: V_eff={best_v:.1f}cm³ (gap={best_v/4.9:.2f}mm), RMSE={best_rmse:.3f}%')

    # =========================================================================
    # PLASMA PARAMETERS
    # =========================================================================
    print(f'\n{"="*110}')
    print(f'  PLASMA PARAMETERS')
    print(f'{"="*110}')

    hdr2 = f'  {"V_eff":>8} |'
    for T in T_LIST:
        hdr2 += f'  Te/ne @ {T-273}°C      '
    print(hdr2)
    print(f'  {"-" * (len(hdr2) + 5)}')

    for v in V_EFF_LIST:
        line = f'  {v:>8.2f} |'
        for T in T_LIST:
            _, Te, ne, tau, _ = results[v][T]
            line += f'  {Te:.3f}eV {ne:.1e}'
        print(line)

    # =========================================================================
    # MONOTONICITY CHECK
    # =========================================================================
    print(f'\n{"="*110}')
    print(f'  MONOTONICITY CHECK (conversion vs V_eff)')
    print(f'{"="*110}')

    for T in T_LIST:
        convs = [results[v][T][0] for v in V_EFF_LIST]
        diffs = [convs[i+1] - convs[i] for i in range(len(convs)-1)]
        mono = 'MONOTONE increasing' if all(d > 0 for d in diffs) else \
               'MONOTONE decreasing' if all(d < 0 for d in diffs) else 'NON-MONOTONE'
        print(f'  T={T}K ({T-273}°C): {mono}')
        print(f'    convs = {["%.2f" % c for c in convs]}')
        print(f'    diffs = {["%.3f" % d for d in diffs]}')
