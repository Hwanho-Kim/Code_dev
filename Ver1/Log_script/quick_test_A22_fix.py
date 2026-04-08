"""Test A22 fix: P=5W, Q=0.4slm, correct composition N2:O2:CO2:CH4=70:15:14:1, 4 temperatures."""
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
P_W = 5.0
Q_SLM = 0.4
V_EFF_CM3 = 1.6
V_REACTOR_CM3 = 250.0

# Actual experimental composition: N2:O2:CO2:CH4 = 70:15:14:1
INLET = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

print(f'A22 Fix Test: P={P_W}W, Q={Q_SLM}slm, V_eff={V_EFF_CM3}cm³, V_reactor={V_REACTOR_CM3}cm³')
print(f'Inlet: {INLET}')
print(f'{"T(°C)":>6} {"CH4_sim":>8} {"CH4_exp":>8} {"Δ(%p)":>8} {"Te(eV)":>8} {"n_e":>12} {"tau(s)":>8}')
print('-' * 75)

for T_K in EXP_DATA:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))
    cfg['V_eff'] = V_EFF_CM3 * 1e-6
    cfg['reactor']['volume'] = V_REACTOR_CM3 * 1e-6
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_SLM
    cfg['T_wall'] = T_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_K
    cfg['inlet_composition'] = INLET

    Q_actual = Q_SLM * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau = V_REACTOR_CM3 * 1e-6 / Q_actual
    t_end = max(10.0, 5.0 * tau)

    cfg['solver'] = {
        't_end': t_end, 'n_points': 150, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
    }

    with contextlib.redirect_stdout(io.StringIO()):
        solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)
        scfg = cfg_out['solver']
        result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                              rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    sm = solver.sm
    ch4_idx = sm.index('CH4')
    c0 = result.concentrations[ch4_idx, 0]
    cf = result.concentrations[ch4_idx, -1]
    conv = (c0 - cf) / c0 * 100 if c0 > 0 else 0

    n_sp = sm.n_species
    y = result.y[:, -1]
    n_e = max(y[0], 1e-30) * NA
    ne_eps = y[sm.idx_energy]
    T_gas = y[sm.idx_Tgas]
    eps_th = 1.5 * KB * max(T_gas, 200) / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean

    delta = conv - EXP_DATA[T_K]
    print(f'{T_K-273:6d} {conv:8.2f} {EXP_DATA[T_K]:8.2f} {delta:+8.2f} {Te_eV:8.3f} {n_e:12.2e} {tau:8.2f}')

# Also print comparison with old values
print(f'\nComparison:')
print(f'  Old (no A22, CH4=5%):  17.67%, 20.49%, 25.62%, 29.91%')
print(f'  A22 fix (CH4=5%):       2.17%,  2.76%,  3.83%,  4.82%')
print(f'  Exp (CH4=1%):           5.26%,  8.05%, 14.36%, 20.02%')
