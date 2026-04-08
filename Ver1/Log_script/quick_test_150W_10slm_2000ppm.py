"""Quick test: P=150W, Q=10slm, CH4=2000ppm, 4 temperatures."""
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
P_W = 150.0
Q_SLM = 10.0
V_EFF_CM3 = 1.6
V_REACTOR_CM3 = 250.0

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

    # CH4 2000 ppm, CO2 5%, rest adjusted
    cfg['inlet_composition'] = {
        'CH4': 0.002,    # 2000 ppm
        'CO2': 0.05,
        'N2': 0.836,     # balance
        'O2': 0.112,
    }

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

    SEI = P_W / (Q_SLM / 60.0)
    print(f'T={T_K-273:3d}°C  CH4={conv:5.2f}%  Te={Te_eV:.3f}eV  n_e={n_e:.2e}  '
          f'tau={tau:.2f}s  SEI={SEI:.0f}J/L  CH4_in=2000ppm  CH4_out={cf/c0*2000:.0f}ppm')
