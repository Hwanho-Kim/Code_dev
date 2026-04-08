"""Test PFR parcel cycling vs PFR(1τ) baseline.

Compares:
  1. PFR(1τ): standard Lagrangian batch, fresh gas → integrate τ
  2. PFR cycling: feed gas reset each cycle, radicals/products carry over
"""
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
P_W = 5.0
Q_SLM = 0.4
V_REACTOR_CM3 = 250.0
INLET = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

print(f'PFR Cycling Test: P={P_W}W, Q={Q_SLM}slm, V_reactor={V_REACTOR_CM3}cm³')
print(f'Inlet: {INLET}')
print()

# ── Header ──
print(f'{"T(°C)":>6} {"PFR(1τ)":>9} {"Cycling":>9} {"Exp":>8} '
      f'{"Δ_PFR":>8} {"Δ_Cyc":>8} {"Cycles":>7}')
print('-' * 68)

for T_K in EXP_DATA:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plasma0d_v2')
    cfg_base = load_config(os.path.join(base_dir, 'config.yaml'))

    # Common settings
    for cfg in [cfg_base]:
        cfg['V_eff'] = V_REACTOR_CM3 * 1e-6  # V_eff = V_reactor (baseline)
        cfg['reactor']['volume'] = V_REACTOR_CM3 * 1e-6
        cfg['power_mode'] = 'constant'
        cfg['P_input_W'] = P_W
        cfg['flow']['Q_slm'] = Q_SLM
        cfg['flow']['model'] = 'PFR'
        cfg['T_wall'] = T_K
        cfg['wall_loss_freq'] = 10000.0
        cfg['initial']['T_gas'] = T_K
        cfg['inlet_composition'] = INLET

    Q_actual = Q_SLM * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau = V_REACTOR_CM3 * 1e-6 / Q_actual

    cfg_base['solver'] = {
        't_end': tau, 'n_points': 150, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
    }

    # ── 1) PFR(1τ) baseline ──
    with contextlib.redirect_stdout(io.StringIO()):
        solver, y0, t_span, cfg_out = setup_simulation(cfg_base, base_dir)
        scfg = cfg_out['solver']
        result_pfr = solver.solve(y0, t_span, n_points=scfg['n_points'],
                                   method=scfg['method'], rtol=scfg['rtol'],
                                   atol=scfg['atol'], max_step=scfg['max_step'])

    sm = solver.sm
    ch4_idx = sm.index('CH4')
    c0 = result_pfr.concentrations[ch4_idx, 0]
    cf_pfr = result_pfr.concentrations[ch4_idx, -1]
    conv_pfr = (c0 - cf_pfr) / c0 * 100

    # ── 2) PFR cycling ──
    # Re-setup (fresh solver state)
    cfg_cyc = load_config(os.path.join(base_dir, 'config.yaml'))
    cfg_cyc['V_eff'] = V_REACTOR_CM3 * 1e-6
    cfg_cyc['reactor']['volume'] = V_REACTOR_CM3 * 1e-6
    cfg_cyc['power_mode'] = 'constant'
    cfg_cyc['P_input_W'] = P_W
    cfg_cyc['flow']['Q_slm'] = Q_SLM
    cfg_cyc['flow']['model'] = 'PFR'
    cfg_cyc['T_wall'] = T_K
    cfg_cyc['wall_loss_freq'] = 10000.0
    cfg_cyc['initial']['T_gas'] = T_K
    cfg_cyc['inlet_composition'] = INLET
    cfg_cyc['solver'] = {
        't_end': tau, 'n_points': 150, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
    }

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver2, y0_2, _, cfg_out2 = setup_simulation(cfg_cyc, base_dir)
        scfg2 = cfg_out2['solver']
        result_cyc = solver2.solve_pfr_cycling(
            y0_2, inlet_composition=INLET,
            P_gas=101325.0, T_gas_init=T_K,
            max_cycles=50, conv_atol=0.01,
            method=scfg2['method'], rtol=scfg2['rtol'],
            atol=scfg2['atol'], max_step=scfg2['max_step'],
            n_points=scfg2['n_points'])

    # Extract cycling log for cycle count
    log = buf.getvalue()
    n_cycles = log.count('Cycle')

    cf_cyc = result_cyc.y[ch4_idx, -1]
    c0_cyc = INLET['CH4'] * (101325.0 / (R_GAS * T_K))
    conv_cyc = (c0_cyc - cf_cyc) / c0_cyc * 100

    d_pfr = conv_pfr - EXP_DATA[T_K]
    d_cyc = conv_cyc - EXP_DATA[T_K]

    print(f'{T_K-273:6d} {conv_pfr:9.2f} {conv_cyc:9.2f} {EXP_DATA[T_K]:8.2f} '
          f'{d_pfr:+8.2f} {d_cyc:+8.2f} {n_cycles:7d}')

print()
print('PFR(1τ) baseline: 4.95/7.18/13.21/19.94%')
print('Exp:              5.26/8.05/14.36/20.02%')
