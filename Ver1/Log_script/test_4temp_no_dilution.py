"""4-temperature reference test after V_eff/V_reactor dilution removal.
Conditions: P=5W, Q=0.4slm, N2:O2:CO2:CH4=70:15:14:1, PFR, V_reactor=250cm³.
Compare with pre-dilution-removal PFR results: 4.95/7.18/13.21/19.92%.
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
PRE_DILUTION = {303: 4.95, 373: 7.18, 453: 13.21, 523: 19.92}  # PFR results before dilution removal
P_W = 5.0
Q_SLM = 0.4
V_EFF_CM3 = 1.6
V_REACTOR_CM3 = 250.0
INLET = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

print('=' * 90)
print('  4-Temperature Reference Test (V_eff/V_reactor dilution REMOVED)')
print('=' * 90)
print(f'  P={P_W}W, Q={Q_SLM}slm, V_eff={V_EFF_CM3}cm³, V_reactor={V_REACTOR_CM3}cm³')
print(f'  Inlet: {INLET}')
print(f'  Flow: PFR, t_end=5×τ')
print()

results = []
t_total = time.time()

for T_K in EXP_DATA:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))
    cfg['V_eff'] = V_EFF_CM3 * 1e-6
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
    t_end = max(10.0, 5.0 * tau)

    cfg['solver'] = {
        't_end': t_end, 'n_points': 150, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
    }

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)
        scfg = cfg_out['solver']
        result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                              rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])
    wt = time.time() - t0

    sm = solver.sm
    ch4_idx = sm.index('CH4'); co2_idx = sm.index('CO2')
    c0_ch4 = result.concentrations[ch4_idx, 0]
    cf_ch4 = result.concentrations[ch4_idx, -1]
    c0_co2 = result.concentrations[co2_idx, 0]
    cf_co2 = result.concentrations[co2_idx, -1]
    conv_ch4 = (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0
    conv_co2 = (c0_co2 - cf_co2) / c0_co2 * 100 if c0_co2 > 0 else 0

    n_sp = sm.n_species
    y = result.y[:, -1]
    n_e = max(y[0], 1e-30) * NA
    ne_eps = y[sm.idx_energy]
    T_gas = y[sm.idx_Tgas]
    eps_th = 1.5 * KB * max(T_gas, 200) / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean

    r = {'T_K': T_K, 'ch4': conv_ch4, 'co2': conv_co2,
         'Te': Te_eV, 'n_e': n_e, 'Tg': T_gas, 'tau': tau, 'wt': wt}
    results.append(r)
    print(f'  T={T_K-273:3d}°C ({T_K}K): CH4={conv_ch4:6.2f}%  CO2={conv_co2:6.2f}%  '
          f'Te={Te_eV:.3f}eV  n_e={n_e:.2e}  ({wt:.0f}s)', flush=True)

total_time = time.time() - t_total
print(f'\n  Total: {total_time:.0f}s ({total_time/60:.1f} min)')

# ============================================================
# COMPARISON TABLE
# ============================================================
print(f'\n{"="*90}')
print(f'  CH4 CONVERSION: Experiment vs Pre-dilution vs Post-dilution')
print(f'{"="*90}')
print(f'  {"T(°C)":>6} {"T(K)":>6} {"Exp(%)":>8} {"Pre(%)":>8} {"Post(%)":>8} '
      f'{"Δexp(%p)":>9} {"Δpre(%p)":>9} {"Te(eV)":>8} {"n_e(m⁻³)":>12}')
print(f'  {"-"*85}')

errs_exp = []; errs_pre = []
for r in results:
    T_K = r['T_K']
    exp = EXP_DATA[T_K]; pre = PRE_DILUTION[T_K]; post = r['ch4']
    d_exp = post - exp; d_pre = post - pre
    errs_exp.append(d_exp); errs_pre.append(d_pre)
    print(f'  {T_K-273:6d} {T_K:6d} {exp:8.2f} {pre:8.2f} {post:8.2f} '
          f'{d_exp:+9.2f} {d_pre:+9.2f} {r["Te"]:8.3f} {r["n_e"]:12.2e}')

e = np.array(errs_exp); p = np.array(errs_pre)
print(f'  {"-"*85}')
print(f'  {"RMSE":>6} {"":>6} {"":>8} {"":>8} {"":>8} '
      f'{np.sqrt(np.mean(e**2)):9.2f} {np.sqrt(np.mean(p**2)):9.2f}')
print(f'  {"MAE":>6} {"":>6} {"":>8} {"":>8} {"":>8} '
      f'{np.mean(np.abs(e)):9.2f} {np.mean(np.abs(p)):9.2f}')

print(f'\n  CO2 CONVERSION:')
print(f'  {"T(°C)":>6} {"CO2(%)":>8}')
print(f'  {"-"*20}')
for r in results:
    print(f'  {r["T_K"]-273:6d} {r["co2"]:8.2f}')
