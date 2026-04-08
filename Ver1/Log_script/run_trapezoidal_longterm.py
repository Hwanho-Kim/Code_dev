"""Trapezoidal pulsed mode long-term test.
PRF=1333Hz, dc=20%, P_peak=8.1W (P_avg=1.62W), rise=100ns.
T=303K, PFR, Q=0.4slm.

Usage: cd /home/hawn/work && python Log_script/run_trapezoidal_longterm.py
"""
import sys, os, time
# Add parent dir (/home/hawn/work) so plasma0d_v2 package is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import T_STP, P_STP, NA, QE, KB

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plasma0d_v2')
cfg = load_config(os.path.join(base_dir, 'config.yaml'))

# --- Trapezoidal pulsed ---
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {
    'PRF_Hz': 1333.0,
    'duty_cycle': 0.20,
    'P_peak_W': 8.1,
    'rise_time_s': 1.0e-7,
    'waveform': 'trapezoidal',
}

# --- Reactor / flow ---
cfg['reactor'] = {'volume': 250e-6, 'pressure': 101325.0}
cfg['V_eff'] = 1.6e-6
cfg['flow'] = {'Q_slm': 0.4, 'model': 'PFR'}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

# --- Initial / thermal ---
T_K = 303.0
cfg['initial'] = {'T_gas': T_K, 'ne': 1e8, 'Te_eV': 0.026}
cfg['T_wall'] = T_K
cfg['wall_loss_freq'] = 10000.0

# --- Solver ---
# tau_flow ~ 30s, but start with 50ms (67 pulses) to gauge speed
Q_actual = 0.4 * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
tau = 250e-6 / Q_actual
t_end = 0.050  # 50 ms = 67 pulses

cfg['solver'] = {
    't_end': t_end,
    'n_points': 2000,
    'method': 'BDF',
    'rtol': 1e-5,
    'atol': 1e-10,
    'max_step': 1e-6,
    'constrained': False,
}

PRF = 1333.0
period = 1.0 / PRF
n_pulses = int(t_end / period)
print(f"Trapezoidal pulsed long-term test")
print(f"  PRF={PRF:.0f}Hz, dc=20%, P_peak=8.1W, rise=100ns")
print(f"  t_end={t_end*1e3:.1f}ms ({n_pulses} pulses)")
print(f"  tau_flow={tau:.1f}s ({t_end/tau*100:.2f}% of tau)")
print(f"  T={T_K:.0f}K, PFR, Q=0.4slm")
print()

solver, y0, t_span, co = setup_simulation(cfg, base_dir)
scfg = co['solver']

t0_wall = time.time()
result = solver.solve(
    y0, t_span,
    n_points=scfg['n_points'],
    method=scfg['method'],
    rtol=scfg['rtol'],
    atol=scfg['atol'],
    max_step=scfg['max_step'],
    constrained=scfg.get('constrained', False),
)
wall = time.time() - t0_wall

# --- Results ---
sm = solver.sm
ch4_i = sm.index('CH4')
c0 = result.concentrations[ch4_i, 0]
cf = result.concentrations[ch4_i, -1]
conv = (c0 - cf) / c0 * 100 if c0 > 0 else 0

y_f = result.y[:, -1]
n_e = max(y_f[0], 1e-30) * NA
ne_eps = y_f[sm.idx_energy]
T_gas = y_f[sm.idx_Tgas]
eps_th = 1.5 * KB * max(T_gas, 200) / QE
eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
Te_eV = (2.0 / 3.0) * eps_mean

print(f"\n{'='*60}")
print(f"  RESULTS ({t_end*1e3:.1f}ms, {n_pulses} pulses)")
print(f"{'='*60}")
print(f"  CH4 conversion : {conv:.4f}%")
print(f"  Final n_e      : {n_e:.3e} m-3")
print(f"  Final Te       : {Te_eV:.4f} eV")
print(f"  Final T_gas    : {T_gas:.1f} K")
print(f"  Wall time      : {wall:.1f}s ({wall/60:.1f}min)")
print(f"  RHS evals      : {result.n_rhs_evals}")
print(f"  Speed          : {n_pulses/wall:.2f} pulses/s")
print(f"  Extrapolation  : tau_flow({tau:.1f}s) would take ~{tau*wall/t_end/3600:.0f}h")

# Save
out_path = os.path.join(os.path.dirname(__file__), 'trapezoidal_longterm_result.npz')
np.savez(out_path,
         t=result.t, y=result.y,
         species_names=result.species_names,
         wall_time=wall, n_pulses=n_pulses, t_end=t_end)
print(f"\n  Saved: {out_path}")
