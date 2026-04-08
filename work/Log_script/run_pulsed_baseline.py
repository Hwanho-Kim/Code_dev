"""Run pulsed PFR to steady state using validated baseline solver.
scipy constrained BDF + scaled FD Jacobian + Numba RHS + clamping.

Usage: nohup python run_pulsed_baseline.py > baseline_run.log 2>&1 &
Expected: ~165 hours (7 days)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import T_STP, P_STP

cfg = load_config(os.path.join(os.path.dirname(__file__), 'plasma0d_v2', 'config.yaml'))
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {'PRF_Hz': 10000.0, 'duty_cycle': 0.20, 'P_peak_W': 8.1,
                'rise_time_s': 2e-6, 'waveform': 'trapezoidal'}
cfg['reactor'] = {'volume': 250e-6, 'pressure': 101325.0}
cfg['flow'] = {'Q_slm': 0.4, 'model': 'PFR'}
cfg['T_wall'] = 303.0; cfg['wall_loss_freq'] = 10000.0
cfg['initial'] = {'T_gas': 303.0, 'ne': 1e8, 'Te_eV': 0.026}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

Q = 0.4 * (303/T_STP) * (P_STP/101325.0) / 60000.0
tau = 250e-6 / Q
n_pulses = int(tau * 10000)

cfg['solver'] = {'t_end': tau, 'n_points': 2000, 'method': 'BDF',
                 'rtol': 1e-5, 'atol': 1e-10, 'max_step': 0.5e-6, 'constrained': True}

print(f"Pulsed PFR baseline: tau={tau:.1f}s, {n_pulses} pulses")
solver, y0, t_span, co = setup_simulation(cfg, os.path.join(os.path.dirname(__file__), 'plasma0d_v2'))
scfg = co['solver']
result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                      rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'],
                      constrained=True)

ch4_i = result.species_names.index('CH4')
conv = (result.concentrations[ch4_i,0]-result.concentrations[ch4_i,-1])/result.concentrations[ch4_i,0]*100
print(f"\nCH4 conv: {conv:.2f}%")
print(f"Wall time: {result.wall_time:.0f}s ({result.wall_time/3600:.1f}h)")

np.savez(os.path.join(os.path.dirname(__file__), 'pulsed_baseline_result.npz'),
         t=result.t, y=result.y, species_names=result.species_names,
         wall_time=result.wall_time)
