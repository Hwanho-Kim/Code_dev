"""Run pulsed CSTR to steady state (5τ ≈ 170s, ~1.7M pulses).

Usage:
    nohup python run_pulsed_steady_state.py > pulsed_run.log 2>&1 &

Expected runtime: ~24-44 hours (depends on CPU).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import T_STP, P_STP

# Configuration
cfg = load_config(os.path.join(os.path.dirname(__file__), 'plasma0d_v2', 'config.yaml'))
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {
    'PRF_Hz': 10000.0,
    'duty_cycle': 0.20,
    'P_peak_W': 8.1,
    'rise_time_s': 2e-6,
    'waveform': 'trapezoidal',
}
cfg['reactor'] = {'volume': 250e-6, 'pressure': 101325.0}
cfg['flow'] = {'Q_slm': 0.4, 'model': 'CSTR'}
cfg['T_wall'] = 303.0
cfg['wall_loss_freq'] = 10000.0
cfg['initial'] = {'T_gas': 303.0, 'ne': 1e12, 'Te_eV': 2.0}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

solver, y0, _, _ = setup_simulation(cfg, os.path.join(os.path.dirname(__file__), 'plasma0d_v2'))

# Residence time
Q = 0.4 * (303 / T_STP) * (P_STP / 101325.0) / 60000.0
tau = 250e-6 / Q
t_end = 5.0 * tau  # ~170s

print(f"\nτ = {tau:.1f}s, t_end = {t_end:.1f}s")
print(f"Estimated pulses: {int(t_end * 10000)}")
print(f"Starting pulsed steady-state run...\n")

result = solver.solve_pulsed(
    y0, t_end=t_end,
    n_output=2000,
    progress_every=10000,
)

# Save results
idx = result.species_names.index('CH4')
c0, cf = result.concentrations[idx, 0], result.concentrations[idx, -1]
conv = (c0 - cf) / c0 * 100

print(f"\n=== FINAL RESULTS ===")
print(f"CH4 conversion: {conv:.2f}%")
print(f"ne_final: {result.ne_m3[-1]:.2e} m⁻³")
print(f"T_gas_final: {result.T_gas[-1]:.2f} K")
print(f"Wall time: {result.wall_time/3600:.1f} hours")

# Save numpy arrays
np.savez(os.path.join(os.path.dirname(__file__), 'pulsed_steady_state_result.npz'),
         t=result.t, y=result.y,
         species_names=result.species_names,
         ne_m3=result.ne_m3, Te_eV=result.Te_eV,
         T_gas=result.T_gas, concentrations=result.concentrations,
         wall_time=result.wall_time)
print("Results saved to pulsed_steady_state_result.npz")
