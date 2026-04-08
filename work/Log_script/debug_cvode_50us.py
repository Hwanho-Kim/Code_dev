"""Debug CVODE pulsed output around t=50µs anomaly."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import NA

_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
cfg = load_config(os.path.join(_base, 'plasma0d_v2', 'config.yaml'))

cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {'PRF_Hz': 10000.0, 'duty_cycle': 0.20, 'P_peak_W': 8.1,
                'rise_time_s': 2e-6, 'waveform': 'trapezoidal'}
cfg['reactor'] = {'volume': 250e-6, 'pressure': 101325.0}
cfg['V_eff'] = 4.0e-7
cfg['flow'] = {'Q_slm': 0.4, 'model': 'PFR'}
cfg['T_wall'] = 303.0; cfg['wall_loss_freq'] = 10000.0
cfg['initial'] = {'T_gas': 303.0, 'ne': 1e8, 'Te_eV': 0.026}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}
cfg['Lambda'] = 1.0e-3

# Only 1 period, dense output in OFF phase
t_end = 100e-6  # 1 period
n_points = 5000  # very dense

cfg['solver'] = {'t_end': t_end, 'n_points': n_points, 'method': 'BDF',
                 'rtol': 1e-5, 'atol': 1e-10, 'max_step': 0.5e-6, 'constrained': True}

solver, y0, t_span, co = setup_simulation(cfg, os.path.join(_base, 'plasma0d_v2'))

# Run CVODE only
print(f"\n{'='*60}")
print(f"  CVODE debug: 1 period, {n_points} points")
print(f"{'='*60}")
result = solver.solve(y0, t_span, n_points=n_points, method='CVODE',
                      rtol=1e-5, atol=1e-10, max_step=0.5e-6)

# Dump ne around the OFF phase (20-100µs)
print(f"\n  Raw output around OFF phase:")
print(f"  {'t(µs)':>8s}  {'c_e(mol/m3)':>14s}  {'ne(m-3)':>12s}  {'ne_eps':>12s}  {'eps_mean':>10s}")

for i in range(len(result.t)):
    t_us = result.t[i] * 1e6
    if 19 < t_us < 101:
        if t_us < 22 or 29 < t_us < 31 or 39 < t_us < 41 or \
           44 < t_us < 56 or 59 < t_us < 61 or 79 < t_us < 81 or 98 < t_us:
            c_e = result.y[0, i]
            ne = c_e * NA
            ne_eps = result.y[result.n_species, i]  # idx_energy = n_species
            eps_mean = ne_eps / ne if ne > 1e-10 else 0
            print(f"  {t_us:8.3f}  {c_e:14.6e}  {ne:12.4e}  {ne_eps:12.4e}  {eps_mean:10.4f}")

# Check for anomalous points in OFF phase
print(f"\n  Anomalous points (ne < 1e4 m⁻³) in OFF phase:")
n_anomaly = 0
for i in range(len(result.t)):
    t_us = result.t[i] * 1e6
    if 20 < t_us < 100:
        ne = result.y[0, i] * NA
        if ne < 1e4:
            n_anomaly += 1
            if n_anomaly <= 20:
                print(f"  t={t_us:.3f} µs, ne={ne:.4e}, c_e={result.y[0,i]:.4e}")
print(f"  Total anomalous: {n_anomaly} / {sum(1 for t in result.t if 20e-6 < t < 100e-6)}")
