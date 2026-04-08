"""CVODE vs BDF pulsed trapezoidal test — 4 periods (400µs).

Compare CVODE ctypes wrapper against scipy BDF constrained baseline.
Same settings as pulsed_baseline.txt.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import time

from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import NA

_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
cfg = load_config(os.path.join(_base, 'plasma0d_v2', 'config.yaml'))

# --- Pulsed trapezoidal config (same as pulsed_baseline) ---
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {
    'PRF_Hz': 10000.0,
    'duty_cycle': 0.20,
    'P_peak_W': 8.1,
    'rise_time_s': 2e-6,
    'waveform': 'trapezoidal',
}
cfg['reactor'] = {'volume': 250e-6, 'pressure': 101325.0}
cfg['V_eff'] = 4.0e-7
cfg['flow'] = {'Q_slm': 0.4, 'model': 'PFR'}
cfg['T_wall'] = 303.0
cfg['wall_loss_freq'] = 10000.0
cfg['initial'] = {'T_gas': 303.0, 'ne': 1e8, 'Te_eV': 0.026}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}
cfg['Lambda'] = 1.0e-3

n_periods = 4
T_period = 1.0 / 10000.0  # 100µs
t_end = n_periods * T_period  # 400µs
n_points = n_periods * 500  # 500 points per period

cfg['solver'] = {
    't_end': t_end,
    'n_points': n_points,
    'method': 'BDF',
    'rtol': 1e-5,
    'atol': 1e-10,
    'max_step': 0.5e-6,
    'constrained': True,
}

print(f"="*60)
print(f"  CVODE vs BDF Pulsed Trapezoidal Test")
print(f"  {n_periods} periods = {t_end*1e6:.0f} µs")
print(f"  PRF=10kHz, dc=20%, P_peak=8.1W, rise=2µs")
print(f"="*60)

solver, y0, t_span, co = setup_simulation(
    cfg, os.path.join(_base, 'plasma0d_v2'))
scfg = co['solver']

# ===== Run 1: scipy BDF constrained (reference) =====
print(f"\n{'='*60}")
print(f"  [1] scipy BDF constrained (reference)")
print(f"{'='*60}")

result_bdf = solver.solve(
    y0, t_span, n_points=scfg['n_points'], method='BDF',
    rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'],
    constrained=True)

# ===== Run 2: CVODE ctypes =====
print(f"\n{'='*60}")
print(f"  [2] CVODE ctypes + constraints")
print(f"{'='*60}")

result_cvode = solver.solve(
    y0, t_span, n_points=scfg['n_points'], method='CVODE',
    rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

# ===== Compare =====
print(f"\n{'='*60}")
print(f"  Comparison: {n_periods} periods ({t_end*1e6:.0f} µs)")
print(f"{'='*60}")

# Extract key diagnostics at specific times
diag_times_us = [5, 10, 18, 20, 21, 30, 50, 99,
                 105, 110, 199, 205, 210, 299, 305, 310, 399]
diag_times_us = [t for t in diag_times_us if t <= t_end * 1e6]

print(f"\n  {'t(µs)':>7s}  {'BDF_ne':>10s}  {'CVODE_ne':>10s}  {'ratio':>7s}  "
      f"{'BDF_Te':>7s}  {'CVODE_Te':>7s}  {'ΔTe':>6s}")
print(f"  {'-'*70}")

for t_us in diag_times_us:
    t_s = t_us * 1e-6

    # Find closest index in each result
    i_bdf = np.argmin(np.abs(result_bdf.t - t_s))
    i_cvode = np.argmin(np.abs(result_cvode.t - t_s))

    ne_bdf = result_bdf.ne_m3[i_bdf] if result_bdf.ne_m3 is not None else 0
    ne_cvode = result_cvode.ne_m3[i_cvode] if result_cvode.ne_m3 is not None else 0

    Te_bdf = result_bdf.Te_eV[i_bdf] if result_bdf.Te_eV is not None else 0
    Te_cvode = result_cvode.Te_eV[i_cvode] if result_cvode.Te_eV is not None else 0

    ratio = ne_cvode / ne_bdf if ne_bdf > 0 else float('inf')
    dTe = Te_cvode - Te_bdf

    print(f"  {t_us:7.1f}  {ne_bdf:10.2e}  {ne_cvode:10.2e}  {ratio:7.3f}  "
          f"{Te_bdf:7.3f}  {Te_cvode:7.3f}  {dTe:+6.3f}")

# Performance summary
print(f"\n  Performance:")
print(f"  {'':>15s}  {'BDF':>12s}  {'CVODE':>12s}  {'speedup':>8s}")
print(f"  {'wall time':>15s}  {result_bdf.wall_time:10.2f}s  "
      f"{result_cvode.wall_time:10.2f}s  "
      f"{result_bdf.wall_time/result_cvode.wall_time:7.1f}x")
print(f"  {'RHS evals':>15s}  {result_bdf.n_rhs_evals:>12d}  "
      f"{result_cvode.n_rhs_evals:>12d}  "
      f"{result_bdf.n_rhs_evals/max(1,result_cvode.n_rhs_evals):7.1f}x")
print(f"  {'message':>15s}  {result_bdf.solver_message[:40]}")
print(f"  {'':>15s}  {result_cvode.solver_message[:40]}")

# CH4 at end
ch4_i = result_bdf.species_names.index('CH4')
ch4_bdf = result_bdf.concentrations[ch4_i, -1]
ch4_cvode = result_cvode.concentrations[ch4_i, -1]
ch4_init = result_bdf.concentrations[ch4_i, 0]
conv_bdf = (ch4_init - ch4_bdf) / ch4_init * 100
conv_cvode = (ch4_init - ch4_cvode) / ch4_init * 100
print(f"\n  CH4 conversion @ {t_end*1e6:.0f}µs: BDF={conv_bdf:.4f}%, CVODE={conv_cvode:.4f}%")
