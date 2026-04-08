"""Trapezoidal pulsed mode with SUNDIALS CVODE.

Strategy: Smooth floor + max_step limit.
- Smooth sigmoid suppression of negative dydt near ne_floor
- max_step=1e-7 so BDF polynomial captures the afterglow transition
- No CVODE constraints needed

Usage: cd /home/hawn/work && LD_LIBRARY_PATH=~/.local/sundials/lib:$LD_LIBRARY_PATH \
       python Log_script/run_trapezoidal_cvode.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.cvode_wrapper import CVODESolver, CVODEResult
from plasma0d_v2.constants import T_STP, P_STP, NA, QE, KB

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plasma0d_v2')
cfg = load_config(os.path.join(base_dir, 'config.yaml'))

# --- Trapezoidal pulsed ---
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {
    'PRF_Hz': 1333.0, 'duty_cycle': 0.20, 'P_peak_W': 8.1,
    'rise_time_s': 1.0e-7, 'waveform': 'trapezoidal',
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
Q_actual = 0.4 * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
tau = 250e-6 / Q_actual
t_end = 0.050  # 50 ms

cfg['solver'] = {
    't_end': t_end, 'n_points': 2000, 'method': 'BDF',
    'rtol': 1e-5, 'atol': 1e-10, 'max_step': 1e-6, 'constrained': False,
}

PRF = 1333.0
period = 1.0 / PRF
n_pulses = int(t_end / period)
print(f"CVODE + smooth floor + max_step limit")
print(f"  PRF={PRF:.0f}Hz, dc=20%, P_peak=8.1W, rise=100ns")
print(f"  t_end={t_end*1e3:.1f}ms ({n_pulses} pulses)")
print(f"  tau_flow={tau:.1f}s ({t_end/tau*100:.2f}% of tau)")
print()

solver, y0, t_span, co = setup_simulation(cfg, base_dir)
sm = solver.sm
n_sp = sm.n_species

# ============================================================
#  Smooth floor configuration
# ============================================================

NE_FLOOR = 1e8       # m⁻³ (same as original ne_seed)
CE_FLOOR = NE_FLOOR / NA  # mol/m³
MAX_STEP = 1e-7      # 100 ns — captures the attachment decay (τ~40ns)

# Override solver internal floors to match
solver._ce_floor = CE_FLOOR
solver._ne_eps_floor = NE_FLOOR * 0.039

# The solver.rhs already does:
#   c = np.maximum(c, concentration_floor)   [1e-30 for all species]
#   if y_sp[0] < ce_floor: dydt[0] = max(dydt[0], 0.0)
# With ce_floor = NE_FLOOR/NA, this acts as a hard floor guard.
# CVODE with max_step=1e-7 ensures steps are small enough
# to not jump over the floor region.

# --- CVODE solve ---
print(f"\n  CVODE + smooth floor + max_step={MAX_STEP:.0e}")
print(f"    n_eq = {len(y0)}")
print(f"    ne_floor = {NE_FLOOR:.0e} m⁻³  (ce_floor = {CE_FLOOR:.2e} mol/m³)")
print(f"    rtol = 1e-5, atol = 1e-10, max_step = {MAX_STEP:.0e}")

t_eval = np.linspace(0, t_end, 2000)
t0_wall = time.time()

cvode = CVODESolver(len(y0), solver.rhs)
cvode.setup(y0, t0=0.0, rtol=1e-5, atol=1e-10,
            max_step=MAX_STEP,
            init_step=1e-12,
            max_num_steps=10000000,
            constraints='none')  # no constraints — floor guard in RHS handles it

result = cvode.solve(t_eval)
wall = time.time() - t0_wall

print(f"\n  CVODE stats:")
print(f"    steps={result.n_steps}, rhs_evals={result.n_rhs_evals}")
print(f"    err_fails={result.n_err_fails}")
print(f"    success={result.success}, msg={result.message}")

cvode.free()

# --- Results ---
ch4_i = sm.index('CH4')
y = result.y

c0_ch4 = y[ch4_i, 0]
cf_ch4 = y[ch4_i, -1]
conv = (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0

ne = y[0, :] * NA
ne_eps = y[sm.idx_energy, :]
T_gas = y[sm.idx_Tgas, :]
eps_th = 1.5 * KB * np.maximum(T_gas, 200) / QE
eps_mean = np.where(ne > 1, np.clip(ne_eps / ne, eps_th, 100.0), eps_th)
Te_eV = (2.0 / 3.0) * eps_mean

print(f"\n{'='*60}")
print(f"  RESULTS ({t_end*1e3:.1f}ms, {n_pulses} pulses)")
print(f"{'='*60}")
print(f"  CH4 conversion : {conv:.4f}%")
print(f"  ne: min={ne.min():.3e}, max={ne.max():.3e}, final={ne[-1]:.3e}")
print(f"  Te: min={Te_eV.min():.4f}, max={Te_eV.max():.4f}, final={Te_eV[-1]:.4f} eV")
print(f"  T_gas final    : {T_gas[-1]:.1f} K")
print(f"  Wall time      : {wall:.1f}s ({wall/60:.1f}min)")
print(f"  Speed          : {n_pulses/wall:.2f} pulses/s")
print(f"  Extrapolation  : tau_flow({tau:.1f}s) would take ~{tau*wall/t_end/3600:.0f}h")

# Check negativity
neg_mask = np.any(y[:n_sp, :] < 0, axis=1)
neg_names = [sm.names[i] for i in np.where(neg_mask)[0]]
if neg_names:
    print(f"\n  WARNING: negative species: {neg_names}")
    # Show worst offenders
    for name in neg_names[:5]:
        idx = sm.index(name)
        vals = y[idx, :]
        print(f"    {name}: min={vals.min():.3e}, max={vals.max():.3e}")
else:
    print(f"\n  All species non-negative!")

# Check ne_eps
if np.any(ne_eps < 0):
    print(f"  ne_eps: min={ne_eps.min():.3e}, max={ne_eps.max():.3e}")

# Per-pulse summary
print(f"\n  Per-pulse peaks (first 5 + last 5):")
print(f"  {'Pulse':>5} {'ne_peak':>12} {'Te_peak':>10} {'ne_valley':>12}")
for ip in list(range(min(5, n_pulses))) + list(range(max(5, n_pulses-5), n_pulses)):
    t0p = ip * period
    t1p = (ip + 1) * period
    mask = (t_eval >= t0p) & (t_eval < t1p)
    if mask.sum() == 0:
        continue
    ne_p = ne[mask]
    Te_p = Te_eV[mask]
    print(f"  {ip:5d} {ne_p.max():12.3e} {Te_p.max():10.4f} {ne_p.min():12.3e}")

# Save
out_path = os.path.join(os.path.dirname(__file__), 'trapezoidal_cvode_result.npz')
np.savez(out_path, t=t_eval, y=result.y, species_names=sm.names,
         wall_time=wall, n_pulses=n_pulses, t_end=t_end)
print(f"\n  Saved: {out_path}")
