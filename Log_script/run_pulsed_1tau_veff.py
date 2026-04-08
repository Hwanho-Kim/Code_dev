"""Pulsed trapezoidal 1τ with V_eff=4.9cm³.

V_eff 모델로 τ가 0.66s(303K)로 짧아져서 ~880 pulses만 돌리면 됨.
이전 V_reactor=250cm³에서는 τ=33.8s → 45,000 pulses(불가능)였음.

전략: CVODE ON/OFF operator splitting
  - ON phase: full RHS (Numba fast path), CVODE with constraints
  - OFF phase: rhs_off (frozen electrons), CVODE without constraints
  - OFF phase에서 trace species 음수 발생 → 다음 ON 전 clamp
"""
import sys, os, time, io, contextlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import T_STP, P_STP, NA, QE, KB
from plasma0d_v2.cvode_wrapper import CVODESolver
from plasma0d_v2.numba_core import pulsed_power_numba

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plasma0d_v2')
cfg = load_config(os.path.join(base_dir, 'config.yaml'))

# --- Pulsed trapezoidal ---
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {
    'PRF_Hz': 1333.0, 'duty_cycle': 0.20, 'P_peak_W': 8.1,
    'rise_time_s': 1.0e-7, 'waveform': 'trapezoidal',
}

# --- V_eff model ---
V_EFF_CM3 = 4.9
cfg['V_eff'] = V_EFF_CM3 * 1e-6
cfg['reactor']['volume'] = V_EFF_CM3 * 1e-6
cfg['reactor']['pressure'] = 101325.0

# --- Flow ---
Q_SLM = 0.4
cfg['flow'] = {'Q_slm': Q_SLM, 'model': 'PFR'}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

# --- Temperature ---
T_K = 303.0
cfg['initial'] = {'T_gas': T_K, 'ne': 1e8, 'Te_eV': 0.026}
cfg['T_wall'] = T_K
cfg['wall_loss_freq'] = 10000.0

# --- Timing ---
Q_actual = Q_SLM * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
tau = V_EFF_CM3 * 1e-6 / Q_actual
PRF = 1333.0
period = 1.0 / PRF
n_pulses_tau = int(tau * PRF)

# Minimal solver config for setup
cfg['solver'] = {
    't_end': tau, 'n_points': 1000, 'method': 'BDF',
    'rtol': 1e-5, 'atol': 1e-10, 'max_step': 1e-6, 'constrained': False,
}

P_avg = 8.1 * 0.20
print(f"Pulsed trapezoidal 1τ with V_eff={V_EFF_CM3}cm³")
print(f"  PRF={PRF:.0f}Hz, dc=20%, P_peak=8.1W, P_avg={P_avg:.2f}W")
print(f"  V_eff={V_EFF_CM3}cm³, Q={Q_SLM}slm, T={T_K:.0f}K")
print(f"  τ = {tau:.3f}s = {tau*1e3:.1f}ms")
print(f"  n_pulses = {n_pulses_tau}")
print(f"  P_dep = {P_avg/(V_EFF_CM3*1e-6):.3e} W/m³")
print()

with contextlib.redirect_stdout(io.StringIO()):
    solver, y0, t_span, co = setup_simulation(cfg, base_dir)

sm = solver.sm
n_sp = sm.n_species
n_state = y0.shape[0]
power = solver.power

# Build fast RHS for ON phase
_nb_fn = solver._rhs_numba
_nb_args = solver._nb_args
_period = float(power._pulse_period)
_t_on = float(power._pulse_t_on)
_rise = float(power._pulse_rise_time)
_P_on = float(power._pulse_P_on_Wm3)

def fast_rhs_on(t, y):
    Pdep = pulsed_power_numba(t, _period, _t_on, _rise, _P_on)
    return _nb_fn(t, y, *_nb_args, Pdep)

rhs_off = solver.rhs_off

# Constraints for ON phase
constraints_on = np.ones(n_state, dtype=np.float64)
constraints_on[sm.idx_Tgas] = 0.0

# --- Solve with CVODE ON/OFF splitting ---
print(f"Solving with CVODE ON/OFF operator splitting...")
print(f"  ON: full RHS (Numba), CVODE constrained")
print(f"  OFF: rhs_off (frozen electrons), CVODE unconstrained")
print()

t0_wall = time.time()
y = y0.copy()
total_rhs_on = 0
total_rhs_off = 0
n_fail_on = 0
n_fail_off = 0

# Snapshots for per-pulse analysis
snap_set = set(list(range(min(5, n_pulses_tau))) +
               list(range(max(5, n_pulses_tau-5), n_pulses_tau)))
snapshots = {}

for ip in range(n_pulses_tau):
    t_start = ip * period
    t_on_end = t_start + _t_on
    t_end_p = (ip + 1) * period

    # Clamp before ON phase
    y[:n_sp] = np.maximum(y[:n_sp], 0.0)

    # ON phase: CVODE with constraints
    cvode_on = CVODESolver(n_state, fast_rhs_on)
    cvode_on.setup(y, t_start, rtol=1e-5, atol=1e-10,
                   max_step=0.0, init_step=1e-12,
                   max_num_steps=100000, constraints=constraints_on)
    t_r, y_on, ret = cvode_on.step_to(t_on_end)
    total_rhs_on += cvode_on._get_stats()['n_rhs_evals']
    cvode_on.free()
    if ret < 0: n_fail_on += 1
    y = y_on.copy()
    ne_peak = y[0] * NA

    # Clamp between ON/OFF
    y[:n_sp] = np.maximum(y[:n_sp], 0.0)

    # OFF phase: CVODE without constraints
    cvode_off = CVODESolver(n_state, rhs_off)
    cvode_off.setup(y, t_on_end, rtol=1e-4, atol=1e-8,
                    max_step=0.0, init_step=1e-9,
                    max_num_steps=50000, constraints='none')
    t_r, y_off, ret = cvode_off.step_to(t_end_p)
    total_rhs_off += cvode_off._get_stats()['n_rhs_evals']
    cvode_off.free()
    if ret < 0: n_fail_off += 1
    y = y_off.copy()
    ne_valley = y[0] * NA

    if ip in snap_set:
        snapshots[ip] = {'ne_peak': ne_peak, 'ne_valley': ne_valley}

    if (ip+1) % 100 == 0 or ip == n_pulses_tau - 1:
        elapsed = time.time() - t0_wall
        pct = (ip+1) / n_pulses_tau * 100
        eta = elapsed / (ip+1) * (n_pulses_tau - ip - 1)
        ch4_now = (y0[sm.index('CH4')] - y[sm.index('CH4')]) / y0[sm.index('CH4')] * 100
        print(f"  [{pct:5.1f}%] Pulse {ip+1}/{n_pulses_tau}: ne={ne_valley:.3e}, "
              f"CH4={ch4_now:.3f}%, fail={n_fail_on}+{n_fail_off}, "
              f"{elapsed:.0f}s, ETA={eta:.0f}s")

wall = time.time() - t0_wall

# --- Results ---
ch4_idx = sm.index('CH4')
conv = (y0[ch4_idx] - y[ch4_idx]) / y0[ch4_idx] * 100

ne_final = y[0] * NA
T_gas_final = y[sm.idx_Tgas]
ne_eps_final = y[sm.idx_energy]
eps_th = 1.5 * KB * T_gas_final / QE
eps_mean = np.clip(ne_eps_final / ne_final, eps_th, 100.0) if ne_final > 1 else eps_th
Te_final = (2.0/3.0) * eps_mean

print(f"\n{'='*60}")
print(f"  RESULTS: Pulsed 1τ ({tau*1e3:.1f}ms, {n_pulses_tau} pulses)")
print(f"{'='*60}")
print(f"  CH4 conversion : {conv:.4f}%")
print(f"  ne_final       : {ne_final:.3e} m⁻³")
print(f"  Te_final       : {Te_final:.4f} eV")
print(f"  T_gas_final    : {T_gas_final:.1f} K")
print(f"  Wall time      : {wall:.1f}s ({wall/60:.1f}min)")
print(f"  Speed          : {n_pulses_tau/wall:.2f} pulses/s")
print(f"  RHS            : ON={total_rhs_on}, OFF={total_rhs_off}")
print(f"  Fails          : ON={n_fail_on}, OFF={n_fail_off}")
print(f"{'='*60}")

print(f"\nPer-pulse peaks:")
print(f"  {'Pulse':>5} {'ne_peak':>12} {'ne_valley':>12}")
for ip in sorted(snapshots.keys()):
    s = snapshots[ip]
    print(f"  {ip:5d} {s['ne_peak']:12.3e} {s['ne_valley']:12.3e}")

# Negative species check
neg = [sm.names[i] for i in range(n_sp) if y[i] < 0]
if neg:
    print(f"\n  WARNING: negative species at end: {neg}")
else:
    print(f"\n  All species non-negative!")

print(f"\n  Comparison:")
print(f"    Continuous PFR(1τ, V_eff=4.9): 4.35%")
print(f"    Pulsed PFR(1τ, V_eff=4.9):    {conv:.2f}%")
print(f"    P_avg(pulsed) = {P_avg:.2f}W vs P(cont) = 5.0W")
print(f"    P-scaled: {conv:.2f}% × (5.0/{P_avg:.2f}) = {conv*5.0/P_avg:.2f}%")
