"""Pulsed 1τ + last 10 pulse continuous time-series plot.

Reference format: pulsed_last15_profiles.png
  - Panel 1: ne (m⁻³)
  - Panel 2: Te (eV)
  - Panel 3: O, OH, H (cm⁻³)
  - Panel 4: O3, HO2, CH3, NO (cm⁻³)
  - ON phase shaded, continuous time axis in µs
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

P_PEAK = 32.5
DC = 0.20
P_AVG = P_PEAK * DC

cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {
    'PRF_Hz': 1333.0, 'duty_cycle': DC, 'P_peak_W': P_PEAK,
    'rise_time_s': 1.0e-7, 'waveform': 'trapezoidal',
}
V_EFF_CM3 = 4.9
cfg['V_eff'] = V_EFF_CM3 * 1e-6
cfg['reactor']['volume'] = V_EFF_CM3 * 1e-6
cfg['reactor']['pressure'] = 101325.0
Q_SLM = 0.4
cfg['flow'] = {'Q_slm': Q_SLM, 'model': 'PFR'}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}
T_K = 303.0
cfg['initial'] = {'T_gas': T_K, 'ne': 1e8, 'Te_eV': 0.026}
cfg['T_wall'] = T_K
cfg['wall_loss_freq'] = 10000.0
Q_actual = Q_SLM * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
tau = V_EFF_CM3 * 1e-6 / Q_actual
PRF = 1333.0
period = 1.0 / PRF
n_pulses_tau = int(tau * PRF)

cfg['solver'] = {
    't_end': tau, 'n_points': 1000, 'method': 'BDF',
    'rtol': 1e-5, 'atol': 1e-10, 'max_step': 1e-6, 'constrained': False,
}

print(f"Pulsed 1τ: P_peak={P_PEAK}W, P_avg={P_AVG}W, {n_pulses_tau} pulses")

with contextlib.redirect_stdout(io.StringIO()):
    solver, y0, t_span, co = setup_simulation(cfg, base_dir)

sm = solver.sm
n_sp = sm.n_species
n_state = y0.shape[0]
power = solver.power

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
constraints = np.ones(n_state, dtype=np.float64)
constraints[sm.idx_Tgas] = 0.0

N_LAST = 15
last_start = n_pulses_tau - N_LAST

# --- Run ---
t0_wall = time.time()
y = y0.copy()
n_fail = 0

all_ts = []
all_ys = []

for ip in range(n_pulses_tau):
    t_start = ip * period
    t_on_end = t_start + _t_on
    t_end_p = (ip + 1) * period

    y[:n_sp] = np.maximum(y[:n_sp], 0.0)
    ce_seed = 1e8 / NA
    if y[0] < ce_seed:
        y[0] = ce_seed
        y[sm.idx_energy] = 1e8 * 1.5 * 0.026

    rec = (ip >= last_start)

    if rec:
        all_ts.append(t_start)
        all_ys.append(y.copy())

    # ON
    cvode_on = CVODESolver(n_state, fast_rhs_on)
    cvode_on.setup(y, t_start, rtol=1e-5, atol=1e-10,
                   max_step=0.0, init_step=1e-12,
                   max_num_steps=100000, constraints=constraints)
    if rec:
        dt_sub = _t_on / 20
        t_cur = t_start
        for _ in range(20):
            t_next = min(t_cur + dt_sub, t_on_end)
            t_r, y_step, ret = cvode_on.step_to(t_next)
            all_ts.append(t_r)
            all_ys.append(y_step.copy())
            t_cur = t_r
            if ret < 0: n_fail += 1; break
        cvode_on.free()
        y = y_step.copy()
    else:
        t_r, y_on, ret = cvode_on.step_to(t_on_end)
        cvode_on.free()
        if ret < 0: n_fail += 1
        y = y_on.copy()

    y[:n_sp] = np.maximum(y[:n_sp], 0.0)
    eps_th = 1.5 * KB * T_K / QE
    y[sm.idx_energy] = y[0] * NA * eps_th

    if rec:
        all_ts.append(t_on_end)
        all_ys.append(y.copy())

    # OFF
    cvode_off = CVODESolver(n_state, rhs_off)
    cvode_off.setup(y, t_on_end, rtol=1e-4, atol=1e-8,
                    max_step=0.0, init_step=1e-9,
                    max_num_steps=200000, constraints=constraints)
    if rec:
        dt_sub = (t_end_p - t_on_end) / 30
        t_cur = t_on_end
        for _ in range(30):
            t_next = min(t_cur + dt_sub, t_end_p)
            t_r, y_step, ret = cvode_off.step_to(t_next)
            all_ts.append(t_r)
            all_ys.append(y_step.copy())
            t_cur = t_r
            if ret < 0: n_fail += 1; break
        cvode_off.free()
        y = y_step.copy()
    else:
        t_r, y_off, ret = cvode_off.step_to(t_end_p)
        cvode_off.free()
        if ret < 0: n_fail += 1
        y = y_off.copy()

    if (ip+1) % 100 == 0 or ip == n_pulses_tau - 1:
        elapsed = time.time() - t0_wall
        pct = (ip+1) / n_pulses_tau * 100
        ch4_now = (y0[sm.index('CH4')] - y[sm.index('CH4')]) / y0[sm.index('CH4')] * 100
        print(f"  [{pct:5.1f}%] Pulse {ip+1}/{n_pulses_tau}: "
              f"ne={y[0]*NA:.3e}, CH4={ch4_now:.3f}%, {elapsed:.0f}s")

wall = time.time() - t0_wall
ch4_conv = (y0[sm.index('CH4')] - y[sm.index('CH4')]) / y0[sm.index('CH4')] * 100
print(f"\n  Done: CH4={ch4_conv:.4f}%, {wall:.0f}s, fails={n_fail}")

# --- Plot (matching pulsed_last15_profiles.png format) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ts = np.array(all_ts)
ys = np.array(all_ys)

t0_plot = ts[0]
t_us = (ts - t0_plot) * 1e6  # µs

ne_arr = ys[:, 0] * NA  # m⁻³
ne_eps_arr = ys[:, sm.idx_energy]
eps_arr = np.where(ne_arr > 1, ne_eps_arr / ne_arr, 0.039)
Te_arr = (2.0/3.0) * eps_arr

# Species in cm⁻³
def get_cm3(name):
    if sm.has(name):
        return np.maximum(ys[:, sm.index(name)] * NA * 1e-6, 1e-1)
    return np.ones(len(ts)) * 1e-1

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'Last {N_LAST} pulses (quasi-steady), P_peak={P_PEAK}W, V_eff={V_EFF_CM3}cm³\n'
             f'EI enabled in OFF phase, CH4 conv={ch4_conv:.2f}%',
             fontsize=12)

# ON phase shading
for ip_rel in range(N_LAST):
    t_on_s = ip_rel * period * 1e6
    t_on_e = (ip_rel * period + _t_on) * 1e6
    for ax in axes:
        ax.axvspan(t_on_s, t_on_e, alpha=0.12, color='orange', linewidth=0)

# Panel 1: ne
axes[0].semilogy(t_us, np.maximum(ne_arr, 1), 'b-', linewidth=0.8)
axes[0].set_ylabel('$n_e$ (m$^{-3}$)')
axes[0].set_ylim(1e4, 1e15)

# Panel 2: Te
axes[1].plot(t_us, Te_arr, 'r-', linewidth=0.8)
axes[1].set_ylabel('$T_e$ (eV)')
axes[1].set_ylim(-0.1, 3.0)

# Panel 3: Short-lived radicals (O, OH, H) in cm⁻³
axes[2].semilogy(t_us, get_cm3('O'), 'g-', linewidth=0.8, label='O')
axes[2].semilogy(t_us, get_cm3('OH'), 'b-', linewidth=0.8, label='OH')
axes[2].semilogy(t_us, get_cm3('H'), 'r-', linewidth=0.8, label='H')
axes[2].set_ylabel('Density (cm$^{-3}$)')
axes[2].legend(fontsize=9)

# Panel 4: Longer-lived species (O3, HO2, CH3, NO) in cm⁻³
axes[3].semilogy(t_us, get_cm3('O3'), 'm-', linewidth=0.8, label='O3')
axes[3].semilogy(t_us, get_cm3('HO2'), 'r-', linewidth=0.8, label='HO2')
axes[3].semilogy(t_us, get_cm3('CH3'), color='orange', linewidth=0.8, label='CH3')
axes[3].semilogy(t_us, get_cm3('NO'), 'k--', linewidth=0.8, label='NO')
axes[3].set_ylabel('Density (cm$^{-3}$)')
axes[3].set_xlabel('Time (µs)')
axes[3].legend(fontsize=9)

plt.tight_layout()
out_path = os.path.join(base_dir, 'output', 'pulsed_last10_nofreeze.png')
plt.savefig(out_path, dpi=150)
print(f"  Plot saved: {out_path}")
