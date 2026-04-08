"""Pulsed trapezoidal 1τ: P_peak=32.5W (P_avg=6.5W), V_eff=4.9cm³.

EI reactions enabled in OFF phase (CX-based k_ei_conc).
Last 10 pulses saved as output/pulsed_last10_nofreeze.png.
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
P_PEAK = 32.5
DC = 0.20
P_AVG = P_PEAK * DC  # 6.5W

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

print(f"Pulsed trapezoidal 1τ: P_peak={P_PEAK}W, P_avg={P_AVG}W")
print(f"  PRF={PRF:.0f}Hz, dc={DC*100:.0f}%, V_eff={V_EFF_CM3}cm³")
print(f"  τ = {tau:.3f}s, n_pulses = {n_pulses_tau}")
print(f"  OFF phase: EI enabled (CX-based), constrained CVODE")
print()

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

# --- Last 10 pulse recording ---
N_LAST = 10
last10_start = n_pulses_tau - N_LAST
last10_data = []  # list of (t_array, y_array) per pulse

# --- Main loop ---
t0_wall = time.time()
y = y0.copy()
total_rhs_on = 0
total_rhs_off = 0
n_fail_on = 0
n_fail_off = 0

for ip in range(n_pulses_tau):
    t_start = ip * period
    t_on_end = t_start + _t_on
    t_end_p = (ip + 1) * period

    y[:n_sp] = np.maximum(y[:n_sp], 0.0)

    # Re-seed
    ce_seed = 1e8 / NA
    if y[0] < ce_seed:
        y[0] = ce_seed
        y[sm.idx_energy] = 1e8 * 1.5 * 0.026

    recording = (ip >= last10_start)
    pulse_ts = []
    pulse_ys = []

    if recording:
        pulse_ts.append(t_start)
        pulse_ys.append(y.copy())

    # ON phase
    cvode_on = CVODESolver(n_state, fast_rhs_on)
    cvode_on.setup(y, t_start, rtol=1e-5, atol=1e-10,
                   max_step=0.0, init_step=1e-12,
                   max_num_steps=100000, constraints=constraints)

    if recording:
        # Step through ON phase recording intermediate points
        dt_rec = (_t_on) / 20  # 20 sub-points in ON
        t_cur = t_start
        for _ in range(20):
            t_next = min(t_cur + dt_rec, t_on_end)
            t_r, y_step, ret = cvode_on.step_to(t_next)
            pulse_ts.append(t_r)
            pulse_ys.append(y_step.copy())
            t_cur = t_r
            if ret < 0:
                n_fail_on += 1
                break
        total_rhs_on += cvode_on._get_stats()['n_rhs_evals']
        cvode_on.free()
        y = y_step.copy()
    else:
        t_r, y_on, ret = cvode_on.step_to(t_on_end)
        total_rhs_on += cvode_on._get_stats()['n_rhs_evals']
        cvode_on.free()
        if ret < 0: n_fail_on += 1
        y = y_on.copy()

    ne_peak = y[0] * NA

    # Clamp + thermal reset
    y[:n_sp] = np.maximum(y[:n_sp], 0.0)
    eps_th = 1.5 * KB * T_K / QE
    y[sm.idx_energy] = y[0] * NA * eps_th

    if recording:
        pulse_ts.append(t_on_end)
        pulse_ys.append(y.copy())

    # OFF phase
    cvode_off = CVODESolver(n_state, rhs_off)
    cvode_off.setup(y, t_on_end, rtol=1e-4, atol=1e-8,
                    max_step=0.0, init_step=1e-9,
                    max_num_steps=200000, constraints=constraints)

    if recording:
        dt_rec = (t_end_p - t_on_end) / 30  # 30 sub-points in OFF
        t_cur = t_on_end
        for _ in range(30):
            t_next = min(t_cur + dt_rec, t_end_p)
            t_r, y_step, ret = cvode_off.step_to(t_next)
            pulse_ts.append(t_r)
            pulse_ys.append(y_step.copy())
            t_cur = t_r
            if ret < 0:
                n_fail_off += 1
                break
        total_rhs_off += cvode_off._get_stats()['n_rhs_evals']
        cvode_off.free()
        y = y_step.copy()
        last10_data.append((np.array(pulse_ts), np.array(pulse_ys)))
    else:
        t_r, y_off, ret = cvode_off.step_to(t_end_p)
        total_rhs_off += cvode_off._get_stats()['n_rhs_evals']
        cvode_off.free()
        if ret < 0: n_fail_off += 1
        y = y_off.copy()

    ne_valley = y[0] * NA

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

print(f"\n{'='*60}")
print(f"  RESULTS: Pulsed 1τ ({tau*1e3:.1f}ms, {n_pulses_tau} pulses)")
print(f"{'='*60}")
print(f"  CH4 conversion : {conv:.4f}%")
print(f"  ne_final       : {ne_final:.3e} m⁻³")
print(f"  T_gas_final    : {T_gas_final:.1f} K")
print(f"  Wall time      : {wall:.1f}s ({wall/60:.1f}min)")
print(f"  Speed          : {n_pulses_tau/wall:.2f} pulses/s")
print(f"  Fails          : ON={n_fail_on}, OFF={n_fail_off}")
print(f"{'='*60}")
print(f"\n  Reference (prev nofreeze): CH4=5.56% @ P_avg=6.5W")
print(f"  Continuous V_eff=4.9:      CH4=4.35% @ P=5.0W")

# --- Plot last 10 pulses ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'Pulsed last 10 pulses (P_peak={P_PEAK}W, P_avg={P_AVG}W, EI enabled in OFF)',
             fontsize=13)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, N_LAST))

for i, (ts, ys) in enumerate(last10_data):
    t_us = (ts - ts[0]) * 1e6  # relative µs
    ne_arr = ys[:, 0] * NA
    ne_eps_arr = ys[:, sm.idx_energy]
    eps_arr = np.where(ne_arr > 1, ne_eps_arr / ne_arr, 0.039)
    Te_arr = (2.0/3.0) * eps_arr

    pulse_num = last10_start + i
    lbl = f'P{pulse_num}'

    axes[0].semilogy(t_us, np.maximum(ne_arr, 1), color=colors[i], label=lbl, linewidth=0.8)
    axes[1].plot(t_us, Te_arr, color=colors[i], linewidth=0.8)

    if sm.has('O-'):
        axes[2].semilogy(t_us, np.maximum(ys[:, sm.index('O-')] * NA, 1),
                         color=colors[i], linewidth=0.8, linestyle='-')
    if sm.has('O2-'):
        axes[2].semilogy(t_us, np.maximum(ys[:, sm.index('O2-')] * NA, 1),
                         color=colors[i], linewidth=0.8, linestyle='--')

    if sm.has('CH4'):
        axes[3].plot(t_us, ys[:, sm.index('CH4')] / y0[sm.index('CH4')] * 100,
                     color=colors[i], linewidth=0.8)

axes[0].set_ylabel('ne [m⁻³]')
axes[0].legend(fontsize=7, ncol=5, loc='upper right')
axes[0].set_ylim(bottom=1e0)
axes[0].axvline(x=_t_on*1e6, color='red', linestyle=':', alpha=0.5, label='ON/OFF')

axes[1].set_ylabel('Te [eV]')
axes[1].axvline(x=_t_on*1e6, color='red', linestyle=':', alpha=0.5)

axes[2].set_ylabel('O⁻ (solid), O₂⁻ (dash) [m⁻³]')
axes[2].axvline(x=_t_on*1e6, color='red', linestyle=':', alpha=0.5)
axes[2].set_ylim(bottom=1e0)

axes[3].set_ylabel('CH4 remaining [%]')
axes[3].set_xlabel('Time within pulse [µs]')
axes[3].axvline(x=_t_on*1e6, color='red', linestyle=':', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(base_dir, 'output', 'pulsed_last10_nofreeze.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"\n  Plot saved: {out_path}")
