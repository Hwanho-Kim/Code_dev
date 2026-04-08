"""Test: OFF phase WITHOUT thermal 3-body O₂ attachment only.

3체 부착(rxn 165)을 OFF phase에서 비활성화.
DR(e+ion⁺), 탈착, diffusion은 그대로 활성.
ne가 어떻게 변하는지 관찰.
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

cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {
    'PRF_Hz': 1333.0, 'duty_cycle': 0.20, 'P_peak_W': 8.1,
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
cfg['solver'] = {
    't_end': tau, 'n_points': 1000, 'method': 'BDF',
    'rtol': 1e-5, 'atol': 1e-10, 'max_step': 1e-6, 'constrained': False,
}

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

constraints_on = np.ones(n_state, dtype=np.float64)
constraints_on[sm.idx_Tgas] = 0.0

N_TEST_PULSES = 15

# Disable thermal attachment injection in rhs_off
print("Disabling thermal 3-body attachment (rxn 165) in OFF phase")
print(f"  solver._att165_global_idx was: {solver._att165_global_idx}")
solver._att165_global_idx = None
print(f"  solver._att165_global_idx now: {solver._att165_global_idx}")
print()

solver._diag_off = False
rhs_off = solver.rhs_off

y = y0.copy()
t0_wall = time.time()
n_fail_on = 0
n_fail_off = 0

print(f"{'='*70}")
print(f"  WITHOUT thermal 3-body att - {N_TEST_PULSES} pulses")
print(f"  Active OFF-phase e⁻ processes: DR, O₃ att, detachment, diffusion")
print(f"  Disabled: thermal 3-body O₂ attachment (rxn 165)")
print(f"{'='*70}")

for ip in range(N_TEST_PULSES):
    t_start = ip * period
    t_on_end = t_start + _t_on
    t_end_p = (ip + 1) * period

    y[:n_sp] = np.maximum(y[:n_sp], 0.0)

    # Re-seed
    ce_seed = 1e8 / NA
    if y[0] < ce_seed:
        y[0] = ce_seed
        y[sm.idx_energy] = 1e8 * 1.5 * 0.026

    # ON phase
    cvode_on = CVODESolver(n_state, fast_rhs_on)
    cvode_on.setup(y, t_start, rtol=1e-5, atol=1e-10,
                   max_step=0.0, init_step=1e-12,
                   max_num_steps=100000, constraints=constraints_on)
    t_r, y_on, ret_on = cvode_on.step_to(t_on_end)
    rhs_on = cvode_on._get_stats()['n_rhs_evals']
    cvode_on.free()
    if ret_on < 0: n_fail_on += 1
    y = y_on.copy()
    ne_peak = y[0] * NA

    # Clamp + thermal reset
    y[:n_sp] = np.maximum(y[:n_sp], 0.0)
    eps_th = 1.5 * KB * T_K / QE
    y[sm.idx_energy] = y[0] * NA * eps_th

    ne_off_start = y[0] * NA

    # OFF phase
    cvode_off = CVODESolver(n_state, rhs_off)
    cvode_off.setup(y, t_on_end, rtol=1e-4, atol=1e-8,
                    max_step=0.0, init_step=1e-9,
                    max_num_steps=200000, constraints='none')
    t_r, y_off, ret_off = cvode_off.step_to(t_end_p)
    rhs_off_count = cvode_off._get_stats()['n_rhs_evals']
    cvode_off.free()
    if ret_off < 0: n_fail_off += 1
    y = y_off.copy()
    ne_valley = y[0] * NA

    # Key species
    c_Om = y[sm.index('O-')] * NA if sm.has('O-') else 0
    c_O2m = y[sm.index('O2-')] * NA if sm.has('O2-') else 0

    # Ion totals for DR estimate
    n_ion_pos = sum(y[idx] * NA for idx in solver._positive_ion_indices)
    n_ion_neg = sum(y[idx] * NA for idx in solver._negative_ion_indices)

    decades_drop = np.log10(max(ne_peak, 1)) - np.log10(max(abs(ne_valley), 1))
    flag = ""
    if ret_on < 0: flag += " [ON FAIL]"
    if ret_off < 0: flag += " [OFF FAIL]"

    print(f"  Pulse {ip:3d}: ne_peak={ne_peak:.3e} → ne_valley={ne_valley:.3e} "
          f"(Δ={decades_drop:.1f} dec), "
          f"O⁻={c_Om:.2e}, O₂⁻={c_O2m:.2e}, "
          f"n+={n_ion_pos:.2e}, n-={n_ion_neg:.2e}, "
          f"rhs={rhs_on}+{rhs_off_count}{flag}")

wall = time.time() - t0_wall
print(f"\n  Total: {wall:.1f}s, fails ON={n_fail_on} OFF={n_fail_off}")

print(f"\n  Baseline comparison (WITH thermal att):")
print(f"    Pulse 0: ne_valley ~ 7.5e+07 → 3.1e+06 → 1.9e+05 → ... → ~10-100 m⁻³")
print(f"    (6+ decades drop within a few pulses)")
