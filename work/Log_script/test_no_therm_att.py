"""Test: OFF phase with/without thermal 3-body O₂ attachment.

질문: rhs_off에서 thermal 3체 부착(rxn 165)을 끄면 ne가 어떻게 되는가?
- DR(e+ion⁺)이 있으므로 ne가 무한히 오르진 않고 준정상 상태 도달 예상
- 탈착이 DR보다 크면 ne 일시 상승 → DR이 커져서 평형
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

N_TEST_PULSES = 20

def run_test(label, disable_therm_att=False):
    """Run N_TEST_PULSES with optional thermal attachment disable."""
    print(f"\n{'='*70}")
    print(f"  TEST: {label}")
    print(f"  Thermal 3-body att (rxn 165) in OFF phase: {'OFF' if disable_therm_att else 'ON'}")
    print(f"{'='*70}")

    saved_idx = solver._att165_global_idx
    if disable_therm_att:
        solver._att165_global_idx = None

    # NO diagnostics (too slow)
    solver._diag_off = False
    rhs_off = solver.rhs_off

    y = y0.copy()
    t0_wall = time.time()
    n_fail_on = 0
    n_fail_off = 0
    records = []

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
                        max_num_steps=50000, constraints='none')
        t_r, y_off, ret_off = cvode_off.step_to(t_end_p)
        rhs_off_count = cvode_off._get_stats()['n_rhs_evals']
        cvode_off.free()
        if ret_off < 0: n_fail_off += 1
        y = y_off.copy()
        ne_valley = y[0] * NA

        # Key species at end of OFF
        c_Om = y[sm.index('O-')] * NA if sm.has('O-') else 0
        c_O2m = y[sm.index('O2-')] * NA if sm.has('O2-') else 0

        records.append({
            'pulse': ip, 'ne_peak': ne_peak, 'ne_off_start': ne_off_start,
            'ne_valley': ne_valley, 'rhs_off': rhs_off_count,
            'fail_on': ret_on < 0, 'fail_off': ret_off < 0,
            'c_Om': c_Om, 'c_O2m': c_O2m,
        })

        status = f"  Pulse {ip:3d}: ne_peak={ne_peak:.3e}, ne_valley={ne_valley:.3e}"
        status += f", O⁻={c_Om:.2e}, O₂⁻={c_O2m:.2e}"
        status += f", rhs={rhs_on}+{rhs_off_count}"
        if ret_on < 0: status += " [ON FAIL]"
        if ret_off < 0: status += " [OFF FAIL]"
        print(status)

    wall = time.time() - t0_wall
    solver._att165_global_idx = saved_idx

    print(f"\n  Total: {wall:.1f}s, fails ON={n_fail_on} OFF={n_fail_off}")
    return records

# --- Run both tests ---
rec_with = run_test("WITH thermal 3-body att (baseline)", disable_therm_att=False)
rec_without = run_test("WITHOUT thermal 3-body att", disable_therm_att=True)

# --- Comparison ---
print(f"\n{'='*70}")
print(f"  COMPARISON: ne_valley [m⁻³]")
print(f"{'='*70}")
print(f"  {'Pulse':>5} {'WITH att':>14} {'WITHOUT att':>14} {'ratio':>10}")
for rw, rwo in zip(rec_with, rec_without):
    ratio = rwo['ne_valley'] / rw['ne_valley'] if rw['ne_valley'] > 0 else float('inf')
    print(f"  {rw['pulse']:5d} {rw['ne_valley']:14.3e} {rwo['ne_valley']:14.3e} {ratio:10.1f}x")

# Print ne evolution for the WITHOUT case
print(f"\n  WITHOUT att - ne evolution:")
for r in rec_without:
    peak_log = np.log10(max(r['ne_peak'], 1))
    valley_log = np.log10(max(r['ne_valley'], 1))
    print(f"    Pulse {r['pulse']:3d}: peak={r['ne_peak']:.3e}  valley={r['ne_valley']:.3e}  "
          f"(drop={peak_log-valley_log:.1f} decades)")
