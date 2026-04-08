"""Newton-Krylov shooting for pulsed CSTR periodic steady state.

Usage:
    source ~/work/.venv/bin/activate
    nohup python run_pulsed_shooting.py > shooting_run.log 2>&1 &
    tail -f shooting_run.log

Expected runtime: ~30-120 minutes
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import io, contextlib
from scipy.optimize import newton_krylov
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import NA

cfg = load_config(os.path.join(os.path.dirname(__file__), 'plasma0d_v2', 'config.yaml'))
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {'PRF_Hz': 10000.0, 'duty_cycle': 0.20, 'P_peak_W': 8.1,
                'rise_time_s': 2e-6, 'waveform': 'trapezoidal'}
cfg['reactor'] = {'volume': 250e-6, 'pressure': 101325.0}
cfg['flow'] = {'Q_slm': 0.4, 'model': 'CSTR'}
cfg['T_wall'] = 303.0
cfg['wall_loss_freq'] = 10000.0
cfg['initial'] = {'T_gas': 303.0, 'ne': 1e8, 'Te_eV': 0.026}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

with contextlib.redirect_stdout(io.StringIO()):
    solver, y0, _, _ = setup_simulation(cfg, os.path.join(os.path.dirname(__file__), 'plasma0d_v2'))

T_period = 100e-6
w = np.load(os.path.join(os.path.dirname(__file__), 'plasma0d_v2', 'scale_weights.npy'))
n_flow = [0]
t_global = [time.time()]

def flow_map(y0_in):
    n_flow[0] += 1
    y_c = np.maximum(y0_in, 1e-30)
    y_c[0] = max(y_c[0], solver._ce_floor)
    y_c[solver.sm.idx_energy] = max(y_c[solver.sm.idx_energy], solver._ne_eps_floor)
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver._solve_constrained(
            y_c, (0.0, T_period),
            np.array([0.0, T_period]), 2,
            method='BDF', rtol=1e-5, atol=1e-8, max_step=1e-6)
    return result.y[:, -1]

def residual_scaled(y0_in):
    return (flow_map(y0_in) - y0_in) / w

# Warmup
print("=== Pulsed CSTR Newton-Krylov Shooting ===")
print(f"PRF=10kHz, dc=20%, P_peak=8.1W, T_period={T_period*1e6:.0f}us")
print(f"\nPhase 1: Warmup (20 periods)...")
sys.stdout.flush()

y = y0.copy()
n_warmup = 100
print(f"\nPhase 1: Warmup ({n_warmup} periods)...")
sys.stdout.flush()
for i in range(n_warmup):
    t0 = time.time()
    y = flow_map(y)
    dt = time.time() - t0
    if (i+1) % 20 == 0:
        ne = y[0]*NA
        ch4_i = solver.sm.index('CH4')
        conv = (y0[ch4_i]-y[ch4_i])/y0[ch4_i]*100
        res_norm = np.linalg.norm((flow_map(y) - y) / w)
        elapsed = time.time() - t_global[0]
        print(f"  period {i+1}: ne={ne:.2e}, conv={conv:.4f}%, "
              f"|F|={res_norm:.1f}, dt={dt:.1f}s, total={elapsed:.0f}s")
        sys.stdout.flush()

print(f"\nPhase 2: Newton-Krylov shooting...")
print(f"  f_tol=100, maxiter=30, inner_maxiter=10")
sys.stdout.flush()

t0_nk = time.time()
try:
    y_pss = newton_krylov(
        residual_scaled, y,
        method='lgmres',
        verbose=True,
        f_tol=100.0,
        inner_maxiter=10,
        maxiter=30,
        rdiff=1e-5
    )
    elapsed_nk = time.time() - t0_nk
    elapsed_total = time.time() - t_global[0]

    ne = y_pss[0] * NA
    ch4_i = solver.sm.index('CH4')
    conv = (y0[ch4_i] - y_pss[ch4_i]) / y0[ch4_i] * 100
    ne_eps = y_pss[solver.sm.idx_energy]
    eps = ne_eps / ne if ne > 1 else 0.039
    Te = 2/3 * eps

    print(f"\n=== PSS FOUND ===")
    print(f"Newton-Krylov: {elapsed_nk:.0f}s, {n_flow[0]} flow maps")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"CH4 conversion: {conv:.4f}%")
    print(f"ne = {ne:.2e} m-3")
    print(f"Te = {Te:.3f} eV")
    print(f"T_gas = {y_pss[solver.sm.idx_Tgas]:.2f} K")
    print(f"\nExpected (continuous P_avg=1.62W CSTR): ~1.44%")

    np.savez(os.path.join(os.path.dirname(__file__), 'pulsed_pss_result.npz'),
             y_pss=y_pss, y0=y0, w=w,
             species_names=solver.sm.names,
             n_flow_maps=n_flow[0],
             wall_time=elapsed_total)
    print("Saved to pulsed_pss_result.npz")

except Exception as e:
    elapsed_total = time.time() - t_global[0]
    print(f"\nDid not converge: {elapsed_total:.0f}s, {n_flow[0]} flow maps")
    ne = y[0]*NA
    ch4_i = solver.sm.index('CH4')
    conv = (y0[ch4_i]-y[ch4_i])/y0[ch4_i]*100
    print(f"Last state: ne={ne:.2e}, conv={conv:.4f}%")
    print(f"Error: {str(e)[:200]}")

    np.savez(os.path.join(os.path.dirname(__file__), 'pulsed_pss_result.npz'),
             y_last=y, y0=y0, w=w,
             species_names=solver.sm.names,
             n_flow_maps=n_flow[0],
             wall_time=elapsed_total,
             converged=False)
