"""Diagnostic script: OFF-phase electron budget analysis.

Runs 1 pulse (ON + OFF) and records per-RHS-call electron budget
to identify the dominant loss mechanism in afterglow.

Usage:
    cd /home/hawn/work && python plasma0d_v2/diag_off_budget.py
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import NA, KB, QE


def run_diagnostic():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, 'config.yaml')
    cfg = load_config(cfg_path)

    # Override to pulsed mode matching reference conditions
    cfg['power_mode'] = 'pulsed'
    cfg['pulse'] = {
        'PRF_Hz': 1333.0,
        'duty_cycle': 0.20,
        'P_peak_W': 32.5,
        'rise_time_s': 1.0e-7,
        'waveform': 'trapezoidal',
    }
    cfg['V_eff'] = 4.9e-6
    cfg['initial']['T_gas'] = 303.0
    cfg['T_wall'] = 303.0

    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)

    T_pulse = solver.power.period
    t_on = solver.power._pulse_duty_cycle * T_pulse
    rise = solver.power._pulse_rise_time
    t_on_eff = max(t_on - rise, rise)
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC: OFF-phase electron budget")
    print(f"{'='*60}")
    print(f"  T_pulse = {T_pulse*1e6:.1f} µs")
    print(f"  t_on_eff = {t_on_eff*1e6:.1f} µs")
    print(f"  t_off = {(T_pulse - t_on_eff)*1e6:.1f} µs")

    # --- Step 1: Run ON phase (1 pulse) ---
    print(f"\n  [Step 1] ON phase: 0 → {t_on_eff*1e6:.1f} µs")
    sol_on = solve_ivp(
        solver.rhs, [0.0, t_on_eff], y0,
        method='BDF', rtol=1e-5, atol=1e-10,
        max_step=1e-7,
    )
    if not sol_on.success:
        print(f"    WARNING: ON phase solver failed: {sol_on.message}")
    y_on_end = sol_on.y[:, -1]
    ne_on = y_on_end[0] * NA
    ne_eps_on = y_on_end[solver.sm.idx_energy]
    Te_on = (2.0/3.0) * ne_eps_on / ne_on if ne_on > 1 else 0
    print(f"    ON end: ne = {ne_on:.3e} m⁻³, Te = {Te_on:.3f} eV")

    # --- Step 2: Thermal reset (as in production solver) ---
    y_off_start = y_on_end.copy()
    T_gas = y_off_start[solver.sm.idx_Tgas]
    eps_th = 1.5 * KB * max(T_gas, 200.0) / QE
    n_e_start = y_off_start[0] * NA
    y_off_start[solver.sm.idx_energy] = n_e_start * eps_th
    print(f"    Thermal reset: eps_th = {eps_th:.4f} eV, "
          f"ne_eps = {y_off_start[solver.sm.idx_energy]:.3e}")

    # --- Step 3: Run OFF phase with diagnostics ---
    print(f"\n  [Step 2] OFF phase: {t_on_eff*1e6:.1f} → {T_pulse*1e6:.1f} µs (diagnostics ON)")
    solver._diag_off = True
    solver._diag_off_records = []

    sol_off = solve_ivp(
        solver.rhs_off,
        [t_on_eff, T_pulse],
        y_off_start,
        method='BDF', rtol=1e-3, atol=1e-8,
    )
    if not sol_off.success:
        print(f"    WARNING: OFF phase solver failed: {sol_off.message}")

    solver._diag_off = False

    y_off_end = sol_off.y[:, -1]
    ne_off = y_off_end[0] * NA
    print(f"    OFF end: ne = {ne_off:.3e} m⁻³")
    print(f"    ne decay: {ne_on:.2e} → {ne_off:.2e} "
          f"({np.log10(max(ne_on, 1)) - np.log10(max(ne_off, 1)):.1f} orders)")

    # --- Step 4: Analyze budget ---
    records = solver._diag_off_records
    if not records:
        print("    No diagnostic records collected!")
        return

    print(f"\n  [Step 3] Electron budget analysis ({len(records)} RHS calls)")

    # Convert to arrays for analysis
    t_arr = np.array([r['t'] for r in records])
    ne_arr = np.array([r['n_e'] for r in records])
    dr_arr = np.array([r['DR'] for r in records])        # mol/(m³·s), negative
    o3att_arr = np.array([r['O3_att'] for r in records])  # mol/(m³·s), negative
    thatt_arr = np.array([r.get('therm_att', 0.0) for r in records])  # thermal 3-body att
    det_arr = np.array([r['detach'] for r in records])    # mol/(m³·s), positive
    diff_arr = np.array([r['diff'] for r in records])     # mol/(m³·s), negative
    flow_arr = np.array([r['flow'] for r in records])
    dydt_arr = np.array([r['dydt_e'] for r in records])
    Te_arr = np.array([r['Te_eV'] for r in records])
    diff_freq_arr = np.array([r['diff_freq'] for r in records])

    # Species concentrations
    c_O_neg = np.array([r['c_O-'] for r in records])
    c_O2_neg = np.array([r['c_O2-'] for r in records])
    c_O = np.array([r['c_O'] for r in records])
    c_O3 = np.array([r['c_O3'] for r in records])
    c_N2A = np.array([r['c_N2A'] for r in records])
    c_O2a = np.array([r['c_O2a'] for r in records])

    # Convert mol/(m³·s) to m⁻³/s for easier interpretation
    dr_m3s = dr_arr * NA
    o3att_m3s = o3att_arr * NA
    thatt_m3s = thatt_arr * NA
    det_m3s = det_arr * NA
    diff_m3s = diff_arr * NA

    # Print time-resolved snapshots
    print(f"\n  {'t(µs)':>8} {'ne(m⁻³)':>12} {'Te(eV)':>8} "
          f"{'DR':>12} {'O3att':>12} {'ThAtt':>12} {'Detach':>12} {'Diff':>12} "
          f"{'Net':>12} {'τ_e(µs)':>10}")
    print(f"  {'-'*124}")

    # Print 20 evenly spaced snapshots
    n_print = min(20, len(records))
    indices = np.linspace(0, len(records)-1, n_print, dtype=int)
    for i in indices:
        r = records[i]
        t_us = (r['t'] - t_on_eff) * 1e6
        net = dr_m3s[i] + o3att_m3s[i] + thatt_m3s[i] + det_m3s[i] + diff_m3s[i]
        tau_e = -ne_arr[i] / net if abs(net) > 1e-10 else float('inf')
        print(f"  {t_us:8.2f} {ne_arr[i]:12.3e} {Te_arr[i]:8.4f} "
              f"{dr_m3s[i]:12.3e} {o3att_m3s[i]:12.3e} "
              f"{thatt_m3s[i]:12.3e} {det_m3s[i]:12.3e} {diff_m3s[i]:12.3e} "
              f"{net:12.3e} {tau_e*1e6:10.2f}")

    # Time-integrated budget
    # Sort by time for proper integration
    sort_idx = np.argsort(t_arr)
    t_sorted = t_arr[sort_idx]
    # Use unique time points only (BDF may re-evaluate at same t)
    t_unique, unique_idx = np.unique(t_sorted, return_index=True)

    if len(t_unique) > 1:
        dr_sorted = dr_m3s[sort_idx][unique_idx]
        o3att_sorted = o3att_m3s[sort_idx][unique_idx]
        thatt_sorted = thatt_m3s[sort_idx][unique_idx]
        det_sorted = det_m3s[sort_idx][unique_idx]
        diff_sorted = diff_m3s[sort_idx][unique_idx]

        int_dr = np.trapezoid(dr_sorted, t_unique)
        int_o3att = np.trapezoid(o3att_sorted, t_unique)
        int_thatt = np.trapezoid(thatt_sorted, t_unique)
        int_det = np.trapezoid(det_sorted, t_unique)
        int_diff = np.trapezoid(diff_sorted, t_unique)
        int_total = int_dr + int_o3att + int_thatt + int_det + int_diff

        print(f"\n  Time-integrated electron budget (OFF phase):")
        print(f"    DR (dissociative recomb.)  : {int_dr:+.4e} m⁻³")
        print(f"    O3 attachment              : {int_o3att:+.4e} m⁻³")
        print(f"    Thermal 3-body attachment  : {int_thatt:+.4e} m⁻³")
        print(f"    Detachment (Arrhenius)     : {int_det:+.4e} m⁻³")
        print(f"    Diffusion loss             : {int_diff:+.4e} m⁻³")
        print(f"    Total                      : {int_total:+.4e} m⁻³")
        print(f"    Δne (actual)               : {ne_off - ne_on:+.4e} m⁻³")

        total_loss = abs(int_dr) + abs(int_o3att) + abs(int_thatt) + abs(int_diff)
        if total_loss > 0:
            print(f"\n  Loss fractions:")
            print(f"    DR       : {abs(int_dr)/total_loss*100:6.1f}%")
            print(f"    O3 att   : {abs(int_o3att)/total_loss*100:6.1f}%")
            print(f"    ThAtt    : {abs(int_thatt)/total_loss*100:6.1f}%")
            print(f"    Diffusion: {abs(int_diff)/total_loss*100:6.1f}%")

    # Print key species at OFF start and end
    print(f"\n  Key species at OFF start/end:")
    print(f"    {'Species':>12} {'Start':>12} {'End':>12} {'Ratio':>8}")
    for name, arr_val in [('O-', c_O_neg), ('O2-', c_O2_neg),
                           ('O', c_O), ('O3', c_O3),
                           ('N2(A)', c_N2A), ('O2(a1Dg)', c_O2a)]:
        v0 = arr_val[0] * NA if len(arr_val) > 0 else 0
        vf = arr_val[-1] * NA if len(arr_val) > 0 else 0
        ratio = vf / v0 if v0 > 0 else 0
        print(f"    {name:>12} {v0:12.3e} {vf:12.3e} {ratio:8.3f}")

    # Diffusion length analysis
    print(f"\n  Diffusion parameters:")
    print(f"    Lambda = {solver.ekin.Lambda:.2e} m")
    print(f"    Lambda² = {solver.ekin.Lambda_sq:.2e} m²")
    print(f"    diff_freq at OFF start = {diff_freq_arr[0]:.2e} 1/s")
    print(f"    diff_freq at OFF end   = {diff_freq_arr[-1]:.2e} 1/s")
    print(f"    D_a/Λ² timescale = {1.0/diff_freq_arr[0]*1e6:.2f} µs")

    print(f"\n  Elapsed OFF time = {(T_pulse - t_on_eff)*1e6:.1f} µs")
    print(f"  Done.")


if __name__ == '__main__':
    run_diagnostic()
