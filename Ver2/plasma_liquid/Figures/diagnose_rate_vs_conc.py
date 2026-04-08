#!/usr/bin/env python3
"""
Diagnostic: compare volume-averaged concentration change (ΔC/Δt)
with the rate-budget net dC/dt for NO2-, H2O2, NO3-, O3.

If the rate budget is correct, they should match.
"""
import sys, os, time as time_mod
from pathlib import Path
import numpy as np

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from gen_fig2_rate_evolution import (
    load_gas_data, run_simulation, compute_rates_snapshot,
    species_contribution, TARGET_SPECIES, SPECIES_LABEL,
    SPEC_TO_TOTAL, _smooth,
)


def main():
    os.chdir(_project_root)
    times, gas_conc = load_gas_data()
    snap_times, snaps, solver = run_simulation(times, gas_conc)

    chem = solver.chem
    dz = solver.dz_cells
    L = solver.L
    nt = len(snap_times)

    # 1. Volume-averaged concentrations at each snapshot
    conc_avg = {}
    for sp_name in TARGET_SPECIES:
        # Map species name to solver index
        # For total variables (e.g. HONO2_total for NO3-), use total
        total_name = SPEC_TO_TOTAL.get(sp_name, sp_name)
        if total_name in chem.species_idx:
            idx = chem.species_idx[total_name]
        elif sp_name in chem.species_idx:
            idx = chem.species_idx[sp_name]
        else:
            print(f"  WARNING: {sp_name} not found in species_idx")
            continue
        c_arr = np.zeros(nt)
        for i, y2d in enumerate(snaps):
            c_arr[i] = np.dot(y2d[:, idx], dz) / L
        conc_avg[sp_name] = c_arr
        print(f"\n{SPECIES_LABEL.get(sp_name, sp_name)} "
              f"(idx={idx}, var={total_name if total_name != sp_name else sp_name}):")
        print(f"  C(0)  = {c_arr[0]:.4e} M")
        print(f"  C(end)= {c_arr[-1]:.4e} M")
        print(f"  ΔC    = {c_arr[-1]-c_arr[0]:.4e} M")

    # 2. Numerical dC/dt from concentration differences
    print("\n" + "="*70)
    print("Comparison: numerical ΔC/Δt vs rate-budget net")
    print("="*70)

    # Compute rate budget
    print("\nComputing per-reaction rates at each snapshot...")
    all_rxn_rates, all_mt_flux = [], []
    for i, (tv, y2d) in enumerate(zip(snap_times, snaps)):
        rr, mf = compute_rates_snapshot(solver, y2d, tv)
        all_rxn_rates.append(rr)
        all_mt_flux.append(mf)

    for sp_name in TARGET_SPECIES:
        if sp_name not in conc_avg:
            continue
        c = conc_avg[sp_name]
        sp_lbl = SPECIES_LABEL.get(sp_name, sp_name)

        # Numerical dC/dt (central difference)
        dt = np.diff(snap_times)
        dcdt_num = np.diff(c) / dt  # length nt-1

        # Rate-budget net at each snapshot
        net_budget = np.zeros(nt)
        for i in range(nt):
            contribs = species_contribution(
                all_rxn_rates[i], sp_name, all_mt_flux[i])
            net_budget[i] = sum(rate for _, rate in contribs)

        # Mid-point average for comparison with numerical dC/dt
        net_mid = 0.5 * (net_budget[:-1] + net_budget[1:])

        # Print comparison at several time points
        print(f"\n{sp_lbl}:")
        print(f"  {'t(s)':>6s}  {'C(M)':>12s}  {'ΔC/Δt(M/s)':>12s}  "
              f"{'net_budget':>12s}  {'ratio':>8s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
        for j in [0, 5, 10, 20, 35, 50, 71]:
            if j >= len(dcdt_num):
                continue
            t = 0.5*(snap_times[j] + snap_times[j+1])
            ratio = dcdt_num[j] / net_mid[j] if abs(net_mid[j]) > 1e-30 else float('inf')
            print(f"  {t:6.0f}  {c[j]:12.4e}  {dcdt_num[j]:+12.4e}  "
                  f"{net_mid[j]:+12.4e}  {ratio:8.2f}")

        # Overall integral check
        integral_num = c[-1] - c[0]
        integral_budget = np.trapezoid(net_budget, snap_times)
        print(f"  ∫net dt (budget) = {integral_budget:+.4e} M")
        print(f"  ΔC (actual)      = {integral_num:+.4e} M")
        if abs(integral_num) > 1e-20:
            print(f"  ratio            = {integral_budget/integral_num:.4f}")


if __name__ == '__main__':
    main()
