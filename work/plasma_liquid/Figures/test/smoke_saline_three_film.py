#!/usr/bin/env python3
"""DIW + Saline 3-voltage verification under new defaults:
  - bc_type = 'three_film' (project default 2026-04-23)
  - H2O2/O3 ratio = 0.003 (re-fit 2026-04-23)
  - N2O4 from Humid_fit-rescaled NO2 (gen_all_figures fix)

Imports load_gas_data from gen_all_figures.py so gas preprocessing matches
the canonical pipeline exactly (Dry shape × SS_80/SS_dry rescale).

Total runs: 3 voltages × {DIW, Saline} = 6 sims, ~5min each.
"""
from __future__ import annotations

import functools
import sys
import time as time_mod
from pathlib import Path

import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))
sys.path.insert(0, str(_root / 'Figures'))

import gen_all_figures as gaf
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

print = functools.partial(print, flush=True)

EXP_BY_MODE = {'DIW': gaf.EXP_DIW_ALL, 'Saline': gaf.EXP_SALINE_ALL}
EXP_KEY_MAP = {'pH': 'pH', 'NO3-': 'NO3', 'NO2-': 'NO2', 'H2O2': 'H2O2'}


def load_humid_fit(voltage: str):
    """Use gen_all_figures.load_gas_data() to match canonical gas processing
    (Humid_fit shape preservation + N2O4 from rescaled NO2)."""
    gaf.DEFAULT_GAS_SHEET = voltage
    gaf.CONDITION_LABEL = 'Humid_fitting'
    times, gas_conc = gaf.load_gas_data()
    return times, gas_conc, gaf.HONO_GAS, gaf.HONO2_GAS, gaf.H2O2_GAS


def run_one(voltage: str, mode: str) -> dict:
    """mode: 'DIW' or 'Saline'."""
    is_saline = (mode == 'Saline')
    cation = 0.154 if is_saline else 0.0

    print()
    print("=" * 70)
    print(f"{mode:6s} {voltage} Humid_fitting  "
          f"(three_film, H2O2/O3={gaf.H2O2_RATIO})")
    print("=" * 70)

    times, gas, hono_g, hono2_g, h2o2_g = load_humid_fit(voltage)
    h2o2_peak = float(np.nanmax(h2o2_g)) if hasattr(h2o2_g, '__len__') else float(h2o2_g)
    print(f"  Gas peaks (cm^-3): O3={np.nanmax(gas['O3']):.2e}  "
          f"NO2={np.nanmax(gas['NO2']):.2e}  N2O5={np.nanmax(gas['N2O5']):.2e}  "
          f"N2O4={np.nanmax(gas['N2O4']):.2e}")
    print(f"  Unmeasured H2O2 peak: {h2o2_peak:.2e}")

    chem = AqueousChemistry1D(saline_mode=is_saline)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=is_saline, fixed_cation_conc=cation,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas,
        hono_gas=hono_g, hono2_gas=hono2_g, h2o2_gas=h2o2_g,
    )

    t_end = float(times[-1])
    t_eval = np.arange(2.0, t_end + 0.1, 2.0)
    y0 = solver.build_initial_condition(initial_pH=7.0)

    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval, y0=y0,
        verbose=False, dt_poisson=None,
    )
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    out = {
        'voltage': voltage,
        'mode': mode,
        'success': bool(result['success']),
        'wall_s': wall,
        'pH': float(result['pH_avg']),
        'NO3-': float(avg.get('NO3-', 0)) * 1e6,
        'NO2-': float(avg.get('NO2-', 0)) * 1e6,
        'H2O2': float(avg.get('H2O2', 0)) * 1e6,
        'Cl-': float(avg.get('Cl-', 0)) * 1e3,
    }
    print(f"  done {wall:.1f}s  success={out['success']}")
    return out


def main():
    voltages = ['2.6kV', '3.2kV', '3.6kV']
    results = []
    for v in voltages:
        for mode in ['DIW', 'Saline']:
            results.append(run_one(v, mode))

    by_key = {(r['voltage'], r['mode']): r for r in results}

    print()
    print("=" * 90)
    print(f"SUMMARY  (three_film, H2O2/O3={gaf.H2O2_RATIO}, "
          f"δ_liq=100µm, δ_gas=10mm)")
    print("=" * 90)
    print(f"{'V':6s} {'metric':7s} | "
          f"{'DIW sim':>10s} {'DIW exp':>10s} {'r':>5s}  | "
          f"{'Sal sim':>10s} {'Sal exp':>10s} {'r':>5s}")
    print("-" * 90)
    for V in voltages:
        d = by_key[(V, 'DIW')]
        s = by_key[(V, 'Saline')]
        ed = gaf.EXP_DIW_ALL[V]
        es = gaf.EXP_SALINE_ALL[V]
        for sim_key, exp_key in [('pH', 'pH'), ('NO3-', 'NO3'),
                                 ('NO2-', 'NO2'), ('H2O2', 'H2O2')]:
            ds, ss = d[sim_key], s[sim_key]
            de, se = ed[exp_key], es[exp_key]
            dr = f"{ds/de:.2f}" if de > 0 else "--"
            sr = f"{ss/se:.2f}" if se > 0 else "--"
            print(f"{V:6s} {sim_key:7s} | "
                  f"{ds:>10.3f} {de:>10.3f} {dr:>5s}  | "
                  f"{ss:>10.3f} {se:>10.3f} {sr:>5s}")
        print(f"{V:6s} {'Cl-(mM)':7s} | "
              f"{'--':>10s} {'--':>10s} {'--':>5s}  | "
              f"{s['Cl-']:>10.3f} {154.0:>10.3f} {'OK':>5s}")
        wd, ws = d['wall_s'], s['wall_s']
        print(f"{V:6s} wall=DIW {wd:.0f}s, Saline {ws:.0f}s")
        print("-" * 90)


if __name__ == '__main__':
    main()
