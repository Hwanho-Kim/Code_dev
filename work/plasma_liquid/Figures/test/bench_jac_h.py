#!/usr/bin/env python3
"""Benchmark FD Jacobian h_threshold change (Issue #4 fix).

3.2kV DIW Humid_fitting single run. Reports wall time + nfev/njev.
Baseline (h_threshold=1e-10): expected ~1.7min, nfev ~30-40K.
After fix (h_threshold=atol=1e-15): expected ~30-60s, nfev ~5-10K.
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


def main():
    voltage = '3.2kV'
    gaf.DEFAULT_GAS_SHEET = voltage
    gaf.CONDITION_LABEL = 'Humid_fitting'
    times, gas = gaf.load_gas_data()
    hono_g, hono2_g, h2o2_g = gaf.HONO_GAS, gaf.HONO2_GAS, gaf.H2O2_GAS

    print(f"3.2kV DIW Humid_fitting benchmark")
    print(f"  Gas peaks (cm^-3): O3={np.nanmax(gas['O3']):.2e}  "
          f"NO2={np.nanmax(gas['NO2']):.2e}  N2O5={np.nanmax(gas['N2O5']):.2e}")

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, fixed_cation_conc=0.0,
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
        verbose=True, dt_poisson=None,
    )
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    print(f"\n=== RESULT ===")
    print(f"  success      : {result['success']}")
    print(f"  wall_time    : {wall:.1f}s ({wall/60:.2f}min)")
    print(f"  nfev         : {result.get('nfev', 0)}")
    print(f"  njev         : {result.get('njev', 0)}")
    print(f"  pH_avg       : {result['pH_avg']:.3f}")
    print(f"  NO3- (uM)    : {avg.get('NO3-', 0)*1e6:.2f}")
    print(f"  NO2- (uM)    : {avg.get('NO2-', 0)*1e6:.3f}")
    print(f"  H2O2 (uM)    : {avg.get('H2O2', 0)*1e6:.2f}")


if __name__ == '__main__':
    main()
