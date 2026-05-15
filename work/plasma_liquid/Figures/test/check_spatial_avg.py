#!/usr/bin/env python3
"""Quick verification — dump spatial_avg dict from DIW 3.2 kV run."""
from __future__ import annotations
import sys, functools
from pathlib import Path
import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / 'Ver4_1D'))
sys.path.insert(0, str(_root / 'Figures'))

import gen_all_figures as gaf
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

print = functools.partial(print, flush=True)

gaf.DEFAULT_GAS_SHEET = '3.2kV'
gaf.CONDITION_LABEL = 'Humid_fitting'
times, gas_conc = gaf.load_gas_data()

chem = AqueousChemistry1D(saline_mode=False)
solver = PDESolver1D(
    chemistry=chem,
    dz_min=5e-6, stretch_ratio=1.12,
    saline_mode=False, fixed_cation_conc=0.0,
)
solver.set_gas_data(
    times=times, gas_conc_molecules=gas_conc,
    hono_gas=gaf.HONO_GAS, hono2_gas=gaf.HONO2_GAS, h2o2_gas=gaf.H2O2_GAS,
)
t_end = float(times[-1])
t_eval = np.arange(2.0, t_end + 0.1, 2.0)
y0 = solver.build_initial_condition(initial_pH=7.0)

result = solver.solve(t_span=(0, t_end), t_eval=t_eval, y0=y0,
                      verbose=False, dt_poisson=None)

avg = result['spatial_avg']
print(f"\nSUCCESS: pH_avg = {result['pH_avg']:.4f}")
print(f"\nspatial_avg keys ({len(avg)}):")
for k in sorted(avg.keys()):
    v = avg[k]
    if abs(v) > 1e-30:
        print(f"  {k:20s} = {v:.4e}")

# Direct check on key species
print(f"\n--- KEY VALUES ---")
print(f"NO2-          = {avg.get('NO2-', 0)*1e6:.4f} µM")
print(f"NO3-          = {avg.get('NO3-', 0)*1e6:.4f} µM")
print(f"H2O2          = {avg.get('H2O2', 0)*1e6:.4f} µM")
print(f"HONO_total    = {avg.get('HONO_total', 0)*1e6:.4f} µM")
print(f"HONO2_total   = {avg.get('HONO2_total', 0)*1e6:.4f} µM")
print(f"H2O2_total    = {avg.get('H2O2_total', 0)*1e6:.4f} µM")
print(f"HONO          = {avg.get('HONO', 0)*1e6:.4e} µM")
print(f"H+            = {avg.get('H+', 0):.4e} M")
print(f"OH-           = {avg.get('OH-', 0):.4e} M")

# Manual speciation check at pH_avg
import math
H = avg.get('H+', 1e-7)
pKa_HONO = 3.398
pKa_HONO2 = -1.4   # HNO3, very strong acid
pKa_H2O2 = 11.65
Ka_HONO = 10**(-pKa_HONO)
print(f"\n--- Manual speciation at H_avg = {H:.4e} (pH = {-math.log10(H):.3f}) ---")
print(f"NO2- fraction (HONO_total → NO2-): "
      f"{Ka_HONO/(Ka_HONO+H):.4f}")
print(f"Implied NO2- = HONO_total × frac = "
      f"{avg.get('HONO_total', 0)*Ka_HONO/(Ka_HONO+H)*1e6:.4f} µM")
