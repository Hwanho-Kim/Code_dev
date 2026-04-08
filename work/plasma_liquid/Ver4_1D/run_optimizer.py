#!/usr/bin/env python3
"""
Joint DI + Saline optimizer: parallel DE, ratio constraints, early termination.
Module-level initialization for multiprocessing compatibility.
"""
import math, sys, time, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent))
from config_1d import (PHYSICAL, HENRY_CONSTANTS, N2O4_EQ,
                        MASS_TRANSFER, GRID, POISSON, DEFAULTS)
object.__setattr__(POISSON, 'enabled', False)
DEFAULTS.trace_concentration = 1e-15
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

# ── Module-level data (shared across forked workers) ──
_CSV = (Path(__file__).parent.parent
        / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv')
_df = pd.read_csv(_CSV)
TIMES = np.arange(len(_df), dtype=float)
GAS_CONC = {}
for _c in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
    if _c in _df.columns:
        GAS_CONC[_c] = np.maximum(_df[_c].values.astype(float), 0.0)
    else:
        GAS_CONC[_c] = np.zeros(len(_df))
if np.all(GAS_CONC.get('N2O4', np.zeros(1)) == 0):
    T = 298.15
    Kp = math.exp(math.log(N2O4_EQ.KP_298) +
                   (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
    GAS_CONC['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * GAS_CONC['NO2']**2

NO2_MAX = float(np.max(GAS_CONC['NO2']))
N2O5_MAX = float(np.max(GAS_CONC['N2O5']))
O3_MAX = float(np.max(GAS_CONC['O3']))
T_END = float(len(TIMES) - 1)

TARGETS_DI = {'pH': 3.61, 'NO2-': 3.0, 'NO3-': 63.0, 'H2O2': 11.0}
TARGETS_SAL = {'pH': 3.60, 'NO2-': 0.222, 'NO3-': 102.0, 'H2O2': 5.0}
WEIGHTS = {'pH': 1/0.015**2, 'NO2-': 1/0.30**2,
           'NO3-': 1/0.10**2, 'H2O2': 1/0.15**2}


def _cost(res, tgt):
    c = 0.0
    for k in tgt:
        s, e = res[k], tgt[k]
        if k == 'pH':
            r = (10**(-s) - 10**(-e)) / 10**(-e)
        elif e < 0.5:
            r = s / 1.0
        else:
            r = (s - e) / e
        c += WEIGHTS[k] * r * r
    return c


def _run(hono, hono2, h2o2, dg_m, saline=False):
    orig = MASS_TRANSFER.delta_x_gas
    MASS_TRANSFER.delta_x_gas = dg_m
    try:
        chem = AqueousChemistry1D(saline_mode=saline)
        solver = PDESolver1D(
            chemistry=chem, N_z=10, mass_transfer_eta=1.0,
            saline_mode=saline,
            fixed_cation_conc=0.154 if saline else 0.0,
        )
        solver.set_gas_data(TIMES, GAS_CONC,
                            hono_gas=hono, hono2_gas=hono2, h2o2_gas=h2o2)
        result = solver.solve(
            t_span=(0, T_END), t_eval=np.array([0, T_END]),
            verbose=False, dt_poisson=10.0,
        )
    except Exception:
        return None
    finally:
        MASS_TRANSFER.delta_x_gas = orig
    if not result['success']:
        return None
    avg = result['spatial_avg']
    return {'pH': result['pH_avg'], 'NO2-': avg.get('NO2-', 0)*1e6,
            'NO3-': avg.get('NO3-', 0)*1e6, 'H2O2': avg.get('H2O2', 0)*1e6}


_BEST = {'cost': 1e9, 'n': 0}


def objective(x):
    r_hono, r_hono2, r_h2o2, log_dg = x
    hono = r_hono * NO2_MAX
    hono2 = r_hono2 * N2O5_MAX
    h2o2 = r_h2o2 * O3_MAX
    dg_m = (10.0 ** log_dg) * 1e-3

    # DI water first (fast ~30s)
    rd = _run(hono, hono2, h2o2, dg_m, saline=False)
    if rd is None:
        return 1e6
    cd = _cost(rd, TARGETS_DI)

    # Early termination: skip saline if DI is terrible
    if cd > 5000:
        return cd * 2

    # Saline (~10min)
    rs = _run(hono, hono2, h2o2, dg_m, saline=True)
    if rs is None:
        return cd + 1e6
    cs = _cost(rs, TARGETS_SAL)
    cost = cd + cs

    _BEST['n'] += 1
    if cost < _BEST['cost']:
        _BEST['cost'] = cost
        print(f"  [{_BEST['n']:3d}] cost={cost:.1f} (DI={cd:.1f} SAL={cs:.1f})  "
              f"DI:pH={rd['pH']:.2f} NO3={rd['NO3-']:.0f} H2O2={rd['H2O2']:.1f}  "
              f"SAL:pH={rs['pH']:.2f} NO3={rs['NO3-']:.0f} H2O2={rs['H2O2']:.1f}  "
              f"δg={10**log_dg:.0f}mm r=[{r_hono:.3f},{r_hono2:.3f},{r_h2o2:.4f}]",
              flush=True)
    return cost


if __name__ == '__main__':
    print("=" * 70)
    print("Joint DI+Saline Optimizer (parallel DE, ratio constraints)")
    print("=" * 70)
    print(f"  NO₂={NO2_MAX:.2e}, N₂O₅={N2O5_MAX:.2e}, O₃={O3_MAX:.2e} cm⁻³")
    print(f"  Cores: {os.cpu_count()}, Strang dt=10s, N_z=10")

    bounds = [
        (0.03, 0.30),   # HONO/NO₂
        (0.01, 1.0),    # HONO₂/N₂O₅
        (0.005, 0.30),  # H₂O₂/O₃
        (1.5, 3.5),     # log₁₀(δ_gas mm): 30–3000 mm
    ]
    print(f"  Bounds: {bounds}")
    print()

    t0 = time.time()
    result = differential_evolution(
        objective, bounds,
        maxiter=25, popsize=10,
        tol=0.01, mutation=(0.5, 1.5), recombination=0.8,
        workers=-1, updating='deferred',
        polish=False, disp=True, seed=42,
    )
    dt1 = time.time() - t0

    bx = result.x
    hono_f = bx[0] * NO2_MAX
    hono2_f = bx[1] * N2O5_MAX
    h2o2_f = bx[2] * O3_MAX
    dg_f = 10.0 ** bx[3]

    print(f"\n{'='*70}\nRESULTS\n{'='*70}")
    print(f"  HONO  = {hono_f:.3e} (ratio={bx[0]:.3f})")
    print(f"  HONO₂ = {hono2_f:.3e} (ratio={bx[1]:.3f})")
    print(f"  H₂O₂  = {h2o2_f:.3e} (ratio={bx[2]:.4f})")
    print(f"  δ_gas = {dg_f:.0f} mm")
    print(f"  Cost  = {result.fun:.2f}, Time = {dt1/60:.1f}min")

    rd = _run(hono_f, hono2_f, h2o2_f, dg_f*1e-3, saline=False)
    rs = _run(hono_f, hono2_f, h2o2_f, dg_f*1e-3, saline=True)
    for lbl, r, t in [("DI", rd, TARGETS_DI), ("SAL", rs, TARGETS_SAL)]:
        print(f"\n  {lbl}:")
        if r is None:
            print("    FAILED"); continue
        for k in ['pH', 'NO2-', 'NO3-', 'H2O2']:
            print(f"    {k:>6s}: {r[k]:8.3f}  (exp={t[k]:.2f}, "
                  f"err={abs(r[k]-t[k])/max(t[k],0.1)*100:.1f}%)")

    import yaml
    yaml.dump({
        'gas': {'HONO': float(hono_f), 'HONO2': float(hono2_f),
                'H2O2': float(h2o2_f)},
        'ratios': {'HONO_NO2': float(bx[0]), 'HONO2_N2O5': float(bx[1]),
                   'H2O2_O3': float(bx[2])},
        'mass_transfer': {'delta_x_gas': float(dg_f*1e-3)},
    }, open(Path(__file__).parent / 'optimal_params_1d.yaml', 'w'),
        default_flow_style=False)
    print(f"\n  Saved optimal_params_1d.yaml")
