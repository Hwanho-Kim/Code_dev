#!/usr/bin/env python3
"""
1D Saline Simulation: Run with optimal 1D params + saline chemistry.

Uses DI-water-fitted params (optimal_params_1d.yaml) to test whether the
1D spatial model naturally resolves the O3/OH radical starvation that
kills Cl chemistry in the 0D model.

Experimental targets (3.2 kVpp, 0.9% NaCl, 6 min):
    pH=3.60, NO2-~0 uM, NO3-~102 uM, H2O2~5 uM
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_1d import (
    PHYSICAL, MASS_TRANSFER, GRID,
    GAS_TO_AQUEOUS_MAP, ACID_BASE_PAIRS,
    SALINE_ACID_BASE_PAIRS,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D


SALINE_TARGETS = {
    'pH': 3.60,
    'NO2-': 0.0,
    'NO3-': 102.0,
    'H2O2': 5.0,
}

DEFAULT_CSV = Path(__file__).parent.parent / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'


def load_gas_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float) * 2.0  # OAS data: 2-second intervals

    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))

    if 'N2O4' not in df.columns or np.all(gas_conc['N2O4'] == 0):
        from config_1d import N2O4_EQ, PHYSICAL as P
        no2 = gas_conc['NO2']
        T = 298.15
        import math
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298) +
            (N2O4_EQ.DELTA_H / P.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        factor = P.KB_T_OVER_P * T
        gas_conc['N2O4'] = Kp * factor * (no2 ** 2)

    return times, gas_conc


def run_saline_1d():
    params_path = Path(__file__).parent / 'optimal_params_1d.yaml'
    if not params_path.exists():
        print(f"ERROR: {params_path} not found")
        sys.exit(1)

    with open(params_path) as f:
        params = yaml.safe_load(f)

    gas = params.get('gas_phase_unmeasured', {})
    mt = params.get('mass_transfer', {})
    grid = params.get('grid', {})

    hono_gas = gas.get('HONO', 6.535e14)
    hono2_gas = gas.get('HONO2', 9.949e14)
    h2o2_gas = gas.get('H2O2', 1.128e15)
    # η=1.0: no fitting correction — use physical two-film k_L as-is.
    # Fine geometric grid (dz_min=5μm) resolves liquid boundary layer,
    # so δ_liq in D_adj acts only as gas-side coupling (not sub-grid model).
    eta = 1.0

    print("=" * 70)
    print("1D SALINE Simulation — eta=1.0 (no fitting), geometric grid")
    print("=" * 70)
    print(f"  HONO={hono_gas:.3e}, HONO2={hono2_gas:.3e}, H2O2={h2o2_gas:.3e} cm^-3")
    print(f"  eta={eta:.4f} (physical, no correction)")
    print(f"  Grid: geometric, dz_min=5 um, ratio=1.02")
    print(f"  Solution: 0.9% NaCl (Cl- = 0.154 M, Na+ = 0.154 M)")
    print()

    csv_path = DEFAULT_CSV
    times, gas_conc = load_gas_data(csv_path)
    print(f"  CSV: {len(times)} timesteps from {csv_path.name}")

    chem = AqueousChemistry1D(saline_mode=True)
    print(f"  Reactions: {len(chem.reactions)} (base + saline)")
    print(f"  Species: {chem.n_species}")

    from config_1d import ODE_CONFIG, DEFAULTS
    # Raise trace floor so scipy num_jac perturbations stay finite.
    # 1e-15 M is still 1e9× below μM products of interest.
    DEFAULTS.trace_concentration = 1e-15

    # --- Monkey-patch scipy num_jac to cap factor growth ---
    # Prevents overflow when Cl- (0.154M) coexists with trace (1e-15) species.
    import scipy.integrate._ivp.common as _ivp_common
    _orig_sparse_num_jac = _ivp_common._sparse_num_jac

    def _patched_sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups):
        import numpy as _np
        # Cap factor to prevent overflow (original has no upper bound)
        MAX_FACTOR = 1e38
        _np.clip(factor, None, MAX_FACTOR, out=factor)
        result = _orig_sparse_num_jac(fun, t, y, f, h, factor, y_scale, structure, groups)
        # Also cap the returned factor
        _np.clip(result[1], None, MAX_FACTOR, out=result[1])
        return result

    _ivp_common._sparse_num_jac = _patched_sparse_num_jac
    print()

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,          # 5 μm = δ_liq/20 (CFD best practice)
        stretch_ratio=1.02,   # ~72 cells in BL, ~187 total
        mass_transfer_eta=eta,
        saline_mode=True,
        fixed_cation_conc=0.154,
        bc_type='film_alpha',     # Heirman 2025 Eq.7
        alpha_b=0.03,             # from DIW calibration
    )

    solver.set_gas_data(
        times=times,
        gas_conc_molecules=gas_conc,
        hono_gas=hono_gas,
        hono2_gas=hono2_gas,
        h2o2_gas=h2o2_gas,
    )

    t_end = float(times[-1])  # actual end time in seconds
    t0 = time.time()

    result = solver.solve(
        t_span=(0, t_end),
        t_eval=np.array([0, t_end / 4, t_end / 2, 3 * t_end / 4, t_end]),
        verbose=True,
        dt_poisson=60.0,    # macro-step for electroneutrality enforcement
    )

    wall = time.time() - t0

    avg = result['spatial_avg']
    no2_uM = avg.get('NO2-', 0) * 1e6
    no3_uM = avg.get('NO3-', 0) * 1e6
    h2o2_uM = avg.get('H2O2', 0) * 1e6
    pH = result['pH_avg']
    cl_M = avg.get('Cl-', 0.154)
    oh_M = avg.get('OH', 0)
    o3_M = avg.get('O3', 0)
    hclo_M = avg.get('HClO', 0)

    print()
    print("=" * 70)
    print("SALINE 1D RESULTS vs EXPERIMENT")
    print("=" * 70)
    print(f"  {'':15s}  {'Sim':>10s}  {'Exp':>8s}  {'Error':>8s}")
    print(f"  {'-'*50}")
    for label, sim_val, exp_val, unit in [
        ('pH', pH, SALINE_TARGETS['pH'], ''),
        ('NO2- (uM)', no2_uM, SALINE_TARGETS['NO2-'], 'uM'),
        ('NO3- (uM)', no3_uM, SALINE_TARGETS['NO3-'], 'uM'),
        ('H2O2 (uM)', h2o2_uM, SALINE_TARGETS['H2O2'], 'uM'),
    ]:
        if exp_val != 0:
            err_pct = abs(sim_val - exp_val) / exp_val * 100
            print(f"  {label:15s}  {sim_val:10.3f}  {exp_val:8.2f}  {err_pct:7.1f}%")
        else:
            print(f"  {label:15s}  {sim_val:10.3f}  {exp_val:8.2f}  {'--':>8s}")

    print()
    print(f"  Cl- final: {cl_M:.6f} M (initial: 0.154, change: {(cl_M - 0.154)*1e6:+.1f} uM)")
    print(f"  OH:  {oh_M:.3e} M")
    print(f"  O3:  {o3_M:.3e} M")
    print(f"  HClO: {hclo_M:.3e} M")
    print(f"  Wall time: {wall:.1f}s ({wall/60:.1f}min)")
    print(f"  Success: {result['success']}")

    sfc = result.get('surface', {})
    if sfc:
        print()
        print("  --- Surface (z=0) ---")
        print(f"  pH_surface: {result.get('pH_surface', 0):.3f}")
        print(f"  OH(0):  {sfc.get('OH', 0):.3e} M")
        print(f"  O3(0):  {sfc.get('O3', 0):.3e} M")
        print(f"  HClO(0): {sfc.get('HClO', 0):.3e} M")
        print(f"  Cl-(0): {sfc.get('Cl-', 0):.6f} M")


if __name__ == '__main__':
    run_saline_1d()
