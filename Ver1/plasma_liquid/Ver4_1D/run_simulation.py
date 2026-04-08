#!/usr/bin/env python3
"""
1D Plasma-Liquid Simulation Runner.

Usage:
    python run_simulation.py                     # Default Ver3 optimal params
    python run_simulation.py --hono 6.2e14 --hono2 6.8e14 --h2o2 1.1e15
    python run_simulation.py --nz 20             # Coarse grid (fast)

Loads gas-phase CSV data and runs the 1D diffusion-reaction simulation.
Reports final concentrations compared to experimental targets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_1d import (
    PHYSICAL, HENRY_CONSTANTS, MASS_TRANSFER, GRID,
    GAS_TO_AQUEOUS_MAP, ACID_BASE_PAIRS
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D


# =============================================================================
# Experimental Targets (3.2 kVpp, DI water, 6 min)
# =============================================================================
TARGETS = {
    'pH': 3.61,
    'NO2-': 3.0,    # μM
    'NO3-': 63.0,   # μM
    'H2O2': 11.0,   # μM
}

# Default CSV path
DEFAULT_CSV = Path(__file__).parent.parent / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'

# Default gas-phase parameters (from Ver3 optimal fit)
DEFAULT_HONO = 6.219e14    # molecules/cm³
DEFAULT_HONO2 = 6.761e14   # molecules/cm³
DEFAULT_H2O2 = 1.061e15    # molecules/cm³


def load_gas_data(csv_path: Path):
    """Load gas-phase CSV data."""
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float)

    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))

    # Estimate N2O4 from NO2 if not in CSV or all zeros
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


def run_single_simulation(
    csv_path: Path,
    hono_gas: float,
    hono2_gas: float,
    h2o2_gas: float,
    N_z: int = 100,
    dz_min: float = None,
    stretch_ratio: float = None,
    verbose: bool = True,
) -> dict:
    """
    Run a single 1D simulation.

    Parameters
    ----------
    csv_path : Path
        Path to gas-phase CSV.
    hono_gas, hono2_gas, h2o2_gas : float
        Gas-phase concentrations [molecules/cm³].
    N_z : int
        Number of grid points.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with simulation results.
    """
    # Load gas data
    times, gas_conc = load_gas_data(csv_path)

    if verbose:
        print("=" * 70)
        print("1D Plasma-Liquid Diffusion-Reaction Simulation")
        print("=" * 70)
        print(f"CSV: {csv_path}")
        print(f"Time steps: {len(times)}")
        print(f"Gas species: HONO={hono_gas:.3e}, HONO₂={hono2_gas:.3e}, "
              f"H₂O₂={h2o2_gas:.3e} cm⁻³")
        print()

    # Initialize chemistry
    chem = AqueousChemistry1D()

    if verbose:
        print(f"Loaded {len(chem.reactions)} reactions, "
              f"{chem.n_species} aqueous species")

    # Initialize PDE solver
    if dz_min is not None:
        solver = PDESolver1D(chemistry=chem, dz_min=dz_min, stretch_ratio=stretch_ratio)
    else:
        solver = PDESolver1D(chemistry=chem, N_z=N_z)

    # Set gas-phase boundary conditions
    solver.set_gas_data(
        times=times,
        gas_conc_molecules=gas_conc,
        hono_gas=hono_gas,
        hono2_gas=hono2_gas,
        h2o2_gas=h2o2_gas,
    )

    # Solve
    t_end = float(len(times) - 1)
    result = solver.solve(
        t_span=(0, t_end),
        t_eval=np.array([0, t_end / 4, t_end / 2, 3 * t_end / 4, t_end]),
        verbose=verbose,
    )

    return result


def print_results(result: dict):
    """Print simulation results vs experimental targets."""
    print()
    print("=" * 70)
    print("RESULTS (Volume-Averaged Final Concentrations)")
    print("=" * 70)

    avg = result['spatial_avg']

    # NO2- (from HONO_total speciation)
    no2_uM = avg.get('NO2-', 0) * 1e6
    # NO3- (from HONO2_total speciation)
    no3_uM = avg.get('NO3-', 0) * 1e6
    # H2O2 (from H2O2_total speciation)
    h2o2_uM = avg.get('H2O2', 0) * 1e6
    # pH
    pH = result['pH_avg']

    print(f"  {'Species':>8s}  {'Sim':>10s}  {'Exp':>8s}  {'Error':>8s}")
    print(f"  {'-'*8:>8s}  {'-'*10:>10s}  {'-'*8:>8s}  {'-'*8:>8s}")

    for label, sim_val, exp_val, unit in [
        ('pH', pH, TARGETS['pH'], ''),
        ('NO₂⁻', no2_uM, TARGETS['NO2-'], 'μM'),
        ('NO₃⁻', no3_uM, TARGETS['NO3-'], 'μM'),
        ('H₂O₂', h2o2_uM, TARGETS['H2O2'], 'μM'),
    ]:
        if exp_val != 0:
            err_pct = abs(sim_val - exp_val) / exp_val * 100
        else:
            err_pct = 0
        print(f"  {label:>8s}  {sim_val:>10.3f}  {exp_val:>8.2f}  {err_pct:>7.1f}%  {unit}")

    print()
    print(f"  Surface pH: {result['pH_surface']:.3f}")
    print(f"  Wall time: {result['wall_time']:.1f}s")
    print(f"  Solver success: {result['success']}")


def main():
    parser = argparse.ArgumentParser(description='1D Plasma-Liquid Simulation')
    parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV),
                        help='Path to gas-phase CSV')
    parser.add_argument('--hono', type=float, default=DEFAULT_HONO,
                        help='HONO gas concentration [molecules/cm³]')
    parser.add_argument('--hono2', type=float, default=DEFAULT_HONO2,
                        help='HONO₂ gas concentration [molecules/cm³]')
    parser.add_argument('--h2o2', type=float, default=DEFAULT_H2O2,
                        help='H₂O₂ gas concentration [molecules/cm³]')
    parser.add_argument('--nz', type=int, default=GRID.N_z,
                        help='Number of grid points (uniform mode)')
    parser.add_argument('--geometric', action='store_true',
                        help='Use geometric grid (resolves Debye layer)')
    parser.add_argument('--dz-min', type=float, default=GRID.dz_min,
                        help='Min cell width at interface [m]')
    parser.add_argument('--stretch', type=float, default=GRID.stretch_ratio,
                        help='Geometric stretching ratio')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if args.geometric:
        result = run_single_simulation(
            csv_path=Path(args.csv),
            hono_gas=args.hono,
            hono2_gas=args.hono2,
            h2o2_gas=args.h2o2,
            dz_min=args.dz_min,
            stretch_ratio=args.stretch,
            verbose=not args.quiet,
        )
    else:
        result = run_single_simulation(
            csv_path=Path(args.csv),
            hono_gas=args.hono,
            hono2_gas=args.hono2,
            h2o2_gas=args.h2o2,
            N_z=args.nz,
            verbose=not args.quiet,
        )

    print_results(result)


if __name__ == '__main__':
    main()
