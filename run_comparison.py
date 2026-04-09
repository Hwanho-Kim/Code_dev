#!/usr/bin/env python3
"""
Run DIW reference case on Ver1 and Ver2 with identical conditions.
Time axis corrected: times *= 2.0 (OAS 2-second intervals).
"""
import sys
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_gas_data(csv_path: Path):
    """Load gas-phase CSV with ×2 time correction."""
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float) * 2.0  # OAS 2-sec intervals

    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))

    # Estimate N2O4 from NO2
    if 'N2O4' not in df.columns or np.all(gas_conc['N2O4'] == 0):
        no2 = gas_conc['NO2']
        T = 298.15
        KP_298 = 6.75
        DELTA_H = -57120.0
        REF_TEMP = 298.15
        R = 8.314462618
        KB_T_OVER_P = 1.3625946e-22
        Kp = math.exp(math.log(KP_298) + (DELTA_H / R) * (1/REF_TEMP - 1/T))
        factor = KB_T_OVER_P * T
        gas_conc['N2O4'] = Kp * factor * (no2 ** 2)

    return times, gas_conc


def run_version(ver_name: str, ver_root: Path):
    """Run a single version and return results dict."""
    ver4_dir = ver_root / 'plasma_liquid' / 'Ver4_1D'
    csv_path = ver_root / 'plasma_liquid' / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'

    # Add ver4_dir to path (front)
    sys.path.insert(0, str(ver4_dir))

    # Force reimport of modules for this version
    for mod_name in list(sys.modules.keys()):
        if mod_name in ('config_1d', 'chemistry_1d', 'pde_solver'):
            del sys.modules[mod_name]

    from config_1d import GRID
    from chemistry_1d import AqueousChemistry1D
    from pde_solver import PDESolver1D

    # Load gas data with ×2 correction
    times, gas_conc = load_gas_data(csv_path)

    print(f"\n{'='*70}")
    print(f"  {ver_name}: DIW Reference Case (Film+α_b=0.03, 측정종만)")
    print(f"  t_end={times[-1]:.0f}s, {len(times)} time points")
    print(f"{'='*70}")

    # Same conditions as run_alpha_analysis α_b=0.03
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.02,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='film_alpha',
        alpha_b=0.03,
    )
    solver.set_gas_data(
        times=times,
        gas_conc_molecules=gas_conc,
        hono_gas=0,
        hono2_gas=0,
        h2o2_gas=0,
    )

    t_end = float(times[-1])
    y0 = solver.build_initial_condition(initial_pH=7.0)

    t0 = time.time()
    result = solver.solve(
        t_span=(0, t_end),
        t_eval=np.array([0, t_end/4, t_end/2, 3*t_end/4, t_end]),
        y0=y0,
        verbose=True,
        dt_poisson=None,  # single BDF
    )
    wall = time.time() - t0

    # Clean up path
    sys.path.remove(str(ver4_dir))

    # Extract results
    avg = result['spatial_avg']
    no2_uM = avg.get('NO2-', 0) * 1e6
    no3_uM = avg.get('NO3-', 0) * 1e6
    h2o2_uM = avg.get('H2O2', 0) * 1e6
    o3_nM = avg.get('O3', 0) * 1e9
    oh_pM = avg.get('OH', 0) * 1e12
    pH = result['pH_avg']

    return {
        'ver': ver_name,
        'pH': pH,
        'NO2-': no2_uM,
        'NO3-': no3_uM,
        'H2O2': h2o2_uM,
        'O3_nM': o3_nM,
        'OH_pM': oh_pM,
        'wall_s': wall,
        'success': result['success'],
        'nfev': result.get('nfev', 0),
    }


def main():
    work_root = Path("/mnt/d/HH Kim/Opencode_backup/work")
    ver1_root = work_root / 'Ver1'
    ver2_root = work_root / 'Ver2'

    # Experimental targets
    targets = {'pH': 3.61, 'NO2-': 3.0, 'NO3-': 63.0, 'H2O2': 11.0}

    results = []
    for name, root in [('Ver1', ver1_root), ('Ver2', ver2_root)]:
        r = run_version(name, root)
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY (DIW, Film+α_b=0.03, 측정종만, t=720s)")
    print(f"{'='*70}")
    print(f"  {'':>10s}  {'Exp':>8s}  {'Ver1':>10s}  {'Ver2':>10s}  {'Diff':>10s}")
    print(f"  {'-'*10:>10s}  {'-'*8:>8s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}")

    r1, r2 = results[0], results[1]
    for label, key, unit, exp in [
        ('pH', 'pH', '', targets['pH']),
        ('NO₂⁻', 'NO2-', 'µM', targets['NO2-']),
        ('NO₃⁻', 'NO3-', 'µM', targets['NO3-']),
        ('H₂O₂', 'H2O2', 'µM', targets['H2O2']),
        ('O₃', 'O3_nM', 'nM', None),
        ('OH', 'OH_pM', 'pM', None),
    ]:
        v1 = r1[key]
        v2 = r2[key]
        diff = v1 - v2
        exp_str = f"{exp:.2f}" if exp is not None else "—"
        print(f"  {label:>10s}  {exp_str:>8s}  {v1:10.3f}  {v2:10.3f}  {diff:+10.4f}  {unit}")

    print(f"\n  {'Wall time':>10s}  {'':>8s}  {r1['wall_s']:10.1f}s {r2['wall_s']:10.1f}s")
    print(f"  {'nfev':>10s}  {'':>8s}  {r1['nfev']:10d}  {r2['nfev']:10d}")
    print(f"  {'Success':>10s}  {'':>8s}  {str(r1['success']):>10s}  {str(r2['success']):>10s}")


if __name__ == '__main__':
    main()
