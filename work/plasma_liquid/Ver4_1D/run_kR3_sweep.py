#!/usr/bin/env python3
"""k_R3 sweep runner — injects R_TPA3 (hTPA + OH) at arbitrary k values.

Reuses run_tpa_alkaline pipeline but appends an R_TPA3 reaction to the
chemistry after init, then re-precomputes numba arrays. Caches to
`{voltage}_tpa2000uM_humidfitting_kR3-{k:.0e}.npz`.

k_R3 = 0 uses the existing `{voltage}_tpa2000uM_humidfitting.npz` cache
(reactions_tpa.yaml has R_TPA3 commented out — Tampieri practice).
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D
from run_tpa_alkaline import (
    load_gas_data, build_tpa_initial_condition,
    DZ_MIN, STRETCH, REF_BC, REF_ALPHA, REF_DELTA_GAS, DT_SNAPSHOT,
    CACHE_DIR, EXPERIMENT,
)


def _cache_path(voltage: str, k_R3: float) -> Path:
    if k_R3 == 0:
        return CACHE_DIR / f"{voltage}_tpa2000uM_humidfitting.npz"
    return CACHE_DIR / f"{voltage}_tpa2000uM_humidfitting_kR3-{k_R3:.0e}.npz"


def run_one(voltage: str, k_R3: float, rerun: bool = False,
            condition: str = 'Humid_fitting'):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(voltage, k_R3)

    if cache_file.exists() and not rerun:
        d = dict(np.load(cache_file, allow_pickle=True))
        print(f"[{voltage} kR3={k_R3:.1e}] loading from cache "
              f"(hTPA={float(d['hTPA_uM']):.2f} µM)")
        return d

    print(f"\n{'='*70}")
    print(f"Run: voltage={voltage}, k_R3={k_R3:.3e} M⁻¹s⁻¹")
    print(f"{'='*70}")

    times, gas_conc, hono_gas, hono2_gas, h2o2_gas = load_gas_data(
        voltage, condition=condition,
    )

    tpa_conc = 2e-3
    na_conc = 10e-3 + 2.0 * tpa_conc
    chem = AqueousChemistry1D(saline_mode=False, tpa_mode=True)

    if k_R3 > 0:
        chem.reactions.append({
            'type': 'irr',
            'reactants': {'hTPA': 1, 'OH': 1},
            'products': {},
            'k': float(k_R3),
            'label': f'R_TPA3: hTPA + OH → decomposition (k={k_R3:.2e})',
        })
        chem._precompute_reaction_data()
        chem._precompute_numba_arrays()
        print(f"  Injected R_TPA3 with k={k_R3:.3e} "
              f"(reactions total = {len(chem.reactions)})")

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=DZ_MIN,
        stretch_ratio=STRETCH,
        mass_transfer_eta=1.0,
        saline_mode=False,
        fixed_cation_conc=na_conc,
        bc_type=REF_BC,
        alpha_b=REF_ALPHA,
        delta_gas=REF_DELTA_GAS,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=hono_gas, hono2_gas=hono2_gas, h2o2_gas=h2o2_gas,
    )

    y0 = build_tpa_initial_condition(solver, tpa_conc=tpa_conc, initial_pH=12.0)
    t_end = float(times[-1])
    t_eval = np.arange(DT_SNAPSHOT, t_end + 0.1, DT_SNAPSHOT)
    t_eval = t_eval[t_eval <= t_end + 0.1]

    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval, y0=y0,
        verbose=False, dt_poisson=None,
    )
    wall = time_mod.time() - t0

    avg = result['spatial_avg']
    htpa_uM = avg.get('hTPA', 0.0) * 1e6
    tpa_uM = avg.get('TPA', 0.0) * 1e6
    pH = result['pH_avg']

    exp_htpa = EXPERIMENT.get(voltage, None)
    err = (htpa_uM - exp_htpa) / exp_htpa * 100 if exp_htpa else None
    print(f"  Wall {wall:.1f}s | pH={pH:.3f} | [hTPA]={htpa_uM:.2f} µM "
          f"(exp {exp_htpa:.2f}, err {err:+.1f}%) | [TPA]={tpa_uM*1e3:.1f} µM")

    N_z, N_s = solver.N_z, solver.N_s
    snap_t = np.array([0.0] + [float(tv) for tv in result['t_eval']])
    snap_y = [y0.reshape(N_z, N_s).copy()]
    for yv in result['y_eval']:
        snap_y.append(np.array(yv).reshape(N_z, N_s))
    snap_y_arr = np.array(snap_y)

    data = {
        'voltage': voltage, 'tpa_conc': np.float64(tpa_conc),
        'k_R3': np.float64(k_R3),
        'pH_avg': np.float64(pH),
        'TPA_uM': np.float64(tpa_uM), 'hTPA_uM': np.float64(htpa_uM),
        'wall_s': np.float64(wall),
        'success': np.bool_(result['success']),
        'z_centers': solver.z_centers, 'dz_cells': solver.dz_cells,
        'L': np.float64(solver.L),
        'N_z': np.int64(N_z), 'N_s': np.int64(N_s),
        'snap_t': snap_t, 'snap_y': snap_y_arr,
        'species_idx_keys': np.array(list(solver.species_idx.keys())),
        'species_idx_vals': np.array(list(solver.species_idx.values())),
    }
    np.savez_compressed(cache_file, **data)
    print(f"  cached → {cache_file.name}")
    return data


def main():
    VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
    K_R3_LIST = [0.0, 1.0e9, 6.3e9]

    rows = []
    for k in K_R3_LIST:
        for v in VOLTAGES:
            d = run_one(v, k, rerun=False)
            rows.append({
                'k_R3': k, 'voltage': v,
                'hTPA_uM': float(d['hTPA_uM']),
                'pH': float(d['pH_avg']),
            })

    print("\n" + "="*70)
    print("k_R3 SWEEP SUMMARY (Humid_fitting, 600 s)")
    print("="*70)
    print(f"{'k_R3 [M⁻¹s⁻¹]':<18} {'V':<8} {'hTPA [µM]':>12} {'pH':>8} {'exp':>8}")
    for r in rows:
        exp = EXPERIMENT[r['voltage']]
        print(f"{r['k_R3']:<18.3e} {r['voltage']:<8} "
              f"{r['hTPA_uM']:>12.2f} {r['pH']:>8.3f} {exp:>8.2f}")


if __name__ == '__main__':
    main()
