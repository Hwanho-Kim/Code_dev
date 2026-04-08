#!/usr/bin/env python3
"""
BC Comparison: Run DI water simulation with different gas-liquid interface BCs.

Compares 6 boundary condition models:
  C: Two-film (Lee 2023, current default)
  A: Dirichlet (C(0) ≈ C_eq via stiff relaxation)
  D: Film theory (Heirman 2025 Eq.6, α_b=1)
  E1: Film + α_b=0.1
  E2: Film + α_b=0.05
  E3: Film + α_b=0.01

Reference: Heirman 2025, J. Phys. D: Appl. Phys. 58 085206
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
    HENRY_CONSTANTS, GAS_DIFFUSIVITY, LIQUID_DIFFUSIVITY,
    D_GAS_DEFAULT, D_LIQ_DEFAULT,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D, compute_k_mt

DIW_TARGETS = {
    'pH': 3.61,
    'NO2-': 3.0,
    'NO3-': 63.0,
    'H2O2': 11.0,
}

DEFAULT_CSV = Path(__file__).parent.parent / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'

BC_CASES = [
    {'label': 'C: Two-film',       'bc_type': 'two_film',    'alpha_b': 1.0},
    {'label': 'A: Dirichlet',      'bc_type': 'dirichlet',   'alpha_b': 1.0},
    {'label': 'D: Film',           'bc_type': 'film',        'alpha_b': 1.0},
    {'label': 'E: Film+α_b=0.1',  'bc_type': 'film_alpha',  'alpha_b': 0.1},
    {'label': 'E: Film+α_b=0.05', 'bc_type': 'film_alpha',  'alpha_b': 0.05},
    {'label': 'E: Film+α_b=0.01', 'bc_type': 'film_alpha',  'alpha_b': 0.01},
]


FOCUS_SPECIES = ['N2O5', 'O3', 'NO2', 'H2O2', 'NO', 'NO3', 'N2O4', 'HONO', 'HONO2']


def print_kmt_table(gas_conc_final: dict, hono_gas: float, hono2_gas: float,
                    h2o2_gas: float):
    """Print analytical k_mt / C_eq / Max flux comparison across BC types.

    Args:
        gas_conc_final: dict of gas species → final concentration [molecules/cm³]
        hono_gas, hono2_gas, h2o2_gas: unmeasured species [molecules/cm³]
    """
    delta_gas = MASS_TRANSFER.delta_x_gas  # 0.01 m
    delta_liq = MASS_TRANSFER.delta_x_liq  # 0.0001 m
    dz0 = 5e-6  # m (surface cell width)
    conv = 1000.0 / PHYSICAL.AVOGADRO

    # Gas concentrations in mol/L for C_eq calculation
    gas_molar = {}
    for sp in FOCUS_SPECIES:
        if sp == 'HONO':
            gas_molar[sp] = max(hono_gas, 0.0) * conv
        elif sp == 'HONO2':
            gas_molar[sp] = max(hono2_gas, 0.0) * conv
        elif sp == 'H2O2':
            gas_molar[sp] = max(h2o2_gas, 0.0) * conv
        else:
            gas_molar[sp] = max(gas_conc_final.get(sp, 0.0), 0.0) * conv

    lines = []
    lines.append("=" * 100)
    lines.append("ANALYTICAL k_mt COMPARISON (no simulation)")
    lines.append("=" * 100)
    lines.append(f"  delta_gas={delta_gas*1000:.0f}mm, delta_liq={delta_liq*1e6:.0f}µm, "
                 f"dz₀={dz0*1e6:.0f}µm")
    lines.append("")

    # Table 1: k_mt values
    bc_labels = ['Two-film', 'Dirichlet', 'Film', 'Film+0.1', 'Film+0.05', 'Film+0.01']
    bc_params = [
        ('two_film', 1.0), ('dirichlet', 1.0), ('film', 1.0),
        ('film_alpha', 0.1), ('film_alpha', 0.05), ('film_alpha', 0.01),
    ]

    lines.append(f"{'Species':<8s}  {'H':>10s}  {'C_gas(M)':>10s}  {'C_eq(M)':>10s}  "
                 + "  ".join(f"{lb:>10s}" for lb in bc_labels))
    lines.append("─" * (8 + 2 + 10 + 2 + 10 + 2 + 10 + len(bc_labels) * 12))

    # Sub-header: k_mt [m/s]
    lines.append("  k_mt [m/s]:")
    for sp in FOCUS_SPECIES:
        H = HENRY_CONSTANTS.get(sp, 1.0)
        C_gas = gas_molar[sp]
        C_eq = H * C_gas
        vals = []
        for bc_type, alpha_b in bc_params:
            k = compute_k_mt(sp, delta_gas, delta_liq, bc_type=bc_type, alpha_b=alpha_b)
            vals.append(k)
        lines.append(f"  {sp:<8s}  {H:10.4g}  {C_gas:10.3e}  {C_eq:10.3e}  "
                     + "  ".join(f"{v:10.3e}" for v in vals))

    lines.append("")
    lines.append("  Max flux = k_mt × C_eq / dz₀ [M/s]  (C_surface=0 upper bound):")
    for sp in FOCUS_SPECIES:
        H = HENRY_CONSTANTS.get(sp, 1.0)
        C_gas = gas_molar[sp]
        C_eq = H * C_gas
        if C_eq < 1e-30:
            continue
        vals = []
        for bc_type, alpha_b in bc_params:
            k = compute_k_mt(sp, delta_gas, delta_liq, bc_type=bc_type, alpha_b=alpha_b)
            flux = k * C_eq / dz0
            vals.append(flux)
        lines.append(f"  {sp:<8s}  {H:10.4g}  {C_gas:10.3e}  {C_eq:10.3e}  "
                     + "  ".join(f"{v:10.3e}" for v in vals))

    # Ratio relative to Two-film
    lines.append("")
    lines.append("  k_mt ratio vs Two-film:")
    for sp in FOCUS_SPECIES:
        k_ref = compute_k_mt(sp, delta_gas, delta_liq, bc_type='two_film', alpha_b=1.0)
        if k_ref < 1e-30:
            continue
        vals = []
        for bc_type, alpha_b in bc_params:
            k = compute_k_mt(sp, delta_gas, delta_liq, bc_type=bc_type, alpha_b=alpha_b)
            vals.append(k / k_ref)
        lines.append(f"  {sp:<8s}  {'':>10s}  {'':>10s}  {'':>10s}  "
                     + "  ".join(f"{v:10.2f}" for v in vals))

    lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


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


def run_single_case(case: dict, times, gas_conc, params: dict):
    """Run one BC case (DI water). Returns dict with results, solver, y_final."""
    gas = params.get('gas_phase_unmeasured', {})
    hono_gas = gas.get('HONO', 6.219e14)
    hono2_gas = gas.get('HONO2', 6.761e14)
    h2o2_gas = gas.get('H2O2', 1.061e15)

    chem = AqueousChemistry1D(saline_mode=False)

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.02,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type=case['bc_type'],
        alpha_b=case['alpha_b'],
    )

    solver.set_gas_data(
        times=times,
        gas_conc_molecules=gas_conc,
        hono_gas=hono_gas,
        hono2_gas=hono2_gas,
        h2o2_gas=h2o2_gas,
    )

    t_end = float(times[-1])
    t0 = time.time()

    try:
        result = solver.solve(
            t_span=(0, t_end),
            t_eval=np.array([0, t_end / 4, t_end / 2, 3 * t_end / 4, t_end]),
            verbose=True,
            dt_poisson=10.0,
        )
        wall = time.time() - t0

        y_final = result['y_final']
        if y_final.ndim == 1:
            y_final = y_final.reshape(solver.N_z, solver.N_s)

        avg = result['spatial_avg']
        return {
            'pH': result['pH_avg'],
            'NO2-': avg.get('NO2-', 0) * 1e6,
            'NO3-': avg.get('NO3-', 0) * 1e6,
            'H2O2': avg.get('H2O2', 0) * 1e6,
            'wall_s': wall,
            'success': result['success'],
            'solver': solver,
            'y_final': y_final,
        }
    except Exception as e:
        wall = time.time() - t0
        return {
            'pH': float('nan'),
            'NO2-': float('nan'),
            'NO3-': float('nan'),
            'H2O2': float('nan'),
            'wall_s': wall,
            'success': False,
            'error': str(e),
        }


def extract_mt_flux(solver, y_final):
    """Extract instantaneous MT flux for each interface species at final time.

    Returns list of dicts with gas_sp, aq_sp, k_mt, C_eq, C_surface, flux, flux_vol.
    """
    t_idx = solver._n_times - 1
    idx_to_name = {v: k for k, v in solver.species_idx.items()}
    L = solver.L
    dz0 = solver.dz_cells[0]

    mt_data = []
    for aq_idx, k_mt, gas_sp, H_val, Ka in solver._interface_species:
        C_eq = solver._get_C_eq_fast(gas_sp, t_idx)
        C_surface = y_final[0, aq_idx]
        flux = k_mt * (C_eq - C_surface)          # mol/(m²·s)
        flux_vol = flux * dz0 / L                   # M/s (volume-averaged source in dz₀)
        # More useful: flux per surface cell volume = k_mt/dz₀ * (C_eq - C_s) [M/s]
        flux_cell = k_mt / dz0 * (C_eq - C_surface)

        mt_data.append({
            'gas_sp': gas_sp,
            'aq_sp': idx_to_name.get(aq_idx, f'idx{aq_idx}'),
            'k_mt': k_mt,
            'C_eq': C_eq,
            'C_surface': C_surface,
            'flux': flux,               # [mol/(m²·s)]
            'flux_cell': flux_cell,     # [M/s] in surface cell
        })
    return mt_data


def print_mt_comparison(results, t_end):
    """Print MT flux comparison table across all BC cases.

    Focus on N₂O₅, O₃, NO₂, H₂O₂.
    """
    focus = ['N2O5', 'O3', 'NO2', 'H2O2']
    lines = []

    lines.append("")
    lines.append("=" * 100)
    lines.append("MASS TRANSFER FLUX COMPARISON (instantaneous at t_end)")
    lines.append("=" * 100)

    for sp in focus:
        lines.append(f"\n  {sp}:")
        hdr = (f"    {'BC':<22s}  {'k_mt(m/s)':>10s}  {'C_eq(M)':>10s}  "
               f"{'C_surf(M)':>10s}  {'Flux(M/s)':>10s}  {'Accum(µM)':>10s}")
        lines.append(hdr)
        lines.append("    " + "─" * (len(hdr) - 4))

        for r in results:
            if not r['success'] or 'mt_data' not in r:
                lines.append(f"    {r['label']:<22s}  {'FAIL':>10s}")
                continue
            entry = next((d for d in r['mt_data'] if d['gas_sp'] == sp), None)
            if entry is None:
                continue
            # Estimated accumulation: flux_cell * t_end * (dz₀/L) gives vol-avg [M]
            # But more direct: NO₃⁻ final ≈ 2 × N₂O₅ integrated (R98 pathway)
            # Here just show flux_cell * t_end as surface-cell-based estimate [µM]
            accum_est = entry['flux_cell'] * t_end * 1e6  # µM (in surface cell only)
            lines.append(f"    {r['label']:<22s}  {entry['k_mt']:10.3e}  "
                         f"{entry['C_eq']:10.3e}  {entry['C_surface']:10.3e}  "
                         f"{entry['flux_cell']:10.3e}  {accum_est:10.1f}")

    # Cross-check: N₂O₅ flux × 2 (R98: N₂O₅→2 NO₃⁻) × t_end vs actual NO₃⁻
    lines.append("")
    lines.append("  Cross-check: N₂O₅ MT flux × 2 × t_end vs actual NO₃⁻")
    lines.append(f"    {'BC':<22s}  {'N2O5 flux*2*t(µM)':>18s}  {'NO3⁻ actual(µM)':>16s}  {'Ratio':>8s}")
    lines.append("    " + "─" * 70)
    for r in results:
        if not r['success'] or 'mt_data' not in r:
            continue
        n2o5 = next((d for d in r['mt_data'] if d['gas_sp'] == 'N2O5'), None)
        if n2o5 is None:
            continue
        # Volume-averaged accumulation estimate
        solver = r['solver']
        dz0 = solver.dz_cells[0]
        L = solver.L
        # MT goes into surface cell. Volume-avg = flux_cell * dz₀/L * t_end
        n2o5_vol_uM = n2o5['flux_cell'] * (dz0 / L) * t_end * 1e6
        est_no3 = 2.0 * n2o5_vol_uM
        no3_actual = r['NO3-']
        ratio = est_no3 / no3_actual if no3_actual > 0 else float('inf')
        lines.append(f"    {r['label']:<22s}  {est_no3:18.1f}  {no3_actual:16.1f}  {ratio:8.2f}")

    lines.append("")
    text = "\n".join(lines)
    print(text)
    return text


def main():
    params_path = Path(__file__).parent / 'optimal_params_1d.yaml'
    if not params_path.exists():
        print(f"ERROR: {params_path} not found")
        sys.exit(1)

    with open(params_path) as f:
        params = yaml.safe_load(f)

    csv_path = DEFAULT_CSV
    times, gas_conc = load_gas_data(csv_path)
    t_end = float(times[-1])

    # --- Step 1: Analytical k_mt table (no simulation) ---
    gas = params.get('gas_phase_unmeasured', {})
    hono_gas = gas.get('HONO', 6.219e14)
    hono2_gas = gas.get('HONO2', 6.761e14)
    h2o2_gas = gas.get('H2O2', 1.061e15)

    # Use final-timestep gas concentrations for C_eq
    gas_conc_final = {sp: arr[-1] for sp, arr in gas_conc.items()}
    kmt_text = print_kmt_table(gas_conc_final, hono_gas, hono2_gas, h2o2_gas)

    # --- Step 2: Run simulations + extract MT flux ---
    print("=" * 78)
    print("BC COMPARISON — DI Water 1D (3.2 kVpp)")
    print("=" * 78)
    print(f"  CSV: {len(times)} timesteps, t_end={t_end:.0f}s")
    print(f"  Cases: {len(BC_CASES)}")
    print()

    results = []

    for i, case in enumerate(BC_CASES):
        print("-" * 78)
        print(f"[{i+1}/{len(BC_CASES)}] {case['label']}  "
              f"(bc_type='{case['bc_type']}', alpha_b={case['alpha_b']})")
        print("-" * 78)

        res = run_single_case(case, times, gas_conc, params)
        res['label'] = case['label']

        # Extract MT flux if simulation succeeded
        if res['success'] and 'solver' in res:
            res['mt_data'] = extract_mt_flux(res['solver'], res['y_final'])

        results.append(res)

        if res['success']:
            print(f"  → pH={res['pH']:.3f}, NO3⁻={res['NO3-']:.1f}µM, "
                  f"NO2⁻={res['NO2-']:.1f}µM, H2O2={res['H2O2']:.2f}µM, "
                  f"time={res['wall_s']:.0f}s")
        else:
            err = res.get('error', 'unknown')
            print(f"  → FAILED ({err}), time={res['wall_s']:.0f}s")
        print()

    # Print comparison table
    print()
    print("=" * 78)
    print("COMPARISON TABLE")
    print("=" * 78)

    header = f"{'BC':<22s}  {'pH':>6s}  {'NO2⁻(µM)':>9s}  {'NO3⁻(µM)':>10s}  {'H2O2(µM)':>9s}  {'Time':>6s}"
    print(header)
    print("─" * len(header))

    for r in results:
        if r['success']:
            print(f"{r['label']:<22s}  {r['pH']:6.3f}  {r['NO2-']:9.1f}  "
                  f"{r['NO3-']:10.1f}  {r['H2O2']:9.2f}  {r['wall_s']/60:5.1f}m")
        else:
            print(f"{r['label']:<22s}  {'FAIL':>6s}  {'FAIL':>9s}  "
                  f"{'FAIL':>10s}  {'FAIL':>9s}  {r['wall_s']/60:5.1f}m")

    print("─" * len(header))
    print(f"{'실험값':<22s}  {3.61:6.2f}  {3.0:9.1f}  "
          f"{63.0:10.1f}  {11.0:9.2f}  {'—':>6s}")
    print()

    # Print MT flux comparison
    mt_text = print_mt_comparison(results, t_end)

    # Save all output
    out_dir = Path(__file__).parent.parent / 'Figures'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'mt_comparison_output.txt'
    with open(out_path, 'w') as f:
        f.write(kmt_text + "\n\n")
        # Comparison table
        f.write("=" * 78 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * 78 + "\n")
        f.write(header + "\n")
        f.write("─" * len(header) + "\n")
        for r in results:
            if r['success']:
                f.write(f"{r['label']:<22s}  {r['pH']:6.3f}  {r['NO2-']:9.1f}  "
                        f"{r['NO3-']:10.1f}  {r['H2O2']:9.2f}  {r['wall_s']/60:5.1f}m\n")
            else:
                f.write(f"{r['label']:<22s}  {'FAIL':>6s}  {'FAIL':>9s}  "
                        f"{'FAIL':>10s}  {'FAIL':>9s}  {r['wall_s']/60:5.1f}m\n")
        f.write("─" * len(header) + "\n")
        f.write(f"{'실험값':<22s}  {3.61:6.2f}  {3.0:9.1f}  "
                f"{63.0:10.1f}  {11.0:9.2f}  {'—':>6s}\n\n")
        f.write(mt_text + "\n")
    print(f"Output saved to: {out_path}")


if __name__ == '__main__':
    main()
