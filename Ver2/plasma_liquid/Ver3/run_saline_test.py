#!/usr/bin/env python3
"""
Saline Validation Test: Run baseline simulation with saline chemistry
using optimal parameters fitted from DI water.

Same as run_baseline.py --validate, but with:
  - saline_mode=True (101 base + 93 Cl reactions)
  - Initial Cl⁻ = 0.154 M (0.9% NaCl)

Experimental targets (3.2 kVpp, saline, 6 min):
  pH=3.60, NO₂⁻≈0 μM, NO₃⁻≈102 μM, H₂O₂≈5 μM
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    GAS_PHASE_SPECIES, HENRY_CONSTANTS, DEFAULTS,
    ACID_BASE_PAIRS, MASS_TRANSFER
)
from chemistry import CompleteAqueousChemistry
from chemistry_utils import (
    molecules_to_molar, apply_henry_law, calculate_pH, h_from_pH,
    speciate_acid_base
)
from utils import get_logger
import pandas as pd


# Experimental targets (3.2 kVpp, saline, 6 min)
SALINE_TARGETS = {
    'pH': 3.60,
    'NO2-': 0.0,      # below detection limit
    'NO3-': 102.0,     # μM
    'H2O2': 5.0,       # μM
}


def run_saline_validation():
    logger = get_logger()

    # =========================================================================
    # 0. Load optimal parameters (fitted from DI water)
    # =========================================================================
    params_path = Path(__file__).parent / "optimal_params.yaml"
    if not params_path.exists():
        print(f"ERROR: {params_path} not found. Run run_optimizer.py first.")
        sys.exit(1)

    with open(params_path) as f:
        params = yaml.safe_load(f)

    mt = params.get('mass_transfer', {})
    gas = params.get('gas_phase_unmeasured', {})

    # Apply mass transfer overrides
    for key, val in mt.items():
        if hasattr(MASS_TRANSFER, key):
            setattr(MASS_TRANSFER, key, val)

    gas_overrides = {
        'HONO': gas.get('HONO', 0.0),
        'HONO2': gas.get('HONO2', 0.0),
        'H2O2': gas.get('H2O2', 0.0),
    }

    print("=" * 70)
    print("SALINE VALIDATION — Using DI-water optimal params")
    print("=" * 70)
    print(f"  Mass transfer: δ_gas={mt.get('delta_x_gas', 0)*1000:.0f}mm, "
          f"δ_liq={mt.get('delta_x_liq', 0)*1e6:.0f}μm, "
          f"A/V={mt.get('area_to_volume_ratio', 0):.2f} m⁻¹")
    print(f"  Gas overrides: HONO={gas_overrides['HONO']:.3e}, "
          f"HONO2={gas_overrides['HONO2']:.3e}, "
          f"H2O2={gas_overrides['H2O2']:.3e}")
    print(f"  Solution: 0.9% NaCl (Cl⁻ = 0.154 M)")

    # =========================================================================
    # 1. Load gas-phase data
    # =========================================================================
    csv_path = str(Path(__file__).parent.parent
                   / "empty chamber" / "empty chamber" / "1kHz3.2kVpp.csv")
    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    n_steps = len(df)
    print(f"  CSV: {n_steps} timesteps from {Path(csv_path).name}")

    gas_species_in_csv = [s for s in GAS_PHASE_SPECIES if s in df.columns]

    # =========================================================================
    # 2. Initialize chemistry solver (SALINE MODE)
    # =========================================================================
    chemistry = CompleteAqueousChemistry(saline_mode=True)
    print(f"  Reactions: {len(chemistry.reactions)} "
          f"(base + saline Cl reactions)")

    # =========================================================================
    # 3. Initialize accumulated liquid concentrations
    # =========================================================================
    initial_pH = 7.0
    H_init = h_from_pH(initial_pH)
    accumulated = {}
    accumulated['H+'] = H_init
    accumulated['OH-'] = 1e-14 / H_init
    accumulated['O2'] = 2.5e-4
    accumulated['N2'] = 5e-4
    accumulated['OH'] = 1e-12
    # Saline: Cl⁻ initial
    accumulated['Cl-'] = 0.154  # 0.9% NaCl

    trace = DEFAULTS.trace_concentration
    time_step = 1.0

    history = {
        'time': [], 'pH': [], 'H+': [],
        'NO2-': [], 'NO3-': [], 'H2O2': [],
        'HONO_total': [], 'HONO2_total': [], 'H2O2_total': [],
        'ONOOH_total': [], 'O3': [], 'NO': [],
        'Cl-': [], 'OH': [], 'O3-': [], 'HClO': [],
    }

    # =========================================================================
    # 4. Main simulation loop
    # =========================================================================
    t_start = time.time()
    print(f"\n{'Step':>5s}  {'Time':>6s}  {'pH':>6s}  {'NO2-':>10s}  "
          f"{'NO3-':>10s}  {'H2O2':>10s}  {'OH':>10s}  "
          f"{'O3':>10s}  {'HClO':>10s}  {'elapsed':>8s}")
    print("-" * 105)

    for step_idx in range(n_steps):
        row = df.iloc[step_idx]
        t_current = row.get('Time', step_idx)

        # 4a. Mass transfer: gas → liquid
        C_aq_from_gas = {}
        all_gas_species = set(gas_species_in_csv) | set(gas_overrides.keys())

        for species in all_gas_species:
            if species in gas_overrides and gas_overrides[species] > 0:
                gas_conc = gas_overrides[species]
            else:
                gas_conc = row.get(species, 0.0)
                if pd.isna(gas_conc) or gas_conc <= 0:
                    continue

            if species in ('HONO',):
                aq_key = 'HONO_total'
            elif species in ('HONO2',):
                aq_key = 'HONO2_total'
            elif species in ('H2O2',):
                aq_key = 'H2O2_total'
            else:
                aq_key = species

            current_aq = accumulated.get(aq_key, 0.0)
            new_aq = apply_henry_law(
                species, gas_conc,
                method='two_film', delta_t=time_step,
                current_aq_conc=current_aq
            )
            C_aq_from_gas[aq_key] = new_aq

        # 4b. Update accumulated
        for aq_key, new_conc in C_aq_from_gas.items():
            accumulated[aq_key] = new_conc

        # 4c. Solve aqueous chemistry ODE
        C_aq_initial = {}
        for species in chemistry.aqueous_species:
            if species in accumulated:
                C_aq_initial[species] = accumulated[species]
        for total_name in ACID_BASE_PAIRS:
            if total_name in accumulated:
                C_aq_initial[total_name] = accumulated[total_name]

        current_pH = calculate_pH(accumulated.get('H+', H_init))

        try:
            C_final, contributions = chemistry.solve(
                C_aq_initial, current_pH, time_step=time_step,
                cl_concentration=accumulated.get('Cl-', 0.154)
            )
        except Exception as e:
            logger.warning(f"Step {step_idx} solve failed: {e}")
            C_final = C_aq_initial
            C_final['pH'] = current_pH

        # 4d. Update accumulated with ODE results
        for total_name, (acid_name, base_name, pKa) in ACID_BASE_PAIRS.items():
            acid_conc = C_final.get(acid_name, 0.0)
            base_conc = C_final.get(base_name, 0.0)
            total_conc = acid_conc + base_conc
            if total_conc > trace:
                accumulated[total_name] = total_conc

        speciated_species = set()
        for total_name, (acid_name, base_name, pKa) in ACID_BASE_PAIRS.items():
            speciated_species.add(acid_name)
            speciated_species.add(base_name)

        for species, conc in C_final.items():
            if species == 'pH':
                continue
            if species in speciated_species:
                continue
            if species in ACID_BASE_PAIRS:
                continue
            accumulated[species] = conc

        # 4e. Record history
        pH_now = C_final.get('pH', 7.0)
        H_now = C_final.get('H+', H_init)

        HONO_total = C_final.get('HONO_total', accumulated.get('HONO_total', trace))
        HONO2_total = C_final.get('HONO2_total', accumulated.get('HONO2_total', trace))
        H2O2_total = C_final.get('H2O2_total', accumulated.get('H2O2_total', trace))
        ONOOH_total = C_final.get('ONOOH_total', accumulated.get('ONOOH_total', trace))

        _, NO2_minus = speciate_acid_base(HONO_total, 3.4, H_now)
        _, NO3_minus = speciate_acid_base(HONO2_total, -1.34, H_now)
        H2O2_conc, _ = speciate_acid_base(H2O2_total, 11.65, H_now)

        history['time'].append(t_current)
        history['pH'].append(pH_now)
        history['H+'].append(H_now)
        history['NO2-'].append(NO2_minus)
        history['NO3-'].append(NO3_minus)
        history['H2O2'].append(H2O2_conc)
        history['HONO_total'].append(HONO_total)
        history['HONO2_total'].append(HONO2_total)
        history['H2O2_total'].append(H2O2_total)
        history['ONOOH_total'].append(ONOOH_total)
        history['O3'].append(C_final.get('O3', trace))
        history['NO'].append(C_final.get('NO', trace))
        history['Cl-'].append(C_final.get('Cl-', accumulated.get('Cl-', 0.154)))
        history['OH'].append(C_final.get('OH', trace))
        history['O3-'].append(C_final.get('O3-', trace))
        history['HClO'].append(C_final.get('HClO', trace))

        if step_idx % 60 == 0 or step_idx == n_steps - 1:
            wall = time.time() - t_start
            cl_now = history['Cl-'][-1]
            oh_now = history['OH'][-1]
            o3_now = history['O3'][-1]
            hclo_now = history['HClO'][-1]
            print(f"{step_idx:5d}  {t_current:6.0f}  {pH_now:6.2f}  "
                  f"{NO2_minus*1e6:10.3f}  {NO3_minus*1e6:10.3f}  "
                  f"{H2O2_conc*1e6:10.3f}  {oh_now:10.3e}  "
                  f"{o3_now:10.3e}  {hclo_now:10.3e}  "
                  f"{wall:7.1f}s", flush=True)

    elapsed = time.time() - t_start

    # =========================================================================
    # 5. Final results & comparison
    # =========================================================================
    final_pH = history['pH'][-1]
    final_NO2 = history['NO2-'][-1] * 1e6
    final_NO3 = history['NO3-'][-1] * 1e6
    final_H2O2 = history['H2O2'][-1] * 1e6
    final_Cl = history['Cl-'][-1]

    print("\n" + "=" * 70)
    print("SALINE SIMULATION RESULTS (t = 360s)")
    print("=" * 70)
    print(f"\n  {'':15s}  {'Simulation':>12s}  {'Experiment':>12s}  {'Error':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'pH':15s}  {final_pH:12.3f}  {SALINE_TARGETS['pH']:12.3f}  "
          f"{final_pH - SALINE_TARGETS['pH']:+10.3f}")
    print(f"  {'NO2- (μM)':15s}  {final_NO2:12.3f}  {SALINE_TARGETS['NO2-']:12.3f}  "
          f"{final_NO2 - SALINE_TARGETS['NO2-']:+10.3f}")
    print(f"  {'NO3- (μM)':15s}  {final_NO3:12.3f}  {SALINE_TARGETS['NO3-']:12.3f}  "
          f"{final_NO3 - SALINE_TARGETS['NO3-']:+10.3f}")
    print(f"  {'H2O2 (μM)':15s}  {final_H2O2:12.3f}  {SALINE_TARGETS['H2O2']:12.3f}  "
          f"{final_H2O2 - SALINE_TARGETS['H2O2']:+10.3f}")
    print(f"\n  Cl⁻ final: {final_Cl:.4f} M (initial: 0.154 M, "
          f"change: {(final_Cl - 0.154)*1e3:+.3f} mM)")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save results
    output_path = Path(__file__).parent / "saline_results.csv"
    df_out = pd.DataFrame(history)
    df_out.to_csv(output_path, index=False)
    print(f"  Results saved to: {output_path}")

    return history


if __name__ == "__main__":
    run_saline_validation()
