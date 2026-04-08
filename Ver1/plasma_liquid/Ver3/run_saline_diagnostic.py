#!/usr/bin/env python3
"""
Saline Diagnostic: Track ALL species concentrations to identify bottlenecks.

Runs the same simulation as run_saline_test.py but dumps EVERY species
concentration at each timestep. Goal: find which intermediate is
unrealistically low compared to literature, explaining why Cl chemistry
doesn't fire.
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    GAS_PHASE_SPECIES, HENRY_CONSTANTS, DEFAULTS,
    ACID_BASE_PAIRS, MASS_TRANSFER, AQUEOUS_SPECIES, SALINE_SPECIES
)
from chemistry import CompleteAqueousChemistry
from chemistry_utils import (
    molecules_to_molar, apply_henry_law, calculate_pH, h_from_pH,
    speciate_acid_base, calculate_mass_transfer_coefficient
)
from utils import get_logger
import pandas as pd


# Key species to analyze for Cl chemistry bottleneck
DIAGNOSTIC_SPECIES = {
    # Cl entry-point reactants (what feeds into Cl reactions)
    'entry_reactants': ['OH', 'O3', 'H2O2', 'HO2', 'O2-', 'ONOOH', 'O2NOOH', 'O3-', 'NO3'],
    # Cl intermediates (should build up if Cl chemistry works)
    'cl_intermediates': ['Cl', 'Cl2-', 'HOCl-', 'Cl2', 'HClO', 'ClO-', 'ClNO2', 'HOClH',
                         'ClO', 'ClO2', 'Cl2O2', 'HClO2', 'ClO2-'],
    # NOx species affected by Cl
    'nox': ['NO2-', 'NO3-', 'NO', 'NO2', 'N2O3', 'N2O4', 'N2O5', 'HONO', 'HONO2',
            'ONOO-', 'O2NOO-'],
    # Other ROS
    'ros': ['H2O2', 'HO2-', 'O-', 'H', 'HO3'],
    # Conserved
    'conserved': ['Cl-', 'H+', 'OH-', 'O2'],
}

# Flatten for tracking
ALL_DIAG_SPECIES = []
for group in DIAGNOSTIC_SPECIES.values():
    ALL_DIAG_SPECIES.extend(group)
ALL_DIAG_SPECIES = list(dict.fromkeys(ALL_DIAG_SPECIES))  # deduplicate preserving order


def run_diagnostic():
    logger = get_logger()

    # =========================================================================
    # 0. Load optimal parameters
    # =========================================================================
    params_path = Path(__file__).parent / "optimal_params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    mt = params.get('mass_transfer', {})
    gas = params.get('gas_phase_unmeasured', {})

    for key, val in mt.items():
        if hasattr(MASS_TRANSFER, key):
            setattr(MASS_TRANSFER, key, val)

    gas_overrides = {
        'HONO': gas.get('HONO', 0.0),
        'HONO2': gas.get('HONO2', 0.0),
        'H2O2': gas.get('H2O2', 0.0),
    }

    print("=" * 80)
    print("SALINE DIAGNOSTIC — Full Species Concentration Tracking")
    print("=" * 80)
    print(f"  A/V={mt.get('area_to_volume_ratio', 0):.2f} m⁻¹")
    print(f"  Gas: HONO={gas_overrides['HONO']:.3e}, HONO2={gas_overrides['HONO2']:.3e}, "
          f"H2O2={gas_overrides['H2O2']:.3e}")

    # =========================================================================
    # 1. Calculate mass transfer coefficients for reference
    # =========================================================================
    print("\n--- Mass Transfer Coefficients ---")
    for species in GAS_PHASE_SPECIES:
        H = HENRY_CONSTANTS.get(species, 1.0)
        k_mt = calculate_mass_transfer_coefficient(species, H)
        print(f"  {species:8s}: H={H:.4e}, k_L*a={k_mt:.4e} s⁻¹")

    # =========================================================================
    # 2. Load gas-phase data
    # =========================================================================
    csv_path = str(Path(__file__).parent.parent
                   / "empty chamber" / "empty chamber" / "1kHz3.2kVpp.csv")
    df = pd.read_csv(csv_path)
    n_steps = len(df)
    gas_species_in_csv = [s for s in GAS_PHASE_SPECIES if s in df.columns]

    # Print gas-phase O3 stats
    if 'O3' in df.columns:
        o3_gas = df['O3'].values
        o3_molar = [molecules_to_molar(v) for v in o3_gas if v > 0]
        if o3_molar:
            H_o3 = HENRY_CONSTANTS.get('O3', 0.2298)
            c_eq_o3 = H_o3 * np.mean(o3_molar)
            print(f"\n  Gas O3: mean={np.mean(o3_gas):.3e} cm⁻³, "
                  f"C_gas_molar={np.mean(o3_molar):.3e} M, "
                  f"C_eq(aq)={c_eq_o3:.3e} M")

    # =========================================================================
    # 3. Initialize chemistry solver (SALINE MODE)
    # =========================================================================
    chemistry = CompleteAqueousChemistry(saline_mode=True)
    print(f"  Reactions: {len(chemistry.reactions)} total")
    print(f"  Species in ODE: {len(chemistry.aqueous_species)}")

    # =========================================================================
    # 4. Initialize
    # =========================================================================
    initial_pH = 7.0
    H_init = h_from_pH(initial_pH)
    accumulated = {}
    accumulated['H+'] = H_init
    accumulated['OH-'] = 1e-14 / H_init
    accumulated['O2'] = 2.5e-4
    accumulated['N2'] = 5e-4
    accumulated['OH'] = 1e-12
    accumulated['Cl-'] = 0.154

    trace = DEFAULTS.trace_concentration
    time_step = 1.0

    # History: track ALL diagnostic species
    history = {'time': []}
    for sp in ALL_DIAG_SPECIES:
        history[sp] = []
    history['pH'] = []

    # Also track totals for acid-base pairs
    for total_name in ACID_BASE_PAIRS:
        history[total_name] = []

    # =========================================================================
    # 5. Main simulation loop
    # =========================================================================
    t_start = time.time()
    print(f"\n{'Step':>5s}  {'Time':>6s}  {'pH':>6s}  {'OH':>12s}  {'O3':>12s}  "
          f"{'O3-':>12s}  {'Cl':>12s}  {'HClO':>12s}  {'NO2-':>10s}")
    print("-" * 100)

    for step_idx in range(n_steps):
        row = df.iloc[step_idx]
        t_current = row.get('Time', step_idx)

        # 5a. Mass transfer
        C_aq_from_gas = {}
        all_gas_species = set(gas_species_in_csv) | set(gas_overrides.keys())

        for species in all_gas_species:
            if species in gas_overrides and gas_overrides[species] > 0:
                gas_conc = gas_overrides[species]
            else:
                gas_conc = row.get(species, 0.0)
                if pd.isna(gas_conc) or gas_conc <= 0:
                    continue

            if species == 'HONO':
                aq_key = 'HONO_total'
            elif species == 'HONO2':
                aq_key = 'HONO2_total'
            elif species == 'H2O2':
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

        for aq_key, new_conc in C_aq_from_gas.items():
            accumulated[aq_key] = new_conc

        # 5b. Solve ODE
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

        # 5c. Update accumulated
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

        # 5d. Record ALL species
        pH_now = C_final.get('pH', 7.0)
        H_now = C_final.get('H+', H_init)
        history['time'].append(t_current)
        history['pH'].append(pH_now)

        for total_name in ACID_BASE_PAIRS:
            history[total_name].append(C_final.get(total_name,
                                       accumulated.get(total_name, trace)))

        for sp in ALL_DIAG_SPECIES:
            # For speciated species, get from C_final directly
            val = C_final.get(sp, trace)
            # For species tracked as totals, also check accumulated
            if val <= trace and sp in accumulated:
                val = accumulated[sp]
            history[sp].append(val)

        # Print progress
        if step_idx % 60 == 0 or step_idx == n_steps - 1:
            oh = C_final.get('OH', trace)
            o3 = C_final.get('O3', trace)
            o3m = C_final.get('O3-', trace)
            cl = C_final.get('Cl', trace)
            hclo = C_final.get('HClO', trace)
            no2m = C_final.get('NO2-', trace)
            print(f"{step_idx:5d}  {t_current:6.0f}  {pH_now:6.3f}  "
                  f"{oh:12.3e}  {o3:12.3e}  {o3m:12.3e}  "
                  f"{cl:12.3e}  {hclo:12.3e}  {no2m*1e6:10.3f}μM",
                  flush=True)

    elapsed = time.time() - t_start

    # =========================================================================
    # 6. Final Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SPECIES CONCENTRATIONS (t=360s)")
    print("=" * 80)

    for group_name, species_list in DIAGNOSTIC_SPECIES.items():
        print(f"\n--- {group_name.upper()} ---")
        print(f"  {'Species':15s}  {'Concentration':>14s}  {'Unit':>6s}  Notes")
        print(f"  {'-'*60}")
        for sp in species_list:
            val = history[sp][-1] if sp in history and history[sp] else trace
            if val > 1e-3:
                print(f"  {sp:15s}  {val:14.4e}  {'M':>6s}  {val*1e3:.2f} mM")
            elif val > 1e-6:
                print(f"  {sp:15s}  {val:14.4e}  {'M':>6s}  {val*1e6:.2f} μM")
            elif val > 1e-9:
                print(f"  {sp:15s}  {val:14.4e}  {'M':>6s}  {val*1e9:.2f} nM")
            elif val > 1e-12:
                print(f"  {sp:15s}  {val:14.4e}  {'M':>6s}  {val*1e12:.2f} pM")
            elif val > trace * 10:
                print(f"  {sp:15s}  {val:14.4e}  {'M':>6s}  {val*1e15:.2f} fM")
            else:
                print(f"  {sp:15s}  {val:14.4e}  {'M':>6s}  ≈ TRACE")

    # =========================================================================
    # 7. Cl Chemistry Rate Analysis at final timestep
    # =========================================================================
    print("\n" + "=" * 80)
    print("Cl⁻ ENTRY-POINT REACTION RATES (final timestep)")
    print("=" * 80)

    cl_m = history['Cl-'][-1]
    oh = history['OH'][-1]
    o3 = history['O3'][-1]
    h2o2 = history['H2O2'][-1]
    ho2 = history['HO2'][-1]
    o2m = history['O2-'][-1]
    onooh = history['ONOOH'][-1]
    o2nooh = history['O2NOOH'][-1]
    o3m = history['O3-'][-1]
    no3_rad = history['NO3'][-1] if 'NO3' in history else trace
    h_plus = history['H+'][-1]

    reactions = [
        ("S19: Cl⁻+O₃→ClO⁻", 2e-3, cl_m * o3, "ClO⁻"),
        ("S20: Cl⁻+H₂O₂→ClO⁻", 1.8e-9, cl_m * h2o2, "ClO⁻"),
        ("S21: Cl⁻+H₂O₂+H⁺→HClO", 8.3e-7, cl_m * h2o2 * h_plus, "HClO"),
        ("S22: Cl⁻+OH→Cl", 4.3e9, cl_m * oh, "Cl"),
        ("S55: NO₃+Cl⁻→NO₃⁻+Cl", 1e8, no3_rad * cl_m, "Cl"),
        ("S96: Cl⁻+O₂NOOH→HClO", 1.4e-2, cl_m * o2nooh, "HClO"),
        ("S4f: OH+Cl⁻→HOCl⁻", 4.3e9, oh * cl_m, "HOCl⁻"),
    ]

    print(f"\n  {'Reaction':35s}  {'Rate (M/s)':>12s}  {'360s cumul':>12s}  Product")
    print(f"  {'-'*80}")
    for label, k, reactant_product, product in reactions:
        rate = k * reactant_product
        cumul = rate * 360
        if cumul > 1e-6:
            cumul_str = f"{cumul*1e6:.2f} μM"
        elif cumul > 1e-9:
            cumul_str = f"{cumul*1e9:.2f} nM"
        else:
            cumul_str = f"{cumul:.2e} M"
        print(f"  {label:35s}  {rate:12.3e}  {cumul_str:>12s}  {product}")

    # =========================================================================
    # 8. O3 Budget
    # =========================================================================
    print("\n" + "=" * 80)
    print("O₃ BUDGET ANALYSIS")
    print("=" * 80)

    H_o3 = HENRY_CONSTANTS.get('O3', 0.2298)
    k_mt_o3 = calculate_mass_transfer_coefficient('O3', H_o3)
    # Get typical gas O3
    if 'O3' in df.columns:
        o3_gas_last = df['O3'].iloc[-1]
        o3_gas_molar = molecules_to_molar(o3_gas_last)
        c_eq_o3 = H_o3 * o3_gas_molar
        mt_rate = k_mt_o3 * (c_eq_o3 - o3)
        print(f"  Gas O₃ = {o3_gas_last:.3e} cm⁻³ = {o3_gas_molar:.3e} M")
        print(f"  C_eq(aq) = H × C_gas = {H_o3} × {o3_gas_molar:.3e} = {c_eq_o3:.3e} M")
        print(f"  Actual [O₃(aq)] = {o3:.3e} M")
        print(f"  k_L*a = {k_mt_o3:.4e} s⁻¹")
        print(f"  Mass transfer rate = k_L*a × (C_eq - C_aq) = {mt_rate:.3e} M/s")

    no2m_final = history['NO2-'][-1]
    print(f"\n  Key O₃ sinks:")
    print(f"    R32: O₃+NO₂⁻ (k=5e5): rate = {5e5 * o3 * no2m_final:.3e} M/s")
    print(f"    R25: O₃+O₂⁻ (k=1.6e9): rate = {1.6e9 * o3 * o2m:.3e} M/s")
    print(f"    R27: O₃+OH (k=3e9): rate = {3e9 * o3 * oh:.3e} M/s")
    print(f"    R30: O₃+H₂O₂ (k=6.5e-3): rate = {6.5e-3 * o3 * h2o2:.3e} M/s")
    print(f"    S19: O₃+Cl⁻ (k=2e-3): rate = {2e-3 * o3 * cl_m:.3e} M/s")

    # =========================================================================
    # 9. OH Budget
    # =========================================================================
    print("\n" + "=" * 80)
    print("OH BUDGET ANALYSIS")
    print("=" * 80)

    print(f"  [OH] = {oh:.3e} M")
    print(f"\n  OH Production:")
    print(f"    R38: O₃⁻+H⁺→OH (k=9e10): rate = {9e10 * o3m * h_plus:.3e} M/s")
    print(f"    R33: O₃⁻→OH+O₂+OH⁻ (k=25): rate = {25 * o3m:.3e} M/s")
    print(f"    R30: O₃+H₂O₂→OH (k=6.5e-3): rate = {6.5e-3 * o3 * h2o2:.3e} M/s")
    print(f"    R15f: ONOOH→NO₂+OH (k=0.35): rate = {0.35 * onooh:.3e} M/s")
    ho2m = history.get('HO2-', [trace])[-1]
    print(f"    R26: O₃+HO₂⁻→OH (k=5.5e6): rate = {5.5e6 * o3 * ho2m:.3e} M/s")

    print(f"\n  OH Consumption:")
    print(f"    R77: OH+NO₂⁻ (k=1e9): rate = {1e9 * oh * no2m_final:.3e} M/s")
    hono = history['HONO'][-1]
    hono2 = history['HONO2'][-1]
    print(f"    R78: OH+HONO (k=1e9): rate = {1e9 * oh * hono:.3e} M/s")
    print(f"    R79: OH+HONO₂ (k=5.3e7): rate = {5.3e7 * oh * hono2:.3e} M/s")
    print(f"    R41: OH+H₂O₂ (k=2.7e7): rate = {2.7e7 * oh * h2o2:.3e} M/s")
    print(f"    S4f: OH+Cl⁻→HOCl⁻ (k=4.3e9): rate = {4.3e9 * oh * cl_m:.3e} M/s")
    print(f"    S22: OH+Cl⁻→Cl (k=4.3e9): rate = {4.3e9 * oh * cl_m:.3e} M/s")
    print(f"    S4+S22 total Cl⁻ sink: {2 * 4.3e9 * oh * cl_m:.3e} M/s")
    non_cl_sink = 1e9*oh*no2m_final + 1e9*oh*hono + 5.3e7*oh*hono2 + 2.7e7*oh*h2o2
    cl_sink = 2 * 4.3e9 * oh * cl_m
    print(f"    Non-Cl OH sinks: {non_cl_sink:.3e} M/s")
    print(f"    Cl OH sinks: {cl_sink:.3e} M/s")
    if non_cl_sink > 0:
        print(f"    Cl/Non-Cl ratio: {cl_sink/non_cl_sink:.1f}×")

    # =========================================================================
    # 10. Save full data
    # =========================================================================
    output_path = Path(__file__).parent / "saline_diagnostic.csv"
    df_out = pd.DataFrame(history)
    df_out.to_csv(output_path, index=False)
    print(f"\n  Full diagnostic data saved to: {output_path}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return history


if __name__ == "__main__":
    run_diagnostic()
