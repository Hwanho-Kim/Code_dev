#!/usr/bin/env python3
"""
Baseline Simulation: Accumulating liquid-phase chemistry over 360 seconds.

Reads gas-phase OAS data from 1kHz3.2kVpp.csv and runs:
  1. Mass transfer (two-film theory) at each 1s timestep
  2. Aqueous-phase ODE chemistry (101 reactions from reactions_full.yaml)
  3. Accumulates liquid concentrations across timesteps

Baseline condition: HONO=0, HONO2=0, H2O2=0 (unmeasured species set to zero).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    GAS_PHASE_SPECIES, HENRY_CONSTANTS, DEFAULTS,
    ACID_BASE_PAIRS, MASS_TRANSFER
)
from chemistry import CompleteAqueousChemistry
from chemistry_utils import (
    molecules_to_molar, apply_henry_law, calculate_pH, h_from_pH,
    speciate_acid_base, apply_mass_transfer_limited_henry
)
from utils import get_logger


def run_baseline(csv_path: str, initial_pH: float = 7.0, time_step: float = 1.0,
                  override_mass_transfer: dict = None,
                  override_gas: dict = None):
    """
    Run simulation with accumulated liquid concentrations.

    Parameters
    ----------
    csv_path : str
        Path to gas-phase OAS CSV file
    initial_pH : float
        Initial pH of DI water
    time_step : float
        Time step in seconds (matches CSV time resolution)
    override_mass_transfer : dict, optional
        Override mass transfer params: {delta_x_gas, delta_x_liq, area_to_volume_ratio}
    override_gas : dict, optional
        Override gas-phase concentrations: {HONO, HONO2, H2O2} in molecules/cm³
    """
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("SIMULATION — Accumulated Liquid-Phase Chemistry")
    logger.info("=" * 70)

    # =========================================================================
    # 0. Apply parameter overrides
    # =========================================================================
    if override_mass_transfer:
        for key, val in override_mass_transfer.items():
            if hasattr(MASS_TRANSFER, key):
                setattr(MASS_TRANSFER, key, val)
                logger.info(f"Override: MASS_TRANSFER.{key} = {val}")

    gas_overrides = override_gas or {}

    # =========================================================================
    # 1. Load gas-phase data
    # =========================================================================
    df = pd.read_csv(csv_path)
    n_steps = len(df)
    logger.info(f"Loaded {n_steps} rows from {Path(csv_path).name}")

    # Check available gas species
    gas_species_in_csv = [s for s in GAS_PHASE_SPECIES if s in df.columns]
    logger.info(f"Gas species in CSV: {gas_species_in_csv}")

    # =========================================================================
    # 2. Initialize chemistry solver
    # =========================================================================
    chemistry = CompleteAqueousChemistry()
    logger.info(f"Loaded {len(chemistry.reactions)} aqueous reactions")

    # Species indices for reading results
    species_idx = chemistry.species_idx

    # =========================================================================
    # 3. Initialize accumulated liquid concentrations
    # =========================================================================
    # Start with DI water at given pH
    accumulated = {}  # species -> concentration [mol/L]

    # Set initial H+ and OH-
    H_init = h_from_pH(initial_pH)
    accumulated['H+'] = H_init
    accumulated['OH-'] = 1e-14 / H_init

    # Atmospheric dissolved gases
    accumulated['O2'] = 2.5e-4  # mol/L (saturated)
    accumulated['N2'] = 5e-4    # mol/L (saturated)

    # Initial OH radical
    accumulated['OH'] = 1e-12

    # All other species start at trace
    trace = DEFAULTS.trace_concentration

    # Track time history for key species
    history = {
        'time': [],
        'pH': [],
        'H+': [],
        'NO2-': [],
        'NO3-': [],
        'H2O2': [],
        'HONO_total': [],
        'HONO2_total': [],
        'H2O2_total': [],
        'ONOOH_total': [],
        'O3': [],       # aqueous O3
        'NO': [],       # aqueous NO
        'NO2_aq': [],   # aqueous NO2
        'N2O4': [],     # aqueous N2O4
        'N2O5': [],     # aqueous N2O5
    }

    # =========================================================================
    # 4. Main simulation loop
    # =========================================================================
    t_start = time.time()
    print(f"\n{'Step':>5s}  {'Time(s)':>8s}  {'pH':>6s}  {'NO2-(uM)':>10s}  "
          f"{'NO3-(uM)':>10s}  {'H2O2(uM)':>10s}  {'ONOO-(uM)':>10s}  {'elapsed':>8s}")
    print("-" * 85)

    for step_idx in range(n_steps):
        row = df.iloc[step_idx]
        t_current = row.get('Time', step_idx)

        # =====================================================================
        # 4a. Mass transfer: gas → liquid for each gas species
        # =====================================================================
        C_aq_from_gas = {}  # Fresh mass transfer contributions this timestep

        all_gas_species = set(gas_species_in_csv) | set(gas_overrides.keys())
        for species in all_gas_species:
            # Override species use constant gas concentration
            if species in gas_overrides and gas_overrides[species] > 0:
                gas_conc = gas_overrides[species]
            else:
                gas_conc = row.get(species, 0.0)
                if pd.isna(gas_conc) or gas_conc <= 0:
                    continue

            # Current aqueous concentration for this species
            # Map gas species to aqueous species / total name
            if species in ('HONO',):
                aq_key = 'HONO_total'
            elif species in ('HONO2',):
                aq_key = 'HONO2_total'
            elif species in ('H2O2',):
                aq_key = 'H2O2_total'
            else:
                aq_key = species

            current_aq = accumulated.get(aq_key, 0.0)

            # Apply two-film mass transfer
            # apply_henry_law returns new aqueous concentration after mass transfer
            new_aq = apply_henry_law(
                species,
                gas_conc,
                method='two_film',
                delta_t=time_step,
                current_aq_conc=current_aq
            )

            C_aq_from_gas[aq_key] = new_aq

        # =====================================================================
        # 4b. Update accumulated concentrations with mass transfer results
        # =====================================================================
        for aq_key, new_conc in C_aq_from_gas.items():
            accumulated[aq_key] = new_conc

        # =====================================================================
        # 4c. Solve aqueous chemistry ODE for this timestep
        # =====================================================================
        # Build initial concentration dict for ODE solver
        C_aq_initial = {}
        for species in chemistry.aqueous_species:
            if species in accumulated:
                C_aq_initial[species] = accumulated[species]

        # Also map acid/base pairs correctly
        # If we have HONO_total in accumulated, pass it
        for total_name in ACID_BASE_PAIRS:
            if total_name in accumulated:
                C_aq_initial[total_name] = accumulated[total_name]

        # Get current pH from accumulated H+
        current_pH = calculate_pH(accumulated.get('H+', H_init))

        # Solve ODE (1 second timestep)
        try:
            C_final, contributions = chemistry.solve(
                C_aq_initial, current_pH, time_step=time_step
            )
        except Exception as e:
            logger.warning(f"Step {step_idx} solve failed: {e}")
            C_final = C_aq_initial
            C_final['pH'] = current_pH

        # =====================================================================
        # 4d. Update accumulated with ODE results
        # =====================================================================
        # C_final contains speciated species (HONO, NO2-, NO3-, etc.)
        # We need to reconstruct TOTAL concentrations for acid-base pairs
        # so that initialize_concentrations() works correctly next timestep.

        # First, reconstruct totals from speciated pairs
        for total_name, (acid_name, base_name, pKa) in ACID_BASE_PAIRS.items():
            acid_conc = C_final.get(acid_name, 0.0)
            base_conc = C_final.get(base_name, 0.0)
            total_conc = acid_conc + base_conc
            if total_conc > trace:
                accumulated[total_name] = total_conc

        # Then update non-equilibrium species (skip speciated acid/base individuals)
        speciated_species = set()
        for total_name, (acid_name, base_name, pKa) in ACID_BASE_PAIRS.items():
            speciated_species.add(acid_name)
            speciated_species.add(base_name)

        for species, conc in C_final.items():
            if species == 'pH':
                continue
            if species in speciated_species:
                continue  # Already handled via totals above
            if species in ACID_BASE_PAIRS:
                continue  # total names shouldn't appear in C_final
            accumulated[species] = conc

        # =====================================================================
        # 4e. Record history
        # =====================================================================
        pH_now = C_final.get('pH', 7.0)
        H_now = C_final.get('H+', H_init)

        # Speciate to get individual NO2-, NO3-, H2O2, ONOO-
        HONO_total = C_final.get('HONO_total', accumulated.get('HONO_total', trace))
        HONO2_total = C_final.get('HONO2_total', accumulated.get('HONO2_total', trace))
        H2O2_total = C_final.get('H2O2_total', accumulated.get('H2O2_total', trace))
        ONOOH_total = C_final.get('ONOOH_total', accumulated.get('ONOOH_total', trace))

        # Use speciation to get individual species
        _, NO2_minus = speciate_acid_base(HONO_total, 3.4, H_now)
        _, NO3_minus = speciate_acid_base(HONO2_total, -1.34, H_now)
        H2O2_conc, _ = speciate_acid_base(H2O2_total, 11.65, H_now)
        _, ONOO_minus = speciate_acid_base(ONOOH_total, 6.6, H_now)

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
        history['NO2_aq'].append(C_final.get('NO2', trace))
        history['N2O4'].append(C_final.get('N2O4', trace))
        history['N2O5'].append(C_final.get('N2O5', trace))

        # Print progress every 30 seconds
        if step_idx % 30 == 0 or step_idx == n_steps - 1:
            wall = time.time() - t_start
            print(f"{step_idx:5d}  {t_current:8.1f}  {pH_now:6.2f}  "
                  f"{NO2_minus * 1e6:10.3f}  {NO3_minus * 1e6:10.3f}  "
                  f"{H2O2_conc * 1e6:10.3f}  {ONOO_minus * 1e6:10.3f}  "
                  f"{wall:7.1f}s", flush=True)

    elapsed = time.time() - t_start

    # =========================================================================
    # 5. Final results
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE SIMULATION RESULTS (t = 360s)")
    print("=" * 70)

    final_pH = history['pH'][-1]
    final_NO2 = history['NO2-'][-1] * 1e6   # to μM
    final_NO3 = history['NO3-'][-1] * 1e6   # to μM
    final_H2O2 = history['H2O2'][-1] * 1e6  # to μM
    final_ONOO = history['ONOOH_total'][-1] * 1e6  # total ONOOH+ONOO-

    print(f"\n  pH:        {final_pH:.3f}")
    print(f"  NO2-:      {final_NO2:.3f} μM")
    print(f"  NO3-:      {final_NO3:.3f} μM")
    print(f"  H2O2(aq):  {final_H2O2:.3f} μM")
    print(f"  ONOO-:     {final_ONOO:.6f} μM (peroxynitrite)")

    print(f"\n--- Experimental Targets (3.2 kVpp, DI water, 6 min) ---")
    print(f"  pH:        3.61 ± 0.04")
    print(f"  NO2-:      ~3 μM")
    print(f"  NO3-:      ~63 μM")
    print(f"  H2O2(aq):  ~11 μM")

    print(f"\n--- Deficit (target - baseline) ---")
    print(f"  pH:        {3.61 - final_pH:+.3f}")
    print(f"  NO2-:      {3.0 - final_NO2:+.3f} μM")
    print(f"  NO3-:      {63.0 - final_NO3:+.3f} μM")
    print(f"  H2O2(aq):  {11.0 - final_H2O2:+.3f} μM")

    print(f"\nElapsed: {elapsed:.1f} seconds")

    # =========================================================================
    # 6. Mass transfer diagnostics
    # =========================================================================
    print(f"\n--- Mass Transfer Config ---")
    print(f"  δ_gas:     {MASS_TRANSFER.delta_x_gas*1000:.1f} mm")
    print(f"  δ_liq:     {MASS_TRANSFER.delta_x_liq*1e6:.0f} μm")
    print(f"  A/V:       {MASS_TRANSFER.area_to_volume_ratio:.0f} m⁻¹")
    print(f"  depth:     {MASS_TRANSFER.liquid_depth*1000:.1f} mm")

    # Gas-phase input at final timestep
    last_row = df.iloc[-1]
    print(f"\n--- Gas-Phase at t=360s (molecules/cm³) ---")
    for sp in gas_species_in_csv:
        val = last_row.get(sp, 0)
        print(f"  {sp:8s}: {val:.3e}")

    # Aqueous concentrations (non-trace)
    print(f"\n--- Aqueous Concentrations at t=360s (mol/L) ---")
    for sp in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5',
               'HONO_total', 'HONO2_total', 'H2O2_total',
               'HO2_total', 'ONOOH_total', 'OH', 'H+', 'OH-']:
        val = accumulated.get(sp, trace)
        if val > trace * 10:
            print(f"  {sp:15s}: {val:.4e} M  ({val*1e6:.3f} μM)")

    # =========================================================================
    # 7. Save time-history to CSV
    # =========================================================================
    output_path = Path(__file__).parent / "baseline_results.csv"
    df_out = pd.DataFrame(history)
    df_out.to_csv(output_path, index=False)
    print(f"\nTime-history saved to: {output_path}")

    return history


if __name__ == "__main__":
    csv_path = str(Path(__file__).parent.parent / "empty chamber" / "empty chamber" / "1kHz3.2kVpp.csv")

    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    if '--validate' in sys.argv:
        # Validation mode: use optimal parameters from optimizer
        import yaml
        params_path = Path(__file__).parent / "optimal_params.yaml"
        if params_path.exists():
            with open(params_path) as f:
                params = yaml.safe_load(f)
            mt = params.get('mass_transfer', {})
            gas = params.get('gas_phase_unmeasured', {})
            run_baseline(
                csv_path,
                override_mass_transfer={
                    'delta_x_gas': mt.get('delta_x_gas', 0.001),
                    'delta_x_liq': mt.get('delta_x_liq', 0.0001),
                    'area_to_volume_ratio': mt.get('area_to_volume_ratio', 100.0),
                },
                override_gas={
                    'HONO': gas.get('HONO', 0.0),
                    'HONO2': gas.get('HONO2', 0.0),
                    'H2O2': gas.get('H2O2', 0.0),
                }
            )
        else:
            print(f"ERROR: {params_path} not found. Run run_optimizer.py first.")
            sys.exit(1)
    else:
        run_baseline(csv_path)
