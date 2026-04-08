#!/usr/bin/env python3
"""
Compute reaction enthalpy (ΔH) for all 177 reactions in reactions.yaml
using Hess's law: ΔH = Σ(coeff × ΔHf°)_products - Σ(coeff × ΔHf°)_reactants

Output: enthalpy_reference.md

Notes:
- Electron ΔHf° = 0 (convention)
- ELECTRON_IMPACT reactions: electron is implicit reactant AND implicit product
  (both sides cancel, so only target species matter)
- For ionization (e + X -> X+ + 2e), products list includes {e, coeff:1}
  meaning ONE net electron is produced. The "2e" in formula means
  1 electron in + 1 electron in products list → net gain of 1e.
  Since ΔHf°(e)=0, this doesn't affect ΔH.
- M (third body) not in reactants/products lists → no ΔHf° contribution
- ΔH in kJ/mol and eV/molecule (1 eV = 96.485 kJ/mol)
"""

import yaml
import sys
from pathlib import Path
from datetime import datetime

# ── ΔHf° data (kJ/mol, 298 K) ──────────────────────────────────────────────
# Sources: NIST-JANAF, ATcT (Active Thermochemical Tables), NIST WebBook
# Ion values computed from: ΔHf°(ion) = ΔHf°(neutral) + IP (positive) or - EA (negative)
# Excited states: ΔHf°(excited) = ΔHf°(ground) + excitation energy

DELTA_HF = {
    # Electron (convention)
    'e': 0.0,

    # Feed gases
    'CH4': -74.8,    # NIST-JANAF
    'CO2': -393.5,   # NIST-JANAF
    'N2':  0.0,      # element
    'O2':  0.0,      # element
    'H2':  0.0,      # element
    'H2O': -241.8,   # NIST-JANAF

    # Atoms & small radicals
    'H':      218.0,   # NIST-JANAF
    'O':      249.2,   # NIST-JANAF
    'C':      716.7,   # NIST-JANAF
    'N':      472.7,   # NIST-JANAF
    'N(2D)':  702.3,   # N + 2.38 eV (229.6 kJ/mol)
    'OH':     39.0,    # NIST-JANAF
    'CH':     596.4,   # NIST WebBook
    'CH2':    391.2,   # NIST WebBook (triplet)
    'CH3':    146.7,   # ATcT

    # Excited N2 states
    'N2(A)':  595.3,   # N2 + 6.17 eV (595.3 kJ/mol)
    'N2(a1)': 810.5,   # N2 + 8.40 eV (810.5 kJ/mol)

    # C1 oxygenated
    'CO':     -110.5,  # NIST-JANAF
    'HCO':    43.5,    # NIST WebBook
    'CH2O':   -115.9,  # NIST-JANAF (ATcT: -109.2)
    'CH2OH':  -16.4,   # NIST WebBook
    'CH3O':   21.8,    # NIST WebBook
    'CH3OH':  -200.9,  # NIST-JANAF
    'CH3O2':  12.7,    # NIST WebBook
    'CH3OOH': -127.9,  # NIST WebBook
    'CH2CO':  -47.5,   # NIST WebBook (±2, disputed)
    'CH3CO':  -12.0,   # NIST WebBook (±3, moderate)
    'CH3CHO': -170.7,  # NIST-JANAF

    # HOx
    'HO2':   2.1,      # ATcT
    'H2O2':  -136.1,   # NIST-JANAF
    'O3':    142.7,     # NIST-JANAF

    # NOx
    'NO':   90.3,       # NIST-JANAF
    'NO2':  33.2,       # NIST-JANAF
    'NO3':  71.1,       # NIST WebBook

    # C2 hydrocarbons
    'C2H2':    226.7,   # NIST-JANAF
    'C2H3':    297.2,   # NIST WebBook
    'C2H4':    52.5,    # NIST-JANAF
    'C2H4OH':  -23.5,   # NIST WebBook
    'C2H5':    120.6,   # NIST WebBook
    'C2H6':    -84.0,   # NIST-JANAF
    'C2HO':    178.2,   # NIST WebBook (HCCO ketenyl)
    'C2H5O':   -8.9,    # NIST WebBook
    'C2H5O2':  -28.4,   # NIST WebBook (±1.5, moderate)
    'C2H5OH':  -234.0,  # NIST-JANAF
    'C2H5OOH': -161.4,  # ATcT (NOT old NIST -210)

    # C3+
    'C3H5':  171.0,     # NIST WebBook (allyl)
    'C3H6':  20.4,      # NIST-JANAF (propene)
    'C3H7':  100.0,     # NIST WebBook (n-propyl)
    'C3H8':  -104.7,    # NIST-JANAF
    'C4H10': -125.6,    # NIST-JANAF (n-butane)

    # Positive ions (ΔHf° = ΔHf°(neutral) + IP)
    'N2+':   1503.2,    # 0 + 15.58 eV
    'O2+':   1164.6,    # 0 + 12.07 eV
    'CO2+':  936.1,     # -393.5 + 13.78 eV
    'CH4+':  1141.9,    # -74.8 + 12.61 eV
    'CH3+':  1096.1,    # 146.7 + 9.84 eV
    'CH5+':  912.0,     # estimated from PA(CH4) = 5.7 eV
    'N4+':   1406.7,    # N2+N2 cluster, from appearance energy
    'O4+':   1087.6,    # O2+O2 cluster, from appearance energy
    'NO+':   983.7,     # 90.3 + 9.26 eV
    'H3O+':  597.2,     # -241.8 + 8.7 eV (estimated PA)

    # Negative ions (ΔHf° = ΔHf°(neutral) - EA)
    'O-':    108.2,     # 249.2 - 1.461 eV
    'O2-':   -43.2,     # 0 - 0.448 eV
}

EV_PER_KJMOL = 1.0 / 96.485  # 1 eV = 96.485 kJ/mol


def compute_reaction_dH(rxn):
    """
    Compute ΔH for a single reaction using Hess's law.

    For ELECTRON_IMPACT type:
      - The implicit electron on reactant side is NOT in reactants list
      - Products may contain 'e' with coeff:1 (for ionization)
      - Since ΔHf°(e)=0, electron terms cancel regardless

    Returns: (dH_kj, missing_species_list)
    """
    missing = []

    # Sum products
    sum_products = 0.0
    for p in rxn.get('products', []):
        sp = p['species']
        coeff = p.get('coeff', 1)
        if sp in DELTA_HF:
            sum_products += coeff * DELTA_HF[sp]
        else:
            missing.append(sp)

    # Sum reactants
    sum_reactants = 0.0
    for r in rxn.get('reactants', []):
        sp = r['species']
        coeff = r.get('coeff', 1)
        if sp in DELTA_HF:
            sum_reactants += coeff * DELTA_HF[sp]
        else:
            missing.append(sp)

    if missing:
        return None, missing

    dH = sum_products - sum_reactants
    return dH, []


def classify_reaction(rxn):
    """Classify reaction for gas heating tier."""
    rtype = rxn['type']
    formula = rxn.get('formula', '')

    if rtype == 'ELECTRON_IMPACT':
        # Check if ionization (produces ion)
        for p in rxn.get('products', []):
            sp = p['species']
            if sp.endswith('+') or sp.endswith('-'):
                return 'IONIZATION'
        return 'EI_DISSOCIATION'

    if rtype == 'TE_DEPENDENT':
        # Check subtype
        subtype = rxn.get('subtype', '')
        if 'AT1' in subtype or 'attachment' in formula.lower():
            return 'ATTACHMENT'
        # Check for DR (e + ion → neutrals)
        has_e_react = any(r['species'] == 'e' for r in rxn.get('reactants', []))
        has_ion_react = any(r['species'].endswith('+') for r in rxn.get('reactants', []))
        if has_e_react and has_ion_react:
            return 'DR'
        # e + neutral attachment
        if has_e_react:
            return 'ATTACHMENT'
        return 'TE_OTHER'

    if rtype == 'ARRHENIUS':
        # Check for ion-ion recombination
        has_pos = any(r['species'].endswith('+') for r in rxn.get('reactants', []))
        has_neg = any(r['species'].endswith('-') for r in rxn.get('reactants', []))
        if has_pos and has_neg:
            return 'ION_ION_RECOMB'

        # Check for charge transfer (ion + neutral → ion + neutral)
        has_ion_r = any(r['species'].endswith('+') or r['species'].endswith('-')
                        for r in rxn.get('reactants', []))
        has_ion_p = any(p['species'].endswith('+') or p['species'].endswith('-')
                        for p in rxn.get('products', []))
        if has_ion_r and has_ion_p:
            return 'CHARGE_TRANSFER'

        # Detachment (negative ion → electron)
        if has_neg:
            has_e_prod = any(p['species'] == 'e' for p in rxn.get('products', []))
            if has_e_prod:
                return 'DETACHMENT'

        # Neutral-neutral
        return 'NEUTRAL'

    return rtype


def get_heating_tier(category, dH_kj):
    """
    Assign gas heating tier based on reaction category and ΔH magnitude.

    Tier 1 (HIGH):  Always heats gas — DR, ion-ion recomb, neutral exothermic
    Tier 2 (MEDIUM): Partially heats gas — charge transfer, detachment
    Tier 3 (LOW/NONE): Electron energy channel — EI, ionization, attachment
    """
    if category in ('DR', 'ION_ION_RECOMB'):
        return 'Tier 1 (HIGH)'
    if category == 'NEUTRAL' and dH_kj is not None and dH_kj < -50:
        return 'Tier 1 (HIGH)'
    if category == 'NEUTRAL' and dH_kj is not None and dH_kj < 0:
        return 'Tier 2 (MEDIUM)'
    if category in ('CHARGE_TRANSFER', 'DETACHMENT'):
        return 'Tier 2 (MEDIUM)'
    if category in ('EI_DISSOCIATION', 'IONIZATION', 'ATTACHMENT'):
        return 'Tier 3 (electron)'
    if category == 'NEUTRAL' and dH_kj is not None and dH_kj >= 0:
        return 'Tier 2 (endo)'
    return '—'


def main():
    # Load reactions
    rxn_path = Path(__file__).parent / 'input' / 'reactions.yaml'
    with open(rxn_path, 'r') as f:
        data = yaml.safe_load(f)

    reactions = data['reactions']
    print(f"Loaded {len(reactions)} reactions from {rxn_path}")

    # Compute ΔH for each reaction
    results = []
    missing_all = set()

    for rxn in reactions:
        rid = rxn['id']
        formula = rxn.get('formula', '???')
        rtype = rxn['type']
        ref = rxn.get('ref', '')
        dH, missing = compute_reaction_dH(rxn)
        category = classify_reaction(rxn)
        tier = get_heating_tier(category, dH)

        if missing:
            missing_all.update(missing)
            print(f"  WARNING R{rid}: Missing ΔHf° for {missing}")

        results.append({
            'id': rid,
            'formula': formula,
            'type': rtype,
            'category': category,
            'dH_kj': dH,
            'dH_eV': dH * EV_PER_KJMOL if dH is not None else None,
            'tier': tier,
            'ref': ref,
            'missing': missing,
        })

    if missing_all:
        print(f"\n  MISSING SPECIES (no ΔHf° data): {sorted(missing_all)}")

    # ── Write enthalpy_reference.md ──────────────────────────────────────────
    out_path = Path(__file__).parent / 'enthalpy_reference.md'
    with open(out_path, 'w') as f:
        f.write("# Enthalpy Reference — plasma0d_v2\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("All values at 298 K, standard state (1 atm).\n")
        f.write("ΔH computed via Hess's law: ΔH = Σ(ν·ΔHf°)_products − Σ(ν·ΔHf°)_reactants\n\n")
        f.write("---\n\n")

        # ── Part 1: Species ΔHf° table ──
        f.write("## Part 1: Species Formation Enthalpy (ΔHf°)\n\n")
        f.write("| Species | ΔHf° (kJ/mol) | ΔHf° (eV) | Source | Notes |\n")
        f.write("|---------|---------------|-----------|--------|-------|\n")

        # Group species by type
        type_order = [
            ('Element / Feed', ['e', 'CH4', 'CO2', 'N2', 'O2', 'H2', 'H2O']),
            ('Atoms & Radicals', ['H', 'O', 'C', 'N', 'N(2D)', 'OH', 'CH', 'CH2', 'CH3']),
            ('Excited States', ['N2(A)', 'N2(a1)']),
            ('C1 Oxygenated', ['CO', 'HCO', 'CH2O', 'CH2OH', 'CH3O', 'CH3OH',
                               'CH3O2', 'CH3OOH', 'CH2CO', 'CH3CO', 'CH3CHO']),
            ('HOx', ['HO2', 'H2O2', 'O3']),
            ('NOx', ['NO', 'NO2', 'NO3']),
            ('C2 Species', ['C2H2', 'C2H3', 'C2H4', 'C2H4OH', 'C2H5', 'C2H6',
                            'C2HO', 'C2H5O', 'C2H5O2', 'C2H5OH', 'C2H5OOH']),
            ('C3+ Species', ['C3H5', 'C3H6', 'C3H7', 'C3H8', 'C4H10']),
            ('Positive Ions', ['N2+', 'O2+', 'CO2+', 'CH4+', 'CH3+', 'CH5+',
                               'N4+', 'O4+', 'NO+', 'H3O+']),
            ('Negative Ions', ['O-', 'O2-']),
        ]

        source_notes = {
            'e': ('Convention', 'ΔHf°(e) ≡ 0'),
            'CH4': ('NIST-JANAF', ''),
            'CO2': ('NIST-JANAF', ''),
            'N2': ('Element', 'reference state'),
            'O2': ('Element', 'reference state'),
            'H2': ('Element', 'reference state'),
            'H2O': ('NIST-JANAF', ''),
            'H': ('NIST-JANAF', ''),
            'O': ('NIST-JANAF', ''),
            'C': ('NIST-JANAF', 'gas phase'),
            'N': ('NIST-JANAF', ''),
            'N(2D)': ('Computed', 'N + 2.38 eV'),
            'OH': ('NIST-JANAF', ''),
            'CH': ('NIST WebBook', ''),
            'CH2': ('NIST WebBook', 'triplet ground state'),
            'CH3': ('ATcT', ''),
            'N2(A)': ('Computed', 'N₂ + 6.17 eV'),
            'N2(a1)': ('Computed', 'N₂ + 8.40 eV'),
            'CO': ('NIST-JANAF', ''),
            'HCO': ('NIST WebBook', ''),
            'CH2O': ('NIST-JANAF', 'ATcT: −109.2'),
            'CH2OH': ('NIST WebBook', ''),
            'CH3O': ('NIST WebBook', ''),
            'CH3OH': ('NIST-JANAF', ''),
            'CH3O2': ('NIST WebBook', ''),
            'CH3OOH': ('NIST WebBook', ''),
            'CH2CO': ('NIST WebBook', '±2 kJ/mol, disputed'),
            'CH3CO': ('NIST WebBook', '±3 kJ/mol'),
            'CH3CHO': ('NIST-JANAF', ''),
            'HO2': ('ATcT', ''),
            'H2O2': ('NIST-JANAF', ''),
            'O3': ('NIST-JANAF', ''),
            'NO': ('NIST-JANAF', ''),
            'NO2': ('NIST-JANAF', ''),
            'NO3': ('NIST WebBook', ''),
            'C2H2': ('NIST-JANAF', ''),
            'C2H3': ('NIST WebBook', 'vinyl'),
            'C2H4': ('NIST-JANAF', ''),
            'C2H4OH': ('NIST WebBook', ''),
            'C2H5': ('NIST WebBook', 'ethyl'),
            'C2H6': ('NIST-JANAF', ''),
            'C2HO': ('NIST WebBook', 'ketenyl HCCO'),
            'C2H5O': ('NIST WebBook', 'ethoxy'),
            'C2H5O2': ('NIST WebBook', '±1.5 kJ/mol'),
            'C2H5OH': ('NIST-JANAF', ''),
            'C2H5OOH': ('ATcT', 'old NIST −210 is WRONG'),
            'C3H5': ('NIST WebBook', 'allyl'),
            'C3H6': ('NIST-JANAF', 'propene'),
            'C3H7': ('NIST WebBook', 'n-propyl'),
            'C3H8': ('NIST-JANAF', ''),
            'C4H10': ('NIST-JANAF', 'n-butane'),
            'N2+': ('Computed', 'N₂ + IP 15.58 eV'),
            'O2+': ('Computed', 'O₂ + IP 12.07 eV'),
            'CO2+': ('Computed', 'CO₂ + IP 13.78 eV'),
            'CH4+': ('Computed', 'CH₄ + IP 12.61 eV'),
            'CH3+': ('Computed', 'CH₃ + IP 9.84 eV'),
            'CH5+': ('Estimated', 'PA(CH₄) ≈ 5.7 eV'),
            'N4+': ('Estimated', 'appearance energy'),
            'O4+': ('Estimated', 'appearance energy'),
            'NO+': ('Computed', 'NO + IP 9.26 eV'),
            'H3O+': ('Estimated', 'PA(H₂O) ≈ 8.7 eV'),
            'O-': ('Computed', 'O − EA 1.461 eV'),
            'O2-': ('Computed', 'O₂ − EA 0.448 eV'),
        }

        for group_name, species_list in type_order:
            f.write(f"| **{group_name}** | | | | |\n")
            for sp in species_list:
                val = DELTA_HF[sp]
                val_ev = val * EV_PER_KJMOL
                src, note = source_notes.get(sp, ('—', ''))
                f.write(f"| {sp} | {val:+.1f} | {val_ev:+.3f} | {src} | {note} |\n")

        # ── Part 2: Reaction ΔH table ──
        f.write("\n---\n\n")
        f.write("## Part 2: Reaction Enthalpy (ΔH)\n\n")
        f.write("Convention: ΔH < 0 = exothermic (releases heat), ΔH > 0 = endothermic (absorbs heat)\n\n")
        f.write("**Category codes:**\n")
        f.write("- `EI_DISS` = Electron-impact dissociation\n")
        f.write("- `IONIZ` = Electron-impact ionization\n")
        f.write("- `DR` = Dissociative recombination (e + ion⁺ → neutrals)\n")
        f.write("- `CT` = Charge transfer (ion + neutral → ion + neutral)\n")
        f.write("- `II_REC` = Ion-ion recombination (A⁻ + B⁺ → neutrals)\n")
        f.write("- `AT` = Attachment (e + neutral → ion⁻)\n")
        f.write("- `DT` = Detachment (ion⁻ + neutral → neutrals + e)\n")
        f.write("- `NEUT` = Neutral-neutral reaction\n\n")

        f.write("**Gas heating tiers:**\n")
        f.write("- Tier 1 (HIGH): Energy deposited directly into gas — DR, ion-ion recomb, strongly exothermic neutral\n")
        f.write("- Tier 2 (MEDIUM): Partial gas heating — charge transfer, detachment, mildly exo/endothermic neutral\n")
        f.write("- Tier 3 (electron): Electron energy channel — EI dissociation, ionization, attachment\n\n")

        # Short category names for table
        cat_short = {
            'EI_DISSOCIATION': 'EI_DISS',
            'IONIZATION': 'IONIZ',
            'DR': 'DR',
            'CHARGE_TRANSFER': 'CT',
            'ION_ION_RECOMB': 'II_REC',
            'ATTACHMENT': 'AT',
            'DETACHMENT': 'DT',
            'NEUTRAL': 'NEUT',
            'TE_OTHER': 'TE_OTHER',
        }

        f.write("| ID | Category | Formula | ΔH (kJ/mol) | ΔH (eV) | Tier | Ref |\n")
        f.write("|---:|----------|---------|------------:|--------:|------|-----|\n")

        # Statistics
        n_exo = 0
        n_endo = 0
        n_missing = 0
        tier1_reactions = []

        for r in results:
            cat = cat_short.get(r['category'], r['category'])
            if r['dH_kj'] is not None:
                dH_str = f"{r['dH_kj']:+.1f}"
                eV_str = f"{r['dH_eV']:+.3f}"
                if r['dH_kj'] < 0:
                    n_exo += 1
                else:
                    n_endo += 1
                if r['tier'] == 'Tier 1 (HIGH)':
                    tier1_reactions.append(r)
            else:
                dH_str = "N/A"
                eV_str = "N/A"
                n_missing += 1

            f.write(f"| {r['id']} | {cat} | {r['formula']} | {dH_str} | {eV_str} | {r['tier']} | {r['ref']} |\n")

        # ── Part 3: Summary & Notes ──
        f.write("\n---\n\n")
        f.write("## Part 3: Summary & Notes\n\n")
        f.write("### Statistics\n\n")
        f.write(f"- Total reactions: {len(results)}\n")
        f.write(f"- Exothermic (ΔH < 0): {n_exo}\n")
        f.write(f"- Endothermic (ΔH ≥ 0): {n_endo}\n")
        f.write(f"- Missing data: {n_missing}\n\n")

        if missing_all:
            f.write("### Missing Species\n\n")
            f.write(f"Species without ΔHf° data: {', '.join(sorted(missing_all))}\n\n")

        f.write("### Tier 1 (HIGH gas heating) Reactions\n\n")
        f.write("These reactions deposit energy directly into the gas and should be prioritized\n")
        f.write("for the gas temperature enthalpy source term.\n\n")
        f.write("| ID | Formula | ΔH (kJ/mol) | ΔH (eV) | Category |\n")
        f.write("|---:|---------|------------:|--------:|----------|\n")
        for r in tier1_reactions:
            cat = cat_short.get(r['category'], r['category'])
            f.write(f"| {r['id']} | {r['formula']} | {r['dH_kj']:+.1f} | {r['dH_eV']:+.3f} | {cat} |\n")

        f.write("\n### Reliability Notes\n\n")
        f.write("1. **NIST-JANAF / ATcT**: Highest reliability (±0.5 kJ/mol typically)\n")
        f.write("2. **NIST WebBook**: Good reliability for most species (±1-3 kJ/mol)\n")
        f.write("3. **Computed (ions)**: Moderate reliability — depends on IP/EA accuracy (±5 kJ/mol)\n")
        f.write("4. **Estimated (cluster ions)**: Lower reliability — N₄⁺, O₄⁺, CH₅⁺, H₃O⁺ (±10-20 kJ/mol)\n\n")
        f.write("### Specific Warnings\n\n")
        f.write("- **CH₂CO (ketene)**: ΔHf° = −47.5 kJ/mol — disputed in literature (±2 kJ/mol)\n")
        f.write("- **CH₃CO (acetyl)**: ΔHf° = −12.0 kJ/mol — moderate uncertainty (±3 kJ/mol)\n")
        f.write("- **C₂H₅O₂ (ethylperoxy)**: ΔHf° = −28.4 kJ/mol — moderate uncertainty (±1.5 kJ/mol)\n")
        f.write("- **C₂H₅OOH**: ATcT value (−161.4 kJ/mol) used. Old NIST value (−210 kJ/mol) is WRONG.\n")
        f.write("- **CH₂O (formaldehyde)**: NIST-JANAF (−115.9) vs ATcT (−109.2) — 6.7 kJ/mol discrepancy.\n")
        f.write("  Using NIST-JANAF for consistency. ATcT is likely more accurate.\n\n")

        f.write("### Implementation Notes for gas_thermal.py\n\n")
        f.write("The gas temperature equation source term from reaction enthalpy:\n\n")
        f.write("```\n")
        f.write("Q_rxn = - Σ_j (ΔH_j × R_j)    [W/m³]\n")
        f.write("```\n\n")
        f.write("where:\n")
        f.write("- ΔH_j = reaction enthalpy [J/mol] (from this table × 1000)\n")
        f.write("- R_j = reaction rate [mol/(m³·s)]\n")
        f.write("- Negative sign: exothermic (ΔH < 0) → Q_rxn > 0 (heats gas)\n\n")
        f.write("**Important considerations:**\n")
        f.write("1. Tier 3 reactions (EI, ionization, attachment) should NOT be included — \n")
        f.write("   their energy comes from the electron energy equation, not gas thermal.\n")
        f.write("2. For DR reactions, the kinetic energy of products goes to gas heating.\n")
        f.write("3. For neutral reactions, full ΔH goes to gas.\n")
        f.write("4. For charge transfer, the exothermicity goes partly to internal energy\n")
        f.write("   of the product ion — simplified assumption: all to gas.\n")

    print(f"\nWrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"  Exothermic: {n_exo}, Endothermic: {n_endo}, Missing: {n_missing}")
    print(f"  Tier 1 (HIGH gas heating): {len(tier1_reactions)} reactions")


if __name__ == '__main__':
    main()
