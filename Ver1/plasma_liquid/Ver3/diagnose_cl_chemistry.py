#!/usr/bin/env python3
"""Diagnose why 92 saline reactions produce zero macro effect."""

import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import AQUEOUS_SPECIES, SALINE_SPECIES, ACID_BASE_PAIRS

# Load final concentrations from saline simulation
import pandas as pd
df = pd.read_csv(Path(__file__).parent / "saline_results.csv")
final = df.iloc[-1]

# Key concentrations from simulation
concs = {
    'OH': final['OH'],
    'O3': final['O3'],
    'H+': final['H+'],
    'Cl-': final['Cl-'],
    'HClO': final.get('HClO', 0),
    'NO2-': final['NO2-'],
    'NO3-': final['NO3-'],
    'H2O2': final['H2O2'],
}

print("=" * 70)
print("DIAGNOSTIC: Why do 92 saline reactions produce zero effect?")
print("=" * 70)

# 1. Check species name coverage
print("\n--- 1. SPECIES NAME COVERAGE ---")
all_model_species = set(AQUEOUS_SPECIES + SALINE_SPECIES)
speciated_species = set()
for total_name, (acid, base, pKa) in ACID_BASE_PAIRS.items():
    speciated_species.add(acid)
    speciated_species.add(base)
all_known = all_model_species | speciated_species | {'H2O'}

# Load all species used in saline reactions
saline_file = Path(__file__).parent / "reactions_saline.yaml"
with open(saline_file) as f:
    saline_data = yaml.safe_load(f)

reaction_species = set()
all_rxns = []
for section in ['reversible_reactions', 'irreversible_reactions']:
    if section not in saline_data:
        continue
    for rxn in saline_data[section]:
        if rxn is None:
            continue
        for sp in rxn.get('reactants', {}):
            reaction_species.add(sp)
        for sp in rxn.get('products', {}):
            reaction_species.add(sp)
        all_rxns.append(rxn)

missing = reaction_species - all_known
in_speciated_only = reaction_species & speciated_species - all_model_species
print(f"  Total species in saline reactions: {len(reaction_species)}")
print(f"  Species in model (direct + speciated): {len(all_known)}")
if missing:
    print(f"  *** MISSING SPECIES (reactions will be DEAD): {missing}")
else:
    print(f"  All species accounted for (no name mismatches)")
print(f"  Species resolved via speciation: {reaction_species & speciated_species}")

# 2. Entry point analysis: what activates Cl⁻?
print("\n--- 2. Cl⁻ ACTIVATION ENTRY POINTS ---")
print(f"  [OH]  = {concs['OH']:.3e} M  (literature: ~1e-12 M, gap: {1e-12/concs['OH']:.0f}x)")
print(f"  [O3]  = {concs['O3']:.3e} M  (literature: ~1e-7 M,  gap: {1e-7/concs['O3']:.0f}x)")
print(f"  [H+]  = {concs['H+']:.3e} M  (pH = {-np.log10(concs['H+']):.2f})")
print(f"  [Cl-] = {concs['Cl-']:.3e} M")

# S4: OH + Cl- -> HOCl-  (k=4.3e9)
rate_S4 = 4.3e9 * concs['OH'] * concs['Cl-']
print(f"\n  S4: OH + Cl- -> HOCl-")
print(f"    rate = 4.3e9 * {concs['OH']:.2e} * {concs['Cl-']:.3f} = {rate_S4:.3e} M/s")
print(f"    Total Cl- processed in 360s: {rate_S4 * 360 * 1e6:.1f} uM")

# S4 reverse: HOCl- -> OH + Cl-  (k=6.1e9)
K_S4 = 4.3e9 / 6.1e9  # = 0.705
HOCl_minus_ss = K_S4 * concs['OH'] * concs['Cl-']
print(f"  S4 reverse: k_b=6.1e9 s-1")
print(f"    [HOCl-]_ss = {HOCl_minus_ss:.3e} M")
print(f"    K_eq = k_f/k_b = {K_S4:.3f} (equilibrium FAVORS reactants)")

# 3. HOCl- branching
print("\n--- 3. HOCl- BRANCHING (where does HOCl- go?) ---")
k_back_S4 = 6.1e9  # HOCl- -> OH + Cl-
k_S5_rev = 2.1e10 * concs['H+']  # HOCl- + H+ -> Cl + H2O
k_S3_f = 1.0e4 * concs['Cl-']  # HOCl- + Cl- -> Cl2- + OH-
k_S7_rev = 3.0e10  # from S7, HOCl- + H+ -> HOClH... wait need to check

total_k = k_back_S4 + k_S5_rev + k_S3_f
print(f"  S4 reverse (back to OH+Cl-):   k = {k_back_S4:.3e} s-1  ({k_back_S4/total_k*100:.4f}%)")
print(f"  S5 reverse (HOCl-+H+ -> Cl):   k = {k_S5_rev:.3e} s-1  ({k_S5_rev/total_k*100:.4f}%)")
print(f"  S3 forward (HOCl-+Cl- -> Cl2-): k = {k_S3_f:.3e} s-1  ({k_S3_f/total_k*100:.6f}%)")

productive_fraction = (k_S5_rev + k_S3_f) / total_k
net_Cl_radical_production = rate_S4 * productive_fraction * 360  # mol/L over 360s
print(f"\n  Productive branching: {productive_fraction*100:.4f}%")
print(f"  Net Cl radical production in 360s: {net_Cl_radical_production*1e6:.3f} uM")

# 4. Compare to required changes
print("\n--- 4. REQUIRED vs AVAILABLE Cl CHEMISTRY FLUX ---")
print(f"  Required NO2- removal:   3 uM")
print(f"  Required extra NO3-:    39 uM")
print(f"  Required H2O2 removal:   6 uM")
print(f"  Available Cl radicals: {net_Cl_radical_production*1e6:.3f} uM")
print(f"  DEFICIT: {39 / max(net_Cl_radical_production*1e6, 1e-10):.0f}x too little for NO3- target")

# 5. What would fix it?
print("\n--- 5. WHAT [OH] IS NEEDED? ---")
for target_OH in [1e-13, 1e-12, 1e-11]:
    rate = 4.3e9 * target_OH * 0.154
    cl_flux = rate * productive_fraction * 360 * 1e6
    print(f"  [OH]={target_OH:.0e}: S4 rate={rate:.2e} M/s, net Cl radicals={cl_flux:.1f} uM")

# 6. Root cause chain
print("\n--- 6. ROOT CAUSE CHAIN ---")
print("""
  A/V = 13.3 (fitted, vs physical 100)
    |
    v
  O3 flux too low (O3 is liquid-side limited)
    |
    v
  [O3(aq)] ~ 5 nM (literature: 100-1000 nM)
    |
    v
  OH production too low (O3 decomposition -> OH)
    |
    v
  [OH] ~ 1e-14 M (literature: 1e-12 M) -- 100x GAP
    |
    v
  S4: OH + Cl- -> HOCl- rate too low
    |
    v  
  99.91% of HOCl- returns to OH + Cl- (unproductive)
    |
    v
  Net Cl radical production: ~0.3 uM (need ~40+ uM)
    |
    v
  92 saline reactions: ALL STARVED OF RADICALS -> ZERO EFFECT

  FUNDAMENTAL ISSUE: 0D model with uniform A/V cannot simultaneously
  provide enough O3 (needs high A/V) and avoid N2O5 overdose (needs low A/V).
  The fitted A/V=13.3 is a compromise that kills O3/OH/Cl chemistry.
""")
