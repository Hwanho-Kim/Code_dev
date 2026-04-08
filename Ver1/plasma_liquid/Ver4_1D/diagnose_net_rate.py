#!/usr/bin/env python3
"""
Diagnostic: QSSA net effective rate vs brute-force reference.

Checks:
  1. dydt comparison (net effective vs brute-force all-reactions)
  2. Cl atom budget (d(total_Cl)/dt)
  3. Tagged vs untagged reaction listing
  4. Analytical S3-S9 formula verification
  5. QSSA species dydt residual
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from chemistry_1d import AqueousChemistry1D
from config_1d import DEFAULTS, ODE_CONFIG


# =====================================================================
# Setup
# =====================================================================

print("=" * 78)
print("QSSA Net Effective Rate Diagnostic")
print("=" * 78)

chem = AqueousChemistry1D(saline_mode=True)
NS = chem.n_species
idx = chem.species_idx
trace = DEFAULTS.trace_concentration

# Realistic saline y_cell
y_cell = np.full(NS, trace)
y_cell[idx['H+']] = 1e-4       # pH ~ 4
y_cell[idx['Cl-']] = 0.154     # 0.9% NaCl
y_cell[idx['OH']] = 1e-11
y_cell[idx['O3']] = 1e-7
y_cell[idx['NO2']] = 1e-9
y_cell[idx['NO3']] = 1e-10
y_cell[idx['H']] = 1e-15
y_cell[idx['O2']] = 2.5e-4
y_cell[idx['HONO_total']] = 1e-6
y_cell[idx['HONO2_total']] = 5e-5
y_cell[idx['H2O2_total']] = 1e-6
y_cell[idx['HO2_total']] = 1e-10
y_cell[idx['HClO_total']] = 1e-6
y_cell[idx['HClO2_total']] = 1e-10
y_cell[idx['Cl2']] = 1e-15
y_cell[idx['Cl3-']] = 1e-15
y_cell[idx['ClO2']] = 1e-15
y_cell[idx['ClO']] = 1e-15
y_cell[idx['ClO3']] = 1e-15
y_cell[idx['ClNO2']] = 1e-15

# Set QSSA species to non-trivial initial values for linearization
y_cell[idx['HOCl-']] = 1e-12
y_cell[idx['Cl2-']] = 1e-12
y_cell[idx['Cl']] = 1e-14
y_cell[idx['HOClH']] = 1e-14

print(f"  Species count: {NS}")
print(f"  Reactions: {len(chem._rxn_data)}")
print(f"  QSSA species: HOCl-, Cl2-, Cl, HOClH")
print(f"  y_cell[H+] = {y_cell[idx['H+']]:.2e} (pH={-np.log10(y_cell[idx['H+']]):.2f})")
print(f"  y_cell[Cl-] = {y_cell[idx['Cl-']]:.4f} M")
print()


# =====================================================================
# 1. Compare dydt: net effective rate vs brute-force reference
# =====================================================================

print("=" * 78)
print("1. dydt COMPARISON: net effective rate vs brute-force")
print("=" * 78)

# --- (A) Net effective rate (current code) ---
y_net = y_cell.copy()
dydt_net = chem.compute_rates(y_net)

# --- (B) Brute-force: manually iterate ALL 193 reactions, no QSSA skip ---
# First, apply QSSA to get consistent QSSA concentrations
y_bf = y_cell.copy()
chem.apply_qssa(y_bf)  # set QSSA species
speciated_bf = chem.speciate(y_bf)

dydt_bf = np.zeros(NS, dtype=np.float64)
for ri, rxn_d in enumerate(chem._rxn_data):
    rate = chem._compute_single_rate(rxn_d, y_bf, speciated_bf)
    if abs(rate) < 1e-30:
        continue
    # Apply to ALL targets (no skip)
    chem._apply_rate(rxn_d, rate, dydt_bf)

# OH- algebraic
if 'OH-' in idx:
    dydt_bf[idx['OH-']] = 0.0

# Sanitize
dydt_bf = np.nan_to_num(dydt_bf, nan=0.0, posinf=0.0, neginf=0.0)
dydt_bf = np.clip(dydt_bf, -chem.max_rate, chem.max_rate)

# Compare
species_names = chem.aqueous_species
print(f"\n{'Species':<18} {'dydt_net':>14} {'dydt_bf':>14} {'abs_diff':>12} {'rel_err':>10}")
print("-" * 78)
n_mismatch = 0
for i in range(NS):
    dn = dydt_net[i]
    db = dydt_bf[i]
    diff = abs(dn - db)
    if max(abs(dn), abs(db)) < 1e-15:
        continue
    denom = max(abs(dn), abs(db), 1e-30)
    rel = diff / denom
    flag = ""
    if rel > 0.01:
        flag = " <-- MISMATCH"
        n_mismatch += 1
    print(f"  {species_names[i]:<16} {dn:>14.6e} {db:>14.6e} {diff:>12.4e} {rel:>9.3e}{flag}")

print(f"\nTotal species with |dydt|>1e-15: {sum(1 for i in range(NS) if max(abs(dydt_net[i]), abs(dydt_bf[i])) > 1e-15)}")
print(f"Mismatches (rel_err > 1%): {n_mismatch}")


# =====================================================================
# 2. Cl atom budget
# =====================================================================

print("\n" + "=" * 78)
print("2. Cl ATOM BUDGET")
print("=" * 78)

# Cl count per species
CL_COUNT = {
    'Cl': 1, 'Cl-': 1, 'Cl2': 2, 'Cl2-': 2, 'Cl3-': 3,
    'HOCl-': 1, 'HOClH': 1,
    'HClO_total': 1, 'HClO2_total': 1,
    'HCl': 1,
    'Cl2O': 2, 'Cl2O2': 2, 'Cl2O3': 2, 'Cl2O4': 2, 'Cl2O5': 2, 'Cl2O6': 2,
    'ClO': 1, 'ClO2': 1, 'ClO3': 1,
    'ClNO2': 1,
    'ClO3-': 1, 'ClO4-': 1,
}

def cl_budget(dydt_vec):
    total = 0.0
    for sp, ncl in CL_COUNT.items():
        if sp in idx:
            total += ncl * dydt_vec[idx[sp]]
    return total

cl_net = cl_budget(dydt_net)
cl_bf = cl_budget(dydt_bf)
print(f"  d(total_Cl)/dt [net effective]: {cl_net:>14.6e} M/s")
print(f"  d(total_Cl)/dt [brute-force]:   {cl_bf:>14.6e} M/s")
print(f"  Difference:                     {abs(cl_net - cl_bf):>14.6e} M/s")

# Breakdown per Cl species
print(f"\n  {'Species':<18} {'n_Cl':>4} {'dydt_net*n':>14} {'dydt_bf*n':>14}")
print("  " + "-" * 54)
for sp, ncl in sorted(CL_COUNT.items()):
    if sp in idx:
        dn = ncl * dydt_net[idx[sp]]
        db = ncl * dydt_bf[idx[sp]]
        if max(abs(dn), abs(db)) > 1e-20:
            print(f"  {sp:<18} {ncl:>4} {dn:>14.6e} {db:>14.6e}")


# =====================================================================
# 3. Tagged vs untagged reactions
# =====================================================================

print("\n" + "=" * 78)
print("3. REACTION TAGGING (rxn_is_qssa)")
print("=" * 78)

qssa_species_set = {'HOCl-', 'Cl2-', 'Cl', 'HOClH'}
qssa_idx_set = {idx[sp] for sp in qssa_species_set if sp in idx}

rxn_is_qssa = chem._nb_rxn_is_qssa
n_tagged = int(rxn_is_qssa.sum())
print(f"  Total reactions: {len(chem._rxn_data)}")
print(f"  Tagged as QSSA: {n_tagged}")
print(f"  QSSA species idx: {qssa_idx_set}")
print()

# Check each reaction
untagged_qssa_involved = []
print(f"  {'#':>3} {'Label':<55} {'Type':<5} {'QSSA':>4}  Species")
print("  " + "-" * 100)
for ri, rxn in enumerate(chem.reactions):
    label = rxn.get('label', f'R{ri}')
    rxn_d = chem._rxn_data[ri]
    rtype = rxn_d['type']
    is_qssa = int(rxn_is_qssa[ri])

    # Check involvement
    involved_species = set()
    for sp_name, coeff, sp_idx in rxn_d['reactants']:
        involved_species.add(sp_name)
    for sp_name, coeff, sp_idx in rxn_d['products']:
        involved_species.add(sp_name)

    involves_qssa = bool(involved_species & qssa_species_set)

    flag = ""
    if involves_qssa and not is_qssa:
        flag = " <-- BUG: involves QSSA but NOT tagged"
        untagged_qssa_involved.append((ri, label))
    elif not involves_qssa and is_qssa:
        flag = " <-- SUSPICIOUS: tagged but no QSSA species"

    # Only print tagged or QSSA-involved reactions to keep output manageable
    if is_qssa or involves_qssa:
        species_str = ", ".join(sorted(involved_species))
        print(f"  {ri:>3} {label:<55} {rtype:<5} {is_qssa:>4}  {species_str}{flag}")

if untagged_qssa_involved:
    print(f"\n  ** WARNING: {len(untagged_qssa_involved)} reactions involve QSSA species but are NOT tagged:")
    for ri, label in untagged_qssa_involved:
        print(f"     [{ri}] {label}")
else:
    print(f"\n  All reactions involving QSSA species are correctly tagged.")


# =====================================================================
# 4. Analytical formula verification for S3-S9
# =====================================================================

print("\n" + "=" * 78)
print("4. S3-S9 ANALYTICAL FORMULA VERIFICATION")
print("=" * 78)

q = chem._qssa
y_v = y_bf.copy()  # use QSSA-applied values
spec = chem.speciate(y_v)

# Read QSSA concentrations
x1 = y_v[q['idx_HOCl-']]; x2 = y_v[q['idx_Cl2-']]
x3 = y_v[q['idx_Cl']];    x4 = y_v[q['idx_HOClH']]
OH_v = y_v[q['idx_OH']]; Clm_v = y_v[q['idx_Cl-']]
H_v = y_v[q['idx_H+']]; OHm_v = spec.get('OH-', trace)

k3f = q['S3_kf']; k3b = q['S3_kb']; k4f = q['S4_kf']; k4b = q['S4_kb']
k5f = q['S5_kf']; k5b = q['S5_kb']; k6f = q['S6_kf']; k6b = q['S6_kb']
k7f = q['S7_kf']; k7b = q['S7_kb']; k8f = q['S8_kf']; k8b = q['S8_kb']
k9f = q['S9_kf']; k9b = q['S9_kb']

# Analytical contribution
anal_Clm = (
    - k4f * OH_v * Clm_v
    + (k4b - (k3f + k9b) * Clm_v) * x1
    + ((k3b + k9f) * OHm_v + k6b + k8f) * x2
    - k6f * Clm_v * x3
    - k8b * Clm_v * x4
)
anal_OH = -k4f * OH_v * Clm_v + k4b * x1
anal_Hp = -(k5b + k7b) * H_v * x1 + k5f * x3 + k7f * x4

# Brute-force S3-S9 contribution
bf_Clm = 0.0; bf_OH = 0.0; bf_Hp = 0.0
bf_HOClm = 0.0; bf_Cl2m = 0.0; bf_Cl = 0.0; bf_HOClH = 0.0

# Find S3-S9 reaction indices
s3s9_labels = {'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'}
s3s9_indices = []
for ri, rxn in enumerate(chem.reactions):
    label = rxn.get('label', '')
    rxn_id = label.split(':')[0].strip()
    if rxn_id in s3s9_labels:
        s3s9_indices.append(ri)

print(f"  S3-S9 reaction indices: {s3s9_indices}")
print(f"  QSSA concentrations: x1(HOCl-)={x1:.4e}, x2(Cl2-)={x2:.4e}, x3(Cl)={x3:.4e}, x4(HOClH)={x4:.4e}")
print(f"  Co-reactants: OH={OH_v:.4e}, Cl-={Clm_v:.4e}, H+={H_v:.4e}, OH-={OHm_v:.4e}")
print()

for ri in s3s9_indices:
    rxn_d = chem._rxn_data[ri]
    label = chem.reactions[ri].get('label', f'R{ri}')
    rate = chem._compute_single_rate(rxn_d, y_v, spec)

    # Accumulate per-species contributions
    for sp_name, coeff, sp_idx in rxn_d['reactants']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx.get('Cl-', -2):
            bf_Clm -= coeff * rate
        elif target == idx.get('OH', -2):
            bf_OH -= coeff * rate
        elif target == idx.get('H+', -2):
            bf_Hp -= coeff * rate
        elif target == idx.get('HOCl-', -2):
            bf_HOClm -= coeff * rate
        elif target == idx.get('Cl2-', -2):
            bf_Cl2m -= coeff * rate
        elif target == idx.get('Cl', -2):
            bf_Cl -= coeff * rate
        elif target == idx.get('HOClH', -2):
            bf_HOClH -= coeff * rate
    for sp_name, coeff, sp_idx in rxn_d['products']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx.get('Cl-', -2):
            bf_Clm += coeff * rate
        elif target == idx.get('OH', -2):
            bf_OH += coeff * rate
        elif target == idx.get('H+', -2):
            bf_Hp += coeff * rate
        elif target == idx.get('HOCl-', -2):
            bf_HOClm += coeff * rate
        elif target == idx.get('Cl2-', -2):
            bf_Cl2m += coeff * rate
        elif target == idx.get('Cl', -2):
            bf_Cl += coeff * rate
        elif target == idx.get('HOClH', -2):
            bf_HOClH += coeff * rate

    print(f"  {label:<50} rate = {rate:>12.4e}")

print(f"\n  {'Target':<12} {'Analytical':>14} {'Brute-force':>14} {'Abs diff':>12} {'Rel err':>10}")
print("  " + "-" * 66)
for name, anal, bf in [
    ('Cl-', anal_Clm, bf_Clm),
    ('OH', anal_OH, bf_OH),
    ('H+', anal_Hp, bf_Hp),
]:
    diff = abs(anal - bf)
    denom = max(abs(anal), abs(bf), 1e-30)
    rel = diff / denom
    flag = " <-- MISMATCH" if rel > 1e-6 else ""
    print(f"  {name:<12} {anal:>14.6e} {bf:>14.6e} {diff:>12.4e} {rel:>9.3e}{flag}")

# Also show QSSA species S3-S9 budget (should be self-consistent)
print(f"\n  S3-S9 brute-force contributions to QSSA species (for reference):")
print(f"    HOCl-:  {bf_HOClm:>14.6e}")
print(f"    Cl2-:   {bf_Cl2m:>14.6e}")
print(f"    Cl:     {bf_Cl:>14.6e}")
print(f"    HOClH:  {bf_HOClH:>14.6e}")

# Check: S3-S9 Cl atom conservation
# S3: HOCl- + Cl- -> Cl2- + OH-  (Cl: 1+1 -> 2)  OK
# S4: OH + Cl- -> HOCl-          (Cl: 1 -> 1)      OK
# S5: Cl + H2O -> HOCl- + H+    (Cl: 1 -> 1)       OK
# S6: Cl + Cl- -> Cl2-           (Cl: 1+1 -> 2)     OK
# S7: HOClH -> HOCl- + H+        (Cl: 1 -> 1)       OK
# S8: Cl2- + H2O -> HOClH + Cl-  (Cl: 2 -> 1+1)    OK
# S9: Cl2- + OH- -> HOCl- + Cl-  (Cl: 2 -> 1+1)    OK
s39_cl = (bf_Clm + bf_HOClm + 2*bf_Cl2m + bf_Cl + bf_HOClH)
print(f"\n  S3-S9 total Cl budget (should be 0): {s39_cl:.6e}")


# =====================================================================
# 5. QSSA species dydt residual
# =====================================================================

print("\n" + "=" * 78)
print("5. QSSA SPECIES dydt RESIDUAL")
print("=" * 78)

qssa_names = ['HOCl-', 'Cl2-', 'Cl', 'HOClH']
print(f"\n  Net effective rate dydt (should be ~0 for QSSA species):")
for sp in qssa_names:
    i = idx[sp]
    print(f"    dydt[{sp:<8}] = {dydt_net[i]:>14.6e}")

print(f"\n  Brute-force dydt (full rates, not ~0 because QSSA not enforced as constraint):")
for sp in qssa_names:
    i = idx[sp]
    print(f"    dydt[{sp:<8}] = {dydt_bf[i]:>14.6e}")

# Also check: if QSSA is exact, the brute-force dydt of QSSA species
# should also be ~0 (production = destruction)
print(f"\n  QSSA residual check: brute-force dydt of QSSA species should be ~0")
print(f"  (This validates the QSSA concentration solve, not the net rate)")
for sp in qssa_names:
    i = idx[sp]
    val = dydt_bf[i]
    status = "OK" if abs(val) < 1e-6 else f"RESIDUAL={val:.4e}"
    print(f"    {sp:<8}: {status}")


# =====================================================================
# Summary
# =====================================================================

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)

# Key differences
print("\n  Key differences between net effective and brute-force:")
diffs = []
for i in range(NS):
    dn = dydt_net[i]
    db = dydt_bf[i]
    if max(abs(dn), abs(db)) < 1e-15:
        continue
    diff = abs(dn - db)
    denom = max(abs(dn), abs(db), 1e-30)
    rel = diff / denom
    if rel > 0.001:
        diffs.append((species_names[i], dn, db, rel))

if diffs:
    diffs.sort(key=lambda x: -x[3])
    for sp, dn, db, rel in diffs[:20]:
        print(f"    {sp:<18}: net={dn:>12.4e}, bf={db:>12.4e}, rel_err={rel:.4e}")
else:
    print("    None (all within 0.1%)")

print(f"\n  Cl budget difference: {abs(cl_net - cl_bf):.6e} M/s")
print(f"  Untagged QSSA reactions: {len(untagged_qssa_involved)}")
print(f"  S3-S9 analytical accuracy: see Section 4")


# =====================================================================
# 6. Deep dive: analytical formula term-by-term vs S3-S9 brute-force
# =====================================================================

print("\n" + "=" * 78)
print("6. ANALYTICAL FORMULA TERM-BY-TERM DECOMPOSITION")
print("=" * 78)

# The analytical formula for dydt[Cl-] from S3-S9 is:
#   -k4f*OH*Cl- + (k4b - (k3f+k9b)*Cl-)*x1 + ((k3b+k9f)*OH- + k6b + k8f)*x2
#   - k6f*Cl-*x3 - k8b*Cl-*x4

# Let's compute each term
t1 = -k4f * OH_v * Clm_v         # S4 forward: OH + Cl- -> HOCl-
t2 = k4b * x1                    # S4 backward: HOCl- -> OH + Cl-
t3 = -(k3f + k9b) * Clm_v * x1  # S3 forward + S9 backward consuming Cl-
t4 = (k3b + k9f) * OHm_v * x2   # S3 backward + S9 forward producing Cl-
t5 = k6b * x2                    # S6 backward: Cl2- -> Cl + Cl-
t6 = k8f * x2                    # S8 forward: Cl2- -> HOClH + Cl-
t7 = -k6f * Clm_v * x3           # S6 forward: Cl + Cl- -> Cl2-
t8 = -k8b * Clm_v * x4           # S8 backward: HOClH + Cl- -> Cl2-

print(f"\n  Analytical dydt[Cl-] term decomposition:")
print(f"    t1 = -k4f*OH*Cl-      = {t1:>14.6e}  (S4 fwd)")
print(f"    t2 = k4b*x1           = {t2:>14.6e}  (S4 bwd)")
print(f"    t3 = -(k3f+k9b)*Cl-*x1= {t3:>14.6e}  (S3f+S9b)")
print(f"    t4 = (k3b+k9f)*OH-*x2 = {t4:>14.6e}  (S3b+S9f)")
print(f"    t5 = k6b*x2           = {t5:>14.6e}  (S6 bwd)")
print(f"    t6 = k8f*x2           = {t6:>14.6e}  (S8 fwd)")
print(f"    t7 = -k6f*Cl-*x3      = {t7:>14.6e}  (S6 fwd)")
print(f"    t8 = -k8b*Cl-*x4      = {t8:>14.6e}  (S8 bwd)")
print(f"    SUM                   = {sum([t1,t2,t3,t4,t5,t6,t7,t8]):>14.6e}")
print(f"    Expected (anal_Clm)   = {anal_Clm:>14.6e}")

# Now compare each S3-S9 rate individually with expected
print(f"\n  Brute-force S3-S9 individual Cl- contributions:")

for ri in s3s9_indices:
    rxn_d = chem._rxn_data[ri]
    label = chem.reactions[ri].get('label', f'R{ri}')
    rate = chem._compute_single_rate(rxn_d, y_v, spec)

    clm_contrib = 0.0
    for sp_name, coeff, sp_idx in rxn_d['reactants']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx.get('Cl-', -2):
            clm_contrib -= coeff * rate
    for sp_name, coeff, sp_idx in rxn_d['products']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx.get('Cl-', -2):
            clm_contrib += coeff * rate

    print(f"    {label:<50} Cl- contrib = {clm_contrib:>14.6e}")

# Key diagnostic: what does the analytical formula assume about H_idx?
print(f"\n  H_idx in Numba kernel points to: idx[H+]={idx['H+']}")
print(f"  qssa_idx[5] (H+) = {chem._nb_qssa_idx[5]}")
print(f"  qssa_idx[9] (H)  = {chem._nb_qssa_idx[9]}")

# The analytical formula line 294 uses H_idx (= species_idx['H+'])
# but the Net rate code uses i_H_q = qssa_idx[9] which is idx_H (H atom)!
# Wait - checking again:
# Line 270: i_H_q = qssa_idx[9]  -- this is the H atom index
# But line 294: dydt[H_idx] += ...  -- H_idx is the H+ index (correct target)
# The H variable on line 91 is yc[H_idx] = H+ concentration (correct)
# So in the Numba kernel, the formula is correct.
# But in Python compute_rates, line 738: H_v = y_cell[q['idx_H+']]
# which is also correct.

# Let's check if the mismatch comes from mass-balance rate limiting
print(f"\n  Checking mass-balance rate limiting effect:")
for ri in s3s9_indices:
    rxn_d = chem._rxn_data[ri]
    label = chem.reactions[ri].get('label', f'R{ri}')

    # Compute rate WITHOUT limiting
    if rxn_d['type'] == 'rev':
        rate_f = rxn_d['k_f']
        rate_b = rxn_d['k_b']
        for sp_name, coeff, sp_idx in rxn_d['reactants']:
            conc = chem._get_conc(sp_name, sp_idx, y_v, spec)
            rate_f *= conc ** min(coeff, 3)
        for sp_name, coeff, sp_idx in rxn_d['products']:
            conc = chem._get_conc(sp_name, sp_idx, y_v, spec)
            rate_b *= conc ** min(coeff, 3)
        rate_raw = rate_f - rate_b
    else:
        rate_raw = rxn_d['k']
        for sp_name, coeff, sp_idx in rxn_d['reactants']:
            conc = chem._get_conc(sp_name, sp_idx, y_v, spec)
            rate_raw *= conc ** min(coeff, 3)

    rate_limited = chem._compute_single_rate(rxn_d, y_v, spec)
    if abs(rate_raw) > 1e-30 and abs(rate_raw - rate_limited) / abs(rate_raw) > 1e-6:
        print(f"    {label:<50} raw={rate_raw:>12.4e} limited={rate_limited:>12.4e}")
    else:
        print(f"    {label:<50} raw={rate_raw:>12.4e} (no limiting)")

# Check S3-S9 rates from analytical vs individual
print(f"\n  Rate constants (from _qssa dict):")
for sid in ['S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']:
    kf = q[f'{sid}_kf']
    kb = q[f'{sid}_kb']
    print(f"    {sid}: kf={kf:.4e}, kb={kb:.4e}")

print(f"\n  Rate constants (from _rxn_data):")
for ri in s3s9_indices:
    rxn_d = chem._rxn_data[ri]
    label = chem.reactions[ri].get('label', '')
    if rxn_d['type'] == 'rev':
        print(f"    {label.split(':')[0]}: kf={rxn_d['k_f']:.4e}, kb={rxn_d['k_b']:.4e}")


# =====================================================================
# 7. QSSA residual: why is Cl2- residual so large?
# =====================================================================

print("\n" + "=" * 78)
print("7. QSSA RESIDUAL ANALYSIS (Cl2-)")
print("=" * 78)

# List all reactions contributing to Cl2- in brute-force
print(f"\n  Reactions contributing to dydt[Cl2-] in brute-force:")
cl2m_idx = idx['Cl2-']
for ri, rxn_d in enumerate(chem._rxn_data):
    rate = chem._compute_single_rate(rxn_d, y_v, spec)
    if abs(rate) < 1e-30:
        continue

    contrib = 0.0
    for sp_name, coeff, sp_idx in rxn_d['reactants']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == cl2m_idx:
            contrib -= coeff * rate
    for sp_name, coeff, sp_idx in rxn_d['products']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == cl2m_idx:
            contrib += coeff * rate

    if abs(contrib) > 1e-15:
        label = chem.reactions[ri].get('label', f'R{ri}')
        is_q = "QSSA" if rxn_is_qssa[ri] else "    "
        print(f"    [{ri:>3}] {is_q} {label:<55} {contrib:>14.6e}")

print(f"\n  Total dydt[Cl2-] brute-force: {dydt_bf[cl2m_idx]:.6e}")
print(f"  This should be ~0 if QSSA solve is exact.")
print(f"  Large residual means QSSA solve is approximate (2-pass Picard).")


# =====================================================================
# 8. Net effective rate: analytical + irreversible loop decomposition
# =====================================================================

print("\n" + "=" * 78)
print("8. NET EFFECTIVE RATE: ANALYTICAL + IRREVERSIBLE LOOP FOR Cl-")
print("=" * 78)

# The net effective rate has two parts:
# (a) S3-S9 analytical formula
# (b) S23-S69 irreversible reactions (applied to non-QSSA targets)

# Part (a) already computed as anal_Clm
print(f"\n  (a) S3-S9 analytical:    dydt[Cl-] = {anal_Clm:>14.6e}")

# Part (b): irreversible tagged reactions, non-QSSA targets
irr_Clm = 0.0
irr_details = []
for ri, rxn_d in enumerate(chem._rxn_data):
    if rxn_is_qssa[ri] == 0:
        continue
    if rxn_d['type'] == 'rev':
        continue  # S3-S9 already handled
    rate = chem._compute_single_rate(rxn_d, y_v, spec)
    if abs(rate) < 1e-30:
        continue

    contrib = 0.0
    for sp_name, coeff, sp_idx in rxn_d['reactants']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx['Cl-'] and target not in {idx[s] for s in qssa_species_set}:
            contrib -= coeff * rate
    for sp_name, coeff, sp_idx in rxn_d['products']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx['Cl-'] and target not in {idx[s] for s in qssa_species_set}:
            contrib += coeff * rate

    if abs(contrib) > 1e-20:
        irr_Clm += contrib
        label = chem.reactions[ri].get('label', f'R{ri}')
        irr_details.append((label, contrib))

print(f"  (b) Irreversible QSSA:   dydt[Cl-] = {irr_Clm:>14.6e}")
for label, c in irr_details:
    print(f"      {label:<55} {c:>14.6e}")

# Part (c): non-QSSA reactions (main loop)
nq_Clm = 0.0
for ri, rxn_d in enumerate(chem._rxn_data):
    if rxn_is_qssa[ri] > 0:
        continue
    rate = chem._compute_single_rate(rxn_d, y_v, spec)
    if abs(rate) < 1e-30:
        continue
    for sp_name, coeff, sp_idx in rxn_d['reactants']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx['Cl-']:
            nq_Clm -= coeff * rate
    for sp_name, coeff, sp_idx in rxn_d['products']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx['Cl-']:
            nq_Clm += coeff * rate

print(f"  (c) Non-QSSA reactions:  dydt[Cl-] = {nq_Clm:>14.6e}")
print(f"\n  Total (a+b+c):           dydt[Cl-] = {anal_Clm + irr_Clm + nq_Clm:>14.6e}")
print(f"  dydt_net[Cl-]:           dydt[Cl-] = {dydt_net[idx['Cl-']]:>14.6e}")
print(f"  dydt_bf[Cl-]:            dydt[Cl-] = {dydt_bf[idx['Cl-']]:>14.6e}")

# Brute-force decomposition for comparison
bf_q_Clm = 0.0
bf_nq_Clm = 0.0
for ri, rxn_d in enumerate(chem._rxn_data):
    rate = chem._compute_single_rate(rxn_d, y_v, spec)
    if abs(rate) < 1e-30:
        continue
    contrib = 0.0
    for sp_name, coeff, sp_idx in rxn_d['reactants']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx['Cl-']:
            contrib -= coeff * rate
    for sp_name, coeff, sp_idx in rxn_d['products']:
        target = chem._get_target_idx(sp_name, sp_idx)
        if target == idx['Cl-']:
            contrib += coeff * rate
    if rxn_is_qssa[ri] > 0:
        bf_q_Clm += contrib
    else:
        bf_nq_Clm += contrib

print(f"\n  BF decomposition:")
print(f"    QSSA-tagged reactions:  {bf_q_Clm:>14.6e}")
print(f"    Non-QSSA reactions:     {bf_nq_Clm:>14.6e}")
print(f"    Total:                  {bf_q_Clm + bf_nq_Clm:>14.6e}")
print(f"\n  Mismatch in QSSA-tagged: anal+irr = {anal_Clm + irr_Clm:>14.6e} vs bf = {bf_q_Clm:>14.6e}")
print(f"  Difference: {anal_Clm + irr_Clm - bf_q_Clm:>14.6e}")
print()
