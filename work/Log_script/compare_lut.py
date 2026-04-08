"""Compare 300K vs 523K BOLSIG+ LUT: ε̄ grids, rate coefficients, EEDF."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.bolsig_parser import parse_bolsig_file, parse_eedf_file
from plasma0d_v2.boltzmann import MeanEnergyLUT
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import NA
import io, contextlib

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2', 'input')

# === 1. Raw BOLSIG+ data comparison ===
print("="*100)
print("[1] RAW BOLSIG+ DATA COMPARISON")
print("="*100)

b300 = parse_bolsig_file(os.path.join(base, 'BOLSIG_parameter/Condition1_300K.txt'))
b523 = parse_bolsig_file(os.path.join(base, 'BOLSIG_parameter/Condition1_523K.txt'))

print(f"\n300K: {b300.n_EN} E/N points, ε̄ = [{b300.mean_energy_eV[0]:.4f}, {b300.mean_energy_eV[-1]:.2f}] eV")
print(f"523K: {b523.n_EN} E/N points, ε̄ = [{b523.mean_energy_eV[0]:.4f}, {b523.mean_energy_eV[-1]:.2f}] eV")
print(f"Rate labels: {b300.n_rates} (300K), {b523.n_rates} (523K)")

# Compare ε̄ at SAME E/N
print(f"\n{'E/N(Td)':>8} {'ε̄_300K':>10} {'ε̄_523K':>10} {'ratio':>8} {'diff%':>8}")
print("-"*55)
for i in range(0, min(b300.n_EN, b523.n_EN), 5):  # every 5th point
    en = b300.EN_Td[i]
    e300 = b300.mean_energy_eV[i]
    e523 = b523.mean_energy_eV[i]
    ratio = e523/e300 if e300 > 0 else 0
    diff_pct = (e523-e300)/e300*100 if e300 > 0 else 0
    print(f'{en:8.2f} {e300:10.4f} {e523:10.4f} {ratio:8.3f} {diff_pct:+8.1f}%')

# === 2. Rate coefficients at same E/N ===
print(f"\n{'='*100}")
print("[2] RATE COEFFICIENTS AT SAME E/N (first 5 reactions)")
print(f"{'='*100}")

# Pick a few E/N points relevant to our operating range
en_targets = [10, 20, 40, 60, 80, 100]
for en_t in en_targets:
    idx300 = np.argmin(np.abs(b300.EN_Td - en_t))
    idx523 = np.argmin(np.abs(b523.EN_Td - en_t))
    print(f"\nE/N ≈ {en_t} Td  (300K: {b300.EN_Td[idx300]:.1f} Td, ε̄={b300.mean_energy_eV[idx300]:.3f} eV | "
          f"523K: {b523.EN_Td[idx523]:.1f} Td, ε̄={b523.mean_energy_eV[idx523]:.3f} eV)")
    print(f"  {'Reaction':>35} {'k_300K':>12} {'k_523K':>12} {'ratio':>8}")
    for j in range(min(5, b300.n_rates)):
        k3 = b300.rate_coefficients[idx300, j]
        k5 = b523.rate_coefficients[idx523, j]
        r = k5/k3 if k3 > 0 else 0
        lbl = b300.rate_labels[j]
        print(f"  {str(lbl):>35} {k3:12.3e} {k5:12.3e} {r:8.3f}")

# === 3. Build LUTs and compare at SAME ε̄ ===
print(f"\n{'='*100}")
print("[3] LUT COMPARISON: k(ε̄) at SAME mean energy")
print("    This is what the solver actually queries")
print(f"{'='*100}")

# Build two LUTs
e300_eedf = parse_eedf_file(os.path.join(base, 'BOLSIG_EEDF/EEDF_300K.dat'))
e523_eedf = parse_eedf_file(os.path.join(base, 'BOLSIG_EEDF/EEDF_523K.dat'))

# Load reactions to get cross sections
cfg = load_config(os.path.join(base, '..', 'config.yaml'))
from plasma0d_v2.species import SpeciesManager
from plasma0d_v2.reactions import ReactionSet
sm = SpeciesManager()
sm.load_from_yaml(os.path.join(base, cfg['species_file']))
sm.finalize()
rxn = ReactionSet()
rxn.load_from_yaml(os.path.join(base, cfg['reactions_file']))
rxn.build(sm)
xsec_dir = os.path.join(base, cfg.get('cross_section_dir', 'cross_sections'))

with contextlib.redirect_stdout(io.StringIO()):
    lut300 = MeanEnergyLUT()
    lut300.load_cross_sections(xsec_dir, rxn.ei_reactions)
    lut300.build(b300, eedf_data=e300_eedf)

    lut523 = MeanEnergyLUT()
    lut523.load_cross_sections(xsec_dir, rxn.ei_reactions)
    lut523.build(b523, eedf_data=e523_eedf)

# Compare at target ε̄ values
eps_targets = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
print(f"\n{'ε̄(eV)':>8} | {'Reaction':>40} | {'k_300K':>12} {'k_523K':>12} {'ratio':>8} {'diff%':>8}")
print("-"*105)

# Get reaction labels from LUT cross sections
cs_names = [cs.name for cs in lut300.cross_sections]

for eps in eps_targets:
    # Check if within both LUT ranges
    if eps < max(lut300.eps_range[0], lut523.eps_range[0]):
        continue
    if eps > min(lut300.eps_range[1], lut523.eps_range[1]):
        continue

    k300, _ = lut300.get_rate_coefficients(eps)
    k523, _ = lut523.get_rate_coefficients(eps)

    for j in range(min(len(k300), len(cs_names))):
        if k300[j] < 1e-50 and k523[j] < 1e-50:
            continue
        r = k523[j]/k300[j] if k300[j] > 1e-50 else float('inf')
        d = (k523[j]-k300[j])/k300[j]*100 if k300[j] > 1e-50 else float('inf')
        name = cs_names[j][:40]
        # Only print first few and key reactions
        if j < 3 or 'O2' in name or 'ioniz' in name.lower() or 'attach' in name.lower():
            print(f'{eps:8.2f} | {name:>40} | {k300[j]:12.3e} {k523[j]:12.3e} {r:8.3f} {d:+8.1f}%')
    print()

# === 4. Transport comparison ===
print(f"{'='*100}")
print("[4] TRANSPORT: elastic power loss A21(ε̄)")
print(f"{'='*100}")
print(f"{'ε̄(eV)':>8} {'A21_300K':>12} {'A21_523K':>12} {'ratio':>8}")
print("-"*50)
for eps in eps_targets:
    if eps < max(lut300.eps_range[0], lut523.eps_range[0]):
        continue
    if eps > min(lut300.eps_range[1], lut523.eps_range[1]):
        continue
    t300 = lut300.get_transport(eps)
    t523 = lut523.get_transport(eps)
    r = t523.elastic_power_N / t300.elastic_power_N if t300.elastic_power_N > 0 else 0
    print(f'{eps:8.2f} {t300.elastic_power_N:12.3e} {t523.elastic_power_N:12.3e} {r:8.3f}')
