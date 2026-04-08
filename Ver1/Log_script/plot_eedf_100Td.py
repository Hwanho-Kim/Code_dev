"""Plot EEDF F₀(ε) at E/N ≈ 100 Td for 300K and 523K."""
import sys, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.bolsig_parser import parse_bolsig_file, parse_eedf_file

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2', 'input')

b300 = parse_bolsig_file(os.path.join(base, 'BOLSIG_parameter/Condition1_300K.txt'))
b523 = parse_bolsig_file(os.path.join(base, 'BOLSIG_parameter/Condition1_523K.txt'))
e300 = parse_eedf_file(os.path.join(base, 'BOLSIG_EEDF/EEDF_300K.dat'))
e523 = parse_eedf_file(os.path.join(base, 'BOLSIG_EEDF/EEDF_523K.dat'))

# Find E/N closest to 100 Td
idx300 = int(np.argmin(np.abs(b300.EN_Td - 100)))
idx523 = int(np.argmin(np.abs(b523.EN_Td - 100)))

blk300 = e300.blocks[idx300]
blk523 = e523.blocks[idx523]

en300 = b300.EN_Td[idx300]
en523 = b523.EN_Td[idx523]
eps300 = b300.mean_energy_eV[idx300]
eps523 = b523.mean_energy_eV[idx523]

print(f"300K: E/N = {en300:.2f} Td, ε̄ = {eps300:.4f} eV, EEDF points = {len(blk300.energy_eV)}")
print(f"523K: E/N = {en523:.2f} Td, ε̄ = {eps523:.4f} eV, EEDF points = {len(blk523.energy_eV)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: linear scale
ax = axes[0]
ax.plot(blk300.energy_eV, blk300.eedf, 'b-', lw=1.8, label=f'300 K  (ε̄={eps300:.3f} eV)')
ax.plot(blk523.energy_eV, blk523.eedf, 'r--', lw=1.8, label=f'523 K  (ε̄={eps523:.3f} eV)')
ax.set_xlabel('Electron energy ε  [eV]', fontsize=13)
ax.set_ylabel('F₀(ε)  [eV⁻³ˡ²]', fontsize=13)
ax.set_title(f'EEDF at E/N ≈ 100 Td  (linear)', fontsize=14)
ax.legend(fontsize=12)
ax.set_xlim(0, 20)
ax.grid(True, alpha=0.3)

# Right: log scale
ax = axes[1]
ax.semilogy(blk300.energy_eV, blk300.eedf, 'b-', lw=1.8, label=f'300 K  (ε̄={eps300:.3f} eV)')
ax.semilogy(blk523.energy_eV, blk523.eedf, 'r--', lw=1.8, label=f'523 K  (ε̄={eps523:.3f} eV)')
ax.set_xlabel('Electron energy ε  [eV]', fontsize=13)
ax.set_ylabel('F₀(ε)  [eV⁻³ˡ²]', fontsize=13)
ax.set_title(f'EEDF at E/N ≈ 100 Td  (log)', fontsize=14)
ax.legend(fontsize=12)
ax.set_xlim(0, 25)
ax.set_ylim(1e-12, 1)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eedf_100Td_comparison.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
