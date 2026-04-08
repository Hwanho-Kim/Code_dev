"""Direct EEDF comparison: 300K vs 523K at same E/N and same ε̄."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.bolsig_parser import parse_bolsig_file, parse_eedf_file
from plasma0d_v2.boltzmann import MeanEnergyLUT, CrossSection
from plasma0d_v2.constants import QE, ME

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2', 'input')

b300 = parse_bolsig_file(os.path.join(base, 'BOLSIG_parameter/Condition1_300K.txt'))
b523 = parse_bolsig_file(os.path.join(base, 'BOLSIG_parameter/Condition1_523K.txt'))
e300 = parse_eedf_file(os.path.join(base, 'BOLSIG_EEDF/EEDF_300K.dat'))
e523 = parse_eedf_file(os.path.join(base, 'BOLSIG_EEDF/EEDF_523K.dat'))

# ============================================================
# [1] EEDF at SAME E/N
# ============================================================
print("="*100)
print("[1] EEDF COMPARISON AT SAME E/N")
print("    F₀(ε) [eV^(-3/2)] at selected energy points")
print("="*100)

en_targets = [0.1, 1.0, 10, 40, 60, 100]
for en_t in en_targets:
    idx300 = np.argmin(np.abs(b300.EN_Td - en_t))
    idx523 = np.argmin(np.abs(b523.EN_Td - en_t))
    
    blk300 = e300.blocks[idx300]
    blk523 = e523.blocks[idx523]
    
    eps300 = b300.mean_energy_eV[idx300]
    eps523 = b523.mean_energy_eV[idx523]
    
    print(f"\nE/N ≈ {en_t} Td  |  ε̄_300K={eps300:.4f} eV  |  ε̄_523K={eps523:.4f} eV  |  ε̄ ratio={eps523/eps300:.3f}")
    
    # Sample F₀ at a few energy points
    e_samples = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    print(f"  {'ε(eV)':>8} {'F₀_300K':>12} {'F₀_523K':>12} {'ratio':>8} {'diff%':>8}")
    print(f"  {'-'*55}")
    for e_s in e_samples:
        if e_s > blk300.energy_eV[-1] or e_s > blk523.energy_eV[-1]:
            continue
        f300 = np.interp(e_s, blk300.energy_eV, blk300.eedf)
        f523 = np.interp(e_s, blk523.energy_eV, blk523.eedf)
        r = f523/f300 if f300 > 1e-50 else 0
        d = (f523-f300)/f300*100 if f300 > 1e-50 else 0
        print(f"  {e_s:8.2f} {f300:12.4e} {f523:12.4e} {r:8.3f} {d:+8.1f}%")

# ============================================================
# [2] EEDF at SAME ε̄ (what the LUT actually maps to)
# ============================================================
print(f"\n{'='*100}")
print("[2] EEDF COMPARISON AT SAME ε̄ (= same LUT query point)")
print("    This is the relevant comparison for our ε̄-indexed model")
print("="*100)

eps_targets = [0.5, 1.0, 1.5, 2.0, 3.0]
for eps_t in eps_targets:
    # Find which E/N index gives closest ε̄
    idx300 = np.argmin(np.abs(b300.mean_energy_eV - eps_t))
    idx523 = np.argmin(np.abs(b523.mean_energy_eV - eps_t))
    
    en300 = b300.EN_Td[idx300]
    en523 = b523.EN_Td[idx523]
    eps300_actual = b300.mean_energy_eV[idx300]
    eps523_actual = b523.mean_energy_eV[idx523]
    
    blk300 = e300.blocks[idx300]
    blk523 = e523.blocks[idx523]
    
    print(f"\nε̄ ≈ {eps_t} eV")
    print(f"  300K: E/N={en300:.2f} Td, ε̄={eps300_actual:.4f} eV")
    print(f"  523K: E/N={en523:.2f} Td, ε̄={eps523_actual:.4f} eV")
    print(f"  E/N ratio: {en523/en300:.3f}")
    
    e_samples = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 13.0, 15.0]
    print(f"  {'ε(eV)':>8} {'F₀_300K':>12} {'F₀_523K':>12} {'ratio':>8} {'diff%':>8}")
    print(f"  {'-'*55}")
    for e_s in e_samples:
        if e_s > min(blk300.energy_eV[-1], blk523.energy_eV[-1]):
            continue
        f300 = np.interp(e_s, blk300.energy_eV, blk300.eedf)
        f523 = np.interp(e_s, blk523.energy_eV, blk523.eedf)
        if f300 < 1e-50 and f523 < 1e-50:
            continue
        r = f523/f300 if f300 > 1e-50 else float('inf')
        d = (f523-f300)/f300*100 if f300 > 1e-50 else float('inf')
        print(f"  {e_s:8.2f} {f300:12.4e} {f523:12.4e} {r:8.3f} {d:+8.1f}%")
    
    # Compute k for O2 attachment (key reaction) using both EEDFs
    # k = sqrt(2e/me) * ∫ σ(ε) * ε * F₀(ε) dε
    # Load O2 attachment cross section
    cs_path = os.path.join(base, 'cross_sections')
    # Find O2 attachment cross section file
    from plasma0d_v2.config import load_config
    from plasma0d_v2.reactions import ReactionSet
    from plasma0d_v2.species import SpeciesManager
    import io, contextlib
    
    # Just manually compute for the attachment cross section
    # O2 -> O- + O is the last custom cross section
    o2_att_file = os.path.join(cs_path, 'O2_attachment_O-.dat')
    if os.path.exists(o2_att_file):
        cs = CrossSection("O2 attach", o2_att_file, 3.6)
        with contextlib.redirect_stdout(io.StringIO()):
            cs.load()
        
        sigma300 = cs.interpolate(blk300.energy_eV)
        v300 = np.sqrt(2.0 * blk300.energy_eV * QE / ME)
        f_eps300 = blk300.eedf * np.sqrt(blk300.energy_eV)
        integrand300 = sigma300 * v300 * f_eps300
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        k300 = max(float(_trapz(integrand300, blk300.energy_eV)), 0.0)
        
        sigma523 = cs.interpolate(blk523.energy_eV)
        v523 = np.sqrt(2.0 * blk523.energy_eV * QE / ME)
        f_eps523 = blk523.eedf * np.sqrt(blk523.energy_eV)
        integrand523 = sigma523 * v523 * f_eps523
        k523 = max(float(_trapz(integrand523, blk523.energy_eV)), 0.0)
        
        r_k = k523/k300 if k300 > 0 else 0
        print(f"  O₂ attachment k: 300K={k300:.4e}, 523K={k523:.4e}, ratio={r_k:.4f}, diff={r_k-1:+.2%}")

# ============================================================
# [3] EEDF normalization check
# ============================================================
print(f"\n{'='*100}")
print("[3] EEDF NORMALIZATION CHECK: ∫F₀(ε)√ε dε should = 1")
print("="*100)
for label, eedf_data, bolsig in [("300K", e300, b300), ("523K", e523, b523)]:
    print(f"\n{label}:")
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    for i in [0, 10, 20, 30, 40]:
        if i >= len(eedf_data.blocks):
            continue
        blk = eedf_data.blocks[i]
        norm = float(_trapz(blk.eedf * np.sqrt(blk.energy_eV), blk.energy_eV))
        en = bolsig.EN_Td[i]
        eps = bolsig.mean_energy_eV[i]
        print(f"  R{i+1:2d}: E/N={en:8.2f} Td, ε̄={eps:.4f} eV, norm={norm:.6f}")
