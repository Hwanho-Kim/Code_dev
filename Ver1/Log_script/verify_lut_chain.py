"""Full-chain verification: LUT load → k_table → simulation query.

Checks:
  1. BOLSIG+ files correctly loaded (tgas, eps_range different)
  2. EEDF files correctly loaded (different F₀)
  3. _k_table values differ between 300K and 523K LUTs
  4. Run two simulations (T=523K with 300K LUT vs 523K LUT)
     and log eps_mean, Te, n_e, key rates at EVERY output point
"""
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.bolsig_parser import parse_bolsig_file, parse_eedf_file
from plasma0d_v2.boltzmann import MeanEnergyLUT
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
input_dir = os.path.join(base_dir, 'input')

# ================================================================
# [1] RAW BOLSIG+ DATA COMPARISON
# ================================================================
print("=" * 90)
print("[1] RAW BOLSIG+ DATA")
print("=" * 90)

b300 = parse_bolsig_file(os.path.join(input_dir, 'BOLSIG_parameter/Condition1_300K.txt'))
b523 = parse_bolsig_file(os.path.join(input_dir, 'BOLSIG_parameter/Condition1_523K.txt'))

print(f"\n300K: tgas={b300.tgas_K}, n_EN={len(b300.EN_Td)}, "
      f"eps=[{b300.mean_energy_eV[0]:.6f}, {b300.mean_energy_eV[-1]:.4f}]")
print(f"523K: tgas={b523.tgas_K}, n_EN={len(b523.EN_Td)}, "
      f"eps=[{b523.mean_energy_eV[0]:.6f}, {b523.mean_energy_eV[-1]:.4f}]")

# Check they're actually different objects
print(f"\nSame object? b300 is b523 = {b300 is b523}")
print(f"eps arrays equal? {np.array_equal(b300.mean_energy_eV, b523.mean_energy_eV)}")

# ================================================================
# [2] EEDF DATA COMPARISON
# ================================================================
print(f"\n{'=' * 90}")
print("[2] RAW EEDF DATA")
print("=" * 90)

e300 = parse_eedf_file(os.path.join(input_dir, 'BOLSIG_EEDF/EEDF_300K.dat'))
e523 = parse_eedf_file(os.path.join(input_dir, 'BOLSIG_EEDF/EEDF_523K.dat'))

print(f"\n300K EEDF: {e300.n_blocks} blocks, E/N=[{e300.EN_Td[0]:.4f}, {e300.EN_Td[-1]:.1f}] Td")
print(f"523K EEDF: {e523.n_blocks} blocks, E/N=[{e523.EN_Td[0]:.4f}, {e523.EN_Td[-1]:.1f}] Td")

# Compare F₀ at block 0 (E/N=0.1 Td)
print(f"\nBlock 0 (E/N≈0.1 Td): F₀(0.01eV) 300K={e300.blocks[0].eedf[0]:.6e}, "
      f"523K={e523.blocks[0].eedf[0]:.6e}")
# Compare at high E/N block (idx ~37, E/N≈100 Td)
idx_hi = 37
print(f"Block {idx_hi} (E/N≈{b300.EN_Td[idx_hi]:.1f} Td): F₀(0.01eV) "
      f"300K={e300.blocks[idx_hi].eedf[0]:.6e}, 523K={e523.blocks[idx_hi].eedf[0]:.6e}")

# ================================================================
# [3] BUILD TWO LUTs AND COMPARE _k_table
# ================================================================
print(f"\n{'=' * 90}")
print("[3] LUT _k_table COMPARISON (300K vs 523K)")
print("=" * 90)

from plasma0d_v2.reactions import ReactionSet
from plasma0d_v2.species import SpeciesManager

cfg = load_config(os.path.join(base_dir, 'config.yaml'))
sm = SpeciesManager()
sm.load_from_yaml(os.path.join(input_dir, cfg['species_file']))
sm.finalize()
rxn = ReactionSet()
rxn.load_from_yaml(os.path.join(input_dir, cfg['reactions_file']))
with contextlib.redirect_stdout(io.StringIO()):
    rxn.build(sm)

xsec_dir = os.path.join(input_dir, 'cross_sections')

# Build LUT with 300K data
lut300 = MeanEnergyLUT()
with contextlib.redirect_stdout(io.StringIO()):
    lut300.load_cross_sections(xsec_dir, rxn.ei_reactions)
lut300.build(b300, eedf_data=e300)

# Build LUT with 523K data
lut523 = MeanEnergyLUT()
with contextlib.redirect_stdout(io.StringIO()):
    lut523.load_cross_sections(xsec_dir, rxn.ei_reactions)
lut523.build(b523, eedf_data=e523)

print(f"\nlut300: tgas={lut300.tgas_K}, eedf_used={lut300._eedf_used}, "
      f"eps=[{lut300._eps_grid[0]:.6f}, {lut300._eps_grid[-1]:.4f}], "
      f"k_table shape={lut300._k_table.shape}")
print(f"lut523: tgas={lut523.tgas_K}, eedf_used={lut523._eedf_used}, "
      f"eps=[{lut523._eps_grid[0]:.6f}, {lut523._eps_grid[-1]:.4f}], "
      f"k_table shape={lut523._k_table.shape}")

# Compare k_table at matching ε̄ points
# Find reaction names
rxn_names = [r.formula for r in rxn.ei_reactions]
n_rxns = len(rxn_names)

# Query at specific ε̄ values relevant to simulation
eps_queries = [0.5, 1.0, 1.5, 2.0, 3.0]
print(f"\nRate coefficients k [m³/s] at specific ε̄ values:")
print(f"{'ε̄(eV)':>8} | {'Reaction':>45} | {'k_300K':>12} {'k_523K':>12} {'ratio':>8} {'diff%':>8}")
print("-" * 105)

for eps_q in eps_queries:
    k300, _ = lut300.get_rate_coefficients(eps_q)
    k523, _ = lut523.get_rate_coefficients(eps_q)
    for j in range(min(n_rxns, len(k300))):
        if k300[j] < 1e-50 and k523[j] < 1e-50:
            continue
        r = k523[j] / k300[j] if k300[j] > 1e-50 else float('inf')
        d = (k523[j] - k300[j]) / k300[j] * 100 if k300[j] > 1e-50 else float('inf')
        print(f"{eps_q:8.2f} | {rxn_names[j]:>45} | {k300[j]:12.4e} {k523[j]:12.4e} {r:8.4f} {d:+8.2f}%")
    if eps_q < eps_queries[-1]:
        print()

# ================================================================
# [4] COMPARE _k_table RAW ARRAYS (are they the same object?)
# ================================================================
print(f"\n{'=' * 90}")
print("[4] _k_table IDENTITY AND DIFF")
print("=" * 90)

print(f"\nSame object? lut300._k_table is lut523._k_table = {lut300._k_table is lut523._k_table}")
print(f"Arrays exactly equal? {np.array_equal(lut300._k_table, lut523._k_table)}")

# Element-wise comparison for first few reactions at all eps grid points
for j in range(min(5, n_rxns)):
    k3 = lut300._k_table[:, j]
    k5 = lut523._k_table[:, j]
    # Only compare where both > 0
    mask = (k3 > 1e-50) & (k5 > 1e-50)
    if mask.sum() > 0:
        ratios = k5[mask] / k3[mask]
        print(f"  Rxn {j} ({rxn_names[j][:40]:40s}): "
              f"max_ratio={ratios.max():.6f}, min_ratio={ratios.min():.6f}, "
              f"mean_diff={((ratios-1)*100).mean():+.3f}%")

# ================================================================
# [5] RUN SIMULATION: T=523K with 300K LUT vs 523K LUT
# ================================================================
print(f"\n{'=' * 90}")
print("[5] SIMULATION COMPARISON: T=523K, 300K-LUT vs 523K-LUT")
print("    (full precision, 15 significant digits)")
print("=" * 90)

def run_sim(bolsig_file, eedf_file, label):
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))
    cfg['V_eff'] = 1.6e-6
    cfg['reactor']['volume'] = 100e-6
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = 5.0
    cfg['flow']['Q_slm'] = 0.4
    cfg['T_wall'] = 523.0
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = 523.0
    cfg['bolsig_files'] = [bolsig_file]
    cfg['eedf_files'] = [eedf_file]
    
    Q_actual = 0.4 * (523.0 / T_STP) * (P_STP / 101325.0) / 60000.0
    tau_est = 100e-6 / Q_actual
    t_end = min(max(3.0, 1.5 * tau_est), 15.0)
    cfg['solver'] = {
        't_end': t_end, 'n_points': 100, 'method': 'BDF',
        'rtol': 1e-5, 'atol': 1e-10, 'max_step': 5e-4, 'constrained': False
    }
    
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
        scfg = cfg['solver']
        result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                              rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])
    
    setup_log = captured.getvalue()
    
    return result, solver, setup_log

print("\nRunning with 300K LUT...")
r300, s300, log300 = run_sim('BOLSIG_parameter/Condition1_300K.txt', 'BOLSIG_EEDF/EEDF_300K.dat', '300K')
print("Running with 523K LUT...")
r523, s523, log523 = run_sim('BOLSIG_parameter/Condition1_523K.txt', 'BOLSIG_EEDF/EEDF_523K.dat', '523K')

# Verify LUTs are different
print(f"\nLUT verification:")
print(f"  300K LUT tgas: {s300.lut.tgas_K}")
print(f"  523K LUT tgas: {s523.lut.tgas_K}")
print(f"  300K LUT eps_range: [{s300.lut._eps_grid[0]:.8f}, {s300.lut._eps_grid[-1]:.6f}]")
print(f"  523K LUT eps_range: [{s523.lut._eps_grid[0]:.8f}, {s523.lut._eps_grid[-1]:.6f}]")
print(f"  300K LUT eedf_used: {s300.lut._eedf_used}")
print(f"  523K LUT eedf_used: {s523.lut._eedf_used}")
print(f"  Same LUT object? {s300.lut is s523.lut}")
print(f"  Same _k_table? {np.array_equal(s300.lut._k_table, s523.lut._k_table)}")

# Compare results at final time with FULL precision
n_sp = s300.sm.n_species
y300 = r300.y[:, -1]
y523 = r523.y[:, -1]

c300 = y300[:n_sp]; c523_f = y523[:n_sp]
ne300 = c300[0] * NA; ne523 = c523_f[0] * NA
ne_eps300 = y300[s300.sm.idx_energy]; ne_eps523 = y523[s523.sm.idx_energy]
Tg300 = y300[s300.sm.idx_Tgas]; Tg523 = y523[s523.sm.idx_Tgas]
eps300 = ne_eps300 / ne300 if ne300 > 1 else 1.0
eps523 = ne_eps523 / ne523 if ne523 > 1 else 1.0

print(f"\nFinal state comparison (T_gas_init=523K):")
print(f"  {'':>20} {'300K LUT':>22} {'523K LUT':>22} {'diff':>12}")
print(f"  {'-'*78}")
print(f"  {'n_e [m⁻³]':>20} {ne300:22.15e} {ne523:22.15e} {(ne523-ne300)/ne300*100:+12.6f}%")
print(f"  {'ne_eps [eV/m³]':>20} {ne_eps300:22.15e} {ne_eps523:22.15e} {(ne_eps523-ne_eps300)/ne_eps300*100:+12.6f}%")
print(f"  {'ε̄ [eV]':>20} {eps300:22.15f} {eps523:22.15f} {(eps523-eps300)/eps300*100:+12.6f}%")
print(f"  {'Te [eV]':>20} {eps300*2/3:22.15f} {eps523*2/3:22.15f} {(eps523-eps300)/eps300*100:+12.6f}%")
print(f"  {'T_gas [K]':>20} {Tg300:22.15f} {Tg523:22.15f} {(Tg523-Tg300)/max(Tg300,1)*100:+12.6f}%")

# Compare key species
for sp_name in ['CH4', 'CO2', 'H2', 'CO', 'O', 'OH', 'O-']:
    try:
        idx = s300.sm.index(sp_name)
        v3 = c300[idx]; v5 = c523_f[idx]
        d = (v5 - v3) / v3 * 100 if abs(v3) > 1e-50 else 0
        print(f"  {sp_name:>20} {v3:22.15e} {v5:22.15e} {d:+12.6f}%")
    except:
        pass

# Compare at multiple time points
print(f"\n  Time evolution comparison (n_e and ε̄):")
print(f"  {'t(s)':>10} | {'n_e_300K':>14} {'n_e_523K':>14} {'diff%':>8} | "
      f"{'ε̄_300K':>10} {'ε̄_523K':>10} {'diff%':>8}")
print(f"  {'-'*90}")

for i_t in [0, 10, 25, 50, 75, 99]:
    if i_t >= r300.y.shape[1] or i_t >= r523.y.shape[1]:
        continue
    t_val = r300.t[i_t]
    ne3 = r300.y[0, i_t] * NA
    ne5 = r523.y[0, i_t] * NA
    neps3 = r300.y[s300.sm.idx_energy, i_t]
    neps5 = r523.y[s523.sm.idx_energy, i_t]
    e3 = neps3 / ne3 if ne3 > 1 else 1.0
    e5 = neps5 / ne5 if ne5 > 1 else 1.0
    d_ne = (ne5 - ne3) / ne3 * 100 if ne3 > 1 else 0
    d_eps = (e5 - e3) / e3 * 100 if e3 > 0.001 else 0
    print(f"  {t_val:10.4f} | {ne3:14.6e} {ne5:14.6e} {d_ne:+8.4f}% | "
          f"{e3:10.6f} {e5:10.6f} {d_eps:+8.4f}%")
