#!/usr/bin/env python3
"""Full reaction budget: all 101 reactions × 25 species × time series to t=4min."""
import numpy as np, sys, time as time_mod
from pathlib import Path

_dir = Path(__file__).parent
sys.path.insert(0, str(_dir.parent.parent / 'Ver4_1D'))
sys.path.insert(0, str(_dir.parent))

from chemistry_1d import AqueousChemistry1D
from gen_all_figures import load_gas_data
from pde_solver import PDESolver1D

times, gas_conc = load_gas_data()
chem = AqueousChemistry1D(saline_mode=False)
solver = PDESolver1D(chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
                     saline_mode=False, bc_type='film_alpha', alpha_b=0.03)
solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                    hono_gas=0, hono2_gas=0, h2o2_gas=0)

ref = dict(np.load(_dir.parent / 'cache' / 'film_alpha_ab0.0300.npz', allow_pickle=True))
snap_t = ref['snap_t']
snap_y = ref['snap_y']
dz = ref['dz_cells']
L = float(ref['L'])
idx = chem.species_idx
N_z = int(ref['N_z'])
N_s = int(ref['N_s'])
n_rxn = len(chem.reactions)

# Time range: 0 ~ 4 min (240s), every 2s snapshot
mask = snap_t <= 242
t_sel = snap_t[mask]
y_sel = snap_y[mask]
nt = len(t_sel)

print(f"Species: {N_s}, Reactions: {n_rxn}, Snapshots: {nt} (0-{t_sel[-1]:.0f}s)")

# ═══════════════════════════════════════════════════════════════
# 1. Compute per-reaction, per-species, volume-averaged rate at each snapshot
# ═══════════════════════════════════════════════════════════════
# rate_contrib[t, rxn, species] = net contribution of reaction rxn to species
# Positive = production, negative = consumption

print("Computing all reaction rates at all snapshots...")
t0 = time_mod.time()

# For each species, which reactions affect it?
# Build mapping: species_name -> list of (rxn_idx, stoich_coeff)
# stoich = +coeff if product, -coeff if reactant

species_names = list(idx.keys())
sp_rxn_map = {sp: [] for sp in species_names}

# Also need speciated species mapping
from config_1d import ACID_BASE_PAIRS
spec_to_total = {}
for total_name, (acid, base, pKa) in ACID_BASE_PAIRS.items():
    spec_to_total[acid] = total_name
    spec_to_total[base] = total_name

for ri, rxn in enumerate(chem.reactions):
    for sp, coeff in rxn['reactants'].items():
        # Map speciated to total
        target = spec_to_total.get(sp, sp)
        if target in idx:
            sp_rxn_map[target].append((ri, -int(coeff)))
    for sp, coeff in rxn.get('products', {}).items():
        target = spec_to_total.get(sp, sp)
        if target in idx:
            sp_rxn_map[target].append((ri, +int(coeff)))

# Compute volume-averaged rate for each reaction at each snapshot
h_idx = idx['H+']
rate_avg = np.zeros((nt, n_rxn))  # (time, rxn)

for ti in range(nt):
    y2d = y_sel[ti]
    rate_cells = np.zeros((n_rxn, N_z))
    for j in range(N_z):
        yc = np.clip(y2d[j, :].copy(), 1e-30, 1.0)
        yc[h_idx] = max(yc[h_idx], 1e-14)
        spec = chem.speciate(yc)
        for ri, rxn_d in enumerate(chem._rxn_data):
            rate_cells[ri, j] = chem._compute_single_rate(rxn_d, yc, spec)
    rate_avg[ti, :] = np.dot(rate_cells, dz) / L

    if (ti + 1) % 20 == 0:
        print(f"  {ti+1}/{nt} snapshots done ({time_mod.time()-t0:.1f}s)")

print(f"Rate computation: {time_mod.time()-t0:.1f}s")

# Compute per-species contribution: contrib[t, sp] from each rxn
# rate_by_sp[sp_name][rxn_label] = array(nt)
rate_by_sp = {}
for sp_name in species_names:
    rate_by_sp[sp_name] = {}
    for ri, stoich in sp_rxn_map[sp_name]:
        label = chem.reactions[ri].get('label', f'R{ri}')
        if label not in rate_by_sp[sp_name]:
            rate_by_sp[sp_name][label] = np.zeros(nt)
        rate_by_sp[sp_name][label] += stoich * rate_avg[:, ri]

# MT contribution per species
idx_to_name = {v: k for k, v in solver.species_idx.items()}
mt_by_sp = {sp: np.zeros(nt) for sp in species_names}

for ti in range(nt):
    y2d = y_sel[ti]
    hp = solver._h_plus_idx
    h_s = max(y2d[0, hp], 1e-14) if hp >= 0 else 1e-7
    for aq_idx, k_mt, gas_sp, _, Ka in solver._interface_species:
        aq_name = idx_to_name[aq_idx]
        C_eq = solver._get_C_eq_fast(gas_sp, t_sel[ti])
        C_0 = y2d[0, aq_idx]
        c_eff = C_0 * h_s / (h_s + Ka) if Ka is not None else C_0
        mt_by_sp[aq_name][ti] = k_mt * (C_eq - c_eff) / L

# Volume-averaged concentrations
conc = np.zeros((nt, N_s))
for ti in range(nt):
    for si in range(N_s):
        conc[ti, si] = np.dot(y_sel[ti][:, si], dz) / L

# dC/dt from finite diff
dcdt = np.zeros((nt, N_s))
dt_arr = np.diff(t_sel)
for si in range(N_s):
    dc = np.diff(conc[:, si])
    d = dc / dt_arr
    dcdt[0, si] = d[0]
    dcdt[-1, si] = d[-1]
    dcdt[1:-1, si] = 0.5 * (d[:-1] + d[1:])

# ═══════════════════════════════════════════════════════════════
# 2. Save full data as CSV for inspection
# ═══════════════════════════════════════════════════════════════
import csv

outfile = _dir / 'full_budget.csv'
with open(outfile, 'w', newline='') as f:
    w = csv.writer(f)
    # Header
    w.writerow(['t(s)', 't(min)', 'species', 'conc(M)', 'dCdt_actual(M/s)',
                'dCdt_budget(M/s)', 'MT(M/s)', 'reaction', 'rate_contrib(M/s)'])

    for sp_name in species_names:
        si = idx[sp_name]
        rxn_dict = rate_by_sp[sp_name]
        mt_arr = mt_by_sp[sp_name]

        for ti in range(nt):
            budget_sum = sum(v[ti] for v in rxn_dict.values()) + mt_arr[ti]

            # Write MT
            if abs(mt_arr[ti]) > 1e-35:
                w.writerow([f'{t_sel[ti]:.1f}', f'{t_sel[ti]/60:.3f}',
                           sp_name, f'{conc[ti,si]:.6e}',
                           f'{dcdt[ti,si]:.6e}', f'{budget_sum:.6e}',
                           f'{mt_arr[ti]:.6e}', 'MT', f'{mt_arr[ti]:.6e}'])

            # Write each reaction
            for label, rarr in sorted(rxn_dict.items(), key=lambda x: -abs(x[1][ti])):
                if abs(rarr[ti]) > 1e-35:
                    w.writerow([f'{t_sel[ti]:.1f}', f'{t_sel[ti]/60:.3f}',
                               sp_name, f'{conc[ti,si]:.6e}',
                               f'{dcdt[ti,si]:.6e}', f'{budget_sum:.6e}',
                               f'{mt_arr[ti]:.6e}', label, f'{rarr[ti]:.6e}'])

print(f"saved: {outfile.name} ({outfile.stat().st_size/1e6:.1f} MB)")

# ═══════════════════════════════════════════════════════════════
# 3. Figure: top reactions per species around transition
# ═══════════════════════════════════════════════════════════════
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 6, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Select key species to plot
key_species = ['OH', 'O3', 'HO2_total', 'O3-', 'O-', 'HO3',
               'ONOOH_total', 'NO2', 'NO3', 'HONO2_total']

t_min_arr = t_sel / 60.0
med_w = 5

n_sp = len(key_species)
fig, axes = plt.subplots(n_sp, 2, figsize=(16, 3.5 * n_sp),
                          gridspec_kw={'width_ratios': [2, 1]})

for pi, sp_name in enumerate(key_species):
    si = idx[sp_name]
    rxn_dict = rate_by_sp[sp_name]
    mt_arr = mt_by_sp[sp_name]

    # Left: rate contributions
    ax_rate = axes[pi, 0]

    # Find top reactions (by max absolute contribution)
    rxn_max = {label: np.max(np.abs(arr)) for label, arr in rxn_dict.items()}
    top_rxns = sorted(rxn_max.keys(), key=lambda x: -rxn_max[x])[:8]

    for label in top_rxns:
        arr = rxn_dict[label]
        ax_rate.plot(t_min_arr, median_filter(arr, size=med_w),
                     lw=0.8, label=label[:45])

    if np.max(np.abs(mt_arr)) > 1e-35:
        ax_rate.plot(t_min_arr, mt_arr, 'k-', lw=1.2, label='MT')

    # dC/dt actual
    ax_rate.plot(t_min_arr, median_filter(dcdt[:, si], size=med_w),
                 'k--', lw=1.5, label='dC/dt (actual)')

    ax_rate.axhline(0, color='gray', lw=0.3)
    ax_rate.axvline(2.1, color='red', ls=':', lw=0.5, alpha=0.5)
    ax_rate.set_ylabel(f'd[{sp_name}]/dt (M/s)')
    ax_rate.set_title(f'{sp_name} — rate budget', fontweight='bold', loc='left')
    ax_rate.legend(fontsize=5, loc='best', ncol=2)
    ax_rate.set_xlabel('Time (min)')

    # Right: concentration
    ax_conc = axes[pi, 1]
    c_arr = conc[:, si]
    # Auto-scale
    c_max = np.max(c_arr[c_arr > 0]) if np.any(c_arr > 0) else 1
    if c_max > 1e-6:
        ax_conc.plot(t_min_arr, c_arr * 1e6, 'b-', lw=1.0)
        ax_conc.set_ylabel(f'[{sp_name}] (uM)')
    elif c_max > 1e-9:
        ax_conc.plot(t_min_arr, c_arr * 1e9, 'b-', lw=1.0)
        ax_conc.set_ylabel(f'[{sp_name}] (nM)')
    elif c_max > 1e-12:
        ax_conc.plot(t_min_arr, c_arr * 1e12, 'b-', lw=1.0)
        ax_conc.set_ylabel(f'[{sp_name}] (pM)')
    else:
        ax_conc.plot(t_min_arr, c_arr * 1e15, 'b-', lw=1.0)
        ax_conc.set_ylabel(f'[{sp_name}] (fM)')
    ax_conc.axvline(2.1, color='red', ls=':', lw=0.5, alpha=0.5)
    ax_conc.set_title(f'{sp_name} — concentration', fontweight='bold', loc='left')
    ax_conc.set_xlabel('Time (min)')
    ax_conc.grid(True, alpha=0.2)

fig.suptitle('Full reaction budget: all species, 0-4 min (101 rxn + MT)',
             fontsize=14, y=1.005)
fig.tight_layout()
fig.savefig(_dir / 'fig_full_budget.png')
fig.savefig(_dir / 'fig_full_budget.pdf')
plt.close(fig)
print(f"saved: fig_full_budget")

# ═══════════════════════════════════════════════════════════════
# 4. Print summary table at key time points
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("SUMMARY: Top 5 reactions per species at t=100s (pre) and t=130s (post)")
print("=" * 100)

for t_target in [100, 130]:
    ti = np.argmin(np.abs(t_sel - t_target))
    print(f"\n{'='*50} t={t_sel[ti]:.0f}s ({t_sel[ti]/60:.1f}min) {'='*50}")
    for sp_name in key_species:
        si = idx[sp_name]
        rxn_dict = rate_by_sp[sp_name]
        mt_val = mt_by_sp[sp_name][ti]

        all_contribs = [(label, arr[ti]) for label, arr in rxn_dict.items() if abs(arr[ti]) > 1e-35]
        if abs(mt_val) > 1e-35:
            all_contribs.append(('MT', mt_val))

        all_contribs.sort(key=lambda x: -abs(x[1]))
        total = sum(v for _, v in all_contribs)

        print(f"\n  {sp_name} [{conc[ti,si]:.3e} M], dC/dt={dcdt[ti,si]:.3e}, budget={total:.3e}:")
        for label, val in all_contribs[:5]:
            sign = '+' if val > 0 else '-'
            print(f"    {sign} {label:50s} {val:+.3e}")

print("\n=== DONE ===")
