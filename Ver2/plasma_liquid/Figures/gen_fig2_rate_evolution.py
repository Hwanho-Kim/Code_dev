#!/usr/bin/env python3
"""
Figure 2: Time-resolved reaction rate contributions.

Runs DIW Film+α_b (α_b=0.03) simulation, collects snapshots every 10s,
computes per-reaction volume-averaged rates for target species,
and generates 4-panel stacked area chart.

Source (+) above zero, sink (-) below zero.
Reactions contributing ≥1% at any time point are shown.
"""

import sys
import os
import time as time_mod
from pathlib import Path
from collections import defaultdict

import numpy as np

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

from config_1d import (
    PHYSICAL, MASS_TRANSFER, GRID,
    GAS_TO_AQUEOUS_MAP, ACID_BASE_PAIRS,
    N2O4_EQ,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

import pandas as pd

# ── Configuration ──
ALPHA_B = 0.03
DT_SNAPSHOT = 2.0           # snapshot interval (s) for rate evaluation
DT_ENFORCE = None            # None = single BDF call (no macro-step restart)
TARGET_SPECIES = ['NO3-', 'O3', 'NO2-', 'H2O2']
PCT_THRESHOLD = 1.0         # ≥1% at any time → include
DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

# Species → total variable mapping (acid-base pairs)
SPEC_TO_TOTAL = {
    'HONO': 'HONO_total', 'NO2-': 'HONO_total',
    'HONO2': 'HONO2_total', 'NO3-': 'HONO2_total',
    'H2O2': 'H2O2_total', 'HO2-': 'H2O2_total',
    'HO2': 'HO2_total', 'O2-': 'HO2_total',
    'ONOOH': 'ONOOH_total', 'ONOO-': 'ONOOH_total',
    'O2NOOH': 'O2NOOH_total', 'O2NOO-': 'O2NOOH_total',
    'HClO': 'HClO_total', 'ClO-': 'HClO_total',
}

# Nice Unicode labels for species
SPECIES_LABEL = {
    'NO3-': 'NO₃⁻', 'O3': 'O₃', 'NO2-': 'NO₂⁻', 'H2O2': 'H₂O₂',
}

# Nice Unicode labels for common reaction species
_CHEM_UNICODE = {
    'O3': 'O₃', 'OH': 'OH', 'HO2': 'HO₂', 'H2O2': 'H₂O₂',
    'NO2': 'NO₂', 'NO3': 'NO₃', 'N2O5': 'N₂O₅', 'N2O4': 'N₂O₄',
    'N2O3': 'N₂O₃', 'O2': 'O₂', 'H2O': 'H₂O', 'H+': 'H⁺',
    'OH-': 'OH⁻', 'NO2-': 'NO₂⁻', 'NO3-': 'NO₃⁻', 'HO2-': 'HO₂⁻',
    'O2-': 'O₂⁻', 'ONOO-': 'ONOO⁻', 'O2NOO-': 'O₂NOO⁻',
    'ONOOH': 'ONOOH', 'O2NOOH': 'O₂NOOH', 'O3-': 'O₃⁻',
    'HO3': 'HO₃', 'HONO_total': 'HONO_t', 'HONO2_total': 'HNO₃_t',
    'H2O2_total': 'H₂O₂_t', 'HO2_total': 'HO₂_t',
    'ONOOH_total': 'ONOOH_t', 'O2NOOH_total': 'O₂NOOH_t',
}


def _nice(sp):
    return _CHEM_UNICODE.get(sp, sp)


# =====================================================================
# 1. Simulation
# =====================================================================

def load_gas_data():
    df = pd.read_csv(DEFAULT_CSV)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))
    if 'N2O4' not in df.columns or np.all(gas_conc['N2O4'] == 0):
        import math
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298)
            + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)
    return times, gas_conc


def run_simulation(times, gas_conc):
    """Run DIW simulation, return snapshots at every DT_SNAPSHOT."""
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='film_alpha',
        alpha_b=ALPHA_B,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=0, hono2_gas=0, h2o2_gas=0,
    )
    t_end = float(times[-1])

    # t_eval at every snapshot interval
    t_eval = np.arange(DT_SNAPSHOT, t_end + 0.1, DT_SNAPSHOT)
    t_eval = t_eval[t_eval <= t_end + 0.1]

    y0 = solver.build_initial_condition(initial_pH=7.0)

    dt_label = f"{DT_ENFORCE}s" if DT_ENFORCE else "None (single BDF)"
    print(f"Running DIW (monolithic BDF): α_b={ALPHA_B}, "
          f"dt_enforce={dt_label}, snapshots every {DT_SNAPSHOT}s, "
          f"t_end={t_end}s, {len(t_eval)} eval points")
    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=t_eval,
        verbose=True, dt_poisson=DT_ENFORCE,
    )
    print(f"Simulation: {time_mod.time()-t0:.1f}s")

    N_z, N_s = solver.N_z, solver.N_s

    # Prepend t=0
    snap_times = [0.0]
    snaps = [y0.reshape(N_z, N_s).copy()]
    for i, tv in enumerate(result['t_eval']):
        snap_times.append(float(tv))
        snaps.append(result['y_eval'][i].reshape(N_z, N_s).copy())

    # Dense output interpolants (list of OdeSolution)
    dense_output = result.get('dense_output', [])

    return np.array(snap_times), snaps, solver, dense_output


def compute_conc_timeseries(snap_times, snaps, solver):
    """Volume-averaged concentration of target species at each snapshot."""
    chem = solver.chem
    dz, L = solver.dz_cells, solver.L
    nt = len(snap_times)
    conc = {}
    for sp_name in TARGET_SPECIES:
        total_name = SPEC_TO_TOTAL.get(sp_name, sp_name)
        idx = chem.species_idx.get(total_name,
              chem.species_idx.get(sp_name))
        if idx is None:
            continue
        c_arr = np.zeros(nt)
        for i, y2d in enumerate(snaps):
            c_arr[i] = np.dot(y2d[:, idx], dz) / L
        conc[sp_name] = c_arr
    return conc


def compute_net_dcdt_from_conc(snap_times, conc):
    """Finite-difference dC/dt from actual concentration evolution.

    Returns arrays of same length as snap_times (forward diff at edges).
    """
    net = {}
    for sp_name, c in conc.items():
        dt = np.diff(snap_times)
        dc = np.diff(c)
        dcdt = dc / dt  # length nt-1
        # Extend to same length: use forward diff at t=0, backward at end
        dcdt_full = np.zeros(len(snap_times))
        dcdt_full[0] = dcdt[0] if len(dcdt) > 0 else 0
        dcdt_full[-1] = dcdt[-1] if len(dcdt) > 0 else 0
        dcdt_full[1:-1] = 0.5 * (dcdt[:-1] + dcdt[1:])  # central diff
        net[sp_name] = dcdt_full
    return net


# =====================================================================
# 2. Per-reaction rate computation
# =====================================================================

def compute_rates_snapshot(solver, y_2d, t):
    """Volume-averaged per-reaction rates and MT flux at one snapshot."""
    chem = solver.chem
    N_z, dz, L = solver.N_z, solver.dz_cells, solver.L
    n_rxn = len(chem.reactions)
    h_idx = chem.species_idx['H+']

    rates_2d = np.zeros((n_rxn, N_z))
    for j in range(N_z):
        yc = np.clip(y_2d[j, :].copy(), chem.trace, 1.0)
        yc[h_idx] = max(yc[h_idx], 1e-14)
        spec = chem.speciate(yc)
        for ri, rxn_d in enumerate(chem._rxn_data):
            rates_2d[ri, j] = chem._compute_single_rate(rxn_d, yc, spec)

    rate_avg = np.dot(rates_2d, dz) / L

    rxn_rates = []
    for ri in range(n_rxn):
        rxn = chem.reactions[ri]
        rxn_rates.append({
            'label': rxn.get('label', f'R{ri}'),
            'rate': rate_avg[ri],
            'reactants': rxn['reactants'],
            'products': rxn.get('products', {}),
        })

    # Mass transfer flux
    mt_flux = {}
    t_idx = max(0, min(int(t / solver._dt_gas), solver._n_times - 1))
    idx_to_name = {v: k for k, v in solver.species_idx.items()}
    hp_idx = solver._h_plus_idx
    h_s = max(y_2d[0, hp_idx], 1e-14) if hp_idx >= 0 else 1e-7
    for aq_idx, k_mt, gas_sp, _, Ka in solver._interface_species:
        C_eq = solver._get_C_eq_fast(gas_sp, t_idx)
        C_0 = y_2d[0, aq_idx]
        c_eff = C_0 * h_s / (h_s + Ka) if Ka is not None else C_0
        mt_flux[idx_to_name[aq_idx]] = k_mt * (C_eq - c_eff) / L

    return rxn_rates, mt_flux


def compute_rates_simpson(solver, dense_sol, t_start, t_end):
    """Simpson-averaged per-reaction rates over [t_start, t_end].

    Evaluates rates at 3 points (t_start, t_mid, t_end) using dense output,
    then applies Simpson's rule: (f(a) + 4f(m) + f(b)) / 6.
    """
    N_z, N_s = solver.N_z, solver.N_s
    t_mid = 0.5 * (t_start + t_end)

    # Evaluate y at 3 points via dense output
    y_a = np.clip(dense_sol(t_start), solver.chem.trace, None).reshape(N_z, N_s)
    y_m = np.clip(dense_sol(t_mid), solver.chem.trace, None).reshape(N_z, N_s)
    y_b = np.clip(dense_sol(t_end), solver.chem.trace, None).reshape(N_z, N_s)

    rr_a, mt_a = compute_rates_snapshot(solver, y_a, t_start)
    rr_m, mt_m = compute_rates_snapshot(solver, y_m, t_mid)
    rr_b, mt_b = compute_rates_snapshot(solver, y_b, t_end)

    # Simpson average: (f(a) + 4*f(m) + f(b)) / 6
    n_rxn = len(rr_a)
    rxn_rates_avg = []
    for ri in range(n_rxn):
        avg_rate = (rr_a[ri]['rate'] + 4 * rr_m[ri]['rate'] + rr_b[ri]['rate']) / 6.0
        rxn_rates_avg.append({
            'label': rr_a[ri]['label'],
            'rate': avg_rate,
            'reactants': rr_a[ri]['reactants'],
            'products': rr_a[ri]['products'],
        })

    mt_flux_avg = {}
    for key in mt_a:
        mt_flux_avg[key] = (mt_a[key] + 4 * mt_m[key] + mt_b[key]) / 6.0

    return rxn_rates_avg, mt_flux_avg


def _total_match_names(species_name):
    """All species that contribute to the same total variable.

    E.g. 'NO3-' → {'NO3-', 'HONO2', 'HONO2_total'}
         'H2O2' → {'H2O2', 'HO2-', 'H2O2_total'}
    """
    names = {species_name}
    total = SPEC_TO_TOTAL.get(species_name)
    if total:
        names.add(total)
        for sp, t in SPEC_TO_TOTAL.items():
            if t == total:
                names.add(sp)
    return names


def species_contribution(rxn_rates, species_name, mt_flux):
    """Net rate contribution of each reaction to one species."""
    match_names = _total_match_names(species_name)

    contribs = []
    for r in rxn_rates:
        in_r = set(r['reactants'].keys()) & match_names
        in_p = set(r['products'].keys()) & match_names
        if not in_r and not in_p:
            continue
        net = 0.0
        for sp in in_p:
            net += int(r['products'][sp]) * r['rate']
        for sp in in_r:
            net -= int(r['reactants'][sp]) * r['rate']
        if abs(net) > 1e-30:
            contribs.append((r['label'], net))

    # Mass transfer
    mt_val = sum(mt_flux.get(n, 0.0) for n in match_names)
    if abs(mt_val) > 1e-30:
        contribs.append(('MT', mt_val))

    return contribs


def build_timeseries(snap_times, all_rxn_rates, all_mt_flux,
                     species_name, pct_thr=1.0):
    """Build {label: rate_array} for one species, filtering ≥pct_thr.

    Returns (sig_labels, sig_rates_dict, net_dcdt).
    net_dcdt is sum of ALL reactions (not just filtered).
    """
    nt = len(snap_times)
    by_label = defaultdict(lambda: np.zeros(nt))

    for i in range(nt):
        for label, rate in species_contribution(
                all_rxn_rates[i], species_name, all_mt_flux[i]):
            by_label[label][i] = rate

    # TRUE net dC/dt = sum of ALL reactions (before filtering)
    net_dcdt = np.zeros(nt)
    for rates in by_label.values():
        net_dcdt = net_dcdt + rates

    # Filter: keep reactions ≥pct_thr at any time point
    # Also require minimum absolute rate (ignore near-zero turnover at t≈0)
    max_total = max(
        sum(abs(r[i]) for r in by_label.values())
        for i in range(nt)
    ) if nt > 0 else 0
    min_total = max_total * 0.01  # ignore time points with <1% of peak turnover

    sig = []
    for label, rates in by_label.items():
        peak = np.max(np.abs(rates))
        if peak < 1e-25:  # absolute floor
            continue
        for i in range(nt):
            total = sum(abs(r[i]) for r in by_label.values())
            if total > min_total and abs(rates[i]) / total * 100 >= pct_thr:
                sig.append(label)
                break

    sig.sort(key=lambda lb: -np.mean(np.abs(by_label[lb])))
    return sig, {lb: by_label[lb] for lb in sig}, net_dcdt


def _short_label(full_label):
    """Extract short ID from YAML label, e.g. 'R98: N2O5 + ...' → 'R98'."""
    if ':' in full_label:
        return full_label.split(':')[0].strip()
    return full_label


def rxn_description(chem, label):
    """Short Unicode reaction string."""
    if label == 'MT':
        return 'mass transfer'
    for rxn in chem.reactions:
        if rxn.get('label') == label:
            r_parts = []
            for sp, c in rxn['reactants'].items():
                c = int(c)
                r_parts.append(f'{c}{_nice(sp)}' if c > 1 else _nice(sp))
            p_parts = []
            for sp, c in rxn.get('products', {}).items():
                c = int(c)
                p_parts.append(f'{c}{_nice(sp)}' if c > 1 else _nice(sp))
            return ' + '.join(r_parts) + ' → ' + ' + '.join(p_parts)
    return label


def _legend_entry(chem, full_label):
    """Build concise legend string: 'R98: N₂O₅ + H₂O → 2NO₃⁻'."""
    short = _short_label(full_label)
    desc = rxn_description(chem, full_label)
    return f'{short}: {desc}'


SMOOTH_TIME_RATE = 60.0     # smoothing window for individual rates (s)
SMOOTH_TIME_NET  = 120.0    # smoothing window for net Σrate (s)


def _smooth(arr, window=5):
    """Median filter + moving average to remove numerical noise."""
    from scipy.ndimage import median_filter, uniform_filter1d
    window = max(3, window) | 1  # ensure odd
    if len(arr) < window:
        return arr
    out = median_filter(arr, size=window, mode='nearest')
    ma_w = max(3, window // 2) | 1
    out = uniform_filter1d(out, size=ma_w, mode='nearest')
    return out


# =====================================================================
# 3. Plotting
# =====================================================================

def plot_fig2(snap_times, species_data, chem):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 7, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
    })

    # Color cycle: distinct colors for individual lines
    line_colors = [
        '#2166ac', '#b2182b', '#4393c3', '#d6604d',
        '#1b7837', '#e08214', '#762a83', '#35978f',
        '#c51b7d', '#543005',
    ]
    line_styles = ['-', '-', '-', '-', '-', '-', '-', '-', '--', '--']

    t_min = snap_times / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    # Smoothing windows scaled by data density
    dt_snap = np.median(np.diff(snap_times)) if len(snap_times) > 1 else 10.0
    sw = max(3, int(SMOOTH_TIME_RATE / dt_snap))
    sw_net = max(3, int(SMOOTH_TIME_NET / dt_snap))

    for pi, sp_name in enumerate(TARGET_SPECIES):
        ax = axes.flat[pi]
        labels, rd, net_dcdt = species_data[sp_name]

        # Smooth all rate timeseries
        rd_s = {lb: _smooth(rd[lb], sw) for lb in labels}

        # Sort by absolute average (largest first)
        sorted_labels = sorted(labels, key=lambda lb: -np.mean(np.abs(rd_s[lb])))

        # Plot each reaction as individual line
        for ci, lb in enumerate(sorted_labels):
            col = line_colors[ci % len(line_colors)]
            ls = line_styles[ci % len(line_styles)]
            lw = 1.8 if ci < 3 else 1.2  # thicker for top contributors
            ax.plot(t_min, rd_s[lb], color=col, ls=ls, lw=lw,
                    label=_legend_entry(chem, lb), zorder=5 + len(sorted_labels) - ci)

        # Net dC/dt = Σ(all reaction rates) — self-consistent with
        # individual lines.  Heavier smoothing because net is a small
        # residual of large canceling source/sink terms.
        net_s = _smooth(net_dcdt, sw_net)
        ax.plot(t_min, net_s, color='black', ls='--', lw=1.5,
                label='Σrate (net)', zorder=10)

        ax.axhline(0, color='black', lw=0.5, zorder=1)
        sp_lbl = SPECIES_LABEL.get(sp_name, sp_name)
        ax.set_title(f'({"abcd"[pi]}) {sp_lbl}', fontweight='bold', loc='left')

        # y-limits: include both reaction rates AND net dC/dt
        all_vals = np.concatenate(
            [rd_s[lb] for lb in labels] + [net_s])
        if len(all_vals) > 0 and np.any(np.abs(all_vals) > 0):
            q_lo, q_hi = np.percentile(all_vals, [1, 99])
            span = q_hi - q_lo
            if span < 1e-30:
                span = max(abs(q_hi), abs(q_lo)) * 0.5
            ax.set_ylim(q_lo - span * 0.15, q_hi + span * 0.15)

        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

        ncol = 2 if len(sorted_labels) > 5 else 1
        ax.legend(loc='best', framealpha=0.92, fontsize=6,
                  handlelength=1.5, borderpad=0.3, labelspacing=0.25,
                  ncol=ncol)

    axes[1, 0].set_xlabel('Time (min)')
    axes[1, 1].set_xlabel('Time (min)')
    axes[0, 0].set_ylabel('Rate (M/s)')
    axes[1, 0].set_ylabel('Rate (M/s)')

    fig.suptitle(
        f'Reaction rate contributions over time '
        f'(DIW, Film+αb, αb={ALPHA_B})',
        fontsize=13, y=1.01)
    fig.tight_layout()

    out_png = _script_dir / 'fig2_rate_evolution.png'
    out_pdf = _script_dir / 'fig2_rate_evolution.pdf'
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  → {out_png.name} / {out_pdf.name} saved")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

CACHE_NPZ = _script_dir / 'fig2_rate_cache.npz'


def save_cache(snap_times, species_data):
    """Save computed timeseries to npz for fast re-plotting."""
    data = {'snap_times': snap_times}
    for sp_name in TARGET_SPECIES:
        labels, rd, net_dcdt = species_data[sp_name]
        data[f'{sp_name}__labels'] = np.array(labels, dtype=object)
        data[f'{sp_name}__net_dcdt'] = net_dcdt
        for lb in labels:
            key = f'{sp_name}__{lb}'
            data[key] = rd[lb]
    np.savez(CACHE_NPZ, **data)
    print(f"  Cache saved: {CACHE_NPZ.name}")


def load_cache():
    """Load cached timeseries. Returns (snap_times, species_data) or None."""
    if not CACHE_NPZ.exists():
        return None
    d = np.load(CACHE_NPZ, allow_pickle=True)
    snap_times = d['snap_times']
    species_data = {}
    for sp_name in TARGET_SPECIES:
        key = f'{sp_name}__labels'
        if key not in d:
            return None
        labels = list(d[key])
        rd = {}
        for lb in labels:
            rd[lb] = d[f'{sp_name}__{lb}']
        net_key = f'{sp_name}__net_dcdt'
        if net_key not in d:
            return None  # force re-run if old cache format
        net_dcdt = d[net_key]
        species_data[sp_name] = (labels, rd, net_dcdt)
    return snap_times, species_data


def main():
    os.chdir(_project_root)

    print("=" * 70)
    print("Figure 2: Time-resolved reaction rate contributions")
    print(f"  α_b={ALPHA_B}, dt_snap={DT_SNAPSHOT}s, dt_enforce={DT_ENFORCE}, species={TARGET_SPECIES}")
    print("=" * 70)

    # Try loading from cache (--replot flag or automatic)
    replot_only = '--replot' in sys.argv
    cached = load_cache() if replot_only else None

    if cached is not None:
        snap_times, species_data = cached
        print(f"  Loaded cache: {len(snap_times)} snapshots")
        # Need chem for legend descriptions
        chem = AqueousChemistry1D(saline_mode=False)
        plot_fig2(snap_times, species_data, chem)
        print("\nDone (from cache)!")
        return

    # 1. Simulation
    times, gas_conc = load_gas_data()
    snap_times, snaps, solver, dense_output = run_simulation(times, gas_conc)
    nt = len(snap_times)
    print(f"  {nt} snapshots collected, "
          f"{len(dense_output)} dense output segment(s)")

    # 2. Per-reaction rates — Simpson average if dense output available
    use_simpson = len(dense_output) > 0
    if use_simpson:
        print("Computing per-reaction rates (Simpson 3-point average)...")
    else:
        print("Computing per-reaction rates (snapshot)...")
    t0 = time_mod.time()
    all_rxn_rates, all_mt_flux = [], []

    if use_simpson:
        # For interval-based Simpson, we compute rates for intervals
        # [t_{i}, t_{i+1}] and assign to the midpoint time.
        # For t=0 (first point), use snapshot rate.
        dense_sol = dense_output[0]  # single BDF → one interpolant

        # First point (t=0): snapshot
        rr0, mf0 = compute_rates_snapshot(solver, snaps[0], snap_times[0])
        all_rxn_rates.append(rr0)
        all_mt_flux.append(mf0)

        # Intervals [t_i, t_{i+1}]: Simpson average assigned to t_{i+1}
        for i in range(1, nt):
            t_a = snap_times[i - 1]
            t_b = snap_times[i]
            rr, mf = compute_rates_simpson(solver, dense_sol, t_a, t_b)
            all_rxn_rates.append(rr)
            all_mt_flux.append(mf)
            if i % 20 == 0:
                print(f"    {i}/{nt-1} intervals (t={t_b:.0f}s)")
    else:
        for i, (tv, y2d) in enumerate(zip(snap_times, snaps)):
            rr, mf = compute_rates_snapshot(solver, y2d, tv)
            all_rxn_rates.append(rr)
            all_mt_flux.append(mf)
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    {i+1}/{nt} (t={tv:.0f}s)")
    print(f"  Rate computation: {time_mod.time()-t0:.1f}s")

    # 3. Volume-averaged concentrations → true net dC/dt
    conc = compute_conc_timeseries(snap_times, snaps, solver)
    net_from_conc = compute_net_dcdt_from_conc(snap_times, conc)
    for sp_name in TARGET_SPECIES:
        sp_lbl = SPECIES_LABEL.get(sp_name, sp_name)
        c = conc.get(sp_name)
        if c is not None:
            print(f"  {sp_lbl}: C(0)={c[0]:.2e}, C(end)={c[-1]:.2e}, "
                  f"ΔC={c[-1]-c[0]:+.2e} M")

    # 4. Build timeseries (per-reaction rates)
    species_data = {}
    print()
    for sp_name in TARGET_SPECIES:
        labels, rd, budget_net = build_timeseries(
            snap_times, all_rxn_rates, all_mt_flux,
            sp_name, PCT_THRESHOLD)
        true_net = net_from_conc.get(sp_name, np.zeros(len(snap_times)))

        # Validation: compare Σrate vs finite-diff (skip t=0)
        mask = snap_times > 0
        if np.any(mask):
            ratio = np.mean(np.abs(budget_net[mask])) / max(
                np.mean(np.abs(true_net[mask])), 1e-30)
            rmse = np.sqrt(np.mean((budget_net[mask] - true_net[mask])**2))
            print(f"  [{sp_name}] Σrate vs ΔC/Δt: "
                  f"ratio={ratio:.3f}, RMSE={rmse:.2e}")
        # Use Σrate (budget_net) as net line — self-consistent with
        # individual rate lines.  Finite-diff ΔC/Δt is too noisy for
        # trace species (O₃ ~nM, H₂O₂ ~0.1nM).
        chosen_net = budget_net

        species_data[sp_name] = (labels, rd, chosen_net)
        sp_lbl = SPECIES_LABEL.get(sp_name, sp_name)
        print(f"  {sp_lbl}: {len(labels)} reactions ≥{PCT_THRESHOLD}%"
              f"  (net dC/dt avg={np.mean(chosen_net):+.3e})")
        for lb in labels:
            desc = rxn_description(solver.chem, lb)
            avg = np.mean(rd[lb])
            short = _short_label(lb)
            print(f"    {short:<8s} avg={avg:+.3e}  ({desc})")

    # 4. Cache + Plot
    save_cache(snap_times, species_data)
    print()
    plot_fig2(snap_times, species_data, solver.chem)
    print("\nDone!")


if __name__ == '__main__':
    main()
