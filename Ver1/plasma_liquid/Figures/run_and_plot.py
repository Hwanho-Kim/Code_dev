#!/usr/bin/env python3
"""
Unified simulation + figure generation script.

Runs all required simulations and generates Fig 1 ~ Fig 5.
No hardcoded data, no cache — always runs fresh from simulation results.

Usage:
    python run_and_plot.py              # Run all
    python run_and_plot.py --quick      # Skip heavy BC comparison (Fig1/1b)
"""

import sys
import os
import time as time_mod
import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, uniform_filter1d

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "Ver4_1D"))

from config_1d import (
    PHYSICAL, MASS_TRANSFER, N2O4_EQ,
    GAS_TO_AQUEOUS_MAP, ACID_BASE_PAIRS,
    LIQUID_DIFFUSIVITY, D_LIQ_DEFAULT,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D

DEFAULT_CSV = (
    _project_root / "empty chamber" / "empty chamber" / "1kHz3.2kVpp.csv"
)
EXP = {"pH": 3.61, "NO3": 63.0, "NO2": 3.0, "H2O2": 11.0}

# ── Plot style ──
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "lines.markersize": 7,
})

# ── Simulation configuration ──
BC_CASES = [
    {"label": "Two-film",      "bc_type": "two_film",   "alpha_b": 1.0},
    {"label": "Dirichlet",     "bc_type": "dirichlet",  "alpha_b": 1.0},
    {"label": "Film (ab=1)",   "bc_type": "film",       "alpha_b": 1.0},
    {"label": "Film+ab=0.05",  "bc_type": "film_alpha", "alpha_b": 0.05},
    {"label": "Film+ab=0.01",  "bc_type": "film_alpha", "alpha_b": 0.01},
]
ALPHA_CASES = [0.01, 0.03, 0.05]
DETAIL_ALPHA = 0.03
DT_SNAPSHOT = 2.0
SNAP_TIMES_MIN = [1, 2, 4, 6, 8, 12]

TARGET_SPECIES = ["NO3-", "O3", "NO2-", "H2O2"]
SPEC_TO_TOTAL = {
    "HONO": "HONO_total", "NO2-": "HONO_total",
    "HONO2": "HONO2_total", "NO3-": "HONO2_total",
    "H2O2": "H2O2_total", "HO2-": "H2O2_total",
    "HO2": "HO2_total", "O2-": "HO2_total",
    "ONOOH": "ONOOH_total", "ONOO-": "ONOOH_total",
    "O2NOOH": "O2NOOH_total", "O2NOO-": "O2NOOH_total",
    "HClO": "HClO_total", "ClO-": "HClO_total",
}
SPECIES_LABEL = {"NO3-": "NO3-", "O3": "O3", "NO2-": "NO2-", "H2O2": "H2O2"}
_CHEM_UNICODE = {
    "O3": "O3", "OH": "OH", "HO2": "HO2", "H2O2": "H2O2",
    "NO2": "NO2", "NO3": "NO3", "N2O5": "N2O5", "N2O4": "N2O4",
    "N2O3": "N2O3", "O2": "O2", "H2O": "H2O", "H+": "H+",
    "OH-": "OH-", "NO2-": "NO2-", "NO3-": "NO3-", "HO2-": "HO2-",
    "O2-": "O2-", "ONOO-": "ONOO-", "O2NOO-": "O2NOO-",
    "ONOOH": "ONOOH", "O2NOOH": "O2NOOH", "O3-": "O3-",
    "HO3": "HO3", "HONO_total": "HONO_t", "HONO2_total": "HNO3_t",
    "H2O2_total": "H2O2_t", "HO2_total": "HO2_t",
}

MT_SPECIES = [("N2O5", "N2O5"), ("O3", "O3"), ("NO2", "NO2"), ("NO3", "NO3")]

# =====================================================================
# Gas data loader
# =====================================================================
def load_gas_data():
    df = pd.read_csv(DEFAULT_CSV)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ["O3", "NO", "NO2", "NO3", "N2O4", "N2O5"]:
        gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0) if col in df.columns else np.zeros(len(df))
    if "N2O4" not in df.columns or np.all(gas_conc["N2O4"] == 0):
        no2 = gas_conc["NO2"]
        T = 298.15
        Kp = math.exp(math.log(N2O4_EQ.KP_298) + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
        gas_conc["N2O4"] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)
    return times, gas_conc


# =====================================================================
# Core simulation runner
# =====================================================================
def run_case(times, gas_conc, bc_type, alpha_b, label,
             t_eval=None, dt_poisson=None, verbose=True):
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        mass_transfer_eta=1.0, saline_mode=False,
        bc_type=bc_type, alpha_b=alpha_b,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=0, hono2_gas=0, h2o2_gas=0)
    t_end = float(times[-1])
    if t_eval is None:
        t_eval = np.array([0, t_end/4, t_end/2, 3*t_end/4, t_end])

    if verbose:
        print(f"  Running: {label} (bc={bc_type}, ab={alpha_b})")
    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=t_eval,
                          verbose=verbose, dt_poisson=dt_poisson)
    wall = time_mod.time() - t0
    if verbose:
        print(f"    -> {wall:.1f}s, success={result["success"]}")
    return result, solver, wall


def extract_summary(result, solver):
    avg = result["spatial_avg"]
    return {
        "pH": result["pH_avg"],
        "NO3": avg.get("NO3-", 0) * 1e6,
        "NO2": avg.get("NO2-", 0) * 1e6,
        "H2O2": avg.get("H2O2", 0) * 1e6,
        "OH": avg.get("OH", 0),
        "O3": avg.get("O3", 0),
        "HO2": avg.get("HO2", 0),
        "O2-": avg.get("O2-", 0),
        "ONOOH": avg.get("ONOOH", 0),
        "O2NOOH": avg.get("O2NOOH", 0),
        "ONOO-": avg.get("ONOO-", 0),
        "O3-": avg.get("O3-", 0),
        "N2O5": avg.get("N2O5", 0),
        "NO2_aq": avg.get("NO2", 0),
        "O2NOO-": avg.get("O2NOO-", 0),
        "pH_surface": result["pH_surface"],
        "success": result["success"],
    }


# =====================================================================
# Rate computation helpers (for Fig 2 / Fig 4)
# =====================================================================
def compute_rates_snapshot(solver, y_2d, t):
    chem = solver.chem
    N_z, dz, L = solver.N_z, solver.dz_cells, solver.L
    n_rxn = len(chem.reactions)
    h_idx = chem.species_idx["H+"]
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
            "label": rxn.get("label", f"R{ri}"),
            "rate": rate_avg[ri],
            "reactants": rxn["reactants"],
            "products": rxn.get("products", {}),
        })
    mt_flux = {}
    idx_to_name = {v: k for k, v in solver.species_idx.items()}
    hp_idx = solver._h_plus_idx
    h_s = max(y_2d[0, hp_idx], 1e-14) if hp_idx >= 0 else 1e-7
    for aq_idx, k_mt, gas_sp, _, Ka in solver._interface_species:
        C_eq = solver._get_C_eq_fast(gas_sp, t)
        C_0 = y_2d[0, aq_idx]
        c_eff = C_0 * h_s / (h_s + Ka) if Ka is not None else C_0
        mt_flux[idx_to_name[aq_idx]] = k_mt * (C_eq - c_eff) / L
    return rxn_rates, mt_flux


def compute_rates_simpson(solver, dense_sol, t_start, t_end_s):
    N_z, N_s = solver.N_z, solver.N_s
    t_mid = 0.5 * (t_start + t_end_s)
    y_a = np.clip(dense_sol(t_start), solver.chem.trace, None).reshape(N_z, N_s)
    y_m = np.clip(dense_sol(t_mid), solver.chem.trace, None).reshape(N_z, N_s)
    y_b = np.clip(dense_sol(t_end_s), solver.chem.trace, None).reshape(N_z, N_s)
    rr_a, mt_a = compute_rates_snapshot(solver, y_a, t_start)
    rr_m, mt_m = compute_rates_snapshot(solver, y_m, t_mid)
    rr_b, mt_b = compute_rates_snapshot(solver, y_b, t_end_s)
    rxn_rates_avg = []
    for ri in range(len(rr_a)):
        avg_rate = (rr_a[ri]["rate"] + 4*rr_m[ri]["rate"] + rr_b[ri]["rate"]) / 6.0
        rxn_rates_avg.append({
            "label": rr_a[ri]["label"], "rate": avg_rate,
            "reactants": rr_a[ri]["reactants"], "products": rr_a[ri]["products"],
        })
    mt_flux_avg = {k: (mt_a[k]+4*mt_m[k]+mt_b[k])/6.0 for k in mt_a}
    return rxn_rates_avg, mt_flux_avg


def _total_match_names(sp):
    names = {sp}
    total = SPEC_TO_TOTAL.get(sp)
    if total:
        names.add(total)
        for s, t in SPEC_TO_TOTAL.items():
            if t == total:
                names.add(s)
    return names


def species_contribution(rxn_rates, sp_name, mt_flux):
    match_names = _total_match_names(sp_name)
    contribs = []
    for r in rxn_rates:
        in_r = set(r["reactants"].keys()) & match_names
        in_p = set(r["products"].keys()) & match_names
        if not in_r and not in_p:
            continue
        net = 0.0
        for sp in in_p:
            net += int(r["products"][sp]) * r["rate"]
        for sp in in_r:
            net -= int(r["reactants"][sp]) * r["rate"]
        if abs(net) > 1e-30:
            contribs.append((r["label"], net))
    mt_val = sum(mt_flux.get(n, 0.0) for n in match_names)
    if abs(mt_val) > 1e-30:
        contribs.append(("MT", mt_val))
    return contribs


def build_timeseries(snap_times, all_rxn_rates, all_mt_flux, sp_name, pct_thr=1.0):
    nt = len(snap_times)
    by_label = defaultdict(lambda: np.zeros(nt))
    for i in range(nt):
        for label, rate in species_contribution(all_rxn_rates[i], sp_name, all_mt_flux[i]):
            by_label[label][i] = rate
    net_dcdt = np.zeros(nt)
    for rates in by_label.values():
        net_dcdt = net_dcdt + rates
    max_total = max((sum(abs(r[i]) for r in by_label.values()) for i in range(nt)), default=0)
    min_total = max_total * 0.01
    sig = []
    for label, rates in by_label.items():
        peak = np.max(np.abs(rates))
        if peak < 1e-25:
            continue
        for i in range(nt):
            total = sum(abs(r[i]) for r in by_label.values())
            if total > min_total and abs(rates[i]) / total * 100 >= pct_thr:
                sig.append(label)
                break
    sig.sort(key=lambda lb: -np.mean(np.abs(by_label[lb])))
    return sig, {lb: by_label[lb] for lb in sig}, net_dcdt


def build_mass_balance(solver, y_final, t_final):
    """Compute source/sink breakdown for Fig 4."""
    y_2d = y_final.reshape(solver.N_z, solver.N_s)
    rxn_rates, mt_flux = compute_rates_snapshot(solver, y_2d, t_final)
    result = {}
    for sp_name in TARGET_SPECIES:
        contribs = species_contribution(rxn_rates, sp_name, mt_flux)
        sources, sinks = [], []
        for label, rate in contribs:
            if rate > 0:
                sources.append((label, rate))
            elif rate < 0:
                sinks.append((label, abs(rate)))
        src_total = sum(r for _, r in sources) or 1e-30
        snk_total = sum(r for _, r in sinks) or 1e-30
        sources = [(lb, r/src_total*100) for lb, r in sorted(sources, key=lambda x: -x[1])]
        sinks = [(lb, r/snk_total*100) for lb, r in sorted(sinks, key=lambda x: -x[1])]
        result[sp_name] = {"sources": sources, "sinks": sinks,
                           "src_total": src_total, "snk_total": snk_total}
    return result


def _smooth(arr, window=5):
    window = max(3, window) | 1
    if len(arr) < window:
        return arr
    out = median_filter(arr, size=window, mode="nearest")
    ma_w = max(3, window // 2) | 1
    return uniform_filter1d(out, size=ma_w, mode="nearest")


def _nice(sp):
    return _CHEM_UNICODE.get(sp, sp)


def _short_label(full_label):
    return full_label.split(":")[0].strip() if ":" in full_label else full_label


def rxn_description(chem, label):
    if label == "MT":
        return "mass transfer"
    for rxn in chem.reactions:
        if rxn.get("label") == label:
            r_parts = [f"{int(c)}{_nice(sp)}" if int(c)>1 else _nice(sp) for sp, c in rxn["reactants"].items()]
            p_parts = [f"{int(c)}{_nice(sp)}" if int(c)>1 else _nice(sp) for sp, c in rxn.get("products",{}).items()]
            return " + ".join(r_parts) + " -> " + " + ".join(p_parts)
    return label


def _legend_entry(chem, full_label):
    return f"{_short_label(full_label)}: {rxn_description(chem, full_label)}"




# =====================================================================
# Figure plotting functions
# =====================================================================

def _format_bar_value(val, unit):
    """Format bar label: show nM/pM for very small uM values."""
    if unit == 'uM':
        if val >= 1.0:
            return f'{val:.1f}'
        elif val >= 0.01:
            return f'{val:.3f}'
        elif val * 1e3 >= 0.1:
            return f'{val*1e3:.1f} nM'
        elif val * 1e6 >= 0.1:
            return f'{val*1e6:.1f} pM'
        elif val > 0:
            return f'{val:.1e}'
        else:
            return '0'
    else:
        return f'{val:.2f}' if val < 10 else f'{val:.1f}'


def plot_fig1(bc_results):
    """Fig 1: BC model comparison bar chart."""
    labels = [r['label'] for r in bc_results]
    n = len(labels)
    x = np.arange(n)
    w = 0.6
    fig, axes = plt.subplots(2, 2, figsize=(max(12, 2.5*n), 8))
    BAR_COLOR = '#4878a8'
    panels = [
        ('pH',        [r['pH'] for r in bc_results],   EXP['pH'],   '',   False, None),
        ('NO2- (uM)', [r['NO2'] for r in bc_results],  EXP['NO2'],  'uM', False, None),
        ('NO3- (uM)', [r['NO3'] for r in bc_results],  EXP['NO3'],  'uM', True,  None),
        ('H2O2 (uM)', [r['H2O2'] for r in bc_results], EXP['H2O2'], 'uM', False, None),
    ]
    for i, (ylabel, data, exp_val, unit, use_log, _) in enumerate(panels):
        ax = axes.flat[i]
        bars = ax.bar(x, data, w, color=BAR_COLOR, edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.axhline(exp_val, color='k', ls='--', lw=1.2, label=f'Exp. {exp_val}')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
        ax.set_title(f'({"abcd"[i]}) {ylabel}')
        if use_log:
            ax.set_yscale('log')
            dmin = min((v for v in data if v > 0), default=1)
            dmax = max(data) if max(data) > 0 else 1
            ax.set_ylim(dmin * 0.3, dmax * 5)
        else:
            dmin, dmax = min(data), max(data)
            span = max(dmax - dmin, abs(dmax) * 0.1, 1.0)
            ax.set_ylim(min(dmin, 0) - span * 0.05, dmax + span * 0.35)
        # Value labels on bars
        for bar, val in zip(bars, data):
            txt = _format_bar_value(val, unit)
            if use_log and val > 0:
                ypos = val * 1.3
            else:
                ypos = max(val, 0) + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.text(bar.get_x() + bar.get_width()/2, ypos, txt,
                    ha='center', va='bottom', fontsize=7.5, rotation=45)
    fig.suptitle('Effect of gas-liquid interface BC model (DIW, 3.2 kVpp, 12 min)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig1_bc_comparison.png')
    fig.savefig(_script_dir / 'fig1_bc_comparison.pdf')
    plt.close(fig)
    print('  -> fig1_bc_comparison saved')


def plot_fig1b(mt_results):
    """Fig 1b: MT flux time series by BC type."""
    n_species = len(MT_SPECIES)
    fig, axes = plt.subplots(2, n_species, figsize=(4.5*n_species, 8), sharex=True)
    bc_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
    for col, (gas_name, sp_label) in enumerate(MT_SPECIES):
        ax_inst, ax_cum = axes[0, col], axes[1, col]
        for ri, res in enumerate(mt_results):
            if res is None:
                continue
            t_min = res['snap_times'] / 60.0
            flux = res['mt_data'][gas_name]
            cumul = np.zeros_like(flux)
            dt = np.diff(res['snap_times'])
            for i in range(1, len(flux)):
                cumul[i] = cumul[i-1] + 0.5*(flux[i-1]+flux[i])*dt[i-1]
            color = bc_colors[ri % len(bc_colors)]
            ax_inst.plot(t_min, flux, color=color, lw=1.5, label=res['label'])
            ax_cum.plot(t_min, cumul*1e6, color=color, lw=1.5, label=res['label'])
        ax_inst.set_title(sp_label, fontweight='bold')
        ax_inst.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        ax_cum.set_xlabel('Time (min)')
        if col == 0:
            ax_inst.set_ylabel('MT flux (M/s)')
            ax_cum.set_ylabel('Cumulative MT (uM)')
            ax_inst.legend(loc='best', fontsize=7)
    fig.suptitle('Mass transfer flux by BC model (DIW, 12 min)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig1b_mt_flux.png')
    fig.savefig(_script_dir / 'fig1b_mt_flux.pdf')
    plt.close(fig)
    print('  -> fig1b_mt_flux saved')


def plot_fig2(snap_times, species_data, chem):
    """Fig 2: Time-resolved reaction rate contributions."""
    line_colors = ['#2166ac','#b2182b','#4393c3','#d6604d','#1b7837','#e08214','#762a83','#35978f','#c51b7d','#543005']
    t_min = snap_times / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    dt_snap = np.median(np.diff(snap_times)) if len(snap_times) > 1 else 10.0
    sw = max(3, int(60.0 / dt_snap))
    sw_net = max(3, int(120.0 / dt_snap))
    for pi, sp_name in enumerate(TARGET_SPECIES):
        ax = axes.flat[pi]
        labels, rd, net_dcdt = species_data[sp_name]
        rd_s = {lb: _smooth(rd[lb], sw) for lb in labels}
        sorted_labels = sorted(labels, key=lambda lb: -np.mean(np.abs(rd_s[lb])))
        for ci, lb in enumerate(sorted_labels):
            col = line_colors[ci % len(line_colors)]
            lw = 1.8 if ci < 3 else 1.2
            ax.plot(t_min, rd_s[lb], color=col, lw=lw, label=_legend_entry(chem, lb), zorder=5+len(sorted_labels)-ci)
        net_s = _smooth(net_dcdt, sw_net)
        ax.plot(t_min, net_s, color='black', ls='--', lw=1.5, label='Sum(rate) net', zorder=10)
        ax.axhline(0, color='black', lw=0.5, zorder=1)
        ax.set_title(f'({"abcd"[pi]}) {SPECIES_LABEL.get(sp_name, sp_name)}', fontweight='bold', loc='left')
        all_vals = np.concatenate([rd_s[lb] for lb in labels] + [net_s])
        if len(all_vals) > 0 and np.any(np.abs(all_vals) > 0):
            q_lo, q_hi = np.percentile(all_vals, [1, 99])
            span = q_hi - q_lo if (q_hi - q_lo) > 1e-30 else max(abs(q_hi), abs(q_lo)) * 0.5
            ax.set_ylim(q_lo - span*0.15, q_hi + span*0.15)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        ncol = 2 if len(sorted_labels) > 5 else 1
        ax.legend(loc='best', framealpha=0.92, fontsize=6, handlelength=1.5, ncol=ncol)
    axes[1,0].set_xlabel('Time (min)')
    axes[1,1].set_xlabel('Time (min)')
    axes[0,0].set_ylabel('Rate (M/s)')
    axes[1,0].set_ylabel('Rate (M/s)')
    fig.suptitle(f'Reaction rate contributions (DIW, Film+ab, ab={DETAIL_ALPHA})', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig2_rate_evolution.png')
    fig.savefig(_script_dir / 'fig2_rate_evolution.pdf')
    plt.close(fig)
    print('  -> fig2_rate_evolution saved')


def plot_fig3(alpha_results):
    """Fig 3: Radical concentrations table vs alpha_b."""
    rad_species = ['O3','O2NOOH','NO2_aq','ONOOH','O2NOO-','HO2','O2-','OH','ONOO-','O3-','N2O5']
    rad_labels  = ['O3','O2NOOH','NO2','ONOOH','O2NOO-','HO2','O2-','OH','ONOO-','O3-','N2O5']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    col_labels = ['Species', 'Order (M)']
    for ar in alpha_results:
        col_labels.append(f'ab = {ar["alpha_b"]}')
    cell_text = []
    for sp, lbl in zip(rad_species, rad_labels):
        vals = [ar['summary'].get(sp, 0) for ar in alpha_results]
        max_val = max(abs(v) for v in vals) if vals else 1e-30
        if max_val > 0:
            exp_order = int(np.floor(np.log10(max_val)))
        else:
            exp_order = -20
        scale = 10**(-exp_order)
        row = [lbl, f'1e{exp_order}']
        for v in vals:
            row.append(f'{v*scale:.2f}')
        cell_text.append(row)
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c4a6e')
        cell.set_text_props(color='white', fontweight='bold')
    for i in range(len(cell_text)):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(color)
            table[i+1, j].set_edgecolor('#cccccc')
        table[i+1, 0].set_text_props(fontweight='bold')
    ax.set_title('Radical and intermediate species concentrations\n(DIW, Film + ab BC, t = 720 s)', fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig3_radicals.png')
    fig.savefig(_script_dir / 'fig3_radicals.pdf')
    plt.close(fig)
    print('  -> fig3_radicals saved')


def plot_fig4(mass_bal, chem):
    """Fig 4: Mass balance horizontal bar chart."""
    src_color, snk_color = '#2166ac', '#c0392b'
    bar_h, gap = 0.6, 0.3
    panels = []
    panel_titles = {'NO3-': '(a) NO3-', 'O3': '(b) O3', 'NO2-': '(c) NO2-', 'H2O2': '(d) H2O2'}
    for sp_name in TARGET_SPECIES:
        mb = mass_bal[sp_name]
        labels, values, colors, ypos = [], [], [], []
        y = 0
        for lb, pct in mb['sources'][:5]:
            desc = _short_label(lb) + ': ' + rxn_description(chem, lb) if lb != 'MT' else 'MT: gas -> liq'
            labels.append(desc); values.append(pct); colors.append(src_color); ypos.append(y); y += 1
        if not mb['sources']:
            labels.append('(negligible)'); values.append(0); colors.append('#cccccc'); ypos.append(y); y += 1
        y += gap
        for lb, pct in mb['sinks'][:5]:
            desc = _short_label(lb) + ': ' + rxn_description(chem, lb) if lb != 'MT' else 'MT: liq -> gas'
            labels.append(desc); values.append(-pct); colors.append(snk_color); ypos.append(y); y += 1
        if not mb['sinks']:
            labels.append('(no sinks)'); values.append(0); colors.append('#cccccc'); ypos.append(y); y += 1
        y_extent = max(ypos) - min(ypos) + 0.6
        panels.append(dict(title=panel_titles[sp_name], labels=labels, values=values,
                           colors=colors, ypos=ypos, y_extent=y_extent,
                           info=f'Src={mb["src_total"]:.2e}, Snk={mb["snk_total"]:.2e} M/s'))
    scale = 10
    h = [max(int(p['y_extent']*scale), 1) for p in panels]
    gap_rows = 8
    n_rows = max(h[0]+gap_rows+h[2], h[1]+gap_rows+h[3])
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, wspace=0.05)
    ax_a = fig.add_subplot(gs[0:h[0], 0])
    ax_c = fig.add_subplot(gs[h[0]+gap_rows:h[0]+gap_rows+h[2], 0])
    ax_b = fig.add_subplot(gs[0:h[1], 1])
    ax_d = fig.add_subplot(gs[h[1]+gap_rows:h[1]+gap_rows+h[3], 1])
    axes_order = [ax_a, ax_b, ax_c, ax_d]
    for ax, p in zip(axes_order, panels):
        ax.barh(p['ypos'], p['values'], height=bar_h, color=p['colors'], edgecolor='black', linewidth=0.7)
        ax.set_yticks([])
        ax.set_ylim(max(p['ypos'])+0.35, min(p['ypos'])-0.35)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for yp, val, label in zip(p['ypos'], p['values'], p['labels']):
            if val <= 0:
                ax.text(2, yp, label, va='center', ha='left', fontsize=8.5)
            else:
                ax.text(-2, yp, label, va='center', ha='right', fontsize=8.5)
        for yp, v in zip(p['ypos'], p['values']):
            av = abs(v)
            if av < 0.1:
                continue
            if av > 40:
                x_txt = v - np.sign(v)*3
                ha = 'right' if v > 0 else 'left'
                color = 'white'
            else:
                x_txt = v + np.sign(v)*1.5
                ha = 'left' if v > 0 else 'right'
                color = 'black'
            ax.text(x_txt, yp, f'{av:.1f}%', va='center', ha=ha, fontsize=9, fontweight='bold', color=color)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlim(-100, 100)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.0f}'))
        ax.set_title(p['title'], fontsize=13, fontweight='bold', loc='left', pad=8)
    ax_a.set_xticklabels([])
    ax_b.set_xticklabels([])
    ax_c.set_xlabel('% of turnover')
    ax_d.set_xlabel('% of turnover')
    src_patch = mpatches.Patch(facecolor=src_color, edgecolor='black', linewidth=0.7, label='Source (+)')
    snk_patch = mpatches.Patch(facecolor=snk_color, edgecolor='black', linewidth=0.7, label='Sink (-)')
    fig.legend(handles=[snk_patch, src_patch], loc='upper right', fontsize=11, framealpha=0.9, bbox_to_anchor=(0.99, 0.98))
    fig.suptitle(f'Mass balance: reaction breakdown (DIW, ab={DETAIL_ALPHA}, 720s)', fontsize=13, y=1.0)
    fig.savefig(_script_dir / 'fig4_mass_balance.png')
    fig.savefig(_script_dir / 'fig4_mass_balance.pdf')
    plt.close(fig)
    print('  -> fig4_mass_balance saved')


def plot_fig5(result, solver, t_eval_5):
    """Fig 5: Spatial concentration profiles."""
    SPECIES_PANELS = [
        ('HONO2_total','NO3-', True, 'uM', 1e6),
        ('O3','O3', False, 'uM', 1e6),
        ('H2O2_total','H2O2', True, 'nM', 1e9),
        ('OH','OH', False, 'pM', 1e12),
        ('HO2_total','HO2', True, 'pM', 1e12),
    ]
    chem = solver.chem
    N_z, N_s = solver.N_z, solver.N_s
    z_mm = solver.z_centers * 1e3
    n_snaps = len(t_eval_5)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_snaps-1, 1)) for i in range(n_snaps)]
    n_panels = len(SPECIES_PANELS) + 1
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5*nrows), sharex=True)
    ax_pH = axes.flat[0]
    h_idx = chem.species_idx['H+']
    for si, tv in enumerate(t_eval_5):
        y2d = result['y_eval'][si].reshape(N_z, N_s)
        h_conc = np.clip(y2d[:, h_idx], 1e-14, None)
        ax_pH.plot(z_mm, -np.log10(h_conc), color=colors[si], lw=1.5, label=f'{tv/60:.0f} min')
    ax_pH.set_ylabel('pH')
    ax_pH.set_title('(a) pH', fontweight='bold', loc='left')
    ax_pH.legend(loc='best', fontsize=7)
    panel_labels = 'bcdefghij'
    for pi, (sp_name, sp_label, is_total, unit, scale) in enumerate(SPECIES_PANELS):
        ax = axes.flat[pi+1]
        idx = chem.species_idx.get(sp_name)
        if idx is None:
            ax.set_visible(False)
            continue
        for si, tv in enumerate(t_eval_5):
            y2d = result['y_eval'][si].reshape(N_z, N_s)
            profile = np.clip(y2d[:, idx], 1e-30, None) * scale
            ax.plot(z_mm, profile, color=colors[si], lw=1.5, label=f'{tv/60:.0f} min')
        ax.set_yscale('log')
        ax.set_ylabel(f'{sp_label} ({unit})')
        ax.set_title(f'({panel_labels[pi]}) {sp_label}', fontweight='bold', loc='left')
        if pi == 0:
            ax.legend(loc='best', fontsize=7)
    for i in range(n_panels, len(axes.flat)):
        axes.flat[i].set_visible(False)
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel('Depth (mm)')
            ax.set_xlim(0, z_mm[-1])
    fig.suptitle(f'Spatial profiles (DIW, Film+ab, ab={DETAIL_ALPHA})', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(_script_dir / 'fig5_spatial.png')
    fig.savefig(_script_dir / 'fig5_spatial.pdf')
    plt.close(fig)
    print('  -> fig5_spatial saved')



# =====================================================================
# Main orchestrator
# =====================================================================

def extract_mt_flux(solver, result, gas_conc_map):
    """Extract MT flux at each snapshot for Fig 1b."""
    N_z, N_s = solver.N_z, solver.N_s
    L = solver.L
    idx_to_name = {v: k for k, v in solver.species_idx.items()}
    mt_map = {}
    for aq_idx, k_mt, gas_sp, _, Ka in solver._interface_species:
        aq_name = idx_to_name[aq_idx]
        for g, a in GAS_TO_AQUEOUS_MAP.items():
            if a == aq_name:
                mt_map[g] = (aq_idx, k_mt, gas_sp, Ka)
                break
    snap_times = result['t_eval']
    mt_data = {gn: np.zeros(len(snap_times)) for gn, _ in MT_SPECIES}
    hp_idx = solver._h_plus_idx
    bc_type = solver.bc_type
    dz0 = solver.dz_cells[0]
    for si, tv in enumerate(snap_times):
        y2d = result['y_eval'][si].reshape(N_z, N_s)
        for gas_name, _ in MT_SPECIES:
            if gas_name not in mt_map:
                continue
            aq_idx, k_mt, gas_sp, Ka = mt_map[gas_name]
            if bc_type == 'dirichlet':
                D_l = LIQUID_DIFFUSIVITY.get(gas_name, D_LIQ_DEFAULT)
                mt_data[gas_name][si] = D_l * (y2d[0, aq_idx] - y2d[1, aq_idx]) / dz0 / L
            else:
                C_eq = solver._get_C_eq_fast(gas_sp, tv)
                C_surface = y2d[0, aq_idx]
                if Ka is not None and hp_idx >= 0:
                    h_s = max(y2d[0, hp_idx], 1e-14)
                    C_surface = C_surface * h_s / (h_s + Ka)
                mt_data[gas_name][si] = k_mt * (C_eq - C_surface) / L
    return snap_times, mt_data


def main():
    parser = argparse.ArgumentParser(description='Run simulations and generate all figures')
    parser.add_argument('--quick', action='store_true',
                        help='Skip heavy BC comparison (Fig1/1b), only run detail case')
    args = parser.parse_args()

    os.chdir(_project_root)
    times, gas_conc = load_gas_data()
    t_end = float(times[-1])
    print('=' * 70)
    print('Unified simulation + figure generation')
    print(f'  Gas data: {len(times)} points, t_end={t_end:.0f}s ({t_end/60:.0f} min)')
    print('=' * 70)

    # ── 1. BC comparison (Fig 1, Fig 1b) ──
    bc_results = []
    mt_results = []

    if not args.quick:
        print('\n[Phase 1] BC comparison (5 cases)...')
        mt_t_eval = np.arange(0, t_end + 0.1, 10.0)
        mt_t_eval = mt_t_eval[mt_t_eval <= t_end]
        for case in BC_CASES:
            result, solver, wall = run_case(
                times, gas_conc, case['bc_type'], case['alpha_b'], case['label'],
                t_eval=mt_t_eval, dt_poisson=None)
            summary = extract_summary(result, solver)
            summary['label'] = case['label']
            summary['wall'] = wall
            bc_results.append(summary)
            # MT flux
            snap_t, mt_data = extract_mt_flux(solver, result, gas_conc)
            mt_results.append({'label': case['label'], 'snap_times': snap_t, 'mt_data': mt_data})
    else:
        print('\n[Phase 1] SKIPPED (--quick mode)')

    # ── 2. Alpha_b sensitivity (Fig 3) ──
    print('\n[Phase 2] Alpha_b sensitivity (3 cases)...')
    alpha_results = []
    for ab in ALPHA_CASES:
        result, solver, wall = run_case(
            times, gas_conc, 'film_alpha', ab, f'Film+ab={ab}',
            dt_poisson=None)
        summary = extract_summary(result, solver)
        summary['alpha_b'] = ab
        summary['wall'] = wall
        alpha_results.append({'alpha_b': ab, 'summary': summary})
        # If this is the detail alpha, also add to bc_results for completeness
        if ab == DETAIL_ALPHA and not args.quick:
            summary_copy = dict(summary)
            summary_copy['label'] = f'Film+ab={ab}'
            bc_results.append(summary_copy)

    # ── 3. Detail simulation for Fig 2 (rate evolution) + Fig 4 (mass balance) + Fig 5 (spatial) ──
    print(f'\n[Phase 3] Detail simulation (ab={DETAIL_ALPHA}, snapshots every {DT_SNAPSHOT}s)...')
    # Dense t_eval for rate evolution
    t_eval_rate = np.arange(DT_SNAPSHOT, t_end + 0.1, DT_SNAPSHOT)
    t_eval_rate = t_eval_rate[t_eval_rate <= t_end + 0.1]

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        mass_transfer_eta=1.0, saline_mode=False,
        bc_type='film_alpha', alpha_b=DETAIL_ALPHA,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=0, hono2_gas=0, h2o2_gas=0)

    y0 = solver.build_initial_condition(initial_pH=7.0)
    N_z, N_s = solver.N_z, solver.N_s

    print(f'  Running detail sim: {len(t_eval_rate)} eval points')
    t0 = time_mod.time()
    result_detail = solver.solve(
        t_span=(0, t_end), t_eval=t_eval_rate,
        verbose=True, dt_poisson=None,
    )
    print(f'  Detail sim: {time_mod.time()-t0:.1f}s')

    # Prepend t=0
    snap_times = np.concatenate([[0.0], result_detail['t_eval']])
    snaps = [y0.reshape(N_z, N_s).copy()]
    for i in range(len(result_detail['t_eval'])):
        snaps.append(result_detail['y_eval'][i].reshape(N_z, N_s).copy())

    dense_output = result_detail.get('dense_output', [])

    # Compute per-reaction rates
    print('  Computing per-reaction rates...')
    t0 = time_mod.time()
    nt = len(snap_times)
    all_rxn_rates, all_mt_flux = [], []
    use_simpson = len(dense_output) > 0
    if use_simpson:
        dense_sol = dense_output[0]
        rr0, mf0 = compute_rates_snapshot(solver, snaps[0], snap_times[0])
        all_rxn_rates.append(rr0)
        all_mt_flux.append(mf0)
        for i in range(1, nt):
            rr, mf = compute_rates_simpson(solver, dense_sol, snap_times[i-1], snap_times[i])
            all_rxn_rates.append(rr)
            all_mt_flux.append(mf)
            if i % 50 == 0:
                print(f'    {i}/{nt-1} intervals')
    else:
        for i, (tv, y2d) in enumerate(zip(snap_times, snaps)):
            rr, mf = compute_rates_snapshot(solver, y2d, tv)
            all_rxn_rates.append(rr)
            all_mt_flux.append(mf)
    print(f'  Rate computation: {time_mod.time()-t0:.1f}s')

    # Build species_data for Fig 2
    species_data = {}
    for sp_name in TARGET_SPECIES:
        labels, rd, budget_net = build_timeseries(snap_times, all_rxn_rates, all_mt_flux, sp_name)
        species_data[sp_name] = (labels, rd, budget_net)

    # Mass balance for Fig 4
    y_final = snaps[-1]
    mass_bal = build_mass_balance(solver, y_final, t_end)

    # Fig 5 spatial: re-run with specific time snapshots
    print(f'\n[Phase 4] Spatial profile simulation...')
    t_eval_5 = np.array([t*60.0 for t in SNAP_TIMES_MIN])
    t_eval_5 = t_eval_5[t_eval_5 <= t_end]
    result_5, solver_5, _ = run_case(
        times, gas_conc, 'film_alpha', DETAIL_ALPHA, 'Spatial',
        t_eval=t_eval_5, dt_poisson=None)

    # ── Generate all figures ──
    print('\n' + '=' * 70)
    print('Generating figures...')
    print('=' * 70)

    if bc_results:
        plot_fig1(bc_results)
    if mt_results:
        plot_fig1b(mt_results)
    plot_fig2(snap_times, species_data, chem)
    plot_fig3(alpha_results)
    plot_fig4(mass_bal, chem)
    plot_fig5(result_5, solver_5, t_eval_5)

    # ── Summary table ──
    print('\n' + '=' * 70)
    print('Summary')
    print('=' * 70)
    if bc_results:
        print(f'{"BC":20s} {"pH":>6s} {"NO3-(uM)":>10s} {"NO2-(uM)":>10s} {"H2O2(uM)":>10s} {"wall(s)":>8s}')
        for r in bc_results:
            print(f'{r["label"]:20s} {r["pH"]:6.3f} {r["NO3"]:10.1f} {r["NO2"]:10.3f} {r["H2O2"]:10.3f} {r["wall"]:8.1f}')
    print()
    print(f'{"alpha_b":>8s} {"pH":>6s} {"NO3-(uM)":>10s} {"OH(M)":>12s} {"O3(M)":>12s} {"HO2(M)":>12s}')
    for ar in alpha_results:
        s = ar['summary']
        print(f'{ar["alpha_b"]:8.3f} {s["pH"]:6.3f} {s["NO3"]:10.1f} {s["OH"]:12.3e} {s["O3"]:12.3e} {s["HO2"]:12.3e}')

    print('\nAll figures saved in Figures/')
    print('Done!')


if __name__ == '__main__':
    main()
