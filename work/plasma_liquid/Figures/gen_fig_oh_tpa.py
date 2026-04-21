#!/usr/bin/env python3
"""
Figure: OH radical quantification via TPA → hTPA fluorescence probe comparison.

Panels:
  (a) [hTPA] bar chart: 3 voltages × Sim vs Exp (inner-filter corrected)
  (b) [OH] spatial profile @ t = 1, 5, 10 min (3 voltages)
  (c) Cumulative [hTPA](t) vs experiment endpoint
  (d) pH(t) time-series (should stay ≥ 10)
  (e) [TPA](t)/[TPA]₀ depletion
  (f) OH sink contribution pie (3.2 kVpp, volume-integrated over 10 min)

Reads .npz caches from Figures/cache/tpa/ produced by run_tpa_alkaline.py.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'Ver4_1D'))

CACHE_DIR = _script_dir / 'cache' / 'tpa'
CONDITION = 'humidfitting'   # 'dry' | 'humidfitting' — matches run_tpa_alkaline.py cond_tag
OUT_PNG = _script_dir / f'fig_oh_tpa_{CONDITION}.png'
OUT_PDF = _script_dir / f'fig_oh_tpa_{CONDITION}.pdf'

VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
EXPERIMENT = {'2.6kV': 12.66, '3.2kV': 57.72, '3.6kV': 43.26}   # µM hTPA
COLORS = {'2.6kV': '#1f77b4', '3.2kV': '#d62728', '3.6kV': '#2ca02c'}


def load_case(voltage: str, tpa: bool = True):
    tag = 'tpa2000uM' if tpa else 'notpa'
    f_new = CACHE_DIR / f"{voltage}_{tag}_{CONDITION}.npz"
    f_old = CACHE_DIR / f"{voltage}_{tag}.npz"
    f = f_new if f_new.exists() else f_old
    if not f.exists():
        return None
    return dict(np.load(f, allow_pickle=True))


def species_idx_map(data):
    keys = data['species_idx_keys']
    vals = data['species_idx_vals']
    return {str(k): int(v) for k, v in zip(keys, vals)}


def volume_avg(snap_y, dz_cells, idx):
    """snap_y shape (T, N_z, N_s) — return (T,) volume-averaged conc."""
    L = dz_cells.sum()
    return (snap_y[:, :, idx] * dz_cells[None, :]).sum(axis=1) / L


def plot_all():
    cases_tpa = {v: load_case(v, True) for v in VOLTAGES}
    missing = [v for v, d in cases_tpa.items() if d is None]
    if missing:
        print(f"WARNING: missing cache for {missing}. Run run_tpa_alkaline.py first.")
        return

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.32)

    # (a) hTPA bar chart
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(VOLTAGES))
    width = 0.35
    sim_vals = [float(cases_tpa[v]['hTPA_uM']) for v in VOLTAGES]
    exp_vals = [EXPERIMENT[v] for v in VOLTAGES]
    ax.bar(x - width/2, sim_vals, width, label='Sim', color='steelblue', edgecolor='k')
    ax.bar(x + width/2, exp_vals, width, label='Exp (IF×2)', color='lightsalmon', edgecolor='k')
    ax.set_xticks(x); ax.set_xticklabels(VOLTAGES)
    ax.set_ylabel('[hTPA] (µM)')
    ax.set_title('(a) hTPA: Simulation vs Experiment')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    for i, (s, e) in enumerate(zip(sim_vals, exp_vals)):
        err = (s - e) / e * 100
        ax.text(i, max(s, e) * 1.04, f'{err:+.0f}%', ha='center', fontsize=9)

    # (b) OH spatial profile
    ax = fig.add_subplot(gs[0, 1])
    snap_times_target = [60.0, 300.0, 600.0]
    linestyles = ['--', '-.', '-']
    for v in VOLTAGES:
        d = cases_tpa[v]
        idx = species_idx_map(d)
        oh_i = idx.get('OH', -1)
        if oh_i < 0:
            continue
        snap_y = d['snap_y']; snap_t = d['snap_t']; z = d['z_centers']
        for tgt, ls in zip(snap_times_target, linestyles):
            i_t = int(np.argmin(np.abs(snap_t - tgt)))
            oh_prof = snap_y[i_t, :, oh_i]
            ax.plot(z * 1e3, oh_prof, color=COLORS[v], ls=ls,
                    label=f'{v} @ {snap_t[i_t]:.0f}s' if v == VOLTAGES[0] else None)
    ax.set_yscale('log')
    ax.set_xlabel('z (mm)'); ax.set_ylabel('[OH] (M)')
    ax.set_title('(b) OH spatial profile')
    ax.grid(alpha=0.3, which='both')
    # legend by voltage
    for v in VOLTAGES:
        ax.plot([], [], color=COLORS[v], label=v)
    ax.legend(fontsize=8, ncol=2)

    # (c) cumulative hTPA(t)
    ax = fig.add_subplot(gs[0, 2])
    for v in VOLTAGES:
        d = cases_tpa[v]
        idx = species_idx_map(d)
        htpa_i = idx.get('hTPA', -1)
        if htpa_i < 0:
            continue
        dz = d['dz_cells']
        htpa_t = volume_avg(d['snap_y'], dz, htpa_i) * 1e6
        ax.plot(d['snap_t'], htpa_t, color=COLORS[v], label=v, lw=2)
        ax.axhline(EXPERIMENT[v], color=COLORS[v], ls=':', alpha=0.6)
    ax.set_xlabel('time (s)'); ax.set_ylabel('[hTPA] (µM)')
    ax.set_title('(c) Cumulative hTPA (dotted = exp endpoint)')
    ax.legend(); ax.grid(alpha=0.3)

    # (d) pH(t)
    ax = fig.add_subplot(gs[1, 0])
    for v in VOLTAGES:
        d = cases_tpa[v]
        idx = species_idx_map(d)
        hp_i = idx.get('H+', -1)
        dz = d['dz_cells']
        h_t = volume_avg(d['snap_y'], dz, hp_i)
        pH = -np.log10(np.maximum(h_t, 1e-14))
        ax.plot(d['snap_t'], pH, color=COLORS[v], label=v, lw=2)
    ax.axhline(10, color='gray', ls=':', label='pH=10')
    ax.set_xlabel('time (s)'); ax.set_ylabel('pH')
    ax.set_title('(d) pH time-series')
    ax.legend(); ax.grid(alpha=0.3)

    # (e) [TPA] depletion
    ax = fig.add_subplot(gs[1, 1])
    for v in VOLTAGES:
        d = cases_tpa[v]
        idx = species_idx_map(d)
        tpa_i = idx.get('TPA', -1)
        dz = d['dz_cells']
        tpa_t = volume_avg(d['snap_y'], dz, tpa_i)
        ax.plot(d['snap_t'], tpa_t / 2e-3, color=COLORS[v], label=v, lw=2)
    ax.set_xlabel('time (s)'); ax.set_ylabel('[TPA] / [TPA]₀')
    ax.set_title('(e) TPA depletion')
    ax.set_ylim([0.9, 1.01])
    ax.legend(); ax.grid(alpha=0.3)

    # (f) OH budget @ 3.2 kVpp: TPA sink vs others (snapshot at t=600s)
    ax = fig.add_subplot(gs[1, 2])
    d = cases_tpa['3.2kV']
    idx = species_idx_map(d)
    dz = d['dz_cells']; L = dz.sum()
    # Compute rough sink rates at final snapshot
    y_final = d['snap_y'][-1]  # (N_z, N_s)
    def _cavg(sp):
        i = idx.get(sp, -1)
        if i < 0: return 0.0
        return float((y_final[:, i] * dz).sum() / L)
    oh = _cavg('OH')
    tpa = _cavg('TPA'); htpa = _cavg('hTPA')
    ohm = _cavg('OH-'); ho2m = _cavg('HO2-'); h2o2 = _cavg('H2O2')
    o3 = _cavg('O3'); hO2 = _cavg('HO2')
    sinks = {
        'TPA':      4.0e9 * tpa * oh,
        'OH⁻':      1.2e10 * ohm * oh,
        'HO₂⁻':     7.5e9 * ho2m * oh,
        'H₂O₂':     2.7e7 * h2o2 * oh,
        'O₃':       3.0e9 * o3 * oh,
        'hTPA':     1.0e9 * htpa * oh,
        'HO₂':      7.1e9 * hO2 * oh,
        '2 OH':     2 * 5.5e9 * oh * oh,
    }
    total = sum(sinks.values())
    if total > 0:
        labels = list(sinks.keys())
        sizes = [sinks[k] / total * 100 for k in labels]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=plt.cm.tab10.colors[:len(labels)])
        ax.set_title(f'(f) OH sinks @ 3.2kV, t=600s\n(total rate = {total:.2e} M/s)')
    else:
        ax.text(0.5, 0.5, 'No OH present', ha='center', va='center',
                transform=ax.transAxes)

    fig.suptitle(f'OH radical quantification: TPA→hTPA fluorescence probe '
                 f'vs plasma-liquid 1D simulation  [{CONDITION.upper()}]',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == '__main__':
    plot_all()
