#!/usr/bin/env python3
"""Reaction pathway figures (absolute rate, M/s) for H2O2/NO2-/NO3-.

One figure per species (3 figures total), each with the three voltages
(2.6/3.2/3.6 kV) plotted on the same axis as a grouped diverging bar chart.

Layout:
    y-axis  : reaction labels (sources on top, sinks below)
    x-axis  : net rate to the mass pool [M/s], symlog scale
    color   : voltage (3 levels)
    sign    : sign of bar (positive = source, negative = sink)

Sources/sinks are pooled across voltages: the top-N reactions by
maximum absolute rate (across voltages) are kept, so a reaction that is
significant only at one voltage still appears.

Reference cache: {V}kV_Humid_fitting_three_film_HONOvar_v2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_FIG_DIR))
import gen_all_figures as gaf  # noqa: E402

VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']
FOLDER_TPL = 'DIW results/{V}_Humid_fitting_three_film_HONOvar_v2'
CACHE_NAME = 'three_film_abspecies_dg0.0100.npz'
TARGET_SPECIES = ['H2O2', 'NO2-', 'NO3-']
MIN_FRAC = 0.01            # drop reactions whose share of Σsrc/Σsnk is <1%
                           # at every voltage
VOLTAGE_COLORS = {
    '2.6kV': '#fdae61',    # light orange
    '3.2kV': '#d7301f',    # red
    '3.6kV': '#7f0000',    # dark red
}


def _load_voltage(voltage: str):
    gaf.DEFAULT_GAS_SHEET = voltage
    times, gas_conc = gaf.load_gas_data()
    cache_path = _FIG_DIR / FOLDER_TPL.format(V=voltage) / 'cache' / CACHE_NAME
    cache = dict(np.load(cache_path, allow_pickle=True))
    y_final = cache['y_final']
    solver = gaf._get_solver(times, gas_conc)
    rxn_rates, mt_flux = gaf.compute_rates_snapshot(
        solver, y_final, float(times[-1]))
    return rxn_rates, mt_flux


def _contribs_dict(rxn_rates, mt_flux, species: str) -> dict[str, float]:
    return dict(gaf.species_contribution(rxn_rates, species, mt_flux))


def _select_rows(species: str, all_data: dict):
    """Return ((src_rows, snk_rows), totals).

    Each row: (label, [rate_2.6, rate_3.2, rate_3.6]).
    Reaction kept if its rate at *any* voltage is ≥ MIN_FRAC of that voltage's
    Σsrc (for sources) or Σsnk (for sinks). Sign assigned by largest-|rate|
    voltage.
    """
    contribs_by_v = {v: _contribs_dict(*all_data[v], species) for v in VOLTAGES}
    all_labels = set().union(*[c.keys() for c in contribs_by_v.values()])

    totals = {}
    for v in VOLTAGES:
        tot_s = sum(max(0.0, r) for r in contribs_by_v[v].values())
        tot_k = -sum(min(0.0, r) for r in contribs_by_v[v].values())
        totals[v] = (tot_s, tot_k)

    src_rows, snk_rows = [], []
    for lab in all_labels:
        rates = [contribs_by_v[v].get(lab, 0.0) for v in VOLTAGES]
        if max(abs(r) for r in rates) == 0:
            continue
        dominant = max(rates, key=lambda r: abs(r))
        sign = 1 if dominant > 0 else -1

        if sign > 0:
            keep = any(rates[i] / totals[v][0] >= MIN_FRAC
                       for i, v in enumerate(VOLTAGES) if totals[v][0] > 0)
            if keep:
                src_rows.append((lab, rates, max(rates)))
        else:
            keep = any(-rates[i] / totals[v][1] >= MIN_FRAC
                       for i, v in enumerate(VOLTAGES) if totals[v][1] > 0)
            if keep:
                snk_rows.append((lab, rates, min(rates)))

    src_rows.sort(key=lambda x: -x[2])
    snk_rows.sort(key=lambda x: x[2])
    src_rows = [(lab, rates) for lab, rates, _ in src_rows]
    snk_rows = [(lab, rates) for lab, rates, _ in snk_rows]
    return (src_rows, snk_rows), totals


def _plot_species(species: str, all_data: dict):
    (src_rows, snk_rows), totals = _select_rows(species, all_data)
    n_src = len(src_rows)
    n_snk = len(snk_rows)
    n_total = n_src + n_snk
    if n_total == 0:
        print(f'  no rows for {species}')
        return

    fig, ax = plt.subplots(figsize=(13, max(5, 0.55 * n_total + 2.5)))
    n_v = len(VOLTAGES)
    bar_h = 0.22

    # Collect (y_center, label, rates, role) so source rows sit on top.
    layout = []
    y = 0
    for lab, rates in src_rows:
        layout.append((y, lab, rates, 'src'))
        y += 1
    if n_src and n_snk:
        y += 0.4   # gap between source and sink groups
    snk_top_y = y
    for lab, rates in snk_rows:
        layout.append((y, lab, rates, 'snk'))
        y += 1

    # Plot grouped bars.
    for y_c, lab, rates, role in layout:
        for i, v in enumerate(VOLTAGES):
            offset = (i - (n_v - 1) / 2) * bar_h
            val = rates[i] if role == 'src' else -abs(rates[i])
            # If a "source" reaction happens to have a sink-direction value at
            # this voltage (or vice versa), keep the actual signed value:
            val = rates[i]
            ax.barh(y_c + offset, val, height=bar_h * 0.92,
                    color=VOLTAGE_COLORS[v], edgecolor='black', lw=0.5)

    yticks = [y_c for y_c, *_ in layout]
    yticklabels = [lab for _, lab, *_ in layout]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.invert_yaxis()

    # Group separator line and section labels.
    if n_src and n_snk:
        sep_y = (yticks[n_src - 1] + yticks[n_src]) / 2
        ax.axhline(sep_y, color='gray', lw=0.6, ls='--', alpha=0.6)
        ax.text(0.99, yticks[0] - 0.7, 'Sources',
                transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontweight='bold',
                color='#2166ac', fontsize=11)
        ax.text(0.99, yticks[-1] + 0.7, 'Sinks',
                transform=ax.get_yaxis_transform(),
                ha='right', va='top', fontweight='bold',
                color='#c0392b', fontsize=11)

    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Net rate to mass pool [M/s]   (positive = source, negative = sink)')
    ax.grid(True, axis='x', alpha=0.25)

    # Symmetric linear x-limit, padded slightly above max |rate|.
    all_vals = [rates[i] for _, _, rates, _ in layout for i in range(n_v)]
    if all_vals:
        max_abs = max(abs(v) for v in all_vals)
        x_max = max_abs * 1.08
        ax.set_xlim(-x_max, x_max)

    # Voltage legend.
    handles = [mpatches.Patch(facecolor=VOLTAGE_COLORS[v],
                              edgecolor='black', lw=0.5, label=v)
               for v in VOLTAGES]
    ax.legend(handles=handles, loc='lower right', fontsize=10,
              title='Voltage', title_fontsize=10, frameon=True)

    # Title with totals per voltage.
    title_sp = gaf._uni(species)
    sub = '   '.join(
        f'{v}: Σsrc={totals[v][0]:.2e} / Σsnk={totals[v][1]:.2e}'
        for v in VOLTAGES)
    fig.suptitle(
        f'Reaction pathway: {title_sp}   '
        f'(DIW, three_film, HONOvar_v2, t=600 s, volume-averaged)\n'
        + sub,
        fontsize=11, y=0.995)
    fig.tight_layout()

    out_dir = _FIG_DIR / 'DIW results'
    fname = species.replace('-', 'm').replace('+', 'p')
    fig.savefig(out_dir / f'fig_pathway_{fname}.png',
                dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / f'fig_pathway_{fname}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved fig_pathway_{fname}.{{png,pdf}}'
          f'   ({n_src} src + {n_snk} snk rows)')


def main():
    all_data = {}
    for v in VOLTAGES:
        print(f'Loading {v}...')
        all_data[v] = _load_voltage(v)
    for sp in TARGET_SPECIES:
        print(f'\nGenerating pathway figure for {sp}...')
        _plot_species(sp, all_data)


if __name__ == '__main__':
    main()
