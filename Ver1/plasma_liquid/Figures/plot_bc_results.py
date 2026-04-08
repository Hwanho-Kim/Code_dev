#!/usr/bin/env python3
"""
Plot BC comparison and α_b sensitivity results from CLAUDE.md records.

Figures:
  1. BC model comparison (bar chart): pH, NO2⁻, NO3⁻, H2O2
  2. α_b sensitivity (line+scatter): pH, NO3⁻ with experimental intersection
  3. Radical concentrations vs α_b
  4. Mass balance summary
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Style ──
BAR_COLOR = '#4878a8'
BAR_EDGE = '#2c4a6e'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
})

# ── Experimental targets (DIW, 3.2 kVpp) ──
EXP = {'pH': 3.61, 'NO3': 63.0, 'NO2': 3.0, 'H2O2': 11.0}

# ═══════════════════════════════════════════════════════════════
# Data from monolithic BDF runs (2026-04-01, atol=1e-12, dt_enforce=None)
# ═══════════════════════════════════════════════════════════════

# BC comparison (DIW, 측정종만 mode, monolithic BDF)
BC_DATA = {
    'labels': ['Two-film', 'Dirichlet', 'Film\n(αb=1)',
               'Film\nαb=0.05', 'Film\nαb=0.01'],
    'pH':     [2.40,  2.92,  3.36,  3.72,  4.27],
    'NO3':    [3936.2, 1199.3, 438.6, 191.9, 54.1],
    'NO2':    [2.399e-3, 3.043e-2, 5.666e-3, 3.196e-3, 2.438e-3],
    'H2O2':   [3.190e-3, 1.389e-3, 1.448e-3, 1.092e-3, 6.556e-4],
}

# α_b sensitivity (DIW, 측정종만 mode, monolithic BDF)
ALPHA_DATA = {
    'alpha_b': [0.01,   0.03,   0.05],
    'pH':      [4.267,  3.869,  3.717],
    'NO3':     [54.1,   135.1,  191.9],
    'NO2':     [0.0,    0.0,    0.0],
    'H2O2':    [0.0,    0.0,    0.0],
    # Radical bulk avg (M)
    'OH':      [1.718e-11, 2.353e-11, 2.729e-11],
    'O3':      [5.716e-08, 1.177e-07, 1.682e-07],
    'HO2':     [5.035e-11, 1.032e-10, 1.353e-10],
}


def fig1_bc_comparison():
    """Bar chart comparing BC model performance: pH, NO2⁻, NO3⁻, H2O2."""
    labels = BC_DATA['labels']
    x = np.arange(len(labels))
    w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    panels = [
        ('pH',        BC_DATA['pH'],   EXP['pH'],   '',    False, (1.5, 5.0)),
        ('NO₂⁻ (µM)', BC_DATA['NO2'],  EXP['NO2'],  'µM',  False, (-0.5, 5.0)),
        ('NO₃⁻ (µM)', BC_DATA['NO3'],  EXP['NO3'],  'µM',  True,  (20, 15000)),
        ('H₂O₂ (µM)', BC_DATA['H2O2'], EXP['H2O2'], 'µM',  False, (-1, 15)),
    ]

    for i, (ylabel, data, exp_val, unit, use_log, ylim) in enumerate(panels):
        ax = axes.flat[i]
        bars = ax.bar(x, data, w, color=BAR_COLOR, edgecolor='black',
                      linewidth=0.8, alpha=0.85)

        exp_label = f'Exp. ({exp_val}{" " + unit if unit else ""})'
        ax.axhline(exp_val, color='k', ls='--', lw=1.2, label=exp_label)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(ylim)
        ax.set_title(f'({"abcd"[i]}) {ylabel}')

        if use_log:
            ax.set_yscale('log')

        # value labels
        for bar, val in zip(bars, data):
            if use_log and val > 0:
                ypos = val * 1.2
                txt = f'{val:.0f}'
            else:
                ypos = max(val, 0) + (ylim[1] - ylim[0]) * 0.02
                if val >= 0.1:
                    txt = f'{val:.1f}'
                elif val > 0:
                    # Show in nM for sub-0.1 µM values
                    val_nM = val * 1e3
                    txt = f'{val_nM:.1f} nM' if val_nM >= 1 else f'{val_nM:.2f} nM'
                else:
                    txt = '0'
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    txt, ha='center', va='bottom', fontsize=8.5)

    # Collect one legend handle from any subplot (dashed line = Exp.)
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, ['Exp.'], loc='upper right', framealpha=0.9,
               fontsize=10, bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('Effect of gas–liquid interface BC model (DIW, 3.2 kVpp, 12 min)',
                 fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.93, 1.0])
    fig.savefig('Figures/fig1_bc_comparison.png')
    fig.savefig('Figures/fig1_bc_comparison.pdf')
    print("  → fig1_bc_comparison.png/pdf saved")
    plt.close(fig)


def fig2_alpha_sensitivity():
    """α_b sensitivity: pH and NO3⁻ vs accommodation coefficient."""
    ab = np.array(ALPHA_DATA['alpha_b'])
    pH = np.array(ALPHA_DATA['pH'])
    NO3 = np.array(ALPHA_DATA['NO3'])

    fig, ax1 = plt.subplots(figsize=(6, 4.5))

    color_pH = '#2c4a6e'
    color_NO3 = '#a03030'

    # pH (left axis)
    ax1.plot(ab, pH, 'o-', color=color_pH, label='pH', zorder=5)
    ax1.axhline(EXP['pH'], color=color_pH, ls='--', lw=1, alpha=0.5)
    ax1.set_xlabel('Accommodation coefficient αb')
    ax1.set_ylabel('pH', color=color_pH)
    ax1.tick_params(axis='y', labelcolor=color_pH)
    ax1.set_ylim(3.4, 4.5)

    # NO3⁻ (right axis)
    ax2 = ax1.twinx()
    ax2.plot(ab, NO3, 's-', color=color_NO3, label='NO₃⁻', zorder=5)
    ax2.axhline(EXP['NO3'], color=color_NO3, ls='--', lw=1, alpha=0.5)
    ax2.set_ylabel('NO₃⁻ (µM)', color=color_NO3)
    ax2.tick_params(axis='y', labelcolor=color_NO3)
    ax2.set_ylim(0, 210)

    # Interpolate to find α_b where NO3⁻ = 63 µM
    f_NO3 = interp1d(NO3, ab, kind='linear', fill_value='extrapolate')
    ab_opt = float(f_NO3(EXP['NO3']))
    f_pH = interp1d(ab, pH, kind='linear', fill_value='extrapolate')
    pH_at_opt = float(f_pH(ab_opt))

    # Mark intersection
    ax1.axvline(ab_opt, color='gray', ls=':', lw=1, alpha=0.6)
    ax2.plot(ab_opt, EXP['NO3'], '*', color=color_NO3, markersize=14, zorder=10)
    ax1.plot(ab_opt, pH_at_opt, '*', color=color_pH, markersize=14, zorder=10)

    ax1.annotate(f'αb ≈ {ab_opt:.3f}\npH ≈ {pH_at_opt:.2f}',
                 xy=(ab_opt, pH_at_opt), xytext=(ab_opt + 0.008, pH_at_opt + 0.15),
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    ax1.set_title(f'αb sensitivity — DIW (Film + αb BC)\n'
                  f'Exp. NO₃⁻ = {EXP["NO3"]} µM → αb ≈ {ab_opt:.3f}')
    fig.tight_layout()
    fig.savefig('Figures/fig2_alpha_sensitivity.png')
    fig.savefig('Figures/fig2_alpha_sensitivity.pdf')
    print(f"  → fig2_alpha_sensitivity.png/pdf saved  (αb_opt ≈ {ab_opt:.4f})")
    plt.close(fig)


def fig3_radicals_vs_alpha():
    """Radical/intermediate species concentration table vs α_b (all 11 species)."""
    # Data from monolithic BDF runs (2026-04-01, atol=1e-12, dt_enforce=None)
    species_data = [
        # (name,       α_b=0.01,    α_b=0.03,    α_b=0.05,    unit, scale)
        ('O₃',         5.716e-08,   1.177e-07,   1.682e-07,   -8,   1e8),
        ('O₂NOOH',     3.999e-09,   1.341e-08,   2.115e-08,   -9,   1e9),
        ('NO₂',        1.005e-09,   1.497e-09,   1.749e-09,   -9,   1e9),
        ('ONOOH',      2.133e-10,   4.101e-10,   5.407e-10,   -10,  1e10),
        ('O₂NOO⁻',     9.302e-11,   1.249e-10,   1.387e-10,   -10,  1e10),
        ('HO₂',        5.035e-11,   1.032e-10,   1.353e-10,   -11,  1e11),
        ('O₂⁻',        1.475e-11,   1.210e-11,   1.118e-11,   -11,  1e11),
        ('OH',          1.718e-11,   2.353e-11,   2.729e-11,   -11,  1e11),
        ('ONOO⁻',      9.903e-13,   7.624e-13,   7.077e-13,   -13,  1e13),
        ('O₃⁻',        2.392e-15,   1.525e-15,   1.301e-15,   -15,  1e15),
        ('N₂O₅',       8.520e-17,   2.559e-16,   4.268e-16,   -16,  1e16),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    # Build table data
    col_labels = ['Species', 'Order (M)', 'αb = 0.01', 'αb = 0.03', 'αb = 0.05']
    cell_text = []
    for name, c01, c03, c05, exp, scale in species_data:
        v01, v03, v05 = c01 * scale, c03 * scale, c05 * scale
        def fmt(v):
            if v >= 100:
                return f'{v:.1f}'
            elif v >= 1:
                return f'{v:.2f}'
            else:
                return f'{v:.2f}'
        order_str = f'$10^{{{exp}}}$'
        cell_text.append([name, order_str, fmt(v01), fmt(v03), fmt(v05)])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c4a6e')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row shading
    for i in range(len(cell_text)):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)
            table[i + 1, j].set_edgecolor('#cccccc')
        # Bold species name
        table[i + 1, 0].set_text_props(fontweight='bold')

    # Header edge color
    for j in range(len(col_labels)):
        table[0, j].set_edgecolor('#cccccc')

    ax.set_title('Radical and intermediate species concentrations\n'
                 r'(DIW, Film + $\alpha_b$ BC, $\alpha_b$ = 0.01 / 0.03 / 0.05, t = 720 s)',
                 fontsize=12, pad=20)

    fig.tight_layout()
    fig.savefig('Figures/fig3_radicals.png')
    fig.savefig('Figures/fig3_radicals.pdf')
    print("  → fig3_radicals.png/pdf saved")
    plt.close(fig)


def fig4_mass_balance_detailed():
    """Reaction-by-reaction source/sink breakdown (α_b=0.03).

    Horizontal bar chart per species: each reaction = separate bar.
    Reaction labels on the LEFT (y-axis), percentage on the RIGHT.
    Source (blue) and Sink (red) groups separated by dashed line.
    Data from run_alpha_analysis.py output.
    """
    src_color = '#2166ac'
    snk_color = '#c0392b'

    species_budget = [
        {
            'title': '(a) NO₃⁻',
            'info': 'Σsrc = 6.30×10⁻⁸ M/s, accumulating',
            'sources': [
                ('R98: N₂O₅ + H₂O → 2NO₃⁻', 99.6),
            ],
            'sinks': [
                ('MT: liq → gas', 100.0),
            ],
        },
        {
            'title': '(b) O₃',
            'info': 'Σsrc = 2.07×10⁻⁹ M/s, near steady state',
            'sources': [
                ('MT: gas → liq', 100.0),
            ],
            'sinks': [
                ('R27: O₃ + OH → HO₂ + O₂', 51.9),
                ('R28: O₃ + HO₂ → O₂ + HO₃', 39.2),
                ('R25: O₃ + O₂⁻ → O₃⁻ + O₂', 8.2),
            ],
        },
        {
            'title': '(c) NO₂⁻',
            'info': 'Σsrc = 6.05×10⁻¹¹ M/s, net production',
            'sources': [
                ('R19: 2NO₂ → NO₂⁻ + NO₃⁻', 52.5),
                ('R95: N₂O₄ + H₂O → NO₂⁻ + NO₃⁻', 30.9),
                ('R87: O₂NOO⁻ → NO₂⁻ + O₂', 14.0),
                ('R86: O₂NOOH → NO₂⁻ + HO₂', 2.5),
            ],
            'sinks': [
                ('R92: NO₃ + NO₂⁻ → NO₃⁻ + NO₂', 57.5),
                ('R32: O₃ + NO₂⁻ → O₂ + NO₃⁻', 24.4),
                ('R77: OH + NO₂⁻ → OH⁻ + NO₂', 11.5),
                ('R78: OH + HONO → NO₂', 6.1),
            ],
        },
        {
            'title': '(d) H₂O₂',
            'info': 'Σsrc = 4.88×10⁻¹³ M/s, net production',
            'sources': [
                ('R45: 2OH → H₂O₂', 88.6),
                ('R22: OH + HO₂ → H₂O₂ + O₂', 6.0),
                ('R53: HO₂ + HO₂ → H₂O₂ + O₂', 2.4),
            ],
            'sinks': [
                ('R41: OH + H₂O₂ → HO₂', 46.2),
                ('MT: liq → gas', 40.8),
                ('R91: NO₃ + H₂O₂ → NO₃⁻ + HO₂', 13.0),
            ],
        },
    ]

    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    bar_h = 0.6
    gap = 0.3

    # ── 1st pass: compute y-positions for each panel ──
    panels = []
    for sp in species_budget:
        labels, values, colors, ypos = [], [], [], []
        y = 0

        n_src = len(sp['sources'])
        for label, pct in sp['sources']:
            labels.append(label); values.append(pct)
            colors.append(src_color); ypos.append(y); y += 1
        if n_src == 0:
            labels.append('(negligible)'); values.append(0)
            colors.append('#cccccc'); ypos.append(y); y += 1; n_src = 1

        y += gap

        n_snk = len(sp['sinks'])
        for label, pct in sp['sinks']:
            labels.append(label); values.append(-pct)
            colors.append(snk_color); ypos.append(y); y += 1
        if n_snk == 0:
            labels.append('(no sinks — accumulating)'); values.append(0)
            colors.append('#cccccc'); ypos.append(y); y += 1

        y_extent = max(ypos) - min(ypos) + 0.6  # data range with padding
        panels.append(dict(sp=sp, labels=labels, values=values,
                           colors=colors, ypos=ypos, y_extent=y_extent))

    # ── GridSpec: panel heights proportional to content ──
    # Layout: col 0 = (a),(c) ; col 1 = (b),(d)
    scale = 10
    h = [max(int(p['y_extent'] * scale), 1) for p in panels]
    gap_rows = 4

    gap_rows = 8
    n_rows = max(h[0] + gap_rows + h[2], h[1] + gap_rows + h[3])

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, wspace=0.05)

    ax_a = fig.add_subplot(gs[0:h[0], 0])
    ax_c = fig.add_subplot(gs[h[0]+gap_rows:h[0]+gap_rows+h[2], 0])
    ax_b = fig.add_subplot(gs[0:h[1], 1])
    ax_d = fig.add_subplot(gs[h[1]+gap_rows:h[1]+gap_rows+h[3], 1])

    axes_order = [ax_a, ax_b, ax_c, ax_d]

    # ── 2nd pass: draw ──
    for ax, p in zip(axes_order, panels):
        sp = p['sp']
        ax.barh(p['ypos'], p['values'], height=bar_h, color=p['colors'],
                edgecolor='black', linewidth=0.7)

        ax.set_yticks([])
        ax.set_ylim(max(p['ypos']) + 0.35, min(p['ypos']) - 0.35)  # inverted
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Labels: source (right-going) → label on LEFT, sink (left-going) → label on RIGHT
        for yp, val, label in zip(p['ypos'], p['values'], p['labels']):
            if val <= 0:
                ax.text(2, yp, label, va='center', ha='left', fontsize=8.5)
            else:
                ax.text(-2, yp, label, va='center', ha='right', fontsize=8.5)

        # Percentage: inside bar if large, outside if small
        for yp, v in zip(p['ypos'], p['values']):
            av = abs(v)
            if av < 0.1:
                continue
            if av > 40:
                # inside bar
                x_txt = v - np.sign(v) * 3
                ha = 'right' if v > 0 else 'left'
                color = 'white'
            else:
                # outside bar
                x_txt = v + np.sign(v) * 1.5
                ha = 'left' if v > 0 else 'right'
                color = 'black'
            ax.text(x_txt, yp, f'{av:.1f}%', va='center', ha=ha,
                    fontsize=9, fontweight='bold', color=color)

        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlim(-100, 100)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.0f}'))

        ax.set_title(sp['title'], fontsize=13, fontweight='bold', loc='left',
                     pad=8)

    # Hide x-ticks on top panels, label only bottom
    ax_a.set_xticklabels([])
    ax_b.set_xticklabels([])
    ax_c.set_xlabel('% of turnover')
    ax_d.set_xlabel('% of turnover')

    # Legend
    import matplotlib.patches as mpatches
    src_patch = mpatches.Patch(facecolor=src_color, edgecolor='black',
                               linewidth=0.7, label='Source (+)')
    snk_patch = mpatches.Patch(facecolor=snk_color, edgecolor='black',
                               linewidth=0.7, label='Sink (−)')
    fig.legend(handles=[snk_patch, src_patch], loc='upper right',
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.99, 0.98))

    fig.suptitle('Mass balance: reaction-by-reaction breakdown '
                 '(DIW, αb = 0.03, 720 s)',
                 fontsize=13, y=1.0)
    fig.savefig('Figures/fig4_mass_balance.png')
    fig.savefig('Figures/fig4_mass_balance.pdf')
    print("  → fig4_mass_balance.png/pdf saved")
    plt.close(fig)


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

    print("=" * 60)
    print("Generating figures from recorded BC comparison data")
    print("=" * 60)

    fig1_bc_comparison()
    # fig2: now generated by gen_fig2_rate_evolution.py (time-resolved reaction rates)
    fig3_radicals_vs_alpha()
    fig4_mass_balance_detailed()

    print()
    print("All figures saved in Figures/")
