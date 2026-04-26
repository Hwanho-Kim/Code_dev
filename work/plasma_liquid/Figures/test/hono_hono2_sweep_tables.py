#!/usr/bin/env python3
"""HONO/NO2 + HONO2/N2O5 sweep 결과 테이블 figure 생성.

각 sweep별 4 metric (pH, NO3-, NO2-, H2O2) × 3 voltage × 4 ratio = 12 sim.
총 24 sim 결과를 8개 테이블로 시각화.
Output: hono_hono2_sweep_tables.{png,pdf}
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ─────────────────────────────────────────────────────────────
# Sweep 결과 데이터 (2026-04-23 실측)
# ─────────────────────────────────────────────────────────────

HONO_RATIOS = [0.007, 0.030, 0.070, 0.100]
HONO2_RATIOS = [0.83, 2.00, 3.00, 5.00]
VOLTAGES = ['2.6kV', '3.2kV', '3.6kV']

# Sweep 1: HONO/NO2 (HONO2=0.83, H2O2/O3=0.003 고정)
SWEEP1 = {
    'pH': {
        '2.6kV': [4.424, 4.360, 4.293, 4.253],
        '3.2kV': [4.228, 4.171, 4.114, 4.085],
        '3.6kV': [4.219, 4.146, 4.081, 4.049],
    },
    'NO3': {
        '2.6kV': [37.57, 41.89, 42.69, 42.89],
        '3.2kV': [59.10, 67.28, 73.01, 74.51],
        '3.6kV': [60.27, 70.99, 76.68, 78.16],
    },
    'NO2': {
        '2.6kV': [0.045, 2.567, 12.447, 19.943],
        '3.2kV': [0.054, 0.328, 7.447, 14.683],
        '3.6kV': [0.051, 0.740, 11.788, 21.666],
    },
    'H2O2': {
        '2.6kV': [5.46, 6.40, 6.43, 6.43],
        '3.2kV': [16.64, 18.90, 19.70, 19.79],
        '3.6kV': [21.27, 24.17, 24.93, 25.04],
    },
}

# Sweep 2: HONO2/N2O5 (HONO=voltage-default, H2O2/O3=0.003 고정)
SWEEP2 = {
    'pH': {
        '2.6kV': [4.416, 4.231, 4.117, 3.953],
        '3.2kV': [4.228, 4.008, 3.880, 3.702],
        '3.6kV': [4.221, 4.011, 3.888, 3.714],
    },
    'NO3': {
        '2.6kV': [38.26, 58.72, 76.23, 111.26],
        '3.2kV': [59.12, 98.22, 131.63, 198.48],
        '3.6kV': [60.08, 97.38, 129.25, 193.00],
    },
    'NO2': {
        '2.6kV': [0.049, 0.046, 0.045, 0.042],
        '3.2kV': [0.054, 0.050, 0.046, 0.041],
        '3.6kV': [0.052, 0.048, 0.045, 0.040],
    },
    'H2O2': {
        '2.6kV': [5.70, 5.70, 5.70, 5.70],
        '3.2kV': [16.65, 16.65, 16.65, 16.65],
        '3.6kV': [21.20, 21.20, 21.20, 21.20],
    },
}

EXP = {
    '2.6kV': {'pH': 5.09, 'NO3': 32.63, 'NO2': 0.00,  'H2O2':  4.76},
    '3.2kV': {'pH': 3.61, 'NO3': 62.74, 'NO2': 3.58,  'H2O2': 11.21},
    '3.6kV': {'pH': 3.25, 'NO3': 70.42, 'NO2': 20.74, 'H2O2': 16.25},
}

METRIC_LABELS = {
    'pH':   'pH',
    'NO3':  'NO₃⁻ (µM)',
    'NO2':  'NO₂⁻ (µM)',
    'H2O2': 'H₂O₂ (µM)',
}


def fmt(x, metric):
    if metric == 'pH':
        return f'{x:.3f}'
    elif metric in ('NO3', 'H2O2'):
        return f'{x:.2f}'
    else:  # NO2
        return f'{x:.3f}'


METRIC_ORDER = ['pH', 'NO3', 'NO2', 'H2O2']


def draw_unified(ax, sweep_data, ratios, ratio_name, title,
                 metric_group_colors=None):
    """한 sweep을 통합 테이블로 그림.
    Rows: metric × voltage (12 rows = 4 metrics × 3 voltages)
    Cols: [Metric, V] + 4 ratios + Exp
    """
    ax.axis('off')

    if metric_group_colors is None:
        metric_group_colors = {
            'pH':   '#f0e5d3',
            'NO3':  '#d3e4c5',
            'NO2':  '#c9d9e8',
            'H2O2': '#e8d3d3',
        }

    col_labels = ['Metric', 'V'] + [f'{ratio_name}={r:g}' for r in ratios] + ['Exp']
    cell_text = []
    row_metric_map = []  # (row_index, metric) for coloring
    for metric in METRIC_ORDER:
        for j, v in enumerate(VOLTAGES):
            metric_label = METRIC_LABELS[metric] if j == 0 else ''
            row = [metric_label, v]
            row.extend(fmt(sweep_data[metric][v][i], metric)
                       for i in range(len(ratios)))
            row.append(fmt(EXP[v][metric], metric))
            cell_text.append(row)
            row_metric_map.append(metric)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.5)

    n_cols = len(col_labels)
    n_data_rows = len(cell_text)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#999')
        if row == 0:
            cell.set_facecolor('#4a6fa5')
            cell.set_text_props(color='white', weight='bold')
        else:
            m = row_metric_map[row - 1]
            # Metric-group tinted rows
            cell.set_facecolor(metric_group_colors[m])
            # First column (Metric label) & second (V) bolded
            if col in (0, 1):
                cell.set_text_props(weight='bold')
            # Exp column darker shade of same tint + bold
            if col == n_cols - 1:
                cell.set_text_props(weight='bold')
                # slightly darker by mixing with gray
                base = metric_group_colors[m]
                cell.set_facecolor(_darken(base, 0.1))

    ax.set_title(title, fontsize=13, weight='bold', pad=12)


def _darken(hex_color, factor=0.1):
    """Darken hex color by given factor (0-1)."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return f'#{r:02x}{g:02x}{b:02x}'


def main():
    fig, axes = plt.subplots(2, 1, figsize=(11, 13))

    draw_unified(
        axes[0], SWEEP1, HONO_RATIOS, 'HONO',
        title='Sweep 1: HONO/NO₂ ratio sweep '
              '(HONO₂/N₂O₅=0.83 fixed, H₂O₂/O₃=0.003)',
    )
    draw_unified(
        axes[1], SWEEP2, HONO2_RATIOS, 'HONO₂',
        title='Sweep 2: HONO₂/N₂O₅ ratio sweep '
              '(HONO/NO₂=voltage-default, H₂O₂/O₃=0.003)',
    )

    fig.suptitle(
        'HONO & HONO₂ ratio sweep — DIW Humid fitting, three_film BC, 600s',
        fontsize=14, weight='bold', y=0.995,
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.98])

    out_dir = Path(__file__).parent
    for ext in ('png', 'pdf'):
        p = out_dir / f'hono_hono2_sweep_tables.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()
