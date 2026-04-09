#!/usr/bin/env python3
"""
Figure: Gas-phase data interpolation methods comparison.

For each species with intermittent zeros (below-LOD readings),
compares 4 interpolation methods:
  1. Raw (zeros as-is)
  2. LOD/2 replacement
  3. Linear interpolation between nonzero segments
  4. Exponential backward extrapolation from first stable detection

Generates one figure per species with all methods overlaid.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent

DEFAULT_CSV = (
    _project_root / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)

# All gas species to process
GAS_SPECIES = ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']

# Unicode labels
_LABELS = {
    'O3': 'O₃', 'NO': 'NO', 'NO2': 'NO₂', 'NO3': 'NO₃',
    'N2O4': 'N₂O₄', 'N2O5': 'N₂O₅',
}


def load_raw():
    df = pd.read_csv(DEFAULT_CSV)
    times = np.arange(len(df), dtype=float) * 2.0
    data = {}
    for col in GAS_SPECIES:
        if col in df.columns:
            data[col] = df[col].values.astype(float)
        else:
            data[col] = np.zeros(len(df))
    return times, data


MIN_STABLE_RUN = 5  # minimum consecutive nonzero points to count as "stable"


def find_stable_start(vals):
    """Find index where stable detection begins (>=MIN_STABLE_RUN consecutive nonzero).

    Returns (stable_start_idx, lod_estimate).
    LOD = median of first stable run values (represents detection threshold).
    Before stable_start: all data treated as below-LOD regardless of blips.
    """
    n = len(vals)
    run_start = -1
    run_len = 0
    for i in range(n):
        if vals[i] > 0:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= MIN_STABLE_RUN:
                # Estimate LOD from first few values of this stable run
                seg = vals[run_start:run_start + min(run_len, 10)]
                lod = np.median(seg[seg > 0])
                return run_start, lod
        else:
            run_len = 0
    return n, 0.0  # no stable run found


def method_raw(vals):
    """Method 1: Raw data (zeros as-is)."""
    return np.maximum(vals, 0.0)


def method_lod_half(vals):
    """Method 2: Replace all below-LOD points (before stable start + intermittent zeros) with LOD/2."""
    stable_start, lod = find_stable_start(vals)
    if lod <= 0:
        return np.maximum(vals, 0.0)
    out = vals.copy()
    # Before stable start: all LOD/2
    out[:stable_start] = lod / 2.0
    # After stable start: fill intermittent zeros with LOD/2
    for i in range(stable_start, len(out)):
        if out[i] <= 0:
            out[i] = lod / 2.0
    return np.maximum(out, 0.0)


def method_linear_interp(vals):
    """Method 3: Linear interpolation.

    - Before stable start: ramp from 0 to first stable value
    - After stable start: linear interp between nonzero points
    """
    stable_start, lod = find_stable_start(vals)
    out = vals.copy()

    if stable_start >= len(vals):
        return np.maximum(out, 0.0)

    # Before stable start: linear ramp from 0 at t=0 to first stable value
    first_val = vals[stable_start]
    for i in range(stable_start):
        out[i] = first_val * (i / max(stable_start, 1))

    # After stable start: fill intermittent zeros by linear interp
    nz_after = [(i, vals[i]) for i in range(stable_start, len(vals)) if vals[i] > 0]
    if len(nz_after) >= 2:
        nz_idx = np.array([x[0] for x in nz_after])
        nz_vals = np.array([x[1] for x in nz_after])
        for i in range(stable_start, len(out)):
            if out[i] <= 0:
                out[i] = np.interp(i, nz_idx, nz_vals)

    return np.maximum(out, 0.0)


def method_exp_extrap(vals):
    """Method 4: Exponential backward extrapolation.

    - Find stable start, fit exponential to first 10 stable points
    - Extrapolate backward to t=0
    - After stable start: linear interp for intermittent zeros
    """
    stable_start, lod = find_stable_start(vals)
    out = vals.copy()

    if stable_start >= len(vals) or lod <= 0:
        return np.maximum(out, 0.0)

    # Collect first 10 stable points for fit
    seg_idx, seg_vals = [], []
    for i in range(stable_start, min(stable_start + 20, len(vals))):
        if vals[i] > 0:
            seg_idx.append(i)
            seg_vals.append(vals[i])
            if len(seg_idx) >= 10:
                break

    if len(seg_idx) < 2:
        return method_linear_interp(vals)

    seg_idx = np.array(seg_idx, dtype=float)
    seg_vals = np.array(seg_vals)

    # Fit: log(C) = a + b*idx
    log_vals = np.log(seg_vals)
    if np.std(log_vals) < 0.01:
        # Nearly constant — use linear ramp instead
        return method_linear_interp(vals)

    coeffs = np.polyfit(seg_idx, log_vals, 1)
    b, a = coeffs

    # Extrapolate backward, floor at LOD/100
    floor = max(lod / 100.0, 1.0)
    for i in range(stable_start):
        extrap = np.exp(a + b * i)
        out[i] = max(extrap, floor)

    # After stable start: fill intermittent zeros by linear interp
    nz_after = [(i, vals[i]) for i in range(stable_start, len(vals)) if vals[i] > 0]
    if len(nz_after) >= 2:
        nz_idx_arr = np.array([x[0] for x in nz_after])
        nz_vals_arr = np.array([x[1] for x in nz_after])
        for i in range(stable_start, len(out)):
            if out[i] <= 0:
                out[i] = np.interp(i, nz_idx_arr, nz_vals_arr)

    return np.maximum(out, 0.0)


def method_sigmoid(vals):
    """Method 5: Sigmoid ramp from 0 to first stable value.

    Before stable start: tanh ramp centered at stable_start.
    After stable start: linear interp for intermittent zeros.
    Smooth and BDF-friendly (no discontinuities).
    """
    stable_start, lod = find_stable_start(vals)
    out = vals.copy()

    if stable_start >= len(vals) or lod <= 0:
        return np.maximum(out, 0.0)

    first_val = vals[stable_start]
    # Sigmoid: C(t) = first_val * 0.5 * (1 + tanh((t - t_center) / width))
    # t_center = stable_start, width controls transition sharpness
    # width = stable_start / 4 → transition spans ~2*width around center
    width = max(stable_start / 4.0, 1.0)
    for i in range(stable_start):
        out[i] = first_val * 0.5 * (1.0 + np.tanh((i - stable_start) / width))
        out[i] = max(out[i], lod / 100.0)

    # After stable start: fill intermittent zeros by linear interp
    nz_after = [(i, vals[i]) for i in range(stable_start, len(vals)) if vals[i] > 0]
    if len(nz_after) >= 2:
        nz_idx = np.array([x[0] for x in nz_after])
        nz_vals = np.array([x[1] for x in nz_after])
        for i in range(stable_start, len(out)):
            if out[i] <= 0:
                out[i] = np.interp(i, nz_idx, nz_vals)

    return np.maximum(out, 0.0)


def method_savgol(vals):
    """Method 6: Savitzky-Golay filter on the full data.

    Replaces zeros with LOD/2 first (SG can't handle true zeros on log scale),
    then applies SG filter for smoothing.
    """
    from scipy.signal import savgol_filter

    stable_start, lod = find_stable_start(vals)
    if lod <= 0:
        return np.maximum(vals, 0.0)

    # Fill zeros with LOD/2 first
    out = method_lod_half(vals).copy()

    # SG filter: window=15 points (30s), poly order=3
    win = min(15, len(out) // 2)
    if win % 2 == 0:
        win += 1
    win = max(win, 5)
    out = savgol_filter(out, window_length=win, polyorder=3)

    return np.maximum(out, 0.0)


def gen_figure(times, raw_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    # Only plot species that have zeros (interesting ones)
    species_to_plot = []
    for sp in GAS_SPECIES:
        vals = raw_data[sp]
        n_zero = np.sum(vals <= 0)
        n_nonzero = np.sum(vals > 0)
        if n_nonzero > 0 and n_zero > 0:
            species_to_plot.append(sp)

    if not species_to_plot:
        print("No species with intermittent zeros found.")
        return

    methods = [
        ('Raw (zeros as-is)', method_raw, 'gray', '-', 1.0, 0.4),
        ('LOD/2 replacement', method_lod_half, '#d62728', '--', 1.5, 0.9),
        ('Linear interp', method_linear_interp, '#1f77b4', '-', 1.5, 0.9),
        ('Exp extrapolation', method_exp_extrap, '#2ca02c', '-.', 1.5, 0.9),
        ('Sigmoid ramp', method_sigmoid, '#9467bd', '-', 1.5, 0.9),
        ('Savitzky-Golay', method_savgol, '#ff7f0e', ':', 1.5, 0.9),
    ]

    t_min = times / 60.0
    n_sp = len(species_to_plot)

    # Two rows per species: full range + zoom on transition
    fig, axes = plt.subplots(n_sp, 2, figsize=(14, 4 * n_sp),
                              gridspec_kw={'width_ratios': [2, 1]})
    if n_sp == 1:
        axes = axes.reshape(1, 2)

    for row, sp in enumerate(species_to_plot):
        vals = raw_data[sp]
        _, lod = find_stable_start(vals)

        ax_full = axes[row, 0]
        ax_zoom = axes[row, 1]

        # Find transition region for zoom (around stable start)
        stable_start, lod = find_stable_start(vals)
        if stable_start < len(vals):
            # Zoom: 60s before stable start to 60s after
            zoom_start = max(0, stable_start - 30)  # 60s before
            zoom_end = min(len(vals), stable_start + 30)  # 60s after
        else:
            zoom_start, zoom_end = 0, len(vals)

        for label, method, color, ls, lw, alpha in methods:
            processed = method(vals)
            ax_full.plot(t_min, processed, color=color, ls=ls, lw=lw,
                        alpha=alpha, label=label)
            ax_zoom.plot(t_min[zoom_start:zoom_end],
                        processed[zoom_start:zoom_end],
                        color=color, ls=ls, lw=lw, alpha=alpha, label=label)

        # Mark zero points on raw data
        zero_mask = vals <= 0
        if np.any(zero_mask):
            ax_full.scatter(t_min[zero_mask],
                           np.zeros(np.sum(zero_mask)),
                           color='red', s=8, zorder=5, alpha=0.3,
                           label='Below LOD')

        # LOD line + stable start marker
        if lod > 0:
            for ax in [ax_full, ax_zoom]:
                ax.axhline(lod, color='orange', ls=':', lw=0.8, alpha=0.5,
                          label='LOD' if ax is ax_full else None)
                ax.axhline(lod / 2, color='red', ls=':', lw=0.8, alpha=0.3,
                          label='LOD/2' if ax is ax_full else None)
            if stable_start < len(vals):
                ax_full.axvline(t_min[stable_start], color='purple',
                               ls='--', lw=1, alpha=0.5,
                               label=f'Stable start (t={times[stable_start]:.0f}s)')

        sp_label = _LABELS.get(sp, sp)
        ax_full.set_ylabel(f'{sp_label} (cm⁻³)')
        ax_full.set_title(f'{sp_label} — full range', fontweight='bold',
                         loc='left')
        ax_full.set_xlabel('Time (min)')
        ax_full.set_yscale('log')
        y_nz = vals[vals > 0]
        if len(y_nz) > 0:
            ax_full.set_ylim(bottom=max(lod / 10, 1))

        zoom_t_start = t_min[zoom_start]
        zoom_t_end = t_min[min(zoom_end, len(t_min) - 1)]
        ax_zoom.set_title(
            f'{sp_label} — zoom ({zoom_t_start:.1f}–{zoom_t_end:.1f} min)',
            fontweight='bold', loc='left')
        ax_zoom.set_xlabel('Time (min)')
        ax_zoom.set_yscale('log')
        if len(y_nz) > 0:
            ax_zoom.set_ylim(bottom=max(lod / 10, 1))

        if row == 0:
            ax_full.legend(loc='best', fontsize=8)

    fig.suptitle('Gas-phase data: below-LOD interpolation methods',
                 fontsize=14, y=1.01)
    fig.tight_layout()

    out_png = _script_dir / 'fig_gas_interpolation.png'
    out_pdf = _script_dir / 'fig_gas_interpolation.pdf'
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  → {out_png.name} / {out_pdf.name} saved")
    plt.close(fig)

    # Also print summary table
    print("\n=== Interpolation Summary ===")
    for sp in species_to_plot:
        vals = raw_data[sp]
        _, lod = find_stable_start(vals)
        n_zero = np.sum(vals <= 0)
        sp_label = _LABELS.get(sp, sp)
        print(f"\n{sp_label}:")
        print(f"  LOD estimate: {lod:.3e} cm⁻³")
        print(f"  Zero points: {n_zero}/{len(vals)}")
        for label, method, *_ in methods:
            processed = method(vals)
            # Stats in the formerly-zero regions
            zero_region = processed[vals <= 0]
            if len(zero_region) > 0:
                print(f"  {label:25s}: fill range = [{zero_region.min():.3e}, {zero_region.max():.3e}]")


def main():
    import os
    os.chdir(_project_root)

    print("=" * 60)
    print("Gas-phase data interpolation comparison")
    print("=" * 60)

    times, raw_data = load_raw()
    gen_figure(times, raw_data)
    print("\nDone!")


if __name__ == '__main__':
    main()
