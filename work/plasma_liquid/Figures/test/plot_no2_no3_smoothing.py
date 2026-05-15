#!/usr/bin/env python3
"""Compare raw OAS vs current SG (window=31) vs strong SG (window=151)
for NO2 and NO3 gas species at 2.6 kV.

No simulation — just preprocessing visualization.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Figures"))

from gen_all_figures import DEFAULT_GAS_XLSX, MIN_STABLE_RUN  # noqa: E402


def preprocess(vals, sg_win):
    """Same logic as gen_all_figures._preprocess_below_lod but parameterized."""
    out = vals.copy()
    n = len(vals)
    run_start, run_len = -1, 0
    stable_start = n
    for i in range(n):
        if vals[i] > 0:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= MIN_STABLE_RUN:
                stable_start = run_start
                break
        else:
            run_len = 0
    if stable_start >= n:
        return np.maximum(out, 0.0)
    nz_after = [(i, vals[i]) for i in range(stable_start, n) if vals[i] > 0]
    if len(nz_after) >= 2:
        nz_idx = np.array([x[0] for x in nz_after])
        nz_vals = np.array([x[1] for x in nz_after])
        for i in range(stable_start, n):
            if out[i] <= 0:
                out[i] = np.interp(i, nz_idx, nz_vals)
    stable_region = out[stable_start:]
    if len(stable_region) >= sg_win:
        w = sg_win if sg_win % 2 == 1 else sg_win + 1
        stable_region = savgol_filter(stable_region, window_length=w, polyorder=3)
        out[stable_start:] = np.maximum(stable_region, 0.0)
    first_val = out[stable_start]
    for i in range(stable_start):
        out[i] = first_val * (i / max(stable_start, 1))
    return np.maximum(out, 0.0)


def main():
    VOLTAGE = "2.6kV"
    df = pd.read_excel(DEFAULT_GAS_XLSX, sheet_name=VOLTAGE)
    times = df.iloc[:, 0].values.astype(float)
    raw_no2 = df["NO2"].values.astype(float)
    raw_no3 = df["NO3"].values.astype(float)

    cur_no2  = preprocess(raw_no2, sg_win=31)
    str_no2  = preprocess(raw_no2, sg_win=151)
    str2_no2 = preprocess(raw_no2, sg_win=251)

    cur_no3  = preprocess(raw_no3, sg_win=31)
    str_no3  = preprocess(raw_no3, sg_win=151)
    str2_no3 = preprocess(raw_no3, sg_win=251)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # NO2 panel
    ax = axes[0]
    ax.plot(times, raw_no2, '.', color='gray', alpha=0.4, ms=4, label='raw OAS')
    ax.plot(times, cur_no2, '-', color='#1f77b4', lw=1.5,
            label='current SG (window=31, 62s)')
    ax.plot(times, str_no2, '-', color='#d62728', lw=2.0,
            label='strong SG (window=151, 302s)')
    ax.plot(times, str2_no2, '-', color='#9467bd', lw=2.0, ls='--',
            label='very strong SG (window=251, 502s)')
    ax.set_ylabel(r'NO$_2$ (cm$^{-3}$)')
    ax.set_title(f'(a) NO$_2$ at {VOLTAGE}', fontweight='bold', loc='left')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

    # NO3 panel
    ax = axes[1]
    ax.plot(times, raw_no3, '.', color='gray', alpha=0.4, ms=4, label='raw OAS')
    ax.plot(times, cur_no3, '-', color='#1f77b4', lw=1.5,
            label='current SG (window=31, 62s)')
    ax.plot(times, str_no3, '-', color='#d62728', lw=2.0,
            label='strong SG (window=151, 302s)')
    ax.plot(times, str2_no3, '-', color='#9467bd', lw=2.0, ls='--',
            label='very strong SG (window=251, 502s)')
    ax.set_ylabel(r'NO$_3$ (cm$^{-3}$)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'(b) NO$_3$ at {VOLTAGE}', fontweight='bold', loc='left')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

    fig.suptitle(f'SG smoothing comparison — raw OAS vs current vs strong ({VOLTAGE})',
                 fontsize=12, fontweight='bold', y=1.005)
    fig.tight_layout()

    # CV% over t>60s stable region for quantitative comparison
    mask = times > 60
    def detrended_cv(arr):
        a = arr[mask]; t = times[mask]
        p = np.polyfit(t, a, 1)
        res = a - np.polyval(p, t)
        std = float(np.std(res))
        mean = float(np.mean(a))
        return (std / mean * 100) if mean > 0 else float('nan')

    print(f"\n=== Detrended CV% (t>60s) — {VOLTAGE} ===")
    print(f"{'species':<10s} {'raw':>8s} {'w=31':>8s} {'w=151':>8s} {'w=251':>8s}")
    for sp, raw, cur, strong, vstrong in [
        ('NO2', raw_no2, cur_no2, str_no2, str2_no2),
        ('NO3', raw_no3, cur_no3, str_no3, str2_no3),
    ]:
        cvs = [detrended_cv(x) for x in (raw, cur, strong, vstrong)]
        print(f"{sp:<10s} {cvs[0]:>8.2f} {cvs[1]:>8.2f} "
              f"{cvs[2]:>8.2f} {cvs[3]:>8.2f}")

    out = Path(__file__).parent
    for ext in ('png', 'pdf'):
        p = out / f'fig_no2_no3_smoothing_{VOLTAGE}.{ext}'
        fig.savefig(p, dpi=200 if ext == 'png' else None, bbox_inches='tight')
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
