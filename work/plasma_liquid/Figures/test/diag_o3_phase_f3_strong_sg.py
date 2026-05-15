#!/usr/bin/env python3
"""Phase F3: Strong SG smoothing on NO2/NO3 for 2.6 kV O3 oscillation.

Apply SG window=151 (302s averaging) to NO2 and NO3 raw OAS data; keep
O3/N2O5/NO defaults (window=31). Re-derive HONO/N2O4 from smoothed NO2,
HONO2 from default N2O5, H2O2 from default O3.

Baseline: default preprocessing (window=31 for all).

Default v2 baseline applied (atol=1e-20, STRETCH=1.02, seedmin IC).
"""
from __future__ import annotations

import functools
import math
import sys
import time as time_mod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

from chemistry_1d import AqueousChemistry1D  # noqa: E402
from config_1d import AQUEOUS_SPECIES, N2O4_EQ, PHYSICAL  # noqa: E402
from pde_solver import PDESolver1D  # noqa: E402

print = functools.partial(print, flush=True)
VOLTAGE = "2.6kV"
STRONG_WIN = 151


def _setup_gaf():
    import gen_all_figures as gaf
    gaf.IS_SALINE = False
    gaf.DEFAULT_GAS_SHEET = VOLTAGE
    gaf.SOLUTION_LABEL = "DIW"
    gaf.FIXED_CATION_CONC = 0.0
    gaf.CONDITION_LABEL = "Humid_fitting"
    gaf.EXP = gaf.EXP_DIW_ALL[VOLTAGE]
    return gaf


def _preprocess_custom(vals, sg_win):
    """Mirror gen_all_figures._preprocess_below_lod with parameterized window."""
    from gen_all_figures import MIN_STABLE_RUN
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


def _build_strong_gas_conc(gaf):
    """Same pipeline as gaf.load_gas_data() but NO2/NO3 use sg_win=STRONG_WIN.

    Returns (times, gas_conc) including N2O4 re-derived from smoothed NO2.
    """
    df = pd.read_excel(gaf.DEFAULT_GAS_XLSX, sheet_name=VOLTAGE)
    times = df.iloc[:, 0].values.astype(float)
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            raw = df[col].values.astype(float)
            win = STRONG_WIN if col in ('NO2', 'NO3') else 31
            gas_conc[col] = _preprocess_custom(raw, sg_win=win)
        else:
            gas_conc[col] = np.zeros(len(df))

    # Mirror gaf.load_gas_data() RH80 ratio chain (post-smoothing)
    r = gaf.RH80_RATIOS.get(VOLTAGE, None)
    if r and 'Humid_fitting' in gaf.CONDITION_LABEL:
        mask_ss = times >= (times[-1] - 100)
        def ss(arr):
            return max(float(np.mean(arr[mask_ss])), 1e-30)
        o3_ss_dry   = ss(gas_conc['O3'])
        o3_ss_80    = o3_ss_dry * r['O3_scale']
        no2_ss_dry  = ss(gas_conc['NO2'])
        no2_ss_80   = o3_ss_80 * r['NO2_O3']
        n2o5_ss_dry = ss(gas_conc['N2O5'])
        n2o5_ss_80  = no2_ss_80 * r['N2O5_NO2']
        no3_ss_dry  = ss(gas_conc['NO3'])
        no3_ss_80   = o3_ss_80 * r['NO3_O3']
        gas_conc['O3']   = gas_conc['O3']   * (o3_ss_80   / o3_ss_dry)
        gas_conc['NO2']  = gas_conc['NO2']  * (no2_ss_80  / no2_ss_dry)
        gas_conc['N2O5'] = gas_conc['N2O5'] * (n2o5_ss_80 / n2o5_ss_dry)
        gas_conc['NO3']  = gas_conc['NO3']  * (no3_ss_80  / no3_ss_dry)

    # N2O4 — re-derived from current NO2 (after smoothing + RH80 scaling).
    # Always overwrite here so the equilibrium reflects the smoothed NO2.
    no2 = gas_conc['NO2']
    T = 298.15
    Kp = math.exp(
        math.log(N2O4_EQ.KP_298)
        + (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
    )
    gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * (no2 ** 2)

    return times, gas_conc


def run_case(label, strong):
    gaf = _setup_gaf()
    if strong:
        times, gas_conc = _build_strong_gas_conc(gaf)
    else:
        times, gas_conc = gaf.load_gas_data()
    HONO_GAS  = gas_conc["NO2"]  * gaf.RH80_RATIOS[VOLTAGE]["HONO_NO2"]
    HONO2_GAS = gas_conc["N2O5"] * gaf.HONO2_RATIO
    H2O2_GAS  = gas_conc["O3"]   * gaf.H2O2_RATIO

    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=gaf.DZ_MIN,
        stretch_ratio=gaf.STRETCH,
        saline_mode=False,
        bc_type="three_film",
        alpha_b=None,
        delta_gas=0.01,
        delta_liq=1e-4,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=HONO_GAS, hono2_gas=HONO2_GAS, h2o2_gas=H2O2_GAS,
    )
    t_end = float(times[-1])
    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    res = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0,
                       verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0
    snap_t = np.asarray(res["t_eval"])
    snap_y = np.asarray(res["y_eval"]).reshape(len(snap_t), solver.N_z, solver.N_s)

    iO3 = AQUEOUS_SPECIES.index("O3")
    z_mm = solver.z_centers * 1e3
    dz = solver.dz_cells
    L = solver.L
    mask_b = z_mm > 0.1
    Lb = float(dz[mask_b].sum())
    bulk_O3 = np.array([np.dot(snap_y[k, mask_b, iO3], dz[mask_b]) / Lb
                        for k in range(len(snap_t))])
    vol_O3 = np.array([np.dot(snap_y[k, :, iO3], dz) / L
                        for k in range(len(snap_t))])
    surf_O3 = snap_y[:, 0, iO3]
    print(f"  [{label}] wall={wall:.0f}s, pH={res['pH_avg']:.3f}, "
          f"nfev={res['nfev']}, NO3aq={res['spatial_avg'].get('NO3-', 0)*1e6:.2f}uM, "
          f"NO2aq={res['spatial_avg'].get('NO2-', 0)*1e6:.3f}uM")
    return {
        "label": label,
        "snap_t": snap_t,
        "vol_O3": vol_O3,
        "bulk_O3": bulk_O3,
        "surf_O3": surf_O3,
        "wall": wall,
        "nfev": res["nfev"],
        "pH": float(res["pH_avg"]),
        "no3": float(res["spatial_avg"].get("NO3-", 0)) * 1e6,
        "no2": float(res["spatial_avg"].get("NO2-", 0)) * 1e6,
        "h2o2": float(res["spatial_avg"].get("H2O2", 0)) * 1e6,
    }


def main():
    print("=" * 80)
    print(f"Phase F3: NO2/NO3 SG strong (window={STRONG_WIN}) — {VOLTAGE}")
    print("=" * 80)
    cases = [
        ("baseline (w=31 all)", False),
        (f"strong NO2/NO3 (w={STRONG_WIN})", True),
    ]
    results = []
    for label, strong in cases:
        print(f"\n--- {label} ---")
        results.append(run_case(label, strong))

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    cmap = ["#1f77b4", "#d62728"]
    for d, c in zip(results, cmap):
        t = d["snap_t"] / 60.0
        axes[0].plot(t, d["surf_O3"] * 1e6, color=c, lw=1.4, label=d["label"])
        axes[1].plot(t, d["vol_O3"] * 1e9, color=c, lw=1.4, label=d["label"])
        axes[2].plot(t, d["bulk_O3"] * 1e9, color=c, lw=1.4, label=d["label"])
    axes[0].set_ylabel("Surface [O₃] (µM)")
    axes[0].set_title("(a) Surface (z=0)", fontweight="bold", loc="left")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Vol-weighted [O₃] (nM)")
    axes[1].set_title("(b) Vol-weighted avg", fontweight="bold", loc="left")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[2].set_ylabel("Bulk-only [O₃] (nM)")
    axes[2].set_title("(c) Bulk-only (z>0.1mm)", fontweight="bold", loc="left")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Time (min)")
    fig.suptitle(f"Phase F3: NO2/NO3 strong SG (w={STRONG_WIN}) — {VOLTAGE}",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_f3.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    print("\n=== Detrended std (t > 180s) ===")
    summ = [f"Phase F3 detrended std (t>180s) — NO2/NO3 SG w={STRONG_WIN}:", "",
            f"{'case':<32s} {'wall':>6s} {'nfev':>9s} "
            f"{'surf%':>8s} {'vol%':>8s} {'bulk%':>8s} "
            f"{'pH':>6s} {'NO3uM':>7s} {'NO2uM':>7s} {'H2O2uM':>8s}"]
    for d in results:
        mask = d["snap_t"] > 180
        for sname, arr in [("surf", d["surf_O3"]),
                           ("vol", d["vol_O3"]),
                           ("bulk", d["bulk_O3"])]:
            p = np.polyfit(d["snap_t"][mask], arr[mask], 1)
            res_arr = arr[mask] - np.polyval(p, d["snap_t"][mask])
            std = float(np.std(res_arr))
            mean = float(np.mean(arr[mask]))
            cv = (std / mean * 100) if abs(mean) > 0 else float("nan")
            d[f"{sname}_cv"] = cv
        line = (f"{d['label']:<32s} {d['wall']:>6.0f} {d['nfev']:>9d} "
                f"{d['surf_cv']:>8.3f} {d['vol_cv']:>8.3f} "
                f"{d['bulk_cv']:>8.3f} {d['pH']:>6.3f} "
                f"{d['no3']:>7.2f} {d['no2']:>7.3f} {d['h2o2']:>8.3f}")
        print(line)
        summ.append(line)
    (out / "diag_o3_f3.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
