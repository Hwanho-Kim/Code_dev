#!/usr/bin/env python3
"""Phase F1: All-gas-const test for 2.6 kV O3 oscillation.

Replaces every gas species (O3/NO2/NO3/N2O5/N2O4) with its time-mean.
Compares vs baseline. If oscillation collapses, gas-side residual variation
is the forced-response source. If unchanged, internal R32 amplification
dominates.

Default v2 baseline applied (config_1d.atol=1e-20, gen_all_figures.STRETCH=1.02,
seedmin IC, no perturbation).
"""
from __future__ import annotations

import functools
import sys
import time as time_mod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

from chemistry_1d import AqueousChemistry1D  # noqa: E402
from config_1d import AQUEOUS_SPECIES  # noqa: E402
from pde_solver import PDESolver1D  # noqa: E402

print = functools.partial(print, flush=True)
VOLTAGE = "2.6kV"


def _setup_gaf():
    import gen_all_figures as gaf
    gaf.IS_SALINE = False
    gaf.DEFAULT_GAS_SHEET = VOLTAGE
    gaf.SOLUTION_LABEL = "DIW"
    gaf.FIXED_CATION_CONC = 0.0
    gaf.CONDITION_LABEL = "Humid_fitting"
    gaf.EXP = gaf.EXP_DIW_ALL[VOLTAGE]
    return gaf


def _make_const(gas_conc):
    out = {}
    for k, v in gas_conc.items():
        v = np.asarray(v, dtype=float)
        out[k] = np.full_like(v, float(np.mean(v)))
    return out


def run_case(label, all_const):
    gaf = _setup_gaf()
    times, gas_conc = gaf.load_gas_data()
    if all_const:
        gas_conc = _make_const(gas_conc)
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
          f"nfev={res['nfev']}")
    return {
        "label": label,
        "snap_t": snap_t,
        "vol_O3": vol_O3,
        "bulk_O3": bulk_O3,
        "surf_O3": surf_O3,
        "wall": wall,
        "nfev": res["nfev"],
    }


def main():
    print("=" * 80)
    print(f"Phase F1: All-gas-const test — {VOLTAGE}")
    print("=" * 80)
    cases = [
        ("baseline (variable)", False),
        ("all gas const (mean)", True),
    ]
    results = []
    for label, all_const in cases:
        print(f"\n--- {label} ---")
        results.append(run_case(label, all_const))

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
    fig.suptitle(f"Phase F1: gas-side noise removal — {VOLTAGE}",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_f1.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    print("\n=== Detrended std (t > 180s) ===")
    summ = ["Phase F1 detrended std (t>180s):", "",
            f"{'case':<28s} {'wall':>5s} {'nfev':>8s} "
            f"{'surf_std%':>10s} {'vol_std%':>10s} {'bulk_std%':>10s}"]
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
        line = (f"{d['label']:<28s} {d['wall']:>5.0f} {d['nfev']:>8d} "
                f"{d['surf_cv']:>10.3f} {d['vol_cv']:>10.3f} "
                f"{d['bulk_cv']:>10.3f}")
        print(line)
        summ.append(line)
    (out / "diag_o3_f1.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
