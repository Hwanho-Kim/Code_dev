#!/usr/bin/env python3
"""Phase B redo — correct chemistry disable via _precompute_numba_arrays.

Previous Phase B1/B2 had bug: modifying chem._rxn_data['k'] does NOT
update Numba JIT compute_rates_batch arrays. Must call _precompute_numba_arrays.

Also adds per-cell O3 diagnostic to identify which cells oscillate.

Tests (2.6 kV only):
  baseline       — re-uses HONOvar cache (with full precompute)
  r32_off        — R32 k=0, _precompute_numba_arrays() called
  o3_sinks_off   — R22-R32 k=0, precompute called
  no_diffusion   — D=0 (skip if hard); else gas-only sim with k_mt=0
  per-cell trace — visualize each cell time series for baseline
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
from config_1d import ACID_BASE_PAIRS, AQUEOUS_SPECIES  # noqa: E402
from pde_solver import PDESolver1D  # noqa: E402

print = functools.partial(print, flush=True)

VOLTAGE = "2.6kV"
O3_SINK_LABELS_ALL = [f"R{i}" for i in range(22, 33)]


def disable_reactions_correct(chem, labels_to_zero):
    """Modify _rxn_data k=0 AND call _precompute_numba_arrays."""
    n = 0
    for ri, r in enumerate(chem.reactions):
        label = str(r.get("label", ""))
        if any(label.startswith(f"{lab}:") or label.startswith(f"{lab} ")
               for lab in labels_to_zero):
            r["k"] = 0.0
            chem._rxn_data[ri]["k"] = 0.0
            n += 1
    chem._precompute_numba_arrays()  # ★ rebuild Numba arrays
    print(f"  disabled {n} reactions, _precompute_numba_arrays() called")
    return n


def run_case(label: str, disable_labels=None) -> dict:
    import gen_all_figures as gaf

    gaf.IS_SALINE = False
    gaf.DEFAULT_GAS_SHEET = VOLTAGE
    gaf.SOLUTION_LABEL = "DIW"
    gaf.FIXED_CATION_CONC = 0.0
    gaf.CONDITION_LABEL = "Humid_fitting"
    gaf.EXP = gaf.EXP_DIW_ALL[VOLTAGE]

    times, gas_conc = gaf.load_gas_data()
    HONO_GAS = gas_conc["NO2"] * gaf.RH80_RATIOS[VOLTAGE]["HONO_NO2"]
    HONO2_GAS = gas_conc["N2O5"] * gaf.HONO2_RATIO
    H2O2_GAS = gas_conc["O3"] * gaf.H2O2_RATIO

    chem = AqueousChemistry1D(saline_mode=False)
    if disable_labels:
        disable_reactions_correct(chem, disable_labels)

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type="three_film", alpha_b=None,
        delta_gas=0.01, delta_liq=1e-4,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=HONO_GAS, hono2_gas=HONO2_GAS,
                        h2o2_gas=H2O2_GAS)
    t_end = float(times[-1])
    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    res = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0, verbose=False,
                       dt_poisson=None)
    wall = time_mod.time() - t0
    snap_t = np.asarray(res["t_eval"])
    snap_y = np.asarray(res["y_eval"]).reshape(len(snap_t), solver.N_z, solver.N_s)
    print(f"  [{label}] wall={wall:.0f}s, pH={res['pH_avg']:.3f}, "
          f"nfev={res['nfev']}")

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
    return {
        "label": label,
        "snap_t": snap_t,
        "snap_y_O3": snap_y[:, :, iO3],  # (nt, N_z) — full per-cell
        "surf_O3": surf_O3,
        "bulk_O3": bulk_O3,
        "vol_O3": vol_O3,
        "z_mm": z_mm,
        "dz": dz,
        "L": L,
    }


def main():
    print("=" * 80)
    print(f"Phase B redo (fixed chemistry disable) — {VOLTAGE}")
    print("=" * 80)

    cases = [
        ("baseline (full chem)", None),
        ("R32 only off",       ["R32"]),
        ("R22-R32 all off",    O3_SINK_LABELS_ALL),
    ]
    results = {}
    for label, dl in cases:
        print(f"\n--- {label} ---")
        results[label] = run_case(label, dl)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    colors = {"baseline (full chem)": "#d62728",
              "R32 only off":       "#2ca02c",
              "R22-R32 all off":    "#1f77b4"}

    for label, d in results.items():
        t = d["snap_t"] / 60.0
        c = colors[label]
        axes[0].plot(t, d["surf_O3"] * 1e6, color=c, lw=1.6, label=label)
        axes[1].plot(t, d["vol_O3"] * 1e9, color=c, lw=1.6, label=label)
        axes[2].plot(t, d["bulk_O3"] * 1e9, color=c, lw=1.6, label=label)

    axes[0].set_ylabel("Surface [O₃] (µM)")
    axes[0].set_title("(a) Surface (z=0)", fontweight="bold", loc="left")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("Vol-weighted [O₃] (nM)")
    axes[1].set_title("(b) Vol-weighted avg [0..L] (current fig1c)",
                      fontweight="bold", loc="left")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].set_ylabel("Bulk-only [O₃] (nM)")
    axes[2].set_title("(c) Bulk-only (z>0.1mm)", fontweight="bold", loc="left")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    # Per-cell trace (baseline only): show cells 0, 1, 2, 5, 10, 20, 40
    d = results["baseline (full chem)"]
    cells_to_plot = [0, 1, 2, 5, 10, 20, 40]
    cell_colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(cells_to_plot)))
    ax = axes[3]
    for ci, j in enumerate(cells_to_plot):
        if j < d["snap_y_O3"].shape[1]:
            ax.plot(d["snap_t"]/60.0, d["snap_y_O3"][:, j],
                    color=cell_colors[ci], lw=1.2,
                    label=f"cell {j} (z={d['z_mm'][j]:.3g}mm)")
    ax.set_yscale("log")
    ax.set_ylabel("[O₃] (M)")
    ax.set_title("(d) baseline per-cell trace — find which cells oscillate",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("Time (min)")

    fig.suptitle(
        f"Phase B redo ({VOLTAGE}): fixed chemistry disable + per-cell O3",
        fontsize=12, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_b_redo.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    print("\n=== Detrended std (t > 180s) ===")
    summ = ["Phase B redo detrended std (t>180s):", ""]
    for label in cases_keys(cases):
        d = results[label]
        for sname, arr in [("vol", d["vol_O3"]), ("bulk-only", d["bulk_O3"])]:
            mask = d["snap_t"] > 180
            p = np.polyfit(d["snap_t"][mask], arr[mask], 1)
            res_arr = arr[mask] - np.polyval(p, d["snap_t"][mask])
            std = float(np.std(res_arr))
            mean = float(np.mean(arr[mask]))
            cv = (std / mean * 100) if mean > 0 else float("nan")
            line = (f"  {label:>22s} {sname:>10s}: mean={mean*1e9:7.3f} nM, "
                    f"std={std*1e9:7.4f} nM ({cv:5.2f}%)")
            print(line)
            summ.append(line)
    (out / "diag_o3_b_redo.txt").write_text("\n".join(summ))


def cases_keys(cases):
    return [c[0] for c in cases]


if __name__ == "__main__":
    main()
