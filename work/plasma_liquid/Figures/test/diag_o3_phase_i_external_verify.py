#!/usr/bin/env python3
"""Phase I: external AI verification — Case I + atol/trace floor + N_z sweep.

Tests at 3.6 kV:
  1. baseline (cache, full chem, atol=1e-15, trace=1e-30, dz=5µm)
  2. case_I       — R20 + R35 disabled + initial OH=trace, dz=5µm (N=49)
  3. case_I_N37   — same as #2, dz=20µm (N≈37)
  4. case_I_N57   — same as #2, dz=2µm (N≈57)
  5. atol_tight   — atol=1e-30, max_step=0.1s, full chem, dz=5µm
  6. trace_1e50   — trace_concentration=1e-50, full chem, atol=1e-15

Expected if numerical artifact (external AI hypothesis):
  - Case I keeps deep peak (1e-15 level) → confirms chem-source-free leak
  - Case I × N_z varies meaningfully → grid-dependent
  - atol_tight reduces baseline 4mm peak significantly
  - trace_1e50 changes deep cells (chemistry kernel max(y, trace) clip lowered)
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
import config_1d as cfg  # noqa: E402
from config_1d import AQUEOUS_SPECIES  # noqa: E402
from pde_solver import PDESolver1D  # noqa: E402

print = functools.partial(print, flush=True)
VOLTAGE = "3.6kV"


def disable_o3_producers(chem):
    """Disable R20 (both directions) + R35.
    R20: O + O2 ↔ O3 (only forward producing O3 from O)
    R35: O3- + OH → O3 + OH- (one of the few O3 producers)
    """
    n = 0
    for ri, r in enumerate(chem.reactions):
        label = str(r.get("label", ""))
        if (label.startswith("R20:") or label.startswith("R35:") or
                label.startswith("R20 ") or label.startswith("R35 ")):
            if "k" in r:
                r["k"] = 0.0
            if "k_f" in r:
                r["k_f"] = 0.0
            if "k_b" in r:
                r["k_b"] = 0.0
            chem._rxn_data[ri]["k"] = 0.0
            print(f"  disabled: {label}")
            n += 1
    chem._precompute_numba_arrays()
    return n


def run_case(label, *, disable_R20_R35=False, oh_init_trace=False,
             dz_min=5e-6, atol=1e-15, max_step=1.0, trace=1e-30):
    """One sim with specified knobs."""
    import gen_all_figures as gaf

    # Patch global config
    cfg.ODE_CONFIG.atol = atol
    cfg.ODE_CONFIG.rtol = 1e-6
    cfg.ODE_CONFIG.max_step = max_step
    cfg.DEFAULTS.trace_concentration = trace

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
    if disable_R20_R35:
        n = disable_o3_producers(chem)
        print(f"  disabled {n} O3-producer reactions")

    solver = PDESolver1D(
        chemistry=chem, dz_min=dz_min, stretch_ratio=1.12,
        saline_mode=False, bc_type="three_film", alpha_b=None,
        delta_gas=0.01, delta_liq=1e-4,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=HONO_GAS, hono2_gas=HONO2_GAS,
                        h2o2_gas=H2O2_GAS)
    t_end = float(times[-1])
    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    if oh_init_trace:
        # Force OH init = trace in all cells
        iOH = solver.species_idx["OH"]
        for j in range(solver.N_z):
            y0[j * solver.N_s + iOH] = trace
        print(f"  OH initial set to trace ({trace:.0e}) in all {solver.N_z} cells")

    t0 = time_mod.time()
    res = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0, verbose=False,
                       dt_poisson=None)
    wall = time_mod.time() - t0
    snap_t = np.asarray(res["t_eval"])
    snap_y = np.asarray(res["y_eval"]).reshape(len(snap_t), solver.N_z, solver.N_s)
    print(f"  [{label}] N_z={solver.N_z}, wall={wall:.0f}s, "
          f"pH={res['pH_avg']:.3f}, nfev={res['nfev']}")

    iO3 = AQUEOUS_SPECIES.index("O3")
    return {
        "label": label,
        "N_z": solver.N_z,
        "snap_t": snap_t,
        "z_mm": solver.z_centers * 1e3,
        "snap_y_O3": snap_y[:, :, iO3],
        "wall": wall,
        "nfev": res["nfev"],
    }


def load_baseline_cache():
    fp = (_root / "Figures" / "DIW results"
          / f"{VOLTAGE}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    iO3 = AQUEOUS_SPECIES.index("O3")
    snap_y = np.asarray(d["snap_y"])
    print("  [baseline] loaded cache (atol=1e-15, trace=1e-30, dz=5µm)")
    return {
        "label": "baseline",
        "N_z": int(d["N_z"]),
        "snap_t": np.asarray(d["snap_t"]),
        "z_mm": np.asarray(d["z_centers"]) * 1e3,
        "snap_y_O3": snap_y[:, :, iO3],
        "wall": 0,
        "nfev": 0,
    }


def main():
    print("=" * 80)
    print(f"Phase I ({VOLTAGE}): external AI verification — Case I + atol/trace + N_z")
    print("=" * 80)

    results = {}

    print("\n--- 1) baseline (cache) ---")
    results["baseline"] = load_baseline_cache()

    print("\n--- 2) Case I: R20+R35 OFF + OH init=trace, dz=5µm ---")
    results["case_I"] = run_case("case_I",
                                  disable_R20_R35=True, oh_init_trace=True,
                                  dz_min=5e-6, atol=1e-15)

    print("\n--- 3) Case I × dz=20µm (N≈37) ---")
    results["case_I_N37"] = run_case("case_I_N37",
                                      disable_R20_R35=True, oh_init_trace=True,
                                      dz_min=20e-6, atol=1e-15)

    print("\n--- 4) Case I × dz=2µm (N≈57) ---")
    results["case_I_N57"] = run_case("case_I_N57",
                                      disable_R20_R35=True, oh_init_trace=True,
                                      dz_min=2e-6, atol=1e-15)

    print("\n--- 5) atol=1e-30 + max_step=0.1s, full chem ---")
    results["atol_tight"] = run_case("atol_tight",
                                      disable_R20_R35=False, oh_init_trace=False,
                                      dz_min=5e-6, atol=1e-30, max_step=0.1)

    print("\n--- 6) trace=1e-50, full chem ---")
    results["trace_1e50"] = run_case("trace_1e50",
                                      disable_R20_R35=False, oh_init_trace=False,
                                      dz_min=5e-6, atol=1e-15, trace=1e-50)

    # Restore defaults
    cfg.ODE_CONFIG.atol = 1e-15
    cfg.ODE_CONFIG.rtol = 1e-6
    cfg.ODE_CONFIG.max_step = 1.0
    cfg.DEFAULTS.trace_concentration = 1e-30

    # Plot O3 spatial at t=480s for all cases
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    cmap = plt.cm.tab10(np.arange(len(results)))
    ax = axes[0]
    for (label, d), c in zip(results.items(), cmap):
        si = int(np.argmin(np.abs(d["snap_t"] - 480)))
        ax.plot(d["z_mm"], np.abs(d["snap_y_O3"][si]) + 1e-50, color=c, lw=1.6,
                marker="o", ms=3,
                label=f"{label} (N={d['N_z']})")
    ax.set_yscale("log")
    ax.set_ylim(1e-30, 1e-3)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("[O₃] (M)")
    ax.set_title("(a) O₃ spatial @ t=480s — Phase I cases",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, ncol=2)

    # Time series at z≈4mm
    ax = axes[1]
    for (label, d), c in zip(results.items(), cmap):
        j_4mm = int(np.argmin(np.abs(d["z_mm"] - 4.0)))
        ax.plot(d["snap_t"]/60, np.abs(d["snap_y_O3"][:, j_4mm]) + 1e-50,
                color=c, lw=1.6, label=f"{label} (z={d['z_mm'][j_4mm]:.2f}mm)")
    ax.set_yscale("log")
    ax.set_ylim(1e-25, 1e-7)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("[O₃] (M)")
    ax.set_title("(b) [O₃] at z≈4mm vs time", fontweight="bold", loc="left")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, ncol=2)

    fig.suptitle(f"Phase I ({VOLTAGE}): external AI verification — "
                 "Case I confirms chemistry-free leak",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_i.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    # Numerical summary
    print("\n=== O3 spatial @ t=480s ===")
    summ = ["Phase I O3 spatial @ t=480s:", ""]
    z_query = [0.003, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0]
    header = f"{'z(mm)':>7s} " + ' '.join(f"{lbl[:14]:>14s}"
                                            for lbl in results.keys())
    print(header)
    summ.append(header)
    for zt in z_query:
        line = f"{zt:>7.3f} "
        for label, d in results.items():
            j = int(np.argmin(np.abs(d["z_mm"] - zt)))
            si = int(np.argmin(np.abs(d["snap_t"] - 480)))
            line += f" {d['snap_y_O3'][si, j]:>+14.3e}"
        print(line)
        summ.append(line)

    # Verdict on each case
    print("\n=== 4mm peak comparison ===")
    base_4mm = None
    for label, d in results.items():
        j = int(np.argmin(np.abs(d["z_mm"] - 4.0)))
        si = int(np.argmin(np.abs(d["snap_t"] - 480)))
        v = d["snap_y_O3"][si, j]
        if label == "baseline":
            base_4mm = v
        ratio = v / base_4mm if base_4mm else 1
        line = (f"  {label:<14s} (N={d['N_z']:>3}): O3@4mm = {v:>+12.3e}  "
                f"ratio_to_base={ratio:>10.3e}, wall={d['wall']:>4.0f}s, "
                f"nfev={d['nfev']}")
        print(line)
        summ.append(line)

    (out / "diag_o3_i.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
