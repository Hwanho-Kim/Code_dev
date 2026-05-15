#!/usr/bin/env python3
"""Phase B1: R32 off vs baseline — isolate predator-prey hypothesis.

If the 2.6 kV bulk O3 oscillation disappears with R32 disabled, the
mechanism is confirmed as O3↔NO2-↔R32 weak-damping resonance.

Test matrix:
  2.6 kV baseline   (R32 active)  — re-uses HONOvar cache
  2.6 kV R32 off
  3.2 kV R32 off    (control: would 3.2 kV oscillate without R32?)

Each new sim ~60-100 s. Total ~3-4 min.
Output: fig_diag_o3_b1.{png,pdf}, diag_o3_b1.txt
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

CASES = [
    ("2.6kV", "baseline"),
    ("2.6kV", "r32off"),
    ("3.2kV", "r32off"),
]
COLORS = {
    ("2.6kV", "baseline"): "#d62728",
    ("2.6kV", "r32off"):   "#ff9896",
    ("3.2kV", "r32off"):   "#aec7e8",
}


def disable_r32(chem: AqueousChemistry1D) -> int:
    """Find R32 in chem.reactions and zero its rate constant. Returns n_disabled."""
    n = 0
    for ri, r in enumerate(chem.reactions):
        label = str(r.get("label", ""))
        if label.startswith("R32:") or label.startswith("R32 "):
            r["k"] = 0.0
            chem._rxn_data[ri]["k"] = 0.0
            print(f"  disabled: {label}")
            n += 1
    return n


def run_case(voltage: str, mode: str) -> dict:
    """Run a 600s DIW sim. Returns dict with bulk avg time series."""
    import gen_all_figures as gaf

    gaf.IS_SALINE = False
    gaf.DEFAULT_GAS_SHEET = voltage
    gaf.SOLUTION_LABEL = "DIW"
    gaf.FIXED_CATION_CONC = 0.0
    gaf.CONDITION_LABEL = "Humid_fitting"
    gaf.EXP = gaf.EXP_DIW_ALL[voltage]

    times, gas_conc = gaf.load_gas_data()

    if mode == "baseline":
        # Reuse cache
        fp = (_root / "Figures" / "DIW results"
              / f"{voltage}_Humid_fitting_three_film_HONOvar"
              / "cache" / "three_film_abspecies_dg0.0100.npz")
        d = dict(np.load(fp, allow_pickle=True))
        print(f"  [{voltage}/{mode}] loaded cache")
        return _extract(d)

    chem = AqueousChemistry1D(saline_mode=False)
    if mode == "r32off":
        n = disable_r32(chem)
        if n == 0:
            raise RuntimeError("R32 not found in reactions")

    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type="three_film", alpha_b=None,
        delta_gas=0.01, delta_liq=1e-4,
    )
    # Same gas data path as gen_all_figures.run_case
    HONO_GAS = gas_conc["NO2"] * gaf.RH80_RATIOS[voltage]["HONO_NO2"]
    HONO2_GAS = gas_conc["N2O5"] * gaf.HONO2_RATIO
    H2O2_GAS = gas_conc["O3"] * gaf.H2O2_RATIO
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=HONO_GAS, hono2_gas=HONO2_GAS,
                        h2o2_gas=H2O2_GAS)
    t_end = float(times[-1])
    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)

    t0 = time_mod.time()
    result = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0,
                          verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0

    snap_t = np.asarray(result["t_eval"])
    snap_y = np.asarray(result["y_eval"]).reshape(len(snap_t), solver.N_z, solver.N_s)
    dz = solver.dz_cells
    L = solver.L
    z_mm = solver.z_centers * 1e3
    print(f"  [{voltage}/{mode}] wall={wall:.0f}s, n={len(snap_t)}, "
          f"final pH={result['pH_avg']:.3f}")

    return _extract({
        "snap_t": snap_t,
        "snap_y": snap_y,
        "dz_cells": dz,
        "L": L,
        "z_centers": solver.z_centers,
    })


def _extract(d: dict) -> dict:
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    dz = np.asarray(d["dz_cells"])
    L = float(d["L"])
    z_mm = np.asarray(d["z_centers"]) * 1e3
    iO3 = AQUEOUS_SPECIES.index("O3")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    iHp = AQUEOUS_SPECIES.index("H+")

    # bulk-only avg z>0.1mm
    mask = z_mm > 0.1
    dz_b = dz[mask]
    Lb = float(dz_b.sum())
    bulk_O3 = np.array([np.dot(snap_y[k, mask, iO3], dz_b) / Lb
                        for k in range(len(snap_t))])
    surf_O3 = snap_y[:, 0, iO3]
    # NO2- via speciation, bulk
    from config_1d import ACID_BASE_PAIRS
    pKa = ACID_BASE_PAIRS["HONO_total"][2]
    Ka = 10.0 ** (-pKa)
    hono_t = np.array([np.dot(snap_y[k, mask, iHONO_t], dz_b) / Lb
                       for k in range(len(snap_t))])
    hp = np.array([np.dot(snap_y[k, mask, iHp], dz_b) / Lb
                   for k in range(len(snap_t))])
    hp = np.maximum(hp, 1e-14)
    no2m = hono_t * Ka / (hp + Ka)
    return {
        "snap_t": snap_t,
        "bulk_O3": bulk_O3,
        "surf_O3": surf_O3,
        "no2m": no2m,
    }


def detrended_std(t: np.ndarray, y: np.ndarray, t_min: float) -> float:
    mask = t > t_min
    if mask.sum() < 16:
        return float("nan")
    p = np.polyfit(t[mask], y[mask], 1)
    return float(np.std(y[mask] - np.polyval(p, t[mask])))


def main() -> None:
    print("=" * 80)
    print("Phase B1: R32 off vs baseline")
    print("=" * 80)

    results: dict[tuple, dict] = {}
    for v, mode in CASES:
        print(f"\n--- {v} / {mode} ---")
        results[(v, mode)] = run_case(v, mode)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for (v, mode), d in results.items():
        t_min = d["snap_t"] / 60.0
        c = COLORS[(v, mode)]
        ls = "-" if mode == "baseline" else "--"
        lab = f"{v} {mode}"
        axes[0].plot(t_min, d["surf_O3"] * 1e6, color=c, ls=ls, lw=1.6, label=lab)
        axes[1].plot(t_min, d["bulk_O3"] * 1e9, color=c, ls=ls, lw=1.6, label=lab)
        axes[2].plot(t_min, d["no2m"] * 1e6, color=c, ls=ls, lw=1.6, label=lab)

    axes[0].set_ylabel("Surface [O₃] (µM)")
    axes[0].set_title("(a) Surface [O₃] — z=0", fontweight="bold", loc="left")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_ylabel("Bulk-only [O₃] (nM)")
    axes[1].set_title("(b) Bulk-only [O₃] — z>0.1mm vol-avg",
                      fontweight="bold", loc="left")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    axes[2].set_ylabel("Bulk-only [NO₂⁻] (µM)")
    axes[2].set_title("(c) Bulk-only [NO₂⁻] — chemistry coupling partner",
                      fontweight="bold", loc="left")
    axes[2].grid(alpha=0.3)
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=9)
    axes[2].set_xlabel("Time (min)")

    fig.suptitle(
        "Phase B1: R32 (O₃+NO₂⁻→O₂+NO₃⁻) disable vs baseline — "
        "test predator-prey hypothesis",
        fontsize=12, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_b1.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    # Numerical summary: detrended std of bulk O3 in t > 3min
    print("\n=== Detrended std of bulk-only [O₃] (t > 180s) ===")
    summ = ["Phase B1 detrended std (t>180s) of bulk-only [O₃]:", ""]
    for v, mode in CASES:
        d = results[(v, mode)]
        s = detrended_std(d["snap_t"], d["bulk_O3"], 180.0)
        avg = float(np.mean(d["bulk_O3"][d["snap_t"] > 180]))
        cv = (s / avg * 100) if avg > 0 else float("nan")
        line = (f"  {v:>6} / {mode:>10}:  mean={avg*1e9:7.3f} nM, "
                f"detrended std={s*1e9:7.3f} nM  ({cv:5.2f}%)")
        print(line)
        summ.append(line)
    (out / "diag_o3_b1.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
