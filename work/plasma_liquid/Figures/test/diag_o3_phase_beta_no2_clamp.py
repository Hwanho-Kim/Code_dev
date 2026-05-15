#!/usr/bin/env python3
"""Phase β: NO2⁻ clamp test — verify oscillation mechanism.

User analysis (2026-05-06): R32 (mutual annihilation, dx/dt=dy/dt=-kxy) cannot
generate sustained limit cycle. If oscillation persists with NO2⁻ frozen at
quasi-SS, then NO2⁻ is not the dynamical variable; if oscillation disappears,
then NO2⁻/HONO_total is the driver.

Test: 2.6 kV with HONO_total dydt forced to 0 in all cells (rhs monkey patch).
HONO_total stays at initial value throughout sim → NO2⁻ ≈ const.
Compare bulk-only [O3] oscillation amplitude with baseline.

Initial condition: cache spatial profile at t=200s (after wall formation,
oscillation phase).
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


def run_case(label, *, clamp_hono_total=False, t_start=200.0, t_end=600.0):
    """Run 2.6 kV from t_start to t_end with optional HONO_total clamp.

    Initial state: from baseline cache at t_start.
    If clamp_hono_total: solver.rhs patched to set dydt[HONO_total]=0 in all cells.
    """
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
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type="three_film", alpha_b=None,
        delta_gas=0.01, delta_liq=1e-4,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=HONO_GAS, hono2_gas=HONO2_GAS,
                        h2o2_gas=H2O2_GAS)

    # Load initial state from cache at t_start
    fp = (_root / "Figures" / "DIW results"
          / f"{VOLTAGE}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t_cache = np.asarray(d["snap_t"])
    snap_y_cache = np.asarray(d["snap_y"])
    si = int(np.argmin(np.abs(snap_t_cache - t_start)))
    y0_2d = snap_y_cache[si].copy()
    y0 = y0_2d.ravel()

    # Patch rhs if clamping
    if clamp_hono_total:
        original_rhs = solver.rhs
        HONO_idx = solver.species_idx["HONO_total"]
        def patched_rhs(t, y):
            dydt = original_rhs(t, y)
            dydt_2d = dydt.reshape(solver.N_z, solver.N_s)
            dydt_2d[:, HONO_idx] = 0.0
            return dydt_2d.ravel()
        solver.rhs = patched_rhs
        print(f"  HONO_total dydt CLAMPED to 0 in all {solver.N_z} cells")

    te = np.arange(t_start + 2, t_end + 0.1, 2)
    t0 = time_mod.time()
    res = solver.solve(t_span=(t_start, t_end), t_eval=te, y0=y0,
                       verbose=False, dt_poisson=None)
    wall = time_mod.time() - t0
    snap_t = np.asarray(res["t_eval"])
    snap_y = np.asarray(res["y_eval"]).reshape(len(snap_t), solver.N_z, solver.N_s)
    print(f"  [{label}] wall={wall:.0f}s, pH={res['pH_avg']:.3f}, "
          f"nfev={res['nfev']}")

    iO3 = AQUEOUS_SPECIES.index("O3")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    iHp = AQUEOUS_SPECIES.index("H+")
    z_mm = solver.z_centers * 1e3
    dz = solver.dz_cells
    L = solver.L
    pKa = ACID_BASE_PAIRS["HONO_total"][2]
    Ka = 10.0 ** (-pKa)

    mask_b = z_mm > 0.1
    Lb = float(dz[mask_b].sum())
    bulk_O3 = np.array([np.dot(snap_y[k, mask_b, iO3], dz[mask_b]) / Lb
                        for k in range(len(snap_t))])
    hono_b = np.array([np.dot(snap_y[k, mask_b, iHONO_t], dz[mask_b]) / Lb
                       for k in range(len(snap_t))])
    hp_b = np.maximum(np.array([np.dot(snap_y[k, mask_b, iHp], dz[mask_b]) / Lb
                                 for k in range(len(snap_t))]), 1e-14)
    no2m = hono_b * Ka / (hp_b + Ka)

    return {
        "label": label,
        "snap_t": snap_t,
        "bulk_O3": bulk_O3,
        "no2m": no2m,
        "wall": wall,
    }


def main():
    print("=" * 80)
    print(f"Phase β ({VOLTAGE}): NO2⁻ clamp test")
    print("=" * 80)

    print("\n--- baseline (no clamp, t=200→600s) ---")
    base = run_case("baseline", clamp_hono_total=False, t_start=200.0, t_end=600.0)

    print("\n--- HONO_total clamped (NO2⁻ ~const) ---")
    clamp = run_case("clamp_HONO", clamp_hono_total=True, t_start=200.0, t_end=600.0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax = axes[0]
    ax.plot(base["snap_t"]/60, base["bulk_O3"]*1e9, "r-", lw=1.6,
            label="baseline")
    ax.plot(clamp["snap_t"]/60, clamp["bulk_O3"]*1e9, "b--", lw=1.6,
            label="HONO_total clamped (NO2⁻ ~const)")
    ax.set_ylabel("Bulk-only [O₃] (nM)")
    ax.set_title("(a) Bulk-only [O₃] — NO2⁻ as dynamical variable test",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(base["snap_t"]/60, base["no2m"]*1e6, "r-", lw=1.6, label="baseline")
    ax.plot(clamp["snap_t"]/60, clamp["no2m"]*1e6, "b--", lw=1.6,
            label="clamped (should be ~const)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Bulk-only [NO₂⁻] (µM)")
    ax.set_title("(b) Bulk-only [NO₂⁻] — verification of clamp",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    fig.suptitle(f"Phase β ({VOLTAGE}): NO2⁻ clamp removes oscillation?",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_beta.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    # Detrended std comparison
    print("\n=== Detrended std comparison ===")
    for d in [base, clamp]:
        # Use t > 250s (after initial transient from cache restart)
        mask = d["snap_t"] > 250
        p = np.polyfit(d["snap_t"][mask], d["bulk_O3"][mask], 1)
        resid = d["bulk_O3"][mask] - np.polyval(p, d["snap_t"][mask])
        std = float(np.std(resid))
        mean = float(np.mean(d["bulk_O3"][mask]))
        cv = (std / mean * 100) if mean > 0 else float("nan")
        print(f"  {d['label']:>20}: bulk mean={mean*1e9:.3f} nM, "
              f"std={std*1e9:.4f} nM ({cv:.2f}%)")


if __name__ == "__main__":
    main()
