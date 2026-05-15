#!/usr/bin/env python3
"""Phase D: atol sweep — verify atol-floor causes deep bulk pollution.

Test at 3.6 kV with O3 spatial profile at t=480s:
  Cases × atol levels:
    chem_off  × {1e-15, 1e-20, 1e-25, 1e-30}
    baseline  × {1e-15, 1e-20, 1e-25, 1e-30}

Expected if atol-floor is the cause:
  - chem_off: deep cells should follow erfc analytical (down to ~1e-22 at 4mm)
  - baseline: 4mm peak (1.83e-10) should drop dramatically as atol tightens
"""
from __future__ import annotations

import functools
import math
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
ATOL_VALUES = [1e-15, 1e-20, 1e-25, 1e-30]


def disable_all_chemistry(chem):
    for ri, r in enumerate(chem.reactions):
        if "k" in r:
            r["k"] = 0.0
        if "k_f" in r:
            r["k_f"] = 0.0
        if "k_b" in r:
            r["k_b"] = 0.0
        chem._rxn_data[ri]["k"] = 0.0
    chem._precompute_numba_arrays()


def run_case(label: str, atol: float, chem_off: bool) -> dict:
    import gen_all_figures as gaf

    cfg.ODE_CONFIG.atol = atol
    cfg.ODE_CONFIG.rtol = 1e-6
    cfg.ODE_CONFIG.max_step = 1.0

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
    if chem_off:
        disable_all_chemistry(chem)

    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
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
    print(f"  [{label} atol={atol:.0e}] wall={wall:.0f}s, "
          f"pH={res['pH_avg']:.3f}, nfev={res['nfev']}")

    iO3 = AQUEOUS_SPECIES.index("O3")
    return {
        "label": label,
        "atol": atol,
        "snap_t": snap_t,
        "z_mm": solver.z_centers * 1e3,
        "snap_y_O3": snap_y[:, :, iO3],
        "wall": wall,
        "nfev": res["nfev"],
        "D_O3": float(solver.D_species[iO3]),
    }


def main():
    print("=" * 80)
    print(f"Phase D: atol sweep — {VOLTAGE}")
    print("=" * 80)

    runs = {}
    for atol in ATOL_VALUES:
        for label, chem_off in [("chem_off", True), ("baseline", False)]:
            key = f"{label}_atol{atol:.0e}"
            print(f"\n--- {key} ---")
            runs[key] = run_case(label, atol, chem_off)

    # Plot at t=480s
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # erfc analytical (using tightest atol chem_off as reference)
    d_ref = runs[f"chem_off_atol{ATOL_VALUES[-1]:.0e}"]
    si = int(np.argmin(np.abs(d_ref["snap_t"] - 480)))
    c_surf_ref = d_ref["snap_y_O3"][si, 0]
    sqrt_Dt = math.sqrt(d_ref["D_O3"] * 480)
    z_smooth = np.linspace(d_ref["z_mm"].min(), d_ref["z_mm"].max(), 200)
    erfc_ana = np.array([c_surf_ref * math.erfc(z*1e-3/(2*sqrt_Dt)) for z in z_smooth])
    erfc_ana = np.maximum(erfc_ana, 1e-40)

    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(ATOL_VALUES)))
    for label_idx, label_kind in enumerate(["chem_off", "baseline"]):
        ax = axes[label_idx]
        for ai, atol in enumerate(ATOL_VALUES):
            d = runs[f"{label_kind}_atol{atol:.0e}"]
            si = int(np.argmin(np.abs(d["snap_t"] - 480)))
            o3 = d["snap_y_O3"][si]
            ax.plot(d["z_mm"], np.abs(o3) + 1e-40, color=cmap[ai], lw=1.6,
                    label=f"atol={atol:.0e}")
        ax.plot(z_smooth, erfc_ana, "k--", lw=1.2,
                label=f"erfc analytical (c̄_surf={c_surf_ref:.2e})")
        ax.set_yscale("log")
        ax.set_ylim(1e-40, 1e-3)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("|[O₃]| (M)")
        ax.set_title(f"{label_kind} t=480s", fontweight="bold", loc="left")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle(f"Phase D ({VOLTAGE}): atol effect on O₃ spatial — "
                 "verify atol-floor as deep bulk pollution source",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_d.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    # Numerical summary
    print("\n=== Spatial @ t=480s ===")
    cells_show = [0, 10, 20, 30, 40, 48]
    z_show = runs[f"chem_off_atol{ATOL_VALUES[0]:.0e}"]["z_mm"][cells_show]
    summ = ["Phase D O3 at t=480s, key cells:", "",
            f"{'case':<25s} " + ' '.join(f"z={z:.2f}mm" for z in z_show)]
    print(summ[2])
    for label_kind in ["chem_off", "baseline"]:
        for atol in ATOL_VALUES:
            d = runs[f"{label_kind}_atol{atol:.0e}"]
            si = int(np.argmin(np.abs(d["snap_t"] - 480)))
            vals = [d["snap_y_O3"][si, j] for j in cells_show]
            line = (f"{label_kind:<10s} atol={atol:>6.0e} "
                    + ' '.join(f"{v:>+10.2e}" for v in vals)
                    + f" wall={d['wall']:>4.0f}s nfev={d['nfev']}")
            print(line)
            summ.append(line)
    (out / "diag_o3_d.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
