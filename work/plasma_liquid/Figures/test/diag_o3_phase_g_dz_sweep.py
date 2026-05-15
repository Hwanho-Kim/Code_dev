#!/usr/bin/env python3
"""Phase G: dz_min grid sweep — verify numerical leak hypothesis.

User analysis (2026-05-06):
  - R20-mediated O atom chain DOES NOT explain 4mm peak (Phase F kept)
  - Surface MT is sole O3 source. Reactive penetration ∫dz/λ ~300 makes wall.
  - chem_off → erfc OK (diffusion solver fine).
  - chemistry-ON only activates spurious leak channel.
  - Wall edge BDF Jacobian condition number ~10^15 → potential numerical mass leak.

Test: dz_min sweep at 3.6 kV, fixed stretch=1.12, fixed L=10mm.
  - dz_min = 1, 5, 10, 20 µm → cell count varies (~63 / 49 / 43 / 37 cells)

Decision criteria:
  - 4mm O3 STRONG dz dependence → NUMERICAL LEAK (depends on grid spacing)
  - 4mm O3 weak dz dependence (<10x variation) → physical (grid-independent)

For physical reactive-diffusion at SS, the depth profile depends on continuous
NO2⁻(z) and λ(z), not on numerical discretization. Grid sensitivity reveals
discretization-induced artifacts.
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
VOLTAGE = "3.6kV"
DZ_MIN_VALUES = [1e-6, 5e-6, 10e-6, 20e-6]  # 1, 5, 10, 20 µm


def run_case(dz_min: float) -> dict:
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
    t0 = time_mod.time()
    res = solver.solve(t_span=(0, t_end), t_eval=te, y0=y0, verbose=False,
                       dt_poisson=None)
    wall = time_mod.time() - t0
    snap_t = np.asarray(res["t_eval"])
    snap_y = np.asarray(res["y_eval"]).reshape(len(snap_t), solver.N_z, solver.N_s)
    z_mm = solver.z_centers * 1e3
    print(f"  [dz_min={dz_min*1e6:.0f}µm, N_z={solver.N_z}] wall={wall:.0f}s, "
          f"pH={res['pH_avg']:.3f}, nfev={res['nfev']}")

    iO3 = AQUEOUS_SPECIES.index("O3")
    return {
        "dz_min_um": dz_min * 1e6,
        "N_z": solver.N_z,
        "snap_t": snap_t,
        "z_mm": z_mm,
        "snap_y_O3": snap_y[:, :, iO3],
        "wall": wall,
        "nfev": res["nfev"],
    }


def main():
    print("=" * 80)
    print(f"Phase G: dz_min grid sweep — {VOLTAGE} (numerical leak verification)")
    print("=" * 80)

    runs = []
    for dz in DZ_MIN_VALUES:
        print(f"\n--- dz_min = {dz*1e6:.0f} µm ---")
        runs.append(run_case(dz))

    # Plot O3 spatial at t=480s on common z-axis (interpolated for comparison)
    z_query = np.array([0.003, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0,
                         4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.8])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(runs)))

    # (a) O3 spatial profile at t=480s
    ax = axes[0]
    for d, c in zip(runs, cmap):
        si = int(np.argmin(np.abs(d["snap_t"] - 480)))
        ax.plot(d["z_mm"], np.abs(d["snap_y_O3"][si]) + 1e-40, color=c, lw=1.6,
                marker="o", ms=3,
                label=f"dz_min={d['dz_min_um']:.0f}µm (N={d['N_z']})")
    ax.set_yscale("log")
    ax.set_ylim(1e-30, 1e-3)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("[O₃] (M)")
    ax.set_title("(a) O₃ spatial @ t=480s — dz_min sweep",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # (b) O3 at z=4mm time series
    ax = axes[1]
    for d, c in zip(runs, cmap):
        # Find cell at z≈4mm
        j_4mm = int(np.argmin(np.abs(d["z_mm"] - 4.0)))
        z_actual = d["z_mm"][j_4mm]
        ax.plot(d["snap_t"]/60, d["snap_y_O3"][:, j_4mm], color=c, lw=1.6,
                label=f"dz_min={d['dz_min_um']:.0f}µm (z={z_actual:.2f}mm)")
    ax.set_yscale("log")
    ax.set_ylim(1e-25, 1e-7)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("[O₃] (M)")
    ax.set_title("(b) O₃ at z≈4mm vs time", fontweight="bold", loc="left")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    fig.suptitle(f"Phase G ({VOLTAGE}): dz_min grid sweep — "
                 "verify numerical leak hypothesis",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_g.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    # Numerical summary at t=480s for various depths
    print("\n=== O3 spatial @ t=480s — dz_min sensitivity ===")
    summ = ["Phase G O3 spatial @ t=480s vs dz_min:", "",
            f"{'z(mm)':>7s} " +
            ' '.join(f"dz={d['dz_min_um']:.0f}µm" for d in runs)]
    print(summ[2])
    for zt in [0.003, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 9.5]:
        line = f"{zt:>7.3f} "
        for d in runs:
            j = int(np.argmin(np.abs(d["z_mm"] - zt)))
            si = int(np.argmin(np.abs(d["snap_t"] - 480)))
            v = d["snap_y_O3"][si, j]
            line += f" {v:>+12.3e}"
        print(line)
        summ.append(line)

    # Sensitivity ratio @ 4mm (most discriminating)
    o3_4mm = []
    for d in runs:
        j = int(np.argmin(np.abs(d["z_mm"] - 4.0)))
        si = int(np.argmin(np.abs(d["snap_t"] - 480)))
        o3_4mm.append(d["snap_y_O3"][si, j])
    print("\n=== 4mm peak sensitivity to dz_min ===")
    for d, v in zip(runs, o3_4mm):
        print(f"  dz={d['dz_min_um']:>3.0f}µm (N={d['N_z']:>3}): O3@4mm = {v:.3e}")
    if len(o3_4mm) >= 2:
        ratio = max(o3_4mm) / max(min(o3_4mm), 1e-40)
        print(f"\n  max/min ratio across dz: {ratio:.2e}")
        verdict = ("STRONG dz dependence → NUMERICAL LEAK"
                   if ratio > 10 else
                   "Weak dz dependence (<10×) → physical")
        print(f"  Verdict: {verdict}")
        summ.append(f"\nmax/min ratio at 4mm: {ratio:.2e}  ({verdict})")

    (out / "diag_o3_g.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
