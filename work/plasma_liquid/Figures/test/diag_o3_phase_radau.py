#!/usr/bin/env python3
"""Phase Radau: scipy BDF → Radau (implicit RK5) test.

User suggestion (2026-05-06): BDF may have step-size adaptation issues at
extreme gradients (wall edge). Radau is implicit RK 5th order, often more
robust for stiff transients. Test if 4mm peak changes.

Method:
  Monkey-patch pde_solver.solve_ivp to inject method='Radau'.
  Run 3.6 kV baseline. Compare 4mm peak vs BDF.

If Radau gives different 4mm peak → BDF artifact at extreme gradient.
If same → solver-method-independent (deeper SG flux issue or PDE solution).
"""
from __future__ import annotations

import functools
import sys
import time as time_mod
from pathlib import Path

import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

# Import pde_solver first, then monkey-patch its solve_ivp
import pde_solver as _ps  # noqa: E402

_orig_solve_ivp = _ps.solve_ivp


def patched_solve_ivp(fun, t_span, y0, **kwargs):
    kwargs["method"] = "Radau"
    return _orig_solve_ivp(fun, t_span, y0, **kwargs)


_ps.solve_ivp = patched_solve_ivp

# Now safe to import the rest
from chemistry_1d import AqueousChemistry1D  # noqa: E402
from config_1d import AQUEOUS_SPECIES  # noqa: E402

print = functools.partial(print, flush=True)
VOLTAGE = "3.6kV"


def run(label, *, use_radau=True):
    if use_radau:
        _ps.solve_ivp = patched_solve_ivp
    else:
        _ps.solve_ivp = _orig_solve_ivp

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
    solver = _ps.PDESolver1D(
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
    print(f"  [{label}] wall={wall:.0f}s, pH={res['pH_avg']:.3f}, "
          f"nfev={res['nfev']}")

    iO3 = AQUEOUS_SPECIES.index("O3")
    return {
        "label": label,
        "snap_t": snap_t,
        "z_mm": solver.z_centers * 1e3,
        "snap_y_O3": snap_y[:, :, iO3],
        "wall": wall,
        "nfev": res["nfev"],
    }


def main():
    print("=" * 80)
    print(f"Phase Radau ({VOLTAGE}): scipy BDF vs Radau (implicit RK5) comparison")
    print("=" * 80)

    print("\n--- Radau (this run) ---")
    r_radau = run("Radau", use_radau=True)

    # Load BDF baseline from cache
    fp = (_root / "Figures" / "DIW results"
          / f"{VOLTAGE}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    iO3 = AQUEOUS_SPECIES.index("O3")
    snap_y_bdf = np.asarray(d["snap_y"])
    r_bdf = {
        "label": "BDF (cache)",
        "snap_t": np.asarray(d["snap_t"]),
        "z_mm": np.asarray(d["z_centers"]) * 1e3,
        "snap_y_O3": snap_y_bdf[:, :, iO3],
        "wall": 0,
        "nfev": 0,
    }

    print("\n=== Spatial @ t=480s ===")
    print(f"{'z(mm)':>7s} {'BDF':>14s} {'Radau':>14s} {'ratio':>10s}")
    for zt in [0.003, 0.04, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 2.3, 4.0, 7.0, 9.8]:
        j_bdf = int(np.argmin(np.abs(r_bdf["z_mm"] - zt)))
        j_rad = int(np.argmin(np.abs(r_radau["z_mm"] - zt)))
        si_bdf = int(np.argmin(np.abs(r_bdf["snap_t"] - 480)))
        si_rad = int(np.argmin(np.abs(r_radau["snap_t"] - 480)))
        v_bdf = r_bdf["snap_y_O3"][si_bdf, j_bdf]
        v_rad = r_radau["snap_y_O3"][si_rad, j_rad]
        ratio = v_rad / v_bdf if v_bdf else float("inf")
        print(f"{zt:>7.3f} {v_bdf:>+14.3e} {v_rad:>+14.3e} {ratio:>10.2e}")

    # 4mm key value
    j = int(np.argmin(np.abs(r_radau["z_mm"] - 4.0)))
    si = int(np.argmin(np.abs(r_radau["snap_t"] - 480)))
    j_bdf = int(np.argmin(np.abs(r_bdf["z_mm"] - 4.0)))
    si_bdf = int(np.argmin(np.abs(r_bdf["snap_t"] - 480)))
    print(f"\n4mm @ t=480s:")
    print(f"  BDF:   {r_bdf['snap_y_O3'][si_bdf, j_bdf]:.3e}")
    print(f"  Radau: {r_radau['snap_y_O3'][si, j]:.3e}")
    print(f"  ratio: {r_radau['snap_y_O3'][si, j] / r_bdf['snap_y_O3'][si_bdf, j_bdf]:.3e}")
    print(f"  Radau nfev={r_radau['nfev']}, wall={r_radau['wall']:.0f}s")


if __name__ == "__main__":
    main()
