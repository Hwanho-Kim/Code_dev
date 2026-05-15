#!/usr/bin/env python3
"""Phase C: isolate diffusion vs chemistry as source of bulk O3 pollution.

Three tests at 3.6 kV:
  baseline       : full sim (cache)
  chem_off       : ALL reaction rates k=0 → only diffusion + MT BC
  diff_off       : D_species[O3] = 0 → only chemistry + surface BC

Comparison criteria at t=480s spatial profile:
  - chem_off should match analytical erfc(z / 2√(Dt)) × c_surf(t)
  - diff_off should keep deep cells at trace floor (no source)
  - baseline U-shape origin localized

Output: fig_diag_o3_c.{png,pdf}, diag_o3_c.txt, plus erfc analytical compare.
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
from config_1d import AQUEOUS_SPECIES  # noqa: E402
from pde_solver import PDESolver1D  # noqa: E402

print = functools.partial(print, flush=True)
VOLTAGE = "3.6kV"


def disable_all_chemistry(chem):
    """Set all reaction rates to zero."""
    n = 0
    for ri, r in enumerate(chem.reactions):
        if "k" in r:
            r["k"] = 0.0
        if "k_f" in r:
            r["k_f"] = 0.0
        if "k_b" in r:
            r["k_b"] = 0.0
        # _rxn_data was built by _precompute_reaction_data
        chem._rxn_data[ri]["k"] = 0.0
        n += 1
    chem._precompute_numba_arrays()
    print(f"  disabled {n} reactions (chem_off)")


def disable_diffusion(solver, species_names=None):
    """Zero out D_species for given species (or all if None)."""
    if species_names is None:
        solver.D_species[:] = 0.0
        print("  zeroed D_species for ALL species")
    else:
        for sp in species_names:
            i = solver.species_idx.get(sp)
            if i is not None:
                solver.D_species[i] = 0.0
        print(f"  zeroed D for: {species_names}")


def run_case(label: str, *, chem_off=False, diff_off=False) -> dict:
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
    if chem_off:
        disable_all_chemistry(chem)

    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        saline_mode=False, bc_type="three_film", alpha_b=None,
        delta_gas=0.01, delta_liq=1e-4,
    )
    if diff_off:
        # Zero diffusion for ALL species (cleanest test).
        disable_diffusion(solver, species_names=None)

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
    print(f"  [{label}] wall={wall:.0f}s, pH={res['pH_avg']:.3f}, nfev={res['nfev']}")

    iO3 = AQUEOUS_SPECIES.index("O3")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    return {
        "label": label,
        "snap_t": snap_t,
        "z_mm": solver.z_centers * 1e3,
        "snap_y_O3": snap_y[:, :, iO3],
        "snap_y_HONO_t": snap_y[:, :, iHONO_t],
        "D_O3": solver.D_species[iO3],
    }


def load_baseline() -> dict:
    fp = (_root / "Figures" / "DIW results"
          / f"{VOLTAGE}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    iO3 = AQUEOUS_SPECIES.index("O3")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    snap_y = np.asarray(d["snap_y"])
    print("  [baseline] loaded cache")
    return {
        "label": "baseline",
        "snap_t": np.asarray(d["snap_t"]),
        "z_mm": np.asarray(d["z_centers"]) * 1e3,
        "snap_y_O3": snap_y[:, :, iO3],
        "snap_y_HONO_t": snap_y[:, :, iHONO_t],
        "D_O3": 1.5e-9,  # Liu 2015 value
    }


def erfc_analytical(z_mm, t_s, D_O3, c_surf_t, t_grid):
    """Pure-diffusion erfc analytical with time-varying surface BC.

    For instantaneous step BC: c(z,t) = c_surf × erfc(z/(2√(Dt)))
    For time-varying c_surf(t'), use Duhamel principle (superposition):
      c(z,t) = ∫₀ᵗ (z/(2√(πD(t-t')³))) × exp(−z²/(4D(t-t'))) × c_surf(t') dt'

    Simple approximation: use final c_surf(t) and erfc — overestimates depth.
    Better: use convolution with c_surf'(t).
    Here: take median c_surf in [0, t] for cleaner comparison.
    """
    z_m = z_mm * 1e-3
    sqrt_Dt = math.sqrt(D_O3 * t_s)
    c_surf_avg = float(np.mean(c_surf_t[t_grid > 0])) if (t_grid > 0).any() else 0.0
    out = np.zeros_like(z_m)
    for i, z in enumerate(z_m):
        out[i] = c_surf_avg * math.erfc(z / (2 * sqrt_Dt))
    return out


def main():
    print("=" * 80)
    print(f"Phase C: chem_off / diff_off tests — {VOLTAGE}")
    print("=" * 80)

    runs = {}
    print("\n--- baseline ---")
    runs["baseline"] = load_baseline()
    print("\n--- chem_off ---")
    runs["chem_off"] = run_case("chem_off", chem_off=True, diff_off=False)
    print("\n--- diff_off ---")
    runs["diff_off"] = run_case("diff_off", chem_off=False, diff_off=True)

    # Plot O3 spatial at t=480s for all 3 cases + erfc analytical
    snap_targets = [60, 180, 300, 480]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    axes = axes.flat

    for ti, t_target in enumerate(snap_targets):
        ax = axes[ti]
        for name, d in runs.items():
            si = int(np.argmin(np.abs(d["snap_t"] - t_target)))
            z = d["z_mm"]
            o3 = d["snap_y_O3"][si]
            ax.plot(z, np.abs(o3) + 1e-30, lw=1.6, label=name)
        # erfc analytical for chem_off (using its own surface time series)
        d = runs["chem_off"]
        c_surf = d["snap_y_O3"][:, 0]
        si = int(np.argmin(np.abs(d["snap_t"] - t_target)))
        # Approximate: erfc with time-averaged surface up to t_target
        mask_t = d["snap_t"] <= t_target
        if mask_t.any():
            c_surf_avg = float(np.mean(c_surf[mask_t]))
            sqrt_Dt = math.sqrt(d["D_O3"] * t_target)
            erfc_pred = np.array([
                c_surf_avg * math.erfc(z_i*1e-3 / (2 * sqrt_Dt))
                for z_i in d["z_mm"]
            ])
            ax.plot(d["z_mm"], np.maximum(erfc_pred, 1e-30), 'k--', lw=1.0,
                    label=f"erfc analytical (c̄_surf={c_surf_avg:.2e})")

        ax.set_yscale("log")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("|[O₃]| (M)")
        ax.set_title(f"t = {t_target} s", fontweight="bold", loc="left")
        ax.set_ylim(1e-30, 1e-4)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle(f"Phase C ({VOLTAGE}): isolate diffusion vs chemistry "
                 "as source of bulk O₃",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_c.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    # Numerical summary at t=480s for select cells
    print("\n=== Spatial profile comparison at t=480s ===")
    cells_to_show = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]
    header = f"{'z(mm)':>8s} | " + ' '.join(f"{name:>14s}" for name in runs.keys())
    print(header)
    summ = ["Phase C spatial @ t=480s:", "", header]
    for j in cells_to_show:
        line = f"{runs['baseline']['z_mm'][j]:>8.3f} |"
        for name, d in runs.items():
            si = int(np.argmin(np.abs(d["snap_t"] - 480)))
            line += f" {d['snap_y_O3'][si, j]:>+14.3e}"
        print(line)
        summ.append(line)
    (out / "diag_o3_c.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
