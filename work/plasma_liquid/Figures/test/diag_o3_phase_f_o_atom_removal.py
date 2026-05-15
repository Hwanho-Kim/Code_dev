#!/usr/bin/env python3
"""Phase F: O atom removal — verify noise-fed R20 hypothesis.

User analysis (2026-05-06):
  Deep cell [O] ≈ 1e-18 (atol-floor noise) × [O2] 2.5e-4 × k=4e9
  → R20_f rate ≈ 1e-12 M/s × 480s ≈ 1e-10 M
  ↔ fig5 4mm peak 1.83e-10 (정확히 같은 자릿수)

Test: zero out all 6 reactions involving O atom:
  R20:  O + O2 ↔ O3       (kf=4e9, kb=3e-6)  [main suspect — R20_f]
  R73:  O + H2O → 2OH     (k=50)
  R106: O + OH- → HO2-    (k=4.2e8)
  R107: O + H2O2 → OH + HO2 (k=1.6e9)
  R108: O + HO2- → OH + O2- (k=5.3e9)
  R109: 2O → O2           (k=2.8e10)

Then [O] is mathematically isolated (no source no sink). Should stay at trace.

Predictions if user hypothesis is correct:
  ✓ fig5 4mm peak DRAMATICALLY reduced (deep bulk O3 → atol noise level)
  ✓ 2.6 kV bulk O3 oscillation amplitude reduced
  ✓ Spatial U-shape replaced by monotonic exponential decay
  ✓ Mid cells (1-2 mm) similar to baseline (atol-noise)

Test 2 voltages: 2.6 kV (oscillation) + 3.6 kV (4mm peak).
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

O_REACTION_LABELS = ["R20", "R73", "R106", "R107", "R108", "R109"]
VOLTAGES = ["2.6kV", "3.6kV"]


def disable_o_atom_reactions(chem):
    """Zero out all reactions involving O atom (reactant or product).
    Calls _precompute_numba_arrays() to actually take effect (★ critical)."""
    n = 0
    for ri, r in enumerate(chem.reactions):
        label = str(r.get("label", ""))
        for tag in O_REACTION_LABELS:
            if label.startswith(f"{tag}:") or label.startswith(f"{tag} "):
                if "k" in r:
                    r["k"] = 0.0
                if "k_f" in r:
                    r["k_f"] = 0.0
                if "k_b" in r:
                    r["k_b"] = 0.0
                chem._rxn_data[ri]["k"] = 0.0
                print(f"  disabled: {label}")
                n += 1
                break
    chem._precompute_numba_arrays()
    print(f"  total {n} O-involved reactions disabled, Numba arrays rebuilt")
    return n


def run_case(voltage: str, label: str, *, o_off=False, rerun=False) -> dict:
    import gen_all_figures as gaf

    gaf.IS_SALINE = False
    gaf.DEFAULT_GAS_SHEET = voltage
    gaf.SOLUTION_LABEL = "DIW"
    gaf.FIXED_CATION_CONC = 0.0
    gaf.CONDITION_LABEL = "Humid_fitting"
    gaf.EXP = gaf.EXP_DIW_ALL[voltage]

    times, gas_conc = gaf.load_gas_data()
    HONO_GAS = gas_conc["NO2"] * gaf.RH80_RATIOS[voltage]["HONO_NO2"]
    HONO2_GAS = gas_conc["N2O5"] * gaf.HONO2_RATIO
    H2O2_GAS = gas_conc["O3"] * gaf.H2O2_RATIO

    if not o_off and not rerun:
        # Use cached baseline
        fp = (_root / "Figures" / "DIW results"
              / f"{voltage}_Humid_fitting_three_film_HONOvar"
              / "cache" / "three_film_abspecies_dg0.0100.npz")
        d = dict(np.load(fp, allow_pickle=True))
        print(f"  [{voltage}/{label}] loaded cache")
        return _extract(d, voltage, label)

    chem = AqueousChemistry1D(saline_mode=False)
    if o_off:
        disable_o_atom_reactions(chem)

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
    print(f"  [{voltage}/{label}] wall={wall:.0f}s, pH={res['pH_avg']:.3f}, "
          f"nfev={res['nfev']}")

    z_centers = solver.z_centers
    return _extract({
        "snap_t": snap_t,
        "snap_y": snap_y,
        "z_centers": z_centers,
        "dz_cells": solver.dz_cells,
        "L": solver.L,
    }, voltage, label)


def _extract(d, voltage, label):
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    dz = np.asarray(d["dz_cells"])
    L = float(d["L"])
    z_mm = np.asarray(d["z_centers"]) * 1e3
    iO3 = AQUEOUS_SPECIES.index("O3")
    iO = AQUEOUS_SPECIES.index("O")
    iOH = AQUEOUS_SPECIES.index("OH")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    iHp = AQUEOUS_SPECIES.index("H+")

    mask_b = z_mm > 0.1
    Lb = float(dz[mask_b].sum())
    bulk_O3 = np.array([np.dot(snap_y[k, mask_b, iO3], dz[mask_b]) / Lb
                        for k in range(len(snap_t))])
    surf_O3 = snap_y[:, 0, iO3]

    pKa = ACID_BASE_PAIRS["HONO_total"][2]
    Ka = 10.0 ** (-pKa)
    hono_b = np.array([np.dot(snap_y[k, mask_b, iHONO_t], dz[mask_b]) / Lb
                       for k in range(len(snap_t))])
    hp_b = np.maximum(np.array([np.dot(snap_y[k, mask_b, iHp], dz[mask_b]) / Lb
                                 for k in range(len(snap_t))]), 1e-14)
    no2m = hono_b * Ka / (hp_b + Ka)

    return {
        "voltage": voltage, "label": label,
        "snap_t": snap_t, "z_mm": z_mm,
        "snap_y_O3": snap_y[:, :, iO3],
        "snap_y_O":  snap_y[:, :, iO],
        "snap_y_OH": snap_y[:, :, iOH],
        "surf_O3": surf_O3, "bulk_O3": bulk_O3, "no2m_bulk": no2m,
    }


def main():
    print("=" * 80)
    print("Phase F: O atom reactions removal — noise-fed R20 hypothesis test")
    print("=" * 80)

    runs = {}
    for V in VOLTAGES:
        print(f"\n--- {V} baseline ---")
        runs[(V, "baseline")] = run_case(V, "baseline", o_off=False)
        print(f"\n--- {V} O-reactions OFF ---")
        runs[(V, "o_off")] = run_case(V, "o_off", o_off=True)

    # Spatial profile at t=480s (3.6 kV) and t=240s (2.6 kV oscillation phase)
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    snap_targets = {"2.6kV": [60, 240, 480], "3.6kV": [60, 240, 480]}

    for ci, V in enumerate(VOLTAGES):
        # Row 0: O3 spatial at multiple times
        ax = axes[0, ci]
        for d_label, ls in [("baseline", "-"), ("o_off", "--")]:
            d = runs[(V, d_label)]
            for tt in snap_targets[V]:
                si = int(np.argmin(np.abs(d["snap_t"] - tt)))
                ax.plot(d["z_mm"], np.abs(d["snap_y_O3"][si]) + 1e-40,
                        ls=ls, lw=1.4,
                        label=f"{d_label} t={tt}s")
        ax.set_yscale("log")
        ax.set_ylim(1e-30, 1e-3)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("[O₃] (M)")
        ax.set_title(f"({chr(97+ci)}) {V} O₃ spatial",
                     fontweight="bold", loc="left")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)

        # Row 1: O atom spatial
        ax = axes[1, ci]
        for d_label, ls in [("baseline", "-"), ("o_off", "--")]:
            d = runs[(V, d_label)]
            for tt in snap_targets[V]:
                si = int(np.argmin(np.abs(d["snap_t"] - tt)))
                ax.plot(d["z_mm"], np.abs(d["snap_y_O"][si]) + 1e-40,
                        ls=ls, lw=1.4,
                        label=f"{d_label} t={tt}s")
        ax.set_yscale("log")
        ax.set_ylim(1e-35, 1e-15)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("[O atom] (M)")
        ax.set_title(f"({chr(99+ci)}) {V} O atom spatial",
                     fontweight="bold", loc="left")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)

        # Row 2: bulk-only O3 time series
        ax = axes[2, ci]
        for d_label, color, ls in [
            ("baseline", "#d62728", "-"),
            ("o_off", "#1f77b4", "--"),
        ]:
            d = runs[(V, d_label)]
            ax.plot(d["snap_t"]/60, d["bulk_O3"]*1e9, color=color, ls=ls,
                    lw=1.6, label=d_label)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Bulk-only [O₃] (nM)")
        ax.set_title(f"({chr(101+ci)}) {V} bulk-only [O₃]",
                     fontweight="bold", loc="left")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Phase F: O atom reactions OFF vs baseline — "
                 "test noise-fed R20 hypothesis",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_f.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    # Numerical summary
    print("\n=== Spatial @ t=480s (3.6 kV) ===")
    summ = ["Phase F O3 spatial summary @ t=480s:", "",
            f"{'voltage':>8s} {'case':>10s} {'z(mm)':>7s} {'O3 (M)':>14s} "
            f"{'[O atom] (M)':>14s} {'[OH] (M)':>14s}"]
    print(summ[2])
    for V in VOLTAGES:
        for case in ["baseline", "o_off"]:
            d = runs[(V, case)]
            si = int(np.argmin(np.abs(d["snap_t"] - 480)))
            for zt in [0.003, 0.2, 1.3, 4.0, 9.8]:
                j = int(np.argmin(np.abs(d["z_mm"] - zt)))
                line = (f"{V:>8s} {case:>10s} {d['z_mm'][j]:>7.3f} "
                        f"{d['snap_y_O3'][si, j]:>+14.3e} "
                        f"{d['snap_y_O'][si, j]:>+14.3e} "
                        f"{d['snap_y_OH'][si, j]:>+14.3e}")
                print(line)
                summ.append(line)

    # Detrended std of bulk-only O3 (t > 180s) — limit cycle test
    print("\n=== Detrended std of bulk-only [O3] (t > 180s) ===")
    summ.append("\nDetrended std (t > 180s):")
    for V in VOLTAGES:
        for case in ["baseline", "o_off"]:
            d = runs[(V, case)]
            mask = d["snap_t"] > 180
            p = np.polyfit(d["snap_t"][mask], d["bulk_O3"][mask], 1)
            resid = d["bulk_O3"][mask] - np.polyval(p, d["snap_t"][mask])
            std = float(np.std(resid))
            mean = float(np.mean(d["bulk_O3"][mask]))
            cv = std / mean * 100 if mean > 0 else float("nan")
            line = (f"  {V:>6} {case:>10}: bulk mean={mean*1e9:7.3f} nM, "
                    f"std={std*1e9:7.4f} nM ({cv:5.2f}%)")
            print(line)
            summ.append(line)

    (out / "diag_o3_f.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
