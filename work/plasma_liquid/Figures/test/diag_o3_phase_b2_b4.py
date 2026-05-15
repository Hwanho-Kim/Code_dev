#!/usr/bin/env python3
"""Phase B2 + B4: chemistry vs input variation isolation.

B2: All O3 sinks disabled (R22-R32, 11 reactions). If oscillation persists,
    chemistry-mediated coupling is NOT the cause.
B4: Gas conc held constant at last-100s mean. If oscillation persists,
    input variation is NOT the cause.

Test cases (2.6 kV only — voltage where oscillation appears):
  baseline                — re-uses HONOvar cache
  o3_sinks_off            — R22-R32 disabled (11 rxns)
  gas_constant            — gas time series constant after t=60s
  o3_sinks_off + gas_const — both
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

# All R22-R32 (O3 as reactant)
O3_SINK_LABELS = [f"R{i}" for i in range(22, 33)]
VOLTAGE = "2.6kV"


def disable_o3_sinks(chem: AqueousChemistry1D) -> int:
    n = 0
    for ri, r in enumerate(chem.reactions):
        label = str(r.get("label", ""))
        if any(label.startswith(f"{lab}:") or label.startswith(f"{lab} ")
               for lab in O3_SINK_LABELS):
            r["k"] = 0.0
            chem._rxn_data[ri]["k"] = 0.0
            n += 1
    print(f"  disabled {n} O3 sink reactions (R22-R32)")
    return n


def hold_gas_constant(times, gas_conc, hono, hono2, h2o2, t_freeze=60.0):
    """Replace each species' time series with constant SS value (last 100s mean)
    after t_freeze."""
    mask_ss = times >= (times[-1] - 100)

    def _avg(arr):
        return float(np.mean(arr[mask_ss]))

    new_gas = {}
    for sp, arr in gas_conc.items():
        ss = _avg(arr)
        a = arr.copy()
        a[times >= t_freeze] = ss
        # Linear interp from a[t_freeze] to ss across [t_freeze - 10, t_freeze]
        # to avoid step discontinuity
        new_gas[sp] = a

    def _const(arr):
        ss = _avg(arr)
        a = arr.copy()
        a[times >= t_freeze] = ss
        return a

    return new_gas, _const(hono), _const(hono2), _const(h2o2)


def run_case(label: str, *, sinks_off: bool, gas_const: bool) -> dict:
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

    if gas_const:
        gas_conc, HONO_GAS, HONO2_GAS, H2O2_GAS = hold_gas_constant(
            times, gas_conc, HONO_GAS, HONO2_GAS, H2O2_GAS, t_freeze=60.0
        )
        print(f"  [{label}] gas conc held constant after t=60s")

    chem = AqueousChemistry1D(saline_mode=False)
    if sinks_off:
        disable_o3_sinks(chem)

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
    print(f"  [{label}] wall={wall:.0f}s, pH={res['pH_avg']:.3f}")

    iO3 = AQUEOUS_SPECIES.index("O3")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    iHp = AQUEOUS_SPECIES.index("H+")
    z_mm = solver.z_centers * 1e3
    dz = solver.dz_cells
    L = solver.L
    mask_b = z_mm > 0.1
    Lb = float(dz[mask_b].sum())
    bulk_O3 = np.array([np.dot(snap_y[k, mask_b, iO3], dz[mask_b]) / Lb
                        for k in range(len(snap_t))])
    surf_O3 = snap_y[:, 0, iO3]
    pKa = ACID_BASE_PAIRS["HONO_total"][2]
    Ka = 10.0 ** (-pKa)
    hono_t = np.array([np.dot(snap_y[k, mask_b, iHONO_t], dz[mask_b]) / Lb
                       for k in range(len(snap_t))])
    hp = np.maximum(np.array([np.dot(snap_y[k, mask_b, iHp], dz[mask_b]) / Lb
                               for k in range(len(snap_t))]), 1e-14)
    no2m = hono_t * Ka / (hp + Ka)
    return {"snap_t": snap_t, "bulk_O3": bulk_O3, "surf_O3": surf_O3,
            "no2m": no2m, "label": label}


def load_baseline() -> dict:
    fp = (_root / "Figures" / "DIW results"
          / f"{VOLTAGE}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    dz = np.asarray(d["dz_cells"])
    L = float(d["L"])
    z_mm = np.asarray(d["z_centers"]) * 1e3
    iO3 = AQUEOUS_SPECIES.index("O3")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    iHp = AQUEOUS_SPECIES.index("H+")
    mask_b = z_mm > 0.1
    Lb = float(dz[mask_b].sum())
    bulk_O3 = np.array([np.dot(snap_y[k, mask_b, iO3], dz[mask_b]) / Lb
                        for k in range(len(snap_t))])
    surf_O3 = snap_y[:, 0, iO3]
    pKa = ACID_BASE_PAIRS["HONO_total"][2]
    Ka = 10.0 ** (-pKa)
    hono_t = np.array([np.dot(snap_y[k, mask_b, iHONO_t], dz[mask_b]) / Lb
                       for k in range(len(snap_t))])
    hp = np.maximum(np.array([np.dot(snap_y[k, mask_b, iHp], dz[mask_b]) / Lb
                               for k in range(len(snap_t))]), 1e-14)
    no2m = hono_t * Ka / (hp + Ka)
    print(f"  [baseline] loaded cache")
    return {"snap_t": snap_t, "bulk_O3": bulk_O3, "surf_O3": surf_O3,
            "no2m": no2m, "label": "baseline"}


def detrended_std(t: np.ndarray, y: np.ndarray, t_min: float) -> tuple:
    mask = t > t_min
    if mask.sum() < 16:
        return float("nan"), float("nan")
    p = np.polyfit(t[mask], y[mask], 1)
    res = y[mask] - np.polyval(p, t[mask])
    return float(np.std(res)), float(np.mean(y[mask]))


def main():
    print("=" * 80)
    print(f"Phase B2+B4 isolation tests — {VOLTAGE}")
    print("=" * 80)

    cases = [
        ("baseline", lambda: load_baseline()),
        ("o3_sinks_off", lambda: run_case("o3_sinks_off",
                                           sinks_off=True, gas_const=False)),
        ("gas_constant", lambda: run_case("gas_constant",
                                           sinks_off=False, gas_const=True)),
        ("sinks_off+gas_const", lambda: run_case("sinks_off+gas_const",
                                                  sinks_off=True, gas_const=True)),
    ]
    results = {}
    for name, fn in cases:
        print(f"\n--- {name} ---")
        results[name] = fn()

    colors = {"baseline": "#d62728", "o3_sinks_off": "#2ca02c",
              "gas_constant": "#1f77b4", "sinks_off+gas_const": "#9467bd"}

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for name, d in results.items():
        t = d["snap_t"] / 60.0
        c = colors[name]
        axes[0].plot(t, d["surf_O3"] * 1e6, color=c, lw=1.5, label=name)
        axes[1].plot(t, d["bulk_O3"] * 1e9, color=c, lw=1.5, label=name)
        axes[2].plot(t, d["no2m"] * 1e6, color=c, lw=1.5, label=name)

    axes[0].set_ylabel("Surface [O₃] (µM)")
    axes[0].set_title("(a) Surface [O₃]", fontweight="bold", loc="left")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_ylabel("Bulk-only [O₃] (nM)")
    axes[1].set_title("(b) Bulk-only [O₃] — z>0.1mm vol-avg",
                      fontweight="bold", loc="left")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    axes[2].set_ylabel("Bulk-only [NO₂⁻] (µM)")
    axes[2].set_title("(c) Bulk-only [NO₂⁻]", fontweight="bold", loc="left")
    axes[2].grid(alpha=0.3)
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=9)
    axes[2].set_xlabel("Time (min)")

    fig.suptitle(
        f"Phase B2+B4 ({VOLTAGE}): O3 sinks disable + gas constant — "
        "isolate oscillation source",
        fontsize=12, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out = Path(__file__).parent
    for ext in ("png", "pdf"):
        p = out / f"fig_diag_o3_b2b4.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"saved: {p}")

    print("\n=== Detrended std of bulk O3 (t > 180s) ===")
    summ = ["Phase B2+B4 detrended std (t>180s):", ""]
    for name, d in results.items():
        s, m = detrended_std(d["snap_t"], d["bulk_O3"], 180.0)
        cv = (s / m * 100) if m > 0 else float("nan")
        line = (f"  {name:>22s}: bulk mean={m*1e9:7.3f} nM, "
                f"std={s*1e9:7.4f} nM ({cv:5.2f}%)")
        print(line)
        summ.append(line)
    (out / "diag_o3_b2b4.txt").write_text("\n".join(summ))


if __name__ == "__main__":
    main()
