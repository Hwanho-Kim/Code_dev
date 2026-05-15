#!/usr/bin/env python3
"""HONO/NO2 voltage-specific fine-tune sweep.

이전 sweep (0.007/0.030/0.070/0.100)에서 NO2- log-linear interpolation으로
voltage별 narrow range fine-tune 수행. NO2- 실험값 매칭이 목표.

  2.6 kV (exp NO2-=0.00):    HONO/NO2 ∈ {0.000, 0.001, 0.003, 0.005}
  3.2 kV (exp NO2-=3.58):    HONO/NO2 ∈ {0.045, 0.055, 0.060}
  3.6 kV (exp NO2-=20.74):   HONO/NO2 ∈ {0.090, 0.097}

three_film BC + H2O2/O3=0.003 + HONO2/N2O5=0.83 고정.
"""
from __future__ import annotations

import functools
import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))

from chemistry_1d import AqueousChemistry1D  # noqa: E402
from config_1d import N2O4_EQ, PHYSICAL  # noqa: E402
from pde_solver import PDESolver1D  # noqa: E402

print = functools.partial(print, flush=True)

GAS_XLSX = _root / "OAS data" / "Dry" / "(P-L) 가스활성종 농도.xlsx"

RH80_ALL = {
    "2.6kV": {"O3_scale": 0.493, "NO2_O3": 0.222, "N2O5_NO2": 0.043, "NO3_O3": 0.0179},
    "3.2kV": {"O3_scale": 0.647, "NO2_O3": 0.091, "N2O5_NO2": 0.054, "NO3_O3": 0.00442},
    "3.6kV": {"O3_scale": 0.762, "NO2_O3": 0.095, "N2O5_NO2": 0.037, "NO3_O3": 0.00337},
}
EXP_NO2 = {"2.6kV": 0.00, "3.2kV": 3.58, "3.6kV": 20.74}

H2O2_RATIO = 0.003
HONO2_RATIO = 0.83

SWEEP: dict[str, list[float]] = {
    "2.6kV": [0.000, 0.001, 0.003, 0.005],
    "3.2kV": [0.045, 0.055, 0.060],
    "3.6kV": [0.090, 0.097],
}


def _preprocess(vals: np.ndarray, min_run: int = 5) -> np.ndarray:
    out = np.maximum(vals.copy(), 0.0)
    n = len(out)
    run_start, run_len, stable_start = -1, 0, n
    for i in range(n):
        if out[i] > 0:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= min_run:
                stable_start = run_start
                break
        else:
            run_len = 0
    if stable_start >= n:
        return out
    nz = [(i, out[i]) for i in range(stable_start, n) if out[i] > 0]
    if len(nz) >= 2:
        idx = np.array([x[0] for x in nz])
        vs = np.array([x[1] for x in nz])
        for i in range(stable_start, n):
            if out[i] <= 0:
                out[i] = np.interp(i, idx, vs)
    if stable_start > 0:
        first = out[stable_start]
        out[:stable_start] = np.linspace(0, first, stable_start + 1)[:-1]
    return out


def load_gas(voltage: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_excel(GAS_XLSX, sheet_name=voltage)
    times = df.iloc[:, 0].values.astype(float)
    gas: dict[str, np.ndarray] = {}
    for sp in ("O3", "NO2", "NO3", "N2O5"):
        for col in df.columns:
            if sp in str(col):
                gas[sp] = _preprocess(df[col].values.astype(float))
                break
    return times, gas


def apply_rh80(
    gas_dry: dict[str, np.ndarray],
    times: np.ndarray,
    voltage: str,
    hono_ratio: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    r = RH80_ALL[voltage]
    mask = times >= (times[-1] - 100)

    def ss(arr: np.ndarray) -> float:
        return max(float(np.mean(arr[mask])), 1e-30)

    o3d, no2d, n2o5d, no3d = (
        ss(gas_dry["O3"]),
        ss(gas_dry["NO2"]),
        ss(gas_dry["N2O5"]),
        ss(gas_dry["NO3"]),
    )
    o3_80 = o3d * r["O3_scale"]
    no2_80 = o3_80 * r["NO2_O3"]
    n2o5_80 = no2_80 * r["N2O5_NO2"]
    no3_80 = o3_80 * r["NO3_O3"]
    g = {
        "O3": gas_dry["O3"] * (o3_80 / o3d),
        "NO2": gas_dry["NO2"] * (no2_80 / no2d),
        "N2O5": gas_dry["N2O5"] * (n2o5_80 / n2o5d),
        "NO3": gas_dry["NO3"] * (no3_80 / no3d),
    }
    T = N2O4_EQ.REF_TEMP
    Kp = float(np.exp(np.log(N2O4_EQ.KP_298)))
    g["N2O4"] = Kp * PHYSICAL.KB_T_OVER_P * T * (g["NO2"] ** 2)

    hono = g["NO2"] * hono_ratio
    hno3 = g["N2O5"] * HONO2_RATIO
    h2o2 = g["O3"] * H2O2_RATIO
    return g, hono, hno3, h2o2


def run(voltage: str, hono_ratio: float, t_end: float = 600.0) -> dict[str, float]:
    times, gas_dry = load_gas(voltage)
    gas, hono, hno3, h2o2 = apply_rh80(gas_dry, times, voltage, hono_ratio)
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        saline_mode=False,
        bc_type="three_film",
        alpha_b=None,
        delta_gas=0.01,
        delta_liq=1e-4,
    )
    solver.set_gas_data(
        times=times,
        gas_conc_molecules=gas,
        hono_gas=hono,
        hono2_gas=hno3,
        h2o2_gas=h2o2,
    )
    te = np.arange(2, t_end + 0.1, 2)
    y0 = solver.build_initial_condition(initial_pH=7.0)
    t0 = time_mod.time()
    result = solver.solve(
        t_span=(0, t_end), t_eval=te, y0=y0, verbose=False, dt_poisson=None
    )
    wall = time_mod.time() - t0
    avg = result["spatial_avg"]
    return {
        "pH": float(result["pH_avg"]),
        "NO3": float(avg.get("NO3-", 0.0)) * 1e6,
        "NO2": float(avg.get("NO2-", 0.0)) * 1e6,
        "H2O2": float(avg.get("H2O2", 0.0)) * 1e6,
        "wall": wall,
    }


def main() -> None:
    print("=" * 80)
    print("HONO/NO2 voltage-specific fine-tune sweep")
    print("=" * 80)

    results: dict[tuple[str, float], dict[str, float]] = {}
    for v in ("2.6kV", "3.2kV", "3.6kV"):
        print(f"\n--- {v}  (exp NO2-={EXP_NO2[v]:.2f} µM) ---")
        for r in SWEEP[v]:
            print(f"  HONO/NO2={r:.4f} ...", end=" ")
            res = run(v, r)
            results[(v, r)] = res
            err = res["NO2"] - EXP_NO2[v]
            print(
                f"NO2-={res['NO2']:7.3f} (Δ={err:+7.3f}), "
                f"NO3-={res['NO3']:6.2f}, pH={res['pH']:.3f}, "
                f"H2O2={res['H2O2']:6.2f}, {res['wall']:.0f}s"
            )

    print("\n" + "=" * 80)
    print("Best HONO/NO2 per voltage (min |NO2- error|)")
    print("=" * 80)
    best: dict[str, tuple[float, dict[str, float]]] = {}
    for v in ("2.6kV", "3.2kV", "3.6kV"):
        cands = [(r, results[(v, r)]) for r in SWEEP[v]]
        cands.sort(key=lambda x: abs(x[1]["NO2"] - EXP_NO2[v]))
        best[v] = cands[0]
        r_best, res = cands[0]
        print(
            f"  {v}: HONO/NO2={r_best:.4f}  "
            f"NO2-={res['NO2']:.3f} (exp {EXP_NO2[v]:.2f}, Δ={res['NO2']-EXP_NO2[v]:+.3f})  "
            f"NO3-={res['NO3']:.2f}, pH={res['pH']:.3f}, H2O2={res['H2O2']:.2f}"
        )

    out = Path(__file__).parent / "hono_finetune_results.txt"
    with out.open("w") as f:
        f.write("HONO/NO2 voltage-specific fine-tune sweep\n")
        f.write("three_film, H2O2/O3=0.003, HONO2/N2O5=0.83\n\n")
        for v in ("2.6kV", "3.2kV", "3.6kV"):
            f.write(f"\n--- {v} (exp NO2-={EXP_NO2[v]:.2f} µM) ---\n")
            for r in SWEEP[v]:
                res = results[(v, r)]
                f.write(
                    f"  HONO/NO2={r:.4f}  NO2-={res['NO2']:7.3f}  "
                    f"NO3-={res['NO3']:6.2f}  pH={res['pH']:.3f}  H2O2={res['H2O2']:6.2f}\n"
                )
        f.write("\n=== Best per voltage ===\n")
        for v, (rb, res) in best.items():
            f.write(
                f"  {v}: HONO/NO2={rb:.4f}  NO2-={res['NO2']:.3f}  "
                f"NO3-={res['NO3']:.2f}  pH={res['pH']:.3f}  H2O2={res['H2O2']:.2f}\n"
            )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
