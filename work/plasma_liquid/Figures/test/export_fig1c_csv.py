#!/usr/bin/env python3
"""Export fig1c raw timeseries data to CSV for Origin replotting.

Reproduces the spatial-averaging + pH-dependent speciation used in
gen_all_figures.gen_fig1c(). One CSV per voltage.

Columns:
  time_min,
  NO3- (uM), NO2- (uM), O3 (uM), H2O2 (uM),
  OH (pM), HO2 (pM), O2- (pM),
  ONOOH (pM), ONOO- (pM),
  O2NOOH (pM), O2NOO- (pM),
  O3- (pM), HO3 (pM), O (pM)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).parent.parent.parent  # work/plasma_liquid
sys.path.insert(0, str(_root / "Ver4_1D"))

from config_1d import AQUEOUS_SPECIES, ACID_BASE_PAIRS  # noqa: E402


SUFFIX = "HONOvar_v3"
VOLTAGES = ["2.6kV", "3.2kV", "3.6kV"]
RESULTS_DIR = _root / "Figures" / "DIW results"


def export_one(voltage: str) -> None:
    folder = RESULTS_DIR / f"{voltage}_Humid_fitting_three_film_{SUFFIX}"
    cache = folder / "cache" / "three_film_abspecies_dg0.0100.npz"
    if not cache.exists():
        print(f"  [SKIP] cache not found: {cache}")
        return

    d = dict(np.load(cache, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"], dtype=float)
    snap_y = np.asarray(d["snap_y"], dtype=float)
    dz = np.asarray(d["dz_cells"], dtype=float)
    L = float(d["L"])
    sp_idx = {sp: i for i, sp in enumerate(AQUEOUS_SPECIES)}

    def avg(species: str) -> np.ndarray:
        i = sp_idx.get(species)
        if i is None:
            return np.zeros(len(snap_t))
        return np.array([np.dot(snap_y[k, :, i], dz) / L
                         for k in range(len(snap_t))])

    hp = np.maximum(avg("H+"), 1e-14)
    Ka_hono  = 10.0 ** (-ACID_BASE_PAIRS["HONO_total"][2])
    Ka_hono2 = 10.0 ** (-ACID_BASE_PAIRS["HONO2_total"][2])
    Ka_h2o2  = 10.0 ** (-ACID_BASE_PAIRS["H2O2_total"][2])
    Ka_HO2     = 10.0 ** (-ACID_BASE_PAIRS["HO2_total"][2])
    Ka_ONOOH   = 10.0 ** (-ACID_BASE_PAIRS["ONOOH_total"][2])
    Ka_O2NOOH  = 10.0 ** (-ACID_BASE_PAIRS["O2NOOH_total"][2])

    f_HO2     = hp / (hp + Ka_HO2)
    f_ONOOH   = hp / (hp + Ka_ONOOH)
    f_O2NOOH  = hp / (hp + Ka_O2NOOH)

    # Long-lived (mol/L → µM; O3 → nM)
    NO3m = avg("HONO2_total") * Ka_hono2 / (hp + Ka_hono2) * 1e6
    NO2m = avg("HONO_total")  * Ka_hono  / (hp + Ka_hono)  * 1e6
    O3   = avg("O3") * 1e9  # nM
    H2O2 = avg("H2O2_total")  * hp / (hp + Ka_h2o2) * 1e6

    # Short-lived (mol/L → pM)
    HO2_t    = avg("HO2_total")
    ONOOH_t  = avg("ONOOH_total")
    O2NOOH_t = avg("O2NOOH_total")
    HO2_mol  = HO2_t    * f_HO2       * 1e12
    O2m      = HO2_t    * (1 - f_HO2) * 1e12
    ONOOH    = ONOOH_t  * f_ONOOH       * 1e12
    ONOOm    = ONOOH_t  * (1 - f_ONOOH) * 1e12
    O2NOOH   = O2NOOH_t * f_O2NOOH       * 1e12
    O2NOOm   = O2NOOH_t * (1 - f_O2NOOH) * 1e12
    OH       = avg("OH")  * 1e12
    O3m_rad  = avg("O3-") * 1e12
    HO3      = avg("HO3") * 1e12
    O_atom   = avg("O")   * 1e12

    tmin = snap_t / 60.0
    header = ["time_min",
              "NO3- (uM)", "NO2- (uM)", "O3 (nM)", "H2O2 (uM)",
              "OH (pM)", "HO2 (pM)", "O2- (pM)",
              "ONOOH (pM)", "ONOO- (pM)",
              "O2NOOH (pM)", "O2NOO- (pM)",
              "O3- (pM)", "HO3 (pM)", "O (pM)"]
    cols = [tmin, NO3m, NO2m, O3, H2O2,
            OH, HO2_mol, O2m,
            ONOOH, ONOOm,
            O2NOOH, O2NOOm,
            O3m_rad, HO3, O_atom]
    arr = np.column_stack(cols)

    out_csv = folder / f"fig1c_data_{voltage}.csv"
    np.savetxt(out_csv, arr, delimiter=",",
               header=",".join(header), comments="",
               fmt=["%.4f"] + ["%.6e"] * (len(header) - 1))
    print(f"  saved: {out_csv}  ({len(tmin)} rows)")


def main() -> None:
    print(f"=== Exporting fig1c CSV for {SUFFIX} ({len(VOLTAGES)} voltages) ===")
    for v in VOLTAGES:
        print(f"\n--- {v} ---")
        export_one(v)


if __name__ == "__main__":
    main()
