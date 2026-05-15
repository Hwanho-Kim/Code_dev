#!/usr/bin/env python3
"""Phase α: full wall cells dump from baseline cache (t=480s)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
from config_1d import ACID_BASE_PAIRS, AQUEOUS_SPECIES  # noqa: E402

PKA_HONO = ACID_BASE_PAIRS["HONO_total"][2]
KA_HONO = 10.0 ** (-PKA_HONO)


def dump(voltage):
    fp = (_root / "Figures" / "DIW results"
          / f"{voltage}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    z_mm = np.asarray(d["z_centers"]) * 1e3

    si = int(np.argmin(np.abs(snap_t - 480)))
    iO3 = AQUEOUS_SPECIES.index("O3")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    iHp = AQUEOUS_SPECIES.index("H+")
    iOH = AQUEOUS_SPECIES.index("OH")

    o3 = snap_y[si, :, iO3]
    hono_t = snap_y[si, :, iHONO_t]
    hp = np.maximum(snap_y[si, :, iHp], 1e-14)
    no2m = hono_t * KA_HONO / (hp + KA_HONO)
    oh = snap_y[si, :, iOH]

    print(f"\n=== {voltage} t=480s — all 49 cells (baseline) ===")
    print(f"{'cell':>4} {'z(mm)':>8} {'O3 (M)':>14} {'NO2- (M)':>14} "
          f"{'OH (M)':>14} {'H+ (M)':>14} {'pH':>5}")
    for j in range(len(z_mm)):
        print(f"{j:>4} {z_mm[j]:>8.4f} {o3[j]:>+14.3e} {no2m[j]:>+14.3e} "
              f"{oh[j]:>+14.3e} {hp[j]:>+14.3e} {-np.log10(hp[j]):>5.2f}")


for V in ("2.6kV", "3.2kV", "3.6kV"):
    dump(V)
