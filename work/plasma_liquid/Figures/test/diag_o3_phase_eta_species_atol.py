#!/usr/bin/env python3
"""Phase η: 모든 25 species의 bulk-only avg vs atol.

atol 미만 농도를 갖는 species가 O atom 외 또 있는지, 그리고 그것들이
reactant로 들어가는 reactions의 의미를 검증.

각 voltage × 3 시점 (60s, 240s, 480s)에서 모든 25 species의
bulk-only avg + surface 값 dump.

atol 기준:
  default: 1e-15
  tight:   1e-17 for {OH, O-, O3-, HO3, NO3}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
from config_1d import AQUEOUS_SPECIES  # noqa: E402

ATOL_DEFAULT = 1e-15
ATOL_TIGHT = 1e-17
TIGHT_SPECIES = {"OH", "O-", "O3-", "HO3", "NO3"}


def dump_voltage(voltage):
    fp = (_root / "Figures" / "DIW results"
          / f"{voltage}_Humid_fitting_three_film_HONOvar"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    dz = np.asarray(d["dz_cells"])
    z_mm = np.asarray(d["z_centers"]) * 1e3
    L = float(d["L"])

    mask_b = z_mm > 0.1
    Lb = float(dz[mask_b].sum())

    print(f"\n{'='*100}")
    print(f"  {voltage} — bulk-only avg (z>0.1mm) at t=480s")
    print(f"{'='*100}")
    print(f"{'Species':>12s} {'atol':>9s} {'bulk avg (M)':>14s} {'surf (M)':>14s} "
          f"{'bulk/atol':>10s} {'verdict':>20s}")
    print("-" * 100)

    si = int(np.argmin(np.abs(snap_t - 480)))
    rows = []
    for i, sp in enumerate(AQUEOUS_SPECIES):
        atol = ATOL_TIGHT if sp in TIGHT_SPECIES else ATOL_DEFAULT
        bulk_avg = np.dot(snap_y[si, mask_b, i], dz[mask_b]) / Lb
        surf = snap_y[si, 0, i]
        ratio = abs(bulk_avg) / atol if atol > 0 else 0
        if ratio < 1:
            verdict = "★ BELOW ATOL"
        elif ratio < 100:
            verdict = "marginal"
        else:
            verdict = "OK"
        rows.append((sp, atol, bulk_avg, surf, ratio, verdict))
        print(f"{sp:>12s} {atol:>9.0e} {bulk_avg:>+14.3e} {surf:>+14.3e} "
              f"{ratio:>10.2e} {verdict:>20s}")

    return rows


def main():
    all_rows = {}
    for V in ("2.6kV", "3.2kV", "3.6kV"):
        all_rows[V] = dump_voltage(V)

    # Summary: which species are below atol in any voltage
    print("\n" + "=" * 80)
    print("SUMMARY: species below atol in bulk-only avg at t=480s")
    print("=" * 80)
    print(f"{'Species':>12s} {'2.6kV':>14s} {'3.2kV':>14s} {'3.6kV':>14s} "
          f"{'atol':>9s}")
    for i, sp in enumerate(AQUEOUS_SPECIES):
        atol = ATOL_TIGHT if sp in TIGHT_SPECIES else ATOL_DEFAULT
        vals = [all_rows[V][i][2] for V in ("2.6kV", "3.2kV", "3.6kV")]
        below_atol_voltages = sum(1 for v in vals if abs(v) < atol)
        if below_atol_voltages > 0:
            mark = "★ " if below_atol_voltages > 0 else "  "
            print(f"{mark}{sp:>10s} {vals[0]:>+14.3e} {vals[1]:>+14.3e} "
                  f"{vals[2]:>+14.3e} {atol:>9.0e}  "
                  f"({below_atol_voltages}/3 voltages below atol)")


if __name__ == "__main__":
    main()
