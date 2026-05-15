#!/usr/bin/env python3
"""3.6 kV with smaller dz_min + seedmin + no-perturbation (default after edit).

Test if cell-specific anomalies (cell 37, 47 in dz=5µm grid) move/disappear
with finer grid. This validates whether anomalies are SG-flux artifacts
tied to specific cell positions.
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

import config_1d as cfg
import pde_solver as _ps
from chemistry_1d import AqueousChemistry1D
from config_1d import AQUEOUS_SPECIES

print = functools.partial(print, flush=True)
TRACE = cfg.DEFAULTS.trace_concentration


def build_ic_minimal(self, initial_pH=7.0):
    """Uniform trace IC: only H+, OH- explicitly set. No perturbation."""
    y0 = np.full(self.N_total, TRACE)
    H_conc = 10.0 ** (-initial_pH)
    OH_conc = 1e-14 / H_conc
    for j in range(self.N_z):
        off = j * self.N_s
        if 'H+' in self.species_idx:
            y0[off + self.species_idx['H+']] = H_conc
        if 'OH-' in self.species_idx:
            y0[off + self.species_idx['OH-']] = OH_conc
        if self.saline_mode and 'Cl-' in self.species_idx:
            y0[off + self.species_idx['Cl-']] = 0.154
    np.clip(y0, TRACE, None, out=y0)
    return y0


_ps.PDESolver1D.build_initial_condition = build_ic_minimal
print("[patch] Uniform trace IC (H+, OH-만 explicit), NO perturbation")


def main():
    import gen_all_figures as gaf

    # Patch DZ_MIN globally
    DZ_NEW = 1e-6  # 1 µm (was 5 µm default)
    gaf.DZ_MIN = DZ_NEW
    print(f"[patch] DZ_MIN = {DZ_NEW*1e6:.1f} µm (was 5 µm)")

    V = "3.6kV"
    print(f"\n=== {V} smalldz (dz_min={DZ_NEW*1e6:.0f}µm) + seedmin + no-perturb ===")
    sys.argv = [
        "gen_all_figures.py",
        "--voltage", V,
        "--condition", "Humid_fitting",
        "--label-suffix", f"HONOvar_seedmin_dz1um",
        "--fig", "5",
    ]
    t0 = time_mod.time()
    try:
        gaf.main()
    except SystemExit:
        pass
    wall = time_mod.time() - t0
    print(f"  wall = {wall:.0f}s")

    # Inspect OH spatial profile at t=60, 480 — count negative cells
    fp = (_root / "Figures" / "DIW results"
          / f"{V}_Humid_fitting_three_film_HONOvar_seedmin_dz1um"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    z = np.asarray(d["z_centers"]) * 1e3
    iOH = AQUEOUS_SPECIES.index("OH")
    N_z = snap_y.shape[1]

    print(f"\n=== OH anomaly check — N_z = {N_z}, z range {z[0]:.3f}–{z[-1]:.3f}mm ===")
    for tt in (60, 240, 480):
        si = int(np.argmin(np.abs(snap_t - tt)))
        oh = snap_y[si, :, iOH]
        neg_idx = np.where(oh < 0)[0]
        print(f"\n  t={tt}s — # negative cells: {len(neg_idx)}")
        for j in neg_idx:
            print(f"    cell {j} (z={z[j]:.3f}mm): OH = {oh[j]:.3e}")


if __name__ == "__main__":
    main()
