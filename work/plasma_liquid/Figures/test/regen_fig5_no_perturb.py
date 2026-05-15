#!/usr/bin/env python3
"""Verification: seedmin + remove initial perturbation.

Tests user-confirmed hypothesis: cell-37 (1min) / cell-47 (8min) negative spikes
are caused by deterministic random initial perturbation
(`y0 *= 1 + 1e-6 × np.random.standard_normal(seed=42)`).

If correct, removing the perturbation should eliminate cell-specific anomalies.

3.6 kV only (cheapest verification).
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
from chemistry_1d import AqueousChemistry1D  # noqa: E402
from config_1d import AQUEOUS_SPECIES  # noqa: E402

print = functools.partial(print, flush=True)
TRACE = cfg.DEFAULTS.trace_concentration


def build_ic_no_perturb(self, initial_pH=7.0):
    """Same as orig build_initial_condition but:
    1. O2/N2/OH set to trace (seedmin)
    2. No multiplicative random perturbation
    """
    y0 = np.full(self.N_total, TRACE)

    H_conc = 10.0 ** (-initial_pH)
    OH_conc = 1e-14 / H_conc

    for j in range(self.N_z):
        off = j * self.N_s
        # H+ and OH- ONLY (seedmin: skip O2, N2, OH)
        if 'H+' in self.species_idx:
            y0[off + self.species_idx['H+']] = H_conc
        if 'OH-' in self.species_idx:
            y0[off + self.species_idx['OH-']] = OH_conc
        # NO O2 = 2.5e-4
        # NO N2 = 5e-4
        # NO OH = 1e-12
        if self.saline_mode and 'Cl-' in self.species_idx:
            y0[off + self.species_idx['Cl-']] = 0.154

    # NO perturbation (skip rng block)
    np.clip(y0, TRACE, None, out=y0)
    return y0


_ps.PDESolver1D.build_initial_condition = build_ic_no_perturb
print("[patch] build_initial_condition: seedmin (O2/N2/OH=trace) + NO perturbation")


def main():
    import gen_all_figures as gaf

    V = "3.6kV"
    print(f"\n=== {V} seedmin + no-perturbation ===")
    sys.argv = [
        "gen_all_figures.py",
        "--voltage", V,
        "--condition", "Humid_fitting",
        "--label-suffix", "HONOvar_seedmin_noperturb",
        "--fig", "5",
    ]
    t0 = time_mod.time()
    try:
        gaf.main()
    except SystemExit:
        pass
    wall = time_mod.time() - t0
    print(f"  wall = {wall:.0f}s")

    # Inspect OH spatial profile at t=60, 480
    fp = (_root / "Figures" / "DIW results"
          / f"{V}_Humid_fitting_three_film_HONOvar_seedmin_noperturb"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    z = np.asarray(d["z_centers"]) * 1e3
    iOH = AQUEOUS_SPECIES.index("OH")

    print("\n=== OH spatial @ t=60s, 480s — check cell 37, 47 ===")
    for tt in (60, 480):
        si = int(np.argmin(np.abs(snap_t - tt)))
        oh = snap_y[si, :, iOH]
        # Find any negative cells
        neg_idx = np.where(oh < 0)[0]
        print(f"\n  t={tt}s — # negative cells: {len(neg_idx)}")
        if len(neg_idx) > 0:
            for j in neg_idx:
                print(f"    cell {j} (z={z[j]:.3f}mm): OH = {oh[j]:.3e}")
        else:
            print(f"    NO NEGATIVE CELLS — perturbation hypothesis confirmed")
        # Specifically cell 37, 47
        for j in (37, 47):
            print(f"    cell {j:>2} (z={z[j]:.3f}mm): OH = {oh[j]:+.3e}")


if __name__ == "__main__":
    main()
