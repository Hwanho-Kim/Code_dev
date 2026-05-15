#!/usr/bin/env python3
"""Regen fig5 with minimal initial seeds: O2, N2, OH → trace.
Keep H+ (pH=7) and OH- (1e-7).

Tests user hypothesis: OH plateau at 1e-13 in deep cells = initial seed.
If trace OH initial → deep OH falls to true zero (atol-noise).

Output: Figures/DIW results/{V}_Humid_fitting_three_film_HONOvar_seedmin/fig5_spatial.{png,pdf}
"""
from __future__ import annotations

import sys
import time as time_mod
from pathlib import Path

import numpy as np

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

import config_1d as cfg
import pde_solver as _ps

# Monkey-patch build_initial_condition to remove O2, N2, OH seeds
_orig_build_ic = _ps.PDESolver1D.build_initial_condition
TRACE = cfg.DEFAULTS.trace_concentration


def patched_build_ic(self, initial_pH=7.0):
    y0 = _orig_build_ic(self, initial_pH=initial_pH)
    # Reset O2, N2, OH to trace in all cells (KEEP H+ and OH-)
    for sp in ('O2', 'N2', 'OH'):
        if sp in self.species_idx:
            idx = self.species_idx[sp]
            for j in range(self.N_z):
                y0[j * self.N_s + idx] = TRACE
    return y0


_ps.PDESolver1D.build_initial_condition = patched_build_ic
print(f"[patch] Initial seeds set to trace ({TRACE:.0e}): O2, N2, OH")
print(f"[patch] Kept: H+ (1e-pH), OH- (Kw/H+)")


def main():
    import gen_all_figures as gaf

    for V in ("2.6kV", "3.2kV", "3.6kV"):
        print(f"\n{'='*70}")
        print(f"  Re-running with seedmin (O2,N2,OH=trace) for {V}")
        print(f"{'='*70}")
        sys.argv = [
            "gen_all_figures.py",
            "--voltage", V,
            "--condition", "Humid_fitting",
            "--label-suffix", "HONOvar_seedmin",
            "--fig", "5",
        ]
        t0 = time_mod.time()
        try:
            gaf.main()
        except SystemExit:
            pass
        wall = time_mod.time() - t0
        print(f"  [{V}] wall = {wall:.0f}s")


if __name__ == "__main__":
    main()
