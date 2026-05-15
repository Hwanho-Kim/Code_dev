#!/usr/bin/env python3
"""Regenerate fig5_spatial with atol=1e-30 for 3 voltages.

Patches ODE_CONFIG.atol globally then re-runs gen_all_figures with a unique
label suffix so existing HONOvar caches are not overwritten.

Output: Figures/DIW results/{V}_Humid_fitting_three_film_HONOvar_atol1e30/fig5_spatial.{png,pdf}
"""
from __future__ import annotations

import sys
import time as time_mod
from pathlib import Path

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "Ver4_1D"))
sys.path.insert(0, str(_root / "Figures"))

# Patch atol BEFORE importing gen_all_figures
import config_1d as cfg

cfg.ODE_CONFIG.atol = 1e-30
cfg.ODE_CONFIG.rtol = 1e-6
cfg.ODE_CONFIG.max_step = 1.0
print(f"[patch] ODE_CONFIG.atol = {cfg.ODE_CONFIG.atol:.0e}")
print(f"[patch] ODE_CONFIG.rtol = {cfg.ODE_CONFIG.rtol:.0e}")
print(f"[patch] ODE_CONFIG.max_step = {cfg.ODE_CONFIG.max_step:.1f}")


def main():
    import gen_all_figures as gaf

    # Override main() argv via sys.argv
    for V in ("2.6kV", "3.2kV", "3.6kV"):
        print(f"\n{'='*70}")
        print(f"  Re-running with atol=1e-30 for {V}")
        print(f"{'='*70}")
        sys.argv = [
            "gen_all_figures.py",
            "--voltage", V,
            "--condition", "Humid_fitting",
            "--label-suffix", "HONOvar_atol1e30",
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
