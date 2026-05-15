#!/usr/bin/env python3
"""3.6 kV verification with all settings combined:
  - stretch_ratio = 1.02 (188 cells, dz 5~199µm — much smoother than current 1.12 → 1028µm)
  - atol = 1e-30 (vs default 1e-15)
  - rtol = 1e-6 (default)
  - max_step = 1.0s (default)
  - Seedmin: O2/N2/OH = trace, only H+/OH- explicit (pH=7)
  - Perturbation: ALREADY removed permanently in pde_solver.py:858 (2026-05-07 edit)

Output:
  - Figures/DIW results/3.6kV_Humid_fitting_three_film_HONOvar_full/fig5_spatial.{png,pdf}
  - cache for follow-up analysis
  - Console: wall time, OH negative cell count, O3 4mm peak value, comparison with previous

Expected:
  - OH cliff (cell-specific negative spike) eliminated by smoother grid
  - O3 4mm peak: still present (Phase G dz sweep 8% variation suggests grid-independent)
  - wall time: estimate to scale to 3 voltages
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
from config_1d import AQUEOUS_SPECIES, ACID_BASE_PAIRS

print = functools.partial(print, flush=True)
TRACE = cfg.DEFAULTS.trace_concentration


# ──────────────────────────────────────────────────────────────────────
# Patch 1: Initial condition — uniform trace + H+, OH- only
# ──────────────────────────────────────────────────────────────────────
def build_ic_minimal(self, initial_pH=7.0):
    """Uniform trace IC (seedmin): O2/N2/OH all trace, only H+/OH- explicit.
    Perturbation already removed in pde_solver.py source code.
    """
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
print("[patch 1] IC = uniform trace, only H+/OH- explicit (seedmin)")
print(f"          Perturbation: REMOVED (permanent in pde_solver.py:858)")


# ──────────────────────────────────────────────────────────────────────
# Patch 2: ODE_CONFIG — atol = 1e-30
# ──────────────────────────────────────────────────────────────────────
cfg.ODE_CONFIG.atol = 1e-20
cfg.ODE_CONFIG.rtol = 1e-6
cfg.ODE_CONFIG.max_step = 1.0
print(f"[patch 2] ODE_CONFIG.atol = {cfg.ODE_CONFIG.atol:.0e}")
print(f"          ODE_CONFIG.rtol = {cfg.ODE_CONFIG.rtol:.0e}")
print(f"          ODE_CONFIG.max_step = {cfg.ODE_CONFIG.max_step:.1f}s")


# ──────────────────────────────────────────────────────────────────────
# Patch 3: Grid — stretch 1.12 → 1.02
# ──────────────────────────────────────────────────────────────────────
def main():
    import gen_all_figures as gaf
    gaf.STRETCH = 1.02  # was 1.12
    gaf.DZ_MIN = 5e-6   # keep 5 µm
    print(f"[patch 3] STRETCH = {gaf.STRETCH:.3f} (was 1.12)")
    print(f"          DZ_MIN = {gaf.DZ_MIN*1e6:.1f} µm (kept)")
    print()

    V = "3.6kV"
    print(f"{'='*72}")
    print(f"  Running 3.6 kV with full combo (stretch=1.02, atol=1e-30, seedmin)")
    print(f"{'='*72}")

    sys.argv = [
        "gen_all_figures.py",
        "--voltage", V,
        "--condition", "Humid_fitting",
        "--label-suffix", "HONOvar_FULL_atol1e20",
        "--fig", "5",
    ]
    t_start = time_mod.time()
    try:
        gaf.main()
    except SystemExit:
        pass
    wall_total = time_mod.time() - t_start
    print(f"\n  Total wall time: {wall_total:.0f}s = {wall_total/60:.1f} min")

    # ──────────────────────────────────────────────────────
    # Detailed analysis
    # ──────────────────────────────────────────────────────
    fp = (_root / "Figures" / "DIW results"
          / f"{V}_Humid_fitting_three_film_HONOvar_FULL_atol1e20"
          / "cache" / "three_film_abspecies_dg0.0100.npz")
    d = dict(np.load(fp, allow_pickle=True))
    snap_t = np.asarray(d["snap_t"])
    snap_y = np.asarray(d["snap_y"])
    z = np.asarray(d["z_centers"]) * 1e3
    dz = np.asarray(d["dz_cells"])
    L_total = float(d["L"])

    iO3 = AQUEOUS_SPECIES.index("O3")
    iOH = AQUEOUS_SPECIES.index("OH")
    iHONO_t = AQUEOUS_SPECIES.index("HONO_total")
    iHp = AQUEOUS_SPECIES.index("H+")

    N_z = snap_y.shape[1]
    print(f"\n{'='*72}")
    print(f"  RESULTS — {V}, full combo")
    print(f"{'='*72}")
    print(f"  Grid: N_z = {N_z}, L = {L_total*1e3:.1f} mm")
    print(f"  dz range: {dz.min()*1e6:.1f} ~ {dz.max()*1e6:.1f} µm")
    print()

    # 1) OH negative cell count at multiple times
    print("  --- OH spatial — negative cell check ---")
    for tt in (60, 120, 240, 360, 480):
        si = int(np.argmin(np.abs(snap_t - tt)))
        oh = snap_y[si, :, iOH]
        neg_idx = np.where(oh < 0)[0]
        print(f"    t={tt:>4}s — # negative cells: {len(neg_idx)}", end="")
        if len(neg_idx) > 0:
            for j in neg_idx[:5]:  # show first 5
                print(f"\n      cell {j} (z={z[j]:.3f}mm): OH = {oh[j]:.3e}", end="")
            if len(neg_idx) > 5:
                print(f"\n      ... ({len(neg_idx)-5} more)", end="")
        print()

    # 2) O3 4mm peak value
    print("\n  --- O3 4mm peak (problem 3) ---")
    j_4mm = int(np.argmin(np.abs(z - 4.0)))
    print(f"  cell at z≈4mm: cell {j_4mm}, z={z[j_4mm]:.3f}mm")
    for tt in (60, 240, 480):
        si = int(np.argmin(np.abs(snap_t - tt)))
        o3 = snap_y[si, j_4mm, iO3]
        print(f"    t={tt:>4}s: [O3]@4mm = {o3:.3e} M")

    # Find true peak position around 4mm
    si_480 = int(np.argmin(np.abs(snap_t - 480)))
    o3_prof_480 = snap_y[si_480, :, iO3]
    # Find max in deep zone (z > 1.5mm)
    deep_mask = z > 1.5
    j_peak_deep = np.argmax(o3_prof_480[deep_mask]) + np.where(deep_mask)[0][0]
    print(f"    Deep peak position at t=480s: cell {j_peak_deep} "
          f"(z={z[j_peak_deep]:.3f}mm), [O3]={o3_prof_480[j_peak_deep]:.3e} M")

    # 3) Wall structure — find where O3 hits atol-band at t=480s
    si_480 = int(np.argmin(np.abs(snap_t - 480)))
    o3_480 = snap_y[si_480, :, iO3]
    # Find first cell where |O3| drops below 1e-15
    surf_o3 = o3_480[0]
    print(f"\n  --- Wall structure at t=480s ---")
    print(f"  Surface [O3]: {surf_o3:.3e} M")
    for j in range(N_z):
        if abs(o3_480[j]) < 1e-15:
            print(f"  First cell |O3|<1e-15: cell {j} (z={z[j]:.3f}mm), O3={o3_480[j]:.3e}")
            break

    # 4) Wall time projection for 3 voltages
    print(f"\n  --- Time projection ---")
    print(f"  3.6 kV wall: {wall_total:.0f}s = {wall_total/60:.1f} min")
    print(f"  3 voltages estimate: {3*wall_total:.0f}s = {3*wall_total/60:.1f} min "
          f"= {3*wall_total/3600:.2f} hours")


if __name__ == "__main__":
    main()
