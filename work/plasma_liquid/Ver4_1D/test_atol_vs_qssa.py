#!/usr/bin/env python3
"""
Compare three approaches for handling ultra-fast radical intermediates:

  Case 1: Baseline (scalar atol=1e-15)
  Case 2: Species-specific atol (trace species get atol=1e-20)
  Case 3: QSSA for O3-, HO3, O- + instant conversion for N2O5(aq)

Run:
    .venv/bin/python Ver4_1D/test_atol_vs_qssa.py
"""

import sys
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_1d import PHYSICAL, N2O4_EQ
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D, compute_k_mt
from scipy.integrate import solve_ivp

DEFAULT_CSV = (
    Path(__file__).parent.parent
    / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv'
)


def load_gas_data(csv_path: Path):
    import math
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float) * 2.0
    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = np.maximum(df[col].values.astype(float), 0.0)
        else:
            gas_conc[col] = np.zeros(len(df))
    if np.all(gas_conc['N2O4'] == 0):
        no2 = gas_conc['NO2']
        T = 298.15
        Kp = math.exp(math.log(N2O4_EQ.KP_298) +
                       (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1/N2O4_EQ.REF_TEMP - 1/T))
        factor = PHYSICAL.KB_T_OVER_P * T
        gas_conc['N2O4'] = Kp * factor * (no2 ** 2)
    return times, gas_conc


def make_solver(times, gas_conc):
    """Create a standard solver instance."""
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem, dz_min=5e-6, stretch_ratio=1.12,
        mass_transfer_eta=1.0, saline_mode=False,
        bc_type='film_alpha', alpha_b=0.03,
    )
    solver.set_gas_data(times=times, gas_conc_molecules=gas_conc,
                        hono_gas=0, hono2_gas=0, h2o2_gas=0)
    return solver


# =========================================================================
# Case 2: Species-specific atol
# =========================================================================
class SpeciesAtolSolver:
    """Wrapper that builds a per-DOF atol array."""

    # Species that need tighter atol (bulk conc < 1e-13 M)
    TIGHT_SPECIES = ['O3-', 'HO3', 'O-', 'HO2-', 'N2O5', 'NO', 'H',
                     'N2O', 'N2O3', 'H2', 'O']

    def __init__(self, solver, tight_atol=1e-20, default_atol=1e-15):
        self.solver = solver
        self.tight_atol = tight_atol
        self.default_atol = default_atol
        self._build_atol_array()

    def _build_atol_array(self):
        N_s = self.solver.N_s
        N_z = self.solver.N_z
        atol_per_species = np.full(N_s, self.default_atol)
        for sp in self.TIGHT_SPECIES:
            if sp in self.solver.species_idx:
                idx = self.solver.species_idx[sp]
                atol_per_species[idx] = self.tight_atol
        # Tile across all grid cells
        self.atol_array = np.tile(atol_per_species, N_z)

    def solve(self, **kwargs):
        # Monkey-patch the solve_ivp call to use our atol array
        import pde_solver as mod
        orig_solve_ivp = mod.solve_ivp

        atol_arr = self.atol_array

        def patched_solve_ivp(fun, t_span, y0, **kw):
            kw['atol'] = atol_arr
            return orig_solve_ivp(fun, t_span, y0, **kw)

        mod.solve_ivp = patched_solve_ivp
        try:
            result = self.solver.solve(**kwargs)
        finally:
            mod.solve_ivp = orig_solve_ivp
        return result


# =========================================================================
# Case 3: QSSA for O3-, HO3, O- + N2O5 instant conversion
# =========================================================================
class QSSASolver:
    """Proper QSSA: compute analytical steady-state concentrations for
    fast intermediates, inject them into y before rate calculation.

    QSSA species (lifetime << BDF dt):
      HO3  (τ ≈ 9 µs):  C_ss = P_HO3 / D_HO3
      O3-  (τ ≈ 30 ns):  C_ss = P_O3m / D_O3m
      O-   (τ ≈ ns):     C_ss = P_Om  / D_Om

    N2O5(aq) (τ ≈ 0.4 ps): instant conversion at surface.
    """

    def __init__(self, solver):
        self.solver = solver
        sidx = solver.species_idx
        self.N_s = solver.N_s
        self.N_z = solver.N_z
        self.trace = 1e-30

        # Species indices
        self.i_HO3  = sidx.get('HO3', -1)
        self.i_O3m  = sidx.get('O3-', -1)
        self.i_Om   = sidx.get('O-', -1)
        self.i_N2O5 = sidx.get('N2O5', -1)
        self.i_O3   = sidx.get('O3', -1)
        self.i_OH   = sidx.get('OH', -1)
        self.i_HO2t = sidx.get('HO2_total', -1)
        self.i_O2m  = -1  # O2- comes from HO2_total speciation
        self.i_O2   = sidx.get('O2', -1)
        self.i_Hp   = sidx.get('H+', -1)
        self.i_OHm  = sidx.get('OH-', -1)
        self.i_H2O2t = sidx.get('H2O2_total', -1)
        self.i_H    = sidx.get('H', -1)
        self.i_NO3m = sidx.get('NO3-', -1)

        # pKa for speciation
        self.Ka_HO2 = 10**(-4.88)   # HO2 ↔ H+ + O2-
        self.Ka_H2O2 = 10**(-11.62) # H2O2 ↔ H+ + HO2-

        # Rate constants for QSSA expressions
        # HO3 production
        self.k28 = 5.0e8    # O3 + HO2 → HO3
        # HO3 destruction
        self.k9f  = 1.4e5   # HO3 → O3- + H+
        self.k40  = 5.0e9   # OH + HO3 → H2O2 + O2
        self.k53  = 5.0e9   # HO2 + HO3 → H2O2 + O2
        self.k55  = 1.1e5   # HO3 → O2 + OH
        self.k57  = 1.0e10  # HO3 + O2- → OH- + 2O2

        # O3- production (besides from HO3 via R9f)
        self.k25  = 1.6e9   # O3 + O2- → O3- + O2
        # O3- destruction
        self.k9b  = 5.0e10  # O3- + H+ → HO3 (reverse of R9)
        self.k33  = 25.0    # O3- → OH + O2 + OH-
        self.k34  = 6.0e9   # O3- + OH → O2- + HO2
        self.k35  = 2.5e9   # O3- + OH → O3 + OH-
        self.k38  = 9.0e10  # O3- + H+ → O2 + OH
        self.k50  = 6.0e9   # HO2 + O3- → OH- + 2O2

        # O- production
        self.k21f = 1.3e10  # OH + OH- → O-
        # O- destruction
        self.k21b = 1.7e6   # O- → OH + OH- (reverse)
        self.k10f = 3.6e9   # O- + O2 → O3-
        self.k24  = 5.0e9   # O3 + O- → O2- + O2
        self.k44  = 2.6e10  # OH + O- → HO2-
        self.k48  = 6.0e9   # HO2 + O- → O2 + OH-
        self.k60  = 5.0e7   # O- + H2O2 → O2-
        self.k64  = 6.0e8   # O- + O2- → 2OH- + O2

        # N2O5: NOT suitable for QSSA (MT flux is cell-specific, can't be
        # included in volumetric steady-state). Let BDF handle it normally.

    def _compute_qssa_ss(self, y_2d):
        """Compute steady-state concentrations for QSSA species per cell."""
        tr = self.trace
        N_z = self.N_z

        for j in range(N_z):
            yc = y_2d[j]
            H   = max(yc[self.i_Hp], tr)
            O3  = max(yc[self.i_O3], tr)
            OH  = max(yc[self.i_OH], tr)
            OHm = max(yc[self.i_OHm], tr)
            O2  = max(yc[self.i_O2], tr)

            # Speciate HO2_total → HO2, O2-
            HO2_total = max(yc[self.i_HO2t], 0.0) if self.i_HO2t >= 0 else 0.0
            den_ho2 = H + self.Ka_HO2
            HO2 = HO2_total * H / den_ho2 if den_ho2 > tr else tr
            O2m = HO2_total * self.Ka_HO2 / den_ho2 if den_ho2 > tr else tr
            HO2 = max(HO2, tr)
            O2m = max(O2m, tr)

            # Speciate H2O2_total → H2O2, HO2-
            H2O2_total = max(yc[self.i_H2O2t], 0.0) if self.i_H2O2t >= 0 else 0.0
            den_h2o2 = H + self.Ka_H2O2
            H2O2 = H2O2_total * H / den_h2o2 if den_h2o2 > tr else tr
            H2O2 = max(H2O2, tr)

            # --- HO3 steady state ---
            P_HO3 = self.k28 * O3 * HO2
            D_HO3 = self.k9f + self.k40 * OH + self.k53 * HO2 + self.k55 + self.k57 * O2m
            HO3_ss = P_HO3 / max(D_HO3, 1e-30)

            # --- O3- steady state ---
            P_O3m = self.k9f * HO3_ss + self.k25 * O3 * O2m
            D_O3m = (self.k9b * H + self.k38 * H + self.k33
                     + self.k34 * OH + self.k35 * OH + self.k50 * HO2)
            O3m_ss = P_O3m / max(D_O3m, 1e-30)

            # --- O- steady state ---
            P_Om = self.k21f * OH * OHm
            D_Om = (self.k21b + self.k10f * O2 + self.k24 * O3
                    + self.k44 * OH + self.k48 * HO2 + self.k60 * H2O2
                    + self.k64 * O2m)
            Om_ss = P_Om / max(D_Om, 1e-30)

            # Inject QSSA concentrations
            if self.i_HO3 >= 0:
                y_2d[j, self.i_HO3] = max(HO3_ss, tr)
            if self.i_O3m >= 0:
                y_2d[j, self.i_O3m] = max(O3m_ss, tr)
            if self.i_Om >= 0:
                y_2d[j, self.i_Om] = max(Om_ss, tr)

    def _patched_rhs(self, original_rhs):
        solver = self.solver
        N_s = self.N_s
        N_z = self.N_z
        i_HO3 = self.i_HO3
        i_O3m = self.i_O3m
        i_Om  = self.i_Om
        i_N2O5 = self.i_N2O5
        compute_qssa = self._compute_qssa_ss

        def rhs(t, y):
            # 1. Inject QSSA steady-state concentrations into y
            y_mod = y.copy()
            y_2d = y_mod.reshape(N_z, N_s)
            compute_qssa(y_2d)

            # 2. Call original rhs with QSSA-corrected concentrations
            #    R98 (N2O5+H2O→products) uses N2O5_ss, producing correct NO3-/H+.
            #    MT flux enters N2O5(aq) normally via _interface_species.
            dydt = original_rhs(t, y_mod)
            dydt_2d = dydt.reshape(N_z, N_s)

            # 3. QSSA species: relaxation toward C_ss
            tau_relax = 1e-6
            y_2d_orig = y.reshape(N_z, N_s)
            for idx in [i_HO3, i_O3m, i_Om]:
                if idx >= 0:
                    c_ss = y_2d[:, idx]      # QSSA-computed value
                    c_cur = y_2d_orig[:, idx] # BDF's current value
                    dydt_2d[:, idx] = (c_ss - c_cur) / tau_relax

            return dydt.ravel()

        return rhs

    def solve(self, **kwargs):
        original_rhs = self.solver.rhs
        self.solver.rhs = self._patched_rhs(original_rhs)
        try:
            result = self.solver.solve(**kwargs)
        finally:
            self.solver.rhs = original_rhs
        return result


# =========================================================================
# Main
# =========================================================================
def print_result(label, result, wall):
    avg = result['spatial_avg']
    sfc = result.get('surface', {})
    print(f"\n  [{label}]")
    print(f"  wall={wall:.1f}s ({wall/60:.1f}min), nfev={result.get('nfev','?')}, "
          f"njev={result.get('njev','?')}")
    print(f"  pH_avg    = {result['pH_avg']:.3f}")
    print(f"  pH_sfc    = {result.get('pH_surface', 0):.3f}")
    print(f"  NO3⁻      = {avg.get('NO3-', 0)*1e6:.1f} µM")
    print(f"  NO2⁻      = {avg.get('NO2-', 0)*1e6:.4f} µM")
    print(f"  H2O2      = {avg.get('H2O2', 0)*1e6:.4f} µM")
    print(f"  O3        = {avg.get('O3', 0)*1e9:.1f} nM")
    print(f"  OH        = {avg.get('OH', 0)*1e12:.2f} pM")
    print(f"  HO2       = {avg.get('HO2', 0)*1e12:.1f} pM")
    print(f"  O3_sfc    = {sfc.get('O3', 0)*1e6:.3f} µM")
    print(f"  OH_sfc    = {sfc.get('OH', 0)*1e9:.3f} nM")
    # QSSA species
    print(f"  --- QSSA target species ---")
    print(f"  O3⁻       = {avg.get('O3-', 0):.3e} M")
    print(f"  HO3       = {avg.get('HO3', 0):.3e} M")
    print(f"  O⁻        = {avg.get('O-', 0):.3e} M")
    print(f"  N2O5(aq)  = {avg.get('N2O5', 0):.3e} M")


def main():
    times, gas_conc = load_gas_data(DEFAULT_CSV)
    t_end = float(times[-1])
    solve_kw = dict(t_span=(0, t_end), t_eval=np.array([0, t_end]),
                    verbose=True, dt_poisson=None)

    print("=" * 70)
    print("atol vs QSSA COMPARISON — DIW, Film+α_b=0.03")
    print("=" * 70)

    # Case 1: Baseline (scalar atol=1e-15)
    print("\n" + "=" * 70)
    print("  Case 1: Baseline (atol=1e-15)")
    print("=" * 70)
    s1 = make_solver(times, gas_conc)
    t0 = time.time()
    r1 = s1.solve(**solve_kw)
    w1 = time.time() - t0
    print_result("Baseline atol=1e-15", r1, w1)

    # Case 2: Species-specific atol
    print("\n" + "=" * 70)
    print("  Case 2: Species-specific atol (trace→1e-20, default→1e-15)")
    print("=" * 70)
    s2 = make_solver(times, gas_conc)
    wrapper2 = SpeciesAtolSolver(s2, tight_atol=1e-20, default_atol=1e-15)
    t0 = time.time()
    r2 = wrapper2.solve(**solve_kw)
    w2 = time.time() - t0
    print_result("Species-specific atol", r2, w2)

    # Case 3: QSSA + N2O5 instant conversion
    print("\n" + "=" * 70)
    print("  Case 3: QSSA (O3-, HO3, O-) + N2O5 instant conversion")
    print("=" * 70)
    s3 = make_solver(times, gas_conc)
    wrapper3 = QSSASolver(s3)
    t0 = time.time()
    r3 = wrapper3.solve(**solve_kw)
    w3 = time.time() - t0
    print_result("QSSA + N2O5 instant", r3, w3)

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    a1, a2, a3 = r1['spatial_avg'], r2['spatial_avg'], r3['spatial_avg']
    print(f"  {'Metric':>12s}  {'Baseline':>10s}  {'Spec atol':>10s}  {'QSSA':>10s}  {'Experiment':>10s}")
    print("  " + "─" * 60)
    rows = [
        ('pH_avg',     r1['pH_avg'],    r2['pH_avg'],    r3['pH_avg'],    3.61),
        ('NO3⁻ µM',   a1.get('NO3-',0)*1e6, a2.get('NO3-',0)*1e6, a3.get('NO3-',0)*1e6, 63.0),
        ('H2O2 µM',   a1.get('H2O2',0)*1e6, a2.get('H2O2',0)*1e6, a3.get('H2O2',0)*1e6, 11.0),
        ('O3 nM',     a1.get('O3',0)*1e9, a2.get('O3',0)*1e9, a3.get('O3',0)*1e9, None),
        ('OH pM',     a1.get('OH',0)*1e12, a2.get('OH',0)*1e12, a3.get('OH',0)*1e12, None),
        ('HO2 pM',    a1.get('HO2',0)*1e12, a2.get('HO2',0)*1e12, a3.get('HO2',0)*1e12, None),
        ('Time (s)',   w1, w2, w3, None),
    ]
    for name, v1, v2, v3, exp in rows:
        exp_str = f"{exp:.2f}" if exp is not None else "–"
        print(f"  {name:>12s}  {v1:10.4f}  {v2:10.4f}  {v3:10.4f}  {exp_str:>10s}")

    print(f"\n  Speed: baseline={w1:.0f}s, spec_atol={w2:.0f}s ({w2/w1:.2f}x), "
          f"QSSA={w3:.0f}s ({w3/w1:.2f}x)")
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
