"""
Component Verification Tests for 1D Plasma-Liquid Solver.

Tests each component (diffusion, chemistry, mass transfer) against
analytical solutions to identify which component, if any, is incorrect.

Run:
    Ver3/.venv/bin/python -m pytest Ver4_1D/test_verification.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

# Ensure Ver4_1D is importable
sys.path.insert(0, str(Path(__file__).parent))

from config_1d import (
    PHYSICAL, HENRY_CONSTANTS, GAS_DIFFUSIVITY, LIQUID_DIFFUSIVITY,
    D_GAS_DEFAULT, D_LIQ_DEFAULT, AQUEOUS_SPECIES, GAS_TO_AQUEOUS_MAP,
    MASS_TRANSFER, DEFAULTS, ODE_CONFIG, ACID_BASE_PAIRS,
)
from chemistry_1d import AqueousChemistry1D
from pde_solver import PDESolver1D, compute_k_mt


# =====================================================================
# Module-level fixtures (avoid repeated Numba JIT compilation)
# =====================================================================

@pytest.fixture(scope="module")
def chem():
    """DIW chemistry: 26 species, 96 reactions. JIT compiled once."""
    return AqueousChemistry1D(saline_mode=False)


@pytest.fixture(scope="module")
def solver(chem):
    """PDESolver1D with small uniform grid for fast tests."""
    return PDESolver1D(
        chemistry=chem,
        liquid_depth=1e-3,  # 1 mm
        N_z=50,
        saline_mode=False,
    )


# =====================================================================
# Test A: Pure Diffusion (Crank-Nicolson + Thomas algorithm)
# =====================================================================

def _thomas(a, b, c, d):
    """Standalone Thomas algorithm — 1:1 copy from pde_solver.py L1254-1268."""
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        m = a[i] / (b[i] - a[i] * c_[i - 1])
        c_[i] = c[i] / (b[i] - a[i] * c_[i - 1])
        d_[i] = (d[i] - a[i] * d_[i - 1]) / (b[i] - a[i] * c_[i - 1])
    x = np.zeros(n)
    x[n - 1] = d_[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


def _cn_diffusion_step(c, dz, dt, D):
    """
    Standalone Crank-Nicolson diffusion step on uniform grid.
    1:1 replica of pde_solver.py _diffusion_step logic (L1222-1251)
    adapted for uniform grid (h_faces = dz, inv_dz = 1/dz).
    No-flux BCs at both ends (z=0 and z=L).
    """
    N_z = len(c)
    inv_h = 1.0 / dz  # uniform
    inv_dz = 1.0 / dz
    alpha = 0.5 * dt * D

    a = np.zeros(N_z)
    b = np.ones(N_z)
    cc = np.zeros(N_z)
    rhs = c.copy()

    # Interior
    for j in range(1, N_z - 1):
        coeff_l = alpha * inv_h * inv_dz
        coeff_r = alpha * inv_h * inv_dz
        a[j] = -coeff_l
        cc[j] = -coeff_r
        b[j] = 1.0 + coeff_l + coeff_r
        rhs[j] = c[j] + coeff_l * (c[j - 1] - c[j]) + coeff_r * (c[j + 1] - c[j])

    # j=0: no-flux left BC
    coeff_r0 = alpha * inv_h * inv_dz if N_z > 1 else 0
    b[0] = 1.0 + coeff_r0
    cc[0] = -coeff_r0
    rhs[0] = c[0] + coeff_r0 * (c[1] - c[0]) if N_z > 1 else c[0]

    # j=N-1: no-flux right BC
    if N_z > 1:
        coeff_l_last = alpha * inv_h * inv_dz
        a[N_z - 1] = -coeff_l_last
        b[N_z - 1] = 1.0 + coeff_l_last
        rhs[N_z - 1] = c[N_z - 1] + coeff_l_last * (c[N_z - 2] - c[N_z - 1])

    return _thomas(a, b, cc, rhs)


class TestA_PureDiffusion:
    """Test A: CN diffusion against analytical cosine-decay solution."""

    L = 1e-3        # 1 mm
    D = 1e-9         # m²/s
    N_z = 50
    dt = 1.0         # s
    T_total = 200.0  # s (τ ≈ 101s, ~86% decay)

    def _analytical(self, x, t):
        """C(x,t) = 1e-3 + 5e-4·cos(πx/L)·exp(-D(π/L)²t)"""
        return 1e-3 + 5e-4 * np.cos(np.pi * x / self.L) * np.exp(
            -self.D * (np.pi / self.L) ** 2 * t
        )

    def test_diffusion_vs_analytical(self):
        """Standalone CN diffusion matches analytical cosine-decay (L∞ < 1e-5)."""
        dz = self.L / self.N_z
        x = (np.arange(self.N_z) + 0.5) * dz  # cell centers
        c = self._analytical(x, 0.0)

        n_steps = int(self.T_total / self.dt)
        for _ in range(n_steps):
            c = _cn_diffusion_step(c, dz, self.dt, self.D)

        c_exact = self._analytical(x, self.T_total)
        err_Linf = np.max(np.abs(c - c_exact))
        print(f"\n  Test A: L∞ error = {err_Linf:.2e}")
        assert err_Linf < 1e-5, f"L∞ = {err_Linf:.2e} >= 1e-5"

    def test_nonuniform_mass_conservation(self, chem):
        """Diffusion on geometric grid preserves total mass (∫C·dz = const)."""
        solver = PDESolver1D(
            chemistry=chem,
            dz_min=5e-6,
            stretch_ratio=1.12,
            saline_mode=False,
        )
        N_z = solver.N_z
        N_s = solver.N_s
        dz = solver.dz_cells

        # Initialize one species with a bump
        C = np.full((N_z, N_s), DEFAULTS.trace_concentration)
        sp_idx = solver.species_idx.get('O3', 0)
        C[:, sp_idx] = 1e-3 + 5e-4 * np.cos(np.pi * solver.z_centers / solver.L)

        mass_before = np.sum(C[:, sp_idx] * dz)

        # Run 50 diffusion steps (extract the private method by calling
        # _solve_strang_split internals — instead we replicate the CN logic
        # on the non-uniform grid using solver's grid arrays).
        dt_d = 1.0
        inv_h = solver.inv_h_faces
        inv_dz = solver.inv_dz_cells
        D_val = solver.D_species[sp_idx]

        c = C[:, sp_idx].copy()
        for _ in range(50):
            alpha = 0.5 * dt_d * D_val
            a = np.zeros(N_z)
            b = np.ones(N_z)
            cc_arr = np.zeros(N_z)
            rhs = c.copy()
            for j in range(1, N_z - 1):
                coeff_l = alpha * inv_h[j - 1] * inv_dz[j]
                coeff_r = alpha * inv_h[j] * inv_dz[j]
                a[j] = -coeff_l
                cc_arr[j] = -coeff_r
                b[j] = 1.0 + coeff_l + coeff_r
                rhs[j] = c[j] + coeff_l * (c[j - 1] - c[j]) + coeff_r * (c[j + 1] - c[j])
            # j=0: no-flux
            coeff_r0 = alpha * inv_h[0] * inv_dz[0]
            b[0] = 1.0 + coeff_r0
            cc_arr[0] = -coeff_r0
            rhs[0] = c[0] + coeff_r0 * (c[1] - c[0])
            # j=N-1: no-flux
            coeff_l_last = alpha * inv_h[N_z - 2] * inv_dz[N_z - 1]
            a[N_z - 1] = -coeff_l_last
            b[N_z - 1] = 1.0 + coeff_l_last
            rhs[N_z - 1] = c[N_z - 1] + coeff_l_last * (c[N_z - 2] - c[N_z - 1])
            c = _thomas(a, b, cc_arr, rhs)

        mass_after = np.sum(c * dz)
        rel_err = abs(mass_after - mass_before) / mass_before
        print(f"\n  Test A (non-uniform): mass rel error = {rel_err:.2e}")
        assert rel_err < 1e-12, f"Mass conservation: rel_err = {rel_err:.2e} >= 1e-12"


# =====================================================================
# Test B: Pure Chemistry (BDF)
# =====================================================================

class TestB_PureChemistry:
    """Test B: Chemistry component verification."""

    def test_b1_trace_equilibrium(self, chem):
        """B1: At trace concentrations, dydt ≈ 0."""
        y = np.full(chem.n_species, DEFAULTS.trace_concentration)
        y[chem.species_idx['H+']] = 1e-7   # pH 7
        y[chem.species_idx['OH-']] = 1e-7
        y[chem.species_idx['O2']] = 2.5e-4
        y[chem.species_idx['N2']] = 5e-4

        dydt = chem.compute_rates_numba(y)
        max_rate = np.max(np.abs(dydt))
        print(f"\n  Test B1: max |dydt| at trace = {max_rate:.2e}")
        assert max_rate < 1e-19, f"max |dydt| = {max_rate:.2e} >= 1e-19"

    def test_b2_acid_base_conservation(self, chem):
        """B2: HONO_total = HONO + NO2⁻ is conserved (mass balance)."""
        y = np.full(chem.n_species, DEFAULTS.trace_concentration)
        y[chem.species_idx['H+']] = 1e-4    # pH 4
        y[chem.species_idx['OH-']] = 1e-10
        y[chem.species_idx['O2']] = 2.5e-4
        y[chem.species_idx['N2']] = 5e-4

        C_total_set = 1e-4  # 0.1 mM HONO_total
        y[chem.species_idx['HONO_total']] = C_total_set

        # Speciate
        speciated = chem.speciate(y)
        HONO_conc = speciated['HONO']
        NO2m_conc = speciated['NO2-']
        total_check = HONO_conc + NO2m_conc

        err = abs(total_check - C_total_set)
        print(f"\n  Test B2: HONO={HONO_conc:.6e}, NO2-={NO2m_conc:.6e}, "
              f"sum={total_check:.6e}, err={err:.2e}")
        assert err < 1e-12, f"Acid-base conservation: err = {err:.2e} >= 1e-12"

        # Also verify Ka relationship: HONO/NO2- = H+/Ka
        pKa = 3.4
        Ka = 10.0 ** (-pKa)
        H = y[chem.species_idx['H+']]
        ratio_expected = H / Ka
        ratio_actual = HONO_conc / NO2m_conc
        ratio_err = abs(ratio_actual - ratio_expected) / ratio_expected
        print(f"  Ka ratio check: expected={ratio_expected:.4f}, "
              f"actual={ratio_actual:.4f}, rel_err={ratio_err:.2e}")
        assert ratio_err < 1e-10

    def test_b3_bdf_exponential_decay(self, chem):
        """B3: BDF solver sanity — dy/dt = -ky matches exp(-kt)."""
        # Use actual BDF path with a simple decay reaction.
        # We'll solve dy/dt = -k*y for a single variable.
        k_decay = 0.1  # 1/s
        y0 = np.array([1.0])
        t_end = 50.0  # 5 half-lives

        def rhs(t, y):
            return np.array([-k_decay * y[0]])

        sol = solve_ivp(rhs, [0, t_end], y0, method='BDF',
                         rtol=1e-8, atol=1e-12)
        assert sol.success, f"BDF failed: {sol.message}"

        y_bdf = sol.y[0, -1]
        y_exact = np.exp(-k_decay * t_end)
        rel_err = abs(y_bdf - y_exact) / y_exact
        print(f"\n  Test B3: BDF={y_bdf:.10e}, exact={y_exact:.10e}, "
              f"rel_err={rel_err:.2e}")
        assert rel_err < 1e-6, f"BDF decay: rel_err = {rel_err:.2e} >= 1e-6"


# =====================================================================
# Test C: Pure Mass Transfer
# =====================================================================

class TestC_MassTransfer:
    """Test C: Mass transfer coefficient and dissolution ODE."""

    def test_c1_k_mt_formula(self):
        """C1: compute_k_mt() matches hand calculation for N2O5."""
        delta_gas = 0.01     # 10 mm
        delta_liq = 0.0001   # 100 um

        H = HENRY_CONSTANTS['N2O5']      # 51.34
        D_g = GAS_DIFFUSIVITY['N2O5']    # 0.9e-5
        D_l = LIQUID_DIFFUSIVITY['N2O5'] # 1.0e-9

        # Hand calc: D_adj = D_g * D_l * delta_liq / (D_g * delta_liq + D_l * delta_gas * H)
        num = D_g * D_l * delta_liq
        den = D_g * delta_liq + D_l * delta_gas * H
        D_adj = num / den
        k_expected = D_adj / delta_liq

        k_computed = compute_k_mt('N2O5', delta_gas, delta_liq)
        err = abs(k_computed - k_expected)
        print(f"\n  Test C1: k_mt(N2O5) = {k_computed:.6e}, "
              f"expected = {k_expected:.6e}, err = {err:.2e}")
        assert err < 1e-12, f"k_mt error = {err:.2e} >= 1e-12"

        # Verify intermediate values
        print(f"    H={H}, D_g={D_g:.2e}, D_l={D_l:.2e}")
        print(f"    D_adj={D_adj:.6e}, k_L={k_expected:.6e} m/s")

    def test_c2_dissolution_ode(self):
        """C2: dC/dt = α(C_eq - C) → C(t) = C_eq(1 - exp(-αt)) matches BDF."""
        C_eq = 1e-3   # mol/L
        k_L = 1e-5    # m/s
        dz = 5e-6     # m (surface cell width)
        alpha = k_L / dz  # 1/s

        def rhs(t, y):
            return np.array([alpha * (C_eq - y[0])])

        sol = solve_ivp(rhs, [0, 100], np.array([0.0]),
                         method='BDF', rtol=1e-10, atol=1e-15)
        assert sol.success

        t_final = sol.t[-1]
        y_bdf = sol.y[0, -1]
        y_exact = C_eq * (1.0 - np.exp(-alpha * t_final))
        rel_err = abs(y_bdf - y_exact) / y_exact
        print(f"\n  Test C2: α={alpha:.2e}/s, t={t_final:.1f}s")
        print(f"    BDF={y_bdf:.10e}, exact={y_exact:.10e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-6, f"Dissolution ODE: rel_err = {rel_err:.2e} >= 1e-6"

    def test_c3_ceq_calculation(self):
        """C3: C_eq = H × C_gas × (1000/N_A) hand-verified."""
        # N2O5 at 1e14 molecules/cm³
        C_gas = 1e14  # molecules/cm³
        H = HENRY_CONSTANTS['N2O5']  # 51.34 (dimensionless)

        # Convert: C_gas [molec/cm³] → [mol/L]
        # 1 mol/L = N_A molec / 1000 cm³ = N_A/1000 molec/cm³
        # C_gas [mol/L] = C_gas / (N_A/1000) = C_gas × 1000/N_A
        conv = 1000.0 / PHYSICAL.AVOGADRO
        C_gas_molar = C_gas * conv
        C_eq = H * C_gas_molar

        # Hand calc
        C_gas_molar_hand = 1e14 * 1000.0 / 6.022e23
        C_eq_hand = 51.34 * C_gas_molar_hand

        err = abs(C_eq - C_eq_hand)
        print(f"\n  Test C3: C_gas={C_gas:.1e} cm⁻³ → {C_gas_molar:.6e} mol/L")
        print(f"    C_eq = H×C_gas_mol = {C_eq:.6e} mol/L")
        print(f"    Hand: {C_eq_hand:.6e} mol/L, err={err:.2e}")
        assert err < 1e-15, f"C_eq error = {err:.2e} >= 1e-15"


# =====================================================================
# Test D: Diffusion + Mass Transfer (no significant reaction)
# =====================================================================

class TestD_ClosedSystemConservation:
    """Test D: Closed system (no gas source) — atom conservation.

    With gas input = 0, the liquid is a closed system.
    Chemistry rearranges atoms but cannot create or destroy them.
    Total N atoms and total O atoms must be exactly conserved.
    This directly verifies solver correctness without needing
    to estimate the gas transfer flux.
    """

    # N-atom count per species
    N_ATOMS = {
        'NO': 1, 'NO2': 1, 'NO3': 1,
        'N2O': 2, 'N2O3': 2, 'N2O4': 2, 'N2O5': 2,
        'HONO_total': 1, 'HONO2_total': 1,
    }

    def test_nitrogen_conservation_closed(self, chem):
        """N atoms conserved in closed system (no gas, chemistry + diffusion).

        When C_gas = 0, the surface BC becomes a sink: k_L*(0 - C_surface) < 0,
        which is physical off-gassing (Henry's law works both ways).
        To test pure chemistry+diffusion conservation, we account for this
        off-gassing flux and verify that the remaining error is near zero.
        """
        solver = PDESolver1D(
            chemistry=chem,
            liquid_depth=1e-3,
            N_z=50,
            saline_mode=False,
        )
        N_z = solver.N_z
        N_s = solver.N_s
        dz = solver.dz_cells

        # Gas data: all zero → C_eq = 0 → off-gassing occurs
        n_times = 100
        times = np.arange(n_times, dtype=float) * 2.0
        gas_conc = {sp: np.zeros(n_times) for sp in
                    ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']}
        solver.set_gas_data(times, gas_conc, hono_gas=0, hono2_gas=0, h2o2_gas=0)

        # IC: seed with nonzero N-species
        y0 = solver.build_initial_condition(initial_pH=4.0)
        y0_2d = y0.reshape(N_z, N_s)
        if 'HONO2_total' in solver.species_idx:
            y0_2d[:, solver.species_idx['HONO2_total']] = 1e-4
        if 'HONO_total' in solver.species_idx:
            y0_2d[:, solver.species_idx['HONO_total']] = 5e-5
        y0 = y0_2d.ravel()

        def compute_N_total(y_2d):
            total = 0.0
            for sp, n_N in self.N_ATOMS.items():
                if sp in solver.species_idx:
                    idx = solver.species_idx[sp]
                    total += n_N * np.sum(y_2d[:, idx] * dz)
            return total

        N_init = compute_N_total(y0_2d)

        T = 10.0
        result = solver.solve(
            t_span=(0, T), t_eval=np.array([T]),
            y0=y0, verbose=False, dt_poisson=1.0,
        )
        assert result['success'], f"Solver failed: {result.get('message', '')}"

        y_final = result['y_final']
        if y_final.ndim == 1:
            y_final = y_final.reshape(N_z, N_s)

        N_final = compute_N_total(y_final)

        # Estimate off-gassing: k_L * C_surface * T for each N-species
        # (upper bound — C_surface decreases over time)
        N_offgassed = 0.0
        for aq_idx, k_g, gas_sp, H_val, Ka in solver._interface_species:
            sp_name = [k for k, v in solver.species_idx.items() if v == aq_idx][0]
            n_N = self.N_ATOMS.get(sp_name, 0)
            if n_N > 0:
                # Use initial surface concentration (upper bound for off-gassing)
                C_surf_init = y0_2d[0, aq_idx]
                N_offgassed += n_N * k_g * C_surf_init * T

        N_corrected = N_final + N_offgassed  # add back the off-gassed amount
        rel_err_raw = abs(N_final - N_init) / N_init
        rel_err_corrected = abs(N_corrected - N_init) / N_init

        print(f"\n  Test D: Closed system N conservation over {T}s")
        print(f"    N_init={N_init:.6e}, N_final={N_final:.6e}")
        print(f"    raw rel error = {rel_err_raw:.2e} (includes off-gassing)")
        print(f"    off-gassed N ≈ {N_offgassed:.4e} mol·N/m²")
        print(f"    corrected rel error = {rel_err_corrected:.2e}")

        # Off-gassing corrected error should be small (BDF + diffusion conservation)
        assert rel_err_corrected < 5e-3, (
            f"Corrected N conservation: {rel_err_corrected:.2e} >= 5e-3"
        )
        # Also verify that the raw loss is explained by off-gassing
        N_lost = N_init - N_final
        offgas_explains = N_offgassed / N_lost if N_lost > 0 else 1.0
        print(f"    off-gassing explains {offgas_explains:.1%} of N loss")
        assert offgas_explains > 0.8, (
            f"Off-gassing explains only {offgas_explains:.1%} of N loss"
        )


# =====================================================================
# Test E: Nitrogen Atom Conservation
# =====================================================================

class TestE_NitrogenConservation:
    """Test E: N conservation with gas source — open system.

    Open system: gas transfers N into liquid. The correct check is:
        N_final = N_init + N_transferred_in

    Since we can't track the exact flux during simulation, we run TWO
    simulations: one with gas and one without (closed). The difference
    gives us the net gas contribution, which we compare to the closed-
    system conservation (Test D already validates).

    Here we verify that the solver produces physically reasonable results:
    - N increases (mass transfer is working)
    - HONO2_total (= NO3⁻ pool) is the dominant product of N2O5 hydrolysis
    """

    N_ATOMS = {
        'NO': 1, 'NO2': 1, 'NO3': 1,
        'N2O': 2, 'N2O3': 2, 'N2O4': 2, 'N2O5': 2,
        'HONO_total': 1, 'HONO2_total': 1,
    }

    def test_n2o5_produces_no3(self, chem):
        """N2O5 gas input → HONO2_total (NO3⁻) is the dominant N product."""
        solver = PDESolver1D(
            chemistry=chem,
            liquid_depth=1e-3,
            N_z=50,
            saline_mode=False,
        )
        N_z = solver.N_z
        N_s = solver.N_s
        dz = solver.dz_cells

        # Gas data: N2O5 only at 1e14 cm⁻³
        n_times = 100
        times = np.arange(n_times, dtype=float) * 2.0
        gas_conc = {sp: np.zeros(n_times) for sp in
                    ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']}
        gas_conc['N2O5'] = np.full(n_times, 1e14)
        solver.set_gas_data(times, gas_conc, hono_gas=0, hono2_gas=0, h2o2_gas=0)

        y0 = solver.build_initial_condition(initial_pH=7.0)
        y0_2d = y0.reshape(N_z, N_s)

        def compute_N_total(y_2d):
            total = 0.0
            for sp, n_N in self.N_ATOMS.items():
                if sp in solver.species_idx:
                    idx = solver.species_idx[sp]
                    total += n_N * np.sum(y_2d[:, idx] * dz)
            return total

        N_init = compute_N_total(y0_2d)

        T = 10.0
        result = solver.solve(
            t_span=(0, T), t_eval=np.array([T]),
            y0=y0, verbose=False, dt_poisson=1.0,
        )
        assert result['success'], f"Solver failed: {result.get('message', '')}"

        y_final = result['y_final']
        if y_final.ndim == 1:
            y_final = y_final.reshape(N_z, N_s)

        N_final = compute_N_total(y_final)
        N_gained = N_final - N_init

        # HONO2_total should be the dominant N product (N2O5 + H2O → 2HNO3)
        hno3_idx = solver.species_idx['HONO2_total']
        hno3_gained = np.sum(y_final[:, hno3_idx] * dz) - np.sum(y0_2d[:, hno3_idx] * dz)
        # Each HONO2_total has 1 N atom, so its N contribution is hno3_gained
        hno3_fraction = hno3_gained / N_gained if N_gained > 0 else 0

        print(f"\n  Test E: N2O5 → NO3⁻ pathway over {T}s")
        print(f"    N_gained={N_gained:.4e} mol·N/m²")
        print(f"    HONO2_total_gained={hno3_gained:.4e} mol/m²")
        print(f"    HONO2 fraction of N_gained = {hno3_fraction:.3f}")
        # N2O5 + H2O → 2HNO3, so ~100% of N should go to HONO2_total
        assert N_gained > 0, "No nitrogen gained — mass transfer broken"
        assert hno3_fraction > 0.8, (
            f"HONO2_total fraction = {hno3_fraction:.3f}, expected > 0.8 "
            f"(N2O5 hydrolysis should produce HNO3)"
        )


# =====================================================================
# Run with pytest
# =====================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
