"""
1D Drift-Diffusion-Reaction PDE Solver with Poisson (Vectorized).

Method of Lines (MOL): spatial discretization → ODE system.
Solver: scipy.integrate.solve_ivp with BDF + Jacobian sparsity.

Physical system (Nernst-Planck + Poisson):
    ∂C_i/∂t = D_i ∂²C_i/∂z² − ∂/∂z(Z_eff,i μ_i C_i E) + R_i(C)
    ∂²φ/∂z² = −ρ / (ε₀ εᵣ)
    E = −∂φ/∂z

Boundary conditions:
    z=0: -D_i ∂C_i/∂z + Z_eff,i μ_i C_i E = k_L,i (C_eq,i(t) - C_i(0))
    z=L: total flux = 0  (no-flux for diffusion + drift)
    φ(0) = 0 (Dirichlet), ∂φ/∂z(L) = 0 (Neumann)

State vector layout (grid-major):
    y[j * N_s + i] = concentration of species i at grid point j

References:
    Liu et al. (2015) J. Phys. D — Eqs. 6-7 (DDA+Poisson in liquid)
    Ikuse & Hamaguchi (2022) JJAP — Nernst-Planck + Poisson, SG scheme
    Lee et al. (2023) CEJ — Two-film BC
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix, csc_matrix

from scipy.linalg import solve_banded

from config_1d import (
    PHYSICAL, HENRY_CONSTANTS, GAS_DIFFUSIVITY, LIQUID_DIFFUSIVITY,
    D_GAS_DEFAULT, D_LIQ_DEFAULT, AQUEOUS_SPECIES, GAS_TO_AQUEOUS_MAP,
    MASS_TRANSFER, GRID, DEFAULTS, ODE_CONFIG, ACID_BASE_PAIRS,
    POISSON, SPECIES_CHARGE, PKA_VALUES,
    SALINE_SPECIES_CHARGE,
)
from chemistry_1d import AqueousChemistry1D


# =============================================================================
# Mass Transfer Utilities
# =============================================================================

def compute_k_mt(species_gas: str, delta_gas: float, delta_liq: float,
                  bc_type: str = 'two_film', alpha_b: float = 1.0) -> float:
    """Overall mass transfer coefficient k [m/s].

    bc_type controls the interfacial BC model:
      'two_film'   : D_adj / δ_liq  (Lee 2023, two-film resistance)
      'dirichlet'  : 1.0 m/s        (stiff relaxation → C(0) ≈ C_eq)
      'film'       : D_l / δ_liq    (Heirman 2025 Eq.6, liquid-side only)
      'film_alpha' : α_b × D_l / δ_liq  (Heirman 2025 Eq.7)
    """
    if bc_type == 'dirichlet':
        return 1.0

    D_l = LIQUID_DIFFUSIVITY.get(species_gas, D_LIQ_DEFAULT)

    if bc_type == 'film':
        return D_l / delta_liq

    if bc_type == 'film_alpha':
        return alpha_b * D_l / delta_liq

    # Default: two_film (Lee 2023)
    H = HENRY_CONSTANTS.get(species_gas, 1.0)
    D_g = GAS_DIFFUSIVITY.get(species_gas, D_GAS_DEFAULT)
    num = D_g * D_l * delta_liq
    den = D_g * delta_liq + D_l * delta_gas * H
    D_adj = num / max(den, 1e-30)
    return D_adj / delta_liq


# =============================================================================
# PDE Solver
# =============================================================================

class PDESolver1D:
    """
    1D diffusion-reaction PDE solver using Method of Lines.

    Vectorized diffusion, per-cell chemistry loop.
    """

    def __init__(
        self,
        chemistry: AqueousChemistry1D,
        liquid_depth: float = None,
        N_z: int = None,
        dz_min: float = None,
        stretch_ratio: float = None,
        delta_gas: float = None,
        delta_liq: float = None,
        mass_transfer_eta: float = 1.0,
        saline_mode: bool = False,
        fixed_cation_conc: float = 0.0,
        bc_type: str = None,
        alpha_b: float = None,
    ):
        self.chem = chemistry
        self.N_s = chemistry.n_species
        self.species_idx = chemistry.species_idx
        self.saline_mode = saline_mode
        self._fixed_cation_conc = fixed_cation_conc
        self._acid_base_pairs = chemistry.pKa_map
        # Solver routing:
        # _use_strang=True  → Strang operator splitting (legacy, for comparison)
        # _use_strang=False → monolithic BDF (default, no splitting error)
        self._use_strang = False

        # Grid
        self.L = liquid_depth or MASS_TRANSFER.liquid_depth
        if dz_min is not None:
            _ratio = stretch_ratio or GRID.stretch_ratio
            self._build_grid_geometric(dz_min, _ratio)
        else:
            self._build_grid_uniform(N_z or GRID.N_z)

        # Mass transfer
        self.delta_gas = delta_gas or MASS_TRANSFER.delta_x_gas
        self.delta_liq = delta_liq or MASS_TRANSFER.delta_x_liq
        self.mass_transfer_eta = mass_transfer_eta  # η: scales all k_L values
        self.bc_type = bc_type or MASS_TRANSFER.bc_type
        self.alpha_b = alpha_b if alpha_b is not None else MASS_TRANSFER.alpha_b

        # Total ODE size
        self.N_total = self.N_s * self.N_z

        # Diffusion coefficients per species
        self.D_species = self._build_diffusion_array()

        # Pre-compute mass transfer: gas-side only Robin BC.
        # k_g = D_gas / delta_gas — liquid-side resistance is resolved by PDE.
        # Ka = acid dissociation constant for _total species (None for non-dissociating).
        # Used to compute molecular fraction f_mol = [H+]/([H+]+Ka) so that
        # only the neutral molecular form participates in gas-liquid transfer.
        self._interface_species = []
        for gas_sp, aq_sp in GAS_TO_AQUEOUS_MAP.items():
            if aq_sp in self.species_idx:
                aq_idx = self.species_idx[aq_sp]
                k_mt = compute_k_mt(gas_sp, self.delta_gas, self.delta_liq,
                                    bc_type=self.bc_type, alpha_b=self.alpha_b)
                H = HENRY_CONSTANTS.get(gas_sp, 1.0)
                # For _total species, get Ka from ACID_BASE_PAIRS
                Ka = None
                if aq_sp in ACID_BASE_PAIRS:
                    pKa = ACID_BASE_PAIRS[aq_sp][2]
                    Ka = 10.0 ** (-pKa)
                self._interface_species.append((aq_idx, k_mt, gas_sp, H, Ka))

        # Precompute D/V_T only (SG uses h_faces, not scalar dz)
        # _inv_dz_0 caches 1/dz_cells[0] for interface BC in rhs()
        self._inv_dz_0 = self.inv_dz_cells[0]

        # Jacobian sparsity
        self.jac_sparsity = self._build_jac_sparsity()

        # Gas data (set via set_gas_data)
        self._gas_conc_molar = None
        self._gas_times = None
        self._hono_ceq = 0.0
        self._hono2_ceq = 0.0
        self._h2o2_ceq = 0.0

        # OH- index for zeroing
        self._oh_minus_idx = self.species_idx.get('OH-', -1)
        self._cl_minus_idx = self.species_idx.get('Cl-', -1) if saline_mode else -1

        # Cl atom conservation projection indices (for saline)
        self._cl_cons_idx = []
        if saline_mode and self._cl_minus_idx >= 0:
            _cl_atom_count = {
                'Cl-': 1, 'HOCl-': 1, 'Cl2-': 2, 'Cl': 1, 'HOClH': 1,
                'Cl2': 2, 'HCl': 1, 'HClO_total': 1, 'HClO2_total': 1,
                'ClO': 1, 'ClO2': 1, 'ClO3': 1, 'ClNO2': 1,
                'Cl3-': 3, 'ClO3-': 1, 'ClO4-': 1,
                'Cl2O': 2, 'Cl2O2': 2, 'Cl2O3': 2,
                'Cl2O4': 2, 'Cl2O5': 2, 'Cl2O6': 2,
            }
            for sp_name, n_cl in _cl_atom_count.items():
                idx = self.species_idx.get(sp_name, -1)
                if idx >= 0:
                    self._cl_cons_idx.append((idx, n_cl))

        # ================================================================
        # Poisson + Drift-Diffusion pre-computation
        # ================================================================
        if saline_mode:
            self._poisson_enabled = False
        else:
            self._poisson_enabled = POISSON.enabled

        # Permittivity: ε = ε₀ × εᵣ  [F/m]
        self._epsilon = PHYSICAL.EPSILON_0 * POISSON.epsilon_r  # 6.94e-10

        # Thermal voltage: V_T = kB*T / e  [V]
        T = DEFAULTS.temperature_K  # 298.15 K
        self._V_T = PHYSICAL.KB * T / PHYSICAL.E_CHARGE  # 0.02569 V

        # Faraday constant [C/mol] — for charge density ρ = F × Σ Z_i C_i
        self._F = PHYSICAL.FARADAY  # 96485

        all_charges = dict(SPECIES_CHARGE)
        if saline_mode:
            all_charges.update(SALINE_SPECIES_CHARGE)

        self._direct_charge = {}
        for sp, Z in all_charges.items():
            if sp == 'OH-':
                continue
            if sp in self.species_idx:
                self._direct_charge[self.species_idx[sp]] = Z

        self._total_charge_info = []
        for total_name, (acid_name, base_name, pKa) in self._acid_base_pairs.items():
            if total_name in self.species_idx:
                Ka = 10.0 ** (-pKa)
                self._total_charge_info.append((
                    self.species_idx[total_name],
                    0,
                    -1,
                    Ka,
                ))

        # H+ index for speciation
        self._h_plus_idx = self.species_idx.get('H+', -1)

        # --- Mobility array (per species): μ_i = |Z_i| * D_i / V_T ---
        # For total variables, effective Z is computed at runtime
        # Pre-compute D_i / V_T for each species (multiply by |Z_eff| at runtime)
        self._D_over_VT = self.D_species / self._V_T  # [m²/(V·s)] when multiplied by |Z|

        # Precompute Poisson tridiagonal (geometry-only, state-independent)
        self._poisson_ab, self._poisson_rhs_coeff = self._build_poisson_tridiag()

        # Pre-compute C_eq arrays (after set_gas_data)
        self._ceq_arrays = {}  # gas_sp -> ndarray of C_eq values

        # Operator splitting: frozen E-field from outer Poisson loop.
        # When not None, rhs() uses this instead of solving Poisson inline.
        self._E_half_frozen = None

        # --- Grouped FD Jacobian precomputation ---
        self._jac_groups = self._build_jac_groups()
        self._n_jac_groups = int(self._jac_groups.max()) + 1

        # Pre-compute group→column lists as numpy arrays for vectorized FD
        self._jac_groups_np = []
        for g in range(self._n_jac_groups):
            cols = np.where(self._jac_groups == g)[0]
            if len(cols) > 0:
                self._jac_groups_np.append(cols)

        # Pre-compute flat COO row/col arrays per group for vectorized scatter
        sp = self.jac_sparsity
        self._jac_col_rows = {}
        self._jac_group_coo = []  # [(rows_arr, cols_arr)] per group
        for cols_np in self._jac_groups_np:
            g_rows = []
            g_cols = []
            for col in cols_np:
                rows_for_col = sp[:, col].nonzero()[0]
                self._jac_col_rows[col] = rows_for_col
                g_rows.append(rows_for_col)
                g_cols.append(np.full(len(rows_for_col), col, dtype=np.intp))
            self._jac_group_coo.append((
                np.concatenate(g_rows) if g_rows else np.array([], dtype=np.intp),
                np.concatenate(g_cols) if g_cols else np.array([], dtype=np.intp),
            ))

        # Also keep list-of-lists for backward compat
        self._jac_groups_list = [c.tolist() for c in self._jac_groups_np]

    def _build_diffusion_array(self) -> np.ndarray:
        D = np.full(self.N_s, D_LIQ_DEFAULT)
        for sp, D_val in LIQUID_DIFFUSIVITY.items():
            if sp in self.species_idx:
                D[self.species_idx[sp]] = D_val

        for total_sp, gas_sp in [('HONO_total', 'HONO'), ('HONO2_total', 'HONO2'),
                                  ('H2O2_total', 'H2O2'), ('HO2_total', 'HO2'),
                                  ('HClO_total', 'HClO'), ('HClO2_total', 'HClO2')]:
            if total_sp in self.species_idx:
                D[self.species_idx[total_sp]] = LIQUID_DIFFUSIVITY.get(gas_sp, D_LIQ_DEFAULT)

        if 'H+' in self.species_idx:
            D[self.species_idx['H+']] = 9.3e-9
        if 'OH-' in self.species_idx:
            D[self.species_idx['OH-']] = 5.3e-9

        return D

    def _build_grid_uniform(self, N_z: int):
        self.N_z = N_z
        dz = self.L / N_z
        self.dz_cells = np.full(N_z, dz)
        edges = np.linspace(0, self.L, N_z + 1)
        self.z_centers = 0.5 * (edges[:-1] + edges[1:])
        self.h_faces = np.full(max(N_z - 1, 0), dz)
        self.inv_dz_cells = np.full(N_z, 1.0 / dz)
        self.inv_h_faces = np.full(max(N_z - 1, 0), 1.0 / dz)
        self._grid_type = 'uniform'

    def _build_grid_geometric(self, dz_min: float, ratio: float):
        L = self.L
        if ratio <= 1.0 + 1e-10:
            N = max(int(round(L / dz_min)), 2)
            self._build_grid_uniform(N)
            return

        N = int(np.ceil(np.log(1 + L * (ratio - 1) / dz_min) / np.log(ratio)))
        N = max(N, 2)

        dz_cells = dz_min * ratio ** np.arange(N, dtype=np.float64)
        overshoot = np.sum(dz_cells) - L
        dz_cells[-1] -= overshoot
        dz_cells[-1] = max(dz_cells[-1], dz_cells[-2] * 0.5)

        edges = np.concatenate(([0.0], np.cumsum(dz_cells)))
        z_centers = 0.5 * (edges[:-1] + edges[1:])
        h_faces = z_centers[1:] - z_centers[:-1]

        self.N_z = N
        self.dz_cells = dz_cells
        self.z_centers = z_centers
        self.h_faces = h_faces
        self.inv_dz_cells = 1.0 / dz_cells
        self.inv_h_faces = 1.0 / h_faces
        self._grid_type = 'geometric'

    def _build_poisson_tridiag(self):
        N = self.N_z
        if N < 2:
            return np.zeros((3, max(N, 1))), np.zeros(max(N, 1))

        ab = np.zeros((3, N))
        inv_h = self.inv_h_faces

        ab[1, 0] = 1.0  # Dirichlet φ[0] = 0

        if N > 2:
            j = np.arange(1, N - 1)
            ab[2, j - 1] = inv_h[j - 1]
            ab[1, j] = -(inv_h[j - 1] + inv_h[j])
            ab[0, j + 1] = inv_h[j]

        ab[2, N - 2] = inv_h[N - 2]
        ab[1, N - 1] = -inv_h[N - 2]

        rhs_coeff = np.zeros(N)
        rhs_coeff[1:] = -self.dz_cells[1:] / self._epsilon

        return ab, rhs_coeff

    def _build_jac_groups(self) -> np.ndarray:
        """Analytical graph coloring for block-tridiagonal Jacobian.
        color = (cell_index % 3) * N_s + species_index. O(N)."""
        N = self.N_total
        N_s = self.N_s
        N_z = self.N_z
        colors = np.empty(N, dtype=int)
        for j in range(N_z):
            for i in range(N_s):
                colors[j * N_s + i] = (j % 3) * N_s + i
        return colors

    def _build_jac_sparsity(self):
        N = self.N_total
        jac = lil_matrix((N, N), dtype=np.int8)

        # Block-tridiagonal: each cell couples to itself and neighbors
        for j in range(self.N_z):
            i0 = j * self.N_s
            i1 = i0 + self.N_s
            jac[i0:i1, i0:i1] = 1
            if j > 0:
                ip = (j - 1) * self.N_s
                jac[i0:i1, ip:ip + self.N_s] = 1
            if j < self.N_z - 1:
                inxt = (j + 1) * self.N_s
                jac[i0:i1, inxt:inxt + self.N_s] = 1

        # Poisson coupling is quasi-static: solved in RHS but NOT represented
        # in Jacobian sparsity.  Block-tridiagonal structure is preserved →
        # efficient graph coloring (3×N_s groups) and O(N) LU factorization.
        # The missing global coupling makes the Jacobian approximate, but
        # BDF handles this via extra Newton iterations per step.

        return jac.tocsc()

    # =================================================================
    # Poisson + Drift-Diffusion Methods
    # =================================================================

    def _compute_charge_density(self, y_2d: np.ndarray) -> np.ndarray:
        """
        Compute charge density ρ(z) [C/m³] from species concentrations.

        ρ = F × Σ Z_i × C_i   (summed over all charged species)

        For total variables (e.g. HONO2_total = HONO2 + NO3-),
        speciation via H+ determines the charged fraction.

        Concentrations are in mol/L → mol/m³ = mol/L × 1000.

        Returns: rho (N_z,) in C/m³
        """
        N_z = self.N_z
        H_idx = self._h_plus_idx

        # Net charge sum per cell: Z_net[j] in mol/L
        Z_net = np.zeros(N_z)

        # 1) Direct-tracked ions: H+, OH-, O-, O3-
        for sp_idx, Z in self._direct_charge.items():
            Z_net += Z * y_2d[:, sp_idx]

        # 2) OH- contribution (algebraic: Kw / [H+])
        if H_idx >= 0:
            H_plus = np.maximum(y_2d[:, H_idx], 1e-14)
            OH_minus = 1e-14 / H_plus
            Z_net -= OH_minus  # OH- has Z = -1

        # 3) Total variables: compute base fraction → charge contribution
        if H_idx >= 0:
            H_plus = np.maximum(y_2d[:, H_idx], 1e-14)
            for total_idx, Z_acid, Z_base, Ka in self._total_charge_info:
                C_total = y_2d[:, total_idx]
                f_base = Ka / (H_plus + Ka)
                Z_net += Z_base * f_base * C_total

        # 4) Fixed cation background (Na⁺ from NaCl)
        if self._fixed_cation_conc > 0:
            Z_net += self._fixed_cation_conc

        rho = self._F * 1000.0 * Z_net

        return rho

    def _enforce_electroneutrality(self, y: np.ndarray):
        """
        Enforce electroneutrality by solving for [H+] from charge balance.

        The total-variable formulation tracks HONO_total etc. as ODE variables,
        with algebraic speciation (acid/base fractions from Ka and [H+]).
        When total variables change (mass transfer, reactions), the speciation
        adjusts but [H+] is NOT automatically updated.  This breaks
        electroneutrality.

        Fix: solve f(H) = 0 for each cell, where
            f(H) = H - Kw/H - Σ Ka_j/(H+Ka_j) × C_total_j - [O-] - [O3-]

        This is monotonically increasing in H → unique root.
        Uses Newton's method (3-5 iterations).
        """
        N_s = self.N_s
        N_z = self.N_z
        hp = self._h_plus_idx
        if hp < 0:
            return

        Kw = 1e-14
        y_2d = y.reshape(N_z, N_s)

        for j in range(N_z):
            # Gather anion contributions from direct ions (O-, O3-)
            fixed_anion = 0.0
            for sp_idx, Z in self._direct_charge.items():
                if sp_idx == hp:
                    continue  # skip H+
                if Z < 0:
                    fixed_anion += abs(Z) * max(y_2d[j, sp_idx], 0.0)

            # Gather total variable contributions
            total_info = []  # (C_total, Ka) pairs
            for total_idx, Z_acid, Z_base, Ka in self._total_charge_info:
                C_total = max(y_2d[j, total_idx], 0.0)
                if C_total > 1e-30:
                    total_info.append((C_total, Ka))

            H = max(y_2d[j, hp], 1e-14)

            for _ in range(15):
                f_val = H + self._fixed_cation_conc - Kw / H - fixed_anion
                df_val = 1.0 + Kw / (H * H)

                for C_t, Ka in total_info:
                    denom = H + Ka
                    f_val -= Ka / denom * C_t
                    df_val += Ka * C_t / (denom * denom)

                if abs(f_val) < 1e-14:
                    break

                dH = -f_val / df_val
                H_new = H + dH
                if H_new <= 0:
                    H_new = H * 0.5
                H = max(H_new, 1e-14)

            # Update H+ in state vector
            y_2d[j, hp] = H

    def _enforce_cl_conservation(self, y_flat: np.ndarray, y_before_flat: np.ndarray):
        """Per-cell Cl atom conservation projection (Sturm & Silva 2024).

        Projects BDF numerical error onto Cl⁻ (dominant Cl pool).
        """
        if not self._cl_cons_idx or self._cl_minus_idx < 0:
            return
        N_s = self.N_s
        N_z = self.N_z
        cl_idx = self._cl_minus_idx
        trace = DEFAULTS.trace_concentration
        y_2d = y_flat.reshape(N_z, N_s)
        y_before_2d = y_before_flat.reshape(N_z, N_s)
        for j in range(N_z):
            cl_after = sum(n_cl * y_2d[j, si] for si, n_cl in self._cl_cons_idx)
            cl_before = sum(n_cl * y_before_2d[j, si] for si, n_cl in self._cl_cons_idx)
            err = cl_after - cl_before
            y_2d[j, cl_idx] -= err
            if y_2d[j, cl_idx] < trace:
                y_2d[j, cl_idx] = trace

    def _solve_poisson_1d(self, rho: np.ndarray) -> np.ndarray:
        """
        Solve 1D Poisson on non-uniform FV grid (precomputed tridiagonal).

        Returns: E_half (N_z-1,) — E_{j+1/2} = -(φ_{j+1} - φ_j) / h_{j+1/2}
        """
        N = self.N_z
        if N < 2:
            return np.zeros(max(N - 1, 0))

        rhs_vec = self._poisson_rhs_coeff * rho
        rhs_vec[0] = 0.0

        phi = solve_banded((1, 1), self._poisson_ab, rhs_vec)

        E_half = -(phi[1:] - phi[:-1]) * self.inv_h_faces
        return E_half

    @staticmethod
    def _bernoulli(x: np.ndarray) -> np.ndarray:
        """
        Bernoulli function: B(x) = x / (exp(x) - 1).

        Limits: B(0) = 1, B(x→+∞) → 0, B(x→-∞) → -x.
        Uses Taylor series near x=0 for numerical stability.

        Note: np.where evaluates BOTH branches for ALL elements, so we
        must guard x/expm1(x) against x=0 by substituting a safe value
        before evaluating the large-|x| branch.

        Reference: Scharfetter & Gummel (1969), IEEE Trans. Electron Devices.
        """
        small = np.abs(x) < 1e-4
        # Guard: replace small |x| with 1.0 before evaluating expm1
        # (value doesn't matter — np.where will discard it)
        x_safe = np.where(small, 1.0, x)
        return np.where(
            small,
            1.0 - x / 2.0 + x * x / 12.0,
            x_safe / np.expm1(x_safe),
        )

    def _compute_sg_transport(
        self, y_2d: np.ndarray, E_half: np.ndarray,
    ) -> np.ndarray:
        """
        Compute transport (diffusion + drift) using Scharfetter-Gummel flux.

        Replaces separate central-diff diffusion + central-diff drift with
        a unified exponentially-fitted flux that is stable at all Péclet numbers.

        SG flux at interface j+1/2:
            J_{j+1/2} = (D_i/dz) × [B(α) × C_j − B(−α) × C_{j+1}]

        where α = Z_eff,i × (φ_{j+1} − φ_j) / V_T
                = −Z_eff,i × E_{j+1/2} × dz / V_T

        For neutral species (Z_eff=0): α=0, B(0)=1 → standard diffusion flux.
        For charged species: exponential fitting captures drift exactly.

        Divergence (contribution to ∂C/∂t):
            transport[j,i] = −(J_{j+1/2} − J_{j−1/2}) / dz

        Boundary conditions:
            j=0: J_{-1/2} = 0 (no-flux; interface mass transfer added separately)
            j=N-1: J_{N-1/2} = 0 (no-flux at bottom)

        Args:
            y_2d: (N_z, N_s) concentration array [mol/L]
            E_half: (N_z-1,) electric field at cell interfaces [V/m]

        Returns:
            transport: (N_z, N_s) transport contribution to dydt [mol/(L·s)]

        References:
            Scharfetter & Gummel (1969), IEEE Trans. Electron Devices 16, 64.
            Ikuse & Hamaguchi (2022), JJAP 61, 076002.
        """
        N_z = self.N_z
        N_s = self.N_s
        h = self.h_faces                # (N_z-1,) center-to-center distances
        inv_h = self.inv_h_faces        # (N_z-1,)
        inv_dz = self.inv_dz_cells      # (N_z,) 1/cell_width for divergence
        V_T = self._V_T
        H_idx = self._h_plus_idx

        Z_eff = np.zeros((N_z, N_s))
        for sp_idx, Z in self._direct_charge.items():
            Z_eff[:, sp_idx] = Z

        if H_idx >= 0:
            H_plus = np.maximum(y_2d[:, H_idx], 1e-14)
            for total_idx, Z_acid, Z_base, Ka in self._total_charge_info:
                f_base = Ka / (H_plus + Ka)
                Z_eff[:, total_idx] = Z_acid * (1.0 - f_base) + Z_base * f_base

        Z_eff_half = 0.5 * (Z_eff[:-1, :] + Z_eff[1:, :])

        # α_{j+1/2} = -Z_eff × E_{j+1/2} × h_{j+1/2} / V_T
        alpha = -Z_eff_half * (E_half[:, np.newaxis] * h[:, np.newaxis] / V_T)
        np.clip(alpha, -500.0, 500.0, out=alpha)

        Bp = self._bernoulli(alpha)
        Bm = self._bernoulli(-alpha)

        # J_{j+1/2} = (D_i / h_{j+1/2}) × [B(α) C_j − B(−α) C_{j+1}]
        D = self.D_species
        J_half = D * inv_h[:, np.newaxis] * (Bp * y_2d[:-1, :] - Bm * y_2d[1:, :])

        # Divergence: -(J_{j+1/2} - J_{j-1/2}) / Δz_j
        transport = np.zeros((N_z, N_s))
        transport[0, :] = -J_half[0, :] * inv_dz[0]
        if N_z > 2:
            transport[1:-1, :] = -(J_half[1:, :] - J_half[:-1, :]) * inv_dz[1:-1, np.newaxis]
        transport[-1, :] = J_half[-1, :] * inv_dz[-1]

        return transport

    def set_gas_data(
        self,
        times: np.ndarray,
        gas_conc_molecules: Dict[str, np.ndarray],
        hono_gas: float = 0.0,
        hono2_gas: float = 0.0,
        h2o2_gas: float = 0.0,
    ):
        """Set gas-phase boundary conditions."""
        conv = 1000.0 / PHYSICAL.AVOGADRO

        self._gas_times = times
        self._dt_gas = float(times[1] - times[0]) if len(times) > 1 else 1.0
        self._gas_conc_molar = {}

        for gas_sp in ['O', 'O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
            if gas_sp in gas_conc_molecules:
                arr = np.maximum(gas_conc_molecules[gas_sp], 0.0) * conv
            else:
                arr = np.zeros_like(times)
            self._gas_conc_molar[gas_sp] = arr

        self._hono_gas_molar = max(hono_gas, 0.0) * conv
        self._hono2_gas_molar = max(hono2_gas, 0.0) * conv
        self._h2o2_gas_molar = max(h2o2_gas, 0.0) * conv

        # Pre-compute C_eq time series: C_eq = H × C_gas
        self._n_times = len(times)
        self._ceq_lookup = {}
        for aq_idx, k_g_val, gas_sp, H, Ka in self._interface_species:
            if gas_sp == 'HONO':
                ceq = H * self._hono_gas_molar  # scalar
                self._ceq_lookup[gas_sp] = ('const', ceq)
            elif gas_sp == 'HONO2':
                ceq = H * self._hono2_gas_molar
                self._ceq_lookup[gas_sp] = ('const', ceq)
            elif gas_sp == 'H2O2':
                ceq = H * self._h2o2_gas_molar
                self._ceq_lookup[gas_sp] = ('const', ceq)
            elif gas_sp in self._gas_conc_molar:
                ceq_arr = H * self._gas_conc_molar[gas_sp]
                self._ceq_lookup[gas_sp] = ('array', ceq_arr)
            else:
                self._ceq_lookup[gas_sp] = ('const', 0.0)

    def _get_C_eq_fast(self, gas_sp: str, t_idx: int) -> float:
        """Fast C_eq lookup using pre-computed data."""
        kind, data = self._ceq_lookup[gas_sp]
        if kind == 'const':
            return data
        else:
            return data[min(t_idx, self._n_times - 1)]

    def build_initial_condition(self, initial_pH: float = 7.0) -> np.ndarray:
        trace = DEFAULTS.trace_concentration
        y0 = np.full(self.N_total, trace)

        H_conc = 10.0 ** (-initial_pH)
        OH_conc = 1e-14 / H_conc

        # Vectorized initialization across all grid points
        for j in range(self.N_z):
            off = j * self.N_s
            if 'O2' in self.species_idx:
                y0[off + self.species_idx['O2']] = 2.5e-4
            if 'N2' in self.species_idx:
                y0[off + self.species_idx['N2']] = 5e-4
            if 'H+' in self.species_idx:
                y0[off + self.species_idx['H+']] = H_conc
            if 'OH-' in self.species_idx:
                y0[off + self.species_idx['OH-']] = OH_conc
            if 'OH' in self.species_idx:
                y0[off + self.species_idx['OH']] = 1e-12
            if self.saline_mode and 'Cl-' in self.species_idx:
                y0[off + self.species_idx['Cl-']] = 0.154

        if self.N_z > 1:
            rng = np.random.default_rng(42)  # deterministic seed
            y0 *= (1.0 + 1e-6 * rng.standard_normal(self.N_total))
            np.clip(y0, trace, None, out=y0)

        return y0

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Vectorized RHS: reactions + SG transport + Poisson + flux BC.

        Flow:
            1. Reactions (per-cell loop)
            2. Poisson → E_half (quasi-static, if enabled)
            3. SG transport (unified diffusion + drift)
            4. Interface flux BC at j=0 (factor 1/dz)
            5. OH- algebraic constraint

        Transport uses Scharfetter-Gummel flux which replaces the
        old central-diff diffusion + central-diff drift with a single
        exponentially-fitted flux stable at all Péclet numbers.
        """
        N_s = self.N_s
        N_z = self.N_z
        dydt = np.zeros(self.N_total, dtype=np.float64)

        trace = DEFAULTS.trace_concentration
        max_conc = ODE_CONFIG.max_concentration

        # Global clamp: prevent -inf/inf/NaN from corrupting Poisson solver.
        # scipy's num_jac can produce NaN perturbation vectors when its
        # internal factor array overflows for trace-concentration species.
        y_safe = np.where(np.isfinite(y), y, 0.0)
        np.clip(y_safe, -max_conc, max_conc, out=y_safe)

        # --- 1. Reactions: per grid cell (Numba-accelerated) ---
        y_2d_clipped = np.clip(y_safe.reshape(N_z, N_s), trace, max_conc)
        dydt[:] = self.chem.compute_rates_batch(y_2d_clipped).ravel()

        # Reshape for spatial operations
        y_2d = y_safe.reshape(N_z, N_s)
        dydt_2d = dydt.reshape(N_z, N_s)

        # Clamp concentrations for transport (avoid negatives)
        y_2d_clamped = np.maximum(y_2d, trace)

        # --- 2. E-field for SG transport ---
        if self._E_half_frozen is not None:
            # Operator splitting: use frozen E-field from outer loop
            E_half = self._E_half_frozen
        elif self._poisson_enabled and N_z >= 2:
            # Monolithic mode: solve Poisson inline (legacy path)
            rho = self._compute_charge_density(y_2d_clamped)
            E_half = self._solve_poisson_1d(rho)
        else:
            E_half = np.zeros(max(N_z - 1, 0))

        # --- 3. SG Transport (replaces diffusion + drift) ---
        transport = self._compute_sg_transport(y_2d_clamped, E_half)
        dydt_2d += transport

        # --- 4. Interface flux at j=0 (factor 1/dz for SG FV) ---
        # For _total species, only the molecular fraction can transfer:
        #   c_eff = f_mol × c_total,  f_mol = [H+]/([H+]+Ka)
        t_idx = max(0, min(int(t / self._dt_gas), self._n_times - 1))
        hp = self._h_plus_idx
        h_surface = y_2d_clamped[0, hp] if hp >= 0 else 1e-7
        for aq_idx, k_g_val, gas_sp, H, Ka in self._interface_species:
            C_eq = self._get_C_eq_fast(gas_sp, t_idx)
            c0 = y_2d_clamped[0, aq_idx]
            c_eff = c0 * h_surface / (h_surface + Ka) if Ka is not None else c0
            dydt_2d[0, aq_idx] += self._inv_dz_0 * k_g_val * (C_eq - c_eff)

        # --- 5. OH⁻ is algebraic (Kw/H⁺) ---
        if self._oh_minus_idx >= 0:
            dydt_2d[:, self._oh_minus_idx] = 0.0
        # Cl⁻ evolves via reactions in monolithic mode (no zeroing)

        return dydt.ravel()

    def _rhs_fixed_E(self, t: float, y: np.ndarray,
                     E_half: np.ndarray) -> np.ndarray:
        """RHS with externally provided E-field (for Newton within Gummel)."""
        N_s = self.N_s
        N_z = self.N_z
        trace = DEFAULTS.trace_concentration
        y_2d = np.maximum(y.reshape(N_z, N_s), trace)
        dydt_2d = np.zeros((N_z, N_s))

        dydt_2d[:] = self.chem.compute_rates_batch(y_2d)

        if N_z > 1:
            transport = self._compute_sg_transport(y_2d, E_half)
            dydt_2d += transport

        t_idx = max(0, min(int(t / self._dt_gas), self._n_times - 1))
        hp = self._h_plus_idx
        h_surface = y_2d[0, hp] if hp >= 0 else 1e-7
        for aq_idx, k_g_val, gas_sp, H, Ka in self._interface_species:
            C_eq = self._get_C_eq_fast(gas_sp, t_idx)
            c0 = y_2d[0, aq_idx]
            c_eff = c0 * h_surface / (h_surface + Ka) if Ka is not None else c0
            dydt_2d[0, aq_idx] += self._inv_dz_0 * k_g_val * (C_eq - c_eff)

        if self._oh_minus_idx >= 0:
            dydt_2d[:, self._oh_minus_idx] = 0.0

        return dydt_2d.ravel()

    def _compute_jac_fd(self, t: float, y: np.ndarray) -> 'csc_matrix':
        """Vectorized grouped finite-difference Jacobian."""
        N = self.N_total
        f0 = self.rhs(t, y)
        eps_sqrt = 1.4901161193847656e-08
        h_threshold = 1e-10
        h = eps_sqrt * np.maximum(np.abs(y), h_threshold)

        all_rows, all_cols, all_vals = [], [], []
        for gi, cols_np in enumerate(self._jac_groups_np):
            y_pert = y.copy()
            y_pert[cols_np] += h[cols_np]
            f_pert = self.rhs(t, y_pert)
            diff = f_pert - f0
            g_rows, g_cols = self._jac_group_coo[gi]
            if len(g_rows) == 0:
                continue
            vals = diff[g_rows] / h[g_cols]
            nonzero = vals != 0.0
            if np.any(nonzero):
                all_rows.append(g_rows[nonzero])
                all_cols.append(g_cols[nonzero])
                all_vals.append(vals[nonzero])

        if all_rows:
            return csc_matrix(
                (np.concatenate(all_vals),
                 (np.concatenate(all_rows), np.concatenate(all_cols))),
                shape=(N, N))
        return csc_matrix((N, N))

    def _compute_jac_banded(self, t: float, y: np.ndarray,
                            dt: float,
                            E_half: np.ndarray = None,
                            ) -> Tuple[np.ndarray, int]:
        """
        J = I − dt × ∂RHS_fixed_E/∂y in banded format (lu = 2N_s−1).

        Per-cell chemistry FD (diagonal blocks) + analytical SG
        transport (tri-diagonal blocks) + interface flux BC.
        Poisson coupling omitted (handled by Gummel outer loop).

        E_half: frozen electric field at cell faces.  If None,
                computed from current y via Poisson.
        """
        N_s = self.N_s
        N_z = self.N_z
        N = self.N_total
        lu = 2 * N_s - 1
        trace = DEFAULTS.trace_concentration
        max_conc = ODE_CONFIG.max_concentration
        eps_sqrt = 1.4901161193847656e-08

        ab = np.zeros((2 * lu + 1, N))
        ab[lu, :] = 1.0

        y_safe = np.where(np.isfinite(y), y, 0.0)
        np.clip(y_safe, -max_conc, max_conc, out=y_safe)

        for j in range(N_z):
            off = j * N_s
            y_cell = y_safe[off:off + N_s].copy()
            np.clip(y_cell, trace, max_conc, out=y_cell)
            R0 = self.chem.compute_rates_numba(y_cell)

            for i in range(N_s):
                h_i = eps_sqrt * max(abs(y_cell[i]), 1e-10)
                y_cell[i] += h_i
                Rp = self.chem.compute_rates_numba(y_cell)
                y_cell[i] -= h_i

                dR = (Rp - R0) / h_i
                for r in range(N_s):
                    if dR[r] != 0.0:
                        ab[lu + r - i, off + i] -= dt * dR[r]

        if N_z >= 2:
            y_2d = y_safe.reshape(N_z, N_s)
            y_2d_c = np.maximum(y_2d, trace)

            if E_half is None:
                rho = self._compute_charge_density(y_2d_c)
                E_half = self._solve_poisson_1d(rho)

            D = self.D_species
            inv_h = self.inv_h_faces
            inv_dz = self.inv_dz_cells
            V_T = self._V_T
            H_idx = self._h_plus_idx

            Z_eff = np.zeros((N_z, N_s))
            for sp_idx, Z in self._direct_charge.items():
                Z_eff[:, sp_idx] = Z
            if H_idx >= 0:
                Hp = np.maximum(y_2d_c[:, H_idx], 1e-14)
                for ti, Za, Zb, Ka in self._total_charge_info:
                    fb = Ka / (Hp + Ka)
                    Z_eff[:, ti] = Za * (1.0 - fb) + Zb * fb

            Zh = 0.5 * (Z_eff[:-1, :] + Z_eff[1:, :])
            h_faces = self.h_faces
            alpha = -Zh * (E_half[:, np.newaxis]
                           * h_faces[:, np.newaxis] / V_T)
            np.clip(alpha, -500.0, 500.0, out=alpha)
            Bp = self._bernoulli(alpha)
            Bm = self._bernoulli(-alpha)

            for f in range(N_z - 1):
                j = f
                jp = f + 1
                for i in range(N_s):
                    cp = D[i] * inv_h[f] * Bp[f, i]
                    cm = D[i] * inv_h[f] * Bm[f, i]

                    col_j = j * N_s + i
                    col_jp = jp * N_s + i

                    ab[lu, col_j] -= dt * (-cp * inv_dz[j])
                    ab[lu - N_s, col_jp] -= dt * (cm * inv_dz[j])
                    ab[lu, col_jp] -= dt * (-cm * inv_dz[jp])
                    ab[lu + N_s, col_j] -= dt * (cp * inv_dz[jp])

        hp = self._h_plus_idx
        h_surface = max(y_safe[hp], 1e-14) if hp >= 0 else 1e-7
        for aq_idx, k_g_val, gas_sp, H, Ka in self._interface_species:
            f_mol = h_surface / (h_surface + Ka) if Ka is not None else 1.0
            ab[lu, aq_idx] -= dt * (-k_g_val * self._inv_dz_0 * f_mol)

        return ab, lu

    def _solve_custom_implicit(
        self,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray],
        y0: np.ndarray,
        verbose: bool,
    ) -> dict:
        """
        Backward Euler + Gummel (Poisson) + Newton.

        Each timestep: freeze E via Poisson → Newton solves
        transport+chemistry with frozen E → check Gummel convergence.
        Jacobian: per-cell chemistry FD + analytical SG transport.
        Adaptive Δt based on Newton convergence speed.
        """
        t_start, t_end = t_span

        dt = ODE_CONFIG.dt_init
        dt_min = ODE_CONFIG.dt_min
        dt_max = ODE_CONFIG.dt_max
        max_newton = ODE_CONFIG.newton_maxiter
        max_gummel = ODE_CONFIG.gummel_maxiter
        gummel_tol = ODE_CONFIG.gummel_tol
        rtol = ODE_CONFIG.rtol
        atol = ODE_CONFIG.atol
        trace = DEFAULTS.trace_concentration
        max_conc = ODE_CONFIG.max_concentration
        N_s = self.N_s
        N_z = self.N_z
        oh = self._oh_minus_idx
        cl = self._cl_minus_idx
        hp = self._h_plus_idx

        y = y0.copy()
        t = t_start

        t_eval_list = np.sort(t_eval) if t_eval is not None else None
        eval_ptr = 0
        collected_t = []
        collected_y = []

        if t_eval_list is not None:
            while (eval_ptr < len(t_eval_list)
                   and abs(t - t_eval_list[eval_ptr]) < 1e-10):
                collected_t.append(t)
                collected_y.append(y.copy())
                eval_ptr += 1

        nfev = 0
        njev = 0
        steps = 0
        rejects = 0
        consec_rejects = 0  # consecutive rejections (reset on success)

        if verbose:
            print(f"  Custom implicit: Gummel({max_gummel}) × "
                  f"Newton({max_newton}), "
                  f"dt0={dt:.1f}s, range=[{dt_min:.0e},{dt_max:.0f}]s")

        while t < t_end - 1e-10 and (steps + rejects) < 5000:
            dt_step = min(dt, t_end - t)
            if t_eval_list is not None and eval_ptr < len(t_eval_list):
                dt_to_eval = t_eval_list[eval_ptr] - t
                if 1e-10 < dt_to_eval < dt_step:
                    dt_step = dt_to_eval
            if dt_step < 1e-12:
                break

            t_new = t + dt_step
            y_old = y.copy()
            y_work = y.copy()

            step_ok = False
            n_newton_last = 0
            E_final = np.zeros(max(N_z - 1, 0))

            for gi in range(max_gummel):
                if self._poisson_enabled:
                    y_2d = np.maximum(y_work.reshape(N_z, N_s), trace)
                    rho = self._compute_charge_density(y_2d)
                    E_half = self._solve_poisson_1d(rho)
                else:
                    E_half = np.zeros(max(N_z - 1, 0))

                newton_ok = False
                for ni in range(max_newton):
                    rhs_val = self._rhs_fixed_E(t_new, y_work, E_half)
                    nfev += 1

                    F = y_work - y_old - dt_step * rhs_val

                    if oh >= 0:
                        oh_indices = np.arange(oh, self.N_total, N_s)
                        F[oh_indices] = 0.0
                    if cl >= 0:
                        cl_indices = np.arange(cl, self.N_total, N_s)
                        F[cl_indices] = 0.0

                    w = rtol * np.abs(y_work) + atol
                    Fnorm = np.sqrt(np.mean((F / w) ** 2))

                    if Fnorm < 1.0:
                        newton_ok = True
                        n_newton_last = ni + 1
                        break

                    if ni == max_newton - 1:
                        break

                    ab, lu = self._compute_jac_banded(
                        t_new, y_work, dt_step, E_half,
                    )
                    njev += 1

                    # Log-transform: column-scale J by C_j
                    # Solves for δu where δC = C * δu
                    C_scale = np.maximum(np.abs(y_work), trace)
                    bw = lu
                    for col in range(self.N_total):
                        ab[:, col] *= C_scale[col]

                    if oh >= 0:
                        for j in range(N_z):
                            idx = j * N_s + oh
                            ab[:, idx] = 0.0
                            ab[bw, idx] = 1.0
                    if cl >= 0:
                        for j in range(N_z):
                            idx = j * N_s + cl
                            ab[:, idx] = 0.0
                            ab[bw, idx] = 1.0

                    try:
                        du = solve_banded((bw, bw), ab, -F)
                    except Exception:
                        break

                    if not np.all(np.isfinite(du)):
                        break

                    np.clip(du, -5.0, 5.0, out=du)
                    y_work += C_scale * du
                    np.clip(y_work, trace, max_conc, out=y_work)

                    if hp >= 0:
                        hp_indices = np.arange(hp, self.N_total, N_s)
                        y_work[hp_indices] = np.maximum(
                            y_work[hp_indices], 1e-14)

                if not newton_ok:
                    break

                if not self._poisson_enabled:
                    step_ok = True
                    break

                y_2d_new = np.maximum(
                    y_work.reshape(N_z, N_s), trace)
                rho_new = self._compute_charge_density(y_2d_new)
                E_new = self._solve_poisson_1d(rho_new)
                E_final = E_new

                if len(E_half) > 0:
                    E_scale = np.max(np.abs(E_new)) + 1e-10
                    dE_rel = np.max(np.abs(E_new - E_half)) / E_scale
                else:
                    dE_rel = 0.0

                if dE_rel < gummel_tol:
                    step_ok = True
                    break

                if gi == max_gummel - 1:
                    step_ok = True

            if step_ok:
                y = y_work
                # Enforce electroneutrality: solve H+ from charge balance
                self._enforce_electroneutrality(y)
                t = t_new
                steps += 1
                consec_rejects = 0  # reset on success

                if t_eval_list is not None:
                    while (eval_ptr < len(t_eval_list)
                           and abs(t - t_eval_list[eval_ptr]) < 1e-10):
                        collected_t.append(t)
                        collected_y.append(y.copy())
                        eval_ptr += 1

                if n_newton_last <= 3:
                    dt = min(dt * 1.5, dt_max)
                elif n_newton_last >= 7:
                    dt = max(dt * 0.5, dt_min)

                if verbose and (steps % 50 == 0
                                or t >= t_end - 1e-10):
                    if hp >= 0:
                        pH_ = -np.log10(max(
                            np.mean(y.reshape(N_z, N_s)[:, hp]),
                            1e-14))
                    else:
                        pH_ = 7.0
                    Em = (np.max(np.abs(E_final))
                          if len(E_final) > 0 else 0.0)
                    print(f"    t={t:.1f}s  pH={pH_:.3f}  "
                          f"|E|={Em:.2e}V/m  dt={dt:.3f}s  "
                          f"steps={steps}  rej={rejects}")
            else:
                y = y_old.copy()
                dt = max(dt * 0.25, dt_min)
                rejects += 1
                consec_rejects += 1
                if consec_rejects > 200:
                    if verbose:
                        print(f"    [ABORT] {consec_rejects} consecutive "
                              f"rejections at t={t:.1f}s (total {rejects})")
                    break

        ok = t >= t_end - 1e-10
        y_final = y.reshape(N_z, N_s)
        avg = self._compute_spatial_average(y_final)
        sfc = self._extract_cell(y_final, 0)

        if verbose:
            print(f"  Total: {steps} steps, {rejects} rejected, "
                  f"nfev={nfev}, njev={njev}")

        return {
            'success': ok,
            'sol': None,
            'y_final': y_final,
            'spatial_avg': avg,
            'surface': sfc,
            'pH_avg': -np.log10(max(avg.get('H+', 1e-7), 1e-14)),
            'pH_surface': -np.log10(max(sfc.get('H+', 1e-7), 1e-14)),
            'wall_time': None,
            'nfev': nfev,
            'njev': njev,
            'message': '' if ok else f'Failed at t={t:.1f}s',
        }

    def _solve_strang_split(
        self, t_span, t_eval, y0, verbose, dt_split=1.0,
    ) -> dict:
        """Strang splitting: diffusion(dt/2) → chemistry(dt) → diffusion(dt/2)."""
        from scipy.integrate import solve_ivp as _sivp
        import scipy.integrate._ivp.common as _com
        for _fn in ('_dense_num_jac', '_sparse_num_jac'):
            _orig = getattr(_com, _fn, None)
            if _orig is None:
                continue
            def _make_patch(orig):
                def _patched(*args):
                    if len(args) >= 6:
                        np.clip(args[5], None, 1e38, out=args[5])
                    r = orig(*args)
                    if isinstance(r, tuple) and len(r) >= 2:
                        np.clip(r[1], None, 1e38, out=r[1])
                    return r
                return _patched
            setattr(_com, _fn, _make_patch(_orig))

        t_start, t_end = t_span
        N_z, N_s = self.N_z, self.N_s
        trace = DEFAULTS.trace_concentration
        max_conc = ODE_CONFIG.max_concentration
        oh = self._oh_minus_idx
        cl = self._cl_minus_idx
        hp = self._h_plus_idx

        # QSSA for saline fast intermediates (HOCl⁻, Cl₂⁻, Cl, HOClH)
        qssa_on = hasattr(self.chem, '_qssa') and self.chem.saline_mode
        if qssa_on:
            idx_hocl_m = self.chem._qssa['idx_HOCl-']
            idx_cl2_m  = self.chem._qssa['idx_Cl2-']
            idx_cl_rad = self.chem._qssa['idx_Cl']
            idx_hoclh  = self.chem._qssa['idx_HOClH']

        y = y0.copy().reshape(N_z, N_s)
        t = t_start
        nfev = 0
        step_count = 0

        # Cl atom conservation projection (Sturm & Silva 2024):
        # After BDF, project Cl atom error onto Cl⁻ (dominant pool).
        # Cl atoms = sum_i (n_Cl_i * C_i) must be conserved per cell.
        _cl_cons_idx = []  # list of (species_idx, n_Cl_atoms)
        if qssa_on:
            _cl_atom_count = {
                'Cl-': 1, 'HOCl-': 1, 'Cl2-': 2, 'Cl': 1, 'HOClH': 1,
                'Cl2': 2, 'HCl': 1, 'HClO_total': 1, 'HClO2_total': 1,
                'ClO': 1, 'ClO2': 1, 'ClO3': 1, 'ClNO2': 1,
                'Cl3-': 3, 'ClO3-': 1, 'ClO4-': 1,
                'Cl2O': 2, 'Cl2O2': 2, 'Cl2O3': 2,
                'Cl2O4': 2, 'Cl2O5': 2, 'Cl2O6': 2,
            }
            for sp_name, n_cl in _cl_atom_count.items():
                idx = self.chem.species_idx.get(sp_name, -1)
                if idx >= 0:
                    _cl_cons_idx.append((idx, n_cl))

        t_eval_sorted = np.sort(t_eval) if t_eval is not None else None
        eval_ptr = 0
        collected_t, collected_y = [], []

        # QSSA species: algebraic, determined by local chemistry only — skip diffusion
        qssa_skip = set()
        if qssa_on:
            qssa_skip = {idx_hocl_m, idx_cl2_m, idx_cl_rad, idx_hoclh}

        # Diffusion: CN tridiagonal per species
        def _diffusion_step(C, dt_d):
            if N_z <= 1 or dt_d <= 0:
                return C
            dz = self.dz_cells
            h = self.h_faces
            inv_h = self.inv_h_faces
            inv_dz = self.inv_dz_cells
            for s in range(N_s):
                if s == oh or s == cl or s in qssa_skip:
                    continue
                D = self.D_species[s]
                c = C[:, s].copy()
                # CN: (I - 0.5*dt*L) c_new = (I + 0.5*dt*L) c_old + dt*BC
                # Tridiagonal: a_j c_{j-1} + b_j c_j + c_j c_{j+1} = rhs_j
                alpha = 0.5 * dt_d * D
                a = np.zeros(N_z)
                b = np.ones(N_z)
                cc = np.zeros(N_z)
                rhs = c.copy()
                for j in range(1, N_z - 1):
                    coeff_l = alpha * inv_h[j-1] * inv_dz[j]
                    coeff_r = alpha * inv_h[j] * inv_dz[j]
                    a[j] = -coeff_l
                    cc[j] = -coeff_r
                    b[j] = 1.0 + coeff_l + coeff_r
                    rhs[j] = c[j] + coeff_l * (c[j-1] - c[j]) + coeff_r * (c[j+1] - c[j])
                coeff_r0 = alpha * inv_h[0] * inv_dz[0] if N_z > 1 else 0
                b[0] = 1.0 + coeff_r0
                cc[0] = -coeff_r0
                rhs[0] = c[0] + coeff_r0 * (c[1] - c[0]) if N_z > 1 else c[0]
                # BC at z=0: no mass transfer here — handled in chemistry step
                # to couple dissolution with reactions implicitly via BDF.
                # BC z=L: no flux
                j = N_z - 1
                if N_z > 1:
                    coeff_l_last = alpha * inv_h[N_z-2] * inv_dz[N_z-1]
                    a[N_z-1] = -coeff_l_last
                    b[N_z-1] = 1.0 + coeff_l_last
                    rhs[N_z-1] = c[N_z-1] + coeff_l_last * (c[N_z-2] - c[N_z-1])
                # Thomas algorithm
                C[:, s] = _thomas(a, b, cc, rhs)
            np.clip(C, trace, max_conc, out=C)
            return C

        def _thomas(a, b, c, d):
            n = len(d)
            c_ = np.zeros(n)
            d_ = np.zeros(n)
            c_[0] = c[0] / b[0]
            d_[0] = d[0] / b[0]
            for i in range(1, n):
                m = a[i] / (b[i] - a[i] * c_[i-1])
                c_[i] = c[i] / (b[i] - a[i] * c_[i-1])
                d_[i] = (d[i] - a[i] * d_[i-1]) / (b[i] - a[i] * c_[i-1])
            x = np.zeros(n)
            x[n-1] = d_[n-1]
            for i in range(n-2, -1, -1):
                x[i] = d_[i] - c_[i] * x[i+1]
            return x

        # Chemistry RHS for a single cell.
        # OH⁻ is algebraic (Kw/H⁺), so freeze it.
        # Cl⁻ is NOT frozen: 4-species QSSA handles all fast Cl intermediates,
        # so remaining Cl⁻ dynamics are slow and safe for BDF.
        def _chem_rhs(t_local, c_cell):
            c_clipped = np.maximum(c_cell, trace)
            rates = self.chem.compute_rates_numba(c_clipped)
            if oh >= 0:
                rates[oh] = 0.0
            return rates

        # Surface cell (j=0) RHS: chemistry + mass transfer source term.
        # Couples gas-liquid transfer implicitly with BDF chemistry,
        # preventing operator-splitting artifact where dissolved gas
        # is fully consumed before next mass transfer update.
        def _chem_rhs_surface(t_local, c_cell):
            c_clipped = np.maximum(c_cell, trace)
            rates = self.chem.compute_rates_numba(c_clipped)
            if oh >= 0:
                rates[oh] = 0.0
            # Mass transfer: k_L/dz * (C_eq - c_eff) for each interface species
            # c_eff = f_mol × c_total for dissociating species (only molecular form transfers)
            t_idx = max(0, min(int((t + t_local) / self._dt_gas), self._n_times - 1))
            h_surf = c_clipped[hp] if hp >= 0 else 1e-7
            for aq_idx, k_g_val, gas_sp, H_val, Ka in self._interface_species:
                C_eq = self._get_C_eq_fast(gas_sp, t_idx)
                c0 = c_clipped[aq_idx]
                c_eff = c0 * h_surf / (h_surf + Ka) if Ka is not None else c0
                rates[aq_idx] += self._inv_dz_0 * k_g_val * (C_eq - c_eff)
            return rates

        n_macro = max(1, int((t_end - t_start) / dt_split))
        if verbose:
            print(f"  Strang split: {n_macro} macro steps, dt={dt_split:.1f}s, "
                  f"{N_z} cells × {N_s} species")

        for k in range(n_macro):
            t_k = t_start + k * dt_split
            t_k1 = min(t_k + dt_split, t_end)
            dt_k = t_k1 - t_k
            t = t_k

            # 1. Diffusion half-step
            y = _diffusion_step(y, dt_k / 2.0)

            # 2. Chemistry full-step (per cell)
            import time as _time
            import sys as _sys
            # Save pre-chemistry state for Cl conservation projection
            if qssa_on and _cl_cons_idx:
                c_before_chem = y.copy()
            # Per-species absolute tolerance: relax all 4 QSSA species
            _atol = np.full(N_s, 1e-8)
            if qssa_on:
                _atol[idx_hocl_m] = 1.0   # algebraic, kernel overwrites
                _atol[idx_cl2_m] = 1.0
                _atol[idx_cl_rad] = 1.0
                _atol[idx_hoclh] = 1.0
            _chem_t0 = _time.time()
            _slow_cells = []
            for j in range(N_z):
                c_cell = y[j, :].copy()
                _cell_t0 = _time.time()
                try:
                    _rhs_fn = _chem_rhs_surface if j == 0 else _chem_rhs
                    sol = _sivp(_rhs_fn, [0, dt_k], c_cell,
                                method='BDF', rtol=1e-4, atol=_atol,
                                max_step=dt_k)
                    if sol.success:
                        y[j, :] = np.maximum(sol.y[:, -1], trace)
                        nfev += sol.nfev
                except Exception:
                    pass
                _cell_dt = _time.time() - _cell_t0
                if _cell_dt > 2.0:
                    _slow_cells.append((j, _cell_dt, sol.nfev if sol.success else -1))
                # Progress every 20 cells or if a cell is slow
                if verbose and (j % 20 == 0 or _cell_dt > 5.0):
                    _nf = sol.nfev if sol.success else -1
                    print(f"    Step {k+1}, cell {j}/{N_z}: "
                          f"{_cell_dt:.1f}s, nfev={_nf}")
                    _sys.stdout.flush()
            _chem_wall = _time.time() - _chem_t0
            if verbose:
                print(f"    Chem step {k+1} total: {_chem_wall:.1f}s, "
                      f"{len(_slow_cells)} slow cells (>2s)")
                for _j, _dt, _nf in _slow_cells[:5]:
                    print(f"      cell {_j}: {_dt:.1f}s, nfev={_nf}")
                _sys.stdout.flush()

            # Cl atom conservation projection (per cell):
            # BDF may violate Cl conservation due to catastrophic cancellation
            # in fast reversible reactions. Project error onto Cl⁻.
            if qssa_on and _cl_cons_idx and cl >= 0:
                for jj in range(N_z):
                    cl_after = 0.0
                    for (si, n_cl) in _cl_cons_idx:
                        cl_after += n_cl * y[jj, si]
                    cl_before = 0.0
                    for (si, n_cl) in _cl_cons_idx:
                        cl_before += n_cl * c_before_chem[jj, si]
                    err = cl_after - cl_before
                    y[jj, cl] -= err  # project onto Cl⁻
                    if y[jj, cl] < trace:
                        y[jj, cl] = trace

            # Reset QSSA species to algebraic values before diffusion.
            # BDF evolves them with small residual dydt (for mass conservation),
            # but their physical concentrations are the QSSA steady-state values.
            if qssa_on:
                for jj in range(N_z):
                    self.chem.apply_qssa(y[jj, :])

            # 3. Diffusion half-step
            t = t_k1
            y = _diffusion_step(y, dt_k / 2.0)

            # Electroneutrality enforcement
            y_flat = y.ravel()
            self._enforce_electroneutrality(y_flat)
            y = y_flat.reshape(N_z, N_s)

            step_count += 1

            # Collect t_eval
            if t_eval_sorted is not None:
                while (eval_ptr < len(t_eval_sorted)
                       and abs(t - t_eval_sorted[eval_ptr]) < 1e-10):
                    collected_t.append(t)
                    collected_y.append(y.ravel().copy())
                    eval_ptr += 1

            if verbose and ((k + 1) % 10 == 0 or k == n_macro - 1):
                if hp >= 0:
                    h_avg = np.mean(y[:, hp])
                    pH_k = -np.log10(max(h_avg, 1e-14))
                else:
                    pH_k = 7.0
                print(f"    Step {k+1}/{n_macro}: t={t:.0f}s, "
                      f"pH={pH_k:.3f}, nfev={nfev}")

        y_final = y
        avg = self._compute_spatial_average(y_final)
        sfc = self._extract_cell(y_final, 0)
        if hp >= 0:
            pH_avg = -np.log10(max(avg.get('H+', 1e-7), 1e-14))
        else:
            pH_avg = 7.0

        return {
            'success': True,
            'y_final': y.ravel(),
            'spatial_avg': avg,
            'surface': sfc,
            'pH_avg': pH_avg,
            't_eval': np.array(collected_t) if collected_t else np.array([t]),
            'y_eval': collected_y,
            'nfev': nfev,
            'njev': 0,
            'message': '',
        }

    def _solve_bdf_with_electroneutrality(
        self,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray],
        y0: np.ndarray,
        verbose: bool,
        dt_enforce: Optional[float] = 1.0,
    ) -> dict:
        t_start, t_end = t_span

        # dt_enforce=None → single BDF call (no macro-step restart)
        if dt_enforce is None or dt_enforce <= 0:
            dt_enforce = t_end - t_start

        t_macro = list(np.arange(t_start, t_end, dt_enforce))
        if not t_macro or t_macro[-1] < t_end - 1e-10:
            t_macro.append(t_end)
        else:
            t_macro[-1] = t_end

        n_macro = len(t_macro) - 1

        if verbose:
            print(f"  BDF + electroneutrality: {n_macro} macro steps, "
                  f"dt_enforce={dt_enforce:.1f}s")

        self._E_half_frozen = np.zeros(max(self.N_z - 1, 0))

        y = y0.copy()
        success = True
        total_nfev = 0
        total_njev = 0
        fail_message = ''

        collected_t = []
        collected_y = []
        collected_sol = []  # dense output interpolants

        for k in range(n_macro):
            t_k = t_macro[k]
            t_k1 = t_macro[k + 1]

            t_eval_k = None
            if t_eval is not None:
                if k == 0:
                    mask = (t_eval >= t_k) & (t_eval <= t_k1)
                else:
                    mask = (t_eval > t_k) & (t_eval <= t_k1)
                if np.any(mask):
                    t_eval_k = t_eval[mask]

            sol_k = solve_ivp(
                self.rhs,
                (t_k, t_k1),
                y,
                method='BDF',
                t_eval=t_eval_k,
                rtol=ODE_CONFIG.rtol,
                atol=ODE_CONFIG.atol,
                max_step=ODE_CONFIG.max_step,
                jac=lambda t, yy: self._compute_jac_fd(t, yy),
                vectorized=False,
                dense_output=True,
            )

            total_nfev += sol_k.nfev
            total_njev += sol_k.njev

            if not sol_k.success:
                success = False
                fail_message = f"Macro step {k} ({t_k:.0f}-{t_k1:.0f}s): {sol_k.message}"
                if verbose:
                    print(f"    [FAIL] {fail_message}")
                break

            if t_eval_k is not None:
                for i in range(len(sol_k.t)):
                    collected_t.append(sol_k.t[i])
                    collected_y.append(sol_k.y[:, i].copy())

            if sol_k.sol is not None:
                collected_sol.append(sol_k.sol)

            y_before = y.copy()
            y = sol_k.y[:, -1].copy()
            self._enforce_electroneutrality(y)
            if self.saline_mode:
                self._enforce_cl_conservation(y, y_before)
            np.clip(y, DEFAULTS.trace_concentration,
                    ODE_CONFIG.max_concentration, out=y)

            if verbose and ((k + 1) % 10 == 0 or k == n_macro - 1 or k == 0):
                y_2d_k = y.reshape(self.N_z, self.N_s)
                h_idx = self._h_plus_idx
                if h_idx >= 0:
                    h_avg = np.mean(y_2d_k[:, h_idx])
                    pH_k = -np.log10(max(h_avg, 1e-14))
                else:
                    pH_k = 7.0
                print(f"    Step {k+1}/{n_macro}: t={t_k1:.0f}s, "
                      f"pH={pH_k:.3f}, nfev={sol_k.nfev}")

        self._E_half_frozen = None

        y_final = y.reshape(self.N_z, self.N_s)
        spatial_avg = self._compute_spatial_average(y_final)
        surface = self._extract_cell(y_final, 0)
        pH_avg = -np.log10(max(spatial_avg.get('H+', 1e-7), 1e-14))
        pH_surface = -np.log10(max(surface.get('H+', 1e-7), 1e-14))

        return {
            'success': success,
            'sol': None,
            'y_final': y_final,
            'spatial_avg': spatial_avg,
            'surface': surface,
            'pH_avg': pH_avg,
            'pH_surface': pH_surface,
            'wall_time': None,
            'nfev': total_nfev,
            'njev': total_njev,
            'message': fail_message,
            't_eval': np.array(collected_t) if collected_t else np.array([t_span[1]]),
            'y_eval': collected_y,
            'dense_output': collected_sol,
        }

    def _solve_operator_split(
        self,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray],
        y0: np.ndarray,
        verbose: bool,
        dt_poisson: float = 60.0,
    ) -> dict:
        """
        Lie operator splitting: Poisson (explicit) + Chemistry+SG (implicit BDF).

        At each macro step:
          1. Solve Poisson from current state → freeze E-field
          2. BDF for chemistry + SG transport with frozen E for Δt

        Advantages over monolithic approach:
          - Poisson never enters BDF RHS → no global Jacobian coupling
          - Block-tridiagonal sparsity is EXACT → efficient scipy num_jac
          - No trace-species factor overflow (Poisson outside solver)

        Args:
            dt_poisson: Macro timestep for Poisson updates [s].
        """
        t_start, t_end = t_span

        # Build macro step boundaries
        t_macro = list(np.arange(t_start, t_end, dt_poisson))
        if not t_macro or t_macro[-1] < t_end - 1e-10:
            t_macro.append(t_end)
        else:
            t_macro[-1] = t_end  # avoid float rounding overshoot

        n_macro = len(t_macro) - 1

        if verbose:
            print(f"  Operator splitting: {n_macro} macro steps, "
                  f"Δt_Poisson={dt_poisson:.0f}s")

        y = y0.copy()
        success = True
        total_nfev = 0
        total_njev = 0
        fail_message = ''

        # Collect solution at t_eval points
        collected_t = []
        collected_y = []

        for k in range(n_macro):
            t_k = t_macro[k]
            t_k1 = t_macro[k + 1]

            # --- Step 1: Poisson solve → freeze E-field ---
            y_2d = np.maximum(
                y.reshape(self.N_z, self.N_s),
                DEFAULTS.trace_concentration,
            )
            rho = self._compute_charge_density(y_2d)
            E_half = self._solve_poisson_1d(rho)
            self._E_half_frozen = E_half

            # --- Step 2: Determine t_eval points in this interval ---
            t_eval_k = None
            if t_eval is not None:
                if k == 0:
                    mask = (t_eval >= t_k) & (t_eval <= t_k1)
                else:
                    mask = (t_eval > t_k) & (t_eval <= t_k1)
                if np.any(mask):
                    t_eval_k = t_eval[mask]

            # --- Step 3: BDF with frozen E-field (block-tridiagonal Jacobian) ---
            sol_k = solve_ivp(
                self.rhs,
                (t_k, t_k1),
                y,
                method=ODE_CONFIG.method,
                t_eval=t_eval_k,
                rtol=ODE_CONFIG.rtol,
                atol=ODE_CONFIG.atol,
                max_step=ODE_CONFIG.max_step,
                jac_sparsity=self.jac_sparsity,
                vectorized=False,
            )

            total_nfev += sol_k.nfev
            total_njev += sol_k.njev

            if not sol_k.success:
                success = False
                fail_message = f"Macro step {k} ({t_k:.0f}-{t_k1:.0f}s): {sol_k.message}"
                if verbose:
                    print(f"    [FAIL] {fail_message}")
                break

            # Collect t_eval results
            if t_eval_k is not None:
                for i in range(len(sol_k.t)):
                    collected_t.append(sol_k.t[i])
                    collected_y.append(sol_k.y[:, i].copy())

            # Update state for next macro step
            y = sol_k.y[:, -1].copy()

            if verbose:
                y_2d_k = y.reshape(self.N_z, self.N_s)
                h_idx = self._h_plus_idx
                if h_idx >= 0:
                    h_avg = np.mean(y_2d_k[:, h_idx])
                    pH_k = -np.log10(max(h_avg, 1e-14))
                else:
                    pH_k = 7.0
                E_max = np.max(np.abs(E_half)) if len(E_half) > 0 else 0
                print(f"    Step {k+1}/{n_macro}: "
                      f"t={t_k1:.0f}s, pH={pH_k:.3f}, "
                      f"|E|_max={E_max:.2e} V/m, "
                      f"nfev={sol_k.nfev}")

        # Clear frozen E-field
        self._E_half_frozen = None

        # Build result dict (same interface as monolithic solve)
        y_final = y.reshape(self.N_z, self.N_s)
        spatial_avg = self._compute_spatial_average(y_final)
        surface = self._extract_cell(y_final, 0)

        pH_avg = -np.log10(max(spatial_avg.get('H+', 1e-7), 1e-14))
        pH_surface = -np.log10(max(surface.get('H+', 1e-7), 1e-14))

        return {
            'success': success,
            'sol': None,  # no single ODE solution object in splitting mode
            'y_final': y_final,
            'spatial_avg': spatial_avg,
            'surface': surface,
            'pH_avg': pH_avg,
            'pH_surface': pH_surface,
            'wall_time': None,  # set by caller
            'nfev': total_nfev,
            'njev': total_njev,
            'message': fail_message,
        }

    def solve(
        self,
        t_span: Tuple[float, float] = (0, 360),
        t_eval: Optional[np.ndarray] = None,
        y0: Optional[np.ndarray] = None,
        verbose: bool = True,
        dt_poisson: Optional[float] = 60.0,
    ) -> dict:
        if y0 is None:
            y0 = self.build_initial_condition()

        if t_eval is None:
            t_eval = np.array([t_span[0], t_span[1]])

        if verbose:
            print(f"PDESolver1D: {self.N_s} species × {self.N_z} cells = "
                  f"{self.N_total} ODEs")
            dz_min_um = self.dz_cells[0] * 1e6
            dz_max_um = self.dz_cells[-1] * 1e6
            print(f"  Grid: L={self.L*1000:.1f}mm, N_z={self.N_z}, "
                  f"dz=[{dz_min_um:.4g}–{dz_max_um:.4g}]μm ({self._grid_type})")
            print(f"  delta_gas={self.delta_gas*1000:.1f}mm, "
                  f"delta_liq={self.delta_liq*1e6:.1f}um, "
                  f"eta={self.mass_transfer_eta:.4f}")
            print(f"  Time: {t_span[0]}-{t_span[1]}s, "
                  f"method={ODE_CONFIG.method}")
            poisson_mode = 'ENABLED (custom implicit)' if self._poisson_enabled else 'DISABLED'
            print(f"  Poisson: {poisson_mode}"
                  f" (ε_r={POISSON.epsilon_r}, V_T={self._V_T:.5f}V)")

        t0 = time.time()

        if self._use_strang:
            # Legacy Strang splitting path (for comparison)
            result = self._solve_strang_split(
                t_span, t_eval, y0, verbose, dt_split=dt_poisson,
            )
        elif self._poisson_enabled:
            if verbose:
                print(f"  Jacobian: grouped FD → banded "
                      f"({self._n_jac_groups} groups, bw={2*self.N_s-1})")

            result = self._solve_custom_implicit(
                t_span, t_eval, y0, verbose,
            )

        else:
            # ---- Monolithic BDF + electroneutrality (DIW/saline) ----
            # Disable QSSA in monolithic mode: BDF handles Cl stiffness implicitly.
            if not self._use_strang and self.saline_mode:
                self.chem.set_qssa_enabled(False)

            if verbose:
                print(f"  Jacobian: grouped FD ({self._n_jac_groups} color groups)")
                dt_label = f"{dt_poisson:.0f}s" if dt_poisson else "None (single BDF)"
                print(f"  Mode: monolithic BDF + electroneutrality "
                      f"(dt_enforce={dt_label})")

            result = self._solve_bdf_with_electroneutrality(
                t_span, t_eval, y0, verbose, dt_enforce=dt_poisson,
            )

            # Restore QSSA if it was disabled
            if self.saline_mode:
                self.chem.set_qssa_enabled(True)

        wall_time = time.time() - t0
        result['wall_time'] = wall_time

        if verbose and 'wall_time' in result:
            status = "SUCCESS" if result['success'] else f"FAILED ({result.get('message', '')})"
            print(f"  Solver: {status}")
            print(f"  Wall time: {wall_time:.1f}s ({wall_time/60:.1f}min), "
                  f"nfev={result.get('nfev', 0)}, njev={result.get('njev', 0)}")

        return result

    def _compute_spatial_average(self, y_2d: np.ndarray) -> Dict[str, float]:
        avg = {}
        mean_y = np.mean(y_2d, axis=0)

        for sp, idx in self.species_idx.items():
            avg[sp] = float(mean_y[idx])

        H_avg = max(avg.get('H+', 1e-7), 1e-14)
        for total_name, (acid, base, pKa) in self._acid_base_pairs.items():
            if total_name in avg:
                C_total = avg[total_name]
                Ka = 10.0 ** (-pKa)
                denom = H_avg + Ka
                avg[acid] = C_total * H_avg / denom
                avg[base] = C_total * Ka / denom
        avg['OH-'] = 1e-14 / H_avg
        return avg

    def _extract_cell(self, y_2d: np.ndarray, j: int) -> Dict[str, float]:
        cell = {}
        y_cell = y_2d[j, :]

        for sp, idx in self.species_idx.items():
            cell[sp] = float(y_cell[idx])

        H = max(cell.get('H+', 1e-7), 1e-14)
        for total_name, (acid, base, pKa) in self._acid_base_pairs.items():
            if total_name in cell:
                C_total = cell[total_name]
                Ka = 10.0 ** (-pKa)
                denom = H + Ka
                cell[acid] = C_total * H / denom
                cell[base] = C_total * Ka / denom
        cell['OH-'] = 1e-14 / H
        return cell
