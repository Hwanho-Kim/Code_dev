"""Mean-energy-indexed look-up table for electron-impact rate coefficients.

Architecture (GlobalKin-style):
  Layer 1 — BOLSIG+ transport:  ε̄ → {P_el/N, P_elas/N, P_inel/N}
  Layer 2 — Custom reactions:   ε̄ → {k_1, k_2, ..., k_21}  [m³/s]
            (Maxwellian EEDF × custom σ integration at each ε̄)

  All tables are indexed by mean electron energy ε̄ [eV] (= A1 from BOLSIG+).
  The ε̄ grid comes directly from the BOLSIG+ output — no resampling.

  Reverse lookup: ε̄ → E/N [Td] (monotonic, for diagnostics only).

Usage::

    from plasma0d_v2.boltzmann import MeanEnergyLUT
    from plasma0d_v2.bolsig_parser import parse_bolsig_file

    bolsig = parse_bolsig_file("input/Condition1_300K.txt")
    lut = MeanEnergyLUT()
    lut.load_cross_sections(xsec_dir, ei_reactions)
    lut.build(bolsig)

    # Query at a given mean energy
    k_conc, Te_eV = lut.get_rate_coefficients_conc(eps_mean_eV=3.0)
    transport = lut.get_transport(eps_mean_eV=3.0)
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from .constants import QE, ME, KB, NA, PI
from .bolsig_parser import BolsigData, EEDFData


# ---------------------------------------------------------------------------
# Cross section loader (unchanged from original)
# ---------------------------------------------------------------------------

class CrossSection:
    """Single cross-section dataset σ(ε)."""

    def __init__(self, name: str, filepath: str, threshold_eV: float = 0.0):
        self.name = name
        self.filepath = filepath
        self.threshold_eV = threshold_eV
        self.energy_eV: np.ndarray = np.array([])  # [eV]
        self.sigma_m2: np.ndarray = np.array([])    # [m²]

    def load(self):
        """Load two-column data: energy [eV]  sigma [m²].

        Supports both plain two-column files (with optional ``#`` comments)
        and LXCat header format where data sits between ``----`` delimiters.
        """
        with open(self.filepath, 'r') as fh:
            raw = fh.read()

        if '----' in raw:
            rows = self._parse_lxcat(raw)
        else:
            rows = np.loadtxt(self.filepath, dtype=float)

        self.energy_eV = rows[:, 0]
        self.sigma_m2 = rows[:, 1]
        print(f"    Loaded {self.name}: {len(self.energy_eV)} points, "
              f"ε=[{self.energy_eV[0]:.1f}, {self.energy_eV[-1]:.1f}] eV")

    @staticmethod
    def _parse_lxcat(raw: str) -> np.ndarray:
        """Extract numeric data between ``----`` delimiter lines."""
        lines = raw.splitlines()
        in_data = False
        rows = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('----'):
                if in_data:
                    break
                in_data = True
                continue
            if in_data and stripped:
                try:
                    tokens = stripped.split()
                    rows.append([float(tokens[0]), float(tokens[1])])
                except (ValueError, IndexError):
                    continue
        if not rows:
            raise ValueError("No numeric data found between ---- delimiters")
        return np.array(rows, dtype=np.float64)

    def interpolate(self, energy_eV: np.ndarray) -> np.ndarray:
        """Interpolate σ at given energies."""
        return np.interp(energy_eV, self.energy_eV, self.sigma_m2,
                         left=0.0, right=0.0)


# ---------------------------------------------------------------------------
# Transport data container (output of Layer 1 query)
# ---------------------------------------------------------------------------

class TransportData:
    """Transport coefficients at a single ε̄ point."""
    __slots__ = ('power_N', 'elastic_power_N', 'inelastic_power_N',
                 'growth_power_N', 'EN_Td', 'eps_mean_eV')

    def __init__(self):
        self.power_N: float = 0.0           # A20  [eV m³/s]
        self.elastic_power_N: float = 0.0   # A21  [eV m³/s]
        self.inelastic_power_N: float = 0.0 # A22  [eV m³/s]
        self.growth_power_N: float = 0.0    # A23  [eV m³/s]
        self.EN_Td: float = 0.0             # reverse-mapped E/N [Td]
        self.eps_mean_eV: float = 0.0       # the queried ε̄ [eV]


# ---------------------------------------------------------------------------
# Main LUT class
# ---------------------------------------------------------------------------

class MeanEnergyLUT:
    """Look-up table indexed by mean electron energy ε̄ [eV].

    Replaces the old E/N-indexed BoltzmannLUT.  Two layers:

    Layer 1 (transport)
        Interpolates BOLSIG+ transport coefficients A20, A21, A22, A23
        as functions of ε̄.  Also provides the reverse map ε̄ → E/N.

    Layer 2 (custom rate coefficients)
        For the 21 electron-impact reactions that have custom cross-section
        files, computes <σv> at each ε̄ grid point.  When BOLSIG+ EEDF data
        is provided, uses the full EEDF × σ integration; otherwise falls
        back to Maxwellian approximation.
    """

    def __init__(self):
        # Cross sections for custom reactions
        self.cross_sections: List[CrossSection] = []

        # ε̄ grid from BOLSIG+ (sorted ascending, unique)
        self._eps_grid: np.ndarray = np.array([])       # [eV]
        self._log_eps_grid: np.ndarray = np.array([])    # log10(ε̄)

        # Layer 1: transport tables (1-D, same length as _eps_grid)
        self._power_N: np.ndarray = np.array([])         # A20
        self._elastic_power_N: np.ndarray = np.array([]) # A21
        self._inelastic_power_N: np.ndarray = np.array([]) # A22
        self._growth_power_N: np.ndarray = np.array([])  # A23

        # Reverse map: ε̄ → E/N [Td]
        self._EN_Td: np.ndarray = np.array([])

        # Layer 2: custom rate coefficients  (n_eps, n_custom_rxns) [m³/s]
        self._k_table: np.ndarray = np.array([])
        self._log_k_table: np.ndarray = np.array([])     # log10 for interpolation

        # Meta
        self._tgas_K: float = 0.0
        self._built = False

    # ------------------------------------------------------------------
    # Cross section loading
    # ------------------------------------------------------------------

    def add_cross_section(self, name: str, filepath: str,
                          threshold_eV: float = 0.0):
        cs = CrossSection(name, filepath, threshold_eV)
        cs.load()
        self.cross_sections.append(cs)

    def load_cross_sections(self, xsec_dir: str, ei_reactions):
        """Load cross sections referenced by electron-impact reactions."""
        self._sigma_over_N_mask = []
        for rxn in ei_reactions:
            fpath = os.path.join(xsec_dir, rxn.cross_section_file)
            is_sigma_N = getattr(rxn, 'sigma_over_N', False)
            if os.path.exists(fpath):
                self.add_cross_section(
                    name=rxn.formula,
                    filepath=fpath,
                    threshold_eV=rxn.energy_loss_eV
                )
            else:
                print(f"  WARNING: cross section file not found: {fpath}")
                cs = CrossSection(rxn.formula, fpath, rxn.energy_loss_eV)
                cs.energy_eV = np.array([0, 100])
                cs.sigma_m2 = np.array([0, 0])
                self.cross_sections.append(cs)
            self._sigma_over_N_mask.append(is_sigma_N)
        self._sigma_over_N_mask = np.array(self._sigma_over_N_mask, dtype=bool)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, bolsig: BolsigData,
              eedf_data: Optional[EEDFData] = None):
        """Build ε̄-indexed LUT from parsed BOLSIG+ data.

        Parameters
        ----------
        bolsig : BolsigData
            Parsed BOLSIG+ output (from ``bolsig_parser.parse_bolsig_file``).
        eedf_data : EEDFData, optional
            Parsed BOLSIG+ EEDF output.  When provided, Layer 2 uses
            EEDF × σ integration instead of Maxwellian approximation.
            E/N grids must match 1:1 (same number of blocks in same order).
        """
        # --- Layer 1: extract ε̄ grid and transport from BOLSIG+ ----------
        eps = bolsig.mean_energy_eV.copy()
        self._tgas_K = bolsig.tgas_K

        # BOLSIG+ outputs are ordered by ascending E/N.
        # Mean energy ε̄ is monotonically increasing with E/N, so the
        # ε̄ grid is already sorted ascending.
        if not np.all(np.diff(eps) > 0):
            raise ValueError(
                "Mean energy (A1) is not strictly monotonically increasing. "
                "Cannot build ε̄-indexed LUT."
            )

        self._eps_grid = eps
        self._log_eps_grid = np.log10(eps)
        self._EN_Td = bolsig.EN_Td.copy()

        # Transport coefficients (same indexing as eps_grid)
        self._power_N = bolsig.power_N.copy()
        self._elastic_power_N = bolsig.elastic_power_N.copy()
        self._inelastic_power_N = bolsig.inelastic_power_N.copy()
        self._growth_power_N = bolsig.growth_power_N.copy()

        # --- Layer 2: compute custom rate coefficients at each ε̄ ---------
        n_eps = len(eps)
        n_cs = len(self.cross_sections)
        self._k_table = np.zeros((n_eps, n_cs))

        # Determine whether to use EEDF-based integration
        use_eedf = False
        if eedf_data is not None:
            if eedf_data.n_blocks == n_eps:
                # Verify E/N grids match (tolerance 0.1%)
                en_match = np.allclose(
                    eedf_data.EN_Td, bolsig.EN_Td, rtol=1e-3)
                if en_match:
                    use_eedf = True
                    print(f"    EEDF data matched: {n_eps} blocks, "
                          f"E/N grids consistent")
                else:
                    print(f"    WARNING: EEDF E/N grid mismatch — "
                          f"falling back to Maxwellian")
            else:
                print(f"    WARNING: EEDF has {eedf_data.n_blocks} blocks "
                      f"but BOLSIG+ has {n_eps} E/N points — "
                      f"falling back to Maxwellian")

        if n_cs > 0:
            eedf_label = "BOLSIG+ EEDF" if use_eedf else "Maxwellian"
            for i, eps_eV in enumerate(eps):
                for j, cs in enumerate(self.cross_sections):
                    if use_eedf:
                        assert eedf_data is not None  # narrowing for type checker
                        blk = eedf_data.blocks[i]
                        self._k_table[i, j] = self._eedf_rate(
                            cs, blk.energy_eV, blk.eedf)
                    else:
                        Te_eV = max((2.0 / 3.0) * eps_eV, 0.01)
                        self._k_table[i, j] = self._maxwellian_rate(cs, Te_eV)

            self._log_k_table = np.log10(
                np.maximum(self._k_table, 1e-50)
            )
            self._k_dead_mask = np.all(self._k_table < 1e-50, axis=0)

        self._built = True
        self._eedf_used = use_eedf

        eps_min, eps_max = eps[0], eps[-1]
        EN_min, EN_max = self._EN_Td[0], self._EN_Td[-1]
        eedf_label = "BOLSIG+ EEDF" if use_eedf else "Maxwellian"
        print(f"  MeanEnergyLUT built (Tgas={self._tgas_K:.0f} K):")
        print(f"    ε̄ range: [{eps_min:.4f}, {eps_max:.2f}] eV "
              f"({n_eps} points)")
        print(f"    E/N range: [{EN_min:.4f}, {EN_max:.1f}] Td")
        print(f"    Custom cross sections: {n_cs} reactions ({eedf_label})")

    # ------------------------------------------------------------------
    # Layer 2 helpers: rate coefficient integration
    # ------------------------------------------------------------------

    @staticmethod
    def _eedf_rate(cs: CrossSection, eedf_energy: np.ndarray,
                   eedf_F0: np.ndarray) -> float:
        """Compute rate coefficient using BOLSIG+ EEDF.

        k = sqrt(2e/m_e) * integral_0^inf  sigma(eps) * eps * F0(eps) deps

        Equivalently (matching _maxwellian_rate pattern):

        k = integral  sigma(eps) * v(eps) * F0(eps) * sqrt(eps) deps

        where F0 [eV^(-3/2)] is normalised: integral F0 * sqrt(eps) deps = 1.

        Parameters
        ----------
        cs : CrossSection
            Cross-section data (interpolated onto EEDF energy grid).
        eedf_energy : ndarray
            Energy grid from BOLSIG+ EEDF block [eV].
        eedf_F0 : ndarray
            EEDF values F0(eps) [eV^(-3/2)].

        Returns
        -------
        float
            Rate coefficient [m^3/s] (number-density basis).
        """
        if len(eedf_energy) < 2 or len(cs.energy_eV) < 2:
            return 0.0

        sigma = cs.interpolate(eedf_energy)                # sigma on EEDF grid
        v = np.sqrt(2.0 * eedf_energy * QE / ME)           # v(eps)
        f_eps = eedf_F0 * np.sqrt(eedf_energy)             # F0 * sqrt(eps)
        integrand = sigma * v * f_eps

        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        return max(float(_trapz(integrand, eedf_energy)), 0.0)

    @staticmethod
    def _maxwellian_rate(cs: CrossSection, Te_eV: float) -> float:
        """Compute Maxwellian-averaged rate coefficient <σv>.

        <σv> = ∫₀^∞ σ(ε) · v(ε) · f_M(ε) dε

        f_M(ε) = (2/√π) · Te^(-3/2) · √ε · exp(-ε/Te)
        v(ε) = √(2εe/m_e)
        """
        if len(cs.energy_eV) < 2:
            return 0.0

        e_max = min(cs.energy_eV[-1], 10 * Te_eV + cs.threshold_eV + 50)
        e_max = max(e_max, cs.threshold_eV + 5)
        n_int = 500
        eps = np.linspace(0.01, e_max, n_int)

        sigma = cs.interpolate(eps)
        v = np.sqrt(2.0 * eps * QE / ME)
        C = 2.0 / np.sqrt(PI) * Te_eV**(-1.5)
        f_eps = C * np.sqrt(eps) * np.exp(-eps / Te_eV)
        integrand = sigma * v * f_eps
        # np.trapezoid (numpy ≥ 2.0) or np.trapz (numpy < 2.0)
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        rate = _trapz(integrand, eps)
        return max(rate, 0.0)

    # ------------------------------------------------------------------
    # Query: rate coefficients
    # ------------------------------------------------------------------

    def _interp_eps(self, eps_mean_eV: float) -> float:
        """Clamp and return log10(eps) for interpolation."""
        eps_clamped = np.clip(eps_mean_eV,
                              self._eps_grid[0], self._eps_grid[-1])
        return np.log10(eps_clamped)

    def get_rate_coefficients(self, eps_mean_eV: float) -> Tuple[np.ndarray, float]:
        """Interpolate custom rate coefficients at given ε̄.

        Vectorized: single searchsorted + one array operation for all
        cross-section columns, replacing the per-column np.interp loop.
        """
        if not self._built:
            raise RuntimeError("LUT not built yet. Call build() first.")

        Te_eV = (2.0 / 3.0) * eps_mean_eV

        n_cs = self._k_table.shape[1] if self._k_table.ndim == 2 else 0
        if n_cs == 0:
            return np.array([]), Te_eV

        log_eps = self._interp_eps(eps_mean_eV)
        grid = self._log_eps_grid
        idx = min(max(int(np.searchsorted(grid, log_eps)) - 1, 0),
                  len(grid) - 2)
        denom = grid[idx + 1] - grid[idx]
        frac = (log_eps - grid[idx]) / denom if abs(denom) > 1e-30 else 0.0

        log_k_interp = (self._log_k_table[idx, :]
                        + frac * (self._log_k_table[idx + 1, :]
                                  - self._log_k_table[idx, :]))
        k_arr = 10.0 ** log_k_interp
        k_arr[self._k_dead_mask] = 0.0
        return k_arr, Te_eV

    def get_rate_coefficients_maxwellian(self, Te_eV: float) -> np.ndarray:
        """Compute rate coefficients from Maxwellian EEDF at given Te.

        Used as fallback when eps_mean is below the LUT range.
        Uses cached thermal rates for speed (Te ≈ thermal in this regime).
        """
        if not hasattr(self, '_k_thermal_cache'):
            Te_th = max((2.0 / 3.0) * self._eps_grid[0], 0.01)
            n_cs = len(self.cross_sections)
            self._k_thermal_cache = np.zeros(n_cs)
            for j, cs in enumerate(self.cross_sections):
                self._k_thermal_cache[j] = self._maxwellian_rate(cs, Te_th)
        return self._k_thermal_cache.copy()

    def get_rate_coefficients_conc(self, eps_mean_eV: float,
                                    N_gas_cm3: float = 0.0,
                                    fallback_maxwellian: bool = False) -> Tuple[np.ndarray, float]:
        """Get rate coefficients in concentration basis [m³/(mol·s)].

        If fallback_maxwellian=True and eps_mean is below LUT range,
        compute rates from Maxwellian EEDF instead of returning the
        LUT-boundary-clamped values.
        """
        Te_eV = (2.0 / 3.0) * eps_mean_eV

        if fallback_maxwellian and eps_mean_eV < self._eps_grid[0]:
            k_dens = self.get_rate_coefficients_maxwellian(Te_eV)
        else:
            k_dens, Te_eV = self.get_rate_coefficients(eps_mean_eV)

        k_conc = k_dens * NA

        if N_gas_cm3 > 0 and hasattr(self, '_sigma_over_N_mask') and np.any(self._sigma_over_N_mask):
            k_conc[self._sigma_over_N_mask] *= N_gas_cm3

        return k_conc, Te_eV

    # ------------------------------------------------------------------
    # Query: rate coefficient derivatives (for analytical Jacobian)
    # ------------------------------------------------------------------

    def get_rate_derivatives_conc(self, eps_mean_eV: float) -> np.ndarray:
        """Compute dk_conc/dε̄ for all custom EI reactions.

        Uses log-log derivative:
            dk/dε̄ = k × slope / ε̄
        where slope = d(log10 k) / d(log10 ε̄) is the local piecewise-linear
        gradient in the log-log LUT.

        Parameters
        ----------
        eps_mean_eV : float
            Mean electron energy [eV].

        Returns
        -------
        dk_conc : ndarray, shape (n_cs,)
            dk/dε̄ in concentration basis [m³/(mol·s·eV)].
        """
        if not self._built:
            raise RuntimeError("LUT not built yet. Call build() first.")

        n_cs = self._k_table.shape[1] if self._k_table.ndim == 2 else 0
        dk_conc = np.zeros(n_cs)
        if n_cs == 0 or eps_mean_eV <= 0:
            return dk_conc

        log_eps = self._interp_eps(eps_mean_eV)
        eps_clamped = np.clip(eps_mean_eV,
                              self._eps_grid[0], self._eps_grid[-1])

        for j in range(n_cs):
            k_col = self._k_table[:, j]
            if np.all(k_col < 1e-50):
                continue

            log_k_col = self._log_k_table[:, j]

            # Interpolate k at this eps
            log_k_val = np.interp(log_eps, self._log_eps_grid, log_k_col)
            k_val = 10.0 ** log_k_val

            # Compute slope = d(log10 k) / d(log10 eps) via finite difference
            # on the piecewise-linear log-log table
            idx = np.searchsorted(self._log_eps_grid, log_eps) - 1
            idx = max(0, min(idx, len(self._log_eps_grid) - 2))
            d_log_eps = self._log_eps_grid[idx + 1] - self._log_eps_grid[idx]
            d_log_k = log_k_col[idx + 1] - log_k_col[idx]

            if abs(d_log_eps) > 1e-30:
                slope = d_log_k / d_log_eps
            else:
                slope = 0.0

            # dk/deps = k * slope / eps  (number density basis [m³/s/eV])
            # Convert to concentration basis: × NA
            dk_conc[j] = k_val * slope / eps_clamped * NA

        return dk_conc

    def get_inelastic_power_deriv(self, eps_mean_eV: float) -> float:
        """Compute dA22/dε̄ for the inelastic power coefficient.

        Same log-log derivative approach as get_transport_deriv (A21).
        """
        if not self._built or eps_mean_eV <= 0:
            return 0.0
        log_eps = self._interp_eps(eps_mean_eV)
        eps_clamped = np.clip(eps_mean_eV,
                              self._eps_grid[0], self._eps_grid[-1])
        A22 = self._interp_at(
            *self._bracket(log_eps), self._inelastic_power_N)
        if np.all(self._inelastic_power_N > 0):
            log_A22 = np.log10(self._inelastic_power_N)
            idx = max(0, min(int(np.searchsorted(self._log_eps_grid, log_eps)) - 1,
                             len(self._log_eps_grid) - 2))
            d_log_eps = self._log_eps_grid[idx + 1] - self._log_eps_grid[idx]
            d_log_A22 = log_A22[idx + 1] - log_A22[idx]
            slope = d_log_A22 / d_log_eps if abs(d_log_eps) > 1e-30 else 0.0
            return A22 * slope / eps_clamped
        else:
            idx = max(0, min(int(np.searchsorted(self._log_eps_grid, log_eps)) - 1,
                             len(self._log_eps_grid) - 2))
            d_eps = self._eps_grid[idx + 1] - self._eps_grid[idx]
            d_A22 = self._inelastic_power_N[idx + 1] - self._inelastic_power_N[idx]
            return d_A22 / d_eps if abs(d_eps) > 1e-30 else 0.0

    def _bracket(self, log_eps: float):
        """Return (frac, idx) for pre-computed bracket."""
        grid = self._log_eps_grid
        idx = min(max(int(np.searchsorted(grid, log_eps)) - 1, 0),
                  len(grid) - 2)
        denom = grid[idx + 1] - grid[idx]
        frac = (log_eps - grid[idx]) / denom if abs(denom) > 1e-30 else 0.0
        return frac, idx

    def get_transport_deriv(self, eps_mean_eV: float) -> float:
        """Compute dA21/dε̄ for the elastic power coefficient.

        Uses log-log derivative:
            dA21/dε̄ = A21 × slope / ε̄
        where slope = d(log10 A21) / d(log10 ε̄).

        Parameters
        ----------
        eps_mean_eV : float
            Mean electron energy [eV].

        Returns
        -------
        float
            dA21/dε̄ [eV·m³/(s·eV)] = [m³/s].
        """
        if not self._built:
            raise RuntimeError("LUT not built yet. Call build() first.")

        if eps_mean_eV <= 0:
            return 0.0

        log_eps = self._interp_eps(eps_mean_eV)
        eps_clamped = np.clip(eps_mean_eV,
                              self._eps_grid[0], self._eps_grid[-1])

        # Interpolate A21 at this eps
        A21 = self._interp_transport(log_eps, self._elastic_power_N)

        # Compute slope in log-log space
        # A21 can be all positive or mixed sign. For positive case, use log-log.
        if np.all(self._elastic_power_N > 0):
            log_A21 = np.log10(self._elastic_power_N)
            idx = np.searchsorted(self._log_eps_grid, log_eps) - 1
            idx = max(0, min(idx, len(self._log_eps_grid) - 2))
            d_log_eps = self._log_eps_grid[idx + 1] - self._log_eps_grid[idx]
            d_log_A21 = log_A21[idx + 1] - log_A21[idx]
            if abs(d_log_eps) > 1e-30:
                slope = d_log_A21 / d_log_eps
            else:
                slope = 0.0
            return A21 * slope / eps_clamped
        else:
            # Mixed/negative: use linear FD on the table
            idx = np.searchsorted(self._log_eps_grid, log_eps) - 1
            idx = max(0, min(idx, len(self._log_eps_grid) - 2))
            d_eps = self._eps_grid[idx + 1] - self._eps_grid[idx]
            d_A21 = self._elastic_power_N[idx + 1] - self._elastic_power_N[idx]
            if abs(d_eps) > 1e-30:
                return d_A21 / d_eps
            return 0.0

    # ------------------------------------------------------------------
    # Query: A20 derivative (for energy_source='A20' Jacobian)
    # ------------------------------------------------------------------

    def _interp_A20_deriv(self, eps_mean_eV: float) -> float:
        """Compute dA20/dε̄ for the total power coefficient.

        Uses log-log derivative (same approach as get_transport_deriv for A21).
        """
        if not self._built or eps_mean_eV <= 0:
            return 0.0

        log_eps = self._interp_eps(eps_mean_eV)
        eps_clamped = np.clip(eps_mean_eV,
                              self._eps_grid[0], self._eps_grid[-1])

        A20 = self._interp_transport(log_eps, self._power_N)

        if np.all(self._power_N > 0):
            log_A20 = np.log10(self._power_N)
            idx = np.searchsorted(self._log_eps_grid, log_eps) - 1
            idx = max(0, min(idx, len(self._log_eps_grid) - 2))
            d_log_eps = self._log_eps_grid[idx + 1] - self._log_eps_grid[idx]
            d_log_A20 = log_A20[idx + 1] - log_A20[idx]
            if abs(d_log_eps) > 1e-30:
                slope = d_log_A20 / d_log_eps
            else:
                slope = 0.0
            return A20 * slope / eps_clamped
        else:
            idx = np.searchsorted(self._log_eps_grid, log_eps) - 1
            idx = max(0, min(idx, len(self._log_eps_grid) - 2))
            d_eps = self._eps_grid[idx + 1] - self._eps_grid[idx]
            d_A20 = self._power_N[idx + 1] - self._power_N[idx]
            if abs(d_eps) > 1e-30:
                return d_A20 / d_eps
            return 0.0

    def invert_A20(self, A20_target: float) -> float:
        """Find ε̄ such that A20(ε̄) = A20_target, via bisection.

        A20 is monotonically increasing with ε̄, so the root is unique.

        Parameters
        ----------
        A20_target : float
            Target value of A20 [eV·m³/s].

        Returns
        -------
        float
            Mean electron energy ε̄ [eV] that gives the target A20.
            Clamped to LUT ε̄ range if target is out of bounds.
        """
        if not self._built:
            return 1.0

        eps_lo = self._eps_grid[0]
        eps_hi = self._eps_grid[-1]

        # Check bounds
        A20_lo = self._interp_transport(self._log_eps_grid[0], self._power_N)
        A20_hi = self._interp_transport(self._log_eps_grid[-1], self._power_N)

        if A20_target <= A20_lo:
            return eps_lo
        if A20_target >= A20_hi:
            return eps_hi

        # Bisection on log(eps) for efficiency (A20 spans orders of magnitude)
        log_lo = self._log_eps_grid[0]
        log_hi = self._log_eps_grid[-1]

        for _ in range(60):  # ~18 digits of precision
            log_mid = 0.5 * (log_lo + log_hi)
            A20_mid = self._interp_transport(log_mid, self._power_N)
            if A20_mid < A20_target:
                log_lo = log_mid
            else:
                log_hi = log_mid
            if (log_hi - log_lo) < 1e-12:
                break

        eps_result = 10.0 ** (0.5 * (log_lo + log_hi))
        return np.clip(eps_result, eps_lo, eps_hi)

    # ------------------------------------------------------------------
    # Query: transport coefficients (Layer 1)
    # ------------------------------------------------------------------

    def get_transport(self, eps_mean_eV: float) -> TransportData:
        """Interpolate BOLSIG+ transport at given ε̄ (vectorized bracket)."""
        if not self._built:
            raise RuntimeError("LUT not built yet. Call build() first.")

        log_eps = self._interp_eps(eps_mean_eV)
        grid = self._log_eps_grid
        idx = min(max(int(np.searchsorted(grid, log_eps)) - 1, 0),
                  len(grid) - 2)
        denom = grid[idx + 1] - grid[idx]
        frac = (log_eps - grid[idx]) / denom if abs(denom) > 1e-30 else 0.0

        td = TransportData()
        td.eps_mean_eV = eps_mean_eV
        td.power_N = self._interp_at(frac, idx, self._power_N)
        td.elastic_power_N = self._interp_at(frac, idx, self._elastic_power_N)
        td.inelastic_power_N = self._interp_at(frac, idx, self._inelastic_power_N)
        td.growth_power_N = self._interp_at(frac, idx, self._growth_power_N)
        td.EN_Td = self._interp_at(frac, idx, self._EN_Td)
        return td

    def _interp_at(self, frac: float, idx: int, data: np.ndarray) -> float:
        """Log-log interpolation at pre-computed bracket."""
        if np.all(data > 0):
            log_lo = np.log10(data[idx])
            log_hi = np.log10(data[idx + 1])
            return 10.0 ** (log_lo + frac * (log_hi - log_lo))
        elif np.all(data < 0):
            log_lo = np.log10(-data[idx])
            log_hi = np.log10(-data[idx + 1])
            return -(10.0 ** (log_lo + frac * (log_hi - log_lo)))
        else:
            return float(data[idx] + frac * (data[idx + 1] - data[idx]))

    def _interp_transport(self, log_eps: float, data: np.ndarray) -> float:
        """Log-log interpolation for a transport coefficient array.

        Handles sign changes (growth power A23 can be negative)
        by falling back to linear interpolation when values are
        non-positive.
        """
        if np.all(data > 0):
            # Pure log-log
            return 10.0 ** np.interp(
                log_eps, self._log_eps_grid, np.log10(data)
            )
        elif np.all(data < 0):
            # All negative — log-log on absolute value, then negate
            return -(10.0 ** np.interp(
                log_eps, self._log_eps_grid, np.log10(-data)
            ))
        else:
            # Mixed sign — fall back to linear interpolation on log(ε̄) axis
            return float(np.interp(log_eps, self._log_eps_grid, data))

    # ------------------------------------------------------------------
    # Reverse lookup: ε̄ → E/N
    # ------------------------------------------------------------------

    def eps_to_EN(self, eps_mean_eV: float) -> float:
        """Map mean energy to reduced electric field (reverse LUT).

        Parameters
        ----------
        eps_mean_eV : float
            Mean electron energy [eV].

        Returns
        -------
        float
            E/N [Td].
        """
        if not self._built:
            raise RuntimeError("LUT not built. Call build() first.")
        log_eps = self._interp_eps(eps_mean_eV)
        return 10.0 ** np.interp(
            log_eps, self._log_eps_grid, np.log10(self._EN_Td)
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def eps_range(self) -> Tuple[float, float]:
        """Return (ε̄_min, ε̄_max) [eV] of the LUT."""
        if not self._built:
            return (0.0, 0.0)
        return (float(self._eps_grid[0]), float(self._eps_grid[-1]))

    @property
    def EN_range(self) -> Tuple[float, float]:
        """Return (E/N_min, E/N_max) [Td] of the LUT."""
        if not self._built:
            return (0.0, 0.0)
        return (float(self._EN_Td[0]), float(self._EN_Td[-1]))

    @property
    def tgas_K(self) -> float:
        """Gas temperature the BOLSIG+ run was performed at."""
        return self._tgas_K


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# The old code used `BoltzmannLUT`.  Keep the name importable but point
# to the new class.  Any caller that used the old E/N-based interface
# will need updating (get_rate_coefficients now takes ε̄, not E/N).
BoltzmannLUT = MeanEnergyLUT
