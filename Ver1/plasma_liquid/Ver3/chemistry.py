"""
Aqueous Phase Chemistry System for Plasma-Liquid Interaction simulation.

This module implements the complete NOx aqueous chemistry system with:
- ODE solver for reaction kinetics
- Algebraic speciation for acid-base equilibria
- Reaction contribution tracking
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.integrate import solve_ivp
import yaml

from config import (
    WATER, HENRY_CONSTANTS, AQUEOUS_SPECIES, ACID_BASE_PAIRS,
    TRACKED_SPECIES, DEFAULTS, ODE_CONFIG, SALINE_SPECIES,
    DIAGNOSTIC_SPECIES_BASE, DIAGNOSTIC_SPECIES_SALINE
)
from chemistry_utils import (
    molecules_to_molar, apply_henry_law, calculate_pH, h_from_pH,
    speciate_acid_base, get_species_to_total_map, validate_concentration,
    estimate_n2o4_from_no2, estimate_h2o2_from_o3, calculate_hono_hono2
)
from utils import get_logger, safe_array, is_valid_array


class ReactionLoader:
    """Load and parse reaction definitions from YAML file."""

    @staticmethod
    def load_reactions(yaml_path: Path) -> List[Dict]:
        """
        Load reactions from YAML file.

        Parameters
        ----------
        yaml_path : Path
            Path to reactions.yaml file

        Returns
        -------
        List[Dict]
            List of reaction dictionaries
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        reactions = []

        # Collect all reaction categories
        categories = [
            'no_producing_reactions',
            'no2_producing_reactions',
            'o3_reactions',
            'hydrolysis_reactions',
            'oh_reactions',
            'h_atom_reactions',
            'ho2_reactions',
            'peroxynitrite_reactions',
            'additional_reactions',
            # Saline (chlorine) reaction categories
            'reversible_reactions',
            'irreversible_reactions',
        ]

        for category in categories:
            if category in data:
                for rxn in data[category]:
                    # Convert YAML format to internal format
                    converted = ReactionLoader._convert_reaction(rxn)
                    reactions.append(converted)

        return reactions

    @staticmethod
    def _convert_reaction(rxn: Dict) -> Dict:
        """Convert YAML reaction format to internal format."""
        if rxn['type'] == 'reversible':
            return {
                'type': 'rev',
                'reactants': rxn['reactants'],
                'products': rxn.get('products', {}),
                'k_f': rxn['k_f'],
                'k_b': rxn['k_b'],
                'label': rxn['label']
            }
        else:  # irreversible
            return {
                'type': 'irr',
                'reactants': rxn['reactants'],
                'products': rxn.get('products', {}),
                'k': rxn['k'],
                'label': rxn['label']
            }


class CompleteAqueousChemistry:
    """
    Complete NOx aqueous chemistry system with ODE solver and contribution analysis.

    Features:
    - Henry's law for gas-to-aqueous transfer
    - Algebraic speciation for acid-base pairs
    - ODE solver with multiple fallback methods
    - Reaction contribution tracking for NO, NO2, OH, ONOO-
    """

    def __init__(self, reactions_file: Optional[Path] = None, saline_mode: bool = False):
        """
        Initialize chemistry system.

        Parameters
        ----------
        reactions_file : Path, optional
            Path to reactions.yaml file. If None, uses default location.
        saline_mode : bool, optional
            If True, load additional chlorine reactions for saline solution.
        """
        self.logger = get_logger()
        self.saline_mode = saline_mode

        # Henry's law constants
        self.henry_constants = HENRY_CONSTANTS

        # Water properties
        self.Kw = WATER.KW

        # Species configuration - add saline species if in saline mode
        if saline_mode:
            self.aqueous_species = AQUEOUS_SPECIES + SALINE_SPECIES
        else:
            self.aqueous_species = AQUEOUS_SPECIES
        self.pKa_map = ACID_BASE_PAIRS
        self.species_to_total = get_species_to_total_map()

        # Create species index mapping
        self.species_idx = {species: i for i, species in enumerate(self.aqueous_species)}
        self.idx_species = {i: species for i, species in enumerate(self.aqueous_species)}

        # Default trace concentration
        self.trace_concentration = DEFAULTS.trace_concentration

        # Reaction tracking (legacy - for TRACKED_SPECIES)
        self.reaction_contributions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.total_production_rates: Dict[str, float] = defaultdict(float)

        # Diagnostic species tracking (full production/consumption analysis)
        self.diagnostic_species = DIAGNOSTIC_SPECIES_BASE.copy()
        if saline_mode:
            self.diagnostic_species.extend(DIAGNOSTIC_SPECIES_SALINE)

        # Per-timestep contribution storage: {species: {'production': {rxn: rate}, 'consumption': {rxn: rate}}}
        self.diagnostic_contributions: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._init_diagnostic_tracking()

        # Load reactions
        self._load_reactions(reactions_file)

    def _load_reactions(self, reactions_file: Optional[Path] = None):
        """Load reactions from file or use defaults."""
        if reactions_file is None:
            # Try to find reactions_full.yaml first (Liu 2015 full set)
            # Fall back to reactions.yaml if not found
            reactions_file = Path(__file__).parent / 'reactions_full.yaml'
            if not reactions_file.exists():
                reactions_file = Path(__file__).parent / 'reactions.yaml'

        if reactions_file.exists():
            try:
                self.reactions = ReactionLoader.load_reactions(reactions_file)
                self.logger.info(f"Loaded {len(self.reactions)} reactions from {reactions_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load reactions from file: {e}")
                self.reactions = self._get_default_reactions()
        else:
            self.logger.info("Using default reactions (reactions.yaml not found)")
            self.reactions = self._get_default_reactions()

        # Load additional saline reactions if in saline mode
        if self.saline_mode:
            self._load_saline_reactions()

    def _load_saline_reactions(self):
        """Load additional chlorine reactions for saline solution."""
        saline_file = Path(__file__).parent / 'reactions_saline.yaml'

        if saline_file.exists():
            try:
                saline_reactions = ReactionLoader.load_reactions(saline_file)
                self.reactions.extend(saline_reactions)
                self.logger.info(f"Loaded {len(saline_reactions)} saline reactions (total: {len(self.reactions)})")
            except Exception as e:
                self.logger.warning(f"Failed to load saline reactions: {e}")
        else:
            self.logger.warning("Saline reactions file not found: reactions_saline.yaml")

    def _init_diagnostic_tracking(self):
        """Initialize diagnostic contribution tracking for all species."""
        self.diagnostic_contributions = {}
        for species in self.diagnostic_species:
            self.diagnostic_contributions[species] = {
                'production': defaultdict(float),
                'consumption': defaultdict(float)
            }

    def reset_diagnostic_tracking(self):
        """Reset diagnostic tracking for new timestep."""
        self._init_diagnostic_tracking()

    def _get_default_reactions(self) -> List[Dict]:
        """Return minimal set of default reactions."""
        return [
            # Key NO-producing reactions
            {'type': 'rev', 'reactants': {'HONO': 2}, 'products': {'NO': 1, 'NO2': 1},
             'k_f': 13.4, 'k_b': 1.1e9, 'label': 'R13: 2HONO → NO + NO2'},
            {'type': 'irr', 'reactants': {'N2O4': 1}, 'products': {'NO2-': 1, 'NO3-': 1, 'H+': 2},
             'k': 1000, 'label': 'N2O4 + H2O → NO2- + NO3- + 2H+'},
            {'type': 'irr', 'reactants': {'N2O5': 1}, 'products': {'NO3-': 2, 'H+': 2},
             'k': 5e9, 'label': 'N2O5 + H2O → 2NO3- + 2H+'},
        ]

    def apply_henry_law(self, species: str, gas_conc_molecules_cm3: float) -> float:
        """Apply Henry's law for gas-to-aqueous transfer."""
        return apply_henry_law(species, gas_conc_molecules_cm3)

    def calculate_pH(self, H_conc: float) -> float:
        """Calculate pH from H+ concentration."""
        return calculate_pH(H_conc)

    def initialize_concentrations(
        self,
        C_aq_initial: Dict[str, float],
        initial_pH: float,
        cl_concentration: Optional[float] = None
    ) -> np.ndarray:
        """
        Initialize concentration vector for ODE solver.

        Uses TOTAL concentrations for acid-base pairs.

        Parameters
        ----------
        C_aq_initial : Dict[str, float]
            Initial aqueous concentrations in mol/L
        initial_pH : float
            Initial pH
        cl_concentration : float, optional
            Initial Cl- concentration in mol/L (for saline mode).
            Default is 0.154 M (0.9% NaCl).

        Returns
        -------
        np.ndarray
            Initial concentration vector
        """
        y0 = np.full(len(self.aqueous_species), self.trace_concentration)

        # Set pH-dependent species
        # Use accumulated H+ if available, otherwise use initial_pH
        if 'H+' in C_aq_initial and C_aq_initial['H+'] > 0:
            H_conc = C_aq_initial['H+']
        else:
            H_conc = h_from_pH(initial_pH)
        H_conc = np.clip(H_conc, 1e-14, 1.0)
        y0[self.species_idx['H+']] = H_conc
        y0[self.species_idx['OH-']] = self.Kw / H_conc

        # Atmospheric equilibrium (only set if not already accumulated)
        if 'O2' not in C_aq_initial or C_aq_initial.get('O2', 0) <= 0:
            y0[self.species_idx['O2']] = 2.5e-4
        if 'N2' not in C_aq_initial or C_aq_initial.get('N2', 0) <= 0:
            y0[self.species_idx['N2']] = 5e-4

        # Initial OH radical (only set if not already accumulated)
        if 'OH' not in C_aq_initial or C_aq_initial.get('OH', 0) <= 0:
            y0[self.species_idx['OH']] = 1e-12

        # Saline mode: set initial Cl- concentration
        if self.saline_mode and 'Cl-' in self.species_idx:
            # Use provided concentration or default to 0.9% NaCl = 0.154 M
            cl_conc = cl_concentration if cl_concentration is not None else 0.154
            y0[self.species_idx['Cl-']] = cl_conc

        # Track processed species
        processed_species = set()

        # Process acid-base pairs as TOTAL concentrations
        for total_name, (acid, base, _) in self.pKa_map.items():
            if total_name in self.species_idx:
                acid_conc = C_aq_initial.get(acid, 0)
                base_conc = C_aq_initial.get(base, 0)
                total_conc = acid_conc + base_conc

                if total_conc > 0:
                    safe_total = np.clip(total_conc, self.trace_concentration, 1.0)
                    y0[self.species_idx[total_name]] = safe_total

                processed_species.add(acid)
                processed_species.add(base)

        # N2O4, N2O5 hydrolysis: handled dynamically by ODE reactions (R95, R98)
        # Following Liu et al. 2015 J. Phys. D: Appl. Phys. 48 495201
        # R95: N2O4 + H2O → NO2- + NO3- + 2H+  (k = 1×10³ s⁻¹)
        # R98: N2O5 + H2O → 2H+ + 2NO3-        (k = 5×10⁹ s⁻¹)
        # These species are set as initial concentrations and processed by ODE solver.

        # Process non-equilibrium species (including N2O4, N2O5)
        for species, conc in C_aq_initial.items():
            if species in processed_species:
                continue
            if species in self.species_idx:
                safe_conc = np.clip(conc, self.trace_concentration, 1.0)
                y0[self.species_idx[species]] = max(y0[self.species_idx[species]], safe_conc)

        # Final safety check
        y0 = np.clip(y0, self.trace_concentration, 1.0)

        return y0

    def reaction_rates(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate reaction rates with algebraic speciation.

        This function is called by the ODE solver and must be robust
        against numerical issues.

        Parameters
        ----------
        t : float
            Current time
        y : np.ndarray
            Current concentration vector

        Returns
        -------
        np.ndarray
            Rate of change vector (dydt)
        """
        try:
            dydt = np.zeros(len(y), dtype=np.float64)

            # Sanitize input
            y = safe_array(y, min_val=self.trace_concentration, max_val=1.0)

            # Reset tracking
            self.reaction_contributions.clear()
            self.total_production_rates.clear()
            self._init_diagnostic_tracking()

            # Get current H+
            H_idx = self.species_idx['H+']
            y[H_idx] = np.clip(y[H_idx], 1e-14, 1.0)
            H = float(y[H_idx])

        except Exception as e:
            self.logger.error(f"reaction_rates initialization failed: {e}")
            return np.zeros(len(self.aqueous_species), dtype=np.float64)

        # Algebraic speciation
        speciated = self._compute_speciation(y, H)

        # Process reactions
        self._process_reactions(y, speciated, dydt)

        # OH- is algebraic
        if 'OH-' in self.species_idx:
            dydt[self.species_idx['OH-']] = 0.0

        # Final safety checks
        dydt = self._sanitize_rates(dydt)

        return dydt

    def _compute_speciation(self, y: np.ndarray, H: float) -> Dict[str, float]:
        """Compute individual species from total concentrations."""
        speciated = {}

        try:
            for total_name, (HA_name, A_name, pKa) in self.pKa_map.items():
                if total_name not in self.species_idx:
                    continue

                C_total = float(y[self.species_idx[total_name]])
                HA_conc, A_conc = speciate_acid_base(C_total, pKa, H)

                speciated[HA_name] = max(HA_conc, self.trace_concentration)
                speciated[A_name] = max(A_conc, self.trace_concentration)

            # Water equilibrium
            speciated['OH-'] = self.Kw / H

        except Exception as e:
            self.logger.error(f"Speciation error: {e}")

        return speciated

    def _process_reactions(
        self,
        y: np.ndarray,
        speciated: Dict[str, float],
        dydt: np.ndarray
    ):
        """Process all reactions and update dydt."""
        for rxn in self.reactions:
            try:
                rate = self._calculate_reaction_rate(rxn, y, speciated)

                if abs(rate) > 1e-30:
                    self._apply_rate_to_species(rxn, rate, dydt)
                    self._track_contributions(rxn, rate)

            except Exception:
                continue

    def _calculate_reaction_rate(
        self,
        rxn: Dict,
        y: np.ndarray,
        speciated: Dict[str, float]
    ) -> float:
        """Calculate rate for a single reaction with mass-balance limiting."""
        rate = 0.0

        try:
            if rxn['type'] == 'irr':
                rate = float(rxn['k'])

                for species, coeff in rxn['reactants'].items():
                    conc = self._get_concentration(species, y, speciated)
                    rate *= self._safe_power(conc, coeff)

                    if rate > 1e15 or not np.isfinite(rate):
                        return 0.0

                # Mass-balance limiting: rate cannot consume more than available reactants
                # rate × dt × coeff <= concentration
                # => rate <= concentration / (coeff × dt)
                dt = ODE_CONFIG.max_time_step
                for species, coeff in rxn['reactants'].items():
                    conc = self._get_concentration(species, y, speciated)
                    if conc > self.trace_concentration and coeff > 0:
                        max_rate = conc / (coeff * dt)
                        rate = min(rate, max_rate)

            elif rxn['type'] == 'rev':
                rate_f = float(rxn['k_f'])
                rate_b = float(rxn['k_b'])

                # Forward reaction
                for species, coeff in rxn['reactants'].items():
                    conc = self._get_concentration(species, y, speciated)
                    rate_f *= self._safe_power(conc, coeff)

                # Backward reaction
                for species, coeff in rxn['products'].items():
                    conc = self._get_concentration(species, y, speciated)
                    rate_b *= self._safe_power(conc, coeff)

                rate = rate_f - rate_b

                if not np.isfinite(rate) or abs(rate) > 1e15:
                    return 0.0

                # Mass-balance limiting for reversible reactions
                dt = ODE_CONFIG.max_time_step
                if rate > 0:
                    # Forward direction: limit by reactants
                    for species, coeff in rxn['reactants'].items():
                        conc = self._get_concentration(species, y, speciated)
                        if conc > self.trace_concentration and coeff > 0:
                            max_rate = conc / (coeff * dt)
                            rate = min(rate, max_rate)
                elif rate < 0:
                    # Backward direction: limit by products
                    for species, coeff in rxn['products'].items():
                        conc = self._get_concentration(species, y, speciated)
                        if conc > self.trace_concentration and coeff > 0:
                            max_rate = conc / (coeff * dt)
                            rate = max(rate, -max_rate)

        except Exception:
            return 0.0

        return rate

    def _get_concentration(
        self,
        species: str,
        y: np.ndarray,
        speciated: Dict[str, float]
    ) -> float:
        """Get concentration from speciated dict or y vector."""
        if species in speciated:
            return speciated[species]
        elif species in self.species_idx:
            return float(y[self.species_idx[species]])
        return 0.0

    def _safe_power(self, conc: float, coeff: int) -> float:
        """Calculate concentration^coeff safely."""
        conc = min(conc, 1.0)
        coeff = min(coeff, 3)
        return conc ** coeff

    def _apply_rate_to_species(
        self,
        rxn: Dict,
        rate: float,
        dydt: np.ndarray
    ):
        """Apply reaction rate to species in dydt."""
        # Reactants consumed
        for species, coeff in rxn['reactants'].items():
            if species in self.species_to_total:
                total_name = self.species_to_total[species]
                if total_name in self.species_idx:
                    dydt[self.species_idx[total_name]] -= coeff * rate
            elif species in self.species_idx:
                dydt[self.species_idx[species]] -= coeff * rate

        # Products formed
        for species, coeff in rxn['products'].items():
            if species in self.species_to_total:
                total_name = self.species_to_total[species]
                if total_name in self.species_idx:
                    dydt[self.species_idx[total_name]] += coeff * rate
            elif species in self.species_idx:
                dydt[self.species_idx[species]] += coeff * rate

    def _track_contributions(self, rxn: Dict, rate: float):
        """Track reaction contributions for production analysis."""
        label = rxn.get('label', 'Unnamed reaction')

        # Legacy tracking for TRACKED_SPECIES (production only, rate > 0)
        if rate > 0:
            for species, coeff in rxn['products'].items():
                if species in TRACKED_SPECIES:
                    self.reaction_contributions[species][label] += coeff * rate
                    self.total_production_rates[species] += coeff * rate

        # Diagnostic tracking for all species (both production and consumption)
        # For irreversible reactions: rate > 0 means forward
        # For reversible reactions: rate = rate_f - rate_b (can be positive or negative)
        self._track_diagnostic_contributions(rxn, rate, label)

    def _track_diagnostic_contributions(self, rxn: Dict, rate: float, label: str):
        """
        Track production and consumption for all diagnostic species.

        For a reaction A + B → C + D with rate r:
        - A and B are consumed at rate r (consumption)
        - C and D are produced at rate r (production)

        For reversible reactions where rate = rate_f - rate_b:
        - If rate > 0: forward direction dominates
        - If rate < 0: backward direction dominates (products become reactants)
        """
        if abs(rate) < 1e-30:
            return

        # Handle rate direction
        if rate > 0:
            # Forward reaction: reactants consumed, products formed
            consumed_species = rxn['reactants']
            produced_species = rxn['products']
            effective_rate = rate
        else:
            # Backward reaction: products consumed, reactants formed
            consumed_species = rxn['products']
            produced_species = rxn['reactants']
            effective_rate = -rate

        # Track consumption (reactants are consumed)
        for species, coeff in consumed_species.items():
            if species in self.diagnostic_contributions:
                self.diagnostic_contributions[species]['consumption'][label] += coeff * effective_rate

        # Track production (products are formed)
        for species, coeff in produced_species.items():
            if species in self.diagnostic_contributions:
                self.diagnostic_contributions[species]['production'][label] += coeff * effective_rate

    def _sanitize_rates(self, dydt: np.ndarray) -> np.ndarray:
        """Ensure dydt contains valid values."""
        dydt = np.nan_to_num(dydt, nan=0.0, posinf=0.0, neginf=0.0)
        dydt = np.clip(dydt, -ODE_CONFIG.max_rate, ODE_CONFIG.max_rate)
        return dydt.astype(np.float64)

    def solve(
        self,
        C_aq_initial: Dict[str, float],
        initial_pH: float,
        time_step: float = 0.1,
        cl_concentration: Optional[float] = None
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Solve aqueous equilibria using ODE solver.

        Parameters
        ----------
        C_aq_initial : Dict[str, float]
            Initial aqueous concentrations in mol/L
        initial_pH : float
            Initial pH
        time_step : float
            Simulation time step in seconds
        cl_concentration : float, optional
            Initial Cl- concentration in mol/L (for saline mode).
            Default is 0.154 M (0.9% NaCl).

        Returns
        -------
        Tuple[Dict, Dict]
            (final_concentrations, contribution_analysis)
        """
        # Initialize
        y0 = self.initialize_concentrations(C_aq_initial, initial_pH, cl_concentration)

        # Pre-flight check
        if not self._preflight_check(y0):
            return self._fallback_result(y0)

        # Try ODE solvers
        y_final = self._solve_ode(y0, time_step)

        # Convert result to dictionary
        return self._process_solution(y_final)

    def _preflight_check(self, y0: np.ndarray) -> bool:
        """Test reaction_rates before calling ODE solver."""
        try:
            self.logger.debug("Pre-flight check: Testing reaction_rates...")
            test_dydt = self.reaction_rates(0.0, y0)

            if test_dydt is None or len(test_dydt) != len(y0):
                self.logger.warning("Pre-flight FAILED: Invalid return")
                return False

            if not is_valid_array(test_dydt):
                self.logger.warning("Pre-flight FAILED: Non-finite values")
                return False

            if np.max(np.abs(test_dydt)) > 1e10:
                self.logger.warning(f"Pre-flight FAILED: Extreme values")
                return False

            self.logger.debug("Pre-flight PASSED")
            return True

        except Exception as e:
            self.logger.warning(f"Pre-flight FAILED: {e}")
            return False

    def _solve_ode(self, y0: np.ndarray, time_step: float) -> np.ndarray:
        """Solve ODE with multiple fallback methods."""
        y_final = None

        for method, rtol, atol in ODE_CONFIG.methods:
            if y_final is not None:
                break

            try:
                self.logger.debug(f"Trying ODE solver: {method}")

                # time_step: total simulation time
                # max_time_step: maximum step SIZE for ODE solver (not total time)
                max_step_size = ODE_CONFIG.max_time_step

                sol = solve_ivp(
                    self.reaction_rates,
                    (0, time_step),  # Integrate over full time_step
                    y0,
                    method=method,
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step_size,  # Limit individual step size
                    first_step=max_step_size / 10,
                    vectorized=False
                )

                if sol.success and is_valid_array(sol.y[:, -1]):
                    y_final = sol.y[:, -1]
                    self.logger.debug(f"  {method}: SUCCESS")

                    # Calculate contributions at final time
                    _ = self.reaction_rates(time_step, y_final)

            except Exception as e:
                self.logger.debug(f"  {method}: Failed - {str(e)[:50]}")
                continue

        # Fallback to Euler
        if y_final is None:
            y_final = self._euler_integration(y0, time_step)

        return y_final if y_final is not None else y0

    def _euler_integration(
        self,
        y0: np.ndarray,
        time_step: float
    ) -> Optional[np.ndarray]:
        """Simple Euler integration as last resort."""
        self.logger.warning("Using Euler integration fallback")

        try:
            y = y0.copy()
            dt = ODE_CONFIG.euler_dt
            n_steps = min(int(time_step / dt), ODE_CONFIG.euler_max_steps)

            for i in range(n_steps):
                dydt = self.reaction_rates(i * dt, y)
                y = y + dt * dydt
                y = np.clip(y, self.trace_concentration, 1.0)

                if not is_valid_array(y):
                    self.logger.warning(f"Euler failed at step {i}")
                    return y0

            self.logger.debug(f"Euler SUCCESS ({n_steps} steps)")
            return y

        except Exception as e:
            self.logger.error(f"Euler integration failed: {e}")
            return None

    def _fallback_result(
        self,
        y0: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Return initial conditions as fallback."""
        self.logger.warning("Using initial conditions as fallback")
        return self._process_solution(y0)

    def _process_solution(
        self,
        y_final: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Convert solution vector to dictionaries."""
        C_final = {}

        # Get final H+
        H_final = max(y_final[self.species_idx['H+']], self.trace_concentration)
        C_final['H+'] = H_final

        # Speciate acid-base pairs
        for total_name, (HA_name, A_name, pKa) in self.pKa_map.items():
            if total_name in self.species_idx:
                C_total = max(y_final[self.species_idx[total_name]], self.trace_concentration)
                HA_conc, A_conc = speciate_acid_base(C_total, pKa, H_final)

                C_final[HA_name] = max(HA_conc, self.trace_concentration)
                C_final[A_name] = max(A_conc, self.trace_concentration)

        # Water equilibrium
        C_final['OH-'] = max(self.Kw / H_final, self.trace_concentration)

        # Non-equilibrium species
        for i, species in enumerate(self.aqueous_species):
            if species not in C_final and species not in self.pKa_map:
                C_final[species] = max(y_final[i], self.trace_concentration)

        # pH
        C_final['pH'] = self.calculate_pH(H_final)

        # Calculate contributions
        contributions = self._calculate_contributions()

        return C_final, contributions

    def _calculate_contributions(self) -> Dict[str, Dict[str, float]]:
        """Calculate percentage contributions for tracked species."""
        contributions = {}

        for species in TRACKED_SPECIES:
            if self.total_production_rates[species] > 0:
                contributions[species] = {}

                for rxn_label, rate in self.reaction_contributions[species].items():
                    percentage = (rate / self.total_production_rates[species]) * 100
                    contributions[species][rxn_label] = percentage

                # Sort by percentage
                contributions[species] = dict(
                    sorted(contributions[species].items(),
                           key=lambda x: x[1], reverse=True)
                )

        return contributions

    def get_diagnostic_contributions(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get current diagnostic contributions for all species.

        Returns
        -------
        Dict with structure:
            {species: {'production': {rxn: rate}, 'consumption': {rxn: rate}}}
        """
        # Convert defaultdicts to regular dicts and sort by rate
        result = {}
        for species in self.diagnostic_species:
            result[species] = {
                'production': dict(sorted(
                    self.diagnostic_contributions[species]['production'].items(),
                    key=lambda x: x[1], reverse=True
                )),
                'consumption': dict(sorted(
                    self.diagnostic_contributions[species]['consumption'].items(),
                    key=lambda x: x[1], reverse=True
                ))
            }
        return result

    def get_diagnostic_summary(self, top_n: int = 10) -> Dict[str, Dict[str, List[tuple]]]:
        """
        Get summary of top contributors for each species.

        Parameters
        ----------
        top_n : int
            Number of top reactions to include

        Returns
        -------
        Dict with structure:
            {species: {'production': [(rxn, rate, pct), ...], 'consumption': [(rxn, rate, pct), ...]}}
        """
        result = {}

        for species in self.diagnostic_species:
            prod_data = self.diagnostic_contributions[species]['production']
            cons_data = self.diagnostic_contributions[species]['consumption']

            # Calculate totals
            total_prod = sum(prod_data.values()) if prod_data else 0
            total_cons = sum(cons_data.values()) if cons_data else 0

            # Sort and get top N with percentages
            prod_sorted = sorted(prod_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            cons_sorted = sorted(cons_data.items(), key=lambda x: x[1], reverse=True)[:top_n]

            prod_list = [(rxn, rate, (rate / total_prod * 100) if total_prod > 0 else 0)
                         for rxn, rate in prod_sorted]
            cons_list = [(rxn, rate, (rate / total_cons * 100) if total_cons > 0 else 0)
                         for rxn, rate in cons_sorted]

            result[species] = {
                'production': prod_list,
                'consumption': cons_list,
                'total_production': total_prod,
                'total_consumption': total_cons
            }

        return result

    # Legacy method for backward compatibility
    def solve_aqueous_equilibria_with_tracking(
        self,
        C_aq_initial: Dict[str, float],
        initial_pH: float,
        time_step: float = 0.1
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Legacy wrapper."""
        return self.solve(C_aq_initial, initial_pH, time_step)

    def solve_aqueous_equilibria(
        self,
        C_aq_initial: Dict[str, float],
        initial_pH: float,
        time_step: float = 0.1
    ) -> Dict[str, float]:
        """Legacy wrapper returning only concentrations."""
        C_final, _ = self.solve(C_aq_initial, initial_pH, time_step)
        return C_final
