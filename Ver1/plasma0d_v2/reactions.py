"""Reaction system for concentration-based plasma chemistry.

Rate expressions (COMSOL chemical engineering convention):
  ARRHENIUS: k = A * T^n * exp(-E / (R*T))
    - Order 1: k [1/s],           rate = k * c_A
    - Order 2: k [m³/(mol·s)],    rate = k * c_A * c_B
    - Order 3: k [m⁶/(mol²·s)],   rate = k * c_A * c_B * c_M
  
  ELECTRON_IMPACT: k(Te) from Boltzmann solver / cross-section LUT
    - k_bolsig [m³/s] (number density basis)
    - k_conc = k_bolsig * NA [m³/(mol·s)] (concentration basis)
    - rate = k_conc * c_e * c_target [mol/(m³·s)]

All concentrations in [mol/m³], rates in [mol/(m³·s)].
"""

import yaml
import numpy as np
from typing import List, Dict, Tuple, Optional
from .constants import R_GAS, NA, KB


class Reaction:
    """Single reaction with stoichiometry and rate parameters."""
    
    def __init__(self, rxn_id: int, rxn_type: str, formula: str,
                 reactant_names: List[str], reactant_coeffs: List[int],
                 product_names: List[str], product_coeffs: List[int]):
        self.id = rxn_id
        self.type = rxn_type       # 'ARRHENIUS', 'ELECTRON_IMPACT', or 'TE_DEPENDENT'
        self.formula = formula
        self.reactant_names = reactant_names
        self.reactant_coeffs = reactant_coeffs
        self.product_names = product_names
        self.product_coeffs = product_coeffs
        
        # Arrhenius parameters (COMSOL units)
        self.A = 0.0       # pre-exponential factor
        self.n = 0.0       # temperature exponent
        self.E = 0.0       # activation energy [J/mol]
        self.order = 2     # reaction order (1, 2, or 3)
        
        # Electron impact parameters
        self.cross_section_file = None
        self.energy_loss_eV = 0.0
        self.bolsig_index = -1   # index in Boltzmann LUT
        
        # TE_DEPENDENT parameters
        # DR subtype: k_cgs = A_cgs * (300/Te_K)^n_Te  [cm³/s]
        # AT1_KOSSYI subtype: pressure-folded Kossyi 3-body attachment
        self.subtype = None        # 'DR', 'AT1_KOSSYI', or None (constant)
        self.A_cgs = 0.0           # pre-exponential in cgs [cm³/s]
        self.n_Te = 0.0            # Te exponent
        self.k3_cgs = 0.0          # 3-body rate coefficient for AT1_KOSSYI [cm⁶/s]
        
        # Indices in state vector (assigned at build time)
        self.reactant_indices: List[int] = []
        self.product_indices: List[int] = []
        
        # Stoichiometry: net change for each species
        self.stoich_vector: Optional[np.ndarray] = None
        
        # Global index in self.reactions list (set by build())
        self._global_index: int = -1
    
    def __repr__(self):
        return f"R{self.id}: {self.formula} [{self.type}]"


class ReactionSet:
    """Collection of all reactions with rate computation."""
    
    def __init__(self):
        self.reactions: List[Reaction] = []
        self.ei_reactions: List[Reaction] = []        # electron impact subset
        self.arrhenius_reactions: List[Reaction] = []  # Arrhenius subset
        self.te_dependent_reactions: List[Reaction] = []  # Te-dependent subset
        self.n_species = 0
        self.stoich_matrix: Optional[np.ndarray] = None  # (n_species, n_reactions)
        self._electron_index = 0
        self._veff_mask: Optional[np.ndarray] = None  # (n_reactions,) bool: True = V_eff reaction
    
    def load_from_yaml(self, filepath: str):
        """Load reactions from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        for rxn_data in data['reactions']:
            r_names = [x['species'] for x in rxn_data['reactants']]
            r_coeffs = [x['coeff'] for x in rxn_data['reactants']]
            p_names = [x['species'] for x in rxn_data['products']]
            p_coeffs = [x['coeff'] for x in rxn_data['products']]
            
            rxn = Reaction(
                rxn_id=rxn_data['id'],
                rxn_type=rxn_data['type'],
                formula=rxn_data['formula'],
                reactant_names=r_names,
                reactant_coeffs=r_coeffs,
                product_names=p_names,
                product_coeffs=p_coeffs,
            )
            
            if rxn.type == 'ARRHENIUS':
                rxn.A = rxn_data['A']
                rxn.n = rxn_data.get('n', 0.0)
                rxn.E = rxn_data.get('E', 0.0)
                rxn.order = rxn_data.get('order', 2)
            elif rxn.type == 'ELECTRON_IMPACT':
                rxn.cross_section_file = rxn_data.get('cross_section_file')
                rxn.energy_loss_eV = rxn_data.get('energy_loss_eV', 0.0)
                rxn.sigma_over_N = rxn_data.get('sigma_over_N', False)
            elif rxn.type == 'TE_DEPENDENT':
                rxn.A_cgs = rxn_data.get('A_cgs', 0.0)
                rxn.n_Te = rxn_data.get('n_Te', 0.0)
                rxn.k3_cgs = rxn_data.get('k3_cgs', 0.0)
                rxn.subtype = rxn_data.get('subtype', None)
            
            self.reactions.append(rxn)
    
    def build(self, species_manager):
        """Assign species indices and build stoichiometry matrix."""
        self.n_species = species_manager.n_species
        n_rxn = len(self.reactions)
        self.stoich_matrix = np.zeros((self.n_species, n_rxn))
        
        ei_idx = 0
        for j, rxn in enumerate(self.reactions):
            # Cache global index for O(1) lookup in hot loops
            rxn._global_index = j

            # Resolve species indices
            rxn.reactant_indices = []
            for name in rxn.reactant_names:
                if species_manager.has(name):
                    rxn.reactant_indices.append(species_manager.index(name))
                else:
                    print(f"  WARNING: species '{name}' not found (reaction {rxn.formula})")
                    rxn.reactant_indices.append(-1)
            
            rxn.product_indices = []
            for name in rxn.product_names:
                if species_manager.has(name):
                    rxn.product_indices.append(species_manager.index(name))
                else:
                    print(f"  WARNING: species '{name}' not found (reaction {rxn.formula})")
                    rxn.product_indices.append(-1)
            
            # Build stoichiometry vector
            rxn.stoich_vector = np.zeros(self.n_species)
            for k, idx in enumerate(rxn.reactant_indices):
                if idx >= 0:
                    rxn.stoich_vector[idx] -= rxn.reactant_coeffs[k]
            for k, idx in enumerate(rxn.product_indices):
                if idx >= 0:
                    rxn.stoich_vector[idx] += rxn.product_coeffs[k]
            
            self.stoich_matrix[:, j] = rxn.stoich_vector
            
            # Categorize
            if rxn.type == 'ELECTRON_IMPACT':
                rxn.bolsig_index = ei_idx
                ei_idx += 1
                self.ei_reactions.append(rxn)
            elif rxn.type == 'TE_DEPENDENT':
                self.te_dependent_reactions.append(rxn)
            else:
                self.arrhenius_reactions.append(rxn)
        
        self._electron_index = species_manager.index('e') if species_manager.has('e') else 0
        
        # --- Phase 6: Compute ΔH per reaction and classify energy channels ---
        delta_hf = species_manager.get_delta_hf_array()  # [kJ/mol] per species
        self._delta_h_kj = np.zeros(n_rxn)
        for j, rxn in enumerate(self.reactions):
            # ΔH_rxn = Σ(ν_products · ΔHf_products) - Σ(ν_reactants · ΔHf_reactants)
            # = stoich_matrix[:, j] @ delta_hf  (since stoich = +products, -reactants)
            self._delta_h_kj[j] = self.stoich_matrix[:, j] @ delta_hf

        # Gas heating: all non-ELECTRON_IMPACT reactions contribute chemical enthalpy
        # (EI energy already handled by P_inel)
        self._gas_heating_indices = []
        # Electron loss: ALL reactions where net electron stoichiometry < 0
        # Includes DR, AT (TE_DEPENDENT) and EI attachment (e.g. R166: e+O2→O⁻+O)
        # P_e_loss (ε̄ × S_loss) is separate from P_inel (energy_loss_eV × R):
        #   P_inel = fixed threshold energy consumed by the reaction
        #   P_e_loss = electron kinetic energy lost when electron is destroyed
        self._electron_loss_indices = []
        for j, rxn in enumerate(self.reactions):
            if rxn.type != 'ELECTRON_IMPACT':
                self._gas_heating_indices.append(j)
            if self.stoich_matrix[self._electron_index, j] < -0.5:
                self._electron_loss_indices.append(j)

        n_exo = sum(1 for j in self._gas_heating_indices if self._delta_h_kj[j] < 0)
        n_endo = sum(1 for j in self._gas_heating_indices if self._delta_h_kj[j] > 0)
        print(f"    Gas heating reactions: {len(self._gas_heating_indices)} "
              f"({n_exo} exothermic, {n_endo} endothermic)")
        print(f"    Electron loss reactions (DR/AT): {len(self._electron_loss_indices)}")

        n_ei = len(self.ei_reactions)
        n_arr = len(self.arrhenius_reactions)
        n_te = len(self.te_dependent_reactions)
        print(f"  Reactions built: {n_rxn} total "
              f"({n_ei} electron-impact, {n_arr} Arrhenius, {n_te} Te-dependent)")
        print(f"    Order 1: {sum(1 for r in self.arrhenius_reactions if r.order==1)}")
        print(f"    Order 2: {sum(1 for r in self.arrhenius_reactions if r.order==2)}")
        print(f"    Order 3: {sum(1 for r in self.arrhenius_reactions if r.order==3)}")

        self._build_vectorized_arrays()

    def _build_vectorized_arrays(self):
        """Pre-extract reaction parameters into flat numpy arrays for
        vectorized rate computation (eliminates Python for-loops in RHS)."""
        e_idx = self._electron_index

        # --- EI reactions ---
        n_ei = len(self.ei_reactions)
        self._ei_global_idx = np.array([r._global_index for r in self.ei_reactions], dtype=np.intp)
        self._ei_bolsig_idx = np.array([r.bolsig_index for r in self.ei_reactions], dtype=np.intp)
        self._ei_target_idx = np.zeros(n_ei, dtype=np.intp)
        for i, rxn in enumerate(self.ei_reactions):
            for idx in rxn.reactant_indices:
                if idx != e_idx:
                    self._ei_target_idx[i] = idx
                    break

        # --- Arrhenius reactions ---
        n_arr = len(self.arrhenius_reactions)
        self._arr_global_idx = np.array([r._global_index for r in self.arrhenius_reactions], dtype=np.intp)
        self._arr_A = np.array([r.A for r in self.arrhenius_reactions])
        self._arr_n = np.array([r.n for r in self.arrhenius_reactions])
        self._arr_E = np.array([r.E for r in self.arrhenius_reactions])
        self._arr_order = np.array([r.order for r in self.arrhenius_reactions], dtype=np.int32)
        self._arr_idx_a = np.zeros(n_arr, dtype=np.intp)
        self._arr_idx_b = np.zeros(n_arr, dtype=np.intp)
        self._arr_n_reactants = np.zeros(n_arr, dtype=np.int32)
        self._arr_same_reactant = np.zeros(n_arr, dtype=bool)
        for i, rxn in enumerate(self.arrhenius_reactions):
            ri = rxn.reactant_indices
            self._arr_n_reactants[i] = len(ri)
            if len(ri) >= 1:
                self._arr_idx_a[i] = ri[0]
            if len(ri) >= 2:
                self._arr_idx_b[i] = ri[1]
                self._arr_same_reactant[i] = (ri[0] == ri[1])

        # --- TE-dependent reactions ---
        n_te = len(self.te_dependent_reactions)
        self._te_global_idx = np.array([r._global_index for r in self.te_dependent_reactions], dtype=np.intp)
        self._te_subtype = [r.subtype for r in self.te_dependent_reactions]
        self._te_A_cgs = np.array([r.A_cgs for r in self.te_dependent_reactions])
        self._te_n_Te = np.array([r.n_Te for r in self.te_dependent_reactions])
        self._te_k3_cgs = np.array([r.k3_cgs for r in self.te_dependent_reactions])
        self._te_idx_a = np.zeros(n_te, dtype=np.intp)
        self._te_idx_b = np.zeros(n_te, dtype=np.intp)
        self._te_n_reactants = np.zeros(n_te, dtype=np.int32)
        self._te_target_idx = np.zeros(n_te, dtype=np.intp)
        for i, rxn in enumerate(self.te_dependent_reactions):
            ri = rxn.reactant_indices
            self._te_n_reactants[i] = len(ri)
            if len(ri) >= 1:
                self._te_idx_a[i] = ri[0]
            if len(ri) >= 2:
                self._te_idx_b[i] = ri[1]
            for idx in ri:
                if idx != e_idx:
                    self._te_target_idx[i] = idx
                    break

    def _init_veff_mask(self):
        """Initialize _veff_mask as all-True (no volume separation).
        Used when dilution is disabled but Numba gas_heat still needs the mask."""
        self._veff_mask = np.ones(self.n_reactions, dtype=bool)

    def _build_gas_heating_arrays(self):
        """Pre-compute arrays for vectorized gas_heating_split.
        Called after tag_veff_reactions or _init_veff_mask (needs _veff_mask)."""
        idx = np.array(self._gas_heating_indices, dtype=np.intp)
        self._gas_heating_idx_arr = idx
        self._delta_h_J = self._delta_h_kj * 1000.0  # kJ -> J

        vm = self._veff_mask
        mask = np.zeros(len(idx), dtype=bool)
        for i, j in enumerate(idx):
            if vm is not None and vm[j]:
                mask[i] = True
            elif self.reactions[j].type == 'TE_DEPENDENT':
                mask[i] = True
        self._gas_heat_veff_mask = mask

    def tag_veff_reactions(self, ion_species_indices):
        """Tag reactions that occur only in V_eff (discharge volume).

        Reactions in V_eff (need f_species dilution for neutrals):
          - All ELECTRON_IMPACT reactions
          - All TE_DEPENDENT reactions
          - ARRHENIUS reactions with at least one ion reactant

        Reactions in V_reactor (bulk, no dilution):
          - ARRHENIUS reactions with only neutral reactants

        Parameters
        ----------
        ion_species_indices : set or list
            Species indices of all ions (positive + negative).
        """
        ion_set = set(ion_species_indices)
        self._veff_mask = np.zeros(self.n_reactions, dtype=bool)

        n_ion_arr = 0
        ion_arr_ids = []
        for j, rxn in enumerate(self.reactions):
            if rxn.type in ('ELECTRON_IMPACT', 'TE_DEPENDENT'):
                self._veff_mask[j] = True
            elif rxn.type == 'ARRHENIUS':
                for idx in rxn.reactant_indices:
                    if idx in ion_set:
                        self._veff_mask[j] = True
                        n_ion_arr += 1
                        ion_arr_ids.append(rxn.id)
                        break

        n_veff = int(np.sum(self._veff_mask))
        n_bulk = self.n_reactions - n_veff
        print(f"  Volume tagging: {n_veff} V_eff reactions "
              f"({len(self.ei_reactions)} EI + {len(self.te_dependent_reactions)} TE "
              f"+ {n_ion_arr} ion-Arrhenius), {n_bulk} bulk Arrhenius")

        self._build_gas_heating_arrays()

    @property
    def n_reactions(self):
        return len(self.reactions)
    
    @property
    def n_ei(self):
        return len(self.ei_reactions)
    
    def compute_arrhenius_rate_coefficients(self, T_gas: float) -> np.ndarray:
        """Vectorized Arrhenius: k = A * T^n * exp(-E/(R*T))."""
        E = self._arr_E
        k = self._arr_A * (T_gas ** self._arr_n)
        mask = E != 0.0
        k[mask] *= np.exp(-E[mask] / (R_GAS * T_gas))
        return k

    def compute_reaction_rates(self, concentrations: np.ndarray, T_gas: float,
                                c_total: float,
                                k_ei_conc: Optional[np.ndarray] = None,
                                Te_eV: float = 1.0,
                                P_gas: float = 101325.0) -> np.ndarray:
        """Vectorized rate computation — no Python for-loops."""
        rates = np.zeros(self.n_reactions)
        c = np.maximum(concentrations, 0.0)
        e_idx = self._electron_index

        # --- EI rates (vectorized) ---
        if k_ei_conc is not None and len(self._ei_global_idx) > 0:
            c_e = c[e_idx]
            k_vals = k_ei_conc[self._ei_bolsig_idx]
            c_targets = c[self._ei_target_idx]
            rates[self._ei_global_idx] = k_vals * c_e * c_targets

        # --- Arrhenius rates (vectorized) ---
        k_arr = self.compute_arrhenius_rate_coefficients(T_gas)
        gi = self._arr_global_idx
        ca = c[self._arr_idx_a]
        cb = c[self._arr_idx_b]
        order = self._arr_order
        nr = self._arr_n_reactants
        same = self._arr_same_reactant

        r = np.zeros(len(k_arr))
        m1 = order == 1
        r[m1] = k_arr[m1] * ca[m1]

        m2 = (order == 2) & (nr >= 2)
        r[m2] = k_arr[m2] * ca[m2] * cb[m2]
        m2s = (order == 2) & (nr == 1)
        r[m2s] = k_arr[m2s] * ca[m2s] * c_total

        m3 = (order == 3) & (nr >= 2)
        r[m3] = k_arr[m3] * ca[m3] * cb[m3] * c_total
        m3s = (order == 3) & (nr == 1)
        r[m3s] = k_arr[m3s] * ca[m3s] * c_total * c_total

        rates[gi] = r

        # --- TE-dependent rates ---
        if len(self._te_global_idx) > 0:
            Te_K = max(Te_eV * 11604.0, 300.0)
            c_e = c[e_idx]
            T_gas_safe = max(T_gas, 200.0)

            for i in range(len(self._te_global_idx)):
                j = self._te_global_idx[i]
                if self._te_subtype[i] == 'AT1_KOSSYI':
                    M_cm3 = P_gas / (KB * T_gas_safe) * 1e-6
                    k3 = self._te_k3_cgs[i] * (300.0 / Te_K) ** 2
                    k3 *= np.exp(-70.0 / T_gas_safe)
                    k3 *= np.exp(1500.0 * (Te_K - T_gas_safe) / (Te_K * T_gas_safe))
                    k_SI = k3 * M_cm3 * 1e-6 * NA
                    rates[j] = k_SI * c_e * c[self._te_target_idx[i]]
                else:
                    if self._te_n_Te[i] != 0.0:
                        k_cgs = self._te_A_cgs[i] * (300.0 / Te_K) ** self._te_n_Te[i]
                    else:
                        k_cgs = self._te_A_cgs[i]
                    k_SI = k_cgs * 1e-6 * NA
                    if self._te_n_reactants[i] >= 2:
                        rates[j] = k_SI * c[self._te_idx_a[i]] * c[self._te_idx_b[i]]
                    elif self._te_n_reactants[i] == 1:
                        rates[j] = k_SI * c[self._te_idx_a[i]]

        return rates
    
    def compute_source_terms(self, rates: np.ndarray) -> np.ndarray:
        """Compute species source terms from reaction rates.
        
        dc_i/dt|_chem = sum_j (nu_ij * R_j)
        
        Returns source array [mol/(m³·s)], shape (n_species,).
        """
        return self.stoich_matrix @ rates

    def compute_source_terms_split(self, rates: np.ndarray):
        """Compute species source terms split by reaction volume.

        Returns (S_veff, S_bulk):
            S_veff: source from V_eff reactions [mol/(m³·s)]
                    (EI + TE + ion-involving Arrhenius)
            S_bulk: source from bulk reactions [mol/(m³·s)]
                    (neutral-only Arrhenius)

        If tag_veff_reactions() has not been called, falls back to
        legacy behavior (EI+TE vs all Arrhenius).
        """
        n_rxn = self.n_reactions

        if self._veff_mask is not None:
            rates_veff = rates * self._veff_mask
            rates_bulk = rates * (~self._veff_mask)
        else:
            # Legacy fallback: EI+TE vs Arrhenius (no ion distinction)
            rates_veff = np.zeros(n_rxn)
            rates_bulk = np.zeros(n_rxn)
            for rxn in self.ei_reactions:
                rates_veff[rxn._global_index] = rates[rxn._global_index]
            for rxn in self.te_dependent_reactions:
                rates_veff[rxn._global_index] = rates[rxn._global_index]
            for rxn in self.arrhenius_reactions:
                rates_bulk[rxn._global_index] = rates[rxn._global_index]

        S_veff = self.stoich_matrix @ rates_veff
        S_bulk = self.stoich_matrix @ rates_bulk
        return S_veff, S_bulk
    
    def compute_electron_energy_loss(self, rates: np.ndarray) -> float:
        """Compute total electron energy loss [W/m³] from EI reactions.
        
        P_inelastic = sum_j (dE_j * R_j * NA) [W/m³]
        where dE_j is threshold energy [J] and R_j is rate [mol/(m³·s)].
        """
        from .constants import QE
        P_loss = 0.0
        for rxn in self.ei_reactions:
            j = rxn._global_index
            dE_J = rxn.energy_loss_eV * QE  # eV -> J per particle
            P_loss += dE_J * rates[j] * NA   # [J * mol/(m³·s) * (1/mol)] = [W/m³]
        return P_loss

    def compute_gas_heating(self, rates: np.ndarray) -> float:
        """Compute chemical enthalpy release to gas [W/m³].

        Q_rxn = -Σ(ΔH_j [kJ/mol] × 1000 × R_j [mol/(m³·s)])
        Exothermic (ΔH < 0) → Q_rxn > 0 (heats gas).
        Only non-ELECTRON_IMPACT reactions (EI energy from P_inel).
        """
        Q = 0.0
        for j in self._gas_heating_indices:
            # -ΔH * 1000 [J/mol] * R [mol/(m³·s)] = [W/m³]
            Q -= self._delta_h_kj[j] * 1000.0 * rates[j]
        return Q

    def compute_gas_heating_split(self, rates: np.ndarray):
        """Vectorized gas heating split by reaction volume."""
        idx = self._gas_heating_idx_arr
        dQ = -self._delta_h_J[idx] * rates[idx]
        Q_veff = float(np.sum(dQ[self._gas_heat_veff_mask]))
        Q_bulk = float(np.sum(dQ[~self._gas_heat_veff_mask]))
        return Q_veff, Q_bulk

    def compute_rate_derivatives(self, c: np.ndarray, T_gas: float,
                                 c_total: float,
                                 k_ei_conc: np.ndarray,
                                 dk_ei_conc: np.ndarray,
                                 Te_eV: float, eps_mean: float,
                                 c_e: float, ne_eps: float,
                                 P_gas: float = 101325.0,
                                 deps_dc_e_override: float = None,
                                 deps_dne_eps_override: float = None,
                                 deps_dTgas_override: float = None):
        """Compute analytical derivatives of all reaction rates.

        Parameters
        ----------
        deps_dc_e_override : float, optional
            Override ∂ε̄/∂c_e (for A20_power_balance mode where ε̄ comes
            from the algebraic power constraint instead of ne_eps/n_e).
        deps_dne_eps_override : float, optional
            Override ∂ε̄/∂ne_eps (= 0 in A20_power_balance mode).
        deps_dTgas_override : float, optional
            Override ∂ε̄/∂T_gas (nonzero in A20_power_balance mode because
            N_gas = P/(kB·T) depends on T_gas).

        Returns
        -------
        dR_dc : ndarray, shape (n_reactions, n_species)
            ∂R_j/∂c_k for each reaction j and species k.
        dR_dne_eps : ndarray, shape (n_reactions,)
            ∂R_j/∂(ne_eps) for each reaction j.
        dR_dTgas : ndarray, shape (n_reactions,)
            ∂R_j/∂T_gas for each reaction j.
        """
        n_rxn = self.n_reactions
        n_sp = self.n_species
        e_idx = self._electron_index

        dR_dc = np.zeros((n_rxn, n_sp))
        dR_dne_eps = np.zeros(n_rxn)
        dR_dTgas = np.zeros(n_rxn)

        c = np.maximum(c, 0.0)
        n_e = c_e * NA
        eps_mean = max(eps_mean, 1e-6)
        Te_K = Te_eV * 11604.0
        Te_K = max(Te_K, 300.0)
        T_gas_safe = max(T_gas, 200.0)

        # Chain rule quantities for ε̄ dependence:
        # Default (standard mode): ε̄ = ne_eps / n_e
        #   ∂ε̄/∂c_e = -eps_mean / c_e
        #   ∂ε̄/∂ne_eps = 1 / n_e
        #   ∂ε̄/∂T_gas = 0  (no T_gas dependence in standard mode)
        # A20_power_balance mode: ε̄ from A20(ε̄) = P/(QE·n_e·N)
        #   ∂ε̄/∂c_e = -(1/dA20_deps) · A20_target / c_e  (n_e↑ → ε̄↓)
        #   ∂ε̄/∂ne_eps = 0  (ne_eps is dummy)
        #   ∂ε̄/∂T_gas = (1/dA20_deps) · A20_target / T_gas  (T↑ → N↓ → ε̄↑)
        if deps_dc_e_override is not None:
            deps_dc_e = deps_dc_e_override
        else:
            deps_dc_e = -eps_mean / max(c_e, 1e-50)
        if deps_dne_eps_override is not None:
            deps_dne_eps = deps_dne_eps_override
        else:
            deps_dne_eps = 1.0 / max(n_e, 1.0)
        # ∂ε̄/∂T_gas: nonzero only in A20_power_balance mode
        deps_dTgas = deps_dTgas_override if deps_dTgas_override is not None else 0.0

        # --- Electron impact reactions ---
        if k_ei_conc is not None and dk_ei_conc is not None:
            for rxn in self.ei_reactions:
                j = rxn._global_index
                bi = rxn.bolsig_index
                if bi >= len(k_ei_conc):
                    continue

                k = k_ei_conc[bi]        # [m³/(mol·s)]
                dk = dk_ei_conc[bi]       # dk/dε̄ [m³/(mol·s·eV)]

                # Find target species (first non-electron reactant)
                target_idx = -1
                c_target = 0.0
                for idx in rxn.reactant_indices:
                    if idx != e_idx:
                        target_idx = idx
                        c_target = c[idx]
                        break

                # R = k(ε̄) * c_e * c_target
                # ∂R/∂c_e = c_target * (k + dk/dε̄ * ∂ε̄/∂c_e * c_e)
                #         = c_target * (k + dk * deps_dc_e * c_e)
                #         = c_target * (k - dk * eps_mean)
                dR_dc[j, e_idx] = c_target * (k + dk * deps_dc_e * c_e)

                # ∂R/∂c_target = k * c_e
                if target_idx >= 0:
                    dR_dc[j, target_idx] = k * c_e

                dR_dne_eps[j] = dk * deps_dne_eps * c_e * c_target

                # ∂R/∂T_gas: zero in standard mode, nonzero in A20_PB via ε̄(T_gas)
                dR_dTgas[j] = dk * deps_dTgas * c_e * c_target

        # --- Arrhenius reactions ---
        for i, rxn in enumerate(self.arrhenius_reactions):
            j = rxn._global_index

            # k = A * T^n * exp(-E/(R*T))
            # dk/dT = k * (n/T + E/(R*T²))
            if rxn.E != 0.0:
                k_val = rxn.A * (T_gas ** rxn.n) * np.exp(-rxn.E / (R_GAS * T_gas))
                dk_dT = k_val * (rxn.n / T_gas + rxn.E / (R_GAS * T_gas * T_gas))
            else:
                k_val = rxn.A * (T_gas ** rxn.n)
                dk_dT = k_val * rxn.n / T_gas if rxn.n != 0 else 0.0

            if rxn.order == 1:
                # R = k * c_A
                if rxn.reactant_indices:
                    idx_a = rxn.reactant_indices[0]
                    dR_dc[j, idx_a] += k_val
                    dR_dTgas[j] = dk_dT * c[idx_a]

            elif rxn.order == 2:
                if len(rxn.reactant_indices) >= 2:
                    idx_a = rxn.reactant_indices[0]
                    idx_b = rxn.reactant_indices[1]
                    if idx_a == idx_b:
                        # R = k * c_A²
                        dR_dc[j, idx_a] += 2.0 * k_val * c[idx_a]
                        dR_dTgas[j] = dk_dT * c[idx_a] * c[idx_a]
                    else:
                        # R = k * c_A * c_B
                        dR_dc[j, idx_a] += k_val * c[idx_b]
                        dR_dc[j, idx_b] += k_val * c[idx_a]
                        dR_dTgas[j] = dk_dT * c[idx_a] * c[idx_b]
                elif len(rxn.reactant_indices) == 1:
                    # R = k * c_A * c_total, c_total = P/(R*T)
                    idx_a = rxn.reactant_indices[0]
                    dR_dc[j, idx_a] += k_val * c_total
                    # ∂R/∂T = dk/dT * c_A * c_total + k * c_A * dc_total/dT
                    # dc_total/dT = -P/(R*T²) = -c_total/T
                    dR_dTgas[j] = (dk_dT * c_total - k_val * c_total / T_gas) * c[idx_a]

            elif rxn.order == 3:
                if len(rxn.reactant_indices) >= 2:
                    idx_a = rxn.reactant_indices[0]
                    idx_b = rxn.reactant_indices[1]
                    if idx_a == idx_b:
                        # R = k * c_A² * c_total
                        dR_dc[j, idx_a] += 2.0 * k_val * c[idx_a] * c_total
                        dR_dTgas[j] = (dk_dT * c_total - k_val * c_total / T_gas) \
                                      * c[idx_a] * c[idx_a]
                    else:
                        # R = k * c_A * c_B * c_total
                        dR_dc[j, idx_a] += k_val * c[idx_b] * c_total
                        dR_dc[j, idx_b] += k_val * c[idx_a] * c_total
                        dR_dTgas[j] = (dk_dT * c_total - k_val * c_total / T_gas) \
                                      * c[idx_a] * c[idx_b]
                elif len(rxn.reactant_indices) == 1:
                    # R = k * c_A * c_total²
                    idx_a = rxn.reactant_indices[0]
                    ct2 = c_total * c_total
                    dR_dc[j, idx_a] += k_val * ct2
                    dR_dTgas[j] = (dk_dT * ct2 - 2.0 * k_val * ct2 / T_gas) * c[idx_a]

        # --- Te-dependent reactions ---
        if self.te_dependent_reactions:
            c_e_val = c[e_idx]
            # dTe_K/deps = (2/3) * 11604
            dTeK_deps = (2.0 / 3.0) * 11604.0

            for rxn in self.te_dependent_reactions:
                j = rxn._global_index

                if rxn.subtype == 'AT1_KOSSYI':
                    # k3 = k3_cgs * (300/Te_K)^2 * exp(-70/Tg) *
                    #       exp(1500*(Te_K-Tg)/(Te_K*Tg))
                    # k_eff_cgs = k3 * M_cm3
                    # k_SI = k_eff_cgs * 1e-6 * NA
                    # R = k_SI * c_e * c_target
                    M_cm3 = P_gas / (KB * T_gas_safe) * 1e-6
                    k3_base = rxn.k3_cgs * (300.0 / Te_K) ** 2
                    exp_Tg = np.exp(-70.0 / T_gas_safe)
                    exp_cross = np.exp(1500.0 * (Te_K - T_gas_safe) / (Te_K * T_gas_safe))
                    k3 = k3_base * exp_Tg * exp_cross
                    k_eff_cgs = k3 * M_cm3
                    k_SI = k_eff_cgs * 1e-6 * NA

                    # Find target
                    target_idx = -1
                    c_target = 0.0
                    for idx in rxn.reactant_indices:
                        if idx != e_idx:
                            target_idx = idx
                            c_target = c[idx]
                            break

                    R_val = k_SI * c_e_val * c_target

                    # ∂R/∂c_e = k_SI * c_target  (plus ε̄-chain)
                    # dk_SI/dε̄ = k_SI * dln_k/dTe_K * dTe_K/dε̄
                    # dln_k/dTe_K = -2/Te_K + 1500*Tg/(Te_K²*Tg) (simplified)
                    #             = -2/Te_K + 1500/(Te_K²)  ... wait, let me be precise
                    # d/dTe_K [ln(k3)] = -2/Te_K + 1500*(-Tg)/(Te_K*Tg)²*Tg
                    #   = -2/Te_K + 1500*(-1)/(Te_K²)  ... no:
                    # exp_cross = exp(1500*(Te_K-Tg)/(Te_K*Tg))
                    #   = exp(1500/Tg - 1500/Te_K)
                    # d/dTe_K [1500/Tg - 1500/Te_K] = 1500/Te_K²
                    # d/dTe_K [ln(300/Te_K)^2] = -2/Te_K
                    dln_k_dTeK = -2.0 / Te_K + 1500.0 / (Te_K * Te_K)
                    dk_SI_deps = k_SI * dln_k_dTeK * dTeK_deps

                    # ∂R/∂c_e via ε̄ chain: R depends on ε̄ through k_SI
                    dR_dc[j, e_idx] += k_SI * c_target + dk_SI_deps * deps_dc_e * c_e_val * c_target
                    if target_idx >= 0:
                        dR_dc[j, target_idx] += k_SI * c_e_val

                    # ∂R/∂ne_eps
                    dR_dne_eps[j] = dk_SI_deps * deps_dne_eps * c_e_val * c_target

                    # ∂R/∂T_gas: k depends on T_gas through exp(-70/Tg), exp_cross, and M_cm3
                    # dln_k/dTg = 70/Tg² + 1500/Tg² + (-1/Tg) [from M_cm3 = P/(kB*Tg)*1e-6]
                    # Wait: M_cm3 ∝ 1/Tg, so d(ln(M_cm3))/dTg = -1/Tg
                    # d/dTg [exp(-70/Tg)] / exp(-70/Tg) = 70/Tg²
                    # d/dTg [exp(1500/Tg - 1500/Te_K)] / exp(...) = -1500/Tg²
                    # Careful: exp_cross = exp(1500*(Te_K-Tg)/(Te_K*Tg))
                    #        = exp(1500/Tg - 1500/Te_K)
                    # d/dTg = -1500/Tg²
                    dln_k_dTg = 70.0 / (T_gas_safe * T_gas_safe) \
                                - 1500.0 / (T_gas_safe * T_gas_safe) \
                                - 1.0 / T_gas_safe  # from M_cm3
                    dR_dTgas[j] = R_val * dln_k_dTg
                    # A20_PB: ε̄ also depends on T_gas → add dk/dε̄ · ∂ε̄/∂T chain
                    dR_dTgas[j] += dk_SI_deps * deps_dTgas * c_e_val * c_target

                else:
                    # DR or constant: k_cgs = A_cgs * (300/Te_K)^n_Te
                    # k_SI = k_cgs * 1e-6 * NA
                    if rxn.n_Te != 0.0:
                        k_cgs = rxn.A_cgs * (300.0 / Te_K) ** rxn.n_Te
                    else:
                        k_cgs = rxn.A_cgs
                    k_SI = k_cgs * 1e-6 * NA

                    # dk_SI/dε̄ = k_SI * (-n_Te/Te_K) * dTeK_deps
                    if rxn.n_Te != 0.0:
                        dk_SI_deps = k_SI * (-rxn.n_Te / Te_K) * dTeK_deps
                    else:
                        dk_SI_deps = 0.0

                    # R = k_SI * c_A * c_B (bimolecular)
                    if len(rxn.reactant_indices) >= 2:
                        idx_a = rxn.reactant_indices[0]
                        idx_b = rxn.reactant_indices[1]
                        R_val = k_SI * c[idx_a] * c[idx_b]

                        dR_dc[j, idx_a] += k_SI * c[idx_b]
                        dR_dc[j, idx_b] += k_SI * c[idx_a]

                        # ε̄ chain for species that is electron
                        if idx_a == e_idx:
                            dR_dc[j, e_idx] += dk_SI_deps * deps_dc_e * c[idx_a] * c[idx_b]
                        elif idx_b == e_idx:
                            dR_dc[j, e_idx] += dk_SI_deps * deps_dc_e * c[idx_a] * c[idx_b]

                        dR_dne_eps[j] = dk_SI_deps * deps_dne_eps * c[idx_a] * c[idx_b]

                        # A20_PB: ε̄(T_gas) chain rule for DR/TE bimolecular
                        dR_dTgas[j] += dk_SI_deps * deps_dTgas * c[idx_a] * c[idx_b]

                    elif len(rxn.reactant_indices) == 1:
                        idx_a = rxn.reactant_indices[0]
                        R_val = k_SI * c[idx_a]
                        dR_dc[j, idx_a] += k_SI
                        if idx_a == e_idx:
                            dR_dc[j, e_idx] += dk_SI_deps * deps_dc_e * c[idx_a]
                        dR_dne_eps[j] = dk_SI_deps * deps_dne_eps * c[idx_a]

                        # A20_PB: ε̄(T_gas) chain rule for DR/TE unimolecular
                        dR_dTgas[j] += dk_SI_deps * deps_dTgas * c[idx_a]

        return dR_dc, dR_dne_eps, dR_dTgas

    def compute_electron_loss_rate(self, rates: np.ndarray) -> float:
        """Compute total electron destruction rate [mol/(m³·s)] from DR/AT.

        Sum of reaction rates for non-EI reactions where electrons are consumed
        (net electron stoichiometry < 0). Used to compute P_e_loss = ε̄ × S × NA.
        """
        S = 0.0
        for j in self._electron_loss_indices:
            # Each reaction destroys electrons; stoich < 0, so negate
            S -= self.stoich_matrix[self._electron_index, j] * rates[j]
        return S
