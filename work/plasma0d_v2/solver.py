"""ODE solver for 0D plasma chemistry (concentration basis, ε̄-indexed).

State vector y = [c_0, c_1, ..., c_{Ns-1}, ne_eps, T_gas]
  c_i: species concentrations [mol/m³]  (c_0 = electron)
  ne_eps: n_e * eps_mean [eV/m³]
  T_gas: gas temperature [K]

RHS flow:
  1. Extract ε̄ from state: eps_mean = ne_eps / n_e
  2. Query LUT: lut.get_rate_coefficients_conc(ε̄) -> k_conc[], Te
  3. Query LUT: lut.get_transport(ε̄) -> A20, A21, A22, A23, E/N
  4. Compute elastic loss: P_el = n_e * N_gas * A21(ε̄)  [eV/(m³·s)]
  5. Compute Q_elastic for gas: Q_el_Wm3 = P_el * QE    [W/m³]
"""

import numpy as np
from scipy.integrate import solve_ivp
import time as time_module
from dataclasses import dataclass
from typing import Optional

from .constants import QE, NA, KB, ME, R_GAS, total_concentration


@dataclass
class SimulationResult:
    t: np.ndarray
    y: np.ndarray
    species_names: list
    n_species: int

    concentrations: Optional[np.ndarray] = None
    Te_eV: Optional[np.ndarray] = None
    T_gas: Optional[np.ndarray] = None
    ne_m3: Optional[np.ndarray] = None
    EN_Td: Optional[np.ndarray] = None
    eps_mean_eV: Optional[np.ndarray] = None
    power_Wm3: Optional[np.ndarray] = None

    wall_time: float = 0.0
    n_rhs_evals: int = 0
    solver_message: str = ""


class PlasmaODESolver:

    def __init__(self, species_manager, reaction_set, boltzmann_lut,
                 power_source, electron_kinetics, flow_model, gas_thermal,
                 qn_mode: str = 'A', V_eff: float = 1e-6, V_reactor: float = 1e-6,
                 energy_source: str = 'constant'):
        self.sm = species_manager
        self.rxn = reaction_set
        self.lut = boltzmann_lut
        self.power = power_source
        self.ekin = electron_kinetics
        self.flow = flow_model
        self.gth = gas_thermal

        # Volume parameters (used by power.py and flow.py, NOT for dilution)
        self._V_eff = V_eff
        self._V_reactor = V_reactor

        # Energy source mode: 'constant' (external P_dep) or 'A20' (self-consistent)
        # 'A20': P_abs = n_e * N * A20(ε̄)  — power depends on ε̄ and N(T_gas)
        self._energy_source = energy_source

        self._rhs_count = 0
        self._jac_count = 0
        self._nb_params = None
        self._rhs_numba = None
        self._concentration_floor = 1e-30
        self._A22_current = 0.0  # current A22 value, updated each RHS call
        self._ne_seed = 1e8                       # background electron density [m⁻³] (initial cond only)
        self._ce_floor = 1e-30                     # electron floor ≈ 0 (ZDPlasKin style: clower=0)
        self._ne_eps_floor = 1e-20                 # ne*eps floor: small but avoids 0 energy density

        # Quasi-neutrality mode: 'A' = diagnostic only, 'B' = derive n_e from ions
        self._qn_mode = qn_mode
        self._positive_ion_indices: list = []
        self._negative_ion_indices: list = []
        self._build_ion_indices()

        # Initialize veff_mask as all-True (needed by numba_core gas_heat)
        self.rxn._init_veff_mask()
        self.rxn._build_gas_heating_arrays()

        # Cache LUT boundary for 3-stage afterglow transition
        if self.lut is not None:
            self._eps_min_lut = self.lut.eps_range[0]
            transport_boundary = self.lut.get_transport(self._eps_min_lut)
            self._A21_at_boundary = transport_boundary.elastic_power_N
            self._A22_at_boundary = transport_boundary.inelastic_power_N
        else:
            self._eps_min_lut = 0.1
            self._A21_at_boundary = 0.0
            self._A22_at_boundary = 0.0

    def _build_ion_indices(self):
        """Pre-compute positive/negative ion species indices for QN enforcement."""
        from .species import SpeciesType
        for sp in self.sm:
            if sp.type == SpeciesType.ION_POSITIVE:
                self._positive_ion_indices.append(sp.index)
            elif sp.type == SpeciesType.ION_NEGATIVE:
                self._negative_ion_indices.append(sp.index)
        if self._qn_mode == 'B' and self._positive_ion_indices:
            print(f"  QN Mode B: n_e derived from {len(self._positive_ion_indices)} positive, "
                  f"{len(self._negative_ion_indices)} negative ion species")

    def jacobian(self, t, y):
        """Compute the analytical Jacobian matrix ∂(dy/dt)/∂y.

        Returns a (n_state, n_state) dense matrix.
        """
        self._jac_count += 1
        n_sp = self.sm.n_species
        n_state = self.sm.n_state
        idx_energy = self.sm.idx_energy
        idx_Tgas = self.sm.idx_Tgas

        # --- Unpack and clamp state (same as rhs) ---
        c = y[:n_sp].copy()
        ne_eps = y[idx_energy]
        T_gas = y[idx_Tgas]

        c = np.maximum(c, self._concentration_floor)
        ne_eps = np.clip(ne_eps, self._ne_eps_floor, 1e35)
        T_gas = np.clip(T_gas, 200.0, 10000.0)

        c_e = c[0]
        n_e = min(c_e * NA, 1e26)

        T_gas_safe = max(T_gas, 200.0)
        eps_thermal = 1.5 * KB * T_gas_safe / QE

        if n_e > 1.0:
            eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0)
        else:
            eps_mean = max(1.0, eps_thermal)

        c_total = total_concentration(self.power.P_gas, T_gas)
        N_gas = self.power.P_gas / (KB * T_gas_safe)

        # --- A20 power balance: override eps_mean (same as rhs) ---
        if self._energy_source == 'A20_power_balance' and self.lut is not None:
            P_target = self.power.get_power_density(t)
            A20_target = P_target / (QE * max(n_e, 1.0) * max(N_gas, 1e20))
            eps_mean = self.lut.invert_A20(A20_target)

        Te_eV = (2.0 / 3.0) * eps_mean

        # --- LUT queries (same 3-stage logic as rhs) ---
        k_ei_conc = None
        dk_ei_conc = None
        A21 = 0.0
        dA21_deps = 0.0

        if self.lut is not None:
            if eps_mean >= self._eps_min_lut:
                N_gas_cm3_jac = N_gas * 1e-6
                k_ei_conc, _ = self.lut.get_rate_coefficients_conc(eps_mean, N_gas_cm3=N_gas_cm3_jac)
                dk_ei_conc = self.lut.get_rate_derivatives_conc(eps_mean)
                transport = self.lut.get_transport(eps_mean)
                A21 = transport.elastic_power_N
                dA21_deps = self.lut.get_transport_deriv(eps_mean)
            elif eps_mean > eps_thermal:
                # Stage 2: analytical elastic cooling
                denom = self._eps_min_lut - eps_thermal
                if denom > 1e-6:
                    A21 = self._A21_at_boundary * (eps_mean - eps_thermal) / denom
                    dA21_deps = self._A21_at_boundary / denom
                # k_ei_conc = None → EI reactions off

        # --- Compute reaction rates (needed for electron loss terms) ---
        rates = self.rxn.compute_reaction_rates(
            c, T_gas, c_total, k_ei_conc,
            Te_eV=Te_eV, P_gas=self.power.P_gas
        )

        # --- Rate derivatives ---
        if k_ei_conc is None:
            k_ei_conc_safe = np.zeros(self.rxn.n_ei)
        else:
            k_ei_conc_safe = k_ei_conc
        if dk_ei_conc is None:
            dk_ei_conc_safe = np.zeros(self.rxn.n_ei)
        else:
            dk_ei_conc_safe = dk_ei_conc

        # Power-constraint derivatives for A20_PB mode
        # A20(ε̄) = P / (QE·n_e·N)  ⟹  ∂ε̄/∂c_e, ∂ε̄/∂ne_eps, ∂ε̄/∂T_gas
        deps_overrides = {}
        if self._energy_source == 'A20_power_balance' and self.lut is not None:
            dA20_deps_val = self.lut._interp_A20_deriv(eps_mean)
            dA20_deps_safe = max(abs(dA20_deps_val), 1e-50) * (1.0 if dA20_deps_val >= 0 else -1.0)
            A20_at_eps = self.lut.get_transport(eps_mean).power_N
            # ∂ε̄/∂c_e = -(A20 / dA20_deps) / c_e  (n_e = c_e·NA, A20_target ∝ 1/n_e)
            deps_overrides['deps_dc_e_override'] = -(A20_at_eps / dA20_deps_safe) / max(c_e, 1e-50)
            deps_overrides['deps_dne_eps_override'] = 0.0
            # ∂ε̄/∂T_gas = +(A20 / dA20_deps) / T_gas  (N ∝ 1/T → A20_target ∝ T)
            deps_overrides['deps_dTgas_override'] = (A20_at_eps / dA20_deps_safe) / T_gas_safe

        dR_dc, dR_dne_eps, dR_dTgas = self.rxn.compute_rate_derivatives(
            c, T_gas, c_total,
            k_ei_conc_safe, dk_ei_conc_safe,
            Te_eV, eps_mean, c_e, ne_eps,
            P_gas=self.power.P_gas,
            **deps_overrides
        )

        # --- Assemble Jacobian ---
        J = np.zeros((n_state, n_state))
        stoich = self.rxn.stoich_matrix  # (n_sp, n_rxn)

        J[:n_sp, :n_sp] = stoich @ dR_dc
        J[:n_sp, idx_energy] = stoich @ dR_dne_eps
        J[:n_sp, idx_Tgas] = stoich @ dR_dTgas

        # (4) Flow Jacobian for species: dc_i/dt|_flow = (c_in_i - c_i)/tau
        #     ∂(dc_i/dt|_flow)/∂c_i = -1/tau
        #     ∂(dc_i/dt|_flow)/∂T_gas = (c_in_i - c_i) * ∂(1/tau)/∂T
        #     tau ∝ 1/T → 1/tau ∝ T → ∂(1/tau)/∂T = 1/(tau*T)
        tau = self.flow.get_residence_time(T_gas)
        if tau > 0 and tau < 1e9:
            inv_tau = 1.0 / tau
            for i in range(n_sp):
                J[i, i] -= inv_tau
            # ∂/∂T of flow source with T-dependent c_inlet:
            #   flow_i = (x_i * P/(R*T) - c_i) / tau
            #   ∂flow_i/∂T = ∂/∂T[(x_i*P/(R*T))/tau] - ∂/∂T[c_i/tau]
            #   Since tau ∝ T: 1/tau ∝ 1/T, so x_i*P/(R*T) / tau ∝ 1/T²
            #   and c_i/tau ∝ 1/T.  Net: ∂flow_i/∂T = -c_i * inv_tau / T
            #   (the x_i*P/(R*T)/tau term has ∂/∂T = -2*x_i*P/(R*T²*tau) + ... 
            #    but simplifies to: d/dT[(x*P/R - c*T)/(tau_0*T²)] ... )
            # Exact derivation:
            #   flow_i = [x_i*P/(R*T) - c_i] / tau(T)
            #   tau = V/Q_act, Q_act ∝ T → tau = tau_0 * T_STP/T → 1/tau = T/(tau_0*T_STP)
            #   flow_i = [x_i*P/(R*T) - c_i] * T / (tau_0*T_STP)
            #          = x_i*P/(R*tau_0*T_STP) - c_i*T/(tau_0*T_STP)
            #   ∂flow_i/∂T = 0 - c_i/(tau_0*T_STP) = -c_i * inv_tau / T
            J[:n_sp, idx_Tgas] += -c[:n_sp] * inv_tau / T_gas_safe

        # (4b) Ambipolar diffusion loss Jacobian for charged species
        # dydt[i] += -c[i] * diff_freq, where diff_freq = K * Te_eV
        # K = mu_i_N / (N_gas * Λ²),  Te_eV = (2/3)*ε̄
        # ε̄ derivatives depend on mode (standard vs A20_PB)
        diff_freq = self.ekin.compute_diffusion_rate(T_gas, Te_eV)
        Te_eV_safe = max(Te_eV, 1e-3)
        c_e_safe = max(c_e, 1e-30)
        n_e_safe = max(n_e, 1.0)
        K_diff = diff_freq / Te_eV_safe  # = mu_i_N / (N_gas * Λ²)

        # Compute ∂(diff_freq)/∂(state) = K * (2/3) * ∂ε̄/∂(state)
        # Plus ∂K/∂T_gas = K/T (from N_gas ∝ 1/T)
        if self._energy_source == 'A20_power_balance' and deps_overrides:
            deps_dc_e_val = deps_overrides['deps_dc_e_override']
            deps_dne_eps_val = 0.0
            deps_dTgas_val = deps_overrides['deps_dTgas_override']
        else:
            deps_dc_e_val = -eps_mean / c_e_safe
            deps_dne_eps_val = 1.0 / n_e_safe
            deps_dTgas_val = 0.0

        dfreq_dc_e = K_diff * (2.0/3.0) * deps_dc_e_val
        dfreq_dne_eps = K_diff * (2.0/3.0) * deps_dne_eps_val
        # ∂diff_freq/∂T_gas = K*(2/3)*∂ε̄/∂T + diff_freq/T (from ∂K/∂T = K/T)
        dfreq_dTgas = K_diff * (2.0/3.0) * deps_dTgas_val + diff_freq / T_gas_safe

        # Electron: dydt[0] -= c_e * diff_freq
        # ∂/∂c_e = -(diff_freq + c_e * dfreq_dc_e)
        J[0, 0] -= diff_freq + c[0] * dfreq_dc_e
        J[0, idx_energy] -= c[0] * dfreq_dne_eps
        J[0, idx_Tgas] -= c[0] * dfreq_dTgas

        # Ions: dydt[i] -= c[i] * diff_freq
        for idx in self._positive_ion_indices:
            J[idx, idx] -= diff_freq
            J[idx, 0] -= c[idx] * dfreq_dc_e
            J[idx, idx_energy] -= c[idx] * dfreq_dne_eps
            J[idx, idx_Tgas] -= c[idx] * dfreq_dTgas
        for idx in self._negative_ion_indices:
            J[idx, idx] -= diff_freq
            J[idx, 0] -= c[idx] * dfreq_dc_e
            J[idx, idx_energy] -= c[idx] * dfreq_dne_eps
            J[idx, idx_Tgas] -= c[idx] * dfreq_dTgas

        # NOTE: Jacobian row zeroing for clamped species REMOVED.
        # Previously zeroed J[i,:] when y[i] < floor and dydt[i] < 0,
        # but this caused 60/65 species rows to be zeroed at initial conditions
        # (where most species start at 0), making BDF take 76x more steps.
        # The RHS floor guard (max(dydt,0)) is kept for stability;
        # the Jacobian mismatch is small and BDF handles it via DQ correction.

        # (6) Electron energy row
        P_el_eVm3s = n_e * N_gas * A21
        S_e_loss = self.rxn.compute_electron_loss_rate(rates)
        P_e_loss_eVm3s = eps_mean * S_e_loss * NA

        # Determine P_dep mode
        A20 = 0.0
        dA20_deps = 0.0
        if self._energy_source == 'A20' and self.lut is not None and eps_mean >= self._eps_min_lut:
            transport_p = self.lut.get_transport(eps_mean)
            A20 = transport_p.power_N
            P_dep = n_e * N_gas * A20 * QE
            dA20_deps = self.lut._interp_A20_deriv(eps_mean)
        else:
            P_dep = self.power.get_power_density(t)

        # P_inel derivatives: use A22 when active (consistent with RHS)
        use_A22_jac = (self._A22_current > 0 and self.lut is not None
                       and eps_mean >= self._eps_min_lut)
        A22_val = 0.0
        dA22_deps = 0.0
        if use_A22_jac:
            A22_val = self._A22_current
            dA22_deps = self.lut.get_inelastic_power_deriv(eps_mean)

        n_eloss = len(self.rxn._electron_loss_indices)
        dR_dc_eloss = np.zeros((n_eloss, n_sp))
        dR_dne_eps_eloss = np.zeros(n_eloss)
        for i_el, jj in enumerate(self.rxn._electron_loss_indices):
            dR_dc_eloss[i_el, :] = dR_dc[jj, :]
            dR_dne_eps_eloss[i_el] = dR_dne_eps[jj]

        # Build energy Jacobian row
        n_state_loc = n_state
        jac_row = np.zeros(n_state_loc)

        eps_safe = max(eps_mean, 1e-6)
        c_e_safe = max(c_e, 1e-50)
        n_e_safe = max(n_e, 1.0)
        T_gas_safe_e = max(T_gas, 200.0)
        deps_dc_e = -eps_safe / c_e_safe
        deps_dne_eps = 1.0 / n_e_safe

        # P_dep derivatives (only for A20 mode)
        dPdep_dc_e = 0.0; dPdep_dne_eps = 0.0; dPdep_dTgas = 0.0
        if self._energy_source == 'A20' and A20 > 0:
            dPdep_dc_e = NA * N_gas * A20 + n_e * N_gas * dA20_deps * deps_dc_e
            dPdep_dne_eps = n_e * N_gas * dA20_deps * deps_dne_eps
            dPdep_dTgas = n_e * (-N_gas / T_gas_safe_e) * A20

        # P_elastic derivatives: P_el = n_e * N_gas * A21(ε̄)
        dPel_dc_e = NA * N_gas * A21 + n_e * N_gas * dA21_deps * deps_dc_e
        dPel_dne_eps = n_e * N_gas * dA21_deps * deps_dne_eps
        dPel_dTgas = n_e * (-N_gas / T_gas_safe_e) * A21

        # P_inel derivatives
        dPinel_dc = np.zeros(n_sp)
        dPinel_dne_eps = 0.0
        dPinel_dTgas = 0.0
        if use_A22_jac:
            # A22-based: P_inel_eV = n_e * N_gas * A22(ε̄)
            dPinel_dc[0] = NA * N_gas * A22_val + n_e * N_gas * dA22_deps * deps_dc_e
            dPinel_dne_eps = n_e * N_gas * dA22_deps * deps_dne_eps
            dPinel_dTgas = n_e * (-N_gas / T_gas_safe_e) * A22_val
        else:
            # Reaction-based fallback
            for i_ei, rxn in enumerate(self.rxn.ei_reactions):
                j_rxn = rxn._global_index
                dE_eV = rxn.energy_loss_eV
                coeff = dE_eV * NA
                dPinel_dc += coeff * dR_dc[j_rxn, :]
                dPinel_dne_eps += coeff * dR_dne_eps[j_rxn]

        # P_diff derivatives: P_diff = ne_eps * D_a / Λ²
        mu_i_N = 2.8e22
        N_gas_v = 101325.0 / (KB * T_gas_safe_e)
        mu_i = mu_i_N / N_gas_v
        Te_eV_loc = (2.0 / 3.0) * eps_safe
        D_a = mu_i * Te_eV_loc
        dDa_deps = mu_i * (2.0 / 3.0)
        inv_L2 = 1.0 / self.ekin.Lambda_sq

        dPdiff_dc_e = ne_eps * inv_L2 * dDa_deps * deps_dc_e
        dPdiff_dne_eps = D_a * inv_L2 + ne_eps * inv_L2 * dDa_deps * deps_dne_eps
        dPdiff_dTgas = ne_eps * inv_L2 * D_a / T_gas_safe_e

        # P_flow derivatives: P_flow = ne_eps / τ
        inv_tau = 1.0 / tau if tau > 0 else 0.0
        dPflow_dne_eps = inv_tau
        dPflow_dTgas = ne_eps * inv_tau / T_gas_safe_e

        # P_e_loss derivatives: P_eloss = ε̄ * S_e_loss * NA
        e_idx = 0
        dPeloss_dc = np.zeros(n_sp)
        dPeloss_dne_eps = 0.0
        for i_el, jj in enumerate(self.rxn._electron_loss_indices):
            nu_e = -self.rxn.stoich_matrix[e_idx, jj]
            dPeloss_dc += eps_safe * NA * nu_e * dR_dc_eloss[i_el, :]
            dPeloss_dne_eps += eps_safe * NA * nu_e * dR_dne_eps_eloss[i_el]
        dPeloss_dc[e_idx] += deps_dc_e * S_e_loss * NA
        dPeloss_dne_eps += deps_dne_eps * S_e_loss * NA

        # Assemble: rhs = P_dep_eV - P_el - P_inel_eV - P_diff - P_flow - P_eloss
        jac_row[:n_sp] = -dPinel_dc - dPeloss_dc
        jac_row[e_idx] += dPdep_dc_e - dPel_dc_e - dPdiff_dc_e
        jac_row[idx_energy] = (dPdep_dne_eps - dPel_dne_eps - dPinel_dne_eps
                               - dPdiff_dne_eps - dPflow_dne_eps - dPeloss_dne_eps)
        jac_row[idx_Tgas] = dPdep_dTgas - dPel_dTgas - dPinel_dTgas - dPdiff_dTgas - dPflow_dTgas

        J[idx_energy, :] = jac_row

        # (7) Gas temperature row
        Q_elastic_Wm3 = P_el_eVm3s * QE
        Q_rxn_Wm3 = self.rxn.compute_gas_heating(rates)
        Q_e_loss_Wm3 = P_e_loss_eVm3s * QE

        # Q_el = n_e * N_gas * A21(ε̄) * QE
        # Use mode-dependent ε̄ derivatives (same as passed to rate_derivatives)
        if self._energy_source == 'A20_power_balance' and deps_overrides:
            deps_dc_e = deps_overrides['deps_dc_e_override']
            deps_dne_eps = 0.0
            deps_dTgas_eps = deps_overrides['deps_dTgas_override']
        else:
            deps_dc_e = -eps_mean / max(c_e, 1e-50)
            deps_dne_eps = 1.0 / max(n_e, 1.0)
            deps_dTgas_eps = 0.0

        dQel_dc = np.zeros(n_sp)
        dQel_dc[0] = (NA * N_gas * A21 + n_e * N_gas * dA21_deps * deps_dc_e) * QE
        dQel_dne_eps = n_e * N_gas * dA21_deps * deps_dne_eps * QE
        # ∂Q_el/∂T_gas: N_gas dependence + ε̄(T_gas) chain in A20_PB
        dQel_dTgas = (n_e * (-N_gas / T_gas_safe) * A21
                      + n_e * N_gas * dA21_deps * deps_dTgas_eps) * QE

        # dQ_rxn/dc_k: Q_rxn = -Σ(ΔH_j * 1000 * R_j) for gas-heating reactions
        # ∂Q_rxn/∂c_k = -Σ(ΔH_j * 1000 * ∂R_j/∂c_k)
        dQrxn_dc = np.zeros(n_sp)
        dQrxn_dne_eps = 0.0
        dQrxn_dTgas = 0.0
        for jj in self.rxn._gas_heating_indices:
            dh = self.rxn._delta_h_kj[jj]
            coeff = -dh * 1000.0
            dQrxn_dc += coeff * dR_dc[jj, :]
            dQrxn_dne_eps += coeff * dR_dne_eps[jj]
            dQrxn_dTgas += coeff * dR_dTgas[jj]

        # Q_eloss = ε̄ · S_e_loss · NA · QE
        dQeloss_dc = np.zeros(n_sp)
        dQeloss_dne_eps = 0.0
        dQeloss_dTgas = 0.0
        e_idx = 0
        for i_el, jj in enumerate(self.rxn._electron_loss_indices):
            nu_e = -self.rxn.stoich_matrix[e_idx, jj]
            dQeloss_dc += eps_mean * NA * QE * nu_e * dR_dc[jj, :]
            dQeloss_dne_eps += eps_mean * NA * QE * nu_e * dR_dne_eps[jj]
            dQeloss_dTgas += eps_mean * NA * QE * nu_e * dR_dTgas[jj]
        # Chain rule on ε̄ (mode-dependent deps already set above)
        dQeloss_dc[e_idx] += deps_dc_e * S_e_loss * NA * QE
        dQeloss_dne_eps += deps_dne_eps * S_e_loss * NA * QE
        dQeloss_dTgas += deps_dTgas_eps * S_e_loss * NA * QE

        tgas_row = self.gth.compute_Tgas_jacobian_row(
            T_gas, Q_elastic_Wm3, tau, Q_rxn_Wm3, Q_e_loss_Wm3,
            dQel_dc, dQel_dne_eps, dQel_dTgas,
            dQrxn_dc, dQrxn_dne_eps, dQrxn_dTgas,
            dQeloss_dc, dQeloss_dne_eps,
            n_state,
            dQeloss_dTgas=dQeloss_dTgas
        )
        J[idx_Tgas, :] = tgas_row

        return J

    def _setup_numba(self):
        """Setup Numba JIT RHS (called once)."""
        try:
            from .numba_core import rhs_numba, extract_numba_params
            p = extract_numba_params(self)
            self._rhs_numba = rhs_numba
            # Store as tuple in fixed order matching rhs_numba signature
            self._nb_args = (
                p['nsp'], p['ie'], p['iT'],
                p['lgrid'], p['ltab'], p['kdead'], p['snm'], p['egrid'],
                p['elN'], p['inN'], p['poN'],
                p['emin'], p['A21b'], p['A22b'], p['kth'],
                p['nrxn'], p['SM'],
                p['eg'], p['eb'], p['et'], p['nei'], p['eel'],
                p['ag2'], p['aA'], p['an2'], p['aE'], p['ao'],
                p['aa'], p['ab'], p['anr'], p['narr'],
                p['tg2'], p['ts2'], p['tA2'], p['tn2'], p['tk2'],
                p['ta3'], p['tb3'], p['tnr3'], p['tt2'], p['nte'],
                p['hi'], p['dh'],
                p['eli'], p['ese'],
                p['cef'], p['cf'], p['nef'],
                p['pii'], p['nii'], p['npi'], p['nni'],
                p['xi'], p['Vr'], p['Qs'], p['Pg'], p['pfr'],
                p['Tw'], p['wlf'], p['Ma'], p['cp'], p['Lsq'],
            )
            # Warm up JIT
            _y = np.zeros(self.sm.n_species + 2)
            _y[self.sm.idx_Tgas] = 300.0
            _Pdep = self.power.get_power_density(0.0)
            self._rhs_numba(0.0, _y, *self._nb_args, _Pdep)
            print("  Numba RHS: compiled and ready")
        except Exception as e:
            self._nb_args = None
            self._rhs_numba = None
            print(f"  Numba RHS: unavailable ({e})")

    def rhs(self, t, y):
        self._rhs_count += 1

        # Numba fast path
        if self._rhs_numba is not None:
            Pdep = self.power.get_power_density(t)
            return self._rhs_numba(t, y, *self._nb_args, Pdep)

        n_sp = self.sm.n_species

        # --- Unpack state ---
        c = y[:n_sp].copy()
        ne_eps = y[self.sm.idx_energy]
        T_gas = y[self.sm.idx_Tgas]

        c = np.maximum(c, self._concentration_floor)
        ne_eps = np.clip(ne_eps, self._ne_eps_floor, 1e35)
        T_gas = np.clip(T_gas, 200.0, 10000.0)

        c_e = c[0]
        n_e = min(c_e * NA, 1e26)

        # --- Mean electron energy & thermal floor ---
        T_gas_safe = max(T_gas, 200.0)
        eps_thermal = 1.5 * KB * T_gas_safe / QE  # (3/2)kT [eV]

        if n_e > 1.0:
            eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0)
        else:
            eps_mean = max(1.0, eps_thermal)

        # --- Total gas concentration and number density ---
        c_total = total_concentration(self.power.P_gas, T_gas)
        N_gas = self.power.P_gas / (KB * T_gas_safe)

        # --- A20 power balance: override eps_mean from power constraint ---
        # P_input/V_eff = n_e * N_gas * A20(ε̄) * QE  →  solve for ε̄
        # When T_gas↑, N↓ → ε̄ must increase → Te↑ (correct physics)
        if self._energy_source == 'A20_power_balance' and self.lut is not None:
            P_target = self.power.get_power_density(t)  # P_input/V_eff [W/m³]
            A20_target = P_target / (QE * max(n_e, 1.0) * max(N_gas, 1e20))
            eps_mean = self.lut.invert_A20(A20_target)

        # --- 3-stage afterglow transition ---
        k_ei_conc = None
        Te_eV = (2.0 / 3.0) * eps_mean
        P_el_eVm3s = 0.0
        Q_elastic_Wm3 = 0.0
        N_gas_cm3 = N_gas * 1e-6

        if self.lut is not None:
            if eps_mean >= self._eps_min_lut:
                # Stage 1: LUT region — normal BOLSIG+ query
                k_ei_conc, Te_eV = self.lut.get_rate_coefficients_conc(
                    eps_mean, N_gas_cm3=N_gas_cm3)
                transport = self.lut.get_transport(eps_mean)
                P_el_eVm3s = n_e * N_gas * transport.elastic_power_N
                Q_elastic_Wm3 = P_el_eVm3s * QE
                self._A22_current = transport.inelastic_power_N

            elif eps_mean > eps_thermal:
                # Stage 2: Below LUT, above thermal
                # Cooling: A21/A22 linear interpolation to thermal
                denom = self._eps_min_lut - eps_thermal
                if denom > 1e-6:
                    A21_eff = self._A21_at_boundary * (eps_mean - eps_thermal) / denom
                    A22_eff = self._A22_at_boundary * (eps_mean - eps_thermal) / denom
                else:
                    A21_eff = 0.0
                    A22_eff = 0.0
                P_el_eVm3s = n_e * N_gas * A21_eff
                Q_elastic_Wm3 = P_el_eVm3s * QE
                self._A22_current = A22_eff
                # EI rates from Maxwellian fallback (GlobalKin-style)
                # At thermal Te, Maxwellian gives correct low attachment rates
                k_ei_conc, _ = self.lut.get_rate_coefficients_conc(
                    eps_mean, N_gas_cm3=N_gas_cm3, fallback_maxwellian=True)

            else:
                # Stage 3 — at/below thermal
                self._A22_current = 0.0
                # EI rates from Maxwellian at thermal Te
                k_ei_conc, _ = self.lut.get_rate_coefficients_conc(
                    eps_mean, N_gas_cm3=N_gas_cm3, fallback_maxwellian=True)

        # --- QN Method B: derive electron concentration from ion balance ---
        # Only apply when total ion concentration is significant relative to electron
        if self._qn_mode == 'B' and self._positive_ion_indices:
            c_ion_total = 0.0
            for idx in self._positive_ion_indices:
                c_ion_total += c[idx]
            # Activate QN only when ions are significant (>1% of electron concentration)
            if c_ion_total > 0.01 * c_e:
                c_e_qn = c_ion_total
                for idx in self._negative_ion_indices:
                    c_e_qn -= c[idx]
                c_e_qn = max(c_e_qn, self._concentration_floor)
                c[0] = c_e_qn
                c_e = c_e_qn
                n_e = min(c_e * NA, 1e26)
                if n_e > 1.0:
                    eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0)
                Te_eV = (2.0 / 3.0) * eps_mean
                # Re-compute elastic loss with QN-derived n_e
                if self.lut is not None and eps_mean >= self._eps_min_lut:
                    transport = self.lut.get_transport(eps_mean)
                    P_el_eVm3s = n_e * N_gas * transport.elastic_power_N
                    Q_elastic_Wm3 = P_el_eVm3s * QE
                    self._A22_current = transport.inelastic_power_N
                elif self.lut is not None and eps_mean > eps_thermal:
                    denom = self._eps_min_lut - eps_thermal
                    if denom > 1e-6:
                        A21_eff = self._A21_at_boundary * (eps_mean - eps_thermal) / denom
                        A22_eff = self._A22_at_boundary * (eps_mean - eps_thermal) / denom
                    else:
                        A21_eff = 0.0
                        A22_eff = 0.0
                    P_el_eVm3s = n_e * N_gas * A21_eff
                    Q_elastic_Wm3 = P_el_eVm3s * QE
                    self._A22_current = A22_eff
                else:
                    self._A22_current = 0.0

        # --- Reaction rates ---
        rates = self.rxn.compute_reaction_rates(
            c, T_gas, c_total, k_ei_conc,
            Te_eV=Te_eV, P_gas=self.power.P_gas
        )

        # --- Species source terms (no volume dilution) ---
        S = self.rxn.compute_source_terms(rates)
        S_flow = self.flow.compute_flow_source(c, T_gas)

        dydt = np.zeros_like(y)
        dydt[:n_sp] = S[:n_sp] + S_flow[:n_sp]

        # --- Ambipolar diffusion loss for charged species ---
        # dc_charged/dt += -c_charged * D_a / Λ²
        # Same D_a as energy equation (consistency)
        diff_freq = self.ekin.compute_diffusion_rate(T_gas, Te_eV)  # [1/s]
        dydt[0] -= c[0] * diff_freq  # electron
        for idx in self._positive_ion_indices:
            dydt[idx] -= c[idx] * diff_freq
        for idx in self._negative_ion_indices:
            dydt[idx] -= c[idx] * diff_freq

        y_sp = y[:n_sp]
        if y_sp[0] < self._ce_floor:
            dydt[0] = max(dydt[0], 0.0)
        for i in range(1, n_sp):
            if y_sp[i] < self._concentration_floor:
                dydt[i] = max(dydt[i], 0.0)

        # QN Mode B: zero out electron ODE when ions are significant
        if self._qn_mode == 'B' and self._positive_ion_indices:
            c_ion_sum = 0.0
            for idx in self._positive_ion_indices:
                c_ion_sum += c[idx]
            if c_ion_sum > 0.01 * c[0]:
                dydt[0] = 0.0

        # --- Electron energy equation ---
        # Electron energy is solved in V_eff — no dilution on electron terms
        _skip_energy_ode = False
        if self._energy_source == 'A20' and self.lut is not None and eps_mean >= self._eps_min_lut:
            # Unconstrained A20: P_abs = n_e * N_gas * A20(ε̄)  [W/m³]
            transport_for_power = self.lut.get_transport(eps_mean)
            A20 = transport_for_power.power_N
            P_dep = n_e * N_gas * A20 * QE  # [W/m³]
        elif self._energy_source == 'A20_power_balance':
            # ε̄ for rates comes from power constraint (set above).
            # Energy ODE still runs: residual (P_dep - losses) drives ne_eps.
            P_dep = self.power.get_power_density(t)
        else:
            P_dep = self.power.get_power_density(t)

        # Use BOLSIG+ A22 for total inelastic loss (includes vibrational excitation)
        if self._A22_current > 0:
            P_inel = n_e * N_gas * self._A22_current * QE  # [W/m³]
        else:
            P_inel = self.rxn.compute_electron_energy_loss(rates)  # fallback
        tau = self.flow.get_residence_time(T_gas)

        # Phase 6: electron destruction loss (DR/AT + EI attachment)
        S_e_loss = self.rxn.compute_electron_loss_rate(rates)  # [mol/(m³·s)]
        P_e_loss_eVm3s = eps_mean * S_e_loss * NA  # [eV/(m³·s)]

        if not _skip_energy_ode:
            dydt[self.sm.idx_energy] = self.ekin.compute_energy_rhs(
                ne_eps, c_e, T_gas, P_dep, P_el_eVm3s, P_inel, tau,
                P_e_loss_eVm3s=P_e_loss_eVm3s
            )

        # --- Gas temperature equation (no volume dilution) ---
        Q_rxn_Wm3 = self.rxn.compute_gas_heating(rates)
        Q_e_loss_Wm3 = P_e_loss_eVm3s * QE  # [eV/(m³·s)] * [J/eV] = [W/m³]

        dydt[self.sm.idx_Tgas] = self.gth.compute_Tgas_rhs(
            T_gas, Q_elastic_Wm3, tau,
            Q_rxn_Wm3=Q_rxn_Wm3,
            Q_e_loss_Wm3=Q_e_loss_Wm3
        )

        np.nan_to_num(dydt, copy=False, nan=0.0, posinf=1e30, neginf=-1e30)
        return dydt

    def rhs_off(self, t, y):
        """OFF-phase RHS for operator splitting.

        Same as full RHS but P_dep = 0.
        All reactions active: EI (at thermal Te), TE_DEPENDENT, Arrhenius.
        """
        self._rhs_count += 1
        n_sp = self.sm.n_species

        c = y[:n_sp].copy()
        ne_eps = y[self.sm.idx_energy]
        T_gas = y[self.sm.idx_Tgas]

        c = np.maximum(c, self._concentration_floor)
        ne_eps = np.clip(ne_eps, self._ne_eps_floor, 1e35)
        T_gas = np.clip(T_gas, 200.0, 10000.0)

        c_e = c[0]
        n_e = min(c_e * NA, 1e26)
        T_gas_safe = max(T_gas, 200.0)
        eps_thermal = 1.5 * KB * T_gas_safe / QE

        if n_e > 1.0:
            eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0)
        else:
            eps_mean = max(1.0, eps_thermal)

        c_total = total_concentration(self.power.P_gas, T_gas)
        N_gas = self.power.P_gas / (KB * T_gas_safe)
        N_gas_cm3 = N_gas * 1e-6

        # EI rate coefficients at current eps_mean (same 3-stage logic as full RHS)
        k_ei_conc = None
        Te_eV = (2.0 / 3.0) * eps_mean

        if self.lut is not None:
            if eps_mean >= self._eps_min_lut:
                k_ei_conc, Te_eV = self.lut.get_rate_coefficients_conc(
                    eps_mean, N_gas_cm3=N_gas_cm3)
            else:
                k_ei_conc, _ = self.lut.get_rate_coefficients_conc(
                    eps_mean, N_gas_cm3=N_gas_cm3, fallback_maxwellian=True)

        # All reactions: EI + TE_DEPENDENT + Arrhenius
        rates = self.rxn.compute_reaction_rates(
            c, T_gas, c_total, k_ei_conc,
            Te_eV=Te_eV, P_gas=self.power.P_gas
        )

        S = self.rxn.compute_source_terms(rates)
        S_flow = self.flow.compute_flow_source(c, T_gas)

        dydt = np.zeros_like(y)
        dydt[:n_sp] = S[:n_sp] + S_flow[:n_sp]

        # Diffusion decay (electrons and ions)
        diff_freq = self.ekin.compute_diffusion_rate(T_gas, Te_eV)
        dydt[0] -= c[0] * diff_freq
        for idx in self._positive_ion_indices:
            dydt[idx] -= c[idx] * diff_freq
        for idx in self._negative_ion_indices:
            dydt[idx] -= c[idx] * diff_freq

        # Electron energy: ne_eps tracks ne proportionally (eps_mean conserved)
        if c_e > self._ce_floor * 10:
            rel_rate = dydt[0] / c_e
            dydt[self.sm.idx_energy] = ne_eps * rel_rate
        else:
            dydt[self.sm.idx_energy] = 0.0

        # Floor guard
        y_sp = y[:n_sp]
        if y_sp[0] < self._ce_floor:
            dydt[0] = max(dydt[0], 0.0)
        for i in range(1, n_sp):
            if y_sp[i] < self._concentration_floor:
                dydt[i] = max(dydt[i], 0.0)
        if y[self.sm.idx_energy] < self._ne_eps_floor:
            dydt[self.sm.idx_energy] = max(dydt[self.sm.idx_energy], 0.0)

        # Gas temperature: chemical heating + cooling only (P_dep = 0)
        tau = self.flow.get_residence_time(T_gas)
        Q_rxn_Wm3 = self.rxn.compute_gas_heating(rates)
        dydt[self.sm.idx_Tgas] = self.gth.compute_Tgas_rhs(
            T_gas, 0.0, tau,
            Q_rxn_Wm3=Q_rxn_Wm3,
            Q_e_loss_Wm3=0.0
        )

        np.nan_to_num(dydt, copy=False, nan=0.0, posinf=1e30, neginf=-1e30)
        return dydt

    def build_initial_state(self, species_manager, inlet_composition: dict,
                             P_gas: float, T_gas_init: float,
                              ne_init: float = 1e12,
                             Te_init_eV: float = 1.0) -> np.ndarray:
        n_state = species_manager.n_state
        y0 = np.zeros(n_state)

        c_total = total_concentration(P_gas, T_gas_init)
        for name, x in inlet_composition.items():
            if species_manager.has(name):
                idx = species_manager.index(name)
                y0[idx] = x * c_total

        c_e_init = ne_init / NA
        y0[0] = c_e_init

        # ne_eps = n_e * eps_mean = n_e * (3/2)*Te
        eps_mean = 1.5 * Te_init_eV
        y0[species_manager.idx_energy] = ne_init * eps_mean

        y0[species_manager.idx_Tgas] = T_gas_init

        print(f"  Initial state: {n_state} variables")
        print(f"    c_total = {c_total:.2f} mol/m³")
        print(f"    c_e = {c_e_init:.4e} mol/m³ (n_e = {ne_init:.2e} m⁻³)")
        print(f"    T_gas = {T_gas_init:.0f} K, Te = {Te_init_eV:.1f} eV, "
              f"eps_mean = {eps_mean:.2f} eV")

        return y0

    def _clamp_state(self, y: np.ndarray) -> np.ndarray:
        """Clamp species concentrations and ne_eps to physical floors."""
        y_clamped = y.copy()
        n_sp = self.sm.n_species
        y_clamped[0] = max(y_clamped[0], self._ce_floor)
        y_clamped[1:n_sp] = np.maximum(y_clamped[1:n_sp], self._concentration_floor)
        y_clamped[self.sm.idx_energy] = max(y_clamped[self.sm.idx_energy],
                                             self._ne_eps_floor)
        return y_clamped

    def solve(self, y0: np.ndarray, t_span: tuple, n_points: int = 2000,
              method: str = 'BDF', rtol: float = 1e-6, atol: float = 1e-10,
              max_step: float = np.inf, seg_dt: float = 0.0,
              constrained: bool = False) -> SimulationResult:
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        self._rhs_count = 0
        self._jac_count = 0

        if method == 'CVODE':
            return self._solve_cvode(
                y0, t_span, t_eval, n_points, rtol, atol, max_step)
        elif method == 'CVODE_NATIVE':
            return self._solve_cvode_native(
                y0, t_span, t_eval, n_points, rtol, atol, max_step)
        elif method == 'CVODES':
            return self._solve_cvodes(
                y0, t_span, t_eval, n_points, rtol, atol, max_step)

        pulse_edges = self.power.get_pulse_edges(t_span[0], t_span[1])
        if constrained:
            if pulse_edges:
                print(f"  Pulsed mode: {len(pulse_edges)} edges detected "
                      f"(constrained BDF handles discontinuities)")
            return self._solve_constrained(
                y0, t_span, t_eval, n_points, method, rtol, atol, max_step)
        elif pulse_edges:
            print(f"  Pulse-edge segmentation: {len(pulse_edges)} edges detected")
            return self._solve_segmented(
                y0, t_span, t_eval, n_points, method, rtol, atol, max_step,
                seg_dt=0.0, seg_boundaries_explicit=pulse_edges)
        elif seg_dt > 0:
            return self._solve_segmented(
                y0, t_span, t_eval, n_points, method, rtol, atol, max_step, seg_dt)
        else:
            return self._solve_single(
                y0, t_span, t_eval, n_points, method, rtol, atol, max_step)

    def _solve_single(self, y0, t_span, t_eval, n_points, method, rtol, atol, max_step):
        """Single solve_ivp call — fast, with post-process clamping."""
        print(f"\n  Solving ODE system (single call)...")
        print(f"    Method: {method}, t=[{t_span[0]*1e6:.1f}, {t_span[1]*1e6:.1f}] µs")
        print(f"    rtol={rtol}, atol={atol}, max_step={max_step*1e6:.2f} µs")

        kwargs = dict(
            method=method, t_eval=t_eval, rtol=rtol, atol=atol, dense_output=False,
        )
        if max_step is not None and max_step < np.inf:
            kwargs['max_step'] = max_step

        t_start = time_module.time()
        sol = solve_ivp(self.rhs, t_span, y0, **kwargs)
        wall_time = time_module.time() - t_start

        if not sol.success:
            print(f"  WARNING: solver failed - {sol.message}")

        result = SimulationResult(
            t=sol.t, y=sol.y,
            species_names=self.sm.names, n_species=self.sm.n_species,
            wall_time=wall_time, n_rhs_evals=self._rhs_count,
            solver_message=sol.message,
        )
        self._postprocess(result)
        print(f"  Solver completed: {len(sol.t)} points, "
              f"{self._rhs_count} RHS evals, {wall_time:.1f}s wall time")
        return result

    def _solve_cvodes(self, y0, t_span, t_eval, n_points, rtol, atol, max_step):
        """SUNDIALS CVODES via CasADi — production-grade BDF with C performance."""
        import casadi as ca

        n_state = y0.shape[0]
        n_sp = self.sm.n_species
        rhs_func = self.rhs

        print(f"\n  Solving ODE system (CasADi CVODES)...")
        print(f"    t=[{t_span[0]*1e6:.1f}, {t_span[1]*1e6:.1f}] µs, "
              f"n_state={n_state}")
        print(f"    rtol={rtol}, atol={atol}, max_step={max_step*1e6:.2f} µs")

        class PlasmaRHS(ca.Callback):
            def __init__(cb_self, name, n):
                ca.Callback.__init__(cb_self)
                cb_self.n = n
                cb_self.construct(name, {'enable_fd': True})
            def get_n_in(cb_self): return 2
            def get_n_out(cb_self): return 1
            def get_sparsity_in(cb_self, i):
                if i == 0: return ca.Sparsity.dense(cb_self.n, 1)
                if i == 1: return ca.Sparsity.dense(1, 1)
            def get_sparsity_out(cb_self, i):
                return ca.Sparsity.dense(cb_self.n, 1)
            def eval(cb_self, arg):
                x = np.array(arg[0]).flatten()
                t_val = float(arg[1])
                dydt = rhs_func(t_val, x)
                return [ca.DM(dydt)]

        cb = PlasmaRHS('plasma_rhs', n_state)

        x_sym = ca.MX.sym('x', n_state)
        t_sym = ca.MX.sym('t')
        ode_expr = cb(x_sym, t_sym)
        dae = {'x': x_sym, 't': t_sym, 'ode': ode_expr}

        t_out = list(t_eval[1:])

        opts = {
            'abstol': atol,
            'reltol': rtol,
            'max_step_size': max_step,
            'max_num_steps': 500000,
            'linear_multistep_method': 'bdf',
            'disable_internal_warnings': True,
        }

        t_start = time_module.time()
        intg = ca.integrator('plasma', 'cvodes', dae, float(t_span[0]), t_out, opts)
        setup_time = time_module.time() - t_start
        print(f"    CVODES setup: {setup_time:.1f}s")

        result_ca = intg(x0=ca.DM(y0))
        wall_time = time_module.time() - t_start

        xf = np.array(result_ca['xf'])
        y_out = np.zeros((n_state, n_points))
        y_out[:, 0] = y0
        y_out[:, 1:] = xf

        y_out[0, :] = np.maximum(y_out[0, :], self._ce_floor)
        y_out[1:n_sp, :] = np.maximum(y_out[1:n_sp, :], self._concentration_floor)
        y_out[self.sm.idx_energy, :] = np.maximum(
            y_out[self.sm.idx_energy, :], self._ne_eps_floor)

        result = SimulationResult(
            t=t_eval, y=y_out,
            species_names=self.sm.names, n_species=self.sm.n_species,
            wall_time=wall_time, n_rhs_evals=self._rhs_count,
            solver_message="CasADi CVODES (SUNDIALS)",
        )
        self._postprocess(result)
        print(f"  Solver completed: {n_points} points, "
              f"{self._rhs_count} RHS evals, {wall_time:.1f}s wall time")
        return result

    def _solve_segmented(self, y0, t_span, t_eval_all, n_points,
                          method, rtol, atol, max_step, seg_dt,
                          seg_boundaries_explicit=None):
        """Segmented solve_ivp with inter-segment clamping.

        If seg_boundaries_explicit is provided, those are used directly
        (pulse-edge aligned).  Otherwise, uniform segments of width seg_dt.
        """
        t0, tf = t_span
        if seg_boundaries_explicit is not None:
            seg_boundaries = np.array(sorted(set([t0] + list(seg_boundaries_explicit) + [tf])))
        else:
            seg_boundaries = np.arange(t0, tf, seg_dt)
            if seg_boundaries[-1] < tf:
                seg_boundaries = np.append(seg_boundaries, tf)
        n_segs = len(seg_boundaries) - 1

        seg_label = "pulse-edge" if seg_boundaries_explicit is not None else f"dt_seg={seg_dt*1e6:.0f} µs"
        print(f"\n  Solving ODE system (segmented, {seg_label}, "
              f"{n_segs} segments)...")
        print(f"    Method: {method}, t=[{t0*1e6:.1f}, {tf*1e6:.1f}] µs")
        print(f"    rtol={rtol}, max_step={max_step*1e6:.2f} µs")

        t_start = time_module.time()
        collected_t = []
        collected_y = []
        y_current = y0.copy()
        total_message = "Segmented integration completed"
        n_clamp_events = 0

        for i_seg in range(n_segs):
            seg_t0 = seg_boundaries[i_seg]
            seg_tf = seg_boundaries[i_seg + 1]

            mask = (t_eval_all >= seg_t0 - 1e-15) & (t_eval_all <= seg_tf + 1e-15)
            seg_t_eval = t_eval_all[mask]
            if len(seg_t_eval) == 0:
                seg_t_eval = np.array([seg_tf])

            kwargs = dict(
                method=method, t_eval=seg_t_eval, rtol=rtol, atol=atol,
                dense_output=False,
            )
            if max_step is not None and max_step < np.inf:
                kwargs['max_step'] = max_step

            sol = solve_ivp(self.rhs, (seg_t0, seg_tf), y_current, **kwargs)

            if not sol.success:
                print(f"  WARNING: solver failed at segment {i_seg} "
                      f"(t={seg_t0*1e6:.1f}-{seg_tf*1e6:.1f} µs): {sol.message}")
                total_message = f"Failed at segment {i_seg}: {sol.message}"
                if len(sol.t) > 0:
                    collected_t.append(sol.t)
                    collected_y.append(sol.y)
                break

            if i_seg == 0:
                collected_t.append(sol.t)
                collected_y.append(sol.y)
            else:
                # Skip first point to avoid duplicates (close to previous segment end)
                if len(sol.t) > 1:
                    collected_t.append(sol.t[1:])
                    collected_y.append(sol.y[:, 1:])
                else:
                    collected_t.append(sol.t)
                    collected_y.append(sol.y)

            y_end = sol.y[:, -1]
            y_clamped = self._clamp_state(y_end)
            if not np.array_equal(y_end, y_clamped):
                n_clamp_events += 1
            y_current = y_clamped

            # Progress reporting
            if (i_seg + 1) % max(1, n_segs // 10) == 0 or i_seg == n_segs - 1:
                elapsed = time_module.time() - t_start
                print(f"    Segment {i_seg+1}/{n_segs} "
                      f"(t={seg_tf*1e6:.0f} µs), {elapsed:.1f}s, "
                      f"{n_clamp_events} clamps, {self._rhs_count} RHS")

        wall_time = time_module.time() - t_start

        if collected_t:
            all_t = np.concatenate(collected_t)
            all_y = np.concatenate(collected_y, axis=1)
        else:
            all_t = np.array([t_span[0]])
            all_y = y0.reshape(-1, 1)

        result = SimulationResult(
            t=all_t, y=all_y,
            species_names=self.sm.names, n_species=self.sm.n_species,
            wall_time=wall_time, n_rhs_evals=self._rhs_count,
            solver_message=total_message,
        )
        self._postprocess(result)
        print(f"  Solver completed: {len(result.t)} points, "
              f"{self._rhs_count} RHS evals, {wall_time:.1f}s wall time")
        print(f"    Clamp events: {n_clamp_events}/{n_segs} segments")
        return result

    def _solve_constrained(self, y0, t_span, t_eval, n_points,
                              method, rtol, atol, max_step):
        """Manual BDF stepping with variable scaling + clamping.

        Variable scaling: ȳ = y/w → κ(I-hJ̄) drops from ~10^40 to ~10^6
        This enables analytical Jacobian (67× faster than FD).
        """
        from scipy.integrate._ivp.bdf import BDF

        n_sp = self.sm.n_species
        n_state = y0.shape[0]
        idx_energy = self.sm.idx_energy
        idx_Tgas = self.sm.idx_Tgas
        ce_floor = self._ce_floor
        conc_floor = self._concentration_floor
        ne_eps_floor = self._ne_eps_floor
        t_start_wall = time_module.time()

        # Load or compute scaling weights
        w = getattr(self, '_scale_weights', None)
        if w is None:
            import os as _os
            scale_path = _os.path.join(_os.path.dirname(__file__), 'scale_weights.npy')
            if _os.path.exists(scale_path):
                w = np.load(scale_path)
            else:
                w = np.maximum(np.abs(y0), 1e-20)
            self._scale_weights = w
        w_inv = 1.0 / w

        _power_func = self.power.get_power_density
        _nb_fn = self._rhs_numba
        _nb_a = self._nb_args if self._nb_args is not None else None

        def rhs_scaled(t, y_bar):
            y = w * y_bar
            if _nb_fn is not None and _nb_a is not None:
                Pd = _power_func(t)
                return _nb_fn(t, y, *_nb_a, Pd) * w_inv
            return self.rhs(t, y) * w_inv

        def jac_scaled(t, y_bar):
            y = w * y_bar
            J = self.jacobian(t, y)
            return J * (w[np.newaxis, :] * w_inv[:, np.newaxis])

        y0_bar = y0 / w

        print(f"\n  Solving ODE system (scaled constrained BDF)...")
        print(f"    t=[{t_span[0]*1e6:.1f}, {t_span[1]*1e6:.1f}] µs")
        print(f"    rtol={rtol}, atol={atol}, max_step={max_step*1e6:.2f} µs")

        stepper = BDF(rhs_scaled, t_span[0], y0_bar.copy(), t_span[1],
                      max_step=max_step, rtol=rtol, atol=atol)

        # Pre-allocate output array; capture via nearest-step interpolation
        n_eval = len(t_eval)
        y_out = np.zeros((n_state, n_eval))
        y_out[:, 0] = y0
        eval_idx = 1
        prev_t = t_span[0]
        prev_y_bar = y0_bar.copy()

        # Floors in scaled coordinates
        ce_floor_bar = ce_floor / w[0]
        ne_eps_floor_bar = ne_eps_floor / w[idx_energy]

        n_clamp_events = 0
        n_steps = 0
        last_progress_time = t_span[0]
        progress_interval = (t_span[1] - t_span[0]) / 10.0

        while stepper.status == 'running':
            stepper.step()
            n_steps += 1
            cur_t = stepper.t
            y_bar = stepper.y

            clamped = False
            if y_bar[0] < ce_floor_bar:
                y_bar[0] = ce_floor_bar
                clamped = True
            if y_bar[idx_energy] < ne_eps_floor_bar:
                y_bar[idx_energy] = ne_eps_floor_bar
                clamped = True
            if clamped:
                n_clamp_events += 1

            while eval_idx < n_eval and t_eval[eval_idx] <= cur_t:
                te = t_eval[eval_idx]
                if cur_t > prev_t:
                    alpha = (te - prev_t) / (cur_t - prev_t)
                    y_bar_interp = prev_y_bar + alpha * (y_bar - prev_y_bar)
                else:
                    y_bar_interp = y_bar
                y_out[:, eval_idx] = w * y_bar_interp  # back to original
                eval_idx += 1

            prev_t = cur_t
            prev_y_bar = y_bar.copy()

            if cur_t - last_progress_time >= progress_interval:
                elapsed = time_module.time() - t_start_wall
                print(f"    t={cur_t*1e6:.0f} µs, steps={n_steps}, "
                      f"clamps={n_clamp_events}, {elapsed:.1f}s, "
                      f"RHS={self._rhs_count}, JAC={self._jac_count}")
                last_progress_time = cur_t

        wall_time = time_module.time() - t_start_wall

        if stepper.status == 'failed':
            print(f"  WARNING: constrained BDF stepper failed at t={stepper.t*1e6:.1f} µs")

        while eval_idx < n_eval:
            y_out[:, eval_idx] = w * prev_y_bar
            eval_idx += 1

        # Final clamp on interpolated output
        y_out[0, :] = np.maximum(y_out[0, :], ce_floor)
        y_out[1:n_sp, :] = np.maximum(y_out[1:n_sp, :], conc_floor)
        y_out[idx_energy, :] = np.maximum(y_out[idx_energy, :], ne_eps_floor)

        result = SimulationResult(
            t=t_eval, y=y_out,
            species_names=self.sm.names, n_species=self.sm.n_species,
            wall_time=wall_time, n_rhs_evals=self._rhs_count,
            solver_message=f"Constrained BDF: {n_steps} steps, {n_clamp_events} clamps",
        )
        self._postprocess(result)
        print(f"  Solver completed: {len(t_eval)} output points, "
              f"{n_steps} internal steps, {n_clamp_events} clamp events")
        print(f"    {self._rhs_count} RHS evals, {wall_time:.1f}s wall time")
        return result

    def _solve_cvode(self, y0, t_span, t_eval, n_points, rtol, atol, max_step):
        """SUNDIALS CVODE via ctypes wrapper with non-negative constraints.

        Uses CVodeSetConstraints for non-negative enforcement (no manual clamping).
        Trapezoidal pulsed mode: continuous integration (no ReInit).
        Rectangular pulsed mode: pulse-edge ReInit for hard discontinuities.

        Optimization: bypasses solver.rhs() and calls rhs_numba directly with
        Numba-computed pulsed power to eliminate Python dispatch overhead.
        """
        from .cvode_wrapper import CVODESolver

        n_sp = self.sm.n_species
        n_state = y0.shape[0]
        idx_energy = self.sm.idx_energy
        idx_Tgas = self.sm.idx_Tgas

        t0, tf = t_span
        duration = tf - t0

        # Non-negative constraints: species + ne_eps = 1.0 (>= 0), T_gas = 0.0 (unconstrained)
        constraints = np.ones(n_state, dtype=np.float64)
        constraints[idx_Tgas] = 0.0

        # Use scalar atol for consistency with scipy BDF baseline.
        cvode_atol = float(atol) if not isinstance(atol, np.ndarray) else 1e-10

        # Build segment boundaries and determine ON/OFF phase per segment
        _is_pulsed = self.power._mode == 'pulsed'

        if _is_pulsed:
            # ON/OFF operator splitting: each pulse → 2 segments (ON + OFF)
            T_pulse = self.power._pulse_period
            t_on = self.power._pulse_duty_cycle * T_pulse
            rise = self.power._pulse_rise_time
            t_on_eff = max(t_on - rise, rise)
            n_pulses = int(np.ceil((tf - t0) / T_pulse))
            seg_bounds = []
            seg_is_on = []
            for ip in range(n_pulses):
                p_start = t0 + ip * T_pulse
                p_on_end = min(p_start + t_on_eff, tf)
                p_end = min(p_start + T_pulse, tf)
                if p_start >= tf:
                    break
                seg_bounds.append(p_start)
                seg_is_on.append(True)
                if p_on_end < p_end:
                    seg_bounds.append(p_on_end)
                    seg_is_on.append(False)
            seg_bounds.append(tf)
            seg_bounds = np.array(seg_bounds)
        else:
            pulse_edges = self.power.get_pulse_edges(t0, tf)
            if pulse_edges:
                seg_bounds = np.array(sorted(set([t0] + pulse_edges + [tf])))
            else:
                seg_bounds = np.array([t0, tf])
            seg_is_on = [True] * (len(seg_bounds) - 1)
        n_segs = len(seg_bounds) - 1

        # For non-pulsed mode, cap max_step
        cvode_max_step = 0.0
        if max_step < np.inf and not _is_pulsed:
            cvode_max_step = max_step

        n_pulses_est = n_segs // 2 if _is_pulsed else n_segs
        print(f"\n  Solving ODE system (CVODE ctypes + constraints)...")
        print(f"    t=[{t0:.6f}, {tf:.6f}] s ({duration*1e3:.1f} ms)")
        if _is_pulsed:
            print(f"    {n_pulses_est} pulses, {n_segs} segments (ON/OFF splitting)")
        else:
            print(f"    {n_segs} segments")
        print(f"    rtol={rtol}, atol={cvode_atol:.1e}")
        if cvode_max_step > 0:
            print(f"    max_step={cvode_max_step*1e6:.1f} µs")
        else:
            print(f"    max_step=adaptive (CVODE auto)")

        # Build fast RHS functions
        _nb_fn = self._rhs_numba
        _nb_args = self._nb_args
        _power = self.power

        if _nb_fn is not None and _nb_args is not None and _is_pulsed:
            _P_on = float(_power._pulse_P_on_Wm3)

            def fast_rhs_on(t, y):
                return _nb_fn(t, y, *_nb_args, _P_on)

            def fast_rhs_off(t, y):
                return _nb_fn(t, y, *_nb_args, 0.0)

            print(f"    RHS: Numba direct (pulsed ON/OFF splitting)")
        elif _nb_fn is not None and _nb_args is not None and _power._mode == 'constant':
            _P_const = float(_power._P_constant_Wm3)

            def fast_rhs_on(t, y):
                return _nb_fn(t, y, *_nb_args, _P_const)

            fast_rhs_off = fast_rhs_on
            print(f"    RHS: Numba direct (constant power)")
        elif _nb_fn is not None and _nb_args is not None:
            _get_pdep = _power.get_power_density

            def fast_rhs_on(t, y):
                return _nb_fn(t, y, *_nb_args, _get_pdep(t))

            fast_rhs_off = fast_rhs_on
            print(f"    RHS: Numba direct (vi_envelope power)")
        else:
            fast_rhs_on = self.rhs
            fast_rhs_off = self.rhs_off if _is_pulsed else self.rhs
            print(f"    RHS: Python fallback")

        # Create CVODE solvers: ON phase + OFF phase (separate instances for pulsed)
        cvode_on = CVODESolver(n_state, fast_rhs_on)
        cvode_on.setup(y0, t0, rtol=rtol, atol=cvode_atol,
                       max_step=cvode_max_step, init_step=1e-12,
                       max_num_steps=500000, constraints=constraints)

        if _is_pulsed:
            cvode_off = CVODESolver(n_state, fast_rhs_off)
            cvode_off.setup(y0, t0, rtol=max(rtol, 1e-4), atol=max(cvode_atol, 1e-8),
                            max_step=0.0, init_step=1e-10,
                            max_num_steps=500000, constraints=constraints)
        else:
            cvode_off = None

        # ne re-seeding threshold
        ne_seed_conc = self._ne_seed / NA
        eps_thermal = 0.039  # thermal eps_mean at ~300K [eV]

        t_start = time_module.time()
        y_current = y0.copy()
        out_t = []
        out_y = []
        eval_ptr = 0
        n_clamp = 0
        n_on_fail = 0
        n_off_fail = 0
        total_rhs = 0
        progress_interval = max(1, n_pulses_est // 20) * (2 if _is_pulsed else 1)

        for i_seg in range(n_segs):
            seg_t0 = seg_bounds[i_seg]
            seg_tf = seg_bounds[i_seg + 1]
            is_on = seg_is_on[i_seg]

            # Select CVODE instance
            cvode = cvode_on if is_on else cvode_off

            # Phase transitions for pulsed mode
            if _is_pulsed and is_on and i_seg > 0:
                # OFF→ON: ne re-seeding
                if y_current[0] < ne_seed_conc:
                    y_current[0] = ne_seed_conc
                    y_current[idx_energy] = self._ne_seed * eps_thermal
            elif _is_pulsed and not is_on:
                # ON→OFF: ne_eps thermal reset
                ne_now = y_current[0] * NA
                y_current[idx_energy] = ne_now * eps_thermal

            # ReInit for this segment
            total_rhs += cvode._get_stats().get('n_rhs_evals', 0)
            cvode.reinit(seg_t0, y_current)

            # Collect t_eval points within this segment
            seg_eval = []
            while eval_ptr < len(t_eval) and t_eval[eval_ptr] <= seg_tf + 1e-15:
                if t_eval[eval_ptr] >= seg_t0 - 1e-15:
                    seg_eval.append(t_eval[eval_ptr])
                eval_ptr += 1

            seg_failed = False
            if not seg_eval:
                t_reached, y_end, ret = cvode.step_to(seg_tf)
                if ret < 0:
                    if is_on:
                        n_on_fail += 1
                    else:
                        n_off_fail += 1
                    seg_failed = True
                y_current = y_end
            else:
                for t_out in seg_eval:
                    if seg_failed:
                        out_t.append(t_out)
                        out_y.append(y_current.copy())
                        continue
                    t_reached, y_out_pt, ret = cvode.step_to(t_out)
                    if ret < 0:
                        y_out_pt = y_current.copy()
                        if is_on:
                            n_on_fail += 1
                        else:
                            n_off_fail += 1
                        seg_failed = True
                    else:
                        y_current = y_out_pt.copy()
                    out_t.append(t_reached if not seg_failed else t_out)
                    out_y.append(y_out_pt.copy())
                if not seg_failed and seg_eval[-1] < seg_tf - 1e-15:
                    t_reached, y_end, ret = cvode.step_to(seg_tf)
                    if ret >= 0:
                        y_current = y_end

            # Soft clamp for segment hand-off
            clamped = False
            if y_current[0] < self._ce_floor:
                y_current[0] = self._ce_floor
                clamped = True
            if y_current[idx_energy] < self._ne_eps_floor:
                y_current[idx_energy] = self._ne_eps_floor
                clamped = True
            for j in range(1, n_sp):
                if y_current[j] < self._concentration_floor:
                    y_current[j] = self._concentration_floor
                    clamped = True
            if clamped:
                n_clamp += 1

            # Progress report (per pulse = every 2 segments)
            if _is_pulsed:
                i_pulse = (i_seg + 1) // 2
                if i_pulse % max(1, n_pulses_est // 20) == 0 or i_seg == n_segs - 1:
                    elapsed = time_module.time() - t_start
                    pct = (seg_tf - t0) / (tf - t0) * 100
                    seg_rhs = cvode._get_stats().get('n_rhs_evals', 0)
                    rate = max(i_pulse, 1) / elapsed if elapsed > 0 else 1
                    eta = (n_pulses_est - i_pulse) / rate if rate > 0 else 0
                    ne = y_current[0] * NA
                    print(f"    [{pct:5.1f}%] pulse {i_pulse}/{n_pulses_est}, "
                          f"ne={ne:.2e}, ON_fail={n_on_fail}, OFF_fail={n_off_fail}, "
                          f"RHS={total_rhs + seg_rhs}, "
                          f"{elapsed:.0f}s, ETA={eta:.0f}s")
            elif (i_seg + 1) % max(1, n_segs // 20) == 0 or i_seg == n_segs - 1:
                elapsed = time_module.time() - t_start
                pct = (seg_tf - t0) / (tf - t0) * 100
                seg_rhs = cvode._get_stats().get('n_rhs_evals', 0)
                rate = (i_seg + 1) / elapsed if elapsed > 0 else 0
                eta = (n_segs - i_seg - 1) / rate if rate > 0 else 0
                print(f"    [{pct:5.1f}%] t={seg_tf:.6f}s, "
                      f"seg {i_seg+1}/{n_segs}, "
                      f"RHS={total_rhs + seg_rhs}, clamp={n_clamp}, "
                      f"{elapsed:.0f}s elapsed, ETA={eta:.0f}s")

        # Accumulate final segment RHS
        total_rhs += cvode_on._get_stats().get('n_rhs_evals', 0)
        if cvode_off is not None:
            total_rhs += cvode_off._get_stats().get('n_rhs_evals', 0)
        wall_time = time_module.time() - t_start
        cvode_on.free()
        if cvode_off is not None:
            cvode_off.free()

        # Assemble output
        all_t = np.array(out_t)
        all_y = np.array(out_y).T  # shape (n_state, n_points)

        # Final clamp on output
        all_y[0, :] = np.maximum(all_y[0, :], self._ce_floor)
        all_y[1:n_sp, :] = np.maximum(all_y[1:n_sp, :], self._concentration_floor)
        all_y[idx_energy, :] = np.maximum(all_y[idx_energy, :], self._ne_eps_floor)

        msg = (f"CVODE ctypes: {n_segs} segs, {n_clamp} clamps, "
               f"{total_rhs} RHS")
        if _is_pulsed:
            msg += f", ON_fail={n_on_fail}, OFF_fail={n_off_fail}"

        result = SimulationResult(
            t=all_t, y=all_y,
            species_names=self.sm.names, n_species=self.sm.n_species,
            wall_time=wall_time, n_rhs_evals=total_rhs,
            solver_message=msg,
        )
        self._postprocess(result)
        print(f"  Solver completed: {len(all_t)} points, "
              f"{n_segs} segments, {n_clamp} clamps")
        if _is_pulsed:
            print(f"    {n_pulses_est} pulses, ON_fail={n_on_fail}, OFF_fail={n_off_fail}")
        print(f"    {total_rhs} RHS evals, {wall_time:.1f}s wall time")
        return result

    def _solve_cvode_native(self, y0, t_span, t_eval, n_points, rtol, atol, max_step):
        """SUNDIALS CVODE with pulse-edge ReInit.

        Single CVODE instance, re-initialized at each pulse boundary via
        init_step().  Between boundaries CVODE freely adapts step size —
        crucial for OFF phases where it can jump from µs to ms steps.
        """
        from sksundae.cvode import CVODE

        n_sp = self.sm.n_species
        n_state = y0.shape[0]
        idx_energy = self.sm.idx_energy
        ce_floor = self._ce_floor
        ne_eps_floor = self._ne_eps_floor
        rhs_func = self.rhs

        t0, tf = t_span
        duration = tf - t0

        def rhs_inplace(t, y, yp):
            yp[:] = rhs_func(t, y)

        constraints_idx = [0, idx_energy]
        constraints_type = [1, 1]

        # Build pulse-edge segment boundaries
        pulse_edges = self.power.get_pulse_edges(t0, tf)
        if pulse_edges:
            seg_bounds = np.array(sorted(set([t0] + pulse_edges + [tf])))
        else:
            # No pulse edges (continuous/trapezoidal) — use period-based segments
            period = self.power.period
            if period > 0:
                seg_bounds = np.arange(t0, tf, period)
                if seg_bounds[-1] < tf:
                    seg_bounds = np.append(seg_bounds, tf)
            else:
                seg_bounds = np.array([t0, tf])
        n_segs = len(seg_bounds) - 1

        # For pulsed mode, don't cap max_step — let CVODE adapt in OFF phases
        cvode_max_step = 0.0  # 0 = no limit (CVODE default)
        if max_step < np.inf and self.power.mode not in ('pulsed',):
            cvode_max_step = max_step

        n_pulses_est = n_segs // 2 if pulse_edges else n_segs
        print(f"\n  Solving ODE system (CVODE + pulse-edge ReInit)...")
        print(f"    t=[{t0:.6f}, {tf:.6f}] s ({duration*1e3:.1f} ms)")
        print(f"    {n_segs} segments ({n_pulses_est} pulses)")
        print(f"    rtol={rtol}, atol={atol}")
        if cvode_max_step > 0:
            print(f"    max_step={cvode_max_step*1e6:.1f} µs")
        else:
            print(f"    max_step=adaptive (CVODE auto)")

        jac_func = self.jacobian
        def jac_inplace(t, y, yp, JJ):
            JJ[:] = jac_func(t, y)

        cvode = CVODE(
            rhs_inplace,
            method='BDF',
            rtol=rtol,
            atol=float(atol),
            max_step=cvode_max_step,
            first_step=1e-12,
            max_num_steps=500000,
            constraints_idx=constraints_idx,
            constraints_type=constraints_type,
            linsolver='dense',
            jacfn=jac_inplace,
        )

        t_start = time_module.time()
        y_current = y0.copy()
        out_t = [t0]
        out_y = [y0.copy()]
        total_nfev = 0
        n_clamp = 0

        # Pre-compute output time indices per segment for fast lookup
        eval_ptr = 0

        progress_interval = max(1, n_segs // 20)

        for i_seg in range(n_segs):
            seg_t0 = seg_bounds[i_seg]
            seg_tf = seg_bounds[i_seg + 1]

            # ReInit: discard BDF history, restart at order 1
            cvode.init_step(seg_t0, y_current)

            # Step to segment end
            try:
                soln = cvode.step(seg_tf, method='normal')
            except Exception as e:
                if i_seg < 5 or (i_seg + 1) % progress_interval == 0:
                    print(f"  WARNING seg {i_seg}: {e}")
                continue

            total_nfev += soln.nfev
            y_end = soln.y.flatten().copy()

            # Collect output points in this segment
            while eval_ptr < len(t_eval) and t_eval[eval_ptr] <= seg_tf + 1e-15:
                out_t.append(t_eval[eval_ptr])
                out_y.append(y_end.copy())
                eval_ptr += 1

            # Clamp
            if y_end[0] < ce_floor:
                y_end[0] = ce_floor
                n_clamp += 1
            if y_end[idx_energy] < ne_eps_floor:
                y_end[idx_energy] = ne_eps_floor
            y_current = y_end

            if (i_seg + 1) % progress_interval == 0 or i_seg == n_segs - 1:
                elapsed = time_module.time() - t_start
                pct = (seg_tf - t0) / (tf - t0) * 100
                rate = (i_seg + 1) / elapsed if elapsed > 0 else 0
                eta = (n_segs - i_seg - 1) / rate if rate > 0 else 0
                print(f"    [{pct:5.1f}%] t={seg_tf:.6f}s, "
                      f"seg {i_seg+1}/{n_segs}, "
                      f"RHS={total_nfev}, clamp={n_clamp}, "
                      f"{elapsed:.0f}s elapsed, ETA={eta:.0f}s")

        wall_time = time_module.time() - t_start

        # Assemble output
        all_t = np.array(out_t)
        all_y = np.array(out_y).T

        # Final clamp
        all_y[0, :] = np.maximum(all_y[0, :], ce_floor)
        all_y[1:n_sp, :] = np.maximum(all_y[1:n_sp, :], self._concentration_floor)
        all_y[idx_energy, :] = np.maximum(all_y[idx_energy, :], ne_eps_floor)

        result = SimulationResult(
            t=all_t, y=all_y,
            species_names=self.sm.names, n_species=self.sm.n_species,
            wall_time=wall_time, n_rhs_evals=total_nfev,
            solver_message=f"CVODE pulse-ReInit: {n_segs} segs, {n_clamp} clamps",
        )
        self._postprocess(result)
        print(f"  Solver completed: {len(all_t)} points, "
              f"{n_segs} segments, {n_clamp} clamps")
        print(f"    {total_nfev} RHS evals, {wall_time:.1f}s wall time")
        return result

    def solve_pulsed(self, y0, t_end, n_output=1000,
                     rtol=1e-4, atol=1e-10,
                     max_step=2e-6,
                     n_detail=5, n_skip=1000,
                     progress_every=50):
        """Two-timescale pulsed solver with fixed macro-timestep.

        Per macro step:
        1. Solve n_detail full pulse cycles (constrained BDF, unified RHS)
        2. Extract cycle-averaged source terms for NEUTRALS
        3. Advance neutral species by n_skip pulses using forward Euler

        Charged species (e, ions, ne_eps) are set by the detail cycles.
        """
        T_pulse = self.power.period
        n_total = int(t_end / T_pulse)
        n_sp = self.sm.n_species
        n_macro = n_total // (n_detail + n_skip) + 1

        charged_idx = set([0] + list(self._positive_ion_indices)
                          + list(self._negative_ion_indices))
        neutral_mask = np.ones(n_sp, dtype=bool)
        for ci in charged_idx:
            neutral_mask[ci] = False

        print(f"\n  Pulsed macro-timestep solver (fixed skip)")
        print(f"    {n_total} total pulses, ~{n_macro} macro steps")
        print(f"    n_detail={n_detail}, n_skip={n_skip} "
              f"(ratio {n_skip/(n_detail+n_skip)*100:.0f}% skipped)")
        print(f"    t_end = {t_end:.1f}s")

        t_start_wall = time_module.time()
        y = y0.copy()
        t_current = 0.0
        i_pulse = 0
        total_rhs = 0
        macro_steps = 0

        out_t = [0.0]
        out_y = [y0.copy()]
        out_dt = t_end / n_output
        next_out = out_dt

        while i_pulse < n_total:
            # --- Detail cycles: operator-split (ON: full, OFF: frozen e) ---
            from scipy.integrate import solve_ivp as _sivp
            t_on = self.power._pulse_duty_cycle * T_pulse
            rise = self.power._pulse_rise_time
            t_on_eff = max(t_on - rise, rise)
            t_off_eff = T_pulse - t_on_eff

            S_accum = np.zeros(len(y))
            n_resolved = 0
            for k in range(n_detail):
                if i_pulse + k >= n_total:
                    break
                t0_c = t_current + k * T_pulse
                y_start = y.copy()

                # ON phase (DQ Jacobian — analytic Jacobian is 76x slower)
                sol_on = _sivp(self.rhs, [t0_c, t0_c + t_on_eff], y,
                               method='BDF', rtol=rtol, atol=atol)
                y = sol_on.y[:, -1]
                total_rhs += sol_on.nfev

                # OFF phase (frozen electrons)
                sol_off = _sivp(self.rhs_off, [t0_c + t_on_eff, t0_c + T_pulse], y,
                                method='BDF', rtol=1e-3, atol=1e-8)
                y = sol_off.y[:, -1]
                total_rhs += sol_off.nfev

                y[0] = max(y[0], self._ce_floor)
                y[self.sm.idx_energy] = max(y[self.sm.idx_energy], self._ne_eps_floor)

                S_accum += (y - y_start) / T_pulse
                n_resolved += 1

            S_avg = S_accum / max(n_resolved, 1)
            t_current += n_resolved * T_pulse
            i_pulse += n_resolved

            # --- Macro-step: advance neutrals + T_gas ---
            # Analytical CSTR solution for each neutral species:
            #   dc/dt = S_chem + (c_in - c)/τ = S_chem + c_in/τ - c/τ
            # Let R = S_chem + c_in/τ (production), D = 1/τ (destruction freq)
            # Then dc/dt = R - D*c → c(t) = R/D + (c0 - R/D)*exp(-D*t)
            # This is exact for constant R, D and avoids overshoot.
            actual_skip = min(n_skip, n_total - i_pulse)
            if actual_skip > 0:
                dt = actual_skip * T_pulse
                T_gas_now = y[self.sm.idx_Tgas]
                tau_now = self.flow.get_residence_time(T_gas_now)
                inv_tau = 1.0 / tau_now if tau_now > 0 else 0.0
                c_total_now = self.power.P_gas / (8.314 * max(T_gas_now, 200.0))

                for i in range(n_sp):
                    if i not in charged_idx:
                        c_in_i = self.flow.x_inlet[i] * c_total_now
                        R_i = S_avg[i] + c_in_i * inv_tau  # total production
                        D_i = inv_tau  # loss frequency from flow
                        # Add chemical loss if S_avg < 0:
                        if y[i] > self._concentration_floor and S_avg[i] < 0:
                            D_i += (-S_avg[i]) / y[i]
                            R_i = c_in_i * inv_tau  # only flow production

                        if D_i > 1e-30:
                            c_ss = R_i / D_i
                            y[i] = c_ss + (y[i] - c_ss) * np.exp(-D_i * dt)
                        else:
                            y[i] += dt * S_avg[i]

                y[self.sm.idx_Tgas] += dt * S_avg[self.sm.idx_Tgas]
                y[1:n_sp] = np.maximum(y[1:n_sp], self._concentration_floor)
                t_current += dt
                i_pulse += actual_skip

            # Output
            while next_out <= t_current and next_out <= t_end:
                out_t.append(next_out)
                out_y.append(y.copy())
                next_out += out_dt

            macro_steps += 1
            if macro_steps % progress_every == 0:
                elapsed = time_module.time() - t_start_wall
                pct = i_pulse / n_total * 100
                ne = y[0] * NA
                ch4_i = self.sm.index('CH4')
                conv = (y0[ch4_i] - y[ch4_i]) / y0[ch4_i] * 100
                eta = elapsed / max(pct, 0.01) * (100 - pct) / 3600
                print(f"    [{pct:5.1f}%] step {macro_steps}, "
                      f"ne={ne:.2e}, conv={conv:.3f}%, "
                      f"{elapsed:.0f}s, ETA {eta:.1f}h")

        wall_time = time_module.time() - t_start_wall

        all_t = np.array(out_t)
        all_y = np.array(out_y).T

        all_y[0, :] = np.maximum(all_y[0, :], self._ce_floor)
        all_y[1:n_sp, :] = np.maximum(all_y[1:n_sp, :],
                                       self._concentration_floor)
        all_y[self.sm.idx_energy, :] = np.maximum(
            all_y[self.sm.idx_energy, :], self._ne_eps_floor)

        result = SimulationResult(
            t=all_t, y=all_y,
            species_names=self.sm.names, n_species=self.sm.n_species,
            wall_time=wall_time, n_rhs_evals=total_rhs,
            solver_message=(f"Pulsed macro-timestep: {macro_steps} macro steps, "
                           f"{total_rhs} RHS"),
        )
        self._postprocess(result)
        print(f"  Completed: {macro_steps} macro steps, {total_rhs} RHS, "
              f"{wall_time:.0f}s ({wall_time/3600:.1f}h)")
        return result

        wall_time = time_module.time() - t_start

        all_t = np.array(out_t)
        all_y = np.array(out_y).T

        n_sp = self.sm.n_species
        all_y[0, :] = np.maximum(all_y[0, :], self._ce_floor)
        all_y[1:n_sp, :] = np.maximum(all_y[1:n_sp, :],
                                       self._concentration_floor)
        all_y[self.sm.idx_energy, :] = np.maximum(
            all_y[self.sm.idx_energy, :], self._ne_eps_floor)

        result = SimulationResult(
            t=all_t, y=all_y,
            species_names=self.sm.names, n_species=self.sm.n_species,
            wall_time=wall_time, n_rhs_evals=total_rhs,
            solver_message=f"Pulsed operator-split: {n_pulses} pulses",
        )
        self._postprocess(result)
        print(f"  Completed: {n_pulses} pulses, {total_rhs} RHS, "
              f"{wall_time:.0f}s ({wall_time/3600:.1f}h)")
        return result

    def _postprocess(self, result: SimulationResult):
        n_sp = result.n_species
        n_t = len(result.t)

        y_clamped = result.y.copy()
        y_clamped[0, :] = np.maximum(y_clamped[0, :], self._ce_floor)
        y_clamped[1:n_sp, :] = np.maximum(y_clamped[1:n_sp, :], self._concentration_floor)
        y_clamped[self.sm.idx_energy, :] = np.maximum(
            y_clamped[self.sm.idx_energy, :], self._ne_eps_floor)

        result.concentrations = y_clamped[:n_sp, :]
        result.T_gas = result.y[self.sm.idx_Tgas, :]

        ne_eps = y_clamped[self.sm.idx_energy, :]
        c_e = y_clamped[0, :]
        n_e = c_e * NA
        result.ne_m3 = n_e

        result.Te_eV = np.zeros(n_t)
        result.eps_mean_eV = np.zeros(n_t)
        result.EN_Td = np.zeros(n_t)
        result.power_Wm3 = np.zeros(n_t)

        for i in range(n_t):
            T_g = result.T_gas[i] if result.T_gas[i] > 200.0 else 300.0
            eps_th = 1.5 * KB * T_g / QE
            if n_e[i] > 1.0:
                eps = max(ne_eps[i] / n_e[i], eps_th)
            else:
                eps = max(1.0, eps_th)
            result.eps_mean_eV[i] = eps
            result.Te_eV[i] = (2.0 / 3.0) * eps

            if self.lut is not None:
                # E/N only meaningful above LUT range
                if eps >= self._eps_min_lut:
                    result.EN_Td[i] = self.lut.eps_to_EN(eps)
                else:
                    result.EN_Td[i] = 0.0  # afterglow: no external field

            result.power_Wm3[i] = self.power.get_power_density(result.t[i])
