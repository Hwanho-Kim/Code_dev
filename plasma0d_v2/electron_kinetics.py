"""Electron energy balance equation.

State variable: n_e * eps_mean [eV/m³]

d(n_e * eps_mean)/dt = P_dep - P_elastic - P_inelastic - P_diff - P_flow

P_elastic is computed from BOLSIG+ A21 (passed in by solver).
Electron loss uses ambipolar diffusion: D_a / Lambda².
"""

import numpy as np
from .constants import QE, ME, KB, NA, R_GAS


class ElectronKinetics:

    def __init__(self, species_manager, Lambda: float):
        """
        Args:
            species_manager: SpeciesManager instance
            Lambda: diffusion length [m] (sDBD fitting parameter)
        """
        self.sm = species_manager
        self.Lambda = Lambda
        self.Lambda_sq = Lambda * Lambda

    def compute_energy_rhs(self, ne_eps: float, c_e: float, T_gas: float,
                            P_dep_Wm3: float, P_el_eVm3s: float,
                            P_inelastic_Wm3: float, tau: float,
                            P_e_loss_eVm3s: float = 0.0) -> float:
        """Compute d(n_e * eps_mean)/dt [eV/(m³·s)].

        Args:
            ne_eps: n_e * eps_mean [eV/m³]
            c_e: electron concentration [mol/m³]
            T_gas: gas temperature [K]
            P_dep_Wm3: deposited power density [W/m³]
            P_el_eVm3s: elastic loss = n_e * N_gas * A21(ε̄) [eV/(m³·s)]
            P_inelastic_Wm3: inelastic loss power [W/m³]
            tau: residence time [s]
            P_e_loss_eVm3s: electron destruction loss [eV/(m³·s)]
                = ε̄ × Σ(R_loss × NA), from DR/AT reactions
        """
        n_e = min(c_e * NA, 1e26)

        # P_dep [W/m³] -> [eV/(m³·s)]
        P_dep = P_dep_Wm3 / QE

        # Elastic loss (already in eV/(m³·s) from solver)
        P_elastic = min(P_el_eVm3s, 1e30)

        # Inelastic loss [W/m³] -> [eV/(m³·s)]
        P_inel = min(P_inelastic_Wm3 / QE, 1e30)

        # Ambipolar diffusion loss: P_diff = ne_eps * D_a / Lambda²
        # D_a ≈ kB*Te/e * mu_i ≈ kB*Te/(M_ion * nu_in)
        # Simplified: D_a ~ (kB * T_e) / (M_i * nu_in)
        # For now, use a simplified D_a estimate based on ion mobility in N2
        # mu_i ~ 2.2e-4 m²/(V·s) at atmospheric pressure -> D_a ~ mu_i * kB*Te/e
        T_gas_K = max(T_gas, 200.0)
        eps_thermal = 1.5 * KB * T_gas_K / QE
        if n_e > 1.0:
            eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0)
        else:
            eps_mean = max(1.0, eps_thermal)
        Te_eV = (2.0 / 3.0) * eps_mean

        # D_a ≈ mu_i * (Te + Ti) ≈ mu_i * Te  (Te >> Ti)
        # mu_i * N ≈ 2.8e22 [1/(m·V·s)] for N2+ at 300K (Viehland & Mason)
        # mu_i = mu_i_N / N_gas
        N_gas = 101325.0 / (KB * T_gas_K)
        mu_i_N = 2.8e22  # mu_i * N [1/(m·V·s)]
        mu_i = mu_i_N / N_gas
        D_a = mu_i * Te_eV  # [m²/s] (since kB*Te/e = Te_eV [V])

        P_diff = ne_eps * D_a / self.Lambda_sq

        # Flow loss
        P_flow = ne_eps / tau if tau > 0 else 0.0

        # Electron destruction loss (DR/AT): electrons destroyed carry ε̄ energy
        P_eloss = min(P_e_loss_eVm3s, 1e30)

        rhs = P_dep - P_elastic - P_inel - P_diff - P_flow - P_eloss
        return np.clip(rhs, -1e30, 1e30)

    def compute_diffusion_rate(self, T_gas: float, Te_eV: float) -> float:
        """Compute ambipolar diffusion frequency D_a / Λ² [1/s].

        Used for charged species density loss:
            dc_charged/dt += -c_charged * D_a / Λ²

        Same D_a model as in energy equation:
            D_a = mu_i * Te_eV  where mu_i = mu_i_N / N_gas

        Args:
            T_gas: gas temperature [K]
            Te_eV: electron temperature [eV]

        Returns:
            D_a / Λ² [1/s]
        """
        T_gas_safe = max(T_gas, 200.0)
        N_gas = 101325.0 / (KB * T_gas_safe)
        mu_i_N = 2.8e22  # mu_i * N [1/(m·V·s)] for N2+ at 300K
        mu_i = mu_i_N / N_gas
        D_a = mu_i * Te_eV  # [m²/s]
        return D_a / self.Lambda_sq

    def compute_energy_jacobian_row(self, ne_eps: float, c_e: float,
                                    T_gas: float, P_dep_Wm3: float,
                                    P_el_eVm3s: float,
                                    P_inelastic_Wm3: float, tau: float,
                                    P_e_loss_eVm3s: float,
                                    n_e: float, eps_mean: float,
                                    N_gas: float,
                                    A21: float, dA21_deps: float,
                                    dR_dc_inel: np.ndarray,
                                    dR_dne_eps_inel: np.ndarray,
                                    dR_dc_eloss: np.ndarray,
                                    dR_dne_eps_eloss: np.ndarray,
                                    ei_reactions: list,
                                    electron_loss_indices: list,
                                    rates: np.ndarray,
                                    stoich_matrix: np.ndarray,
                                    n_state: int,
                                    A20: float = 0.0,
                                    dA20_deps: float = 0.0,
                                    energy_source: str = 'constant') -> np.ndarray:
        """Compute the Jacobian row for d(ne_eps)/dt.

        Returns
        -------
        jac_row : ndarray, shape (n_state,)
            ∂(d(ne_eps)/dt)/∂y[k] for k = 0..n_state-1.
        """
        jac_row = np.zeros(n_state)
        n_sp = n_state - 2
        idx_energy = n_sp      # y[63] = ne_eps
        idx_Tgas = n_sp + 1    # y[64] = T_gas

        eps_mean = max(eps_mean, 1e-6)
        c_e = max(c_e, 1e-50)
        n_e = max(n_e, 1.0)
        T_gas_safe = max(T_gas, 200.0)

        # Chain rule: ε̄ = ne_eps / n_e
        deps_dc_e = -eps_mean / c_e
        deps_dne_eps = 1.0 / n_e

        # -------------------------------------------------------
        # 1) P_dep: depends on energy_source mode
        #    'constant': no dependence on state (external power)
        #    'A20': P_dep = n_e * N_gas * A20(ε̄) * QE  [W/m³]
        #      P_dep_eV = n_e * N_gas * A20(ε̄)  [eV/(m³·s)]
        #      ∂P_dep_eV/∂c_e = NA * N_gas * A20 + n_e * N_gas * dA20/dε̄ * deps_dc_e
        #      ∂P_dep_eV/∂ne_eps = n_e * N_gas * dA20/dε̄ * deps_dne_eps
        #      ∂P_dep_eV/∂T_gas = n_e * (-N_gas/T_gas) * A20
        # -------------------------------------------------------
        dPdep_dc_e = 0.0
        dPdep_dne_eps = 0.0
        dPdep_dTgas = 0.0
        if energy_source == 'A20' and A20 > 0:
            # Derivatives of P_dep_eV = n_e * N_gas * A20(ε̄)  [eV/(m³·s)]
            dPdep_dc_e = NA * N_gas * A20 + n_e * N_gas * dA20_deps * deps_dc_e
            dPdep_dne_eps = n_e * N_gas * dA20_deps * deps_dne_eps
            dPdep_dTgas = n_e * (-N_gas / T_gas_safe) * A20

        # -------------------------------------------------------
        # 2) P_elastic = n_e * N_gas * A21(ε̄)  [eV/(m³·s)]
        #    ∂P_el/∂c_e = NA * N_gas * A21 + n_e * N_gas * dA21/dε̄ * deps_dc_e
        #    ∂P_el/∂ne_eps = n_e * N_gas * dA21/dε̄ * deps_dne_eps
        #    ∂P_el/∂T_gas = n_e * (-N_gas/T_gas) * A21  (via N_gas = P/(kB*T))
        # -------------------------------------------------------
        dPel_dc_e = NA * N_gas * A21 + n_e * N_gas * dA21_deps * deps_dc_e
        dPel_dne_eps = n_e * N_gas * dA21_deps * deps_dne_eps
        dPel_dTgas = n_e * (-N_gas / T_gas_safe) * A21

        # -------------------------------------------------------
        # 3) P_inel = Σ(ΔE_j * R_j) * NA / QE  where R_j are EI rates
        #    ∂P_inel/∂c_k = Σ(ΔE_j * ∂R_j/∂c_k) * NA / QE
        #    ∂P_inel/∂ne_eps = Σ(ΔE_j * ∂R_j/∂ne_eps) * NA / QE
        # -------------------------------------------------------
        # dR_dc_inel[j, k] and dR_dne_eps_inel[j] are already computed per EI rxn
        dPinel_dc = np.zeros(n_sp)
        dPinel_dne_eps = 0.0
        for i_ei, rxn in enumerate(ei_reactions):
            dE_eV = rxn.energy_loss_eV
            coeff = dE_eV * NA  # [eV * (1/mol)] → after mult by R [mol/(m³·s)] → [eV/(m³·s)]
            # Wait: P_inel [W/m³] = dE_J * R * NA where dE_J = dE_eV * QE
            # Then P_inel [eV/(m³·s)] = P_inel_W / QE = dE_eV * R * NA
            dPinel_dc += coeff * dR_dc_inel[i_ei, :]
            dPinel_dne_eps += coeff * dR_dne_eps_inel[i_ei]

        # -------------------------------------------------------
        # 4) P_diff = ne_eps * D_a / Λ²
        #    D_a = mu_i * Te_eV = (mu_i_N/N_gas) * (2/3)*eps_mean
        #    ∂P_diff/∂ne_eps = D_a/Λ² + ne_eps/Λ² * ∂D_a/∂ne_eps
        #    ∂D_a/∂eps = mu_i * (2/3) = mu_i_N/N_gas * (2/3)
        #    ∂D_a/∂ne_eps = ∂D_a/∂eps * deps_dne_eps
        #    ∂D_a/∂c_e = ∂D_a/∂eps * deps_dc_e
        # -------------------------------------------------------
        mu_i_N = 2.8e22
        N_gas_val = 101325.0 / (KB * T_gas_safe)
        mu_i = mu_i_N / N_gas_val
        Te_eV = (2.0 / 3.0) * eps_mean
        D_a = mu_i * Te_eV

        dDa_deps = mu_i * (2.0 / 3.0)
        inv_L2 = 1.0 / self.Lambda_sq

        dPdiff_dne_eps = D_a * inv_L2 + ne_eps * inv_L2 * dDa_deps * deps_dne_eps
        dPdiff_dc_e = ne_eps * inv_L2 * dDa_deps * deps_dc_e
        # ∂P_diff/∂T_gas via N_gas in mu_i: mu_i = mu_i_N * kB*T/(P) → dmu_i/dT = mu_i/T
        # D_a = mu_i * Te_eV → ∂D_a/∂T = D_a / T  (only T-dependence is via mu_i)
        dPdiff_dTgas = ne_eps * inv_L2 * D_a / T_gas_safe

        # -------------------------------------------------------
        # 5) P_flow = ne_eps / tau
        #    tau = V / Q_actual, Q_actual ∝ T → tau ∝ 1/T → ∂(1/tau)/∂T = 1/(tau*T)
        #    ∂P_flow/∂ne_eps = 1/tau
        #    ∂P_flow/∂T_gas = -ne_eps / (tau * T_gas)  [since tau ∝ 1/T]
        # -------------------------------------------------------
        inv_tau = 1.0 / tau if tau > 0 else 0.0
        dPflow_dne_eps = inv_tau
        # tau = V / Q_actual, Q_actual = Q_slm * T/T_STP * P_STP/P / 60000
        # dtau/dT = -tau/T
        dPflow_dTgas = ne_eps * inv_tau / T_gas_safe  # = ne_eps/(tau*T) with sign: ∂(ne/tau)/∂T
        # Actually: P_flow = ne_eps/tau, tau∝1/T so 1/tau∝T, ∂(1/tau)/∂T = 1/(tau*T)
        # ∂P_flow/∂T = ne_eps * 1/(tau*T) = ne_eps/(tau*T)
        # But this INCREASES P_flow (more outflow at higher T), which is a loss term.
        # Sign in rhs: rhs = P_dep - ... - P_flow
        # So ∂rhs/∂T includes -∂P_flow/∂T = -ne_eps/(tau*T)

        # -------------------------------------------------------
        # 6) P_e_loss = ε̄ * S_e_loss * NA  [eV/(m³·s)]
        #    S_e_loss = Σ(-ν_e,j * R_j) for electron-loss reactions
        #    ∂P_eloss/∂c_k = eps_mean * NA * Σ(-ν_e,j * ∂R_j/∂c_k)
        #                    + ∂eps/∂c_k * S_e_loss * NA  (chain rule on ε̄)
        #    ∂P_eloss/∂ne_eps = eps_mean * NA * Σ(-ν_e,j * ∂R_j/∂ne_eps)
        #                      + deps_dne_eps * S_e_loss * NA
        # -------------------------------------------------------
        e_idx = 0
        S_e_loss = 0.0
        for jj in electron_loss_indices:
            S_e_loss -= stoich_matrix[e_idx, jj] * rates[jj]

        dPeloss_dc = np.zeros(n_sp)
        dPeloss_dne_eps = 0.0
        for i_el, jj in enumerate(electron_loss_indices):
            nu_e = -stoich_matrix[e_idx, jj]  # positive number
            dPeloss_dc += eps_mean * NA * nu_e * dR_dc_eloss[i_el, :]
            dPeloss_dne_eps += eps_mean * NA * nu_e * dR_dne_eps_eloss[i_el]
        # Chain rule on ε̄
        dPeloss_dc[e_idx] += deps_dc_e * S_e_loss * NA
        dPeloss_dne_eps += deps_dne_eps * S_e_loss * NA

        # -------------------------------------------------------
        # Assemble: rhs = P_dep/QE - P_elastic - P_inel/QE - P_diff - P_flow - P_eloss
        # (P_dep is const, P_inel already in eV units via /QE in rhs)
        # -------------------------------------------------------

        # Species derivatives (0..n_sp-1)
        jac_row[:n_sp] = -(dPel_dc_e if False else 0.0)  # placeholder, fill below

        # Actually, let's assemble properly:
        # jac_row[k] = +∂P_dep/∂c_k - ∂P_el/∂c_k - ∂P_inel_eV/∂c_k - ∂P_diff/∂c_k - ∂P_eloss/∂c_k
        # For species k ≠ 0 (electron):
        #   ∂P_el/∂c_k = 0, ∂P_dep/∂c_k = 0
        #   ∂P_diff/∂c_k = 0
        #   Only ∂P_inel/∂c_k and ∂P_eloss/∂c_k matter
        jac_row[:n_sp] = -dPinel_dc - dPeloss_dc
        # Add electron-specific terms (P_dep A20 mode has c_e dependence)
        jac_row[e_idx] += dPdep_dc_e - dPel_dc_e - dPdiff_dc_e

        # ne_eps derivative
        jac_row[idx_energy] = dPdep_dne_eps - dPel_dne_eps - dPinel_dne_eps \
                              - dPdiff_dne_eps - dPflow_dne_eps - dPeloss_dne_eps

        # T_gas derivative
        jac_row[idx_Tgas] = dPdep_dTgas - dPel_dTgas - dPdiff_dTgas - dPflow_dTgas

        return jac_row

    @staticmethod
    def get_Te_eV(ne_eps: float, c_e: float, T_gas: float = 300.0) -> float:
        """Te [eV] = (2/3) * eps_mean."""
        n_e = c_e * NA
        eps_thermal = 1.5 * KB * max(T_gas, 200.0) / QE
        if n_e > 1.0:
            eps_mean = max(ne_eps / n_e, eps_thermal)
        else:
            eps_mean = max(1.0, eps_thermal)
        return (2.0 / 3.0) * eps_mean
