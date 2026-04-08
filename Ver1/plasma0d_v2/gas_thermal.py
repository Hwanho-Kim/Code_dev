"""Gas temperature evolution.

dTg/dt = (Q_elastic + Q_chem - Q_wall - Q_flow) / (rho * cp)

Q_elastic is now passed in directly from the solver (computed from BOLSIG+ A21).
"""

import numpy as np
from .constants import KB, ME, QE, NA, R_GAS


class GasThermal:

    def __init__(self, P_gas: float, T_wall: float = 300.0):
        self.P_gas = P_gas
        self.T_wall = T_wall
        self.wall_loss_freq = 100.0
        self.cp_avg = 1000.0
        self.M_avg = 0.028

    def configure(self, M_avg: float = 0.028, cp_avg: float = 1000.0,
                  wall_loss_freq: float = 100.0):
        self.M_avg = M_avg
        self.cp_avg = cp_avg
        self.wall_loss_freq = wall_loss_freq

    def compute_Tgas_rhs(self, T_gas: float, Q_elastic_Wm3: float,
                          tau: float,
                          Q_rxn_Wm3: float = 0.0,
                          Q_e_loss_Wm3: float = 0.0) -> float:
        """Compute dT_gas/dt [K/s].

        Args:
            T_gas: current gas temperature [K]
            Q_elastic_Wm3: elastic heating [W/m³] = n_e * N_gas * A21(ε̄) * QE
            tau: residence time [s]
            Q_rxn_Wm3: chemical enthalpy release [W/m³]
                = -Σ(ΔH × R) for non-EI reactions (exothermic > 0)
            Q_e_loss_Wm3: electron destruction heating [W/m³]
                = ε̄ × Σ(R_loss × NA) × QE, from DR/AT reactions
        """
        rho = self.P_gas * self.M_avg / (R_GAS * T_gas)
        rho_cp = rho * self.cp_avg

        if rho_cp < 1e-10:
            return 0.0

        Q_elastic = min(Q_elastic_Wm3, 1e15)

        Q_wall = rho_cp * self.wall_loss_freq * (T_gas - self.T_wall)

        Q_flow = 0.0
        if 0 < tau < 1e9:
            T_inlet = 300.0
            Q_flow = rho_cp * (T_gas - T_inlet) / tau

        Q_rxn = min(Q_rxn_Wm3, 1e15)
        Q_e_loss = min(Q_e_loss_Wm3, 1e15)

        return (Q_elastic + Q_rxn + Q_e_loss - Q_wall - Q_flow) / rho_cp

    def compute_Tgas_jacobian_row(self, T_gas: float, Q_elastic_Wm3: float,
                                   tau: float,
                                   Q_rxn_Wm3: float,
                                   Q_e_loss_Wm3: float,
                                   dQel_dc: np.ndarray,
                                   dQel_dne_eps: float,
                                   dQel_dTgas: float,
                                   dQrxn_dc: np.ndarray,
                                   dQrxn_dne_eps: float,
                                   dQrxn_dTgas: float,
                                   dQeloss_dc: np.ndarray,
                                   dQeloss_dne_eps: float,
                                   n_state: int,
                                   dQeloss_dTgas: float = 0.0) -> np.ndarray:
        """Compute the Jacobian row for dT_gas/dt.

        dT/dt = F / G where F = Q_el + Q_rxn + Q_eloss - Q_wall - Q_flow
        and G = rho * cp = P*M/(R*T) * cp.

        ∂(dT/dt)/∂y_k = (∂F/∂y_k) / G  +  (dT/dt) * correction for T-dep of G

        Returns
        -------
        jac_row : ndarray, shape (n_state,)
        """
        jac_row = np.zeros(n_state)
        n_sp = n_state - 2
        idx_energy = n_sp
        idx_Tgas = n_sp + 1

        T_gas_safe = max(T_gas, 200.0)
        rho = self.P_gas * self.M_avg / (R_GAS * T_gas_safe)
        rho_cp = rho * self.cp_avg
        if rho_cp < 1e-10:
            return jac_row
        inv_rho_cp = 1.0 / rho_cp

        # Current dT/dt
        Q_elastic = min(Q_elastic_Wm3, 1e15)
        Q_rxn = min(Q_rxn_Wm3, 1e15)
        Q_e_loss = min(Q_e_loss_Wm3, 1e15)
        Q_wall = rho_cp * self.wall_loss_freq * (T_gas_safe - self.T_wall)
        Q_flow = 0.0
        if 0 < tau < 1e9:
            T_inlet = 300.0
            Q_flow = rho_cp * (T_gas_safe - T_inlet) / tau

        F = Q_elastic + Q_rxn + Q_e_loss - Q_wall - Q_flow
        dTdt = F * inv_rho_cp

        # ∂F/∂c_k (species): from Q_elastic, Q_rxn, Q_e_loss
        jac_row[:n_sp] = (dQel_dc + dQrxn_dc + dQeloss_dc) * inv_rho_cp

        # ∂F/∂ne_eps
        jac_row[idx_energy] = (dQel_dne_eps + dQrxn_dne_eps + dQeloss_dne_eps) * inv_rho_cp

        # ∂F/∂T_gas is more complex:
        # F depends on T through: Q_elastic(T), Q_rxn(T), Q_wall(T), Q_flow(T)
        # G = rho_cp = P*M*cp/(R*T) → dG/dT = -G/T
        # ∂(F/G)/∂T = (∂F/∂T)/G - F*dG/dT / G² = (∂F/∂T)/G + F/(G*T) = (∂F/∂T)/G + dTdt/T

        # ∂Q_wall/∂T = rho_cp * wall_loss_freq + Q_wall * (-1/T)
        #   (Q_wall = rho_cp * wlf * (T-Tw), rho_cp ∝ 1/T)
        #   d(rho_cp*(T-Tw))/dT = rho_cp - rho_cp*(T-Tw)/T = rho_cp * Tw/T
        # So: ∂Q_wall/∂T = self.wall_loss_freq * rho_cp * self.T_wall / T_gas_safe
        # Actually more carefully:
        # Q_wall = (P*M*cp/(R*T)) * wlf * (T - Tw)
        # dQ_wall/dT = P*M*cp/(R) * wlf * [(-1/T²)*(T-Tw) + 1/T]
        #            = P*M*cp/(R*T) * wlf * [-(T-Tw)/T + 1]
        #            = rho_cp * wlf * Tw / T
        dQwall_dT = rho_cp * self.wall_loss_freq * self.T_wall / T_gas_safe

        # ∂Q_flow/∂T = similar
        # Q_flow = rho_cp * (T-Tin)/tau, tau ∝ T → 1/tau ∝ 1/T
        # Q_flow = P*M*cp/(R*T) * (T-Tin) * Q_slm*T*P_STP/(T_STP*P*V*60000)
        # Simplify: Q_flow = const * (T-Tin)/T * T = const * (T - Tin) [independent of T?]
        # Actually: Q_flow = rho_cp * (T-Tin)/tau, rho_cp = P*M*cp/(R*T), tau = V*T_STP*P/(Q_slm*T*P_STP*60000)
        # rho_cp / tau = P*M*cp/(R*T) * Q_slm*T*P_STP/(V*T_STP*P*60000) = M*cp*Q_slm*P_STP/(R*V*T_STP*60000)
        # This is T-independent! So Q_flow = const_flow * (T - Tin)
        # dQ_flow/dT = const_flow = Q_flow / (T - Tin) if T != Tin
        dQflow_dT = 0.0
        if 0 < tau < 1e9:
            T_inlet = 300.0
            dT_inlet = T_gas_safe - T_inlet
            if abs(dT_inlet) > 0.01:
                dQflow_dT = Q_flow / dT_inlet
            else:
                # Near inlet temperature, use limit: dQ_flow/dT = rho_cp/tau * (1 - (T-Tin)/T)
                dQflow_dT = rho_cp / tau

        dF_dT = dQel_dTgas - dQwall_dT - dQflow_dT
        dF_dT += dQrxn_dTgas
        dF_dT += dQeloss_dTgas

        # Full derivative: ∂(dT/dt)/∂T = dF_dT/G + dTdt/T  (denominator derivative)
        jac_row[idx_Tgas] = dF_dT * inv_rho_cp + dTdt / T_gas_safe

        return jac_row
