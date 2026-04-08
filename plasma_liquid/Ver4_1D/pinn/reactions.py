"""
Gas-phase reaction kinetics for PINN physics loss.

14 reactions (R1, R5-R14, NEW1-NEW3) at 298 K, 1 atm.
All rate constants in cm³/s (bimolecular) or cm⁶/s (three-body).
Heterogeneous reactions parameterized by uptake coefficient γ.

All functions are PyTorch-compatible (differentiable).
"""
import torch
import math

# ── Physical constants ──
T = 298.15                        # K
M = 2.46e19                       # cm⁻³ (air number density at 1 atm, 298 K)
KB = 1.381e-23                    # J/K
P = 101325.0                      # Pa
KB_T_OVER_P = KB * T / P          # cm³ (ideal gas law: kBT/P in m³, but we use cm)
# Actually kBT/P = 1.381e-23 * 298.15 / 101325 = 4.065e-20 m³ = 4.065e-14 cm³
KB_T_P_CM3 = KB * T / P * 1e6     # m³ → cm³

# ── Bimolecular rate constants (cm³ molecule⁻¹ s⁻¹) ──
K_R1  = 1.8e-14    # O₃ + NO → NO₂ + O₂
K_R8  = 7.3e-14    # OH + O₃ → HO₂ + O₂
K_R10 = 1.1e-10    # OH + HO₂ → H₂O + O₂
K_R11 = 2.2e-13    # HO₂ + HO₂ → H₂O₂ + O₂
K_R12 = 6.0e-12    # OH + HONO → NO₂ + H₂O
K_R13 = 1.5e-13    # OH + HONO₂ → NO₃ + H₂O
K_R14 = 1.7e-12    # OH + H₂O₂ → HO₂ + H₂O
K_NEW2 = 8.0e-12   # HO₂ + NO → OH + NO₂
K_NEW3 = 2.0e-15   # HO₂ + O₃ → OH + 2 O₂

# ── Three-body effective rate constants at 1 atm ──
# k_eff = k₀·[M] / (1 + k₀·[M]/k_inf) × F_c^(...) ≈ k₀·[M] for low-pressure limit
# Simplified: use effective bimolecular rate = k₀ × [M]
K0_R7   = 9.1e-32   # NO₂ + OH + M → HONO₂ + M (k₀, cm⁶/s)
K0_R9   = 6.9e-31   # OH + OH + M → H₂O₂ + M (k₀, cm⁶/s)
K0_NEW1 = 7.4e-31   # NO + OH + M → HONO (k₀, cm⁶/s)

K_R7  = K0_R7  * M   # effective bimolecular (cm³/s)
K_R9  = K0_R9  * M   # effective bimolecular (cm³/s)
K_NEW1 = K0_NEW1 * M  # effective bimolecular (cm³/s)

# ── N₂O₄ equilibrium ──
# N₂O₄ ⇌ 2 NO₂;  K_eq = [N₂O₄] / [NO₂]² (cm³)
# K_p(298K) = 6.74e-9 (dimensionless in atm); convert to cm³:
# [N₂O₄] = K_p × (kBT/P) × [NO₂]²
KP_N2O4 = 6.74e-9
KEQ_N2O4 = KP_N2O4 * KB_T_P_CM3   # cm³

# ── Heterogeneous reaction parameters ──
# Rate = γ × v̄ × [X] × A_surface / (4 × V_reactor)
# v̄ = sqrt(8 kBT / (π m)) ≈ mean molecular speed
# For simplicity, define effective first-order rate = γ × v̄ × S/V / 4
# S/V for typical DBD reactor: ~100 m⁻¹ = 1 cm⁻¹ (estimate)
GAMMA_R5 = 0.04     # N₂O₅ + H₂O → 2 HONO₂
GAMMA_R6 = 1e-4     # N₂O₄ + H₂O → HONO + HONO₂
S_OVER_V = 1.0      # cm⁻¹ (reactor surface-to-volume, adjustable)

# Mean molecular speeds (cm/s)
M_N2O5 = 108.0 * 1.66e-24  # g → g per molecule
V_BAR_N2O5 = math.sqrt(8 * KB * 1e7 * T / (math.pi * M_N2O5))  # cm/s (KB in erg)
# Actually let's compute properly: v̄ = sqrt(8RT/(πM)) with R=8.314, M in kg/mol
# N₂O₅: M=108 g/mol → v̄ = sqrt(8*8.314*298/(π*0.108)) = sqrt(59186) = 243 m/s = 2.43e4 cm/s
V_BAR_N2O5 = 2.43e4  # cm/s
V_BAR_N2O4 = 2.53e4  # cm/s (M=92 g/mol)

# Effective first-order loss rate (s⁻¹)
K_HET_R5 = GAMMA_R5 * V_BAR_N2O5 * S_OVER_V / 4.0
K_HET_R6 = GAMMA_R6 * V_BAR_N2O4 * S_OVER_V / 4.0


def compute_n2o4(no2: torch.Tensor) -> torch.Tensor:
    """N₂O₄ from NO₂ equilibrium. [N₂O₄] = K_eq × [NO₂]²"""
    return KEQ_N2O4 * no2 ** 2


def compute_rates(
    o3: torch.Tensor,
    no2: torch.Tensor,
    no3: torch.Tensor,
    n2o5: torch.Tensor,
    hono: torch.Tensor,
    no: torch.Tensor,
    hono2: torch.Tensor,
    h2o2: torch.Tensor,
    oh: torch.Tensor,
    ho2: torch.Tensor,
    q_oh: torch.Tensor,
    q_hono: torch.Tensor,
    q_no: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute all reaction rates for physics loss.

    All concentrations in cm⁻³, rates in cm⁻³/s.

    Returns dict with keys:
        'dNO_dt', 'dHONO2_dt', 'dH2O2_dt', 'dHONO_dt' (rate expressions)
    """
    n2o4 = compute_n2o4(no2)

    # ── Individual reaction rates (cm⁻³/s) ──
    r1   = K_R1   * o3 * no               # O₃ + NO → NO₂ + O₂
    r5   = K_HET_R5 * n2o5                # N₂O₅ + H₂O → 2 HONO₂ (heterogeneous)
    r6   = K_HET_R6 * n2o4                # N₂O₄ + H₂O → HONO + HONO₂ (het.)
    r7   = K_R7   * no2 * oh              # NO₂ + OH + M → HONO₂ + M
    r8   = K_R8   * oh * o3               # OH + O₃ → HO₂ + O₂
    r9   = K_R9   * oh * oh               # OH + OH + M → H₂O₂ + M
    r10  = K_R10  * oh * ho2              # OH + HO₂ → H₂O + O₂
    r11  = K_R11  * ho2 * ho2             # HO₂ + HO₂ → H₂O₂ + O₂
    r12  = K_R12  * oh * hono             # OH + HONO → NO₂ + H₂O
    r13  = K_R13  * oh * hono2            # OH + HONO₂ → NO₃ + H₂O
    r14  = K_R14  * oh * h2o2             # OH + H₂O₂ → HO₂ + H₂O
    new1 = K_NEW1 * no * oh               # NO + OH + M → HONO
    new2 = K_NEW2 * ho2 * no              # HO₂ + NO → OH + NO₂
    new3 = K_NEW3 * ho2 * o3              # HO₂ + O₃ → OH + 2 O₂

    # ── Species rate expressions ──
    dNO_dt    = q_no - r1 - new1 - new2
    dHONO2_dt = 2.0 * r5 + r6 + r7 - r13
    dH2O2_dt  = r9 + r11 - r14
    dHONO_dt  = q_hono + new1 + r6 - r12

    return {
        'dNO_dt':    dNO_dt,
        'dHONO2_dt': dHONO2_dt,
        'dH2O2_dt':  dH2O2_dt,
        'dHONO_dt':  dHONO_dt,
        # Individual rates (for diagnostics)
        'r1': r1, 'r5': r5, 'r6': r6, 'r7': r7, 'r8': r8, 'r9': r9,
        'r10': r10, 'r11': r11, 'r12': r12, 'r13': r13, 'r14': r14,
        'new1': new1, 'new2': new2, 'new3': new3,
        # QSSA inputs (for diagnostics)
        'oh': oh, 'ho2': ho2, 'n2o4': n2o4,
    }


# ── QSSA coefficients (used by qssa.py) ──
# Exported for use in QSSA module
RATE_CONSTANTS = {
    'R1': K_R1, 'R7': K_R7, 'R8': K_R8, 'R9': K_R9,
    'R10': K_R10, 'R11': K_R11, 'R12': K_R12, 'R13': K_R13,
    'R14': K_R14, 'NEW1': K_NEW1, 'NEW2': K_NEW2, 'NEW3': K_NEW3,
    'HET_R5': K_HET_R5, 'HET_R6': K_HET_R6,
}
