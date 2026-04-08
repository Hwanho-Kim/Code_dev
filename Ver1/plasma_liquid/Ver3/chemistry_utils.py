"""
Shared chemistry utility functions for Plasma-Liquid Interaction simulation.

This module provides common chemistry calculations used by both
GasPhasePreprocessor and CompleteAqueousChemistry classes.
"""

import math
from typing import Tuple

import numpy as np

from config import (
    PHYSICAL, N2O4_EQ, WATER, HENRY_CONSTANTS, HENRY_CONSTANT_TYPE,
    PKA_VALUES, ACID_BASE_PAIRS, MASS_TRANSFER, GAS_DIFFUSIVITY, LIQUID_DIFFUSIVITY
)
from utils import safe_calculation, get_logger


# =============================================================================
# Mass Transfer (Two-Film Theory)
# Ref: Lee 2023 Chem. Eng. J., Liu 2015 J. Phys. D
# =============================================================================

def calculate_adjusted_diffusivity(
    species: str,
    H: float,
    D_gas: float = None,
    D_liq: float = None
) -> float:
    """
    Calculate adjusted diffusion coefficient for two-film theory.

    Dadj = (Dg × Dl × δxliq) / (Dg × δxliq + Dl × δxgas × H)

    Ref: Lee 2023 Eq. 10

    Parameters
    ----------
    species : str
        Species name
    H : float
        Henry's law constant (dimensionless)
    D_gas : float, optional
        Gas diffusivity [m²/s]
    D_liq : float, optional
        Liquid diffusivity [m²/s]

    Returns
    -------
    float
        Adjusted diffusion coefficient [m²/s]
    """
    # Get diffusivities
    if D_gas is None:
        D_gas = GAS_DIFFUSIVITY.get(species, MASS_TRANSFER.D_gas_default)
    if D_liq is None:
        D_liq = LIQUID_DIFFUSIVITY.get(species, MASS_TRANSFER.D_liq_default)

    delta_gas = MASS_TRANSFER.delta_x_gas
    delta_liq = MASS_TRANSFER.delta_x_liq

    # Calculate adjusted diffusivity (Eq. 10)
    numerator = D_gas * D_liq * delta_liq
    denominator = D_gas * delta_liq + D_liq * delta_gas * H

    if denominator < 1e-30:
        return D_liq  # Fallback

    D_adj = numerator / denominator
    return D_adj


def calculate_mass_transfer_coefficient(species: str, H: float) -> float:
    """
    Calculate volumetric mass transfer coefficient k_L·a [s⁻¹] for 0D model.

    k_L·a = (A/V) × Dadj / δxliq

    where A/V is the surface area to volume ratio.

    Parameters
    ----------
    species : str
        Species name
    H : float
        Henry's law constant (dimensionless)

    Returns
    -------
    float
        Volumetric mass transfer coefficient [s⁻¹]
    """
    D_adj = calculate_adjusted_diffusivity(species, H)

    # Interfacial mass transfer coefficient
    k_L = D_adj / MASS_TRANSFER.delta_x_liq

    # Volumetric mass transfer coefficient (k_L × a)
    # a = A/V = surface area to volume ratio
    k_L_a = k_L * MASS_TRANSFER.area_to_volume_ratio

    return k_L_a


def apply_mass_transfer_limited_henry(
    species: str,
    gas_conc_molar: float,
    current_aq_conc: float = 0.0,
    delta_t: float = 0.1,
    C_eq_override: float = None
) -> float:
    """
    Apply Henry's law with two-film theory mass transfer limitation.

    Based on Lee et al. 2023 Chem. Eng. J. 458:141425 (Eq. 9, 10):
        Γ_g→l = D_adj × (H×n_g - n_l) / δx_liq

    For 0D model, flux is converted to concentration change:
        dC_liq/dt = k_mt × (C_eq - C_liq)
        ΔC = k_mt × (C_eq - C_liq) × Δt

    where:
        k_mt = (D_adj / δx_liq) × (A/V)  [s⁻¹]
        C_eq = H × C_gas  (equilibrium concentration)

    Parameters
    ----------
    species : str
        Species name
    gas_conc_molar : float
        Gas phase concentration [mol/L]
    current_aq_conc : float
        Current aqueous concentration [mol/L]
    delta_t : float
        Time step for mass transfer [s]
    C_eq_override : float, optional
        Pre-calculated equilibrium concentration [mol/L].
        If provided, uses this instead of calculating from Henry's law.

    Returns
    -------
    float
        Aqueous concentration after mass transfer [mol/L]
    """
    if gas_conc_molar <= 0 and C_eq_override is None:
        return current_aq_conc

    # Get Henry constant
    H = HENRY_CONSTANTS.get(species, 1.0)

    # Equilibrium concentration (use override if provided)
    if C_eq_override is not None:
        C_eq = C_eq_override
    else:
        C_eq = H * gas_conc_molar

    if not MASS_TRANSFER.enable_mass_transfer:
        # Instant equilibrium (original behavior)
        return C_eq

    # Calculate mass transfer coefficient k_mt [s⁻¹]
    # k_mt = (D_adj / δx_liq) × (A/V)
    k_mt = calculate_mass_transfer_coefficient(species, H)

    # Flux-based mass transfer (Lee 2023 Eq. 9)
    # dC_liq/dt = k_mt × (C_eq - C_liq)
    # ΔC = k_mt × (C_eq - C_liq) × Δt
    driving_force = C_eq - current_aq_conc
    delta_C = k_mt * driving_force * delta_t

    # New concentration
    C_new = current_aq_conc + delta_C

    return max(C_new, 0.0)


# =============================================================================
# Equilibrium Calculations
# =============================================================================

@safe_calculation(default_return=0.0)
def calculate_n2o4_equilibrium_constant(T: float = 298.15) -> float:
    """
    Calculate N2O4 equilibrium constant at given temperature.

    Uses van't Hoff equation:
    Kp(T) = exp(ln(Kp_298) + (ΔH/R)*(1/298.15 - 1/T))

    Parameters
    ----------
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        Equilibrium constant Kp at temperature T
    """
    Kp = math.exp(
        math.log(N2O4_EQ.KP_298) +
        (N2O4_EQ.DELTA_H / PHYSICAL.R) * (1 / N2O4_EQ.REF_TEMP - 1 / T)
    )
    return Kp


@safe_calculation(default_return=0.0)
def estimate_n2o4_from_no2(no2_conc: float, T: float = 298.15) -> float:
    """
    Estimate N2O4 concentration from NO2 using equilibrium.

    Formula: n_N2O4 = Kp(T) * (kB*T/P) * n_NO2^2

    Parameters
    ----------
    no2_conc : float
        NO2 concentration in molecules/cm³
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        N2O4 concentration in molecules/cm³
    """
    if no2_conc <= 0:
        return 0.0

    Kp = calculate_n2o4_equilibrium_constant(T)
    factor = PHYSICAL.KB_T_OVER_P * T
    n2o4_conc = Kp * factor * (no2_conc ** 2)

    return n2o4_conc


@safe_calculation(default_return=0.0)
def estimate_h2o2_from_o3(o3_conc: float, ratio: float = 5000.0) -> float:
    """
    Estimate H2O2 concentration from O3.

    In plasma, H2O2 is typically produced at a ratio relative to O3.

    Parameters
    ----------
    o3_conc : float
        O3 concentration in molecules/cm³
    ratio : float
        O3/H2O2 ratio (default: 5000)

    Returns
    -------
    float
        H2O2 concentration in molecules/cm³
    """
    if o3_conc <= 0 or ratio <= 0:
        return 0.0
    return o3_conc / ratio


def calculate_hono_hono2(
    n2o4_conc: float,
    n2o5_conc: float,
    no2_conc: float,
    humidity: float = 0.5,
    T: float = 298.15
) -> Tuple[float, float]:
    """
    Estimate gas-phase HONO and HONO2 from unmeasured species.

    These species cannot be measured by OAS but are expected to exist
    in the gas phase. This function estimates their concentrations
    for use as boundary conditions in the liquid-phase simulation.

    NOTE: This is a gas-phase estimation module. It does NOT modify
    the measured OAS values (N2O4, N2O5). The estimated HONO/HONO2
    are independent gas-phase species that dissolve separately.

    Estimation reactions:
    - N2O4 + H2O(surface) → HONO + HONO2
    - N2O5 + H2O(surface) → 2HONO2
    - NO2 + OH → HONO2 (humidity dependent, minor)

    Parameters
    ----------
    n2o4_conc : float
        N2O4 concentration in molecules/cm³ (from equilibrium estimate)
    n2o5_conc : float
        N2O5 concentration in molecules/cm³ (from OAS measurement)
    no2_conc : float
        NO2 concentration in molecules/cm³ (from OAS measurement)
    humidity : float
        Relative humidity (0-1)
    T : float
        Temperature in Kelvin

    Returns
    -------
    Tuple[float, float]
        (HONO, HONO2) in molecules/cm³
    """
    HONO = 0.0
    HONO2 = 0.0

    # N2O4 hydrolysis
    if n2o4_conc > 0 and humidity > 0:
        k_hydrolysis = 0.1 * humidity * np.exp(-2000 / T)
        conversion = min(k_hydrolysis, 0.5)

        n2o4_reacted = n2o4_conc * conversion
        HONO += n2o4_reacted * 0.5
        HONO2 += n2o4_reacted * 0.5

    # N2O5 hydrolysis (faster reaction)
    if n2o5_conc > 0 and humidity > 0:
        k_hydrolysis = 0.8 * humidity
        conversion = min(k_hydrolysis, 0.95)

        n2o5_reacted = n2o5_conc * conversion
        HONO2 += 2 * n2o5_reacted

    # NO2 + OH reaction (high humidity)
    if no2_conc > 0 and humidity > 0.3:
        oh_production = humidity * 1e-4
        k_no2_oh = 1e-11
        HONO2 += no2_conc * oh_production * k_no2_oh * 100

    return HONO, HONO2


# =============================================================================
# Unit Conversions
# =============================================================================

def molecules_to_molar(molecules_per_cm3: float) -> float:
    """
    Convert molecules/cm³ to mol/L.

    Parameters
    ----------
    molecules_per_cm3 : float
        Concentration in molecules/cm³

    Returns
    -------
    float
        Concentration in mol/L
    """
    return molecules_per_cm3 * (1000 / PHYSICAL.AVOGADRO)


def molar_to_molecules(molar: float) -> float:
    """
    Convert mol/L to molecules/cm³.

    Parameters
    ----------
    molar : float
        Concentration in mol/L

    Returns
    -------
    float
        Concentration in molecules/cm³
    """
    return molar * (PHYSICAL.AVOGADRO / 1000)


def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15


def molecules_to_atm(molecules_per_cm3: float, T: float = 298.15) -> float:
    """
    Convert molecules/cm³ to partial pressure in atm.

    Uses ideal gas law: P = n × kB × T
    where kB = 1.380649e-23 J/K and 1 atm = 101325 Pa

    Parameters
    ----------
    molecules_per_cm3 : float
        Concentration in molecules/cm³
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        Partial pressure in atm
    """
    kB = 1.380649e-23  # Boltzmann constant [J/K]
    # Convert molecules/cm³ to molecules/m³
    n_per_m3 = molecules_per_cm3 * 1e6
    # P = n × kB × T [Pa]
    P_pa = n_per_m3 * kB * T
    # Convert to atm
    return P_pa / 101325.0


# =============================================================================
# Henry's Law
# =============================================================================

def apply_henry_law(
    species: str,
    gas_conc_molecules_cm3: float,
    method: str = "two_film",
    delta_t: float = 0.1,
    current_aq_conc: float = 0.0
) -> float:
    """
    Apply Henry's law with two-film theory mass transfer.

    Based on Lee et al. 2023 Chem. Eng. J. 458:141425 (Eq. 9, 10):
        Γ_g→l = D_adj × (H×n_g - n_l) / δx_liq
        ΔC = k_mt × (C_eq - C_liq) × Δt

    Ref: Lee 2023 Chem. Eng. J., Liu 2015 J. Phys. D

    Parameters
    ----------
    species : str
        Species name (must be in HENRY_CONSTANTS)
    gas_conc_molecules_cm3 : float
        Gas phase concentration in molecules/cm³
    method : str, optional
        Mass transfer method (default: 'two_film')
    delta_t : float
        Time step for mass transfer [s] (default: 0.1)
    current_aq_conc : float, optional
        Current aqueous concentration [mol/L].

    Returns
    -------
    float
        Aqueous phase concentration in mol/L (new concentration)
    """
    # If no gas, return current concentration (no change)
    if gas_conc_molecules_cm3 <= 0:
        return current_aq_conc

    # Two-film theory with mass transfer limitation (default)
    # Calculate equilibrium concentration using Liu 2015 constants
    C_gas_molar = molecules_to_molar(gas_conc_molecules_cm3)
    H = HENRY_CONSTANTS.get(species, 1.0)
    C_eq = H * C_gas_molar

    if species not in HENRY_CONSTANTS:
        C_eq = C_gas_molar

    # Apply two-film theory mass transfer (Lee 2023 flux-based)
    C_new = apply_mass_transfer_limited_henry(
        species, C_gas_molar, current_aq_conc, delta_t, C_eq
    )
    return max(C_new, 0.0)

    # --- Other methods (commented out) ---
    # if method == 'disabled':
    #     # No Henry's law: direct unit conversion only (H = 1)
    #     C_eq = molecules_to_molar(gas_conc_molecules_cm3)
    #     return max(current_aq_conc, C_eq)
    #
    # elif method == 'liu2015':
    #     # Instantaneous Henry's law equilibrium
    #     C_gas_molar = molecules_to_molar(gas_conc_molecules_cm3)
    #     H = HENRY_CONSTANTS.get(species, 1.0)
    #     C_eq = H * C_gas_molar
    #     return max(current_aq_conc, C_eq)


# =============================================================================
# pH and Acid-Base Equilibrium
# =============================================================================

def calculate_pH(h_conc: float) -> float:
    """
    Calculate pH from H+ concentration.

    Parameters
    ----------
    h_conc : float
        H+ concentration in mol/L

    Returns
    -------
    float
        pH value
    """
    if h_conc > 0:
        return -np.log10(h_conc)
    return 7.0  # Neutral pH as default


def h_from_pH(pH: float) -> float:
    """
    Calculate H+ concentration from pH.

    Parameters
    ----------
    pH : float
        pH value

    Returns
    -------
    float
        H+ concentration in mol/L
    """
    return 10 ** (-pH)


def speciate_acid_base(
    total_conc: float,
    pKa: float,
    H_conc: float
) -> Tuple[float, float]:
    """
    Calculate acid and base concentrations from total and pH.

    Uses Henderson-Hasselbalch relationship:
    - alpha_A = Ka / (H + Ka)
    - alpha_HA = H / (H + Ka)

    Parameters
    ----------
    total_conc : float
        Total concentration (HA + A-)
    pKa : float
        Acid dissociation constant
    H_conc : float
        H+ concentration

    Returns
    -------
    Tuple[float, float]
        (HA_conc, A_conc) - acid and conjugate base concentrations
    """
    Ka = 10 ** (-pKa)
    denom = H_conc + Ka

    if denom < 1e-30:
        denom = 1e-30

    alpha_A = Ka / denom      # Fraction as base
    alpha_HA = H_conc / denom  # Fraction as acid

    HA_conc = total_conc * alpha_HA
    A_conc = total_conc * alpha_A

    return HA_conc, A_conc


def get_species_to_total_map() -> dict:
    """
    Create reverse mapping from individual species to total names.

    Returns
    -------
    dict
        {species_name: total_name}
    """
    species_to_total = {}
    for total_name, (acid, base, _) in ACID_BASE_PAIRS.items():
        species_to_total[acid] = total_name
        species_to_total[base] = total_name
    return species_to_total


# =============================================================================
# Data Validation
# =============================================================================

def validate_concentration(conc: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Validate and clip concentration to valid range.

    Parameters
    ----------
    conc : float
        Concentration value
    min_val : float
        Minimum valid value
    max_val : float
        Maximum valid value

    Returns
    -------
    float
        Validated concentration
    """
    if np.isnan(conc) or np.isinf(conc):
        return min_val
    return np.clip(conc, min_val, max_val)
