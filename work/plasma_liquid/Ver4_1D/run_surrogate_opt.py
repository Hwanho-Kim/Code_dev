#!/usr/bin/env python3
"""
Surrogate-assisted parameter estimation for unmeasured gas-phase species.

Two-stage approach:
  Stage 1: Fast 0D surrogate (mass-transfer + instant chemistry) finds
           candidate HONO/HONO2/H2O2 gas-phase concentrations via DE.
  Stage 2: Full 1D PDE solver validates the candidate.
           If 1D disagrees, bias correction is fed back to the 0D model
           and the cycle repeats.

The 0D surrogate uses the same gas_alpha BC formulation as the 1D solver:
  1/k_gi = δ_gas/D_g + 4/(α_b·v̄)
  k_mt = k_gi / H_cc

Usage:
    python run_surrogate_opt.py                    # default δ_gas=10mm
    python run_surrogate_opt.py --delta_gas 3      # δ_gas=3mm
    python run_surrogate_opt.py --max_iter 5       # max feedback iterations
"""

import sys
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, brentq

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'Figures'))

from config_1d import (
    PHYSICAL, HENRY_CONSTANTS, GAS_DIFFUSIVITY, LIQUID_DIFFUSIVITY,
    D_GAS_DEFAULT, D_LIQ_DEFAULT, N2O4_EQ, MASS_TRANSFER,
    ACID_BASE_PAIRS, GAS_TO_AQUEOUS_MAP,
)
from pde_solver import MOLAR_MASS

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

R_LATM = 0.08206   # L·atm/(mol·K)
T = 298.15          # K
R_GAS = 8.314       # J/(mol·K)
KW = 1e-14

# pKa values for acid-base speciation
PKA = {
    'HONO': 3.4,
    'H2O2': 11.65,
    'HONO2': -1.34,
}

# Experimental targets (3.2 kVpp, DI water, 12 min)
TARGETS = {
    'pH': 3.61,
    'NO2-': 3.0,     # µM
    'NO3-': 63.0,    # µM
    'H2O2': 11.0,    # µM
}

# Relative weights: 1/(fractional tolerance)².
# NO3⁻ is the primary mass transfer validation target → tightest tolerance.
WEIGHTS = {
    'pH': 1.0 / 0.03**2,     # ±3% relative (≈ ±0.1 pH)
    'NO2-': 1.0 / 0.30**2,   # ±30% relative
    'NO3-': 1.0 / 0.10**2,   # ±10% relative (primary target)
    'H2O2': 1.0 / 0.20**2,   # ±20% relative
}

# ═══════════════════════════════════════════════════════════════════════
# 0D Surrogate: gas_alpha k_mt
# ═══════════════════════════════════════════════════════════════════════

def compute_k_mt_gas_alpha(species: str, delta_gas: float,
                           alpha_b: float) -> float:
    """Gas-side + interfacial resistance k_mt [m/s] in liquid-phase units.

    Identical to pde_solver.py gas_alpha branch for consistency.
    HENRY_CONSTANTS are dimensionless H_cc, but the 1D code applies
    an additional R*T factor.  We replicate that here so 0D and 1D
    use exactly the same k_mt × C_eq product.
    """
    H = HENRY_CONSTANTS.get(species, 1.0)          # dimensionless H_cc
    H_conv = H * R_LATM * T                        # same as 1D code
    D_g = GAS_DIFFUSIVITY.get(species, D_GAS_DEFAULT)
    M = MOLAR_MASS.get(species, 48.0)
    v_th = math.sqrt(8.0 * R_GAS * T / (math.pi * M * 1e-3))

    k_gas = D_g / delta_gas
    k_int = alpha_b * v_th / 4.0
    k_gi = 1.0 / (1.0 / k_gas + 1.0 / k_int)
    return k_gi / H_conv


def molecules_to_molar(n: float) -> float:
    """molecules/cm³ → mol/L."""
    return n * 1e3 / PHYSICAL.AVOGADRO


# Species-specific α_b (same as config_1d.py)
ALPHA_B_SPECIES = {
    'N2O5':  0.03, 'O3':   0.05, 'H2O2': 0.1,
    'NO':    0.001, 'NO2':  0.03, 'NO3':  0.03,
    'N2O4':  0.03, 'HONO': 0.05, 'HONO2': 0.07,
}


@dataclass
class SurrogateResult:
    """Result from 0D surrogate simulation."""
    pH: float
    no2_uM: float    # NO2⁻ [µM]
    no3_uM: float    # NO3⁻ [µM]
    h2o2_uM: float   # H2O2 [µM]
    # Breakdown
    no3_from_n2o5: float
    no3_from_hono2: float
    no3_from_r32: float
    hono_dissolved_net: float
    h2o2_dissolved: float


class FastSurrogate0D:
    """0D mass-transfer + instant-chemistry model with gas_alpha BC.

    Physics:
      - Gas species dissolve via gas_alpha k_mt.
      - N2O5(aq) → 2H⁺ + 2NO3⁻ (instant hydrolysis, R98).
      - HONO2(aq) → H⁺ + NO3⁻ (strong acid, pKa=-1.34).
      - HONO(aq) ↔ H⁺ + NO2⁻ (weak acid, pKa=3.4).
      - H2O2(aq) ↔ H⁺ + HO2⁻ (weak acid, pKa=11.65).
      - R32: O3 + NO2⁻ → NO3⁻ + O2 (k=5e5 M⁻¹s⁻¹, dominant O3 sink).
      - C_surface ≈ 0 for N2O5 (完全消費, k_rxn >> k_mt).
      - C_surface ≈ C_eq for HONO/HONO2/H2O2 (low reactivity → well mixed).

    0D assumes well-mixed liquid for stable species. Reactive species
    (O3, NO3 radical) are treated as surface-consumed.
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.n_steps = len(df)
        self.dt = 2.0  # OAS 2-sec intervals
        self.total_time = self.n_steps * self.dt

        # Load and preprocess gas data
        self._load_gas(df)

        # Precompute C_eq time series for measured species (δ_gas independent)
        self._ceq_n2o5 = self._to_ceq('N2O5')
        self._ceq_o3 = self._to_ceq('O3')

        print(f"FastSurrogate0D: {self.n_steps} steps × {self.dt}s "
              f"= {self.total_time:.0f}s")

    def _load_gas(self, df: pd.DataFrame) -> None:
        """Load gas data with linear interp + SG smoothing (same as gen_all_figures)."""
        from scipy.signal import savgol_filter

        self.gas_raw = {}
        for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
            if col in df.columns:
                raw = np.maximum(df[col].values.astype(float), 0.0)
                self.gas_raw[col] = self._preprocess(raw)
            else:
                self.gas_raw[col] = np.zeros(self.n_steps)

        # N2O4 from NO2 equilibrium if missing
        if np.all(self.gas_raw.get('N2O4', np.zeros(1)) == 0):
            Kp = math.exp(
                math.log(N2O4_EQ.KP_298)
                + (N2O4_EQ.DELTA_H / PHYSICAL.R)
                * (1 / N2O4_EQ.REF_TEMP - 1 / T)
            )
            self.gas_raw['N2O4'] = (
                Kp * PHYSICAL.KB_T_OVER_P * T * self.gas_raw['NO2']**2
            )

        self.n2o5_max = float(np.max(self.gas_raw['N2O5']))
        self.no2_max = float(np.max(self.gas_raw['NO2']))
        self.o3_max = float(np.max(self.gas_raw['O3']))

    def _preprocess(self, vals: np.ndarray) -> np.ndarray:
        """Linear interp + SG smoothing for below-LOD points."""
        from scipy.signal import savgol_filter

        out = vals.copy()
        n = len(vals)

        # Stable detection start (5 consecutive nonzero)
        run_start, run_len = -1, 0
        stable_start = n
        for i in range(n):
            if vals[i] > 0:
                if run_len == 0:
                    run_start = i
                run_len += 1
                if run_len >= 5:
                    stable_start = run_start
                    break
            else:
                run_len = 0

        if stable_start >= n:
            return np.maximum(out, 0.0)

        # Fill intermittent zeros
        nz_after = [(i, vals[i]) for i in range(stable_start, n) if vals[i] > 0]
        if len(nz_after) >= 2:
            nz_idx = np.array([x[0] for x in nz_after])
            nz_vals = np.array([x[1] for x in nz_after])
            for i in range(stable_start, n):
                if out[i] <= 0:
                    out[i] = np.interp(i, nz_idx, nz_vals)

        # SG smoothing
        sg_win = 15
        stable_region = out[stable_start:]
        if len(stable_region) >= sg_win:
            w = sg_win if sg_win % 2 == 1 else sg_win + 1
            stable_region = savgol_filter(stable_region, window_length=w,
                                          polyorder=3)
            out[stable_start:] = np.maximum(stable_region, 0.0)

        # Ramp before stable start
        first_val = out[stable_start]
        for i in range(stable_start):
            out[i] = first_val * (i / max(stable_start, 1))

        return np.maximum(out, 0.0)

    def _to_ceq(self, species: str) -> np.ndarray:
        """Convert gas-phase molecules/cm³ → C_eq [mol/L] = H × c_gas.

        Uses H directly from HENRY_CONSTANTS (dimensionless H_cc),
        matching 1D solver's set_gas_data() convention.
        """
        H = HENRY_CONSTANTS.get(species, 1.0)
        return H * molecules_to_molar(self.gas_raw[species])

    def simulate(self, hono_gas: float, hono2_gas: float,
                 h2o2_gas: float, delta_gas: float) -> SurrogateResult:
        """Run 0D forward simulation.

        Parameters:
            hono_gas, hono2_gas, h2o2_gas: molecules/cm³ (scalar constants)
            delta_gas: gas boundary layer thickness [m]

        Uses same C_eq convention as 1D: C_eq = H × c_gas_molar.
        Converts flux to bulk concentration via /L (liquid depth).
        """
        dt = self.dt
        L = MASS_TRANSFER.liquid_depth  # 0.01 m

        # C_eq for unmeasured species (constant, same as 1D's set_gas_data)
        ceq_hono = HENRY_CONSTANTS['HONO'] * molecules_to_molar(hono_gas)
        ceq_hono2 = HENRY_CONSTANTS['HONO2'] * molecules_to_molar(hono2_gas)
        ceq_h2o2 = HENRY_CONSTANTS['H2O2'] * molecules_to_molar(h2o2_gas)

        # Compute k_mt for this δ_gas, then /L for bulk-average rate
        k_n2o5 = compute_k_mt_gas_alpha('N2O5', delta_gas, ALPHA_B_SPECIES['N2O5']) / L
        k_o3 = compute_k_mt_gas_alpha('O3', delta_gas, ALPHA_B_SPECIES['O3']) / L
        k_hono = compute_k_mt_gas_alpha('HONO', delta_gas, ALPHA_B_SPECIES['HONO']) / L
        k_hono2 = compute_k_mt_gas_alpha('HONO2', delta_gas, ALPHA_B_SPECIES['HONO2']) / L
        k_h2o2 = compute_k_mt_gas_alpha('H2O2', delta_gas, ALPHA_B_SPECIES['H2O2']) / L

        # Accumulators [mol/L]
        no3_n2o5 = 0.0
        no3_hono2 = 0.0
        no3_r32 = 0.0
        hono_gross = 0.0
        h2o2_total = 0.0
        no2_minus = 0.0

        for i in range(self.n_steps):
            # N2O5: C_surface ≈ 0 (instant hydrolysis) → flux = k × C_eq
            no3_n2o5 += 2.0 * k_n2o5 * self._ceq_n2o5[i] * dt

            # HONO2: strong acid, fully dissolved
            no3_hono2 += k_hono2 * ceq_hono2 * dt

            # HONO: weak acid → NO2⁻ accumulation
            hono_in = k_hono * ceq_hono * dt
            hono_gross += hono_in
            no2_minus += hono_in

            # H2O2: dissolution
            h2o2_total += k_h2o2 * ceq_h2o2 * dt

            # R32: O3 + NO2⁻ → NO3⁻ + O2
            # O3 reacts only within reactive penetration depth:
            #   δ_react = sqrt(D_O3 / (k_R32 × [NO2⁻]))
            # Only fraction δ_react/L of bulk NO2⁻ is accessible.
            if no2_minus > 1e-15:
                D_O3 = 1.75e-9  # m²/s
                k_R32 = 5.0e5   # M⁻¹s⁻¹
                delta_react = math.sqrt(D_O3 / max(k_R32 * no2_minus, 1e-30))
                frac_react = min(delta_react / L, 1.0)
                o3_flux = k_o3 * self._ceq_o3[i] * dt
                r32_loss = min(o3_flux, no2_minus * frac_react)
                no2_minus -= r32_loss
                no3_r32 += r32_loss

        total_no3 = no3_n2o5 + no3_hono2 + no3_r32
        hono_net = max(no2_minus, 0.0)

        # pH from charge balance: [H⁺] = [NO3⁻] + [NO2⁻] + [HO2⁻] + [OH⁻]
        Ka_hono = 10**(-PKA['HONO'])
        Ka_h2o2 = 10**(-PKA['H2O2'])

        def charge_residual(log_h: float) -> float:
            h = 10.0 ** log_h
            no2 = Ka_hono / (Ka_hono + h) * hono_net
            ho2 = Ka_h2o2 / (Ka_h2o2 + h) * h2o2_total
            oh = KW / h
            return h - total_no3 - no2 - ho2 - oh

        try:
            log_h = brentq(charge_residual, -14.0, 0.0, xtol=1e-12)
            h_plus = 10.0 ** log_h
        except ValueError:
            h_plus = max(total_no3 + hono_net * 0.5, 1e-14)

        pH = -math.log10(max(h_plus, 1e-14))
        no2_final = Ka_hono / (Ka_hono + h_plus) * hono_net
        h2o2_aq = h_plus / (Ka_h2o2 + h_plus) * h2o2_total

        return SurrogateResult(
            pH=pH,
            no2_uM=no2_final * 1e6,
            no3_uM=total_no3 * 1e6,
            h2o2_uM=h2o2_aq * 1e6,
            no3_from_n2o5=no3_n2o5 * 1e6,
            no3_from_hono2=no3_hono2 * 1e6,
            no3_from_r32=no3_r32 * 1e6,
            hono_dissolved_net=hono_net * 1e6,
            h2o2_dissolved=h2o2_total * 1e6,
        )


# ═══════════════════════════════════════════════════════════════════════
# 0D Optimizer
# ═══════════════════════════════════════════════════════════════════════

def cost_function(result: SurrogateResult,
                  targets: Dict[str, float]) -> float:
    """Weighted sum of squared relative errors."""
    pred = {
        'pH': result.pH,
        'NO2-': result.no2_uM,
        'NO3-': result.no3_uM,
        'H2O2': result.h2o2_uM,
    }
    cost = 0.0
    for key in targets:
        if targets[key] > 0.5:
            r = (pred[key] - targets[key]) / targets[key]
        else:
            r = pred[key] - targets[key]
        cost += WEIGHTS[key] * r * r
    return cost


def decode_params(x: np.ndarray) -> Tuple[float, float, float, float]:
    """Decode optimizer x-vector → physical parameters.

    x[0] = log10(HONO + 1)     [molecules/cm³]
    x[1] = log10(HONO2 + 1)    [molecules/cm³]
    x[2] = log10(H2O2 + 1)     [molecules/cm³]
    x[3] = log10(δ_gas [mm])   → δ_gas [m]

    Returns (hono, hono2, h2o2, delta_gas_m).
    """
    hono = max(10.0**x[0] - 1.0, 0.0)
    hono2 = max(10.0**x[1] - 1.0, 0.0)
    h2o2 = max(10.0**x[2] - 1.0, 0.0)
    delta_gas_m = 10.0**x[3] * 1e-3  # mm → m
    return hono, hono2, h2o2, delta_gas_m


def optimize_0d(surrogate: FastSurrogate0D,
                targets: Dict[str, float],
                delta_gas_bounds: Tuple[float, float] = (5.0, 30.0),
                seed: int = 42) -> Tuple[np.ndarray, SurrogateResult, float]:
    """Run DE optimization on 0D surrogate.

    Free parameters (4): HONO, HONO2, H2O2, δ_gas.
    Returns (x_opt, result_0d, cost).
    """
    def objective(x):
        hono, hono2, h2o2, dg = decode_params(x)
        result = surrogate.simulate(hono, hono2, h2o2, dg)
        return cost_function(result, targets)

    bounds = [
        (0.0, 16.0),                                    # HONO:  0 ~ 1e16 cm⁻³
        (0.0, 16.0),                                    # HONO2: 0 ~ 1e16 cm⁻³
        (0.0, 16.0),                                    # H2O2:  0 ~ 1e16 cm⁻³
        (math.log10(delta_gas_bounds[0]),                # δ_gas [mm]
         math.log10(delta_gas_bounds[1])),
    ]

    best = {'cost': 1e9, 'n': 0}

    def callback(xk, convergence=0):
        best['n'] += 1
        c = objective(xk)
        if c < best['cost']:
            best['cost'] = c
            hono, hono2, h2o2, dg = decode_params(xk)
            r = surrogate.simulate(hono, hono2, h2o2, dg)
            if best['n'] % 20 == 0 or c < 1.0:
                print(f"    Gen {best['n']:3d}: cost={c:.4f}  "
                      f"pH={r.pH:.3f} NO₃⁻={r.no3_uM:.1f} "
                      f"NO₂⁻={r.no2_uM:.2f} H₂O₂={r.h2o2_uM:.1f} "
                      f"δg={dg*1e3:.1f}mm")

    print("\n  0D DE Optimization (4 params: HONO, HONO₂, H₂O₂, δ_gas)...")
    print(f"  δ_gas bounds: {delta_gas_bounds[0]:.0f}–{delta_gas_bounds[1]:.0f} mm")
    t0 = time.time()
    de_result = differential_evolution(
        objective, bounds=bounds,
        maxiter=500, seed=seed, tol=1e-10,
        mutation=(0.5, 1.5), recombination=0.8,
        popsize=25, callback=callback, polish=True,
    )
    dt = time.time() - t0
    print(f"  DE done in {dt:.1f}s, cost={de_result.fun:.6f}")

    x_opt = de_result.x
    hono, hono2, h2o2, dg = decode_params(x_opt)
    result_0d = surrogate.simulate(hono, hono2, h2o2, dg)

    return x_opt, result_0d, de_result.fun


# ═══════════════════════════════════════════════════════════════════════
# 1D Validation
# ═══════════════════════════════════════════════════════════════════════

def run_1d_validation(hono_gas: float, hono2_gas: float, h2o2_gas: float,
                      delta_gas: float) -> Optional[Dict[str, float]]:
    """Run full 1D PDE simulation with given gas-phase inputs.

    Returns dict with pH, NO2-, NO3-, H2O2 (µM) or None on failure.
    """
    from chemistry_1d import AqueousChemistry1D
    from pde_solver import PDESolver1D

    # Load gas data (same as gen_all_figures.py)
    csv_path = (Path(__file__).parent.parent
                / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv')
    df = pd.read_csv(csv_path)
    times = np.arange(len(df), dtype=float) * 2.0

    from scipy.signal import savgol_filter

    def preprocess(vals):
        out = vals.copy()
        n = len(vals)
        run_start, run_len = -1, 0
        stable_start = n
        for i in range(n):
            if vals[i] > 0:
                if run_len == 0:
                    run_start = i
                run_len += 1
                if run_len >= 5:
                    stable_start = run_start
                    break
            else:
                run_len = 0
        if stable_start >= n:
            return np.maximum(out, 0.0)
        nz_after = [(i, vals[i]) for i in range(stable_start, n) if vals[i] > 0]
        if len(nz_after) >= 2:
            nz_idx = np.array([x[0] for x in nz_after])
            nz_vals = np.array([x[1] for x in nz_after])
            for i in range(stable_start, n):
                if out[i] <= 0:
                    out[i] = np.interp(i, nz_idx, nz_vals)
        sg_win = 15
        stable_region = out[stable_start:]
        if len(stable_region) >= sg_win:
            w = sg_win if sg_win % 2 == 1 else sg_win + 1
            stable_region = savgol_filter(stable_region, window_length=w,
                                          polyorder=3)
            out[stable_start:] = np.maximum(stable_region, 0.0)
        first_val = out[stable_start]
        for i in range(stable_start):
            out[i] = first_val * (i / max(stable_start, 1))
        return np.maximum(out, 0.0)

    gas_conc = {}
    for col in ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5']:
        if col in df.columns:
            gas_conc[col] = preprocess(np.maximum(df[col].values.astype(float), 0.0))
        else:
            gas_conc[col] = np.zeros(len(df))
    if np.all(gas_conc.get('N2O4', np.zeros(1)) == 0):
        Kp = math.exp(
            math.log(N2O4_EQ.KP_298)
            + (N2O4_EQ.DELTA_H / PHYSICAL.R)
            * (1 / N2O4_EQ.REF_TEMP - 1 / T)
        )
        gas_conc['N2O4'] = Kp * PHYSICAL.KB_T_OVER_P * T * gas_conc['NO2']**2

    print("  Running 1D PDE validation...")
    chem = AqueousChemistry1D(saline_mode=False)
    solver = PDESolver1D(
        chemistry=chem,
        dz_min=5e-6,
        stretch_ratio=1.12,
        mass_transfer_eta=1.0,
        saline_mode=False,
        bc_type='gas_alpha',
        alpha_b=None,       # species-specific
        delta_gas=delta_gas,
    )
    solver.set_gas_data(
        times=times, gas_conc_molecules=gas_conc,
        hono_gas=hono_gas, hono2_gas=hono2_gas, h2o2_gas=h2o2_gas,
    )

    t_end = float(times[-1])
    t_eval = np.array([t_end])
    y0 = solver.build_initial_condition(initial_pH=7.0)

    t0 = time.time()
    try:
        result = solver.solve(
            t_span=(0, t_end), t_eval=t_eval, y0=y0,
            verbose=True, dt_poisson=None,
        )
    except Exception as e:
        print(f"  1D solver failed: {e}")
        return None
    wall = time.time() - t0
    print(f"  1D done in {wall:.1f}s")

    if not result['success']:
        print("  1D solver did not converge!")
        return None

    avg = result['spatial_avg']
    return {
        'pH': result['pH_avg'],
        'NO2-': avg.get('NO2-', 0) * 1e6,
        'NO3-': avg.get('NO3-', 0) * 1e6,
        'H2O2': avg.get('H2O2', 0) * 1e6,
    }


# ═══════════════════════════════════════════════════════════════════════
# Feedback Loop
# ═══════════════════════════════════════════════════════════════════════

def compute_bias(result_0d: SurrogateResult,
                 result_1d: Dict[str, float]) -> Dict[str, float]:
    """Compute 1D/0D bias ratio for each target."""
    pred_0d = {
        'pH': result_0d.pH,
        'NO2-': result_0d.no2_uM,
        'NO3-': result_0d.no3_uM,
        'H2O2': result_0d.h2o2_uM,
    }
    bias = {}
    for key in TARGETS:
        if key == 'pH':
            # pH bias as additive offset
            bias[key] = result_1d[key] - pred_0d[key]
        else:
            # Multiplicative ratio for concentrations
            if pred_0d[key] > 0.01:
                bias[key] = result_1d[key] / pred_0d[key]
            else:
                bias[key] = 1.0
    return bias


def apply_bias_to_targets(original_targets: Dict[str, float],
                          bias: Dict[str, float]) -> Dict[str, float]:
    """Correct 0D targets to compensate for systematic 1D/0D discrepancy.

    If 1D gives 2× what 0D predicts, we tell 0D to aim for target/2.
    """
    corrected = {}
    for key in original_targets:
        if key == 'pH':
            corrected[key] = original_targets[key] - bias[key]
        else:
            if bias[key] > 0.01:
                corrected[key] = original_targets[key] / bias[key]
            else:
                corrected[key] = original_targets[key]
    return corrected


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_comparison(iteration: int, result_0d: SurrogateResult,
                     result_1d: Optional[Dict[str, float]],
                     hono: float, hono2: float, h2o2: float,
                     delta_gas_m: float,
                     bias: Optional[Dict[str, float]] = None) -> None:
    """Print comparison table."""
    print(f"\n{'='*72}")
    print(f"  ITERATION {iteration} RESULTS")
    print(f"{'='*72}")

    print(f"\n  Fitted parameters:")
    print(f"    HONO  = {hono:.3e} cm⁻³")
    print(f"    HONO₂ = {hono2:.3e} cm⁻³")
    print(f"    H₂O₂  = {h2o2:.3e} cm⁻³")
    print(f"    δ_gas = {delta_gas_m*1e3:.1f} mm")

    pred_0d = {
        'pH': result_0d.pH,
        'NO2-': result_0d.no2_uM,
        'NO3-': result_0d.no3_uM,
        'H2O2': result_0d.h2o2_uM,
    }

    print(f"\n  {'Target':>8s}  {'Exp':>8s}  {'0D':>8s}  ", end='')
    if result_1d:
        print(f"{'1D':>8s}  {'Bias':>8s}  {'1D err':>8s}")
    else:
        print(f"{'0D err':>8s}")

    print(f"  {'-'*60}")
    for key in ['pH', 'NO2-', 'NO3-', 'H2O2']:
        exp = TARGETS[key]
        p0 = pred_0d[key]
        line = f"  {key:>8s}  {exp:8.3f}  {p0:8.3f}  "
        if result_1d:
            p1 = result_1d[key]
            b = bias[key] if bias else '-'
            if key == 'pH':
                err = abs(p1 - exp)
                b_str = f"{b:+.3f}" if isinstance(b, float) else str(b)
                line += f"{p1:8.3f}  {b_str:>8s}  {err:8.3f}"
            else:
                pct = abs(p1 - exp) / max(exp, 0.01) * 100
                b_str = f"{b:.3f}" if isinstance(b, float) else str(b)
                line += f"{p1:8.3f}  {b_str:>8s}  {pct:7.1f}%"
        else:
            if key == 'pH':
                err = abs(p0 - exp)
                line += f"{err:8.3f}"
            else:
                pct = abs(p0 - exp) / max(exp, 0.01) * 100
                line += f"{pct:7.1f}%"
        print(line)

    print(f"\n  0D breakdown:")
    print(f"    NO₃⁻ from N₂O₅:  {result_0d.no3_from_n2o5:.2f} µM")
    print(f"    NO₃⁻ from HONO₂:  {result_0d.no3_from_hono2:.2f} µM")
    print(f"    NO₃⁻ from R32:    {result_0d.no3_from_r32:.2f} µM")
    print(f"    HONO net (→NO₂⁻): {result_0d.hono_dissolved_net:.2f} µM")
    print(f"    H₂O₂ dissolved:   {result_0d.h2o2_dissolved:.2f} µM")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Surrogate-assisted optimization')
    parser.add_argument('--dg_lo', type=float, default=5.0,
                        help='δ_gas lower bound [mm] (default: 5)')
    parser.add_argument('--dg_hi', type=float, default=30.0,
                        help='δ_gas upper bound [mm] (default: 30)')
    parser.add_argument('--max_iter', type=int, default=3,
                        help='Max feedback iterations (default: 3)')
    parser.add_argument('--skip_1d', action='store_true',
                        help='Skip 1D validation (0D only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for DE (default: 42)')
    args = parser.parse_args()

    csv_path = str(Path(__file__).parent.parent
                   / 'empty chamber' / 'empty chamber' / '1kHz3.2kVpp.csv')

    print("=" * 72)
    print("  Surrogate-Assisted Parameter Estimation")
    print(f"  δ_gas bounds: {args.dg_lo:.0f}–{args.dg_hi:.0f} mm")
    print(f"  max iterations = {args.max_iter}")
    print("=" * 72)

    surrogate = FastSurrogate0D(csv_path)

    # ── Feedback loop ──
    current_targets = dict(TARGETS)
    best_params = None
    best_1d_cost = float('inf')

    for iteration in range(1, args.max_iter + 1):
        print(f"\n{'─'*72}")
        print(f"  ITERATION {iteration}")
        if iteration > 1:
            print(f"  Corrected targets: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in current_targets.items()))
        print(f"{'─'*72}")

        # Stage 1: 0D optimization
        x_opt, result_0d, cost_0d = optimize_0d(
            surrogate, current_targets,
            delta_gas_bounds=(args.dg_lo, args.dg_hi),
            seed=args.seed + iteration,
        )
        hono, hono2, h2o2, delta_gas_m = decode_params(x_opt)

        if args.skip_1d:
            print_comparison(iteration, result_0d, None,
                             hono, hono2, h2o2, delta_gas_m)
            best_params = (hono, hono2, h2o2, delta_gas_m)
            break

        # Stage 2: 1D validation
        result_1d = run_1d_validation(hono, hono2, h2o2, delta_gas_m)
        if result_1d is None:
            print("  1D validation failed. Using 0D result.")
            best_params = (hono, hono2, h2o2, delta_gas_m)
            break

        # Compute bias and check convergence
        bias = compute_bias(result_0d, result_1d)
        print_comparison(iteration, result_0d, result_1d,
                         hono, hono2, h2o2, delta_gas_m, bias)

        # Cost against experimental targets
        cost_1d = 0.0
        for key in TARGETS:
            if key == 'pH':
                r = (result_1d[key] - TARGETS[key]) / 0.05
            elif TARGETS[key] > 0.5:
                r = (result_1d[key] - TARGETS[key]) / TARGETS[key]
            else:
                r = result_1d[key] - TARGETS[key]
            cost_1d += r * r

        print(f"\n  1D cost vs experiment: {cost_1d:.4f}")

        if cost_1d < best_1d_cost:
            best_1d_cost = cost_1d
            best_params = (hono, hono2, h2o2, delta_gas_m)

        # Convergence check: all within tolerance
        converged = True
        for key in TARGETS:
            if key == 'pH':
                if abs(result_1d[key] - TARGETS[key]) > 0.1:
                    converged = False
            else:
                if abs(result_1d[key] - TARGETS[key]) / max(TARGETS[key], 0.01) > 0.15:
                    converged = False

        if converged:
            print("\n  Converged: all targets within tolerance")
            break

        if iteration < args.max_iter:
            # Apply bias correction for next iteration
            current_targets = apply_bias_to_targets(TARGETS, bias)
            print(f"\n  -> Bias-corrected targets for next iteration:")
            for k, v in current_targets.items():
                orig = TARGETS[k]
                print(f"    {k:>8s}: {orig:.3f} -> {v:.3f}")

    # ── Final Summary ──
    if best_params:
        hono, hono2, h2o2, delta_gas_m = best_params
        print(f"\n{'='*72}")
        print(f"  FINAL RECOMMENDED PARAMETERS")
        print(f"{'='*72}")
        print(f"  HONO  = {hono:.3e} molecules/cm³")
        print(f"  HONO₂ = {hono2:.3e} molecules/cm��")
        print(f"  H₂O₂  = {h2o2:.3e} molecules/cm³")
        print(f"  δ_gas = {delta_gas_m*1e3:.1f} mm")

        # Save to YAML
        output_path = Path(__file__).parent / 'optimal_params_surrogate.yaml'
        with open(output_path, 'w') as f:
            f.write(f"# Surrogate-assisted optimization result\n")
            f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"# δ_gas bounds: {args.dg_lo:.0f}-{args.dg_hi:.0f} mm\n")
            f.write(f"# Iterations: {iteration}\n\n")
            f.write(f"gas_phase_unmeasured:\n")
            f.write(f"  HONO: {hono:.6e}\n")
            f.write(f"  HONO2: {hono2:.6e}\n")
            f.write(f"  H2O2: {h2o2:.6e}\n\n")
            f.write(f"mass_transfer:\n")
            f.write(f"  delta_x_gas: {delta_gas_m:.6f}  # {delta_gas_m*1e3:.1f} mm\n")
            f.write(f"  bc_type: gas_alpha\n")
        print(f"\n  Saved to: {output_path}")


if __name__ == '__main__':
    main()
