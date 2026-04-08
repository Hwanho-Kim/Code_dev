#!/usr/bin/env python3
"""
Parameter Optimization for Plasma-Liquid Simulation.

Fits 3 free parameters to 4 experimental targets:
  Free:   HONO(g), HONO₂(g), H₂O₂(g) [molecules/cm³]
  Fixed:  δ_gas = 10 mm (Silsby 2021, Lee 2023)
          δ_liq = 100 μm (Silsby 2021, stagnant liquid)
          A/V = 100 m⁻¹ (1/depth, petri dish geometry)

Constraints:
  HONO₂/N₂O₅ ratio ∈ [0.01, 1.0] (Cimerman & Hensel 2023, FTIR)

Targets (3.2 kVpp, DI water, 6 min):
  pH=3.61, NO₂⁻=3μM, NO₃⁻=63μM, H₂O₂=11μM
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, brentq

sys.path.insert(0, str(Path(__file__).parent))

from config import HENRY_CONSTANTS, GAS_DIFFUSIVITY, LIQUID_DIFFUSIVITY
from chemistry_utils import molecules_to_molar


# =============================================================================
# Fixed Mass Transfer Parameters (literature values)
# =============================================================================
FIXED_DELTA_GAS = 0.01     # 10 mm — Lee 2023 (1-10 mm range, upper bound)
FIXED_DELTA_LIQ = 0.0001   # 100 μm — Silsby 2021 (stagnant liquid, 10-100 μm)
FIXED_AV_RATIO = 100.0     # 1/depth for 10mm petri dish

# R32: O₃ + NO₂⁻ → NO₃⁻ + O₂, k = 5.0e5 M⁻¹s⁻¹ (Liu 2015, Table 1)
K_R32 = 5.0e5

# HONO₂/N₂O₅ ratio bounds (Cimerman & Hensel 2023, PCPP, FTIR in O₃ mode)
HONO2_N2O5_RATIO_LO = 0.01
HONO2_N2O5_RATIO_HI = 1.0

# =============================================================================
# Experimental Targets
# =============================================================================
TARGETS = {
    'pH': 3.61,
    'NO2-': 3.0,    # μM
    'NO3-': 63.0,   # μM
    'H2O2': 11.0,   # μM
}

WEIGHTS = {
    'pH': 0.5,
    'NO2-': 1.0 / 1.5**2,
    'NO3-': 1.0 / 5.0**2,
    'H2O2': 1.0 / 2.0**2,
}

NORM = {
    'pH': 3.61,
    'NO2-': 3.0,
    'NO3-': 63.0,
    'H2O2': 11.0,
}

# Physical constants
H_N2O5 = HENRY_CONSTANTS['N2O5']     # 51.34
H_HONO2 = HENRY_CONSTANTS['HONO2']   # 5.1e6
H_HONO = HENRY_CONSTANTS['HONO']     # 1198
H_H2O2 = HENRY_CONSTANTS['H2O2']     # 2.1e6

Ka_HONO = 10**(-3.4)
Ka_H2O2 = 10**(-11.65)
Kw = 1e-14


# =============================================================================
# Mass Transfer Coefficient (two-film theory, Lee 2023 Eq. 9-10)
# =============================================================================
def compute_k_mt(species, delta_gas, delta_liq, av_ratio):
    H = HENRY_CONSTANTS.get(species, 1.0)
    D_gas = GAS_DIFFUSIVITY.get(species, 1.5e-5)
    D_liq = LIQUID_DIFFUSIVITY.get(species, 1.5e-9)

    num = D_gas * D_liq * delta_liq
    den = D_gas * delta_liq + D_liq * delta_gas * H
    D_adj = num / max(den, 1e-30)

    return (D_adj / delta_liq) * av_ratio


# =============================================================================
# Fast Simplified Model
# =============================================================================
class FastSimulator:
    """
    Simplified mass-transfer + instant-chemistry model for optimization.

    δ_gas and δ_liq are FIXED from literature.
    Free parameters: A/V, HONO(g), HONO₂(g), H₂O₂(g).
    """

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.n_steps = len(df)
        self.dt = 1.0

        n2o5 = np.maximum(df['N2O5'].values, 0.0)
        self.n2o5_molar = molecules_to_molar(n2o5)
        self.sum_n2o5_molar = np.sum(self.n2o5_molar)
        self.mean_n2o5 = np.mean(n2o5)
        self.max_n2o5 = np.max(n2o5)
        self.total_time = self.n_steps * self.dt

        o3 = np.maximum(df['O3'].values, 0.0)
        self.o3_molar = molecules_to_molar(o3)

        self.hono2_lo = self.max_n2o5 * HONO2_N2O5_RATIO_LO
        self.hono2_hi = self.max_n2o5 * HONO2_N2O5_RATIO_HI

        print(f"FastSimulator: {self.n_steps} steps, "
              f"Σ[N₂O₅]={self.sum_n2o5_molar:.4e} mol·s/L")
        print(f"  N₂O₅ max={self.max_n2o5:.3e}, mean={self.mean_n2o5:.3e} cm⁻³")
        print(f"  O₃ mean={np.mean(o3):.3e} cm⁻³")
        print(f"  HONO₂ bounds: [{self.hono2_lo:.2e}, {self.hono2_hi:.2e}] cm⁻³")
        print(f"  Fixed: δ_gas={FIXED_DELTA_GAS*1000:.0f}mm, "
              f"δ_liq={FIXED_DELTA_LIQ*1e6:.0f}μm")

    def simulate(self, delta_gas, delta_liq, av_ratio,
                 hono_gas, hono2_gas, h2o2_gas):
        k_n2o5 = compute_k_mt('N2O5', delta_gas, delta_liq, av_ratio)
        k_hono2 = compute_k_mt('HONO2', delta_gas, delta_liq, av_ratio)
        k_hono = compute_k_mt('HONO', delta_gas, delta_liq, av_ratio)
        k_h2o2 = compute_k_mt('H2O2', delta_gas, delta_liq, av_ratio)
        k_o3 = compute_k_mt('O3', delta_gas, delta_liq, av_ratio)

        H_O3 = HENRY_CONSTANTS.get('O3', 0.2298)

        hono_molar = molecules_to_molar(max(hono_gas, 0.0))
        hono2_molar = molecules_to_molar(max(hono2_gas, 0.0))
        h2o2_molar = molecules_to_molar(max(h2o2_gas, 0.0))

        hono_flux = k_hono * H_HONO * hono_molar
        hono2_flux = k_hono2 * H_HONO2 * hono2_molar
        h2o2_flux = k_h2o2 * H_H2O2 * h2o2_molar

        NO3_from_n2o5 = 0.0
        HONO2_dissolved = 0.0
        HONO_dissolved_gross = 0.0
        H2O2_dissolved = 0.0
        NO3_from_R32 = 0.0
        no2_minus = 0.0

        for i in range(self.n_steps):
            NO3_from_n2o5 += 2.0 * k_n2o5 * H_N2O5 * self.n2o5_molar[i] * self.dt
            HONO2_dissolved += hono2_flux * self.dt
            hono_in = hono_flux * self.dt
            HONO_dissolved_gross += hono_in
            H2O2_dissolved += h2o2_flux * self.dt

            no2_minus += hono_in

            # R32: all dissolved O₃ reacts with NO₂⁻ (dominates other O₃ sinks by >5 OoM at pH~4)
            o3_dissolved = k_o3 * H_O3 * self.o3_molar[i] * self.dt
            r32_loss = min(o3_dissolved, no2_minus * 0.99) if no2_minus > 1e-15 else 0.0
            no2_minus -= r32_loss
            NO3_from_R32 += r32_loss

        total_NO3 = NO3_from_n2o5 + HONO2_dissolved + NO3_from_R32
        HONO_dissolved_net = max(no2_minus, 0.0)

        def charge_balance_residual(log_h):
            h = 10.0 ** log_h
            no2 = Ka_HONO / (Ka_HONO + h) * HONO_dissolved_net
            ho2 = Ka_H2O2 / (Ka_H2O2 + h) * H2O2_dissolved
            oh = Kw / h
            return h - total_NO3 - no2 - oh - ho2

        try:
            log_h = brentq(charge_balance_residual, -14.0, 0.0, xtol=1e-12)
            H_plus = 10.0 ** log_h
        except ValueError:
            H_plus = max(total_NO3 + HONO_dissolved_net * 0.5, 1e-14)

        pH = -np.log10(max(H_plus, 1e-14))
        NO2_minus_final = Ka_HONO / (Ka_HONO + H_plus) * HONO_dissolved_net
        H2O2_aq = H_plus / (Ka_H2O2 + H_plus) * H2O2_dissolved

        return {
            'pH': pH,
            'NO2-': NO2_minus_final * 1e6,
            'NO3-': total_NO3 * 1e6,
            'H2O2': H2O2_aq * 1e6,
            '_NO3_from_n2o5': NO3_from_n2o5 * 1e6,
            '_HONO2_dissolved': HONO2_dissolved * 1e6,
            '_NO3_from_R32': NO3_from_R32 * 1e6,
            '_HONO_dissolved_gross': HONO_dissolved_gross * 1e6,
            '_HONO_dissolved_net': HONO_dissolved_net * 1e6,
            '_H2O2_dissolved': H2O2_dissolved * 1e6,
            '_H+': H_plus,
            '_k_n2o5': k_n2o5,
            '_k_hono2': k_hono2,
            '_k_hono': k_hono,
            '_k_h2o2': k_h2o2,
            '_k_o3': k_o3,
        }

    def objective(self, x):
        """x = [log10(HONO+1), log10(HONO₂+1), log10(H₂O₂+1)]"""
        params = self._decode(x)
        result = self.simulate(*params)

        cost = 0.0
        for key in TARGETS:
            r = (result[key] - TARGETS[key]) / NORM[key]
            cost += WEIGHTS[key] * NORM[key]**2 * r**2

        return cost

    def _decode(self, x):
        hono_gas = max(10.0 ** x[0] - 1.0, 0.0)
        hono2_gas = max(10.0 ** x[1] - 1.0, 0.0)
        h2o2_gas = max(10.0 ** x[2] - 1.0, 0.0)
        return FIXED_DELTA_GAS, FIXED_DELTA_LIQ, FIXED_AV_RATIO, hono_gas, hono2_gas, h2o2_gas

    def _encode(self, delta_gas, delta_liq, av_ratio,
                hono_gas, hono2_gas, h2o2_gas):
        return [
            np.log10(hono_gas + 1.0),
            np.log10(hono2_gas + 1.0),
            np.log10(h2o2_gas + 1.0),
        ]


# =============================================================================
# Phase 2: Analytical Initial Estimation
# =============================================================================
def analytical_estimate(sim):
    """Back-calculate starting parameters with fixed δ_gas, δ_liq, A/V."""
    print("\n" + "=" * 70)
    print("PHASE 2: Analytical Initial Estimation")
    print(f"  Fixed: δ_gas={FIXED_DELTA_GAS*1000:.0f}mm, "
          f"δ_liq={FIXED_DELTA_LIQ*1e6:.0f}μm, "
          f"A/V={FIXED_AV_RATIO:.0f} m⁻¹")
    print("=" * 70)

    dg = FIXED_DELTA_GAS
    dl = FIXED_DELTA_LIQ
    av_init = FIXED_AV_RATIO
    print(f"  A/V = {av_init:.1f} m⁻¹ (fixed, 1/depth)")

    # Step 2: HONO₂ for remaining ~8 μM NO₃⁻
    k_hono2 = compute_k_mt('HONO2', dg, dl, av_init)
    remaining_no3 = 8e-6
    if k_hono2 * H_HONO2 * sim.total_time > 0:
        c_molar = remaining_no3 / (k_hono2 * H_HONO2 * sim.total_time)
        hono2_init = c_molar / molecules_to_molar(1.0)
    else:
        hono2_init = sim.hono2_lo
    hono2_init = np.clip(hono2_init, sim.hono2_lo, sim.hono2_hi)
    print(f"  HONO₂(g): {hono2_init:.3e} cm⁻³ (ratio: {hono2_init/sim.max_n2o5:.3f})")

    # Step 3: HONO for NO₂⁻ = 3 μM
    H_plus_approx = 10**(-3.61)
    alpha_no2 = Ka_HONO / (Ka_HONO + H_plus_approx)
    hono_total_needed = 3e-6 / alpha_no2
    k_hono = compute_k_mt('HONO', dg, dl, av_init)
    if k_hono * H_HONO * sim.total_time > 0:
        c_molar = hono_total_needed / (k_hono * H_HONO * sim.total_time)
        hono_init = c_molar / molecules_to_molar(1.0)
    else:
        hono_init = 1e12
    print(f"  HONO(g):  {hono_init:.3e} cm⁻³")

    # Step 4: H₂O₂ for 11 μM
    k_h2o2 = compute_k_mt('H2O2', dg, dl, av_init)
    if k_h2o2 * H_H2O2 * sim.total_time > 0:
        c_molar = 11e-6 / (k_h2o2 * H_H2O2 * sim.total_time)
        h2o2_init = c_molar / molecules_to_molar(1.0)
    else:
        h2o2_init = 1e13
    print(f"  H₂O₂(g): {h2o2_init:.3e} cm⁻³")

    result = sim.simulate(dg, dl, av_init, hono_init, hono2_init, h2o2_init)
    print(f"\n  Initial estimate results:")
    for key in TARGETS:
        print(f"    {key:6s}: {result[key]:.3f}  (target: {TARGETS[key]})")

    return (dg, dl, av_init, hono_init, hono2_init, h2o2_init)


# =============================================================================
# Phase 3: Numerical Optimization
# =============================================================================
def run_optimization(sim, initial_params):
    print("\n" + "=" * 70)
    print(f"PHASE 3: Numerical Optimization (3 free params, A/V={FIXED_AV_RATIO:.0f} fixed)")
    print("=" * 70)

    x0 = sim._encode(*initial_params)
    print(f"  Initial x (log-space): {[f'{v:.3f}' for v in x0]}")

    bounds = [
        (0.0, np.log10(1e17 + 1)),                                  # HONO
        (np.log10(sim.hono2_lo + 1), np.log10(sim.hono2_hi + 1)),  # HONO₂ (ratio constrained)
        (0.0, np.log10(1e16 + 1)),                                  # H₂O₂
    ]

    print(f"  Bounds:")
    labels = ['HONO', 'HONO₂', 'H₂O₂']
    for i, (lo, hi) in enumerate(bounds):
        print(f"    {labels[i]:6s}: [{10**lo:.2e}, {10**hi:.2e}]")

    # Stage 1: differential_evolution
    print("\n  Stage 1: Differential Evolution...")
    t0 = time.time()

    best_cost = [float('inf')]
    eval_count = [0]

    def callback_de(xk, convergence=0):
        eval_count[0] += 1
        cost = sim.objective(xk)
        if cost < best_cost[0]:
            best_cost[0] = cost
            params = sim._decode(xk)
            result = sim.simulate(*params)
            if eval_count[0] % 10 == 0 or cost < 1.0:
                print(f"    Gen {eval_count[0]:4d}: cost={cost:.4f}  "
                      f"pH={result['pH']:.3f}  NO₃⁻={result['NO3-']:.1f}  "
                      f"NO₂⁻={result['NO2-']:.2f}  H₂O₂={result['H2O2']:.1f}")

    de_result = differential_evolution(
        sim.objective,
        bounds=bounds,
        x0=x0,
        maxiter=300,
        seed=42,
        tol=1e-8,
        atol=1e-8,
        mutation=(0.5, 1.5),
        recombination=0.8,
        popsize=25,
        callback=callback_de,
        polish=False,
    )

    t1 = time.time()
    print(f"  DE completed in {t1-t0:.1f}s, {de_result.nfev} evaluations")
    print(f"  DE cost: {de_result.fun:.6f}, success: {de_result.success}")

    # Stage 2: Nelder-Mead polish
    print("\n  Stage 2: Nelder-Mead polish...")
    nm_result = minimize(
        sim.objective,
        de_result.x,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
    )

    t2 = time.time()
    print(f"  NM completed in {t2-t1:.1f}s, {nm_result.nfev} evaluations")
    print(f"  NM cost: {nm_result.fun:.6f}")

    x_opt = nm_result.x if nm_result.fun < de_result.fun else de_result.x
    cost_opt = min(nm_result.fun, de_result.fun)
    params_opt = sim._decode(x_opt)

    return params_opt, cost_opt


# =============================================================================
# Results Display
# =============================================================================
def print_results(sim, params, cost):
    delta_gas, delta_liq, av_ratio, hono_gas, hono2_gas, h2o2_gas = params
    result = sim.simulate(*params)

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\n  --- Fixed Parameters ---")
    print(f"  δ_gas:     {delta_gas*1000:.1f} mm  (Silsby 2021)")
    print(f"  δ_liq:     {delta_liq*1e6:.0f} μm  (Silsby 2021)")
    print(f"  A/V:       {av_ratio:.1f} m⁻¹  (1/depth, petri dish)")

    print(f"\n  --- Fitted Parameters ---")
    print(f"  HONO(g):   {hono_gas:.3e} molecules/cm³")
    print(f"  HONO₂(g):  {hono2_gas:.3e} molecules/cm³  "
          f"(HONO₂/N₂O₅={hono2_gas/sim.max_n2o5:.4f})")
    print(f"  H₂O₂(g):  {h2o2_gas:.3e} molecules/cm³")

    print(f"\n  --- Fit Quality ---")
    print(f"  {'Target':12s}  {'Optimal':>10s}  {'Experiment':>10s}  {'Error':>8s}")
    print(f"  {'-'*50}")
    for key in TARGETS:
        err = result[key] - TARGETS[key]
        pct = abs(err / TARGETS[key]) * 100 if TARGETS[key] != 0 else 0
        print(f"  {key:12s}  {result[key]:10.3f}  {TARGETS[key]:10.3f}  "
              f"{err:+8.3f} ({pct:.1f}%)")

    print(f"\n  Total cost: {cost:.6f}")

    print(f"\n  --- Mass Transfer Breakdown ---")
    print(f"  NO₃⁻ from N₂O₅:   {result['_NO3_from_n2o5']:.2f} μM")
    print(f"  NO₃⁻ from HONO₂:  {result['_HONO2_dissolved']:.2f} μM")
    print(f"  NO₃⁻ from R32:    {result['_NO3_from_R32']:.2f} μM  (O₃+NO₂⁻→NO₃⁻)")
    print(f"  HONO dissolved:    {result['_HONO_dissolved_gross']:.2f} μM "
          f"(net after O₃: {result['_HONO_dissolved_net']:.2f} μM)")
    print(f"  H₂O₂ dissolved:   {result['_H2O2_dissolved']:.2f} μM")

    print(f"\n  --- Mass Transfer Coefficients ---")
    print(f"  k_mt(N₂O₅):  {result['_k_n2o5']:.4e} s⁻¹")
    print(f"  k_mt(HONO₂): {result['_k_hono2']:.4e} s⁻¹")
    print(f"  k_mt(HONO):  {result['_k_hono']:.4e} s⁻¹")
    print(f"  k_mt(H₂O₂): {result['_k_h2o2']:.4e} s⁻¹")
    print(f"  k_mt(O₃):    {result['_k_o3']:.4e} s⁻¹")


# =============================================================================
# Main
# =============================================================================
def main():
    csv_path = str(Path(__file__).parent.parent
                   / "empty chamber" / "empty chamber" / "1kHz3.2kVpp.csv")

    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    t_start = time.time()
    sim = FastSimulator(csv_path)

    # Validate fast model against baseline (original defaults: δ_gas=1mm, δ_liq=100μm, A/V=100)
    print("\n--- Fast Model Validation (baseline params) ---")
    baseline = sim.simulate(0.001, 0.0001, 100.0, 0.0, 0.0, 0.0)
    print(f"  Fast model:  pH={baseline['pH']:.3f}, NO₃⁻={baseline['NO3-']:.1f} μM")
    print(f"  Full ODE:    pH=3.218, NO₃⁻=606.1 μM")
    print(f"  Agreement:   NO₃⁻ fast/ODE = {baseline['NO3-']/606.1:.3f}")

    initial_params = analytical_estimate(sim)
    optimal_params, cost = run_optimization(sim, initial_params)
    print_results(sim, optimal_params, cost)

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    save_params(optimal_params, sim)


def save_params(params, sim):
    delta_gas, delta_liq, av_ratio, hono_gas, hono2_gas, h2o2_gas = params
    output = f"""# Optimal Parameters from run_optimizer.py (constrained fit)
# Fitted to: 3.2 kVpp, DI water, 6 min
# Date: {time.strftime('%Y-%m-%d %H:%M')}
# Fixed: δ_gas={delta_gas*1000:.0f}mm (Silsby 2021), δ_liq={delta_liq*1e6:.0f}μm (Silsby 2021)
# Constraint: HONO₂/N₂O₅ ∈ [{HONO2_N2O5_RATIO_LO}, {HONO2_N2O5_RATIO_HI}] (Cimerman 2023)
# HONO₂/N₂O₅ = {hono2_gas/sim.max_n2o5:.4f}

mass_transfer:
  delta_x_gas: {delta_gas:.6f}    # {delta_gas*1000:.0f} mm (FIXED)
  delta_x_liq: {delta_liq:.6f}   # {delta_liq*1e6:.0f} μm (FIXED)
  area_to_volume_ratio: {av_ratio:.4f}  # m⁻¹ (depth ≈ {1/av_ratio*1000:.1f} mm)

gas_phase_unmeasured:
  HONO: {hono_gas:.6e}    # molecules/cm³
  HONO2: {hono2_gas:.6e}   # molecules/cm³
  H2O2: {h2o2_gas:.6e}    # molecules/cm³
"""
    path = Path(__file__).parent / "optimal_params.yaml"
    with open(path, 'w') as f:
        f.write(output)
    print(f"\nSaved optimal parameters to: {path}")


if __name__ == "__main__":
    main()
