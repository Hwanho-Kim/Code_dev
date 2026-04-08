"""RHS term-by-term diagnostic for all governing equations.

Runs simulation to near-steady-state, then decomposes every RHS term
to check magnitudes and identify over/underestimated contributions.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, ME, R_GAS


def run_and_diagnose(V_eff_m3, V_reactor_m3, P_W, Q_slm, t_end_s=2.0):
    cfg = load_config(os.path.join(os.path.dirname(__file__), 'plasma0d_v2', 'config.yaml'))
    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm
    cfg['solver'] = {
        't_end': t_end_s,
        'n_points': 2000,
        'method': 'BDF',
        'rtol': 1e-6,
        'atol': 1e-10,
        'max_step': 1e-4,
        'constrained': False,
    }

    solver, y0, t_span, cfg = setup_simulation(cfg, os.path.join(os.path.dirname(__file__), 'plasma0d_v2'))
    scfg = cfg['solver']
    result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                          rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    # Use final state for diagnostics
    y_final = result.y[:, -1]
    t_final = result.t[-1]
    diagnose_state(solver, t_final, y_final, result)
    return result


def diagnose_state(solver, t, y, result):
    sm = solver.sm
    rxn = solver.rxn
    lut = solver.lut
    power = solver.power
    ekin = solver.ekin
    flow = solver.flow
    gth = solver.gth

    n_sp = sm.n_species
    f = solver._vol_ratio

    # --- Unpack state ---
    c = y[:n_sp].copy()
    ne_eps = y[sm.idx_energy]
    T_gas = y[sm.idx_Tgas]
    c = np.maximum(c, 1e-30)

    c_e = c[0]
    n_e = c_e * NA
    T_gas_safe = max(T_gas, 200.0)
    eps_thermal = 1.5 * KB * T_gas_safe / QE

    if n_e > 1.0:
        eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0)
    else:
        eps_mean = max(1.0, eps_thermal)

    Te_eV = (2.0 / 3.0) * eps_mean
    c_total = power.P_gas / (R_GAS * T_gas)
    N_gas = power.P_gas / (KB * T_gas_safe)
    tau = flow.get_residence_time(T_gas)

    # --- LUT queries ---
    k_ei_conc = None
    P_el_eVm3s = 0.0
    Q_elastic_Wm3 = 0.0

    if lut is not None and eps_mean >= lut.eps_range[0]:
        k_ei_conc, Te_eV = lut.get_rate_coefficients_conc(eps_mean)
        transport = lut.get_transport(eps_mean)
        P_el_eVm3s = n_e * N_gas * transport.elastic_power_N
        Q_elastic_Wm3 = P_el_eVm3s * QE

    # --- Reaction rates ---
    rates = rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                        Te_eV=Te_eV, P_gas=power.P_gas)

    # =================================================================
    # HEADER
    # =================================================================
    print("\n" + "=" * 90)
    print("  RHS TERM-BY-TERM DIAGNOSTIC")
    print("=" * 90)
    print(f"  t = {t*1e3:.1f} ms")
    print(f"  V_eff = {solver._V_eff*1e6:.2f} cm³, V_reactor = {solver._V_reactor*1e6:.1f} cm³, "
          f"f = {f:.4f}")
    print(f"  n_e = {n_e:.3e} m⁻³, Te = {Te_eV:.3f} eV, ε̄ = {eps_mean:.3f} eV")
    print(f"  T_gas = {T_gas:.2f} K, N_gas = {N_gas:.3e} m⁻³")
    print(f"  τ = {tau*1e3:.1f} ms, c_total = {c_total:.2f} mol/m³")

    # =================================================================
    # (1) SPECIES EQUATIONS
    # =================================================================
    S_electron, S_thermal = rxn.compute_source_terms_split(rates)
    S_flow = flow.compute_flow_source(c, T_gas)

    print("\n" + "-" * 90)
    print("  [1] SPECIES CONSERVATION: dc_i/dt = S_EI/TE × f_i + S_Arr + S_flow")
    print("-" * 90)

    key_species = ['e', 'CH4', 'CO2', 'N2', 'O2', 'H2', 'CO', 'H2O', 'H', 'O', 'OH',
                   'CH3', 'N2+', 'O2+', 'O-', 'CH4+', 'CO2+', 'N', 'NO',
                   'C2H6', 'C2H4', 'C2H2', 'CH2O', 'N2(A)']

    header = f"{'Species':>10} {'c [mol/m³]':>12} {'S_EI/TE':>12} {'×f_i':>12} {'S_Arr':>12} {'S_flow':>12} {'Total':>12} {'dc/c/τ':>10}"
    print(header)
    print("-" * len(header))

    for name in key_species:
        if not sm.has(name):
            continue
        idx = sm.index(name)
        f_i = solver._f_species[idx]
        s_e = S_electron[idx]
        s_ef = s_e * f_i
        s_t = S_thermal[idx]
        s_f = S_flow[idx]
        total = s_ef + s_t + s_f
        c_val = c[idx]
        rel_rate = total / c_val * tau if c_val > 1e-20 else 0.0
        print(f"{name:>10} {c_val:12.3e} {s_e:12.3e} {s_ef:12.3e} {s_t:12.3e} {s_f:12.3e} {total:12.3e} {rel_rate:10.3f}")

    # Top EI reactions by rate
    print(f"\n  Top EI reaction rates:")
    ei_rates = [(rxn_obj, rates[rxn_obj._global_index]) for rxn_obj in rxn.ei_reactions]
    ei_rates.sort(key=lambda x: -abs(x[1]))
    for rxn_obj, rate in ei_rates[:10]:
        print(f"    R{rxn_obj.id:3d}: {rate:10.3e} mol/(m³·s)  {rxn_obj.formula}")

    # Top Arrhenius reactions by rate
    print(f"\n  Top Arrhenius reaction rates:")
    arr_rates = [(rxn_obj, rates[rxn_obj._global_index]) for rxn_obj in rxn.arrhenius_reactions]
    arr_rates.sort(key=lambda x: -abs(x[1]))
    for rxn_obj, rate in arr_rates[:10]:
        print(f"    R{rxn_obj.id:3d}: {rate:10.3e} mol/(m³·s)  {rxn_obj.formula}")

    # Top TE reactions by rate
    print(f"\n  Top TE-dependent reaction rates:")
    te_rates = [(rxn_obj, rates[rxn_obj._global_index]) for rxn_obj in rxn.te_dependent_reactions]
    te_rates.sort(key=lambda x: -abs(x[1]))
    for rxn_obj, rate in te_rates[:10]:
        print(f"    R{rxn_obj.id:3d}: {rate:10.3e} mol/(m³·s)  {rxn_obj.formula}")

    # =================================================================
    # (2) ELECTRON ENERGY EQUATION
    # =================================================================
    P_dep_Wm3 = power.get_power_density(t)
    P_inel_Wm3 = rxn.compute_electron_energy_loss(rates)
    S_e_loss = rxn.compute_electron_loss_rate(rates)
    P_e_loss_eVm3s = eps_mean * S_e_loss * NA

    # Diffusion loss
    mu_i_N = 2.8e22
    mu_i = mu_i_N / N_gas
    D_a = mu_i * Te_eV
    Lambda_sq = ekin.Lambda_sq
    P_diff_eVm3s = ne_eps * D_a / Lambda_sq

    # Flow loss
    P_flow_eVm3s = ne_eps / tau if tau > 0 else 0.0

    # Convert all to eV/(m³·s)
    P_dep_eV = P_dep_Wm3 / QE
    P_el_eV = P_el_eVm3s
    P_inel_eV = P_inel_Wm3 / QE
    P_eloss_eV = P_e_loss_eVm3s

    total_energy = P_dep_eV - P_el_eV - P_inel_eV - P_diff_eVm3s - P_flow_eVm3s - P_eloss_eV

    print("\n" + "-" * 90)
    print("  [2] ELECTRON ENERGY: d(nₑε̄)/dt [eV/(m³·s)]")
    print("-" * 90)
    print(f"    + P_dep      = {P_dep_eV:12.3e}   (P={P_dep_Wm3:.3e} W/m³ = {P_dep_Wm3*solver._V_eff:.2f} W / V_eff)")
    print(f"    - P_elastic  = {P_el_eV:12.3e}   (n_e·N·A₂₁)")
    print(f"    - P_inelast  = {P_inel_eV:12.3e}   (Σ ΔE_j·R_j·N_A)")
    print(f"    - P_diffus   = {P_diff_eVm3s:12.3e}   (nₑε̄·D_a/Λ²,  D_a={D_a:.3e} m²/s)")
    print(f"    - P_flow     = {P_flow_eVm3s:12.3e}   (nₑε̄/τ)")
    print(f"    - P_e_loss   = {P_eloss_eV:12.3e}   (ε̄·S_loss·N_A, DR/AT)")
    print(f"    ─────────────────────────────")
    print(f"    = d(nₑε̄)/dt = {total_energy:12.3e}")
    print(f"\n    Balance check:")
    pcts = {
        'P_dep': P_dep_eV,
        'P_elastic': P_el_eV,
        'P_inelast': P_inel_eV,
        'P_diffus': P_diff_eVm3s,
        'P_flow': P_flow_eVm3s,
        'P_e_loss': P_eloss_eV,
    }
    total_loss = P_el_eV + P_inel_eV + P_diff_eVm3s + P_flow_eVm3s + P_eloss_eV
    for name, val in pcts.items():
        ref = P_dep_eV if P_dep_eV > 0 else total_loss
        pct = val / ref * 100 if ref > 0 else 0
        print(f"      {name:12s} = {pct:6.1f}% of P_dep")

    # =================================================================
    # (3) GAS TEMPERATURE EQUATION
    # =================================================================
    Q_rxn_te, Q_rxn_arr = rxn.compute_gas_heating_split(rates)
    Q_e_loss_Wm3 = P_e_loss_eVm3s * QE

    rho = power.P_gas * gth.M_avg / (R_GAS * T_gas)
    rho_cp = rho * gth.cp_avg
    Q_wall = rho_cp * gth.wall_loss_freq * (T_gas - gth.T_wall)
    Q_flow_gas = rho_cp * (T_gas - 300.0) / tau if tau > 0 and tau < 1e9 else 0.0

    # Diluted values (as actually passed to Tgas equation)
    Q_el_d = Q_elastic_Wm3 * f
    Q_te_d = Q_rxn_te * f
    Q_eloss_d = Q_e_loss_Wm3 * f

    total_Q = Q_el_d + Q_te_d + Q_rxn_arr + Q_eloss_d - Q_wall - Q_flow_gas
    dTdt = total_Q / rho_cp

    print("\n" + "-" * 90)
    print("  [3] GAS TEMPERATURE: dT/dt [K/s]   (all Q in [W/m³])")
    print("-" * 90)
    print(f"    ρ·cp = {rho_cp:.2f} J/(m³·K)")
    print(f"    + Q_elastic×f    = {Q_el_d:12.3e}   (raw: {Q_elastic_Wm3:.3e} × {f:.4f})")
    print(f"    + Q_rxn_TE×f     = {Q_te_d:12.3e}   (raw: {Q_rxn_te:.3e} × {f:.4f})")
    print(f"    + Q_rxn_Arr      = {Q_rxn_arr:12.3e}   (no dilution)")
    print(f"    + Q_e_loss×f     = {Q_eloss_d:12.3e}   (raw: {Q_e_loss_Wm3:.3e} × {f:.4f})")
    print(f"    - Q_wall         = {Q_wall:12.3e}   (ρcp·ν_w·(T-Tw), ν_w={gth.wall_loss_freq})")
    print(f"    - Q_flow         = {Q_flow_gas:12.3e}   (ρcp·(T-Tin)/τ)")
    print(f"    ─────────────────────────────")
    print(f"    = net Q          = {total_Q:12.3e} W/m³")
    print(f"    = dT/dt          = {dTdt:12.3e} K/s  ({dTdt*1e-3:.3f} K/ms)")

    total_heating = Q_el_d + max(Q_te_d, 0) + max(Q_rxn_arr, 0) + Q_eloss_d
    if total_heating > 0:
        print(f"\n    Heating budget:")
        for name, val in [('Q_elastic×f', Q_el_d), ('Q_rxn_TE×f', Q_te_d),
                          ('Q_rxn_Arr', Q_rxn_arr), ('Q_e_loss×f', Q_eloss_d)]:
            pct = val / total_heating * 100
            print(f"      {name:18s} = {pct:6.1f}% of total heating")

    total_cooling = Q_wall + Q_flow_gas
    if total_cooling > 0:
        print(f"\n    Cooling budget:")
        for name, val in [('Q_wall', Q_wall), ('Q_flow', Q_flow_gas)]:
            pct = val / total_cooling * 100
            print(f"      {name:18s} = {pct:6.1f}% of total cooling")

    # =================================================================
    # (4) CONVERSION SUMMARY
    # =================================================================
    print("\n" + "-" * 90)
    print("  [4] CONVERSION SUMMARY")
    print("-" * 90)
    for name in ['CH4', 'CO2', 'O2', 'N2', 'H2', 'CO', 'H2O']:
        if not sm.has(name):
            continue
        idx = sm.index(name)
        c0 = result.concentrations[idx, 0]
        cf = result.concentrations[idx, -1]
        if c0 > 1e-20:
            conv = (c0 - cf) / c0 * 100
            print(f"    {name:6s}: {c0:.4f} → {cf:.4f} mol/m³  ({conv:+.4f}%)")
        else:
            print(f"    {name:6s}: {c0:.4e} → {cf:.4e} mol/m³")

    print("=" * 90)


if __name__ == '__main__':
    V_eff = 4.0e-7    # 0.4 cm³
    V_reactor = 5.0e-6  # 5 cm³
    P_W = 5.0
    Q_slm = 0.4
    t_end = 2.0  # 2 seconds — ~3τ for V_reactor=5cm³

    run_and_diagnose(V_eff, V_reactor, P_W, Q_slm, t_end_s=t_end)
