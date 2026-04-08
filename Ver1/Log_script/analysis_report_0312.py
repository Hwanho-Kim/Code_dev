"""Comprehensive analysis: Sim vs Exp comparison + Plasma-thermal synergy + CO2 behavior.

Generates data for Sim_exp_comparison_0312 report.
V_reactor=250cm³, V_eff=1.6cm³, P=5W, Q=0.4slm, constant power mode.
"""
import sys, os, io, contextlib, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP, ME

# === Fixed conditions ===
V_EFF_CM3 = 1.6
V_REACTOR_CM3 = 250.0
P_W = 5.0
Q_SLM = 0.4

# === Experimental data ===
EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
T_GAS_LIST_K = list(EXP_DATA.keys())

# === Products of interest ===
PRODUCTS = ['H2', 'CO', 'H2O', 'C2H2', 'C2H4', 'C2H6', 'C3H6', 'C3H8', 'CH3OH', 'CH2O', 'O3']
RADICALS = ['O', 'OH', 'H', 'CH3', 'CH2', 'HO2', 'CHO']


def run_full_analysis(T_target_K):
    """Run simulation and extract all analysis data at one temperature."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    V_eff_m3 = V_EFF_CM3 * 1e-6
    V_reactor_m3 = V_REACTOR_CM3 * 1e-6

    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_SLM
    cfg['T_wall'] = T_target_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_target_K

    Q_actual = Q_SLM * (T_target_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau_est = V_reactor_m3 / Q_actual
    t_end = max(10.0, 5.0 * tau_est)  # 5*tau for full CSTR convergence

    cfg['solver'] = {
        't_end': t_end, 'n_points': 200, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
    }

    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
    scfg = cfg['solver']
    result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                          rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    # ===== Extract final state =====
    sm = solver.sm; rxn = solver.rxn; n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    T_gas = y[sm.idx_Tgas]; ne_eps = y[sm.idx_energy]
    c_e = c[0]; n_e = c_e * NA
    T_gs = max(T_gas, 200.0)
    eps_th = 1.5 * KB * T_gs / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean
    c_total = 101325.0 / (R_GAS * T_gas)
    N_gas = 101325.0 / (KB * T_gs)
    tau = solver.flow.get_residence_time(T_gas)

    # ===== LUT + rates =====
    k_ei_conc = None
    if solver.lut and eps_mean >= solver.lut.eps_range[0]:
        k_ei_conc, _ = solver.lut.get_rate_coefficients_conc(eps_mean)

    rates = rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                        Te_eV=Te_eV, P_gas=101325.0)

    # ===== CH4 conversion =====
    ch4_idx = sm.index('CH4')
    c0_ch4 = result.concentrations[ch4_idx, 0]
    cf_ch4 = result.concentrations[ch4_idx, -1]
    conv_ch4 = (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0

    # ===== CO2 conversion =====
    co2_idx = sm.index('CO2')
    c0_co2 = result.concentrations[co2_idx, 0]
    cf_co2 = result.concentrations[co2_idx, -1]
    conv_co2 = (c0_co2 - cf_co2) / c0_co2 * 100 if c0_co2 > 0 else 0

    # ===== Per-reaction CH4 / CO2 contribution =====
    f_sp = solver._f_species
    veff_mask = rxn._veff_mask

    def get_species_contributions(sp_idx):
        """Get per-reaction contributions for a species."""
        stoich_row = rxn.stoich_matrix[sp_idx, :]
        reactions_data = []
        for j in range(rxn.n_reactions):
            nu = stoich_row[j]
            if nu != 0:
                r = rxn.reactions[j]
                is_veff = veff_mask[j] if veff_mask is not None else \
                          (r.type in ('ELECTRON_IMPACT', 'TE_DEPENDENT'))
                f = f_sp[sp_idx] if is_veff else 1.0
                raw_rate = rates[j]
                eff_rate = raw_rate * f
                contribution = nu * eff_rate
                reactions_data.append({
                    'idx': j, 'id': r.id, 'formula': r.formula, 'type': r.type,
                    'nu': int(nu), 'raw_rate': raw_rate, 'f': f,
                    'eff_rate': eff_rate, 'contribution': contribution,
                })
        reactions_data.sort(key=lambda x: abs(x['contribution']), reverse=True)
        return reactions_data

    ch4_reactions = get_species_contributions(ch4_idx)
    co2_reactions = get_species_contributions(co2_idx)

    # CH4 synergy analysis: split EI vs Arrhenius vs TE
    ch4_cons_ei = sum(x['contribution'] for x in ch4_reactions
                      if x['contribution'] < 0 and x['type'] == 'ELECTRON_IMPACT')
    ch4_cons_arr = sum(x['contribution'] for x in ch4_reactions
                       if x['contribution'] < 0 and x['type'] == 'ARRHENIUS')
    ch4_cons_te = sum(x['contribution'] for x in ch4_reactions
                      if x['contribution'] < 0 and x['type'] == 'TE_DEPENDENT')
    ch4_prod_total = sum(x['contribution'] for x in ch4_reactions if x['contribution'] > 0)
    ch4_cons_total = sum(x['contribution'] for x in ch4_reactions if x['contribution'] < 0)

    # CO2 split
    co2_cons_ei = sum(x['contribution'] for x in co2_reactions
                      if x['contribution'] < 0 and x['type'] == 'ELECTRON_IMPACT')
    co2_cons_arr = sum(x['contribution'] for x in co2_reactions
                       if x['contribution'] < 0 and x['type'] == 'ARRHENIUS')
    co2_cons_te = sum(x['contribution'] for x in co2_reactions
                      if x['contribution'] < 0 and x['type'] == 'TE_DEPENDENT')
    co2_prod_total = sum(x['contribution'] for x in co2_reactions if x['contribution'] > 0)
    co2_cons_total = sum(x['contribution'] for x in co2_reactions if x['contribution'] < 0)

    # Flow source
    S_flow = solver.flow.compute_flow_source(c, T_gas)
    S_flow_ch4 = S_flow[ch4_idx]
    S_flow_co2 = S_flow[co2_idx]

    # ===== Product concentrations =====
    products_data = {}
    for name in PRODUCTS:
        if sm.has(name):
            idx = sm.index(name)
            products_data[name] = {
                'c_final': float(c[idx]),       # mol/m³
                'x_final': float(c[idx] / c_total) if c_total > 0 else 0,  # mole fraction
                'c_final_ppm': float(c[idx] / c_total * 1e6) if c_total > 0 else 0,
            }

    # ===== Radical concentrations =====
    radicals_data = {}
    for name in RADICALS:
        if sm.has(name):
            idx = sm.index(name)
            radicals_data[name] = {
                'c_final': float(c[idx]),
                'n_final': float(c[idx] * NA),  # number density m^-3
            }

    # ===== Energy cost =====
    # CH4 molecules converted per second
    delta_c_ch4 = c0_ch4 - cf_ch4  # mol/m³
    ch4_flow_out = delta_c_ch4 * V_reactor_m3 / tau if tau > 0 else 0  # mol/s converted
    ch4_molecules_per_s = ch4_flow_out * NA  # molecules/s
    EC_eV = (P_W / QE) / ch4_molecules_per_s if ch4_molecules_per_s > 0 else float('inf')
    EC_kJ_mol = P_W / ch4_flow_out / 1000 if ch4_flow_out > 0 else float('inf')

    # SEI = P / Q_actual [J/L]
    SEI_JL = P_W / (Q_actual * 1000)  # W / (m³/s * 1000 L/m³) = J/L

    # ===== Energy budget =====
    P_dep_Wm3 = P_W / V_eff_m3
    P_dep_eV = P_dep_Wm3 / QE

    transport = solver.lut.get_transport(eps_mean) if solver.lut else None
    A21 = transport.elastic_power_N if transport else 0
    P_elastic_eV = n_e * N_gas * A21

    P_inel_Wm3 = rxn.compute_electron_energy_loss(rates)
    P_inel_eV = P_inel_Wm3 / QE

    mu_i_N = 2.8e22
    mu_i = mu_i_N / N_gas
    D_a = mu_i * Te_eV
    P_diff_eV = ne_eps * D_a / (solver.ekin.Lambda ** 2)
    P_flow_eV = ne_eps / tau if tau > 0 else 0
    S_e_loss = rxn.compute_electron_loss_rate(rates)
    P_eloss_eV = eps_mean * S_e_loss * NA

    return {
        'T_target': T_target_K,
        'T_gas_actual': float(T_gas),
        'Te_eV': float(Te_eV),
        'eps_mean': float(eps_mean),
        'n_e': float(n_e),
        'N_gas': float(N_gas),
        'tau': float(tau),
        # Conversions
        'conv_ch4': float(conv_ch4),
        'conv_co2': float(conv_co2),
        'c0_ch4': float(c0_ch4),
        'cf_ch4': float(cf_ch4),
        'c0_co2': float(c0_co2),
        'cf_co2': float(cf_co2),
        # CH4 synergy
        'ch4_cons_ei': float(ch4_cons_ei),
        'ch4_cons_arr': float(ch4_cons_arr),
        'ch4_cons_te': float(ch4_cons_te),
        'ch4_prod_total': float(ch4_prod_total),
        'ch4_cons_total': float(ch4_cons_total),
        'ch4_reactions': ch4_reactions,
        'S_flow_ch4': float(S_flow_ch4),
        # CO2 analysis
        'co2_cons_ei': float(co2_cons_ei),
        'co2_cons_arr': float(co2_cons_arr),
        'co2_cons_te': float(co2_cons_te),
        'co2_prod_total': float(co2_prod_total),
        'co2_cons_total': float(co2_cons_total),
        'co2_reactions': co2_reactions,
        'S_flow_co2': float(S_flow_co2),
        # Products / Radicals
        'products': products_data,
        'radicals': radicals_data,
        # Energy cost
        'EC_eV': float(EC_eV),
        'EC_kJ_mol': float(EC_kJ_mol),
        'SEI_JL': float(SEI_JL),
        # Energy budget (eV/(m³·s))
        'P_dep': float(P_dep_eV),
        'P_elastic': float(P_elastic_eV),
        'P_inel': float(P_inel_eV),
        'P_diff': float(P_diff_eV),
        'P_flow': float(P_flow_eV),
        'P_eloss': float(P_eloss_eV),
        # Timing
        'wall_time': float(result.wall_time),
    }


if __name__ == '__main__':
    print(f'=== Comprehensive Sim/Exp Analysis ===')
    print(f'V_eff={V_EFF_CM3}cm³, V_reactor={V_REACTOR_CM3}cm³, P={P_W}W, Q={Q_SLM}slm')
    print()

    all_results = []
    t_start = time.time()

    for i, T_K in enumerate(T_GAS_LIST_K):
        print(f'  [{i+1}/{len(T_GAS_LIST_K)}] T={T_K}K ({T_K-273}°C) ...', end='', flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_full_analysis(T_K)
        all_results.append(r)
        print(f' CH4={r["conv_ch4"]:.2f}% CO2={r["conv_co2"]:.2f}% '
              f'Te={r["Te_eV"]:.3f}eV n_e={r["n_e"]:.2e} ({r["wall_time"]:.0f}s)')

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time:.0f}s')

    # ============================================================
    # TABLE 1: Sim vs Exp - CH4 Conversion
    # ============================================================
    print(f'\n{"="*90}')
    print(f'  [1] CH4 CONVERSION: Simulation vs Experiment')
    print(f'{"="*90}')
    print(f'  {"T(°C)":>6} {"T(K)":>6} {"Exp(%)":>8} {"Sim(%)":>8} {"Δ(%p)":>8} {"Sim/Exp":>8}')
    print(f'  {"-"*50}')
    errors = []
    for r in all_results:
        T_K = r['T_target']
        exp = EXP_DATA[T_K]
        sim = r['conv_ch4']
        delta = sim - exp
        ratio = sim / exp if exp > 0 else 0
        errors.append(delta)
        print(f'  {T_K-273:6d} {T_K:6d} {exp:8.2f} {sim:8.2f} {delta:+8.2f} {ratio:8.3f}')
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    mae = np.mean(np.abs(errors))
    print(f'  {"-"*50}')
    print(f'  RMSE = {rmse:.2f}%p, MAE = {mae:.2f}%p')

    # ============================================================
    # TABLE 2: Plasma Parameters
    # ============================================================
    print(f'\n{"="*100}')
    print(f'  [2] PLASMA PARAMETERS AT STEADY STATE')
    print(f'{"="*100}')
    print(f'  {"T(°C)":>6} {"T(K)":>6} {"Te(eV)":>8} {"ε̄(eV)":>8} {"n_e(m⁻³)":>12} '
          f'{"N_gas(m⁻³)":>12} {"n_e/N":>10} {"τ(s)":>8}')
    print(f'  {"-"*80}')
    for r in all_results:
        print(f'  {r["T_target"]-273:6d} {r["T_target"]:6d} {r["Te_eV"]:8.3f} '
              f'{r["eps_mean"]:8.3f} {r["n_e"]:12.3e} {r["N_gas"]:12.3e} '
              f'{r["n_e"]/r["N_gas"]:10.2e} {r["tau"]:8.2f}')

    # ============================================================
    # TABLE 3: CO2 Conversion
    # ============================================================
    print(f'\n{"="*90}')
    print(f'  [3] CO2 CONVERSION')
    print(f'{"="*90}')
    print(f'  {"T(°C)":>6} {"T(K)":>6} {"CO2 conv(%)":>12} {"c0_CO2":>10} {"cf_CO2":>10} {"Δc_CO2":>10}')
    print(f'  {"-"*60}')
    for r in all_results:
        print(f'  {r["T_target"]-273:6d} {r["T_target"]:6d} {r["conv_co2"]:+12.3f} '
              f'{r["c0_co2"]:10.3e} {r["cf_co2"]:10.3e} {r["c0_co2"]-r["cf_co2"]:+10.3e}')

    # ============================================================
    # TABLE 4: CH4 Reaction Pathway - Synergy Analysis
    # ============================================================
    print(f'\n{"="*110}')
    print(f'  [4] CH4 CONSUMPTION BY MECHANISM TYPE (Plasma-Thermal Synergy)')
    print(f'{"="*110}')
    print(f'  {"T(°C)":>6} {"Σcons":>12} {"EI":>12} {"EI%":>6} '
          f'{"Arrhenius":>12} {"Arr%":>6} {"TE":>12} {"TE%":>6} '
          f'{"Σprod":>12} {"S_flow":>12}')
    print(f'  {"-"*105}')
    for r in all_results:
        cons = r['ch4_cons_total']
        ei_pct = r['ch4_cons_ei'] / cons * 100 if cons != 0 else 0
        arr_pct = r['ch4_cons_arr'] / cons * 100 if cons != 0 else 0
        te_pct = r['ch4_cons_te'] / cons * 100 if cons != 0 else 0
        print(f'  {r["T_target"]-273:6d} {cons:+12.4e} '
              f'{r["ch4_cons_ei"]:+12.4e} {ei_pct:5.1f}% '
              f'{r["ch4_cons_arr"]:+12.4e} {arr_pct:5.1f}% '
              f'{r["ch4_cons_te"]:+12.4e} {te_pct:5.1f}% '
              f'{r["ch4_prod_total"]:+12.4e} {r["S_flow_ch4"]:+12.4e}')

    # ============================================================
    # TABLE 5: Top CH4 consumption reactions at each temperature
    # ============================================================
    for r in all_results:
        T_K = r['T_target']
        cons_rxns = [x for x in r['ch4_reactions'] if x['contribution'] < 0]
        cons_rxns.sort(key=lambda x: x['contribution'])
        total_cons = r['ch4_cons_total']

        print(f'\n{"="*120}')
        print(f'  [5-{T_K-273}°C] TOP CH4 CONSUMPTION REACTIONS at T={T_K}K')
        print(f'{"="*120}')
        print(f'  {"R#":>4} {"Type":>10} {"ν":>3} {"Rate(react)":>12} '
              f'{"Contrib":>12} {"%cons":>7}  {"Formula"}')
        print(f'  {"-"*100}')
        for x in cons_rxns[:10]:
            pct = x['contribution'] / total_cons * 100 if total_cons != 0 else 0
            print(f'  R{x["id"]:>3} {x["type"][:10]:>10} {x["nu"]:>3} '
                  f'{x["eff_rate"]:12.4e} {x["contribution"]:+12.4e} {pct:6.1f}%  '
                  f'{x["formula"]}')

    # ============================================================
    # TABLE 6: CO2 Reaction Pathway
    # ============================================================
    print(f'\n{"="*110}')
    print(f'  [6] CO2 REACTION PATHWAYS BY MECHANISM TYPE')
    print(f'{"="*110}')
    print(f'  {"T(°C)":>6} {"Σcons":>12} {"EI":>12} {"EI%":>6} '
          f'{"Arrhenius":>12} {"Arr%":>6} {"TE":>12} {"TE%":>6} '
          f'{"Σprod":>12} {"S_flow":>12}')
    print(f'  {"-"*105}')
    for r in all_results:
        cons = r['co2_cons_total']
        ei_pct = r['co2_cons_ei'] / cons * 100 if cons != 0 else 0
        arr_pct = r['co2_cons_arr'] / cons * 100 if cons != 0 else 0
        te_pct = r['co2_cons_te'] / cons * 100 if cons != 0 else 0
        print(f'  {r["T_target"]-273:6d} {cons:+12.4e} '
              f'{r["co2_cons_ei"]:+12.4e} {ei_pct:5.1f}% '
              f'{r["co2_cons_arr"]:+12.4e} {arr_pct:5.1f}% '
              f'{r["co2_cons_te"]:+12.4e} {te_pct:5.1f}% '
              f'{r["co2_prod_total"]:+12.4e} {r["S_flow_co2"]:+12.4e}')

    # Top CO2 reactions at each temperature
    for r in all_results:
        T_K = r['T_target']
        co2_rxns = r['co2_reactions']
        total_cons = r['co2_cons_total']
        total_prod = r['co2_prod_total']

        print(f'\n{"="*120}')
        print(f'  [7-{T_K-273}°C] CO2 REACTIONS at T={T_K}K (conv={r["conv_co2"]:+.3f}%)')
        print(f'{"="*120}')

        cons_rxns = [x for x in co2_rxns if x['contribution'] < 0]
        prod_rxns = [x for x in co2_rxns if x['contribution'] > 0]

        if cons_rxns:
            print(f'  [CONSUMPTION] (total = {total_cons:+.4e})')
            print(f'  {"R#":>4} {"Type":>10} {"ν":>3} {"Rate(react)":>12} '
                  f'{"Contrib":>12} {"%cons":>7}  {"Formula"}')
            print(f'  {"-"*100}')
            cons_rxns.sort(key=lambda x: x['contribution'])
            for x in cons_rxns[:8]:
                pct = x['contribution'] / total_cons * 100 if total_cons != 0 else 0
                print(f'  R{x["id"]:>3} {x["type"][:10]:>10} {x["nu"]:>3} '
                      f'{x["eff_rate"]:12.4e} {x["contribution"]:+12.4e} {pct:6.1f}%  '
                      f'{x["formula"]}')

        if prod_rxns:
            print(f'  [PRODUCTION] (total = {total_prod:+.4e})')
            print(f'  {"R#":>4} {"Type":>10} {"ν":>3} {"Rate(react)":>12} '
                  f'{"Contrib":>12} {"%prod":>7}  {"Formula"}')
            print(f'  {"-"*100}')
            prod_rxns.sort(key=lambda x: -x['contribution'])
            for x in prod_rxns[:8]:
                pct = x['contribution'] / total_prod * 100 if total_prod != 0 else 0
                print(f'  R{x["id"]:>3} {x["type"][:10]:>10} {x["nu"]:>3} '
                      f'{x["eff_rate"]:12.4e} {x["contribution"]:+12.4e} {pct:6.1f}%  '
                      f'{x["formula"]}')

    # ============================================================
    # TABLE 8: Product Selectivity
    # ============================================================
    print(f'\n{"="*120}')
    print(f'  [8] PRODUCT CONCENTRATIONS & SELECTIVITY')
    print(f'{"="*120}')
    header = f'  {"Species":>10}'
    for r in all_results:
        header += f' {r["T_target"]-273:>8}°C'
    header += f'  {"Unit"}'
    print(header)
    print(f'  {"-"*70}')

    # Mole fractions (ppm)
    for name in PRODUCTS:
        line = f'  {name:>10}'
        for r in all_results:
            if name in r['products']:
                line += f' {r["products"][name]["c_final_ppm"]:>8.1f}'
            else:
                line += f' {"---":>8}'
        line += f'  ppm'
        print(line)

    # C-selectivity based on CH4 consumed
    print(f'\n  C-Selectivity (% of CH4 C consumed → product C)')
    print(f'  {"-"*70}')
    c_products = {'CO': 1, 'C2H2': 2, 'C2H4': 2, 'C2H6': 2, 'C3H6': 3, 'C3H8': 3,
                  'CH3OH': 1, 'CH2O': 1}
    for name, n_c in c_products.items():
        line = f'  {name:>10}'
        for r in all_results:
            if name in r['products']:
                delta_ch4 = r['c0_ch4'] - r['cf_ch4']
                if delta_ch4 > 0:
                    sel = n_c * r['products'][name]['c_final'] / delta_ch4 * 100
                    line += f' {sel:>8.2f}'
                else:
                    line += f' {"---":>8}'
            else:
                line += f' {"---":>8}'
        line += f'  %'
        print(line)

    # ============================================================
    # TABLE 9: Radical Concentrations
    # ============================================================
    print(f'\n{"="*90}')
    print(f'  [9] RADICAL CONCENTRATIONS (steady-state)')
    print(f'{"="*90}')
    header = f'  {"Species":>10}'
    for r in all_results:
        header += f' {r["T_target"]-273:>12}°C'
    print(header)
    print(f'  {"-"*65}')
    for name in RADICALS:
        line = f'  {name:>10}'
        for r in all_results:
            if name in r['radicals']:
                line += f' {r["radicals"][name]["n_final"]:>12.3e}'
            else:
                line += f' {"---":>12}'
        line += f'  m⁻³'
        print(line)

    # ============================================================
    # TABLE 10: Energy Cost
    # ============================================================
    print(f'\n{"="*90}')
    print(f'  [10] ENERGY COST')
    print(f'{"="*90}')
    print(f'  {"T(°C)":>6} {"CH4%":>8} {"EC(eV/mol)":>12} {"EC(kJ/mol)":>12} '
          f'{"SEI(J/L)":>10} {"τ(s)":>8}')
    print(f'  {"-"*65}')
    for r in all_results:
        print(f'  {r["T_target"]-273:6d} {r["conv_ch4"]:8.2f} '
              f'{r["EC_eV"]:12.1f} {r["EC_kJ_mol"]:12.1f} '
              f'{r["SEI_JL"]:10.1f} {r["tau"]:8.2f}')

    # ============================================================
    # TABLE 11: Electron Energy Budget
    # ============================================================
    print(f'\n{"="*110}')
    print(f'  [11] ELECTRON ENERGY BUDGET [eV/(m³·s)]')
    print(f'{"="*110}')
    print(f'  {"T(°C)":>6} {"P_dep":>12} {"P_elastic":>12} {"P_inel":>12} '
          f'{"P_diff":>12} {"P_flow":>12} {"P_eloss":>12}')
    print(f'  {"-"*85}')
    for r in all_results:
        print(f'  {r["T_target"]-273:6d} {r["P_dep"]:+12.4e} '
              f'{r["P_elastic"]:+12.4e} {r["P_inel"]:+12.4e} '
              f'{r["P_diff"]:+12.4e} {r["P_flow"]:+12.4e} {r["P_eloss"]:+12.4e}')

    # As % of P_dep
    print(f'\n  As % of P_dep:')
    print(f'  {"T(°C)":>6} {"P_el%":>8} {"P_inel%":>8} {"P_diff%":>8} {"P_flow%":>8} {"P_eloss%":>8}')
    print(f'  {"-"*50}')
    for r in all_results:
        P = r['P_dep']
        print(f'  {r["T_target"]-273:6d} {r["P_elastic"]/P*100:8.2f} '
              f'{r["P_inel"]/P*100:8.2f} {r["P_diff"]/P*100:8.2f} '
              f'{r["P_flow"]/P*100:8.2f} {r["P_eloss"]/P*100:8.2f}')

    # Save raw data as JSON
    json_data = []
    for r in all_results:
        d = {k: v for k, v in r.items() if k not in ('ch4_reactions', 'co2_reactions')}
        json_data.append(d)
    with open('analysis_results_0312.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'\n  Raw data saved to analysis_results_0312.json')
    print(f'  Done.')
