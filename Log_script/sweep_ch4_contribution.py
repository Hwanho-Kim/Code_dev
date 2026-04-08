"""CH4 전환율 개별 반응 기여도 분석.

각 T_gas에서 steady-state 후, CH4에 관여하는 모든 반응의
개별 기여율(소모/생성)을 정량적으로 출력한다.

V_eff=1.6cm³, V_reactor=100cm³, P=5W, Q=0.4slm, constant mode.
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

# === Fixed conditions ===
V_eff_cm3 = 1.6
V_reactor_cm3 = 100
P_W = 5.0
Q_slm = 0.4

# === Gas temperature sweep ===
T_gas_list_K = [303, 373, 453, 523]  # 30, 100, 180, 250 °C


def run_ch4_analysis(T_target_K):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    V_eff_m3 = V_eff_cm3 * 1e-6
    V_reactor_m3 = V_reactor_cm3 * 1e-6

    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_reactor_m3
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm

    # Clamp gas temperature
    cfg['T_wall'] = T_target_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_target_K

    # Adaptive t_end
    Q_actual = Q_slm * (T_target_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau_est = V_reactor_m3 / Q_actual
    t_end = min(max(3.0, 1.5 * tau_est), 15.0)

    cfg['solver'] = {
        't_end': t_end, 'n_points': 100, 'method': 'BDF',
        'rtol': 1e-5, 'atol': 1e-10, 'max_step': 5e-4, 'constrained': False
    }

    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
    scfg = cfg['solver']
    result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                          rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    # ===== Final state =====
    sm = solver.sm; rxn = solver.rxn; n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    T_gas = y[sm.idx_Tgas]; ne_eps = y[sm.idx_energy]
    c_e = c[0]; n_e = c_e * NA
    T_gs = max(T_gas, 200.0)
    eps_th = 1.5 * KB * T_gs / QE
    eps_mean = np.clip(ne_eps / n_e, eps_th, 100.0) if n_e > 1 else max(1.0, eps_th)
    Te_eV = (2.0 / 3.0) * eps_mean
    c_total = solver.power.P_gas / (R_GAS * T_gas)

    # LUT query
    k_ei_conc = None
    if solver.lut and eps_mean >= solver.lut.eps_range[0]:
        k_ei_conc, Te_eV_lut = solver.lut.get_rate_coefficients_conc(eps_mean)

    rates = rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                        Te_eV=Te_eV, P_gas=solver.power.P_gas)

    # ===== CH4 per-reaction contribution =====
    ch4_idx = sm.index('CH4')
    ch4_stoich_row = rxn.stoich_matrix[ch4_idx, :]  # (n_reactions,)
    f_sp = solver._f_species

    # Identify all reactions with nonzero CH4 stoichiometry
    ch4_reactions = []
    veff_mask = rxn._veff_mask  # (n_reactions,) bool: True = V_eff reaction
    for j in range(rxn.n_reactions):
        nu = ch4_stoich_row[j]
        if nu != 0:
            r = rxn.reactions[j]
            # Determine effective rate in reactor
            # V_eff reactions (EI + TE + ion-Arrhenius): diluted by f_species
            is_veff = veff_mask[j] if veff_mask is not None else \
                      (r.type in ('ELECTRON_IMPACT', 'TE_DEPENDENT'))
            f = f_sp[ch4_idx] if is_veff else 1.0
            raw_rate = rates[j]                       # mol/(m³·s) in discharge
            eff_rate = raw_rate * f                    # mol/(m³·s) in reactor
            contribution = nu * eff_rate               # mol/(m³·s) CH4 change
            ch4_reactions.append({
                'idx': j, 'id': r.id, 'formula': r.formula, 'type': r.type,
                'nu': int(nu), 'raw_rate': raw_rate, 'f': f,
                'eff_rate': eff_rate, 'contribution': contribution,
            })

    # Sort by absolute contribution (biggest first)
    ch4_reactions.sort(key=lambda x: abs(x['contribution']), reverse=True)

    # CH4 inlet/outlet
    c0_ch4 = result.concentrations[ch4_idx, 0]
    cf_ch4 = result.concentrations[ch4_idx, -1]
    conv = (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0

    # Flow source
    S_flow_ch4 = solver.flow.compute_flow_source(c, T_gas)[ch4_idx]

    # Total consumption / production
    total_consumption = sum(r['contribution'] for r in ch4_reactions if r['contribution'] < 0)
    total_production = sum(r['contribution'] for r in ch4_reactions if r['contribution'] > 0)

    return {
        'T_target': T_target_K, 'T_gas_actual': T_gas,
        'Te': Te_eV, 'n_e': n_e, 'eps': eps_mean,
        'ch4_conv': conv,
        'c0_ch4': c0_ch4, 'cf_ch4': cf_ch4,
        'total_consumption': total_consumption,
        'total_production': total_production,
        'S_flow_ch4': S_flow_ch4,
        'reactions': ch4_reactions,
        'wall_time': result.wall_time,
    }


if __name__ == '__main__':
    print(f'CH4 Conversion Contribution Analysis')
    print(f'V_eff={V_eff_cm3}cm³, V_reactor={V_reactor_cm3}cm³, P={P_W}W, Q={Q_slm}slm')
    print(f'T_gas = {T_gas_list_K} K')
    print()

    all_results = []
    t_start = time.time()

    for i, T_K in enumerate(T_gas_list_K):
        print(f'  [{i+1}/{len(T_gas_list_K)}] Running T={T_K}K ...', end='', flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                r = run_ch4_analysis(T_K)
            except Exception as e:
                print(f' ERROR: {e}')
                continue
        all_results.append(r)
        print(f' Done ({r["wall_time"]:.0f}s), CH4={r["ch4_conv"]:+.1f}%', flush=True)

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time/60:.1f} min\n')

    # =================================================================
    # SUMMARY TABLE 1: Overview
    # =================================================================
    print(f'{"="*100}')
    print(f'  CH4 CONVERSION CONTRIBUTION ANALYSIS')
    print(f'{"="*100}')

    print(f'\n[1] OVERVIEW')
    print(f'{"T(K)":>6} {"Te(eV)":>7} {"n_e(m⁻³)":>11} {"ε̄(eV)":>7} '
          f'{"CH4%":>7} {"c0_CH4":>10} {"cf_CH4":>10} '
          f'{"Σcons":>12} {"Σprod":>12} {"S_flow":>12}')
    print('-' * 110)
    for r in all_results:
        print(f'{r["T_target"]:6.0f} {r["Te"]:7.3f} {r["n_e"]:11.2e} {r["eps"]:7.3f} '
              f'{r["ch4_conv"]:+7.1f} {r["c0_ch4"]:10.3e} {r["cf_ch4"]:10.3e} '
              f'{r["total_consumption"]:+12.4e} {r["total_production"]:+12.4e} '
              f'{r["S_flow_ch4"]:+12.4e}')

    # =================================================================
    # SUMMARY TABLE 2: Per-reaction breakdown at each T_gas
    # =================================================================
    for r in all_results:
        T_K = r['T_target']
        rxns = r['reactions']
        total_cons = r['total_consumption']

        print(f'\n{"="*120}')
        print(f'  T_gas = {T_K}K  |  Te = {r["Te"]:.3f} eV  |  n_e = {r["n_e"]:.2e} m⁻³  |  CH4 conv = {r["ch4_conv"]:+.1f}%')
        print(f'{"="*120}')

        # Consumption reactions
        cons_rxns = [x for x in rxns if x['contribution'] < 0]
        prod_rxns = [x for x in rxns if x['contribution'] > 0]

        if cons_rxns:
            print(f'\n  [CONSUMPTION] (total = {total_cons:+.4e} mol/(m³·s))')
            print(f'  {"R#":>4} {"Type":>8} {"ν":>3} {"Rate(disc)":>12} {"f":>6} '
                  f'{"Rate(react)":>12} {"Contrib":>12} {"%cons":>7}  Formula')
            print(f'  {"-"*110}')
            for x in cons_rxns:
                pct = x['contribution'] / total_cons * 100 if total_cons != 0 else 0
                print(f'  R{x["id"]:>3} {x["type"][:8]:>8} {x["nu"]:>3} '
                      f'{x["raw_rate"]:12.4e} {x["f"]:6.4f} '
                      f'{x["eff_rate"]:12.4e} {x["contribution"]:+12.4e} {pct:6.1f}%  '
                      f'{x["formula"]}')

        if prod_rxns:
            total_prod = r['total_production']
            print(f'\n  [PRODUCTION] (total = {total_prod:+.4e} mol/(m³·s))')
            print(f'  {"R#":>4} {"Type":>8} {"ν":>3} {"Rate(disc)":>12} {"f":>6} '
                  f'{"Rate(react)":>12} {"Contrib":>12} {"%prod":>7}  Formula')
            print(f'  {"-"*110}')
            for x in prod_rxns:
                pct = x['contribution'] / total_prod * 100 if total_prod != 0 else 0
                print(f'  R{x["id"]:>3} {x["type"][:8]:>8} {x["nu"]:>3} '
                      f'{x["raw_rate"]:12.4e} {x["f"]:6.4f} '
                      f'{x["eff_rate"]:12.4e} {x["contribution"]:+12.4e} {pct:6.1f}%  '
                      f'{x["formula"]}')

    # =================================================================
    # SUMMARY TABLE 3: Cross-temperature comparison (top reactions)
    # =================================================================
    # Collect all unique reaction IDs that appear
    all_rxn_ids = set()
    for r in all_results:
        for x in r['reactions']:
            if abs(x['contribution']) > 1e-20:
                all_rxn_ids.add(x['id'])

    # Build cross-temp table: rows=reactions, columns=temperatures
    print(f'\n{"="*130}')
    print(f'  CROSS-TEMPERATURE COMPARISON: CH4 contribution [mol/(m³·s)]')
    print(f'{"="*130}')

    # Header
    header = f'  {"R#":>4} {"Type":>8} '
    for r in all_results:
        header += f' {r["T_target"]:>12.0f}K'
    header += f'  Formula'
    print(header)
    print(f'  {"-"*len(header)}')

    # For each reaction, show contribution at each temperature
    # Sort by max abs contribution across temps
    rxn_max = {}
    rxn_info = {}
    for rid in all_rxn_ids:
        max_abs = 0
        for r in all_results:
            for x in r['reactions']:
                if x['id'] == rid:
                    max_abs = max(max_abs, abs(x['contribution']))
                    rxn_info[rid] = (x['type'], x['formula'])
        rxn_max[rid] = max_abs

    sorted_rids = sorted(rxn_max.keys(), key=lambda x: rxn_max[x], reverse=True)

    for rid in sorted_rids:
        if rxn_max[rid] < 1e-20:
            continue
        rtype, formula = rxn_info[rid]
        line = f'  R{rid:>3} {rtype[:8]:>8} '
        for r in all_results:
            found = False
            for x in r['reactions']:
                if x['id'] == rid:
                    line += f' {x["contribution"]:+12.4e}'
                    found = True
                    break
            if not found:
                line += f' {"---":>12}'
        line += f'  {formula}'
        print(line)

    # Bottom: totals
    line_cons = f'  {"":>4} {"Σcons":>8} '
    line_prod = f'  {"":>4} {"Σprod":>8} '
    line_flow = f'  {"":>4} {"S_flow":>8} '
    line_net  = f'  {"":>4} {"NET":>8} '
    for r in all_results:
        line_cons += f' {r["total_consumption"]:+12.4e}'
        line_prod += f' {r["total_production"]:+12.4e}'
        line_flow += f' {r["S_flow_ch4"]:+12.4e}'
        net = r["total_consumption"] + r["total_production"] + r["S_flow_ch4"]
        line_net += f' {net:+12.4e}'
    print(f'  {"-"*120}')
    print(line_cons)
    print(line_prod)
    print(line_flow)
    print(line_net)

    # =================================================================
    # Top-5 consumption % at each temperature
    # =================================================================
    print(f'\n{"="*100}')
    print(f'  TOP-5 CH4 CONSUMPTION REACTIONS (% of total consumption)')
    print(f'{"="*100}')

    header2 = f'  {"Rank":>4} '
    for r in all_results:
        header2 += f' {"---":>4} {r["T_target"]:>4.0f}K {"---":>4} |'
    print(header2)

    for rank in range(5):
        line = f'  #{rank+1:>3} '
        for r in all_results:
            cons_rxns = [x for x in r['reactions'] if x['contribution'] < 0]
            cons_rxns.sort(key=lambda x: x['contribution'])  # most negative first
            if rank < len(cons_rxns):
                x = cons_rxns[rank]
                total_c = r['total_consumption']
                pct = x['contribution'] / total_c * 100 if total_c != 0 else 0
                line += f' R{x["id"]:>3} {pct:5.1f}% |'
            else:
                line += f' {"---":>4} {"---":>6} |'
        print(line)
