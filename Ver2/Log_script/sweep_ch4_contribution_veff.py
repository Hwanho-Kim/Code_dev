"""CH4 conversion rate contribution analysis (V_eff model, 2026-03-26).

Current code state: V_eff=4.9cm3, FlowModel(V_reactor=V_eff), no dilution.
P=5W constant, Q=0.4slm, T_wall clamp=10000, PFR 1*tau.
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

# === Current conditions ===
V_eff_cm3 = 4.9       # sDBD 70mm x 70mm x 1mm
P_W = 5.0
Q_slm = 0.4

T_gas_list_K = [303, 373, 453, 523]
EXP_VALUES = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}


def run_ch4_analysis(T_target_K):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plasma0d_v2')
    base_dir = os.path.abspath(base_dir)
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    V_eff_m3 = V_eff_cm3 * 1e-6

    cfg['V_eff'] = V_eff_m3
    cfg['reactor']['volume'] = V_eff_m3  # not used by FlowModel (uses V_eff)
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm
    cfg['flow']['model'] = 'PFR'

    # T_gas clamp
    cfg['T_wall'] = T_target_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_target_K

    # PFR 1*tau (tau = V_eff / Q_actual)
    Q_actual = Q_slm * (T_target_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau = V_eff_m3 / Q_actual
    t_end = tau  # 1 * tau

    cfg['solver'] = {
        't_end': t_end, 'n_points': 200, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1,
        'constrained': False,
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

    # ===== CH4 per-reaction contribution (no dilution) =====
    ch4_idx = sm.index('CH4')
    ch4_stoich_row = rxn.stoich_matrix[ch4_idx, :]

    ch4_reactions = []
    for j in range(rxn.n_reactions):
        nu = ch4_stoich_row[j]
        if nu != 0:
            r = rxn.reactions[j]
            raw_rate = rates[j]
            contribution = nu * raw_rate
            ch4_reactions.append({
                'idx': j, 'id': r.id, 'formula': r.formula, 'type': r.type,
                'nu': int(nu), 'raw_rate': raw_rate,
                'contribution': contribution,
            })

    ch4_reactions.sort(key=lambda x: abs(x['contribution']), reverse=True)

    # CH4 inlet/outlet
    c0_ch4 = result.concentrations[ch4_idx, 0]
    cf_ch4 = result.concentrations[ch4_idx, -1]
    conv = (c0_ch4 - cf_ch4) / c0_ch4 * 100 if c0_ch4 > 0 else 0

    # Flow source
    S_flow_ch4 = solver.flow.compute_flow_source(c, T_gas)[ch4_idx]

    total_consumption = sum(r['contribution'] for r in ch4_reactions if r['contribution'] < 0)
    total_production = sum(r['contribution'] for r in ch4_reactions if r['contribution'] > 0)

    return {
        'T_target': T_target_K, 'T_gas_actual': T_gas,
        'Te': Te_eV, 'n_e': n_e, 'eps': eps_mean,
        'ch4_conv': conv, 'tau_ms': tau * 1e3,
        'c0_ch4': c0_ch4, 'cf_ch4': cf_ch4,
        'total_consumption': total_consumption,
        'total_production': total_production,
        'S_flow_ch4': S_flow_ch4,
        'reactions': ch4_reactions,
        'wall_time': result.wall_time,
    }


if __name__ == '__main__':
    print(f'CH4 Conversion Contribution Analysis (V_eff model)')
    print(f'V_eff={V_eff_cm3}cm3, P={P_W}W, Q={Q_slm}slm, PFR 1*tau')
    print(f'No dilution: FlowModel(V_reactor=V_eff)')
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
                import traceback
                print(f' ERROR: {e}')
                traceback.print_exc()
                continue
        all_results.append(r)
        print(f' Done ({r["wall_time"]:.1f}s), CH4={r["ch4_conv"]:.2f}%, tau={r["tau_ms"]:.1f}ms', flush=True)

    total_time = time.time() - t_start
    print(f'\nTotal: {total_time:.1f}s\n')

    if not all_results:
        print("No results. Exiting.")
        sys.exit(1)

    # =================================================================
    # TABLE 1: Overview
    # =================================================================
    print(f'{"="*110}')
    print(f'  CH4 CONVERSION CONTRIBUTION ANALYSIS (V_eff={V_eff_cm3}cm3)')
    print(f'{"="*110}')

    print(f'\n[1] OVERVIEW')
    print(f'{"T(K)":>6} {"tau(ms)":>8} {"Te(eV)":>7} {"n_e(m-3)":>11} {"eps(eV)":>7} '
          f'{"CH4%":>7} {"Exp%":>6} {"Diff":>6} '
          f'{"Scons":>12} {"Sprod":>12} {"S_flow":>12}')
    print('-' * 110)
    for r in all_results:
        exp = EXP_VALUES.get(r['T_target'], 0)
        diff = r['ch4_conv'] - exp
        print(f'{r["T_target"]:6.0f} {r["tau_ms"]:8.1f} {r["Te"]:7.3f} {r["n_e"]:11.2e} {r["eps"]:7.3f} '
              f'{r["ch4_conv"]:+7.2f} {exp:6.2f} {diff:+6.2f} '
              f'{r["total_consumption"]:+12.4e} {r["total_production"]:+12.4e} '
              f'{r["S_flow_ch4"]:+12.4e}')

    # =================================================================
    # TABLE 2: Per-reaction breakdown at each T_gas
    # =================================================================
    for r in all_results:
        T_K = r['T_target']
        rxns = r['reactions']
        total_cons = r['total_consumption']

        print(f'\n{"="*120}')
        print(f'  T_gas = {T_K}K  |  Te = {r["Te"]:.3f} eV  |  n_e = {r["n_e"]:.2e} m-3  |  CH4 conv = {r["ch4_conv"]:.2f}%  (exp={EXP_VALUES.get(T_K,0):.2f}%)')
        print(f'{"="*120}')

        cons_rxns = [x for x in rxns if x['contribution'] < 0]
        prod_rxns = [x for x in rxns if x['contribution'] > 0]

        if cons_rxns:
            print(f'\n  [CONSUMPTION] (total = {total_cons:+.4e} mol/(m3*s))')
            print(f'  {"R#":>4} {"Type":>16} {"v":>3} {"Rate":>12} '
                  f'{"Contrib":>12} {"%cons":>7}  Formula')
            print(f'  {"-"*100}')
            for x in cons_rxns:
                pct = x['contribution'] / total_cons * 100 if total_cons != 0 else 0
                print(f'  R{x["id"]:>3} {x["type"][:16]:>16} {x["nu"]:>3} '
                      f'{x["raw_rate"]:12.4e} '
                      f'{x["contribution"]:+12.4e} {pct:6.1f}%  '
                      f'{x["formula"]}')

        if prod_rxns:
            total_prod = r['total_production']
            print(f'\n  [PRODUCTION] (total = {total_prod:+.4e} mol/(m3*s))')
            print(f'  {"R#":>4} {"Type":>16} {"v":>3} {"Rate":>12} '
                  f'{"Contrib":>12} {"%prod":>7}  Formula')
            print(f'  {"-"*100}')
            for x in prod_rxns:
                pct = x['contribution'] / total_prod * 100 if total_prod != 0 else 0
                print(f'  R{x["id"]:>3} {x["type"][:16]:>16} {x["nu"]:>3} '
                      f'{x["raw_rate"]:12.4e} '
                      f'{x["contribution"]:+12.4e} {pct:6.1f}%  '
                      f'{x["formula"]}')

    # =================================================================
    # TABLE 3: Cross-temperature comparison (top reactions)
    # =================================================================
    all_rxn_ids = set()
    for r in all_results:
        for x in r['reactions']:
            if abs(x['contribution']) > 1e-20:
                all_rxn_ids.add(x['id'])

    print(f'\n{"="*140}')
    print(f'  CROSS-TEMPERATURE COMPARISON: CH4 contribution [mol/(m3*s)]')
    print(f'{"="*140}')

    header = f'  {"R#":>4} {"Type":>16} '
    for r in all_results:
        header += f' {r["T_target"]:>12.0f}K'
    header += f'  Formula'
    print(header)
    print(f'  {"-"*135}')

    rxn_max = {}; rxn_info = {}
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
        line = f'  R{rid:>3} {rtype[:16]:>16} '
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

    # Bottom totals
    print(f'  {"-"*135}')
    for label, key in [('Scons', 'total_consumption'), ('Sprod', 'total_production'), ('S_flow', 'S_flow_ch4')]:
        line = f'  {"":>4} {label:>16} '
        for r in all_results:
            line += f' {r[key]:+12.4e}'
        print(line)
    line_net = f'  {"":>4} {"NET":>16} '
    for r in all_results:
        net = r['total_consumption'] + r['total_production'] + r['S_flow_ch4']
        line_net += f' {net:+12.4e}'
    print(line_net)

    # =================================================================
    # TABLE 4: Top-5 consumption % at each temperature
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
            cons_rxns = sorted([x for x in r['reactions'] if x['contribution'] < 0],
                               key=lambda x: x['contribution'])
            if rank < len(cons_rxns):
                x = cons_rxns[rank]
                total_c = r['total_consumption']
                pct = x['contribution'] / total_c * 100 if total_c != 0 else 0
                line += f' R{x["id"]:>3} {pct:5.1f}% |'
            else:
                line += f' {"---":>4} {"---":>6} |'
        print(line)
