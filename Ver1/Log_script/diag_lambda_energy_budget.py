"""Λ energy budget diagnostic at steady state.

Run Λ=1mm and Λ=10mm to steady state, then decompose the energy equation
to identify which term drives the Te/n_e difference.

Energy equation: d(ne_eps)/dt = P_dep - P_elastic - P_inelastic - P_diff - P_flow - P_e_loss
"""
import sys, os, io, contextlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, ME, R_GAS, T_STP, P_STP

# === Fixed conditions ===
T_GAS_K = 303.0
V_EFF_CM3 = 1.6
V_REACTOR_CM3 = 250.0
P_W = 5.0
Q_SLM = 0.4
T_END = 120.0

LAMBDA_CASES = {
    '1mm':  1.0e-3,
    '10mm': 1.0e-2,
}


def run_and_diagnose(lambda_m, label):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    cfg['V_eff'] = V_EFF_CM3 * 1e-6
    cfg['reactor']['volume'] = V_REACTOR_CM3 * 1e-6
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_SLM
    cfg['Lambda'] = lambda_m
    cfg['T_wall'] = T_GAS_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_GAS_K

    cfg['solver'] = {
        't_end': T_END, 'n_points': 200, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
    }

    solver, y0, t_span, cfg = setup_simulation(cfg, base_dir)
    scfg = cfg['solver']

    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                              rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    # --- Extract final state ---
    sm = solver.sm
    n_sp = sm.n_species
    y_final = result.y[:, -1].copy()

    c = np.maximum(y_final[:n_sp], 1e-30)
    ne_eps = y_final[sm.idx_energy]
    T_gas = y_final[sm.idx_Tgas]

    c_e = c[0]
    n_e = min(c_e * NA, 1e26)
    T_gas_safe = max(T_gas, 200.0)
    eps_thermal = 1.5 * KB * T_gas_safe / QE
    eps_mean = np.clip(ne_eps / n_e, eps_thermal, 100.0) if n_e > 1 else max(1.0, eps_thermal)
    Te_eV = (2.0 / 3.0) * eps_mean

    N_gas = 101325.0 / (KB * T_gas_safe)
    c_total = 101325.0 / (R_GAS * T_gas_safe)

    # --- 1. P_dep ---
    P_dep_Wm3 = solver.power.get_power_density(T_END)
    P_dep = P_dep_Wm3 / QE  # eV/(m³·s)

    # --- 2. P_elastic ---
    transport = solver.lut.get_transport(eps_mean)
    A21 = transport.elastic_power_N
    P_elastic = n_e * N_gas * A21  # eV/(m³·s)

    # --- 3. P_inelastic ---
    k_ei_conc, Te_lut = solver.lut.get_rate_coefficients_conc(eps_mean)
    rates = solver.rxn.compute_reaction_rates(c, T_gas, c_total, k_ei_conc,
                                               Te_eV=Te_eV, P_gas=101325.0)
    P_inel_Wm3 = solver.rxn.compute_electron_energy_loss(rates)
    P_inel = P_inel_Wm3 / QE  # eV/(m³·s)

    # --- 4. P_diff ---
    mu_i_N = 2.8e22
    mu_i = mu_i_N / N_gas
    D_a = mu_i * Te_eV
    diff_freq = D_a / (lambda_m ** 2)
    P_diff = ne_eps * diff_freq  # eV/(m³·s)

    # --- 5. P_flow ---
    tau = solver.flow.get_residence_time(T_gas)
    P_flow = ne_eps / tau if tau > 0 else 0.0  # eV/(m³·s)

    # --- 6. P_e_loss (DR/AT) ---
    S_e_loss = solver.rxn.compute_electron_loss_rate(rates)  # mol/(m³·s)
    P_e_loss = eps_mean * S_e_loss * NA  # eV/(m³·s)

    # --- Species equation: electron loss channels ---
    # Diffusion loss for electron: c_e * diff_freq [mol/(m³·s)]
    diff_loss_e = c_e * diff_freq  # mol/(m³·s)
    diff_loss_per_s = diff_freq  # 1/s (per-electron rate)

    # Collisional loss: S_e_loss [mol/(m³·s)], per-electron = S_e_loss / c_e
    coll_loss_per_s = (S_e_loss / c_e) if c_e > 1e-30 else 0  # 1/s

    # Flow loss: c_e / tau
    flow_loss_per_s = 1.0 / tau if tau > 0 else 0  # 1/s

    # Total loss
    total_loss_per_s = diff_loss_per_s + coll_loss_per_s + flow_loss_per_s

    # --- Compute individual electron loss reaction rates ---
    eloss_details = []
    e_idx = 0  # electron is species 0
    stoich = solver.rxn.stoich_matrix
    for j in solver.rxn._electron_loss_indices:
        rxn = solver.rxn.reactions[j]
        r = rates[j]
        rps = (r / c_e) if c_e > 1e-30 else 0
        eloss_details.append((rxn.formula, r, rps))

    # --- Net ionization ---
    # Sum over reactions with net electron creation (stoich_matrix[0,j] > 0)
    net_ion_rate = 0.0
    ion_details = []
    for j, rxn in enumerate(solver.rxn.reactions):
        nu_e = stoich[e_idx, j]
        if nu_e > 0.5:  # net electron creation (ionization)
            contrib = nu_e * rates[j]
            net_ion_rate += contrib
            ion_details.append((rxn.formula, rates[j], nu_e))

    ion_per_s = (net_ion_rate / c_e) if c_e > 1e-30 else 0

    return {
        'label': label,
        'lambda_m': lambda_m,
        'Te_eV': Te_eV,
        'eps_mean': eps_mean,
        'n_e': n_e,
        'ne_eps': ne_eps,
        'T_gas': T_gas,
        'tau': tau,
        'N_gas': N_gas,
        'D_a': D_a,
        'diff_freq': diff_freq,
        # Energy budget [eV/(m³·s)]
        'P_dep': P_dep,
        'P_elastic': P_elastic,
        'P_inel': P_inel,
        'P_diff': P_diff,
        'P_flow': P_flow,
        'P_e_loss': P_e_loss,
        # Species budget [1/s per electron]
        'diff_loss_ps': diff_loss_per_s,
        'coll_loss_ps': coll_loss_per_s,
        'flow_loss_ps': flow_loss_per_s,
        'total_loss_ps': total_loss_per_s,
        'ion_ps': ion_per_s,
        # Details
        'eloss_details': eloss_details,
    }


if __name__ == '__main__':
    print(f'=== Λ Energy Budget Diagnostic ===')
    print(f'T_gas={T_GAS_K}K, V_eff={V_EFF_CM3}cm³, V_reactor={V_REACTOR_CM3}cm³')
    print(f'P={P_W}W, Q={Q_SLM}slm, t_end={T_END}s')

    all_data = {}
    for label, lam in LAMBDA_CASES.items():
        print(f'\n--- Running {label} ---')
        with contextlib.redirect_stdout(io.StringIO()):
            data = run_and_diagnose(lam, label)
        all_data[label] = data
        print(f'  Done: Te={data["Te_eV"]:.4f}eV, n_e={data["n_e"]:.3e}')

    # === PLASMA STATE ===
    print(f'\n{"="*80}')
    print(f'  PLASMA STATE AT STEADY STATE (t={T_END}s)')
    print(f'{"="*80}')
    print(f'  {"Param":>15} {"1mm":>15} {"10mm":>15} {"Δ(rel%)":>10}')
    print(f'  {"-"*60}')
    d1, d10 = all_data['1mm'], all_data['10mm']
    for key, name, fmt in [
        ('Te_eV', 'Te [eV]', '.4f'),
        ('eps_mean', 'ε̄ [eV]', '.4f'),
        ('n_e', 'n_e [m⁻³]', '.3e'),
        ('ne_eps', 'ne·ε̄ [eV/m³]', '.3e'),
        ('T_gas', 'T_gas [K]', '.1f'),
        ('D_a', 'D_a [m²/s]', '.4e'),
        ('diff_freq', 'D_a/Λ² [1/s]', '.1f'),
    ]:
        v1, v10 = d1[key], d10[key]
        rel = (v10 - v1) / abs(v1) * 100 if abs(v1) > 1e-30 else 0
        print(f'  {name:>15} {v1:>15{fmt}} {v10:>15{fmt}} {rel:>+10.2f}%')

    # === ENERGY BUDGET ===
    print(f'\n{"="*80}')
    print(f'  ELECTRON ENERGY BUDGET [eV/(m³·s)]')
    print(f'  d(ne_eps)/dt = P_dep - P_elastic - P_inel - P_diff - P_flow - P_e_loss')
    print(f'{"="*80}')
    print(f'  {"Term":>15} {"1mm":>15} {"10mm":>15} {"Δ(rel%)":>10} {"frac_1mm%":>10}')
    print(f'  {"-"*65}')

    for key, name, sign in [
        ('P_dep', 'P_dep (+)', +1),
        ('P_elastic', 'P_elastic (-)', -1),
        ('P_inel', 'P_inel (-)', -1),
        ('P_diff', 'P_diff (-)', -1),
        ('P_flow', 'P_flow (-)', -1),
        ('P_e_loss', 'P_e_loss (-)', -1),
    ]:
        v1, v10 = d1[key], d10[key]
        rel = (v10 - v1) / abs(v1) * 100 if abs(v1) > 1e-30 else 0
        frac = abs(v1) / abs(d1['P_dep']) * 100 if abs(d1['P_dep']) > 0 else 0
        print(f'  {name:>15} {v1:>15.4e} {v10:>15.4e} {rel:>+10.2f}% {frac:>10.4f}%')

    residual_1 = d1['P_dep'] - d1['P_elastic'] - d1['P_inel'] - d1['P_diff'] - d1['P_flow'] - d1['P_e_loss']
    residual_10 = d10['P_dep'] - d10['P_elastic'] - d10['P_inel'] - d10['P_diff'] - d10['P_flow'] - d10['P_e_loss']
    print(f'  {"Residual":>15} {residual_1:>15.4e} {residual_10:>15.4e}')

    # === SPECIES BUDGET ===
    print(f'\n{"="*80}')
    print(f'  ELECTRON SPECIES BUDGET [1/s per electron]')
    print(f'  dc_e/dt = ionization - attachment - D_a/Λ² - 1/τ')
    print(f'{"="*80}')
    print(f'  {"Channel":>20} {"1mm [1/s]":>15} {"10mm [1/s]":>15} {"frac_1mm%":>10}')
    print(f'  {"-"*65}')

    for key, name in [
        ('ion_ps', 'Ionization (+)'),
        ('coll_loss_ps', 'Collisional (-)'),
        ('diff_loss_ps', 'Diffusion (-)'),
        ('flow_loss_ps', 'Flow (-)'),
    ]:
        v1, v10 = d1[key], d10[key]
        frac = abs(v1) / d1['total_loss_ps'] * 100 if d1['total_loss_ps'] > 0 else 0
        print(f'  {name:>20} {v1:>15.3e} {v10:>15.3e} {frac:>10.4f}%')

    net_1 = d1['ion_ps'] - d1['coll_loss_ps'] - d1['diff_loss_ps'] - d1['flow_loss_ps']
    net_10 = d10['ion_ps'] - d10['coll_loss_ps'] - d10['diff_loss_ps'] - d10['flow_loss_ps']
    print(f'  {"Net":>20} {net_1:>15.3e} {net_10:>15.3e}')

    # === ELECTRON LOSS DETAILS ===
    print(f'\n{"="*80}')
    print(f'  ELECTRON LOSS REACTION DETAILS')
    print(f'{"="*80}')
    for label in ['1mm', '10mm']:
        d = all_data[label]
        print(f'\n  [{label}]:')
        total = sum(r for _, r, _ in d['eloss_details'])
        for name, r, rps in sorted(d['eloss_details'], key=lambda x: -x[1]):
            frac = r / total * 100 if total > 0 else 0
            print(f'    {name:50s} {r:.3e} mol/(m³·s)  {rps:.2e} 1/s  {frac:5.1f}%')

    print(f'\n  Done.')
