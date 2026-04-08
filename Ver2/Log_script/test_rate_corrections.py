"""Test effect of rate coefficient corrections on CH4 conversion.

Corrections tested:
  A) Reaction 90: OH + OH + M -> H2O2 + M
     Current: A=289132, n=-0.76, E=0  (44-120x too low vs JPL)
     JPL k0: A=7.506e7, n=-1.0, E=0 (low-pressure limit)
     Troe 1atm approx: A=3.33e5, n=-0.21, E=0

  B) Reaction 48: HO2 + HO2 + M -> H2O2 + O2 + M
     Current order=3 only (bimolecular k1 term missing)
     Fix: double A to approximate k1+k2 at 1 atm
"""
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import R_GAS, T_STP, P_STP

EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
P_W = 5.0
Q_SLM = 0.4
V_EFF_CM3 = 4.9
INLET = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}


def patch_arrhenius(rxn_set, rxn_id, A=None, n=None, E=None, A_scale=None):
    """Modify Arrhenius parameters for a given reaction id."""
    for arr_idx, r in enumerate(rxn_set.arrhenius_reactions):
        if r.id == rxn_id:
            if A is not None:
                rxn_set._arr_A[arr_idx] = A
            if A_scale is not None:
                rxn_set._arr_A[arr_idx] *= A_scale
            if n is not None:
                rxn_set._arr_n[arr_idx] = n
            if E is not None:
                rxn_set._arr_E[arr_idx] = E
            return True
    return False


def run_pfr(T_K, corrections=None):
    """Run PFR(1τ) with V_eff=4.9cm³, optional rate corrections."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))

    cfg['V_eff'] = V_EFF_CM3 * 1e-6
    cfg['reactor']['volume'] = V_EFF_CM3 * 1e-6
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_SLM
    cfg['flow']['model'] = 'PFR'
    cfg['T_wall'] = T_K
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_K
    cfg['inlet_composition'] = INLET

    Q_actual = Q_SLM * (T_K / T_STP) * (P_STP / 101325.0) / 60000.0
    tau = V_EFF_CM3 * 1e-6 / Q_actual

    cfg['solver'] = {
        't_end': tau, 'n_points': 150, 'method': 'BDF',
        'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1, 'constrained': False
    }

    with contextlib.redirect_stdout(io.StringIO()):
        solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)

        # Apply corrections
        if corrections:
            for rxn_id, params in corrections.items():
                ok = patch_arrhenius(solver.rxn, rxn_id,
                                     A=params.get('A'),
                                     n=params.get('n'),
                                     E=params.get('E'),
                                     A_scale=params.get('A_scale'))
                if not ok:
                    print(f"  WARNING: reaction id={rxn_id} not found!")

            # Must rebuild numba after modifying rate arrays
            solver._setup_numba()

        scfg = cfg_out['solver']
        result = solver.solve(y0, t_span, n_points=scfg['n_points'],
                              method=scfg['method'], rtol=scfg['rtol'],
                              atol=scfg['atol'], max_step=scfg['max_step'])

    sm = solver.sm
    ch4_idx = sm.index('CH4')
    c0 = result.concentrations[ch4_idx, 0]
    cf = result.concentrations[ch4_idx, -1]
    conv = (c0 - cf) / c0 * 100

    # Also get OH, HO2 final concentrations for diagnostics
    oh_idx = sm.index('OH')
    ho2_idx = sm.index('HO2')
    h2o2_idx = sm.index('H2O2')
    oh_f = result.concentrations[oh_idx, -1]
    ho2_f = result.concentrations[ho2_idx, -1]
    h2o2_f = result.concentrations[h2o2_idx, -1]

    return conv, oh_f, ho2_f, h2o2_f


# ── Scenarios ──
scenarios = {
    'Baseline': None,
    'Fix90_k0': {90: {'A': 7.506e7, 'n': -1.0, 'E': 0.0}},        # JPL low-P limit
    'Fix90_Troe': {90: {'A': 3.33e5, 'n': -0.21, 'E': 0.0}},      # Troe approx 1atm
    'Fix48_2x': {48: {'A_scale': 2.0}},                             # HO2+HO2 bimol term
    'Fix90+48': {90: {'A': 3.33e5, 'n': -0.21, 'E': 0.0},          # Both
                 48: {'A_scale': 2.0}},
}

print(f'Rate Coefficient Correction Test')
print(f'V_eff={V_EFF_CM3}cm³, P={P_W}W, Q={Q_SLM}slm, PFR(1τ)')
print()

# Header
temps = sorted(EXP_DATA.keys())
print(f'{"Scenario":<14}', end='')
for T in temps:
    print(f'  {T-273:3d}°C', end='')
print(f'  {"RMSE":>6}')
print('-' * 60)

# Experiment
print(f'{"Experiment":<14}', end='')
exp_vals = [EXP_DATA[T] for T in temps]
for v in exp_vals:
    print(f'  {v:5.2f}', end='')
print()

# Run scenarios
results = {}
for name, corr in scenarios.items():
    convs = []
    oh_finals = []
    for T_K in temps:
        conv, oh_f, ho2_f, h2o2_f = run_pfr(T_K, corrections=corr)
        convs.append(conv)
        oh_finals.append(oh_f)

    rmse = np.sqrt(np.mean([(c - e)**2 for c, e in zip(convs, exp_vals)]))
    results[name] = convs

    print(f'{name:<14}', end='')
    for c in convs:
        print(f'  {c:5.2f}', end='')
    print(f'  {rmse:6.2f}')

# Delta from baseline
print()
print('Delta from baseline:')
baseline = results['Baseline']
for name, convs in results.items():
    if name == 'Baseline':
        continue
    print(f'  {name:<14}', end='')
    for c, b in zip(convs, baseline):
        print(f'  {c-b:+5.2f}', end='')
    print()
