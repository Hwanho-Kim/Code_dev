"""Test effect of missing reactions on CH4 conversion.

Missing reactions identified (priority order):
  1. O + OH -> O2 + H          (CRITICAL, k~3.3e-11, gas-kinetic)
  2. N + NO -> N2 + O           (HIGH, k~2.5e-11)
  3. N + OH -> NO + H           (HIGH, k~2.9e-11)
  4. NO2 + O -> NO + O2         (MOD-HIGH, k~9.7e-12)
  5. N + O2 -> NO + O           (MOD, slow at 300K)
  6. O + O + M -> O2 + M        (LOW)
"""
import sys, os, io, contextlib, copy, tempfile, yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import R_GAS, T_STP, P_STP

EXP_DATA = {303: 5.26, 373: 8.05, 453: 14.36, 523: 20.02}
P_W = 5.0
Q_SLM = 0.4
V_EFF_CM3 = 4.9
INLET = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

# New reactions to add (Arrhenius format: k = A * T^n * exp(-E/(R*T)))
# Units: A in m³/(mol·s) for order=2, (m³/mol)²/s for order=3
NEW_REACTIONS = {
    'O+OH': {
        'id': 300, 'type': 'ARRHENIUS',
        'formula': 'O + OH -> O2 + H',
        'reactants': [{'coeff': 1, 'species': 'O'},
                      {'coeff': 1, 'species': 'OH'}],
        'products': [{'coeff': 1, 'species': 'O2'},
                     {'coeff': 1, 'species': 'H'}],
        'A': 1.084e7, 'n': 0.0, 'E': -1496.5,
        'order': 2, 'ref': 'JPL 15-10, k=1.8e-11*exp(180/T)',
    },
    'N+NO': {
        'id': 301, 'type': 'ARRHENIUS',
        'formula': 'N + NO -> N2 + O',
        'reactants': [{'coeff': 1, 'species': 'N'},
                      {'coeff': 1, 'species': 'NO'}],
        'products': [{'coeff': 1, 'species': 'N2'},
                     {'coeff': 1, 'species': 'O'}],
        'A': 2.7e7, 'n': 0.0, 'E': 1485.3,
        'order': 2, 'ref': 'GRI-Mech 3.0',
    },
    'N+OH': {
        'id': 302, 'type': 'ARRHENIUS',
        'formula': 'N + OH -> NO + H',
        'reactants': [{'coeff': 1, 'species': 'N'},
                      {'coeff': 1, 'species': 'OH'}],
        'products': [{'coeff': 1, 'species': 'NO'},
                     {'coeff': 1, 'species': 'H'}],
        'A': 3.36e7, 'n': 0.0, 'E': 1610.8,
        'order': 2, 'ref': 'GRI-Mech 3.0',
    },
    'NO2+O': {
        'id': 303, 'type': 'ARRHENIUS',
        'formula': 'NO2 + O -> NO + O2',
        'reactants': [{'coeff': 1, 'species': 'NO2'},
                      {'coeff': 1, 'species': 'O'}],
        'products': [{'coeff': 1, 'species': 'NO'},
                     {'coeff': 1, 'species': 'O2'}],
        'A': 3.9e6, 'n': 0.0, 'E': -1004.2,
        'order': 2, 'ref': 'GRI-Mech 3.0',
    },
    'N+O2': {
        'id': 304, 'type': 'ARRHENIUS',
        'formula': 'N + O2 -> NO + O',
        'reactants': [{'coeff': 1, 'species': 'N'},
                      {'coeff': 1, 'species': 'O2'}],
        'products': [{'coeff': 1, 'species': 'NO'},
                     {'coeff': 1, 'species': 'O'}],
        'A': 9.0e3, 'n': 1.0, 'E': 27196.0,
        'order': 2, 'ref': 'GRI-Mech 3.0, Zeldovich 1',
    },
    'O+O+M': {
        'id': 305, 'type': 'ARRHENIUS',
        'formula': 'O + O -> O2',
        'reactants': [{'coeff': 1, 'species': 'O'},
                      {'coeff': 1, 'species': 'O'}],
        'products': [{'coeff': 1, 'species': 'O2'}],
        'A': 1.2e5, 'n': -1.0, 'E': 0.0,
        'order': 3, 'ref': 'GRI-Mech 3.0',
    },
}

# Scenario definitions: which reactions to add
SCENARIOS = {
    'Baseline':     [],
    '+O+OH':        ['O+OH'],
    '+N_chem':      ['N+NO', 'N+OH', 'N+O2'],
    '+NOx':         ['NO2+O'],
    '+All_pri':     ['O+OH', 'N+NO', 'N+OH', 'NO2+O', 'N+O2'],
    '+All+OOM':     ['O+OH', 'N+NO', 'N+OH', 'NO2+O', 'N+O2', 'O+O+M'],
}


def run_pfr(T_K, extra_rxns=None):
    """Run PFR(1τ) with V_eff=4.9cm³, optionally adding reactions."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'plasma0d_v2')
    input_dir = os.path.join(base_dir, 'input')

    # Load original reactions yaml
    rxn_file = os.path.join(input_dir, 'reactions.yaml')
    with open(rxn_file) as f:
        rxn_data = yaml.safe_load(f)

    # Add extra reactions if specified
    if extra_rxns:
        for key in extra_rxns:
            rxn_data['reactions'].append(NEW_REACTIONS[key])

    # Write to temp file
    tmp_rxn = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                          dir=input_dir, delete=False)
    yaml.dump(rxn_data, tmp_rxn, default_flow_style=False)
    tmp_rxn.close()
    tmp_name = os.path.basename(tmp_rxn.name)

    try:
        cfg = load_config(os.path.join(base_dir, 'config.yaml'))
        cfg['reactions_file'] = tmp_name
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
            'rtol': 1e-6, 'atol': 1e-12, 'max_step': 0.1,
            'constrained': False
        }

        with contextlib.redirect_stdout(io.StringIO()):
            solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)
            scfg = cfg_out['solver']
            result = solver.solve(y0, t_span, n_points=scfg['n_points'],
                                  method=scfg['method'], rtol=scfg['rtol'],
                                  atol=scfg['atol'], max_step=scfg['max_step'])

        sm = solver.sm
        ch4_idx = sm.index('CH4')
        c0 = result.concentrations[ch4_idx, 0]
        cf = result.concentrations[ch4_idx, -1]
        conv = (c0 - cf) / c0 * 100

        # Radical diagnostics
        diag = {}
        for sp_name in ['O', 'OH', 'H', 'HO2', 'N', 'NO', 'NO2']:
            try:
                idx = sm.index(sp_name)
                diag[sp_name] = result.concentrations[idx, -1]
            except (ValueError, KeyError):
                diag[sp_name] = 0.0

        return conv, diag
    finally:
        os.unlink(tmp_rxn.name)


print(f'Missing Reactions Test')
print(f'V_eff={V_EFF_CM3}cm³, P={P_W}W, Q={Q_SLM}slm, PFR(1τ)')
print()

# Header
temps = sorted(EXP_DATA.keys())
print(f'{"Scenario":<14}', end='')
for T in temps:
    print(f'  {T-273:3d}°C', end='')
print(f'  {"RMSE":>6}')
print('-' * 62)

# Experiment
print(f'{"Experiment":<14}', end='')
exp_vals = [EXP_DATA[T] for T in temps]
for v in exp_vals:
    print(f'  {v:5.2f}', end='')
print()

# Run scenarios
results = {}
all_diags = {}
for name, rxn_keys in SCENARIOS.items():
    convs = []
    diags_by_T = {}
    for T_K in temps:
        conv, diag = run_pfr(T_K, extra_rxns=rxn_keys if rxn_keys else None)
        convs.append(conv)
        diags_by_T[T_K] = diag

    rmse = np.sqrt(np.mean([(c - e)**2 for c, e in zip(convs, exp_vals)]))
    results[name] = convs
    all_diags[name] = diags_by_T

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

# Radical concentration comparison at 303K
print()
print('Radical concentrations at 303K (mol/m³):')
print(f'{"Scenario":<14}  {"[O]":>10}  {"[OH]":>10}  {"[H]":>10}  {"[HO2]":>10}  {"[N]":>10}  {"[NO]":>10}')
print('-' * 88)
for name in SCENARIOS:
    d = all_diags[name][303]
    print(f'{name:<14}  {d["O"]:10.3e}  {d["OH"]:10.3e}  {d["H"]:10.3e}  '
          f'{d["HO2"]:10.3e}  {d["N"]:10.3e}  {d["NO"]:10.3e}')
