"""Configuration loading and simulation setup.

Setup flow:
  1. Load species + reactions
  2. Parse BOLSIG+ file(s) -> BolsigData
  3. Build MeanEnergyLUT (ε̄-indexed) from BolsigData + cross sections
  4. Load V-I curve -> PowerSource(V_eff)
  5. Wire ElectronKinetics(Lambda), GasThermal, FlowModel, Solver
"""

import yaml
import os
from .species import SpeciesManager
from .reactions import ReactionSet
from .boltzmann import MeanEnergyLUT
from .bolsig_parser import parse_bolsig_file, parse_eedf_file
from .power import PowerSource
from .electron_kinetics import ElectronKinetics
from .flow import FlowModel
from .gas_thermal import GasThermal
from .solver import PlasmaODESolver


def load_config(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def setup_simulation(cfg: dict, base_dir: str = '.'):
    """Set up complete simulation from configuration.

    Returns:
        (solver, y0, t_span, cfg)
    """
    print("=" * 60)
    print("  Setting up 0D Plasma Simulation (ε̄-indexed LUT)")
    print("=" * 60)

    input_dir = os.path.join(base_dir, cfg.get('input_dir', 'input'))

    # --- [1] Species ---
    print("\n[1] Loading species...")
    sm = SpeciesManager()
    sm.load_from_yaml(os.path.join(input_dir, cfg['species_file']))
    sm.finalize()

    # --- [2] Reactions ---
    print("\n[2] Loading reactions...")
    rxn = ReactionSet()
    rxn.load_from_yaml(os.path.join(input_dir, cfg['reactions_file']))
    rxn.build(sm)

    # --- [3] BOLSIG+ parse & MeanEnergyLUT ---
    print("\n[3] Building ε̄-indexed Boltzmann LUT...")

    bolsig_files = cfg.get('bolsig_files', [])
    if not bolsig_files:
        raise ValueError("No bolsig_files specified in config.yaml")

    # Use the first BOLSIG+ file for the LUT
    bolsig_path = os.path.join(input_dir, bolsig_files[0])
    print(f"  Parsing BOLSIG+ file: {bolsig_path}")
    bolsig_data = parse_bolsig_file(bolsig_path)
    print(f"  {bolsig_data.summary()}")

    lut = MeanEnergyLUT()
    xsec_dir = os.path.join(input_dir, cfg.get('cross_section_dir', 'cross_sections'))
    lut.load_cross_sections(xsec_dir, rxn.ei_reactions)

    eedf_data = None
    eedf_files = cfg.get('eedf_files', [])
    if eedf_files:
        eedf_path = os.path.join(input_dir, eedf_files[0])
        if os.path.exists(eedf_path):
            print(f"  Parsing EEDF file: {eedf_path}")
            eedf_data = parse_eedf_file(eedf_path)
            print(f"  {eedf_data.summary()}")
        else:
            print(f"  WARNING: EEDF file not found: {eedf_path}")
            print(f"  Falling back to Maxwellian EEDF for rate coefficients")

    lut.build(bolsig_data, eedf_data=eedf_data)

    # --- [4] Power source ---
    print("\n[4] Loading power source...")
    reactor = cfg['reactor']
    V_eff = cfg['V_eff']

    ps = PowerSource(
        V_eff=V_eff,
        P_gas=reactor['pressure'],
    )

    power_mode = cfg.get('power_mode', 'vi_envelope')
    if power_mode == 'constant':
        P_input = cfg.get('P_input_W', 0.0)
        ps.set_constant_power(P_input)
    elif power_mode == 'pulsed':
        pulse_cfg = cfg.get('pulse', {})
        ps.set_pulsed_power(
            P_peak_W=pulse_cfg.get('P_peak_W', 5.0),
            PRF_Hz=pulse_cfg.get('PRF_Hz', 1000.0),
            duty_cycle=pulse_cfg.get('duty_cycle', 0.5),
            rise_time_s=pulse_cfg.get('rise_time_s', 0.0),
            waveform=pulse_cfg.get('waveform', 'trapezoidal'),
        )
    else:
        vi_file = os.path.join(input_dir, cfg['vi_curve_file'])
        vi_kwargs = cfg.get('vi_curve_options', {})
        freq = cfg.get('frequency_Hz', None)
        ps.load_vi_curve(vi_file, frequency_Hz=freq, **vi_kwargs)

        if power_mode == 'vi_envelope':
            P_input = cfg.get('P_input_W', ps.P_avg_W)
            ps.set_vi_envelope(P_input)

    print(f"  V_eff = {V_eff:.2e} m³")
    print(f"  P_gas = {reactor['pressure']:.0f} Pa")

    # --- [5] Electron kinetics ---
    print("\n[5] Configuring electron kinetics...")
    Lambda = cfg['Lambda']
    ekin = ElectronKinetics(sm, Lambda=Lambda)
    print(f"  Lambda = {Lambda:.2e} m (diffusion length)")

    # --- [6] Flow model ---
    print("\n[6] Setting up flow model...")
    flow_model_type = cfg.get('flow', {}).get('model', 'CSTR')
    flow = FlowModel(
        V_reactor=V_eff,
        Q_slm=cfg['flow']['Q_slm'],
        P_gas=reactor['pressure'],
        T_gas_init=cfg['initial']['T_gas'],
        flow_model=flow_model_type,
    )
    inlet = cfg['inlet_composition']
    flow.configure(sm, inlet)

    # --- [7] Gas thermal ---
    print("\n[7] Setting up gas thermal model...")
    gth = GasThermal(
        P_gas=reactor['pressure'],
        T_wall=cfg.get('T_wall', 300.0),
    )
    gth.configure(
        M_avg=cfg.get('M_avg', 0.028),
        cp_avg=cfg.get('cp_avg', 1000.0),
        wall_loss_freq=cfg.get('wall_loss_freq', 100.0),
    )

    # --- [8] Solver ---
    print("\n[8] Building solver...")
    qn_mode = cfg.get('qn_mode', 'A')
    energy_source = cfg.get('energy_source', 'constant')
    V_reactor = reactor['volume']
    solver = PlasmaODESolver(sm, rxn, lut, ps, ekin, flow, gth,
                             qn_mode=qn_mode, V_eff=V_eff, V_reactor=V_reactor,
                             energy_source=energy_source)
    energy_labels = {
        'constant': '(external P_dep)',
        'A20': '(P_dep = n_e·N·A20(ε̄), unconstrained)',
        'A20_power_balance': '(ε̄ from power balance, P_dep=P_input/V_eff)',
    }
    print(f"  Energy source: {energy_source} {energy_labels.get(energy_source, '')}")

    T_gas_init = cfg['initial']['T_gas']
    y0 = solver.build_initial_state(
        sm, inlet,
        P_gas=reactor['pressure'],
        T_gas_init=T_gas_init,
        ne_init=cfg['initial'].get('ne', 1e8),
        Te_init_eV=cfg['initial'].get('Te_eV', 1.0),
    )

    # --- Merge solver params: user config + V-I recommended ---
    recommended = ps.get_recommended_solver_params()
    solver_cfg = cfg.get('solver', {})

    if 't_end' not in solver_cfg:
        solver_cfg['t_end'] = recommended['t_end']
    if 'max_step' not in solver_cfg and 'max_step' in recommended:
        solver_cfg['max_step'] = recommended['max_step']
    if 'n_points' not in solver_cfg:
        solver_cfg['n_points'] = recommended['n_points']
    solver_cfg.setdefault('method', 'BDF')
    solver_cfg.setdefault('rtol', 1e-6)
    solver_cfg.setdefault('atol', 1e-12)

    cfg['solver'] = solver_cfg
    t_span = (0.0, solver_cfg['t_end'])

    print(f"\n  Solver params (auto + user override):")
    print(f"    t_end    = {solver_cfg['t_end']*1e6:.1f} us")
    if 'max_step' in solver_cfg:
        print(f"    max_step = {solver_cfg['max_step']*1e9:.0f} ns")
    print(f"    n_points = {solver_cfg['n_points']}")

    eps_min, eps_max = lut.eps_range
    EN_min, EN_max = lut.EN_range
    print("\n" + "=" * 60)
    print(f"  Setup complete: {sm.n_species} species, {rxn.n_reactions} reactions")
    print(f"  LUT: ε̄=[{eps_min:.4f}, {eps_max:.2f}] eV, "
          f"E/N=[{EN_min:.4f}, {EN_max:.1f}] Td")
    if ps.mode == 'constant':
        print(f"  Power: constant, P = {ps.P_avg_W:.2f} W, "
              f"P_dep = {ps._P_constant_Wm3:.3e} W/m³")
    elif ps.mode == 'pulsed':
        print(f"  Power: pulsed ({ps._pulse_waveform}), "
              f"PRF={ps.frequency:.0f} Hz, dc={ps._pulse_duty_cycle*100:.1f}%, "
              f"P_avg = {ps.P_avg_W:.3f} W")
    elif ps.mode == 'vi_envelope':
        print(f"  V-I envelope: {ps.vi_curve.pulse_info.waveform_type}, "
              f"f={ps.frequency:.0f} Hz, P_input = {ps.P_avg_W:.3f} W")
    else:
        print(f"  V-I signed: {ps.vi_curve.pulse_info.waveform_type}, "
              f"f={ps.frequency:.0f} Hz, P_avg = {ps.P_avg_W:.3f} W")
    print(f"  sDBD: V_eff={V_eff:.2e} m³, V_reactor={V_reactor:.2e} m³")
    print(f"  Λ={Lambda:.2e} m")
    print("=" * 60)

    solver._setup_numba()

    return solver, y0, t_span, cfg
