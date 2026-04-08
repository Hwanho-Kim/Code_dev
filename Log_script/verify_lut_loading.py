"""Verify that 300K vs 523K LUT actually loads different data and produces different results."""
import sys, os, io, contextlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.constants import QE, NA, KB, R_GAS, T_STP, P_STP

V_eff_cm3 = 1.6; V_reactor_cm3 = 100; P_W = 5.0; Q_slm = 0.4
T_target = 523.0

for lut_label, bolsig_file, eedf_file in [
    ("300K", "BOLSIG_parameter/Condition1_300K.txt", "BOLSIG_EEDF/EEDF_300K.dat"),
    ("523K", "BOLSIG_parameter/Condition1_523K.txt", "BOLSIG_EEDF/EEDF_523K.dat"),
]:
    print(f"\n{'='*80}")
    print(f"  T_gas={T_target}K with LUT={lut_label}")
    print(f"  bolsig_file: {bolsig_file}")
    print(f"  eedf_file:   {eedf_file}")
    print(f"{'='*80}")

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plasma0d_v2')
    cfg = load_config(os.path.join(base_dir, 'config.yaml'))
    cfg['V_eff'] = V_eff_cm3 * 1e-6
    cfg['reactor']['volume'] = V_reactor_cm3 * 1e-6
    cfg['power_mode'] = 'constant'
    cfg['P_input_W'] = P_W
    cfg['flow']['Q_slm'] = Q_slm
    cfg['T_wall'] = T_target
    cfg['wall_loss_freq'] = 10000.0
    cfg['initial']['T_gas'] = T_target
    cfg['bolsig_files'] = [bolsig_file]
    cfg['eedf_files'] = [eedf_file]

    Q_actual = Q_slm * (T_target / T_STP) * (P_STP / 101325.0) / 60000.0
    tau_est = (V_reactor_cm3 * 1e-6) / Q_actual
    t_end = min(max(3.0, 1.5 * tau_est), 15.0)
    cfg['solver'] = {
        't_end': t_end, 'n_points': 100, 'method': 'BDF',
        'rtol': 1e-5, 'atol': 1e-10, 'max_step': 5e-4, 'constrained': False
    }

    # Don't suppress output for LUT building
    solver, y0, t_span, cfg_out = setup_simulation(cfg, base_dir)

    # Print LUT info
    lut = solver.lut
    print(f"\n  LUT tgas_K = {lut.tgas_K}")
    print(f"  LUT eps_range = {lut.eps_range}")
    print(f"  LUT EEDF used = {lut._eedf_used}")

    # Query a few rate coefficients at eps=1.5 eV
    eps_test = 1.5
    k_conc, Te = lut.get_rate_coefficients_conc(eps_test)
    print(f"\n  k_conc at eps={eps_test} eV (first 5):")
    for i in range(min(5, len(k_conc))):
        print(f"    {lut.cross_sections[i].name}: {k_conc[i]:.10e}")

    # Run simulation
    scfg = cfg_out['solver']
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(y0, t_span, n_points=scfg['n_points'], method=scfg['method'],
                              rtol=scfg['rtol'], atol=scfg['atol'], max_step=scfg['max_step'])

    sm = solver.sm; n_sp = sm.n_species
    y = result.y[:, -1]
    c = np.maximum(y[:n_sp], 1e-30)
    ne_eps = y[sm.idx_energy]; T_gas = y[sm.idx_Tgas]
    c_e = c[0]; n_e = c_e * NA
    eps_mean = np.clip(ne_eps / n_e, 0.01, 100.0)
    Te_eV = (2.0/3.0) * eps_mean

    ch4 = sm.index('CH4')
    c0_ch4 = result.concentrations[ch4, 0]; cf_ch4 = result.concentrations[ch4, -1]
    ch4_conv = (c0_ch4 - cf_ch4) / c0_ch4 * 100

    print(f"\n  === RESULTS (full precision) ===")
    print(f"  n_e     = {n_e:.15e}")
    print(f"  c_e     = {c_e:.15e}")
    print(f"  ne_eps  = {ne_eps:.15e}")
    print(f"  eps     = {eps_mean:.15e}")
    print(f"  Te      = {Te_eV:.15e}")
    print(f"  T_gas   = {T_gas:.15e}")
    print(f"  CH4%    = {ch4_conv:.10f}")
