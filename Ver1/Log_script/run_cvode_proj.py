"""CVODE ctypes + CVodeSetProjFn for non-negativity enforcement.

Uses SUNDIALS v7.2.1 projection mechanism to clamp species concentrations
and ne_eps to physical floor values. The projection correction is
automatically propagated to the Nordsieck history array by CVODE.

Usage:
    # 2-period validation (compare against scipy baseline)
    python run_cvode_proj.py --test

    # Full PFR steady state
    nohup python run_cvode_proj.py > cvode_proj.log 2>&1 &
"""
import ctypes, numpy as np, sys, os, time, io, contextlib
from ctypes import (c_int, c_long, c_double, c_void_p, POINTER,
                    CFUNCTYPE, byref)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Load SUNDIALS shared libraries
# ---------------------------------------------------------------------------
B = '/home/hawn/sundials_build/build/src'
for lib in ['sundials/libsundials_core', 'nvector/serial/libsundials_nvecserial',
            'cvode/libsundials_cvode', 'sunmatrix/dense/libsundials_sunmatrixdense',
            'sunlinsol/dense/libsundials_sunlinsoldense',
            'sunnonlinsol/newton/libsundials_sunnonlinsolnewton']:
    ctypes.CDLL(f'{B}/{lib}.so', mode=ctypes.RTLD_GLOBAL)
core = ctypes.CDLL(f'{B}/sundials/libsundials_core.so')
nv   = ctypes.CDLL(f'{B}/nvector/serial/libsundials_nvecserial.so')
cv   = ctypes.CDLL(f'{B}/cvode/libsundials_cvode.so')
sd   = ctypes.CDLL(f'{B}/sunmatrix/dense/libsundials_sunmatrixdense.so')
sl   = ctypes.CDLL(f'{B}/sunlinsol/dense/libsundials_sunlinsoldense.so')

# ---------------------------------------------------------------------------
# Setup plasma0d_v2 simulation (suppress stdout during init)
# ---------------------------------------------------------------------------
from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.numba_core import rhs_numba, extract_numba_params
from plasma0d_v2.constants import NA, QE, KB, T_STP, P_STP

cfg = load_config(os.path.join(os.path.dirname(__file__),
                               'plasma0d_v2', 'config.yaml'))
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {'PRF_Hz': 10000.0, 'duty_cycle': 0.20, 'P_peak_W': 8.1,
                'rise_time_s': 2e-6, 'waveform': 'trapezoidal'}
cfg['reactor'] = {'volume': 250e-6, 'pressure': 101325.0}
cfg['flow'] = {'Q_slm': 0.4, 'model': 'PFR'}
cfg['T_wall'] = 303.0
cfg['wall_loss_freq'] = 10000.0
cfg['initial'] = {'T_gas': 303.0, 'ne': 1e8, 'Te_eV': 0.026}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

with contextlib.redirect_stdout(io.StringIO()):
    solver, y0, _, _ = setup_simulation(
        cfg, os.path.join(os.path.dirname(__file__), 'plasma0d_v2'))

# ---------------------------------------------------------------------------
# Extract Numba RHS parameters
# ---------------------------------------------------------------------------
p = extract_numba_params(solver)
nb_args = tuple(p[k] for k in [
    'nsp', 'ie', 'iT', 'lgrid', 'ltab', 'kdead', 'snm', 'egrid',
    'elN', 'inN', 'poN', 'emin', 'A21b', 'A22b', 'kth',
    'nrxn', 'SM', 'vm',
    'eg', 'eb', 'et', 'nei', 'eel',
    'ag2', 'aA', 'an2', 'aE', 'ao', 'aa', 'ab', 'anr', 'narr',
    'tg2', 'ts2', 'tA2', 'tn2', 'tk2', 'ta3', 'tb3', 'tnr3', 'tt2', 'nte',
    'hi', 'dh', 'hvm', 'eli', 'ese',
    'fs', 'cef', 'cf', 'nef',
    'pii', 'nii', 'npi', 'nni',
    'xi', 'Vr', 'Qs', 'Pg', 'pfr',
    'Tw', 'wlf', 'Ma', 'cp', 'Lsq', 'vr'])

pfn = solver.power.get_power_density
NEQ = len(y0)
nsp = p['nsp']
ie  = p['ie']      # ne_eps index
iT  = p['iT']      # T_gas index

# Physical floor values (same as solver._ce_floor etc.)
CEF = float(p['cef'])   # electron concentration floor [mol/m³]
CF  = float(p['cf'])    # general concentration floor [mol/m³]
NEF = float(p['nef'])   # ne_eps floor [eV/m³]

RHS_TYPE = CFUNCTYPE(c_int, c_double, c_void_p, c_void_p, c_void_p)
nv.N_VGetArrayPointer_Serial.restype = POINTER(c_double)

_rhs_count = [0]

def rhs_cb(t, ynv, ydnv, ud):
    _rhs_count[0] += 1
    ya = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ynv), shape=(NEQ,))
    ys = ya.copy()
    for i in range(nsp):
        if ys[i] < CF: ys[i] = CF
    if ys[0] < CEF: ys[0] = CEF
    if ys[ie] < NEF: ys[ie] = NEF
    yda = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ydnv), shape=(NEQ,))
    yda[:] = rhs_numba(t, ys, *nb_args, pfn(t))
    return 0

rhs_c = RHS_TYPE(rhs_cb)

libproj = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libplasma_proj.so'))
libproj.proj_set_params(c_int(nsp), c_int(ie), c_int(iT), c_int(NEQ),
                        c_double(CEF), c_double(CF), c_double(NEF), c_double(200.0))
libproj.proj_nonneg.restype = c_int

# ---------------------------------------------------------------------------
# Parse command line
# ---------------------------------------------------------------------------
test_mode = '--test' in sys.argv

# ---------------------------------------------------------------------------
# Create CVODE solver
# ---------------------------------------------------------------------------
ctx = c_void_p()
core.SUNContext_Create(c_void_p(0), byref(ctx))

nv.N_VNew_Serial.restype = c_void_p
ynv = c_void_p(nv.N_VNew_Serial(c_long(NEQ), ctx))
yp = nv.N_VGetArrayPointer_Serial(ynv)
for i in range(NEQ):
    yp[i] = y0[i]

cv.CVodeCreate.restype = c_void_p
mem = c_void_p(cv.CVodeCreate(c_int(2), ctx))   # CV_BDF = 2
cv.CVodeInit(mem, rhs_c, c_double(0.0), ynv)
cv.CVodeSStolerances(mem, c_double(1e-4), c_double(1e-10))
cv.CVodeSetMaxNumSteps(mem, c_long(50000000))
cv.CVodeSetMaxStep(mem, c_double(1e-6))

# Dense linear solver
sd.SUNDenseMatrix.restype = c_void_p
A = sd.SUNDenseMatrix(c_long(NEQ), c_long(NEQ), ctx)
sl.SUNLinSol_Dense.restype = c_void_p
LS = sl.SUNLinSol_Dense(ynv, c_void_p(A), ctx)
cv.CVodeSetLinearSolver(mem, c_void_p(LS), c_void_p(A))

# ★ Enable projection for non-negativity ★
ret = cv.CVodeSetProjFn(mem, libproj.proj_nonneg)
assert ret == 0, f"CVodeSetProjFn failed: ret={ret}"
cv.CVodeSetProjFrequency(mem, c_long(1))    # project every step
cv.CVodeSetProjErrEst(mem, c_int(1))        # also project error estimate

# NOTE: do NOT use CVodeSetConstraints — bug #702 in v7.2.1 causes
# infinite step sizes. CVodeSetProjFn is the correct mechanism.

# ---------------------------------------------------------------------------
# Test mode: 2 periods (200µs) with detailed output
# ---------------------------------------------------------------------------
if test_mode:
    T_period = 100e-6
    t_end = 2.0 * T_period

    # Output times matching baseline
    check_times_us = [5, 10, 18, 20, 21, 30, 50, 99, 105, 110, 199]
    check_times = [t * 1e-6 for t in check_times_us]

    # Baseline reference values: (t_us, ne, Te, ne_eps)
    baseline = [
        (5.0,   2.87e+14, 2.504, 1.08e+15),
        (10.0,  2.87e+14, 2.504, 1.08e+15),
        (18.0,  2.85e+14, 2.503, 1.07e+15),
        (20.0,  6.68e+10, 0.026, 2.01e+09),
        (21.0,  1.43e+10, 0.026, 3.90e+06),
        (30.0,  3.53e+09, 0.026, 3.90e+06),
        (50.0,  1.16e+09, 0.026, 3.90e+06),
        (99.0,  4.22e+08, 0.026, 3.90e+06),
        (105.0, 2.87e+14, 2.504, 1.08e+15),
        (110.0, 2.87e+14, 2.504, 1.08e+15),
        (199.0, 8.40e+08, 0.026, 3.90e+06),
    ]

    print("=== CVODE + ProjFn: 2-Period Validation Test ===")
    print(f"NEQ={NEQ}, CEF={CEF:.2e}, NEF={NEF:.2e}, CF={CF:.2e}")
    print(f"t_end={t_end*1e6:.0f} µs")
    print()
    print(f"{'t(µs)':>7} {'ne_cvode':>12} {'ne_base':>12} {'ratio':>8} "
          f"{'Te':>7} {'ne_eps':>12} {'neps_base':>12} {'proj#':>6}")
    print("-" * 95)

    t_current = 0.0
    t0_wall = time.time()
    ci = 0  # check index

    for ct, (t_us, ne_b, Te_b, neps_b) in zip(check_times, baseline):
        t_out = c_double(0.0)
        ret = cv.CVode(mem, c_double(ct), ynv, byref(t_out), c_int(1))

        ya = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ynv),
                                   shape=(NEQ,))
        ne_cur = ya[0] * NA
        ne_eps_cur = ya[ie]
        Tg = ya[iT]

        if ne_cur > 1.0:
            eps = max(ne_eps_cur / ne_cur, 1.5 * KB * max(Tg, 200.0) / QE)
        else:
            eps = 1.5 * KB * max(Tg, 200.0) / QE
        Te_cur = (2.0 / 3.0) * eps

        ratio = ne_cur / ne_b if ne_b > 0 else float('inf')
        status = "OK" if 0.1 < ratio < 10 else "MISMATCH"

        _np = c_long(0); cv.CVodeGetNumProjEvals(mem, byref(_np))
        print(f"{t_us:7.1f} {ne_cur:12.2e} {ne_b:12.2e} {ratio:8.3f} "
              f"{Te_cur:7.3f} {ne_eps_cur:12.2e} {neps_b:12.2e} "
              f"{_np.value:6d}  {status}")

        if ret != 0:
            print(f"  *** CVODE returned {ret} at t={t_out.value*1e6:.1f} µs")
            break

    wall = time.time() - t0_wall
    nfe = c_long(0); cv.CVodeGetNumRhsEvals(mem, byref(nfe))
    nst = c_long(0); cv.CVodeGetNumSteps(mem, byref(nst))
    nproj = c_long(0); cv.CVodeGetNumProjEvals(mem, byref(nproj))
    npfail = c_long(0); cv.CVodeGetNumProjFails(mem, byref(npfail))

    print()
    print(f"Wall time: {wall:.2f}s ({wall/2*1e3:.1f} ms/period)")
    print(f"Steps: {nst.value}")
    print(f"RHS evals: {nfe.value} ({nfe.value/2:.0f}/period)")
    print(f"Projections: {nproj.value}, fails: {npfail.value}")

    # Final state
    ya = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ynv),
                               shape=(NEQ,))
    print(f"\nFinal state (t={t_out.value*1e6:.0f} µs):")
    print(f"  ne     = {ya[0]*NA:.2e} m⁻³")
    print(f"  ne_eps = {ya[ie]:.2e} eV/m³")
    print(f"  T_gas  = {ya[iT]:.2f} K")
    print(f"  min(y) = {ya[:nsp].min():.2e} (should be >= {CF:.2e})")

    # Check for any negative species
    neg_count = np.sum(ya[:nsp] < 0)
    neg_below_floor = np.sum(ya[:nsp] < CF * 0.99)
    print(f"  Negative species: {neg_count}")
    print(f"  Below floor: {neg_below_floor}")

    if neg_count > 0:
        for i in range(nsp):
            if ya[i] < 0:
                print(f"    y[{i}] ({solver.sm.names[i]}) = {ya[i]:.2e}")

    sys.exit(0)


# ---------------------------------------------------------------------------
# Full PFR run: t_end = tau (residence time)
# ---------------------------------------------------------------------------
Q = 0.4 * (303 / T_STP) * (P_STP / 101325.0) / 60000.0
tau = 250e-6 / Q
n_total = int(tau * 10000)

print(f"=== CVODE + ProjFn: Pulsed PFR ===")
print(f"tau={tau:.1f}s, {n_total} pulses")
print(f"NEQ={NEQ}, CEF={CEF:.2e}, NEF={NEF:.2e}")
print(f"Starting...", flush=True)

# Integrate in chunks for progress reporting
chunk = 10000
t_current = 0.0
t0 = time.time()

while t_current < tau:
    t_target = min(t_current + chunk * 100e-6, tau)
    t_out = c_double(0.0)
    ret = cv.CVode(mem, c_double(t_target), ynv, byref(t_out), c_int(1))
    t_current = t_out.value

    elapsed = time.time() - t0
    pct = t_current / tau * 100
    ya = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ynv),
                               shape=(NEQ,))
    ne = ya[0] * NA
    ch4_i = solver.sm.index('CH4')
    conv = (y0[ch4_i] - ya[ch4_i]) / y0[ch4_i] * 100
    eta = elapsed / max(pct, 0.01) * (100 - pct) / 3600

    nfe_c = c_long(0); cv.CVodeGetNumRhsEvals(mem, byref(nfe_c))
    pulses_done = int(t_current / 100e-6)
    rhs_per_pulse = nfe_c.value / max(pulses_done, 1)

    _npc = c_long(0); cv.CVodeGetNumProjEvals(mem, byref(_npc))
    print(f"[{pct:5.1f}%] t={t_current:.4f}s, ne={ne:.2e}, conv={conv:.4f}%, "
          f"proj={_npc.value}, rhs/pulse={rhs_per_pulse:.0f}, "
          f"{elapsed:.0f}s elapsed, ETA {eta:.1f}h", flush=True)

    if ret != 0:
        print(f"CVODE failed: ret={ret}", flush=True)
        break

wall = time.time() - t0
ya = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ynv), shape=(NEQ,))
ch4_i = solver.sm.index('CH4')
conv = (y0[ch4_i] - ya[ch4_i]) / y0[ch4_i] * 100
nfe = c_long(0); cv.CVodeGetNumRhsEvals(mem, byref(nfe))
nst = c_long(0); cv.CVodeGetNumSteps(mem, byref(nst))
nproj = c_long(0); cv.CVodeGetNumProjEvals(mem, byref(nproj))
npfail = c_long(0); cv.CVodeGetNumProjFails(mem, byref(npfail))

print(f"\n=== RESULT ===")
print(f"Total: {wall:.0f}s ({wall/3600:.1f}h)")
print(f"CH4 conv: {conv:.4f}%")
print(f"Steps: {nst.value}")
print(f"RHS evals: {nfe.value}")
print(f"Projections: {nproj.value}, fails: {npfail.value}")
print(f"Min species: {ya[:nsp].min():.2e}")
print(f"ne = {ya[0]*NA:.2e}, ne_eps = {ya[ie]:.2e}, T_gas = {ya[iT]:.2f}")

np.savez(os.path.join(os.path.dirname(__file__), 'cvode_proj_result.npz'),
         y_final=ya, y0=y0, wall_time=wall, nfev=nfe.value,
         nsteps=nst.value, nproj=nproj.value, npfails=npfail.value,
         species_names=solver.sm.names)
print("Saved to cvode_proj_result.npz")
