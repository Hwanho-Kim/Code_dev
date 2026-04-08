"""CVODE ctypes pulsed integration to steady state.
Usage: nohup python run_cvode_pulsed.py > cvode_run.log 2>&1 &
"""
import ctypes, numpy as np, sys, os, time, io, contextlib
from ctypes import c_int, c_long, c_double, c_void_p, POINTER, CFUNCTYPE, byref
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

B = '/home/hawn/sundials_build/build/src'
for lib in ['sundials/libsundials_core','nvector/serial/libsundials_nvecserial',
            'cvode/libsundials_cvode','sunmatrix/dense/libsundials_sunmatrixdense',
            'sunlinsol/dense/libsundials_sunlinsoldense',
            'sunnonlinsol/newton/libsundials_sunnonlinsolnewton']:
    ctypes.CDLL(f'{B}/{lib}.so', mode=ctypes.RTLD_GLOBAL)
core=ctypes.CDLL(f'{B}/sundials/libsundials_core.so')
nv=ctypes.CDLL(f'{B}/nvector/serial/libsundials_nvecserial.so')
cv=ctypes.CDLL(f'{B}/cvode/libsundials_cvode.so')
sd=ctypes.CDLL(f'{B}/sunmatrix/dense/libsundials_sunmatrixdense.so')
sl=ctypes.CDLL(f'{B}/sunlinsol/dense/libsundials_sunlinsoldense.so')

from plasma0d_v2.config import load_config, setup_simulation
from plasma0d_v2.numba_core import rhs_numba, extract_numba_params
from plasma0d_v2.constants import NA, T_STP, P_STP

cfg = load_config(os.path.join(os.path.dirname(__file__), 'plasma0d_v2', 'config.yaml'))
cfg['power_mode'] = 'pulsed'
cfg['pulse'] = {'PRF_Hz': 10000.0, 'duty_cycle': 0.20, 'P_peak_W': 8.1,
                'rise_time_s': 2e-6, 'waveform': 'trapezoidal'}
cfg['reactor'] = {'volume': 250e-6, 'pressure': 101325.0}
cfg['flow'] = {'Q_slm': 0.4, 'model': 'PFR'}
cfg['T_wall'] = 303.0; cfg['wall_loss_freq'] = 10000.0
cfg['initial'] = {'T_gas': 303.0, 'ne': 1e8, 'Te_eV': 0.026}
cfg['inlet_composition'] = {'N2': 0.70, 'O2': 0.15, 'CO2': 0.14, 'CH4': 0.01}

with contextlib.redirect_stdout(io.StringIO()):
    solver, y0, _, _ = setup_simulation(cfg, os.path.join(os.path.dirname(__file__), 'plasma0d_v2'))

p = extract_numba_params(solver)
nb_args = tuple(p[k] for k in ['nsp','ie','iT','lgrid','ltab','kdead','snm','egrid',
    'elN','inN','poN','emin','A21b','A22b','kth','nrxn','SM','vm',
    'eg','eb','et','nei','eel','ag2','aA','an2','aE','ao','aa','ab','anr','narr',
    'tg2','ts2','tA2','tn2','tk2','ta3','tb3','tnr3','tt2','nte',
    'hi','dh','hvm','eli','ese','fs','cef','cf','nef','pii','nii','npi','nni',
    'xi','Vr','Qs','Pg','pfr','Tw','wlf','Ma','cp','Lsq','vr'])

pfn = solver.power.get_power_density
N = len(y0); nsp=p['nsp']; ie=p['ie']; cef=p['cef']; nef=p['nef']; cf=p['cf']

RHS_TYPE = CFUNCTYPE(c_int, c_double, c_void_p, c_void_p, c_void_p)
nv.N_VGetArrayPointer_Serial.restype = POINTER(c_double)

def rhs_cb(t, ynv, ydnv, ud):
    ya = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ynv), shape=(N,))
    ys = ya.copy()
    for i in range(nsp):
        if ys[i] < cf: ys[i] = cf
    if ys[0] < cef: ys[0] = cef
    if ys[ie] < nef: ys[ie] = nef
    yda = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ydnv), shape=(N,))
    yda[:] = rhs_numba(t, ys, *nb_args, pfn(t))
    return 0
rhs_c = RHS_TYPE(rhs_cb)

ctx = c_void_p(); core.SUNContext_Create(c_void_p(0), byref(ctx))
nv.N_VNew_Serial.restype = c_void_p
ynv = c_void_p(nv.N_VNew_Serial(c_long(N), ctx))
yp = nv.N_VGetArrayPointer_Serial(ynv)
for i in range(N): yp[i] = y0[i]

cv.CVodeCreate.restype = c_void_p
mem = c_void_p(cv.CVodeCreate(c_int(2), ctx))
cv.CVodeInit(mem, rhs_c, c_double(0.0), ynv)
cv.CVodeSStolerances(mem, c_double(1e-4), c_double(1e-10))
cv.CVodeSetMaxNumSteps(mem, c_long(50000000))
cv.CVodeSetMaxStep(mem, c_double(1e-6))

sd.SUNDenseMatrix.restype = c_void_p
A = sd.SUNDenseMatrix(c_long(N), c_long(N), ctx)
sl.SUNLinSol_Dense.restype = c_void_p
LS = sl.SUNLinSol_Dense(ynv, c_void_p(A), ctx)
cv.CVodeSetLinearSolver(mem, c_void_p(LS), c_void_p(A))

# PFR: t_end = tau
Q = 0.4 * (303/T_STP) * (P_STP/101325.0) / 60000.0
tau = 250e-6 / Q
n_total = int(tau * 10000)

print(f"=== CVODE Pulsed PFR ===")
print(f"tau={tau:.1f}s, {n_total} pulses")
print(f"Starting...", flush=True)

# Integrate in chunks for progress reporting
chunk = 10000  # pulses per report
t_current = 0.0
t0 = time.time()

while t_current < tau:
    t_target = min(t_current + chunk * 100e-6, tau)
    t_out = c_double(0.0)
    ret = cv.CVode(mem, c_double(t_target), ynv, byref(t_out), c_int(1))
    t_current = t_out.value

    elapsed = time.time() - t0
    pct = t_current / tau * 100
    ya = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ynv), shape=(N,))
    ne = ya[0] * NA
    ch4_i = solver.sm.index('CH4')
    conv = (y0[ch4_i] - ya[ch4_i]) / y0[ch4_i] * 100
    eta = elapsed / max(pct, 0.01) * (100 - pct) / 3600

    print(f"[{pct:5.1f}%] t={t_current:.4f}s, ne={ne:.2e}, conv={conv:.4f}%, "
          f"{elapsed:.0f}s elapsed, ETA {eta:.1f}h", flush=True)

    if ret != 0:
        print(f"CVODE failed: ret={ret}", flush=True)
        break

wall = time.time() - t0
ya = np.ctypeslib.as_array(nv.N_VGetArrayPointer_Serial(ynv), shape=(N,))
ch4_i = solver.sm.index('CH4')
conv = (y0[ch4_i] - ya[ch4_i]) / y0[ch4_i] * 100
nfe = c_long(0); cv.CVodeGetNumRhsEvals(mem, byref(nfe))

print(f"\n=== RESULT ===")
print(f"Total: {wall:.0f}s ({wall/3600:.1f}h)")
print(f"CH4 conv: {conv:.2f}%")
print(f"nfev: {nfe.value}")

np.savez(os.path.join(os.path.dirname(__file__), 'cvode_pulsed_result.npz'),
         y_final=ya, y0=y0, wall_time=wall, nfev=nfe.value,
         species_names=solver.sm.names)
print("Saved to cvode_pulsed_result.npz")
