"""SUNDIALS CVODE 6.x ctypes wrapper for plasma0d_v2.

Minimal wrapper: BDF + dense linear solver + non-negative constraints.
Uses locally-built SUNDIALS at ~/.local/sundials/.
"""
import ctypes as ct
import numpy as np
import os
from dataclasses import dataclass
from typing import Callable, Optional

# ============================================================
#  Load SUNDIALS shared libraries
# ============================================================
_SUNDIALS_DIR = os.path.expanduser("~/.local/sundials/lib")

def _load_lib(name):
    path = os.path.join(_SUNDIALS_DIR, f"lib{name}.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"SUNDIALS library not found: {path}")
    return ct.CDLL(path)

_nvec    = _load_lib("sundials_nvecserial")
_cvode   = _load_lib("sundials_cvode")
_sunmat  = _load_lib("sundials_sunmatrixdense")
_sunls   = _load_lib("sundials_sunlinsoldense")
_core    = _load_lib("sundials_generic")

# ============================================================
#  Type aliases
# ============================================================
realtype = ct.c_double
sunindextype = ct.c_int64  # SUNDIALS default for 64-bit build

# Opaque pointers
SUNContext  = ct.c_void_p
N_Vector    = ct.c_void_p
SUNMatrix   = ct.c_void_p
SUNLinearSolver = ct.c_void_p
CVodeMem    = ct.c_void_p

# CVODE constants
CV_BDF      = 2
CV_NORMAL   = 1
CV_ONE_STEP = 2
CV_SUCCESS  = 0

# RHS callback: int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
CVRhsFn = ct.CFUNCTYPE(ct.c_int, realtype, N_Vector, N_Vector, ct.c_void_p)

# Jacobian callback: int J(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
#                          void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
CVLsJacFn = ct.CFUNCTYPE(ct.c_int, realtype, N_Vector, N_Vector,
                          SUNMatrix, ct.c_void_p, N_Vector, N_Vector, N_Vector)

# ============================================================
#  Function prototypes
# ============================================================

# SUNContext
_core.SUNContext_Create.restype = ct.c_int
_core.SUNContext_Create.argtypes = [ct.c_void_p, ct.POINTER(SUNContext)]

_core.SUNContext_Free.restype = ct.c_int
_core.SUNContext_Free.argtypes = [ct.POINTER(SUNContext)]

# N_Vector Serial
_nvec.N_VNew_Serial.restype = N_Vector
_nvec.N_VNew_Serial.argtypes = [sunindextype, SUNContext]

_nvec.N_VDestroy.restype = None
_nvec.N_VDestroy.argtypes = [N_Vector]

_nvec.N_VGetArrayPointer_Serial.restype = ct.POINTER(realtype)
_nvec.N_VGetArrayPointer_Serial.argtypes = [N_Vector]

# CVODE
_cvode.CVodeCreate.restype = CVodeMem
_cvode.CVodeCreate.argtypes = [ct.c_int, SUNContext]

_cvode.CVodeInit.restype = ct.c_int
_cvode.CVodeInit.argtypes = [CVodeMem, CVRhsFn, realtype, N_Vector]

_cvode.CVodeReInit.restype = ct.c_int
_cvode.CVodeReInit.argtypes = [CVodeMem, realtype, N_Vector]

_cvode.CVodeSStolerances.restype = ct.c_int
_cvode.CVodeSStolerances.argtypes = [CVodeMem, realtype, realtype]

_cvode.CVodeSVtolerances.restype = ct.c_int
_cvode.CVodeSVtolerances.argtypes = [CVodeMem, realtype, N_Vector]

_cvode.CVodeSetJacFn.restype = ct.c_int
_cvode.CVodeSetJacFn.argtypes = [CVodeMem, ct.c_void_p]

_cvode.CVodeSetMaxNumSteps.restype = ct.c_int
_cvode.CVodeSetMaxNumSteps.argtypes = [CVodeMem, ct.c_long]

_cvode.CVodeSetMaxStep.restype = ct.c_int
_cvode.CVodeSetMaxStep.argtypes = [CVodeMem, realtype]

_cvode.CVodeSetInitStep.restype = ct.c_int
_cvode.CVodeSetInitStep.argtypes = [CVodeMem, realtype]

_cvode.CVodeSetConstraints.restype = ct.c_int
_cvode.CVodeSetConstraints.argtypes = [CVodeMem, N_Vector]

_cvode.CVodeSetLinearSolver.restype = ct.c_int
_cvode.CVodeSetLinearSolver.argtypes = [CVodeMem, SUNLinearSolver, SUNMatrix]

_cvode.CVode.restype = ct.c_int
_cvode.CVode.argtypes = [CVodeMem, realtype, N_Vector, ct.POINTER(realtype), ct.c_int]

_cvode.CVodeFree.restype = None
_cvode.CVodeFree.argtypes = [ct.POINTER(CVodeMem)]

_cvode.CVodeGetNumSteps.restype = ct.c_int
_cvode.CVodeGetNumSteps.argtypes = [CVodeMem, ct.POINTER(ct.c_long)]

_cvode.CVodeGetNumRhsEvals.restype = ct.c_int
_cvode.CVodeGetNumRhsEvals.argtypes = [CVodeMem, ct.POINTER(ct.c_long)]

_cvode.CVodeGetNumErrTestFails.restype = ct.c_int
_cvode.CVodeGetNumErrTestFails.argtypes = [CVodeMem, ct.POINTER(ct.c_long)]

# CVodeGetNumConstrFails not available in SUNDIALS 6.x
# _cvode.CVodeGetNumConstrFails.restype = ct.c_int
# _cvode.CVodeGetNumConstrFails.argtypes = [CVodeMem, ct.POINTER(ct.c_long)]

# Dense matrix
_sunmat.SUNDenseMatrix.restype = SUNMatrix
_sunmat.SUNDenseMatrix.argtypes = [sunindextype, sunindextype, SUNContext]

_sunmat.SUNMatDestroy.restype = None
_sunmat.SUNMatDestroy.argtypes = [SUNMatrix]

_sunmat.SUNDenseMatrix_Data.restype = ct.POINTER(realtype)
_sunmat.SUNDenseMatrix_Data.argtypes = [SUNMatrix]

# Dense linear solver
_sunls.SUNLinSol_Dense.restype = SUNLinearSolver
_sunls.SUNLinSol_Dense.argtypes = [N_Vector, SUNMatrix, SUNContext]

_sunls.SUNLinSolFree.restype = ct.c_int
_sunls.SUNLinSolFree.argtypes = [SUNLinearSolver]


# ============================================================
#  Helper: numpy <-> N_Vector
# ============================================================

def _nvec_from_numpy(arr: np.ndarray, sunctx: SUNContext) -> N_Vector:
    """Create N_Vector_Serial and copy data from numpy array."""
    n = len(arr)
    v = _nvec.N_VNew_Serial(sunindextype(n), sunctx)
    if not v:
        raise RuntimeError("N_VNew_Serial failed")
    ptr = _nvec.N_VGetArrayPointer_Serial(v)
    ct.memmove(ptr, arr.ctypes.data, n * 8)
    return v

def _nvec_to_numpy(v: N_Vector, n: int) -> np.ndarray:
    """Copy N_Vector data to numpy array."""
    ptr = _nvec.N_VGetArrayPointer_Serial(v)
    buf = (realtype * n).from_address(ct.addressof(ptr.contents))
    return np.frombuffer(buf, dtype=np.float64, count=n).copy()

def _nvec_set_from_numpy(v: N_Vector, arr: np.ndarray):
    """Overwrite N_Vector data from numpy array (in-place)."""
    n = len(arr)
    ptr = _nvec.N_VGetArrayPointer_Serial(v)
    ct.memmove(ptr, arr.ctypes.data, n * 8)


# ============================================================
#  CVODESolver class
# ============================================================

@dataclass
class CVODEResult:
    t: np.ndarray
    y: np.ndarray        # shape (n_state, n_points)
    n_steps: int = 0
    n_rhs_evals: int = 0
    n_err_fails: int = 0
    n_constr_fails: int = 0
    success: bool = True
    message: str = ""


class CVODESolver:
    """SUNDIALS CVODE 6.x solver with non-negative constraints.

    Usage:
        solver = CVODESolver(n_eq, rhs_func)
        solver.setup(y0, t0, rtol, atol, constraints='non_negative')
        result = solver.solve(t_end, t_eval)
        solver.reinit(t0_new, y0_new)  # for pulse boundaries
        solver.free()
    """

    def __init__(self, n_eq: int, rhs_func: Callable,
                 jac_func: Optional[Callable] = None):
        """
        Args:
            n_eq: number of equations
            rhs_func: Python callable f(t, y) -> dydt (numpy arrays)
            jac_func: Optional Python callable f(t, y) -> J (n_eq x n_eq numpy array)
        """
        self.n_eq = n_eq
        self._rhs_py = rhs_func
        self._jac_py = jac_func
        self._sunctx = SUNContext()
        self._cvode_mem = None
        self._y_vec = None
        self._atol_vec = None
        self._constraints_vec = None
        self._A = None
        self._LS = None
        self._setup_done = False

        # Temporary arrays for callback
        self._y_tmp = np.zeros(n_eq)
        self._ydot_tmp = np.zeros(n_eq)

        # Create context
        ret = _core.SUNContext_Create(None, ct.byref(self._sunctx))
        if ret != 0:
            raise RuntimeError(f"SUNContext_Create failed: {ret}")

        # Build RHS C callback
        @CVRhsFn
        def _c_rhs(t, y_nvec, ydot_nvec, user_data):
            try:
                y_ptr = _nvec.N_VGetArrayPointer_Serial(y_nvec)
                yd_ptr = _nvec.N_VGetArrayPointer_Serial(ydot_nvec)
                ct.memmove(self._y_tmp.ctypes.data, y_ptr, self.n_eq * 8)
                dydt = self._rhs_py(t, self._y_tmp)
                ct.memmove(yd_ptr, dydt.ctypes.data, self.n_eq * 8)
                return 0
            except Exception as e:
                print(f"  CVODE RHS error: {e}")
                return -1

        self._c_rhs = _c_rhs  # prevent GC

        # Build Jacobian C callback
        self._c_jac = None
        if jac_func is not None:
            @CVLsJacFn
            def _c_jac(t, y_nvec, fy_nvec, Jac, user_data, tmp1, tmp2, tmp3):
                try:
                    y_ptr = _nvec.N_VGetArrayPointer_Serial(y_nvec)
                    ct.memmove(self._y_tmp.ctypes.data, y_ptr, self.n_eq * 8)
                    J = self._jac_py(t, self._y_tmp)
                    # SUNDenseMatrix stores column-major (Fortran order)
                    J_col = np.asfortranarray(J, dtype=np.float64)
                    J_ptr = _sunmat.SUNDenseMatrix_Data(Jac)
                    ct.memmove(J_ptr, J_col.ctypes.data, self.n_eq * self.n_eq * 8)
                    return 0
                except Exception as e:
                    print(f"  CVODE Jacobian error: {e}")
                    return -1

            self._c_jac = _c_jac  # prevent GC

    def setup(self, y0: np.ndarray, t0: float = 0.0,
              rtol: float = 1e-6, atol = 1e-10,
              max_step: float = 0.0, init_step: float = 1e-12,
              max_num_steps: int = 500000,
              constraints = 'non_negative'):
        """Initialize CVODE with BDF + dense solver.

        Args:
            atol: float (scalar) or np.ndarray (vector absolute tolerances)
            constraints: 'non_negative' (all y_i >= 0), 'none',
                         or np.ndarray of per-variable constraint types
                         (0.0=none, 1.0=y_i>=0, -1.0=y_i<=0, 2.0=y_i>0)
        """
        n = self.n_eq
        ctx = self._sunctx

        # Create y vector
        self._y_vec = _nvec_from_numpy(y0, ctx)

        # Create CVODE
        self._cvode_mem = _cvode.CVodeCreate(CV_BDF, ctx)
        if not self._cvode_mem:
            raise RuntimeError("CVodeCreate failed")

        # Init
        ret = _cvode.CVodeInit(self._cvode_mem, self._c_rhs, realtype(t0), self._y_vec)
        if ret != 0:
            raise RuntimeError(f"CVodeInit failed: {ret}")

        # Tolerances: scalar or vector
        if isinstance(atol, np.ndarray):
            self._atol_vec = _nvec_from_numpy(atol.astype(np.float64), ctx)
            ret = _cvode.CVodeSVtolerances(self._cvode_mem, realtype(rtol), self._atol_vec)
            if ret != 0:
                raise RuntimeError(f"CVodeSVtolerances failed: {ret}")
        else:
            ret = _cvode.CVodeSStolerances(self._cvode_mem, realtype(rtol), realtype(float(atol)))
            if ret != 0:
                raise RuntimeError(f"CVodeSStolerances failed: {ret}")

        # Max steps
        _cvode.CVodeSetMaxNumSteps(self._cvode_mem, ct.c_long(max_num_steps))

        # Step size
        if max_step > 0:
            _cvode.CVodeSetMaxStep(self._cvode_mem, realtype(max_step))
        if init_step > 0:
            _cvode.CVodeSetInitStep(self._cvode_mem, realtype(init_step))

        # Dense matrix + solver
        self._A = _sunmat.SUNDenseMatrix(sunindextype(n), sunindextype(n), ctx)
        self._LS = _sunls.SUNLinSol_Dense(self._y_vec, self._A, ctx)
        _cvode.CVodeSetLinearSolver(self._cvode_mem, self._LS, self._A)

        # Jacobian callback (must be after SetLinearSolver)
        if self._c_jac is not None:
            ret = _cvode.CVodeSetJacFn(self._cvode_mem,
                                        ct.cast(self._c_jac, ct.c_void_p))
            if ret != 0:
                print(f"  WARNING: CVodeSetJacFn returned {ret}")

        # Constraints
        if isinstance(constraints, np.ndarray):
            self._constraints_vec = _nvec_from_numpy(constraints.astype(np.float64), ctx)
            ret = _cvode.CVodeSetConstraints(self._cvode_mem, self._constraints_vec)
            if ret != 0:
                print(f"  WARNING: CVodeSetConstraints returned {ret}")
        elif constraints == 'non_negative':
            constr_arr = np.ones(n, dtype=np.float64)  # 1.0 = y_i >= 0
            self._constraints_vec = _nvec_from_numpy(constr_arr, ctx)
            ret = _cvode.CVodeSetConstraints(self._cvode_mem, self._constraints_vec)
            if ret != 0:
                print(f"  WARNING: CVodeSetConstraints returned {ret}")

        self._setup_done = True

    def reinit(self, t0: float, y0: np.ndarray):
        """Reinitialize CVODE at new time/state (discards BDF history)."""
        _nvec_set_from_numpy(self._y_vec, y0)
        ret = _cvode.CVodeReInit(self._cvode_mem, realtype(t0), self._y_vec)
        if ret != 0:
            raise RuntimeError(f"CVodeReInit failed: {ret}")

    def step_to(self, tout: float) -> tuple:
        """Advance to tout. Returns (t_reached, y_array)."""
        tret = realtype(0.0)
        ret = _cvode.CVode(self._cvode_mem, realtype(tout),
                           self._y_vec, ct.byref(tret), CV_NORMAL)
        y = _nvec_to_numpy(self._y_vec, self.n_eq)
        return float(tret.value), y, ret

    def solve(self, t_eval: np.ndarray) -> CVODEResult:
        """Solve to each time in t_eval. Returns CVODEResult."""
        n_pts = len(t_eval)
        y_out = np.zeros((self.n_eq, n_pts))

        # First point = current state
        y_out[:, 0] = _nvec_to_numpy(self._y_vec, self.n_eq)

        for i in range(1, n_pts):
            tret = realtype(0.0)
            ret = _cvode.CVode(self._cvode_mem, realtype(t_eval[i]),
                               self._y_vec, ct.byref(tret), CV_NORMAL)
            y_out[:, i] = _nvec_to_numpy(self._y_vec, self.n_eq)
            if ret < 0:
                # Fill remaining with last good value
                for j in range(i + 1, n_pts):
                    y_out[:, j] = y_out[:, i]
                return CVODEResult(
                    t=t_eval, y=y_out,
                    success=False,
                    message=f"CVODE failed at t={float(tret.value):.6e} (ret={ret})",
                    **self._get_stats(),
                )

        return CVODEResult(
            t=t_eval, y=y_out,
            success=True,
            message="CVODE OK",
            **self._get_stats(),
        )

    def solve_clamped(self, t_eval: np.ndarray, n_species: int,
                      floors: Optional[np.ndarray] = None) -> CVODEResult:
        """Solve with per-step clamping: CV_ONE_STEP + clamp + reinit.

        After each internal CVODE step, checks if any species concentration
        is negative. If so, clamps to floor and calls CVodeReInit to cleanly
        reset BDF history (no polynomial corruption).

        Args:
            t_eval: output time points
            n_species: number of chemical species (first n_species elements of y)
            floors: per-species floor values, shape (n_species,). Default: 1e-30 for all.
        """
        if floors is None:
            floors = np.full(n_species, 1e-30)

        n_pts = len(t_eval)
        y_out = np.zeros((self.n_eq, n_pts))
        y_out[:, 0] = _nvec_to_numpy(self._y_vec, self.n_eq)

        t_idx = 1
        n_reinits = 0
        tret = realtype(0.0)

        while t_idx < n_pts:
            # Take one internal step toward next output time
            ret = _cvode.CVode(self._cvode_mem, realtype(t_eval[t_idx]),
                               self._y_vec, ct.byref(tret), CV_ONE_STEP)

            t_now = float(tret.value)

            if ret < 0:
                # Solver failed — fill remaining and return
                y_fail = _nvec_to_numpy(self._y_vec, self.n_eq)
                for j in range(t_idx, n_pts):
                    y_out[:, j] = y_fail
                return CVODEResult(
                    t=t_eval, y=y_out,
                    success=False,
                    message=f"CVODE failed at t={t_now:.6e} (ret={ret}, reinits={n_reinits})",
                    n_constr_fails=n_reinits,
                    **{k: v for k, v in self._get_stats().items() if k != 'n_constr_fails'},
                )

            # Check for negative species and clamp
            y_now = _nvec_to_numpy(self._y_vec, self.n_eq)
            needs_reinit = False
            for j in range(n_species):
                if y_now[j] < floors[j]:
                    y_now[j] = floors[j]
                    needs_reinit = True

            if needs_reinit:
                _nvec_set_from_numpy(self._y_vec, y_now)
                r2 = _cvode.CVodeReInit(self._cvode_mem, realtype(t_now), self._y_vec)
                if r2 != 0:
                    for j in range(t_idx, n_pts):
                        y_out[:, j] = y_now
                    return CVODEResult(
                        t=t_eval, y=y_out,
                        success=False,
                        message=f"CVodeReInit failed at t={t_now:.6e} (ret={r2})",
                        n_constr_fails=n_reinits,
                        **{k: v for k, v in self._get_stats().items() if k != 'n_constr_fails'},
                    )
                n_reinits += 1

            # Record output for any t_eval points we've passed
            while t_idx < n_pts and t_now >= t_eval[t_idx] - 1e-30:
                y_out[:, t_idx] = y_now
                t_idx += 1

        return CVODEResult(
            t=t_eval, y=y_out,
            success=True,
            message=f"CVODE OK (reinits={n_reinits})",
            n_constr_fails=n_reinits,
            **{k: v for k, v in self._get_stats().items() if k != 'n_constr_fails'},
        )

    def _get_stats(self) -> dict:
        nsteps = ct.c_long(0)
        nrhs = ct.c_long(0)
        nerr = ct.c_long(0)
        _cvode.CVodeGetNumSteps(self._cvode_mem, ct.byref(nsteps))
        _cvode.CVodeGetNumRhsEvals(self._cvode_mem, ct.byref(nrhs))
        _cvode.CVodeGetNumErrTestFails(self._cvode_mem, ct.byref(nerr))
        return {
            'n_steps': nsteps.value,
            'n_rhs_evals': nrhs.value,
            'n_err_fails': nerr.value,
            'n_constr_fails': 0,
        }

    def free(self):
        """Release all SUNDIALS resources."""
        if self._LS:
            _sunls.SUNLinSolFree(self._LS)
            self._LS = None
        if self._A:
            _sunmat.SUNMatDestroy(self._A)
            self._A = None
        if self._constraints_vec:
            _nvec.N_VDestroy(self._constraints_vec)
            self._constraints_vec = None
        if self._atol_vec:
            _nvec.N_VDestroy(self._atol_vec)
            self._atol_vec = None
        if self._y_vec:
            _nvec.N_VDestroy(self._y_vec)
            self._y_vec = None
        if self._cvode_mem:
            mem_ptr = ct.pointer(ct.c_void_p(self._cvode_mem))
            _cvode.CVodeFree(mem_ptr)
            self._cvode_mem = None
        if self._sunctx:
            _core.SUNContext_Free(ct.byref(self._sunctx))
            self._sunctx = None

    def __del__(self):
        self.free()
