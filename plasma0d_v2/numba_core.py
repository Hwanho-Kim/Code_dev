"""Numba @njit compiled RHS for plasma0d_v2."""
import numpy as np
import numba as nb

_NA = 6.02214076e23
_QE = 1.602176634e-19
_KB = 1.380649e-23
_R = 8.314462618
_T0 = 273.15
_P0 = 101325.0


@nb.njit
def _interp_log(log_x, grid, table_row):
    n = len(grid)
    idx = np.searchsorted(grid, log_x) - 1
    if idx < 0: idx = 0
    if idx >= n-1: idx = n-2
    d = grid[idx+1] - grid[idx]
    f = (log_x - grid[idx]) / d if abs(d) > 1e-30 else 0.0
    lo = table_row[idx]; hi = table_row[idx+1]
    if lo > 0 and hi > 0:
        return 10.0**(np.log10(lo) + f*(np.log10(hi)-np.log10(lo)))
    elif lo < 0 and hi < 0:
        return -(10.0**(np.log10(-lo) + f*(np.log10(-hi)-np.log10(-lo))))
    return lo + f*(hi - lo)


@nb.njit
def lut_rates(log_eps, grid, table, dead):
    nc = table.shape[1]; ng = len(grid)
    idx = np.searchsorted(grid, log_eps) - 1
    if idx < 0: idx = 0
    if idx >= ng-1: idx = ng-2
    d = grid[idx+1] - grid[idx]
    f = (log_eps - grid[idx]) / d if abs(d) > 1e-30 else 0.0
    k = np.empty(nc)
    for j in range(nc):
        if dead[j]: k[j] = 0.0
        else: k[j] = 10.0**(table[idx,j] + f*(table[idx+1,j]-table[idx,j]))
    return k


@nb.njit
def calc_rates(c, Tg, ctot, kei, has_ei, Te, Pg, nrxn, eidx,
               eg, eb, et, nei, ag, aA, an, aE, ao, aa, ab, anr, narr,
               tg, ts, tA, tn, tk, ta2, tb2, tnr2, tt, nte):
    r = np.zeros(nrxn)
    if has_ei:
        ce = c[eidx]
        for i in range(nei):
            r[eg[i]] = kei[eb[i]] * ce * c[et[i]]
    for i in range(narr):
        j = ag[i]; k = aA[i] * Tg**an[i]
        if aE[i] != 0.0: k *= np.exp(-aE[i] / (_R * Tg))
        o = ao[i]; nr = anr[i]
        if o == 1: r[j] = k * c[aa[i]]
        elif o == 2:
            if nr >= 2: r[j] = k * c[aa[i]] * c[ab[i]]
            else: r[j] = k * c[aa[i]] * ctot
        elif o == 3:
            if nr >= 2: r[j] = k * c[aa[i]] * c[ab[i]] * ctot
            else: r[j] = k * c[aa[i]] * ctot * ctot
    if nte > 0:
        TK = max(Te * 11604.0, 300.0); Ts = max(Tg, 200.0)
        ce = c[eidx]
        for i in range(nte):
            j = tg[i]
            if ts[i] == 1:
                M = Pg / (_KB * Ts) * 1e-6
                k3 = tk[i] * (300.0/TK)**2 * np.exp(-70.0/Ts) * np.exp(1500.0*(TK-Ts)/(TK*Ts))
                r[j] = k3 * M * 1e-6 * _NA * ce * c[tt[i]]
            else:
                kc = tA[i] * (300.0/TK)**tn[i] if tn[i] != 0.0 else tA[i]
                kS = kc * 1e-6 * _NA
                if tnr2[i] >= 2: r[j] = kS * c[ta2[i]] * c[tb2[i]]
                elif tnr2[i] == 1: r[j] = kS * c[ta2[i]]
    return r


@nb.njit
def src_terms(rates, S, nsp):
    nrxn = len(rates); St = np.zeros(nsp)
    for i in range(nsp):
        for j in range(nrxn):
            s = S[i,j]
            if s != 0.0:
                St[i] += rates[j] * s
    return St


@nb.njit
def gas_heat(rates, hi, dh):
    Qt = 0.0
    for i in range(len(hi)):
        Qt -= dh[hi[i]] * rates[hi[i]]
    return Qt


@nb.njit
def e_eloss(rates, eg, el, nei):
    P = 0.0
    for i in range(nei): P += el[i] * rates[eg[i]] * _NA * _QE
    return P


@nb.njit
def e_lrate(rates, eli, ese):
    S = 0.0
    for i in range(len(eli)): S += ese[i] * rates[eli[i]]
    return S


@nb.njit
def diff_rate(Tg, Te, Lsq):
    N = _P0 / (_KB * max(Tg, 200.0))
    return (2.8e22 / N) * Te / Lsq


@nb.njit
def energy_rhs(neps, ce, Tg, Pdep, Pel, Pinel, tau, Peloss, Lsq):
    ne = min(ce * _NA, 1e26); Pd = Pdep / _QE
    Pe = min(Pel, 1e30); Pi = min(Pinel / _QE, 1e30)
    Tk = max(Tg, 200.0); eth = 1.5*_KB*Tk/_QE
    neps_floor = max(ne, 1.0) * eth  # thermal floor for ne_eps
    if ne > 1.0:
        em = neps/ne
        if em < eth: em = eth
        if em > 100.0: em = 100.0
    else: em = max(1.0, eth)
    Te = (2.0/3.0)*em; N = _P0/(_KB*Tk)
    Da = (2.8e22/N)*Te; Pdiff = neps*Da/Lsq
    Pf = neps/tau if tau > 0 else 0.0; Pl = min(Peloss, 1e30)
    v = Pd - Pe - Pi - Pdiff - Pf - Pl
    # Structural non-negativity: if ne_eps is at/below thermal floor,
    # only allow positive (heating) derivatives. This prevents the solver
    # from driving ne_eps below zero regardless of step size.
    if neps <= neps_floor * 1.01 and v < 0:
        v = 0.0
    if v > 1e30: v = 1e30
    if v < -1e30: v = -1e30
    return v


@nb.njit
def tgas_rhs(Tg, Qel, tau, Qrxn, Qel2, Pg, Tw, wlf, Ma, cp):
    rho = Pg*Ma/(_R*Tg); rc = rho*cp
    if rc < 1e-10: return 0.0
    Qw = rc*wlf*(Tg-Tw); Qf = rc*(Tg-300.0)/tau if 0 < tau < 1e9 else 0.0
    return (min(Qel,1e15)+min(Qrxn,1e15)+min(Qel2,1e15)-Qw-Qf)/rc


@nb.njit
def flow_src(c, Tg, xi, Vr, Qs, Pg, nsp, pfr):
    S = np.zeros(nsp)
    if pfr: return S
    Qa = Qs*(Tg/_T0)*(_P0/Pg)/60000.0
    tau = Vr/Qa if Qa > 0 else 1e10
    ct = Pg/(_R*Tg)
    for i in range(nsp): S[i] = (xi[i]*ct - c[i])/tau
    return S


@nb.njit
def pulsed_power_numba(t, period, t_on, rise, P_on):
    """Trapezoidal/rectangular pulse power density [W/m³]."""
    t_phase = t % period
    if rise <= 0:
        return P_on if t_phase < t_on else 0.0
    if t_phase < rise:
        return P_on * (t_phase / rise)
    elif t_phase < t_on - rise:
        return P_on
    elif t_phase < t_on:
        return P_on * (t_on - t_phase) / rise
    return 0.0


@nb.njit
def rhs_numba(t, y, nsp, ie, iT,
              lgrid, ltab, kdead, snm, egrid, elN, inN, poN,
              emin, A21b, A22b, kth,
              nrxn, SM,
              eg, eb, et, nei, eel,
              ag2, aA, an2, aE, ao, aa, ab, anr, narr,
              tg2, ts2, tA2, tn2, tk2, ta3, tb3, tnr3, tt2, nte,
              hi, dh,
              eli, ese,
              cef, cf, nef,
              pii, nii, npi, nni,
              xi, Vr, Qs, Pg, pfr,
              Tw, wlf, Ma, cp, Lsq,
              Pdep):
    dy = np.zeros(len(y))
    c = y[:nsp].copy()
    for i in range(nsp):
        if c[i] < cf: c[i] = cf
    if c[0] < cef: c[0] = cef
    neps = y[ie]
    if neps < nef: neps = nef
    if neps > 1e35: neps = 1e35
    Tg = y[iT]
    if Tg < 200.0: Tg = 200.0
    if Tg > 10000.0: Tg = 10000.0

    ce = c[0]; ne = min(ce*_NA, 1e26)
    Ts = max(Tg, 200.0); eth = 1.5*_KB*Ts/_QE
    if ne > 1.0:
        em = neps/ne
        if em < eth: em = eth
        if em > 100.0: em = 100.0
    else: em = max(1.0, eth)
    Te = (2.0/3.0)*em
    ct = Pg/(_R*Tg); N = Pg/(_KB*Ts); Ncm = N*1e-6

    has_ei = True; A22c = 0.0; Pel = 0.0; Qel = 0.0
    le = np.log10(max(em, egrid[0]))
    if em > egrid[-1]: le = np.log10(egrid[-1])

    if em >= emin:
        kr = lut_rates(le, lgrid, ltab, kdead)
        kei = np.empty(len(kr))
        for j in range(len(kr)):
            kei[j] = kr[j]*_NA
            if snm[j]: kei[j] *= Ncm
        A21 = _interp_log(le, lgrid, elN)
        A22c = _interp_log(le, lgrid, inN)
        Pel = ne*N*A21; Qel = Pel*_QE
    elif em > eth:
        dn = emin - eth
        if dn > 1e-6:
            fr = (em-eth)/dn; A21e = A21b*fr; A22e = A22b*fr
        else: fr = 0.0; A21e = 0.0; A22e = 0.0
        Pel = ne*N*A21e; Qel = Pel*_QE; A22c = A22e
        kei = np.empty(len(kth))
        for j in range(len(kth)):
            kei[j] = kth[j]*_NA
            if snm[j]: kei[j] *= Ncm
    else:
        A22c = 0.0
        kei = np.empty(len(kth))
        for j in range(len(kth)):
            kei[j] = kth[j]*_NA
            if snm[j]: kei[j] *= Ncm

    rates = calc_rates(c, Tg, ct, kei, has_ei, Te, Pg, nrxn, 0,
                       eg, eb, et, nei, ag2, aA, an2, aE, ao, aa, ab, anr, narr,
                       tg2, ts2, tA2, tn2, tk2, ta3, tb3, tnr3, tt2, nte)

    St = src_terms(rates, SM, nsp)
    Sf = flow_src(c, Tg, xi, Vr, Qs, Pg, nsp, pfr)
    for i in range(nsp): dy[i] = St[i] + Sf[i]

    df = diff_rate(Tg, Te, Lsq)
    dy[0] -= c[0]*df
    for k in range(npi): dy[pii[k]] -= c[pii[k]]*df
    for k in range(nni): dy[nii[k]] -= c[nii[k]]*df

    if y[0] < cef and dy[0] < 0: dy[0] = 0.0
    for i in range(1, nsp):
        if y[i] < cf and dy[i] < 0: dy[i] = 0.0

    if A22c > 0: Pi = ne*N*A22c*_QE
    else: Pi = e_eloss(rates, eg, eel, nei)

    Qa2 = Qs*(Tg/_T0)*(_P0/Pg)/60000.0
    tau = Vr/Qa2 if Qa2 > 0 and not pfr else 1e10

    Sl = e_lrate(rates, eli, ese)
    Ploss = em*Sl*_NA

    dy[ie] = energy_rhs(neps, ce, Tg, Pdep, Pel, Pi, tau, Ploss, Lsq)

    Qrxn = gas_heat(rates, hi, dh)
    Qlw = Ploss*_QE
    dy[iT] = tgas_rhs(Tg, Qel, tau, Qrxn, Qlw, Pg, Tw, wlf, Ma, cp)

    for i in range(len(dy)):
        v = dy[i]
        if v != v: dy[i] = 0.0
        elif v > 1e30: dy[i] = 1e30
        elif v < -1e30: dy[i] = -1e30
    return dy


def extract_numba_params(solver):
    """Extract all data from Python objects for rhs_numba."""
    rxn = solver.rxn; lut = solver.lut; fl = solver.flow; sm = solver.sm
    ts = np.zeros(len(rxn.te_dependent_reactions), dtype=np.int32)
    for i, r in enumerate(rxn.te_dependent_reactions):
        if getattr(r, 'subtype', '') == 'AT1_KOSSYI': ts[i] = 1
    ese = np.array([-rxn.stoich_matrix[0, j] for j in rxn._electron_loss_indices])
    eel = np.array([r.energy_loss_eV for r in rxn.ei_reactions])
    # Use Maxwellian thermal cache for rates below LUT boundary.
    # At thermal Te (~0.027 eV), Maxwellian gives physically correct low attachment
    # rates (most electrons below 0.035 eV threshold). LUT boundary (ε̄=0.04) has
    # non-Maxwellian EEDF with more electrons at the attachment resonance → too high.
    if not hasattr(lut, '_k_thermal_cache'):
        lut.get_rate_coefficients_maxwellian(0.026)
    return dict(
        nsp=sm.n_species, ie=sm.idx_energy, iT=sm.idx_Tgas,
        lgrid=lut._log_eps_grid.copy(), ltab=lut._log_k_table.copy(),
        kdead=lut._k_dead_mask.copy(), snm=lut._sigma_over_N_mask.copy(),
        egrid=lut._eps_grid.copy(), elN=lut._elastic_power_N.copy(),
        inN=lut._inelastic_power_N.copy(), poN=lut._power_N.copy(),
        emin=float(lut._eps_grid[0]), A21b=float(solver._A21_at_boundary),
        A22b=float(solver._A22_at_boundary), kth=lut._k_thermal_cache.copy(),
        nrxn=rxn.n_reactions, SM=rxn.stoich_matrix.astype(np.float64),
        eg=rxn._ei_global_idx.copy(), eb=rxn._ei_bolsig_idx.copy(),
        et=rxn._ei_target_idx.copy(), nei=len(rxn.ei_reactions), eel=eel,
        ag2=rxn._arr_global_idx.copy(), aA=rxn._arr_A.copy(), an2=rxn._arr_n.copy(),
        aE=rxn._arr_E.copy(), ao=rxn._arr_order.copy(), aa=rxn._arr_idx_a.copy(),
        ab=rxn._arr_idx_b.copy(), anr=rxn._arr_n_reactants.copy(),
        narr=len(rxn.arrhenius_reactions),
        tg2=rxn._te_global_idx.copy(), ts2=ts, tA2=rxn._te_A_cgs.copy(),
        tn2=rxn._te_n_Te.copy(), tk2=rxn._te_k3_cgs.copy(),
        ta3=rxn._te_idx_a.copy(), tb3=rxn._te_idx_b.copy(),
        tnr3=rxn._te_n_reactants.copy(), tt2=rxn._te_target_idx.copy(),
        nte=len(rxn.te_dependent_reactions),
        hi=rxn._gas_heating_idx_arr.copy(), dh=rxn._delta_h_J.copy(),
        eli=np.array(rxn._electron_loss_indices, dtype=np.intp), ese=ese,
        cef=float(solver._ce_floor),
        cf=float(solver._concentration_floor), nef=float(solver._ne_eps_floor),
        pii=np.array(list(solver._positive_ion_indices), dtype=np.intp),
        nii=np.array(list(solver._negative_ion_indices), dtype=np.intp),
        npi=len(solver._positive_ion_indices),
        nni=len(solver._negative_ion_indices),
        xi=fl.x_inlet.copy(), Vr=float(fl.V_reactor), Qs=float(fl.Q_slm),
        Pg=float(fl.P_gas), pfr=fl.flow_model == 'PFR',
        Tw=float(solver.gth.T_wall), wlf=float(solver.gth.wall_loss_freq),
        Ma=float(solver.gth.M_avg), cp=float(solver.gth.cp_avg),
        Lsq=float(solver.ekin.Lambda_sq),
    )
