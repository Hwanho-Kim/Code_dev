# plasma0d_v2 Troubleshooting Log — 2026-03-06 (updated 2026-03-09)

## Problem Statement
3ms pulsed vi_envelope simulation produces negative n_e (and other species).
- Pulsed sDBD: 1333 Hz, 0.19% duty cycle, ~1.4 µs effective pulse width (~8 µs with ringing)
- n_e oscillates 10^10 ~ 10^17 per cycle (7 orders of magnitude)
- BDF solver overshoots to negative during afterglow decay phase
- First negative at t ≈ 257 µs (afterglow of first pulse)

## Baseline (before troubleshooting)
- penalty_rate=1e8: prevented negatives but caused solver TIMEOUT (>10 min, stiffness 3.7×)
- No penalty (single solve_ivp call): ran 184s but n_e → -5e12, Te stuck at 0.667 eV

---

## Trial 1: Segmented Integration (10 µs) + max(dydt,0) guard
**Changes:** solver.py: 10 µs segments, clamp between segments
**Parameters:** seg_dt=10µs, max_step=1µs, rtol=1e-6, atol=1e-10
**Result:** TIMEOUT (>15 min). 300 segments × BDF restart overhead = too slow.
**Conclusion:** Segment size too small, BDF Jacobian rebuild overhead dominates.

## Trial 2: BDF single call + post-clamp
**Changes:** solver.py: single solve_ivp, clamp negatives in output only
**Parameters:** t_end=1.5ms, max_step=1µs, rtol=1e-6, atol=1e-10
**Result:** Completed 177.6s. 39,893 negative values clamped. ne_final=6e-7 (floor).
**Conclusion:** Post-clamp masks negatives but n_e still stuck at floor.

## Trial 3: Radau method
**Changes:** method='Radau'
**Result:** TIMEOUT after setup. Overflow in Jacobian computation.
**Conclusion:** Radau no faster than BDF for this problem.

## Trial 4: Segmented (100 µs)
**Changes:** seg_dt=100µs (15 segments for 1.5ms)
**Result:** TIMEOUT (>5 min). Jacobian overflow still occurs per segment.

## Trial 5: Vector atol (species=1e-18)
**Changes:** atol vector: species=1e-18, ne_eps=1.0, Tgas=0.01
**Result:** TIMEOUT (>8 min). BDF forced to track trace species at 1e-18 → tiny steps.
**Root cause analysis:** atol=1e-10 >> c_e ≈ 1e-15 at afterglow end.
BDF treats c_e as "don't care" → freely overshoots to negative.
Fixing with tight atol forces impractically small step sizes.

## Trial 6: Logarithmic transformation (u = ln(c))
**Changes:** State vector u_i = ln(c_i), c_i = exp(u_i) > 0 always
**Result:** "Required step size is less than spacing between numbers" — instant failure.
**Root cause:** For trace species at c=1e-20:
  du/dt = (dc/dt)/c = tiny/1e-20 = 1e+10 to 1e+16
  → BDF needs dt ≈ 1e-16 s, below machine precision for t ≈ 1e-6.
  Log transform INCREASES stiffness for trace species being created from zero.
**Conclusion:** Log transform not viable for 63-species system with trace → major transitions.

## Trial 7: Selective vector atol (electron+ions only at 1e-20)
**Changes:** atol=1e-20 for electron and 12 ion species only
**Result:** TIMEOUT (>3 min for 200µs test).
**Root cause:** Even 12 species at 1e-20 forces ns-scale steps during afterglow.

## Trial 8: Original solver + post-process clamp (BASELINE RESULT)
**Changes:** Reverted to simplest solver (no guards, scalar atol=1e-10, single BDF)
**Parameters:** t_end=3ms, n_points=6000, max_step=1µs
**Result:** Completed 185.8s, 449,899 RHS evals.
**Output (post-clamped):**
```
t=    0 µs: ne=1.0e+12   Te=2.00 eV   Tg=300.00 K
t=    5 µs: ne=4.2e+16   Te=0.98 eV   Tg=300.16 K  (during pulse)
t=   50 µs: ne=5.1e+15   Te=0.03 eV   Tg=300.30 K  (early afterglow)
t=  200 µs: ne=4.0e+12   Te=0.03 eV   Tg=300.33 K
t=  500 µs: ne=1.0e+10   Te=0.64 eV   Tg=300.33 K
t=  745 µs: ne=floor      Te=0.67 eV   Tg=300.33 K  (locked at floor)
t=  750 µs: ne=floor      Te=0.67 eV   Tg=300.33 K  (2nd pulse - NO reignition!)
t= 1500 µs: ne=floor      Te=0.67 eV   Tg=300.30 K  (3rd pulse - still dead)
```
**Negative species:** 17/63 species go negative internally
**First pulse behavior:** CORRECT (ne rises 10^12 → 10^17, then decays)
**Second+ pulse:** FAILS (ne stuck at floor, no reignition)

---

## ROOT CAUSE ANALYSIS

### The fundamental problem chain:
1. **Afterglow decay**: n_e decays from 10^17 to ~10^9 in 750 µs
   (τ_diff = Λ²/D_a ≈ 42 µs → exp(-750/42) ≈ 1.7e-8 → n_e ≈ 1.7e9)
2. **BDF overshoot**: At c_e ≈ 1e-15 mol/m³ and atol=1e-10, BDF treats c_e as
   "don't care" (atol >> c_e by 10^5). Freely overshoots to c_e = -8.4e-12.
3. **RHS clamp masks the problem**: c = max(y, 1e-30) → chemistry sees c_e=0 →
   all ionization rates = 0 → no electron production → no reignition ever.
4. **Permanent death**: Once n_e goes negative (even briefly), BDF's polynomial
   history is corrupted. n_e never recovers.

### Why it's NOT just a numerical issue:
The physical n_e at afterglow end (≈1e9 m⁻³ = c_e ≈ 2e-15 mol/m³) is 10^5×
smaller than atol. ANY polynomial ODE solver (BDF, Radau, LSODA) will have
this problem — it's inherent to the atol/value ratio.

### Professional solutions:
- **ZDPlasKin**: Uses SUNDIALS CVODE with `y_i ≥ 0` constraints (not in scipy)
- **GlobalKin**: Some versions use log-transformed variables (only works when
  species don't start from exactly zero)
- **CHEMKIN**: Uses CVODE with non-negative constraints
- **Common alternative**: Operator splitting with Patankar-type positive schemes

### What WOULD fix this in scipy:
1. **Custom BDF step with clamp**: Modify BDF internals to clamp state after each
   step (not possible with scipy's API)
2. **Implement CVODE wrapper**: Use python-sundials or diffeq package
3. **Hybrid approach**: Use scipy BDF but implement manual stepping with state
   constraint enforcement

---

## Trial 9: Constrained BDF manual stepping (ce_floor = 1e-30)
**Date:** 2026-03-09
**Changes:** Implemented `_solve_constrained` using `scipy.integrate._ivp.bdf.BDF` manual stepping.
After each internal step, clamp all species to 1e-30, ne_eps to 1e10.
**Parameters:** t_end=1.5ms, constrained=True, max_step=1µs, rtol=1e-6, atol=1e-10
**Result:** Completed 182.6s. 35,917 steps, 31,756 clamp events (88%).
n_e_final = 6e-7 (still at floor). No reignition at 2nd pulse.
**Root cause:** ce_floor = 1e-30 → n_e = 6e-7 m⁻³ at floor.
eps_mean = ne_eps/n_e is undefined at this level → ionization rate = 0 → no reignition.

## Trial 10: Constrained BDF + raised electron floor (ne_seed = 1e10 m⁻³) ✅
**Date:** 2026-03-09
**Changes:**
- Electron-specific floor: `ce_floor = 1e10/NA ≈ 1.66e-14 mol/m³` (background ionization)
- ne_eps floor: `ne_seed * eps_thermal(300K) = 1e10 * 0.039 eV`
- RHS non-negativity guard uses ce_floor for electron
- Vector atol: atol[electron] = ce_floor*0.1, atol[ne_eps] = ne_eps_floor*0.1
**Parameters:** t_end=3ms, constrained=True, max_step=1µs, rtol=1e-6, atol=1e-10 (scalar base)
**Result (3ms, 4 full cycles):**
```
  Wall time: 993 s (~16.5 min)
  RHS evals: 2,473,123
  Steps: 183,097 (clamp events: 169,379 = 92%)

  Pulse 1 (t≈0.5 µs):   n_e_peak = 1.30e17 m⁻³
  Pulse 2 (t≈755 µs):   n_e_peak = 3.76e17 m⁻³  ← REIGNITION SUCCESS
  Pulse 3 (t≈1506 µs):  n_e_peak = 4.02e17 m⁻³
  Pulse 4 (t≈2256 µs):  n_e_peak = 4.22e17 m⁻³
  Afterglow decay: n_e 10^17 → 10^10 in ~500 µs (τ ≈ 33 µs)
  T_gas: 300 → 302.1 K (0.5 K/cycle)
  Products: CO=1.8e-3, H2=7.6e-4, H2O=1.4e-3 mol/m³ formed
```
**Assessment:** PHYSICALLY REASONABLE. Multi-cycle pulsed discharge working.
- Peak n_e increases cycle-to-cycle (accumulated ionization products)
- Te cycles correctly: ~1.85 eV during pulse, 0.026 eV (thermal) in afterglow
- Product formation confirms chemistry is active
- Performance: 993s for 3ms is slow but functional

---

## RESOLVED: Root Cause & Solution Summary

**Problem:** BDF solver drifts electron concentration negative during afterglow (atol >> c_e),
killing reignition in subsequent pulses.

**Solution (Trial 10):**
1. Manual BDF stepping via `scipy.integrate._ivp.bdf.BDF` class
2. Per-step constraint enforcement (CVODE-style clamp)
3. Physically motivated electron floor: n_e_seed = 1e10 m⁻³ (background ionization)
4. Separate floors: electron (1.66e-14 mol/m³), other species (1e-30 mol/m³)
5. Vector atol to reduce unnecessary clamp events

**Remaining limitations:**
- 993s for 3ms simulation (would need optimization for longer runs)
- 92% clamp rate indicates BDF history corruption → solver works harder than necessary
- CH4 conversion appears negative because flow inlet replenishment > conversion at 3ms timescale

## Parameter State (solver.py as of Trial 10)
- concentration_floor = 1e-30 (general species)
- ce_floor = 1e10/NA ≈ 1.66e-14 (electron)
- ne_eps_floor = 1e10 * 0.039 ≈ 390 (ne_eps at thermal 300K)
- max(dydt, 0) guard active when y < floor (species-specific)
- Constrained BDF manual stepping (default solver mode)
- Vector atol: electron and ne_eps at floor*0.1, rest at 1e-10
