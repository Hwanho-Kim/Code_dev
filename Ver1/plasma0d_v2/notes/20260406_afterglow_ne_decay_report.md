# Afterglow Electron Density Decay in 0D DBD Plasma Model — Problem Report

**Date**: 2026-04-06
**Project**: plasma0d_v2 — 0D global plasma chemistry model for CH₄/CO₂ dry reforming in DBD (N₂/O₂ carrier)

---

## 1. Executive Summary

Our 0D plasma model simulates a pulsed surface dielectric barrier discharge (sDBD) for CH₄/CO₂ dry reforming. During the OFF phase (afterglow) of each pulse, the **electron density drops by 8 orders of magnitude** (ne ~ 10¹⁴ → 10⁶ m⁻³) within ~600 µs. Published 0D DBD models show only 2–3 orders of magnitude decay. This excessive decay effectively shuts down afterglow chemistry and requires non-physical workarounds (electron re-seeding at each pulse start). The root cause has been identified as **volume-averaging over the full discharge volume** (V_eff = 4.9 cm³) rather than individual filament volumes (~10⁻⁵ cm³), but no satisfactory solution has been found within the current 0D framework.

---

## 2. Model Configuration

### 2.1 Reactor Geometry
- **Reactor type**: Surface DBD (sDBD), coplanar electrode
- **Electrode area**: 70 mm × 70 mm
- **Discharge gap**: ~1 mm
- **Effective discharge volume (V_eff)**: 4.9 × 10⁻⁶ m³ (4.9 cm³)
- **Reactor physical volume (V_reactor)**: 250 cm³ (not used in current model)

### 2.2 Operating Conditions
- **Gas composition**: N₂ = 70%, O₂ = 15%, CO₂ = 14%, CH₄ = 1%
- **Pressure**: 1 atm (101,325 Pa)
- **Gas temperature**: 300–523 K (parametric)
- **Flow rate**: 0.4 slm → residence time τ = V_eff / Q = 0.663 s

### 2.3 Pulsed Power
- **PRF**: 1333 Hz → T_pulse = 750 µs
- **Duty cycle**: 20% → t_ON = 150 µs, t_OFF = 600 µs
- **P_peak**: 32.5 W → P_avg = 6.5 W
- **Power deposition**: P_dep = P_peak / V_eff = 6.63 × 10⁶ W/m³ (ON phase only)
- **Waveform**: Trapezoidal (100 ns rise/fall)

### 2.4 Chemistry
- **Species**: 63 (neutrals, ions, excited states)
- **Reactions**: 218 (electron-impact, Arrhenius, ion-molecule, excited-state)
- **Electron kinetics**: BOLSIG+ lookup tables (LUT) for ε̄ ≥ 0.04 eV, Maxwellian fallback below
- **Solver**: CVODE (SUNDIALS 6.x via ctypes), BDF, rtol=1e-6, atol=vector

### 2.5 Operator Splitting
Each pulse cycle is solved in two phases:
1. **ON phase**: Full RHS with power deposition (Numba-accelerated), CVODE with non-negative constraints
2. **OFF phase**: Same RHS but P_dep = 0 (all reactions active including EI at thermal Te)

Transitions:
- **ON → OFF**: Clamp negatives to zero, reset ne_eps = n_e × ε_thermal (elastic cooling ~100 ns justifies instant thermalization)
- **OFF → ON**: Clamp negatives, re-seed ne ≥ 10⁸ m⁻³ (background ionization proxy — non-physical workaround)

---

## 3. Problem Description

### 3.1 Observed Behavior

During the OFF phase (afterglow) of each pulse:

| Parameter | ON peak | OFF valley | Ratio | Timescale |
|-----------|---------|------------|-------|-----------|
| **n_e** | ~10¹⁴ m⁻³ (2.5 × 10⁶ cm⁻³) | ~3.2 × 10⁶ m⁻³ (3.2 cm⁻³) | **10⁻⁸** | ~600 µs |
| **T_e** | ~1.85 eV | 0.026 eV (thermal) | 10⁻² | ~100 ns |
| **O₂⁻** | ~10⁸ m⁻³ | accumulated | — | — |

The electron density drops by **8 orders of magnitude** in a single OFF phase (600 µs). Each subsequent pulse requires artificial re-seeding to ne = 10⁸ m⁻³ to reignite.

### 3.2 Expected Behavior (Literature)

Published 0D DBD models report afterglow ne decay of only **2–3 orders of magnitude**:

| Reference | Gas | ne_peak (cm⁻³) | ne_valley (cm⁻³) | Decay | Model Volume |
|-----------|-----|:--------------:|:----------------:|:-----:|:------------:|
| Snoeckx 2013 (Bogaerts) | CH₄/CO₂ | ~10¹⁴ | ~10¹²–10¹¹ | 2–3 decades | V_filament |
| van 't Veer 2020 (Bogaerts) | N₂/H₂ | ~10¹⁴ | ~10¹¹ | 3 decades | V_filament |
| Meyer, Hartman, Kushner 2025 | Ar/CH₄/O₂ | 1.8 × 10¹³ | ~10¹¹ | 2 decades | V_reactor ≈ V_filament* |
| Ning 2023 (GlobalKin) | Various | ~10¹³–10¹⁴ | ~10¹¹–10¹² | 2–3 decades | V_filament |

(*) Meyer et al. use a microfluidic channel (500 µm gap) where V_eff ≈ V_reactor, avoiding the volume-averaging issue.

### 3.3 Impact

1. **Afterglow chemistry disabled**: With ne ~ 10⁶ m⁻³, electron-driven reactions in the afterglow (dissociative recombination, vibrational relaxation, attachment-detachment cycling) are effectively zero.
2. **Non-physical re-seeding required**: Each pulse must artificially inject electrons (ne = 10⁸ m⁻³) — no physical basis for this in a continuous discharge.
3. **Pulsed-specific advantages lost**: The model cannot capture pulse-specific afterglow chemistry (e.g., VT relaxation, radical recombination enhanced by moderate ne).
4. **Conversion prediction**: Despite these issues, power-scaled CH₄ conversion agrees with continuous mode (5.52% vs 5.56%), suggesting the ON-phase chemistry dominates. The afterglow problem primarily affects the fidelity of the pulsed model physics.

---

## 4. Root Cause Analysis

### 4.1 Dominant Electron Loss: 3-Body O₂ Attachment

The overwhelming electron sink in the afterglow is:

**e + O₂ + M → O₂⁻ + M** (Reaction 165)

Rate coefficient (Kossyi 1992, validated against Chanin 1962):
- k(M=O₂) = 1.4 × 10⁻²⁹ cm⁶/s
- k(M=N₂) = 1.07 × 10⁻³¹ cm⁶/s (currently missing from code — ~21% underestimate)

At 1 atm, 300 K:
- n(O₂) = 3.6 × 10¹⁸ cm⁻³
- n(M) = 2.4 × 10¹⁹ cm⁻³ (total gas)
- **ν_att = k × n(O₂) × n(M) ≈ 8.6 × 10⁷ s⁻¹** → **τ_att ≈ 12 ns**

This means electrons are removed on a **12 ns timescale** — far faster than the 600 µs OFF phase.

### 4.2 Detachment Cannot Compensate

Nine detachment reactions are included (all from Kossyi 1992):

| Reaction | k (cm³/s) | Partner density (cm⁻³) | ν_det (s⁻¹) | vs ν_att |
|----------|-----------|:---------------------:|:------------:|:--------:|
| O₂⁻ + O → O₃ + e | 3.3 × 10⁻¹⁰ | [O] ~ 10⁶ (afterglow) | 3.3 × 10² | **2.6 × 10⁻⁶** |
| O⁻ + O → O₂ + e | 1.4 × 10⁻¹⁰ | [O] ~ 10⁶ | 1.4 × 10² | — |
| O₂⁻ + N₂(A) → ... + e | 2.1 × 10⁻⁹ | [N₂(A)] ~ 0* | ~0 | — |
| O₂⁻ + O₂(a¹Δ) → ... + e | 2.0 × 10⁻¹⁰ | ~10⁸ | 2.0 × 10⁴ | 2.3 × 10⁻⁴ |

(*) N₂(A) is quenched by O₂ in τ ~ 0.4 µs — unavailable in afterglow.

**Detachment-to-attachment ratio ≈ 1/8000** at our ne_peak.

**In literature models** (ne_peak ~ 10¹⁴ cm⁻³): Atomic O density is ~10¹²–10¹³ cm⁻³, giving ν_det ~ 10⁴–10⁵ s⁻¹. Attachment still dominates initially, but the higher ne means electrons survive longer before reaching the regime where detachment balances attachment, resulting in only 2–3 decades of decay.

### 4.3 Confirmed Non-Issues

Extensive investigation has ruled out the following:

| Hypothesis | Investigation | Result |
|-----------|---------------|--------|
| Missing detachment reactions | Literature audit (Kossyi, Janalizadeh 2021) | All significant pathways included. O⁻+N₂ endothermic at 300K. |
| Incorrect rate coefficients | Comparison with Kossyi 1992, Chanin 1962 | k_att matches standard values (±20%). |
| Penning ionization / N₂(A) detachment | Kinetic analysis | N₂(A) quenched by O₂ in 0.4 µs — unavailable. |
| Superelastic collisions buffering Te | Literature check | Negligible at 1 atm (elastic cooling dominates). |
| Reaction mechanism gaps | Added 6 missing reactions (O+OH, Zeldovich N chemistry) | Max impact −0.11%p on conversion, no ne effect. |
| Rate coefficient uncertainties | Corrected OH+OH+M (44–120× low), HO₂+HO₂ (bimol. missing) | Max impact −0.09%p. |

### 4.4 The Real Root Cause: Volume-Averaging

The fundamental issue is the **8 orders of magnitude difference in ne_peak** between our model and literature:

```
Our model:      ne_peak = 2.5 × 10⁶ cm⁻³    (averaged over V_eff = 4.9 cm³)
Literature:     ne_peak = 10¹⁴ cm⁻³           (single filament, V ~ 10⁻⁵ cm³)
                                                Ratio: 10⁸ ×
```

**Why this matters for afterglow decay:**

1. **Attachment** (e + O₂ + M → O₂⁻ + M) is **first-order in ne**: ν_att is independent of ne. Same τ_att ≈ 12 ns regardless of initial ne.

2. **Detachment** (O₂⁻ + O → O₃ + e) depends on **atomic O density**, which scales with ne (EI production):
   - Our model: [O] ~ 10⁶ cm⁻³ → ν_det ~ 10² s⁻¹
   - Literature: [O] ~ 10¹² cm⁻³ → ν_det ~ 10⁵ s⁻¹

3. The **attachment-detachment equilibrium ne** scales as:

   ne_eq ∝ (ν_det / ν_att) × n(O₂⁻)

   With our low [O], ν_det/ν_att ~ 10⁻⁶, so ne_eq is negligibly small.
   In filament models, ν_det/ν_att ~ 10⁻³, supporting ne_eq ~ 10¹¹–10¹² cm⁻³.

4. **Physical picture**: In a real sDBD, each microdischarge filament has ne ~ 10¹⁴ cm⁻³ in a tiny volume (~10⁻⁵ cm³). After the pulse, this filament's electrons decay by 2–3 decades (to ~10¹¹–10¹²) before attachment-detachment equilibrates. Our 0D model averages the filament ne over the entire 4.9 cm³ discharge volume, yielding ne ~ 10⁶ cm⁻³ — too low for detachment to matter.

---

## 5. Electron Budget in the Afterglow (Quantitative)

At t = 0⁺ (start of OFF phase), T_gas = 303 K, ne = 2.5 × 10¹² m⁻³:

### 5.1 Electron Loss Rates (m⁻³ s⁻¹)

| Process | Rate | Fraction |
|---------|:----:|:--------:|
| 3-body attachment: e + O₂ + M → O₂⁻ + M | 9.4 × 10¹⁸ | **97%** |
| Dissociative recombination: e + O₂⁺ → O + O | ~2 × 10¹⁷ | ~2% |
| Dissociative attachment: e + O₂ → O⁻ + O | ~1 × 10¹⁷ | ~1% |
| Ambipolar diffusion | ~5 × 10¹⁵ | <0.1% |

### 5.2 Electron Source Rates (m⁻³ s⁻¹)

| Process | Rate | vs Attachment |
|---------|:----:|:------------:|
| Detachment: O₂⁻ + O → O₃ + e | ~1.0 × 10¹⁵ | 1/9400 |
| Detachment: O⁻ + O → O₂ + e | ~1.0 × 10¹⁴ | 1/94000 |
| Detachment: O₂⁻ + O₂(a¹Δ) → ... + e | ~5 × 10¹³ | — |
| **Total detachment** | **~1.1 × 10¹⁵** | **1/8000** |

**Net loss rate**: ~9.4 × 10¹⁸ m⁻³ s⁻¹
**Characteristic decay time**: ne / (dne/dt) ≈ 2.5 × 10¹² / 9.4 × 10¹⁸ ≈ **0.3 µs**

After ~10 × τ_decay ≈ 3 µs, ne has dropped below 10⁶ m⁻³. The remaining 597 µs of the OFF phase are chemically inert with respect to electrons.

---

## 6. Current Workarounds

### 6.1 Electron Re-seeding (each pulse start)
```python
if y[0] < ce_seed:                    # ce_seed = ne_seed / NA = 1e8 / 6.022e23
    y[0] = ce_seed
    y[idx_energy] = ne_seed * 1.5 * 0.026   # thermal energy
```
- Injects ne = 10⁸ m⁻³ at each pulse start
- Proxy for background ionization (cosmic rays, UV) — physically weak justification at 10⁸ m⁻³
- Works: pulses reignite and quasi-steady ne established after ~5 cycles

### 6.2 ne_eps Thermal Reset (ON → OFF transition)
```python
eps_th = 1.5 * KB * T_gas / QE        # ε_thermal ≈ 0.039 eV at 300K
y[idx_energy] = y[0] * NA * eps_th     # ne_eps = n_e × ε_thermal
```
- Resets electron energy to thermal at each OFF phase start
- Justified: elastic cooling at 1 atm thermalizes electrons in ~100 ns

### 6.3 ne_eps Proportional Tracking (during OFF phase)
```python
if c_e > ce_floor * 10:
    rel_rate = dydt[0] / c_e           # relative ne change rate
    dydt[idx_energy] = ne_eps * rel_rate  # ne_eps follows ne, preserving eps_mean
else:
    dydt[idx_energy] = 0.0
```
- Ensures ε_mean is conserved as ne decays (avoids Te divergence)
- Without this: ne_eps equation has different eigenvalues than ne equation → Te oscillates wildly

---

## 7. LUT → Maxwellian Discontinuity (Secondary Issue)

At the LUT lower boundary (ε̄ = 0.04 eV), switching from BOLSIG+ EEDF to Maxwellian causes a **5× jump in k_att**:

- **BOLSIG+ EEDF at ε̄ = 0.04 eV**: More electrons at the O₂ attachment resonance (~0.035 eV) → higher k_att
- **Maxwellian at ε̄ = 0.04 eV**: Fewer electrons at 0.035 eV → lower k_att

This discontinuity occurs during the ON→OFF Te cooling transition and briefly accelerates electron loss. However, it is a **secondary issue** — even with perfect Maxwellian rates, τ_att ~ 12 ns guarantees rapid ne decay at our low densities.

**Fix (not yet implemented)**: Extend BOLSIG+ calculations to ε̄ < 0.04 eV to eliminate the discontinuity.

---

## 8. What Has Been Tried

| # | Approach | Result | Why It Failed/Worked |
|---|----------|--------|---------------------|
| 1 | Freeze electrons in OFF phase | ne preserved, but no afterglow chemistry | Non-physical; user rejected |
| 2 | Full RHS with P_dep=0 | 15 s/pulse, convergence failures | Afterglow stiffness (τ_att=12ns vs τ_OFF=600µs) |
| 3 | CVODE non-negative constraints only | ne goes negative anyway | CVODE constraints not 100% guaranteed |
| 4 | ne_eps proportional tracking (no reset) | Te diverges (0.28–3.05 eV) | ne_eps initial value is "hot" (1.5 eV from ON phase) |
| 5 | ne_eps thermal reset + re-seeding | **Works**: 0 fails, 0.22s/pulse | ne still drops 8 decades, but simulation stable |
| 6 | rhs_off with EI activation (CX-based) | ne_valley 5× higher (3.2e6 vs 6.6e5) | EI at thermal Te contributes marginally |
| 7 | Rate coefficient corrections | Max impact −0.09%p | Rates are correct; problem is structural |
| 8 | Missing reaction additions (6 reactions) | Max impact −0.11%p | All significant pathways already included |
| 9 | Kossyi thermal attachment injection | No improvement | Removed; CX-based calculation is correct |
| 10 | LUT boundary clamping | ne 5× worse | BOLSIG+ k_att higher than Maxwellian at boundary |

---

## 9. Possible Solutions (Not Yet Implemented)

### 9.1 Filament-Level 0D Model
- Model a **single microdischarge filament** (V ~ 10⁻⁵ cm³) instead of V_eff
- ne_peak would be ~10¹⁴ cm⁻³ → afterglow decay 2–3 decades (matches literature)
- **Pros**: Physically correct; standard approach in literature (Bogaerts group)
- **Cons**: Major model restructuring; need to determine filament geometry, number of filaments per pulse, inter-filament interactions; requires post-processing to recover reactor-averaged quantities

### 9.2 Effective Attachment Reduction
- Artificially reduce k_att to account for volume-averaging
- Tunable parameter: k_att_eff = k_att × (V_filament / V_eff)^α
- **Pros**: Simple to implement
- **Cons**: No physical basis; would affect ON phase chemistry too

### 9.3 Multi-Zone Model
- Split into **discharge zone** (V_filament) + **post-discharge zone** (V_eff − V_filament)
- Solve filament chemistry at high ne, then mix with background
- **Pros**: Captures both filament physics and volume-averaging
- **Cons**: Significant complexity; need mixing timescale, number of filaments

### 9.4 Maintain Current Re-seeding
- Keep the ne re-seeding workaround (ne ≥ 10⁸ m⁻³ per pulse)
- Accept that afterglow chemistry is not captured
- **Pros**: Working now; CH₄ conversion predictions match experiments
- **Cons**: No afterglow chemistry; non-physical; limits model's predictive capability for pulsed-specific phenomena

### 9.5 LUT Extension (Secondary Fix)
- Extend BOLSIG+ calculations to ε̄ < 0.04 eV
- Eliminates k_att 5× discontinuity
- **Pros**: Straightforward; improves overall model quality
- **Cons**: Does not address the fundamental 8-decade decay issue

---

## 10. Key Physical Constants and Code Parameters

### 10.1 Rate Coefficients

| Reaction | k | Units | Source |
|----------|---|-------|--------|
| e + O₂ + O₂ → O₂⁻ + O₂ | 1.4 × 10⁻²⁹ | cm⁶/s | Kossyi 1992 |
| e + O₂ + N₂ → O₂⁻ + N₂ | 1.07 × 10⁻³¹ | cm⁶/s | Kossyi 1992 (missing in code) |
| O₂⁻ + O → O₃ + e | 3.3 × 10⁻¹⁰ | cm³/s | Kossyi 1992 |
| O⁻ + O → O₂ + e | 1.4 × 10⁻¹⁰ | cm³/s | Kossyi 1992 |
| O₂⁻ + N₂(A) → O₂ + N₂ + e | 2.1 × 10⁻⁹ | cm³/s | Kossyi 1992 |
| O₂⁻ + O₂(a¹Δg) → 2O₂ + e | 2.0 × 10⁻¹⁰ | cm³/s | Kossyi 1992 |

### 10.2 Code Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| ce_floor | 1 × 10⁻³⁰ mol/m³ | solver.py:74 |
| ne_eps_floor | 1 × 10⁻²⁰ | solver.py:75 |
| ne_seed | 1 × 10⁸ m⁻³ | solver.py:73 |
| eps_min_lut | ~0.04 eV | boltzmann.py (data-dependent) |
| V_eff | 4.9 × 10⁻⁶ m³ | config.yaml:42 |

### 10.3 Characteristic Timescales (300 K, 1 atm)

| Process | Timescale |
|---------|-----------|
| 3-body O₂ attachment | τ_att ≈ 12 ns |
| N₂(A) quenching by O₂ | τ_q ≈ 0.4 µs |
| Elastic e-N₂ cooling | τ_cool ≈ 100 ns |
| Pulse ON duration | t_ON = 150 µs |
| Pulse OFF duration | t_OFF = 600 µs |
| Flow residence time | τ = 663 ms |

---

## 11. Relevant Code Structure

```
plasma0d_v2/
├── solver.py         # Main solver, rhs_off() (line 724-814), operator splitting
├── cvode_wrapper.py  # SUNDIALS CVODE ctypes binding
├── reactions.py      # Reaction rate computation, source terms
├── input/
│   └── reactions.yaml  # 218 reactions (attachment: id165, detachment: id168-171, 211-218)
├── boltzmann.py      # BOLSIG+ LUT, Maxwellian fallback (line 393-428)
├── electron_kinetics.py  # Energy balance, diffusion
├── numba_core.py     # Numba-JIT RHS for ON phase
├── config.yaml       # V_eff, power, geometry
└── power.py          # Pulsed power profile
```

---

## 12. References

1. **Kossyi et al. (1992)**, Plasma Sources Sci. Technol. 1, 207 — Standard attachment/detachment rate coefficients for air plasmas
2. **Chanin et al. (1962)**, Phys. Rev. A 128, 671 — Experimental 3-body attachment measurement
3. **Snoeckx et al. (2013)**, J. Phys. Chem. C 117, 4957 — Bogaerts group 0D DBD model, CH₄/CO₂, ne_peak ~ 10¹⁴ cm⁻³
4. **van 't Veer et al. (2020)**, J. Phys. Chem. C 124, 22871 — Bogaerts N₂/H₂ DBD 0D model
5. **Meyer, Hartman, Kushner (2025)**, J. Appl. Phys. — GlobalKin 0D, Ar/CH₄/O₂, microfluidic channel
6. **Ning et al. (2023)** — GlobalKin 10 kHz, 100 s simulation (10⁶ pulses, brute-force)
7. **Lietz & Kushner (2016)** — 5000 pulses @ 10 kHz, GlobalKin
8. **Janalizadeh & Pasko (2021)**, GRL — O⁻ + N₂ detachment rate
9. **Midey & Viggiano (2007)** — O₂⁻ + O₂(a¹Δg) rate

---

## 13. Questions for Discussion

1. **Is there a standard approach** for handling volume-averaged ne in sDBD 0D models where V_eff >> V_filament?
2. Are there **published sDBD-specific 0D models** (not VDBD) that address this volume-averaging issue?
3. Could a **pseudo-filament correction** (scaling ne by V_eff/V_filament during afterglow only) be physically justified?
4. Is the **attachment-detachment equilibrium** approach viable — i.e., solving for ne_eq analytically in the afterglow and using it as a floor?
5. Would a **two-temperature (filament/background) electron model** within the 0D framework be feasible?
6. Are there **alternative formulations** (e.g., using number density per filament, then multiplying by filament number) that avoid this issue while keeping the 0D global model structure?
