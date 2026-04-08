# Sim_exp_comparison_0312

0D Plasma Chemistry Simulation vs Experiment: CH4/CO2/N2/O2 sDBD Reactor  
Date: 2026-03-12

---

## 1. Simulation Conditions

| Parameter | Value | Note |
|---|---|---|
| Power (P) | 5 W | Constant mode |
| Flow rate (Q) | 0.4 slm | At STP |
| Pressure | 1 atm (101325 Pa) | |
| V_eff | 1.6 cm3 | Discharge volume |
| V_reactor | 250 cm3 | Physical reactor volume |
| f = V_eff/V_reactor | 0.0064 | Volume dilution factor |
| Lambda | 1 mm | Diffusion length (insensitive after IIR) |
| Inlet composition | CH4:5%, CO2:5%, N2:78%, O2:12% | |
| Chemistry | 207 reactions, 63 species | 27 EI + 170 Arrhenius + 10 Te-dependent |
| Solver | BDF, rtol=1e-6, atol=1e-12 | |
| Simulation time | 5 x tau (residence time) | Steady-state convergence ensured |

---

## 2. CH4 Conversion: Simulation vs Experiment

| T (C) | T (K) | Exp (%) | Sim (%) | Delta (%p) | Sim/Exp |
|---|---|---|---|---|---|
| 30 | 303 | 5.26 | 17.59 | +12.33 | 3.34 |
| 100 | 373 | 8.05 | 20.49 | +12.44 | 2.55 |
| 180 | 453 | 14.36 | 25.62 | +11.26 | 1.78 |
| 250 | 523 | 20.02 | 29.83 | +9.81 | 1.49 |

**RMSE = 11.51 %p**, MAE = 11.46 %p

**Note on Previous RMSE=1.14%**: The earlier result used `t_end = min(1.5*tau, 15s)`, which at V_reactor=250 cm3 (tau=20-34s) meant simulations ran for only 0.44-0.77 tau. The system had NOT reached steady state, resulting in artificially low (coincidentally well-matched) conversion values. The current analysis uses `t_end = 5*tau` to ensure full CSTR convergence, revealing the true model prediction.

**Interpretation**: The model correctly captures the TREND (increasing conversion with temperature) but OVER-PREDICTS by a factor of 1.5-3.3x. Over-prediction decreases with temperature, suggesting the Arrhenius reaction rates (dominant at high T) are better calibrated than the low-T radical-driven pathways. This indicates V_reactor may need to be significantly larger than 250 cm3, or certain reaction rates (particularly OH + CH4, N2* + CH4) may be over-estimated.

---

## 3. Plasma Parameters at Steady State

| T (C) | T (K) | Te (eV) | mean_eps (eV) | n_e (m-3) | N_gas (m-3) | n_e/N | tau (s) |
|---|---|---|---|---|---|---|---|
| 30 | 303 | 2.305 | 3.458 | 4.07e14 | 2.42e25 | 1.68e-11 | 33.81 |
| 100 | 373 | 2.051 | 3.077 | 6.85e14 | 1.97e25 | 3.48e-11 | 27.46 |
| 180 | 453 | 1.849 | 2.774 | 1.13e15 | 1.62e25 | 6.97e-11 | 22.61 |
| 250 | 523 | 1.801 | 2.702 | 1.42e15 | 1.40e25 | 1.01e-10 | 19.59 |

**Key observations:**
- Te DECREASES with T_gas (2.31 -> 1.80 eV): Higher T_gas -> lower N_gas -> n_e increases to maintain constant P_dep -> eps = ne_eps/n_e decreases
- n_e INCREASES with T_gas (4.1e14 -> 1.4e15 m-3): Self-consistent response to maintain power balance at lower gas density
- Ionization degree n_e/N ~ 10^-11 to 10^-10: Characteristic of weakly ionized atmospheric DBD

---

## 4. CO2 Conversion

| T (C) | T (K) | CO2 conv (%) | c0_CO2 (mol/m3) | cf_CO2 (mol/m3) | Delta_c (mol/m3) |
|---|---|---|---|---|---|
| 30 | 303 | +0.33 | 2.011 | 2.004 | +6.54e-3 |
| 100 | 373 | +0.69 | 1.634 | 1.622 | +1.13e-2 |
| 180 | 453 | +1.07 | 1.345 | 1.331 | +1.44e-2 |
| 250 | 523 | +2.43 | 1.165 | 1.137 | +2.84e-2 |

CO2 is NET consumed in all cases, with conversion increasing from 0.3% to 2.4% with temperature.

---

## 5. Plasma-Thermal Synergy: CH4 Consumption by Mechanism Type

### 5.1 Overview

| T (C) | Total cons (mol/m3/s) | EI | EI (%) | Arrhenius | Arr (%) |
|---|---|---|---|---|---|
| 30 | -1.075e-2 | -4.358e-3 | 40.5 | -6.394e-3 | 59.5 |
| 100 | -1.240e-2 | -3.954e-3 | 31.9 | -8.441e-3 | 68.1 |
| 180 | -1.540e-2 | -3.544e-3 | 23.0 | -1.185e-2 | 77.0 |
| 250 | -1.789e-2 | -3.342e-3 | 18.7 | -1.455e-2 | 81.3 |

**Clear synergy**: Even at 30 C, the Arrhenius (radical) pathway already dominates CH4 consumption (59.5%). At 250 C, it reaches 81.3%. The EI contribution is roughly constant in absolute terms (~3.3-4.4e-3) while Arrhenius grows rapidly (~6.4e-3 -> 14.5e-3), indicating that plasma-generated radicals drive the temperature sensitivity.

### 5.2 Top CH4 Consumption Reactions (by temperature)

#### 30 C (303 K)

| Rank | Reaction | Type | Contribution | % of cons |
|---|---|---|---|---|
| 1 | e + CH4 -> CH3 + H + e | EI | -2.793e-3 | 26.0 |
| 2 | CH4 + OH -> CH3 + H2O | Arrhenius | -1.759e-3 | 16.4 |
| 3 | CH4 + N2(A) -> N2 + CH3 + H | Arrhenius | -1.499e-3 | 13.9 |
| 4 | e + CH4 -> CH2 + H2 + e | EI | -1.431e-3 | 13.3 |
| 5 | CH4 + N2(a1) -> N2 + C + 2H2 | Arrhenius | -8.949e-4 | 8.3 |
| 6 | CH4 + N2(a1) -> N2 + CH3 + H | Arrhenius | -8.949e-4 | 8.3 |
| 7 | CH4 + N2(a1) -> N2 + CH2 + H2 | Arrhenius | -8.949e-4 | 8.3 |

At 30 C, electron-impact dissociation (R1, R2) is #1 and #4, but N2 excited states (N2(A), N2(a1)) collectively contribute 38.9% and OH abstraction 16.4%.

#### 250 C (523 K)

| Rank | Reaction | Type | Contribution | % of cons |
|---|---|---|---|---|
| 1 | CH4 + OH -> CH3 + H2O | Arrhenius | -7.627e-3 | 42.6 |
| 2 | CH4 + O -> CH3 + OH | Arrhenius | -2.920e-3 | 16.3 |
| 3 | e + CH4 -> CH3 + H + e | EI | -2.181e-3 | 12.2 |
| 4 | CH4 + N2(A) -> N2 + CH3 + H | Arrhenius | -1.348e-3 | 7.5 |
| 5 | e + CH4 -> CH2 + H2 + e | EI | -1.104e-3 | 6.2 |
| 6 | CH4 + N2(a1) -> N2 + C + 2H2 | Arrhenius | -8.248e-4 | 4.6 |

At 250 C, OH abstraction alone is 42.6%, and O-atom abstraction (negligible at 30 C) rises to 16.3% (#2). Together, radical abstraction (OH + O) is 58.9% of total CH4 consumption.

### 5.3 Synergy Mechanism

The plasma-thermal synergy operates as follows:

1. **Plasma generates radicals**: e + O2 -> O + O + e, e + H2O -> OH + H + e, e + CO2 -> CO + O + e
2. **Radicals attack CH4 thermally**: OH + CH4 -> CH3 + H2O (Ea = 16.0 kJ/mol), O + CH4 -> CH3 + OH (Ea = 37.7 kJ/mol)
3. **Temperature effect**: Arrhenius rate k = A*exp(-Ea/RT) increases exponentially with T for the radical reactions
4. **N2 excited states**: N2(A) and N2(a1) provide energy transfer pathways unique to N2-containing mixtures (~25% at 30 C, ~13% at 250 C)

The synergy factor can be defined as:

    Synergy = (Arrhenius contribution) / (EI contribution)

| T (C) | Synergy factor |
|---|---|
| 30 | 1.47 |
| 100 | 2.14 |
| 180 | 3.34 |
| 250 | 4.35 |

The synergy factor increases from 1.5x to 4.4x as temperature rises from 30 C to 250 C, quantitatively demonstrating how thermal activation amplifies plasma-generated radical chemistry.

---

## 6. CO2 Reaction Pathway Analysis

### 6.1 CO2 Consumption by Mechanism

| T (C) | Total cons | EI | EI (%) | Arrhenius | Arr (%) |
|---|---|---|---|---|---|
| 30 | -6.128e-3 | -3.778e-3 | 61.6 | -2.351e-3 | 38.4 |
| 100 | -6.367e-3 | -4.177e-3 | 65.6 | -2.190e-3 | 34.4 |
| 180 | -6.560e-3 | -4.539e-3 | 69.2 | -2.022e-3 | 30.8 |
| 250 | -6.577e-3 | -4.623e-3 | 70.3 | -1.955e-3 | 29.7 |

Unlike CH4 (where Arrhenius dominates), CO2 consumption is primarily EI-driven (62-70%) with weaker Arrhenius contribution. This is because CO2 dissociation has high activation energy, making it less sensitive to thermal chemistry.

### 6.2 Top CO2 Consumption Reactions

| Rank | Reaction | Type | 30 C (%) | 250 C (%) |
|---|---|---|---|---|
| 1 | e + CO2 -> CO + O + e | EI | 60.5 | 69.8 |
| 2 | CH2 + CO2 -> CH2O + CO | Arrhenius | 37.9 | 29.3 |
| 3 | e + CO2 -> CO2+ + 2e | EI | 1.1 | 0.5 |
| 4 | N2(A) + CO2 -> N2 + CO + O | Arrhenius | 0.4 | 0.4 |

CO2 decomposition is overwhelmingly dominated by two reactions: electron-impact dissociation (~60-70%) and CH2 radical insertion (~30-38%).

### 6.3 Top CO2 Production Reactions

| Rank | Reaction | Type | 30 C (%) | 250 C (%) |
|---|---|---|---|---|
| 1 | CO + OH -> CO2 + H | Arrhenius | 99.3 | 95.1 |
| 2 | CO + O -> CO2 | Arrhenius | 0.6 | 4.9 |

CO2 re-formation is almost entirely from CO + OH -> CO2 + H. This creates a near-closed cycle:
- CO2 is dissociated by electron impact to form CO + O
- The CO reacts back with OH to regenerate CO2
- Net CO2 conversion is only the difference: 0.3-2.4%

### 6.4 CO2 Mass Balance

| T (C) | Consumption | Production | S_flow | Net (Consumption - Production) |
|---|---|---|---|---|
| 30 | -6.128e-3 | +5.984e-3 | +1.93e-4 | -1.44e-4 |
| 100 | -6.367e-3 | +6.004e-3 | +4.11e-4 | -3.63e-4 |
| 180 | -6.560e-3 | +5.974e-3 | +6.37e-4 | -5.86e-4 |
| 250 | -6.577e-3 | +5.179e-3 | +1.45e-3 | -1.40e-3 |

The CO2 cycle has high throughput (~6e-3 mol/m3/s both ways) but very low net conversion because CO + OH -> CO2 + H efficiently recycles CO back to CO2. The net conversion increases with temperature primarily because OH concentration decreases (OH is consumed more by CH4 + OH), reducing CO2 regeneration efficiency.

---

## 7. Product Concentrations and Selectivity

### 7.1 Product Concentrations (ppm at steady state)

| Species | 30 C | 100 C | 180 C | 250 C |
|---|---|---|---|---|
| CO | 7774 | 9214 | 11574 | 13793 |
| H2O | 13170 | 15975 | 20212 | 22493 |
| H2 | 3261 | 2912 | 2588 | 2542 |
| O3 | 4063 | 1318 | 188 | 17 |
| CH2O | 142 | 174 | 221 | 306 |
| CH3OH | 0.8 | 0.8 | 0.7 | 0.6 |
| C2H6 | <0.1 | <0.1 | <0.1 | 0.1 |
| C2H2, C2H4 | ~0 | ~0 | ~0 | ~0 |

### 7.2 Carbon Selectivity (% of consumed CH4 carbon)

| Product | 30 C | 100 C | 180 C | 250 C |
|---|---|---|---|---|
| CO | 88.4 | 89.9 | 90.3 | 92.5 |
| CH2O | 1.6 | 1.7 | 1.7 | 2.1 |
| CH3OH | <0.1 | <0.1 | <0.1 | <0.1 |
| C2+ hydrocarbons | ~0 | ~0 | ~0 | ~0 |

**CO is overwhelmingly the dominant carbon product (88-93%)** with CH2O as a minor secondary product. C2+ hydrocarbon formation (C2H2, C2H4, C2H6) is negligible. This suggests the CH3 radical, once formed, is quickly oxidized rather than undergoing recombination (CH3 + CH3 -> C2H6).

### 7.3 Notable Trends

- **H2 decreases** with temperature (3261 -> 2542 ppm): At higher T, H atoms are consumed faster by O2 + H -> OH or OH + H2 -> H2O
- **O3 drops dramatically** (4063 -> 17 ppm): Thermal decomposition O3 -> O2 + O accelerates at higher T
- **H2O increases** (13170 -> 22493 ppm): More OH available at higher T, more OH + CH4 -> H2O

---

## 8. Radical Concentrations (Steady-State Predictions)

| Species | 30 C (m-3) | 100 C (m-3) | 180 C (m-3) | 250 C (m-3) |
|---|---|---|---|---|
| O | 4.34e17 | 7.84e17 | 8.51e17 | 5.99e17 |
| OH | 1.50e17 | 1.32e17 | 1.04e17 | 7.57e16 |
| H | 1.82e15 | 4.32e15 | 1.38e16 | 2.61e16 |
| CH3 | 7.30e13 | 2.46e14 | 7.45e14 | 1.42e15 |
| CH2 | 2.98e16 | 3.42e16 | 3.85e16 | 4.35e16 |
| HO2 | 3.02e19 | 5.31e19 | 1.07e20 | 1.69e20 |

**Key observations:**
- **O atom**: Peaks at 180 C (8.5e17), decreases at 250 C as O + CH4 -> CH3 + OH becomes faster
- **OH**: Monotonically decreases (1.5e17 -> 7.6e16) as OH is consumed by CH4 and CO at higher rates
- **H**: Increases 14x (1.8e15 -> 2.6e16) due to enhanced H-abstraction from CH4
- **CH3**: Increases 19x (7.3e13 -> 1.4e15) as CH4 consumption accelerates
- **HO2**: Very high (10^19-10^20 m-3), acting as a radical reservoir (H + O2 + M -> HO2 + M)

---

## 9. Energy Cost

| T (C) | CH4 conv (%) | EC (eV/molecule) | EC (kJ/mol) | SEI (J/L) |
|---|---|---|---|---|
| 30 | 17.59 | 19.8 | 1911 | 676 |
| 100 | 20.49 | 17.0 | 1641 | 549 |
| 180 | 25.62 | 13.6 | 1312 | 452 |
| 250 | 29.83 | 11.7 | 1127 | 392 |

Energy cost decreases from 19.8 to 11.7 eV/molecule as temperature increases. This 41% reduction in EC is the practical benefit of the plasma-thermal synergy: the Arrhenius radical chemistry provides "free" additional conversion using waste heat.

---

## 10. Electron Energy Budget

### 10.1 Absolute Values (eV/m3/s)

| T (C) | P_dep | P_elastic | P_inel | P_diff | P_flow | P_eloss |
|---|---|---|---|---|---|---|
| 30 | +1.95e25 | +2.02e23 | +1.91e25 | +3.75e18 | +4.16e13 | +2.05e23 |
| 100 | +1.95e25 | +2.38e23 | +1.91e25 | +6.16e18 | +7.68e13 | +1.25e23 |
| 180 | +1.95e25 | +2.80e23 | +1.91e25 | +1.00e19 | +1.38e14 | +8.07e22 |
| 250 | +1.95e25 | +2.95e23 | +1.91e25 | +1.38e19 | +1.96e14 | +7.13e22 |

### 10.2 As Fraction of P_dep

| T (C) | P_elastic (%) | P_inel (%) | P_diff (%) | P_flow (%) | P_eloss (%) |
|---|---|---|---|---|---|
| 30 | 1.04 | 97.91 | ~0 | ~0 | 1.05 |
| 100 | 1.22 | 98.14 | ~0 | ~0 | 0.64 |
| 180 | 1.43 | 98.15 | ~0 | ~0 | 0.41 |
| 250 | 1.51 | 98.12 | ~0 | ~0 | 0.37 |

- **~98% of deposited power goes to inelastic collisions** (excitation, dissociation, ionization)
- Elastic loss (gas heating) is only 1-1.5%
- Diffusion and flow losses are negligible (<0.01%)
- Electron destruction loss (DR/AT) is ~1% at 30 C, decreasing to 0.4% at 250 C

---

## 11. Summary of Key Findings

### 11.1 Model Accuracy
- The model correctly captures the temperature trend of CH4 conversion (monotonically increasing)
- Absolute values are over-predicted by 1.5-3.3x (RMSE = 11.51 %p)
- Previous RMSE = 1.14% was an artifact of insufficient convergence time (t_end << tau)
- V_reactor may need to be significantly larger than 250 cm3, or radical-related reaction rates (OH + CH4, N2* + CH4) may need re-evaluation

### 11.2 Plasma-Thermal Synergy
- Even at 30 C, radical chemistry (Arrhenius) accounts for 59.5% of CH4 consumption
- At 250 C, it reaches 81.3%, with OH + CH4 alone contributing 42.6%
- Synergy factor increases from 1.5 to 4.4 with temperature
- EI contribution is roughly constant (~3.3-4.4e-3 mol/m3/s) — temperature sensitivity comes entirely from the radical pathway

### 11.3 CO2 Behavior
- CO2 undergoes a near-closed cycle: EI dissociation (CO2 -> CO + O) followed by regeneration (CO + OH -> CO2 + H)
- Net CO2 conversion is small (0.3-2.4%) compared to throughput (~6e-3 mol/m3/s)
- CO2 conversion increases with temperature because OH is more consumed by CH4 at higher T, reducing CO2 regeneration efficiency

### 11.4 Product Distribution
- CO is the dominant C-product (88-93% selectivity)
- C2+ hydrocarbons are negligible — CH3 radicals are oxidized rather than recombining
- O3 is a significant intermediate at low T (4000 ppm) but thermally decomposes at high T

---

## Appendix: File References

- Analysis script: `/home/hawn/work/analysis_report_0312.py`
- Raw data (JSON): `/home/hawn/work/analysis_results_0312.json`
- Reaction mechanism: `/home/hawn/work/plasma0d_v2/input/reactions.yaml` (207 reactions)
- Species list: `/home/hawn/work/plasma0d_v2/input/species.yaml` (63 species)
- Run command: `source ~/work/.venv/bin/activate && cd ~/work && python analysis_report_0312.py`
