# Reactive-Uptake BC — Ultraplan (v1, SUPERSEDED 2026-04-22)

**Status:** ⚠️ **SUPERSEDED** — PRA α-surface 접근은 driving force diagnostic (2026-04-22)으로 기각. 새 방향 = Finite gas reservoir ODE (CLAUDE.md Session History 2026-04-22 참조).
**이 문서 보존 이유:** (1) Literature survey 45+ citations, (2) K equation audit 결과가 유효, (3) 수학적 동등성 (gas_alpha ≡ 1D gas diffusion) 증명.
**구현 금지:** v1 Phase 0~E는 시작하지 않는다.

**Owner:** HH Kim
**Created:** 2026-04-21
**Superseded:** 2026-04-22

---

## 0. 목적과 철학

**목적:** 현재 `gas_alpha` BC를 **문헌 정합성(α vs γ 구분)**과 **1D-PDE 호환성(bulk 이중계산 제거)**을 동시에 만족하는 canonical reactive-uptake Robin BC로 교체.

**철학 (user confirmed 2026-04-21):**
- NO₃⁻/H₂O₂ 정량 일치를 BC로 강제 맞추지 않는다.
- 물리적으로 올바른 BC를 먼저 구현하고, **잔존 discrepancy는 gas-side input (c_gas(OAS), unmeasured species ratio)의 문제로 honest하게 exposed** 한다.
- η(mass_transfer_eta) 같은 phenomenological knob은 도입하지 않는다.

**Expected outcome:**
- O₃, NO₂: α 문헌 교정으로 kinetic barrier 물리적으로 작동 → surface concentration 현실화.
- N₂O₅/H₂O₂/HONO₂: α 이미 문헌 정합 → BC 교체만으로 MT 거의 불변. 기존 과다 예측은 **c_gas 입력 문제로 재프레임**.
- **Deliverable는 BC 자체의 물리적 정합성이지, fit improvement가 아니다.**

---

## 1. 이론 프레임워크 대조

### 1.1 Schwartz (1986) 직렬 저항 모델 — 지금까지

**원형(Schwartz 1986, NATO ASI):**
```
1/K_total = 1/k_g  +  1/(α·v̄/4)  +  1/(H·β·k_L)
            ────    ─────────────    ──────────────
            기상 film   계면 동역학    액상 film
            (Lewis-     (α = mass     (PDE가 resolve,
            Whitman)    accom. coef.)  본 모델에서 drop)
```

- k_g = D_g / δ_gas — 기상 Fickian film
- α·v̄/4 — Hertz-Knudsen 표면 동역학 (α_s or α_b)
- H·β·k_L — 액상 Fickian film (β = 반응 증강 factor)

**본 연구 채택(2026-04-09 gas_alpha):** 1D PDE가 z=0→L의 확산+반응을 해석하므로 **액상 film 항 drop**:
```
1/k_gi = δ_gas/D_g  +  4/(α·v̄)        [gas-phase units]
k_mt   = k_gi / H_cc                    [liquid-side driving force]
flux   = k_mt · (C_eq − c_surface)      C_eq = H_cc · c_gas(t)
```

구현: `pde_solver.py::compute_k_mt` bc_type='gas_alpha', lines 144-164.

**문헌 계보:**
- Schwartz, S.E. (1986). "Mass-transport considerations pertinent to aqueous phase reactions of gases in liquid-water clouds." In *Chemistry of Multiphase Atmospheric Systems*, NATO ASI, 415–471.
- Lewis, W.K. & Whitman, W.G. (1924). "Principles of gas absorption." *Ind. Eng. Chem.* 16, 1215.
- Hanson, D.R., Ravishankara, A.R., Solomon, S. (1994). "Heterogeneous reactions in sulfuric acid aerosols." *JGR* 99, 3615.

### 1.2 Reactive-Uptake / PRA α-surface 모델 — 제안

**근거:** Pöschl, Rudich & Ammann (2007, *ACP* 7, 5989) **PRA framework**와 IUPAC Task Group (Ammann et al. 2013, *ACP* 13, 8045)의 명시적 지침:

> "γ is a phenomenological parameter … and should not be used as a kinetic input in models that independently represent the processes it lumps, as this leads to double-counting." — Ammann et al. 2013

PRA는 3층으로 분해:
- **α_s**: gas → sorbed surface
- **α_b**: sorbed → bulk liquid  
- **γ (measured)**: net uptake = emergent system output, NOT an input

**1D PDE 모델은 bulk을 명시적으로 해석** → BC에서는 α (surface kinetic)만 사용해야 하고, γ는 시뮬레이션 결과로 **나와야 한다**.

**형식:**
```
-D_l · ∂c/∂z|_{z=0}  =  (v̄·α / 4) · (c_gas(t) − c_surface / H_cc)
```
액상 농도 driving force 변환:
```
k_mt  = v̄·α / (4 · H_cc)                [m/s, liquid-phase units]
flux  = k_mt · (C_eq − c_surface)
```

**Schwartz 형식과의 차이 3가지:**

| 항목 | Schwartz-gas_alpha (현재) | Reactive-uptake (제안) |
|------|--------------------------|----------------------|
| 기상 film 1/k_g | 포함 (δ_gas 의존) | **제거** (stagnant gas에서 비물리적, Silsby 2021) |
| 저항 수 | 2 (gas + interface) | 1 (interface only) |
| α 값의 실질적 기능 | δ_gas가 gas-side를 throttle하여 대부분 종에서 α inert | α가 진짜 knob, kinetic vs diffusion regime 결정 |
| bulk 저항 | PDE가 해결(동일) | PDE가 해결(동일) |
| γ 취급 | 입력으로 혼용 가능(오용 위험) | 출력(시뮬레이션 결과) |

**문헌 계보:**
- Pöschl, U., Rudich, Y., Ammann, M. (2007). "Kinetic model framework for aerosol and cloud surface chemistry and gas-particle interactions — Part 1." *ACP* 7, 5989–6023. (PRA 정식 정의)
- Ammann, M. et al. (2013). "Evaluated kinetic and photochemical data for atmospheric chemistry: Vol VI — heterogeneous reactions with liquid substrates." *ACP* 13, 8045. (IUPAC kinetic database, α vs γ 구분 명시)
- Kolb, C.E. et al. (2010). "An overview of current issues in the uptake of atmospheric trace gases by aerosols and clouds." *ACP* 10, 10561–10605. (α와 γ 혼용을 "가장 빈번한 misuse"로 지적)
- Shiraiwa, M., Pfrang, C., Pöschl, U. (2010). "Kinetic multi-layer model of aerosol surface and bulk chemistry (KM-SUB)." *ACP* 10, 3673. (bulk-resolved 모델에서 α_s·α_b 사용)
- Hanson, D.R. (1997). "Surface-specific reactions on liquids." *J. Phys. Chem. B* 101, 4998.
- Davidovits, P. et al. (2006). "Mass accommodation and chemical reactions at gas-liquid interfaces." *Chem. Rev.* 106, 1323–1354.
- Silsby et al. (2021, *PSST*). "On artificial boundary layers in plasma-liquid transport models." (δ_gas=10mm의 Lewis-Whitman 허구성 지적)
- Zheng, Y., Wang, H., Bruggeman, P.J. (2020). "Modeling of plasma-liquid interactions in He+H₂O RF jet." *JVST A* 38, 063005. (명시적 Robin BC J = α·v̄/4·(c_g − c_l/H), δ_gas 없음)
- Heirman, P. et al. (2025, *J. Phys. D*). α-based Robin BC, DBD 지오메트리에서 Biot 수 기준으로 δ_gas drop 정당화.
- Lindsay, A.D., Graves, D.B., Shannon, S.C. (2016). "Fully coupled plasma-liquid interface simulation." *J. Phys. D* 49, 235204.
- Tian, W. & Kushner, M.J. (2014). *J. Phys. D* 47, 165201. (γ framework, 0D에서 적절)

### 1.3 왜 gas_alpha의 δ_gas=10mm가 비물리적인가

sDBD의 quiescent gas 조건에서:
- 기상 확산 시간상수: τ ~ δ²/D_g = (0.01)²/1.5e-5 = **6.7 초**
- 시뮬레이션 시간: 600 s (plasma treatment 10min)
- → 기상 전체가 수초 내 well-mixed, "10mm film" 개념이 성립하지 않음.

δ_gas=10mm는 Lewis-Whitman 이론의 "fictitious stagnant film" 그대로 차용한 값으로, **대류가 있을 때만 Sherwood 수로 물리적 의미를 갖는다** (Cussler 2009, Bird-Stewart-Lightfoot 2007). sDBD에서는 대류가 약하므로 해당 안 됨.

**결론:** δ_gas term을 drop하는 것이 물리적으로 맞다. 그 대신 c_gas는 OAS 측정값을 chamber-averaged bulk gas로 사용하고, 필요 시 나중에 Sherwood-기반 film을 **Bird-Stewart-Lightfoot 형식**으로 optional 추가 (Phase E).

---

## 2. 현재 구현의 4가지 pathology 진단

### 2.1 α vs γ 혼동 (notes/alpha_b_literature.md)

기존 compilation이 문헌의 "recommended" value를 그대로 가져왔으나 **일부는 γ 기반 측정** (특히 오래된 droplet-train experiment). 교정 필요:

| 종 | 현재 α (config_1d.py:149-159) | 문헌 α (primary) | 비고 |
|----|------------------------------|------------------|------|
| N₂O₅ | 0.03 | 0.03 (Van Doren 1990; γ≈α, 가수분해 지배) | OK |
| **O₃** | **0.05 (Davidovits 2006 과대)** | **1e-3 (Utter 1992), 5e-4 (Magi 1997)** | **50× 과대** |
| H₂O₂ | 0.1 | 0.18 (Worsnop 1989) / 0.11 (Magi 1997) | 약간 과소, OK |
| NO | 0.001 | <5e-5 (상한, Saastad 1993) | poorly constrained |
| **NO₂** | **0.03 (Bruggeman 2016 misread)** | **2e-4 (Cheung 2000, IUPAC)** | **~150× 과대** |
| NO₃ | 0.03 (assumption) | 문헌 측정 없음 (NO₂ 유사 추정) | — |
| N₂O₄ | 0.03 (assumption) | 가수분해 빠름 (NO₂ 유사) | — |
| HONO | 0.05 | 0.04 (Bongartz 1994) | OK |
| HNO₃ | 0.07 | 0.07 (Davidovits 2006, JPL) | OK |

**영향 큰 교정:** O₃ (50×), NO₂ (~150×). 나머지는 이미 문헌 정합.

### 2.2 δ_gas throttle → α dead knob

`gas_alpha` 계산 검증 (O₃, α=0.05 기준):
- v̄(O₃) = √(8RT/πM) = 364 m/s
- α·v̄/4 = 4.55 m/s (계면 속도)
- D_g/δ_gas = 1.5e-5/0.01 = 1.5e-3 m/s (기상 film 속도)
- 합성 k_gi = 1/(1/1.5e-3 + 1/4.55) ≈ 1.5e-3 m/s ← **기상 film이 지배**

α 변화의 영향:
- α=0.05: k_gi = 1.50e-3, k_mt_liq = 6.5e-3 m/s
- α=1e-3: k_gi = 1.48e-3, k_mt_liq = 6.5e-3 m/s (거의 동일!)
- **결론:** δ_gas=10mm 하에서 α는 전혀 작동하지 않는다. 현재 "gas_alpha" BC는 이름과 달리 실질적으로 "one_film_gas" BC와 동일.

### 2.3 kinetic vs Dirichlet regime — 의외의 결과

dz_min = 2e-9 m (Debye resolution, config_1d.py:178):
- D_l/dz_min = 1.75e-9/2e-9 = 0.875 m/s
- 현재 k_mt = 6.5e-3 m/s << D_l/dz_min → **kinetic-limited** (c_surf << C_eq)

즉 현재 BC는 Dirichlet regime이 아니라 **심하게 kinetic-limited** 상태이며, 이는 δ_gas가 과도한 throttle로 작용하여 발생. 이것이 "MT 과다"라는 user의 진단과 **겉보기엔 모순**되지만, 실제로는:

- O₃/NO₂: kinetic-limited이지만 α 값이 너무 높아(0.05/0.03) 여전히 과잉 유입
- H₂O₂/HONO₂: α가 크고 H_cc가 매우 커서 C_eq 자체가 거대 → 아무리 kinetic-limited여도 flux가 큼
- **진짜 원인:** α 값(O₃/NO₂) + c_gas · H_cc(H₂O₂/HONO₂/NO₃⁻). **BC 구조보다 coefficient 값이 지배.**

### 2.4 γ (reactive uptake) literature values은 PDE에 입력으로 쓰면 안 됨

문헌의 γ_O₃ ≈ 1e-5~1e-7 (pure water, Utter 1992; Magi 1997). 만약 이를 BC에 직접 쓰면:
- k_mt = v̄·γ/(4·H_cc) = 364·1e-5/(4·0.23) = 3.96e-3 m/s (현재보다 낮음)

그러나 γ는 이미 **bulk saturation + 확산-반응 coupling**을 lump한 값. PDE가 이를 재해석하면 **double-counting**. Ammann 2013 명시.

→ 올바른 선택: **α (surface-only) 사용**. γ가 더 작다고 γ를 쓰는 것은 잘못된 fit-by-underestimation.

---

## 3. Phased Implementation Plan

### Phase 0 — Regime Diagnosis (예상 1 세션, 코드 변경 없음)

**목적:** 현재 BC의 실제 regime을 숫자로 확인, Phase A-B의 출발점 문서화.

**Tasks:**
1. `Ver4_1D/diagnose_bc_regime.py` 신규 작성:
   - 각 gas-to-aq 종별로 다음 출력:
     - k_mt_liq (현재 gas_alpha)
     - k_mt_liq (reactive_uptake with current α)
     - k_mt_liq (reactive_uptake with corrected α)
     - D_l/dz_min (diffusion rate at surface cell)
     - k_mt·dz_min/D_l (Biot-like number)
     - C_eq_peak(t) (from OAS)
   - kinetic vs diffusion-limited regime 판정
2. DIW 3.2kV baseline 1 run으로 `c_surface(t) / C_eq(t)` 시계열 추출:
   - 종별 saturation ratio 0.1초 해상도로 저장
   - Plot: `Figures/diag_surface_saturation.pdf`
3. α microsweep (O₃만, 5개 값: 1e-6, 1e-4, 0.01, 0.05, 0.5):
   - 현재 gas_alpha와 reactive_uptake 각각에서
   - O₃ bulk-averaged time series 비교

**Output:** `notes/bc_regime_diagnosis.md`. 경험적 근거로 Phase B 전환 정당화.

**Skip 기준:** 이 문서의 §2에서 이미 해석적으로 검증됨. paper용 figure로 값어치 있지만, **원하면 skip 가능**.

### Phase A — Literature Audit + Documentation (0.5 세션)

**Tasks:**
1. `notes/alpha_b_literature.md` → `notes/uptake_coefficients.md` 로 교체:
   - Section 1: α vs α_s vs α_b vs γ 정식 구분 (Pöschl 2007, Ammann 2013)
   - Section 2: 1D-PDE에서 왜 α 사용해야 하는가 (Ammann 2013 quote)
   - Section 3: 종별 α 교정 table (위 §2.1)
   - Section 4: γ_literature values 참조용 (시뮬 출력과 비교할 target)
   - Section 5: 어느 값이 기존과 다른지, 교정 근거
2. `notes/bc_formulation.md` → `notes/bc_formulation_v2.md` 로 승계:
   - Section 1: 이전 gas_alpha 요약 + pathology 4건
   - Section 2: 제안 reactive_uptake 수식 + 유도
   - Section 3: Schwartz vs PRA 대조표 (본 문서 §1.1-1.2 재수록)
   - Section 4: δ_gas drop 정당화 (Silsby 2021, Heirman 2025, Biot 수 해석)
   - Section 5: γ를 출력으로 취급 (시뮬 후 post-hoc 계산 예시 코드)

**Commit:** `docs: α vs γ distinction and reactive-uptake framework notes`

### Phase B — Code Implementation (1-2 세션)

**Tasks:**
1. `Ver4_1D/config_1d.py`:
   - `MassTransfer1DConfig`:
     - `bc_type` default 'two_film' → 'reactive_uptake' (migration 기간 후)
     - `delta_x_gas` 기본값 유지(back-compat), 문서에 reactive_uptake에서 무시됨 명시
     - `alpha_b_species` 값 교정 (O₃: 0.05→1e-3, NO₂: 0.03→2e-4, HNO₃: 0.07 유지)
     - docstring에 "mass accommodation α (not γ); for 1D PDE the interface-only coefficient" 명시
   - HENRY_CONSTANTS 주석에 "dimensionless H_cc = H_cp·R·T (Sander 2023)" 확인 (이미 있음)

2. `Ver4_1D/pde_solver.py`:
   - `compute_k_mt()` (line 114-172):
     - 새 branch `bc_type == 'reactive_uptake'`:
       ```
       H_cc = HENRY_CONSTANTS.get(species_gas, 1.0)
       v_thermal = √(8RT/πM)  # M in kg/mol
       k_int = alpha_b × v_thermal / 4   # gas-phase units
       k_mt_liq = k_int / max(H_cc, 1e-30)
       return k_mt_liq
       ```
     - 기존 `gas_alpha`, `one_film_gas` branch는 legacy로 유지 (BC 비교 실험용)
     - docstring 업데이트: reactive_uptake 설명 + Pöschl 2007 인용
   - `apply_bc` / rhs는 수정 불필요 (flux 공식 동일: k_mt × (C_eq − c_surf))
   - `mass_transfer_eta` 정리: dead code 여부 확인 후 유지 또는 제거 결정

3. `Ver4_1D/run_bc_comparison.py` 업데이트:
   - BC_CASES에 'reactive_uptake' 추가
   - 3 legacy (Dirichlet/gas_alpha/one_film_gas) + 1 new (reactive_uptake) 비교
   - Species별 time series + ss values CSV 저장

4. 테스트:
   - `Ver4_1D/test_bc_reactive_uptake.py` (pytest, @pytest.mark.unit + integration)
     - Unit: k_mt formula 수치 정확성 (O₃: v̄=364, α=1e-3, H_cc=0.2298 → k_mt=0.396 m/s)
     - Unit: α→0 극한에서 k_mt→0
     - Unit: α→1, H_cc→1 극한에서 k_mt = v̄/4 = 91 m/s (O₃)
     - Integration: single-species uptake 600s run → c_surface → C_eq (saturation)
     - Integration: 질량보존 (bulk integral = ∫flux dt)
   - 기존 `test_verification.py` 10개 테스트 전부 pass 유지

**Commit:** `feat: reactive-uptake BC with PRA α-surface framework`

### Phase C — Validation Runs (1-2 세션)

**Tasks:**
1. DIW 3 voltage × 4 BC (Dirichlet, gas_alpha, one_film_gas, reactive_uptake):
   - 600 s, Humid fitting gas input, δ_gas=10mm (gas_alpha용) 유지
   - 12 simulations
   - Output CSV: `Ver4_1D/results/bc_comparison_diw/`

2. Figure: `Figures/fig_bc_reactive_uptake.pdf`:
   - Panel (a): Surface concentration time series, 6 species × 4 BC (3.2kV)
   - Panel (b): Bulk-averaged concentration, 6 species × 4 BC (3.2kV)
   - Panel (c): Voltage scan — final values, 4 BC × 3 voltage for NO₃⁻, H₂O₂, pH
   - Panel (d): **γ_effective = J_net / (v̄/4 · c_gas)** 시계열 — 시뮬레이션 output vs 문헌 γ target

3. Acceptance criteria (BC 정합성, not fit):
   - (i) 모든 솔버 converge, atol=1e-15 유지, step count < 2× baseline
   - (ii) 질량보존: ∫flux dt = [bulk integral] ± 1% (각 종)
   - (iii) Saturation test: 반응 제거 시 c_surface → C_eq 이내 5%
   - (iv) γ_eff(O₃) simulation output ∈ [1e-7, 1e-4] (문헌 range, γ_literature 1e-5~1e-7 포함)
   - (v) γ_eff(N₂O₅) ≈ α_N₂O₅ = 0.03 (가수분해-지배 극한)

4. Fit diagnostics (honestly reported, not acceptance):
   - NO₃⁻ over-prediction factor 변화 (expected: 비슷하거나 약간 증가)
   - H₂O₂ over-prediction factor 변화 (expected: 거의 불변)
   - O₃ surface concentration 변화 (expected: 대폭 감소 — kinetic barrier 작동)
   - NO₂ 관련 species (NO₂⁻ path) 변화 (expected: NO₂ uptake 감소)

**Output:** `notes/bc_validation_results.md`. 결과와 해석. Fit 악화가 발생해도 정당한 이유 기록.

**Commit:** `test: reactive-uptake BC validation runs and comparison figures`

### Phase D — Saline Validation (1 세션)

**Tasks:**
1. Saline 6 cases (2.6/3.2/3.6 kV × 3min/10min) reactive_uptake로 재실행
2. DIW와 동일 acceptance criteria
3. Cl⁻ conservation 확인 (saline-특이적 regression)
4. Saline 특유의 γ 조정 (ClNO₂ path via N₂O₅+Cl⁻) — 현 플랜에서는 α 유지, γ는 emergent

**Output:** `notes/bc_validation_saline.md`

**Commit:** `test: saline validation with reactive-uptake BC`

### Phase E — Optional Enhancements (조건부, 별도 승인)

Phase C/D에서 구조적 mismatch가 드러날 때만:

**E1. Sherwood-number gas-side film:**
- forced convection 있을 때 k_g = Sh·D_g/L, Sh = Ranz-Marshall 또는 Frössling
- sDBD quiescent gas에서는 Sh ≈ 2 (구형 한계) → k_g ~ 2·D_g/L_gap
- 현재 L_gap=10mm에서 k_g = 3e-3 m/s — 여전히 limiting이 아님을 재확인

**E2. Finite gas-reservoir model:**
- OAS는 chamber-bulk 측정값. Interface 경계층 depletion을 별도 변수로:
  ```
  dc_gas_surface/dt = Q_inflow·(c_gas_bulk − c_gas_surface)/V_BL − flux·A/V_BL
  ```
- BL = boundary layer (~1-3mm), Q_inflow = convective replenishment
- Phase C에서 γ_eff(시뮬) >> γ_literature로 드러나면 도입 검토

**E3. Surface reaction (Pöschl α_s → α_b 2층):**
- 현재 α_b만 사용. α_s 추가는 KM-SUB full framework 이전.
- 현 단계에서는 불필요 (α_s ≥ α_b ≥ γ로 α_b만 사용해도 surface-limited flux 정확).

---

## 4. File-level 변경 리스트 (Phase B 기준)

| File | Lines (approx) | Change | Risk |
|------|---------------|--------|------|
| `Ver4_1D/config_1d.py` | 142, 149-159 | bc_type default, α 값 교정 | Low |
| `Ver4_1D/pde_solver.py` | 114-172 | `compute_k_mt` reactive_uptake branch 추가 | Low (legacy 유지) |
| `Ver4_1D/run_bc_comparison.py` | 전체 | BC_CASES 업데이트 | Low |
| `Ver4_1D/test_bc_reactive_uptake.py` | 신규 | 200-300줄 | - |
| `notes/uptake_coefficients.md` | 신규 | ~300줄 | - |
| `notes/bc_formulation_v2.md` | 신규 | ~200줄 | - |
| `notes/reactive_uptake_bc_plan.md` | 본 파일 | - | - |
| `notes/bc_regime_diagnosis.md` | 신규 (Phase 0) | ~150줄 | - |
| `notes/bc_validation_results.md` | 신규 (Phase C) | ~200줄 | - |
| `Figures/gen_fig_bc_reactive_uptake.py` | 신규 | ~250줄 | - |
| `Figures/fig_bc_reactive_uptake.pdf` | 신규 | — | - |
| `CLAUDE.md` | Session History, Key Decisions, WORKING | Phase별 append | - |
| `memory/MEMORY.md` + file | new entry `reactive_uptake_bc` | - | - |

**삭제 대상:** 없음. `bc_formulation.md` → v2로 supersede만.
**Deprecation:** `gas_alpha`, `one_film_gas`는 코드상 유지하되 문서에서 "legacy, use reactive_uptake" 명시.

---

## 5. Validation Protocol 요약

**Level 1 — Solver integrity (필수):**
- 10개 기존 verification test pass
- atol=1e-15, rtol=1e-6 유지
- 600s run 시간 증가 < 2×

**Level 2 — Physical consistency (필수):**
- N atom 보존 (∫flux − bulk_integral < 1%)
- Saturation convergence (반응 off 시 c_surf → C_eq)
- γ_eff(emergent) 문헌 range 이내

**Level 3 — Diagnostic (reporting only, acceptance 아님):**
- 종별 over/under-prediction factor (DIW + Saline)
- Voltage-dependent trend (reactive_uptake가 실험의 voltage-dependence 재현하는지)
- O₃/NO₂ surface concentration 현실성

**Pass condition:** Level 1 & 2 전부 pass. Level 3는 discussion section용.

---

## 6. Risk & Fallback

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Solver stiffness 증가 (reactive_uptake로 k_mt↑) | Medium | Jacobian sparsity 유지, atol/rtol 재튜닝. 이미 기존 atol=1e-15에서 안정함 확인. |
| O₃/NO₂ α 교정으로 radical chain 붕괴 | Low-Medium | Phase C에서 O₃ bulk 크게 감소 시, radical chain(R98 N₂O₅+H₂O) 의존도 확인. NO₃⁻ 생성 메커니즘은 독립적(α_N₂O₅ 불변)이므로 영향 제한적. |
| NO₃⁻ over-prediction 악화 (+) | Medium | **의도된 honest 결과.** Discussion에서 gas-side input(c_gas·H_cc)를 root cause로 재프레임. |
| H₂O₂ ratio input(0.03)이 여전히 불일치 | High | Phase F(별도)에서 gas-side input 재검토: O₃×0.03 → 0.01 혹은 독립 측정 필요. |
| γ_eff(O₃) > 1e-4 로 나와 문헌 범위 초과 | Low | 주로 α 값 문제. literature 하한 3e-4로 조정 검토. |

**Rollback:** 각 phase 별도 commit, revert 가능. `gas_alpha` 구현은 code에 유지되므로 `bc_type` 변경만으로 복원.

---

## 7. Open Questions (승인 전)

1. **Phase 0 skip 여부?** 해석적 검증으로 Phase B 바로 가도 되나, paper figure로 값어치가 있음. → User 결정.
2. **α(O₃) 교정값:** Utter 1992(1e-3) vs Magi 1997(5e-4). 보수적으로 1e-3 권장(flux 더 보존적). → User 결정.
3. **α(NO₂) 교정값:** Cheung 2000 range 1e-4~6e-4. 중간값 2e-4 권장. → User 결정.
4. **α(NO₃, N₂O₄) assumption:** 측정 없음. NO₂와 동일 가정 유지 or N₂O₅와 동일? → User 결정.
5. **bc_type default switch:** `two_film` → `reactive_uptake` 즉시 전환 vs migration 기간 `gas_alpha` → `reactive_uptake`? → User 결정.
6. **H₂O₂ α:** 문헌 0.18 vs 현재 0.1. 온도 의존성 강함(298K에서 0.08-0.11). 현재 0.1 유지 가능. → User 결정.

---

## 8. Commit 순서 (최종)

1. Phase 0: `docs: BC regime diagnosis — gas_alpha pathology analysis`
2. Phase A: `docs: α vs γ distinction and reactive-uptake framework`
3. Phase B-1: `refactor: add reactive_uptake BC branch in compute_k_mt`
4. Phase B-2: `feat: correct α values to primary literature (O3, NO2)`
5. Phase B-3: `test: unit+integration tests for reactive_uptake BC`
6. Phase C: `test: DIW 3-voltage × 4-BC comparison runs`
7. Phase D: `test: saline validation with reactive-uptake`
8. (final): `docs: update CLAUDE.md Session History + memory for BC refactor`

**No force-push to main.** Each phase independently reviewable.

---

## 9. 승인 후 바로 시작 순서

User 승인 시:
1. §7 Open Questions 답변 받기 (특히 α 값 5가지)
2. Phase 0 수행 여부 결정
3. Phase 0 skip이면 → Phase A (literature audit 문서) 시작
4. Phase A 완료 → Phase B 코드 구현

---

## 10. 인용 (Selected; full list in uptake_coefficients.md Phase A)

### α/γ Framework
- Schwartz, S.E. (1986). *NATO ASI* — original resistor model.
- Pöschl, U., Rudich, Y., Ammann, M. (2007). *ACP* 7, 5989.
- Ammann, M. et al. (2013). *ACP* 13, 8045. IUPAC.
- Kolb, C.E. et al. (2010). *ACP* 10, 10561.
- Davidovits, P. et al. (2006). *Chem. Rev.* 106, 1323.
- Shiraiwa, M. et al. (2010). *ACP* 10, 3673. KM-SUB.
- Hanson, D.R. (1997). *J. Phys. Chem. B* 101, 4998.

### α values (primary measurements)
- Utter, R.G. et al. (1992). *J. Phys. Chem.* 96, 4973. (O₃)
- Magi, L. et al. (1997). *J. Phys. Chem. A* 101, 4943. (O₃, H₂O₂)
- Worsnop, D.R. et al. (1989). *J. Phys. Chem.* 93, 1159. (H₂O₂, SO₂)
- Van Doren, J.M. et al. (1990). *J. Phys. Chem.* 94, 3265. (HNO₃, HCl, N₂O₅)
- Kirchner, W. et al. (1990). *J. Atmos. Chem.* 10, 427. (NO₂)
- Cheung, J.L. et al. (2000). *J. Phys. Chem. A* 104, 2655. (NO₂)
- George, C. et al. (1994). *J. Phys. Chem.* 98, 8780. (N₂O₅)
- Schweitzer, F. et al. (1998). *J. Phys. Chem. A* 102, 3942. (N₂O₅)
- Bongartz, A. et al. (1994). *J. Atmos. Chem.* 18, 149. (HONO)
- Hanson, D.R. et al. (1992). *J. Phys. Chem.* 96, 4979. (OH)
- Takami, A. et al. (1998). *Chem. Phys.* 231, 215. (OH)
- Saastad, O.W. et al. (1993). *GRL* 20, 1191. (NO upper bound)

### Henry constants
- Sander, R. (2023). *ACP* 23, 10901. Authoritative compilation.

### Plasma-liquid context
- Bruggeman, P.J. et al. (2016). *PSST* 25, 053002. Review.
- Liu, D.X. et al. (2015–2016). *PSST*/*J. Phys. D* series.
- Lindsay, A.D. et al. (2015, 2016). *J. Phys. D* 48, 424007 / 49, 235204.
- Zheng, Y., Wang, H., Bruggeman, P.J. (2020). *JVST A* 38, 063005. Explicit Robin BC.
- Heirman, P. et al. (2025). *J. Phys. D* 58, 085206.
- Silsby et al. (2021). *PSST*. δ_gas critique.
- Tian, W. & Kushner, M.J. (2014). *J. Phys. D* 47, 165201. γ framework (0D).

### Transport theory
- Lewis, W.K. & Whitman, W.G. (1924). *Ind. Eng. Chem.* 16, 1215.
- Bird, R.B., Stewart, W.E., Lightfoot, E.N. (2007). *Transport Phenomena*, 2nd ed.
- Cussler, E.L. (2009). *Diffusion*, 3rd ed.

---

<!-- Last updated: 2026-04-21 (draft v1) -->
