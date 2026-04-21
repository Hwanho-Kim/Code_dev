# Ultraplan — TPA/hTPA 기반 OH 라디칼 정량 비교

작성일: 2026-04-20
대상 실험: `Article/20260417_weekly meeting (1).pptx` (TPA → hTPA fluorescence, 10 min, 2.6/3.2/3.6 kVpp)
대상 시뮬레이션: `Ver4_1D/` (Monolithic BDF, gas_alpha BC, atol=1e-12)

---

## 1. 목표
1D plasma-liquid 시뮬레이션에서 **TPA/hTPA 경로로 누적되는 OH 농도**를 계산하여, 실험에서 측정한 `[OH]_total = [hTPA]/0.35` 값과 **정량 비교**한다. 3개 전압(2.6 / 3.2 / 3.6 kVpp)에서 측정된 비단조성 (3.2 > 3.6 > 2.6)의 재현성과 물리적 원인까지 규명한다.

### 실험 기준값 (inner filter 1/2 희석 보정 후)
| 전압 | [hTPA]ₘ (μM) | [OH]_total = hTPA/0.35 (μM) | pH |
|---|---|---|---|
| 2.6 kVpp | 12.66 | 36.2 | >10 |
| **3.2 kVpp** | **57.72** | **165** | >10 |
| 3.6 kVpp | 43.26 | 124 | >10 (최저) |

---

## 2. AS-IS (현재 시뮬레이션 능력)

| 항목 | 상태 |
|---|---|
| 솔버 | Monolithic BDF (Strang 대체), Numba, atol=1e-12, rtol=1e-6 |
| BC | **gas_alpha** (1/k_gi=δ_gas/D_g + 4/(α_b·v̄)), δ_gas=10 mm |
| 반응 | 193 aqueous, 47 species, `reactions_full.yaml` |
| 검증 조건 | **DIW** (pH 3~5), **Saline** (NaCl 0.154 M, pH 3~5) |
| 전압별 데이터 | OAS data/Dry: O₃/NO₂/NO₃/N₂O₅ 시계열 (2.6/3.2/3.6 kVpp, 0–600 s) |
| 알칼리 조건 | **미검증** (pH~11.5 시나리오 없음) |
| TPA/hTPA | **없음** (종/반응 모두 부재) |
| OH 수준 (기존) | 5–8 pM (매우 낮음 — 강한 scavenger 없음) |

---

## 3. GAP 분석 — 정량 비교를 위해 필요한 것

### 3.1 화학
- **TPA²⁻ + OH → hTPA²⁻** (branching ≈ 0.35), k = 4.0×10⁹ M⁻¹s⁻¹ (Matthews 1980, Fang 1996)
- **TPA²⁻ + OH → 부산물** (1 − 0.35 = 0.65), 동일 k
- **hTPA²⁻ + OH → 분해** (2차 산화, k ≈ 1×10⁹, 10 min 동안 [hTPA]≪[TPA]이라 제한적 영향)
- TPA pKa₁=3.51, pKa₂=4.82 → pH 11.5에서 **완전 2가음이온(TPA²⁻)**. 시뮬레이션에선 단일 species로 근사 가능.
- **공기 CO₂ 침투에 의한 HCO₃⁻/CO₃²⁻** (~10–100 μM)는 OH scavenger (k=8.5×10⁶). 민감도 확인 필요.

### 3.2 물리 조건 차이 (DIW → TPA 알칼리)
| 항목 | DIW (기존) | TPA 실험 |
|---|---|---|
| pH | 5.09 → 3.61 (산성화) | 11.5 유지 (NaOH 완충) |
| 주 OH scavenger | 없음 | **TPA (2 mM)**, k[TPA] ≈ 8×10⁶ s⁻¹ |
| O₃ 반응 | O₃ + NO₂⁻ 등 | **O₃ + OH⁻ (R22/R23)** 지배 (k=48+70 M⁻¹s⁻¹ × [OH⁻]=3×10⁻³ → 0.35 s⁻¹) |
| N₂O₅ 가수분해 | 빠름 | **H⁺ 극소 + NO₃⁻/NO₂⁻ 축적 허용** |

### 3.3 측정 모델
- 실험은 **bulk 용액 2 mL 취해 96well plate로 형광 측정** → 시뮬레이션에선 **공간평균 [hTPA]** 사용.
- 실험 처리 부피(2 mL)/노출 면적이 기존 petri 설정과 다름 → **체적/표면적 비율(S/V)** 보정 필요. (1D는 liquid_depth로 흡수 — δ_liq만 실험 chamber 깊이와 매칭하면 됨.)

---

## 4. 상세 Phase 계획

### Phase 0 — 문헌/현황 조사 (예상 1 day)

**목적**: 구현 들어가기 전 파라미터 범위와 경쟁 반응을 확정.

**Action items**
1. **TPA-OH rate/branching 문헌** (3개 이상 교차 검증)
   - Matthews (1980), Fang et al. (1996, Free Radical Biology), Saran & Summer (1999), Page et al. (2010 J Environ Monit — PPT 인용), Charbouillot et al. (2011 J Photochem A).
   - `k_TPA+OH ∈ [3.3, 5.0] × 10⁹ M⁻¹s⁻¹`, `branching(→hTPA) ∈ [0.28, 0.35]` 예상.
2. **hTPA 2차 반응**: hTPA + OH → 분해, k ≈ 1×10⁹? Saran & Summer, Fang.
3. **알칼리 pH=11.5 O₃ 메커니즘**:
   - R22 (O₃+OH⁻→O₂+HO₂⁻) k=48 M⁻¹s⁻¹ → HO₂⁻ 축적
   - R23 (O₃+OH⁻→O₂⁻+HO₂) k=70 M⁻¹s⁻¹
   - **R26 (O₃+HO₂⁻→O₂⁻+O₂+OH) k=2.8×10⁶** → 알칼리의 주 OH 공급원 (autocatalytic chain).
   - 이미 `reactions_full.yaml`에 존재함. 추가 반응 필요 없음.
4. **Carbonate scavenging**: `CO₃²⁻ + OH → CO₃·⁻ + OH⁻` k=3.9×10⁸. 대기 개방 여부에 따라 on/off.
5. **OAS data 매칭 확인**: 3.2 kVpp 가스상 데이터와 동일 chamber에서 TPA 측정했는지 확인. 동일 장치 가정.

**Deliverable**: `notes/tpa_chemistry_literature.md` (참고문헌 표 + 선택된 k, branching, 불확실도 범위).

---

### Phase 1 — 반응 메커니즘 확장 (1–2 days)

**File 변경**
- `config_1d.py`
  - `AQUEOUS_SPECIES`에 `'TPA'`, `'hTPA'` 추가 (전하는 암묵적으로 2⁻ — charge balance에는 명시 Z=-2 필요. `SPECIES_CHARGE['TPA'] = -2, SPECIES_CHARGE['hTPA'] = -2`).
  - `'Na+'` species 추가 (charge balance 유지용). `SPECIES_CHARGE['Na+'] = +1`.
  - `LIQUID_DIFFUSIVITY`에 `'TPA': 7.5e-10, 'hTPA': 7.0e-10` (Stokes-Einstein 추정).
  - (optional) carbonate: `CO3--`, `HCO3-` — 민감도에 따라 결정.
  - TRANSFERABLE_SPECIES / GAS_TO_AQUEOUS_MAP: **미수정** (비휘발성).
- `reactions_full.yaml` 또는 신규 `reactions_tpa.yaml`
  ```yaml
  - id: R_TPA1
    reactants: {TPA: 1, OH: 1}
    products:  {hTPA: 1}
    k: 1.4e9        # 0.35 × 4.0e9
    type: irr
    label: "TPA + OH → hTPA (branching 0.35)"
  - id: R_TPA2
    reactants: {TPA: 1, OH: 1}
    products:  {}            # "inert" product 소멸
    k: 2.6e9        # 0.65 × 4.0e9
    type: irr
    label: "TPA + OH → 비형광 부산물"
  - id: R_TPA3
    reactants: {hTPA: 1, OH: 1}
    products:  {}
    k: 1.0e9
    type: irr
    label: "hTPA + OH → 분해"
  ```
- `chemistry_1d.py`: `AQUEOUS_SPECIES` 길이 변경에 따른 인덱스 자동 처리 확인. 충전량 행렬 업데이트.

**Test**
- `test_verification.py`에 TPA 포함 mass balance unit test 1개 추가 (초기 2 mM, OH 주입 시 hTPA 생성 Σ = 0.35×TPA_소모).
- Regression: 기존 DIW run 결과 불변 확인 (TPA=0 초기조건이면 반응 0).

---

### Phase 2 — 알칼리 TPA 시나리오 (1 day)

**File 변경**
- 새 runner: `Ver4_1D/run_tpa_alkaline.py` (base: `run_saline_1d.py`)
  - 초기조건 override:
    ```python
    C0['TPA']   = 2.0e-3      # 2 mM
    C0['Na+']   = 10.0e-3     # charge balance for NaOH + TA²⁻ 고려
    C0['OH-']   = 10.0e-3     # 10 mM NaOH 초기 → pH≈12. TA 해리로 ~11.5 안정.
    C0['H+']    = 1.0e-12
    C0['hTPA']  = 0.0
    ```
  - 실제 pH=11.5는 NaOH 10 mM + TA 2 mM 중화 후 평형. 초기 [OH⁻]=6 mM 정도가 되도록 수동 세팅 또는 charge balance로 자동 풀기 (기존 `_enforce_electroneutrality`).
- `experimental_data.yaml`에 `tpa_alkaline` 섹션 추가 (3개 전압 실험값).

**Validation checkpoint**
- 반응 OFF로 돌려서 **pH가 11.5 유지, Σ charge=0, 수치 안정** 확인.

---

### Phase 3 — 전압별 가스상 입력 매칭 (1 day)

- `OAS data/Dry/(P-L) 가스활성종 농도.xlsx` 로드 (CLAUDE.md에 경로 확인됨).
- 3개 시트/열: 2.6 / 3.2 / 3.6 kVpp, 0–600 s, 2 s 간격, 301 points, 종 = O₃/NO₂/NO₃/N₂O₅.
- 기존 onset filter + linear interpolation + 비측정종 비율 (HONO/NO₂=0.33, HNO₃/N₂O₅=0.83, H₂O₂/O₃=0.03) 재사용.
- **주의**: 이 비율은 DIW에서 fit된 값. TPA/알칼리 조건에서도 동일 가스상 입력이라 가정 (같은 plasma, 같은 chamber, liquid만 다름).
- Output: `gas_input_2p6.csv`, `gas_input_3p2.csv`, `gas_input_3p6.csv`.

---

### Phase 4 — 시뮬레이션 실행 (반나절)

| # | 전압 | TPA | Carbonate | 비고 |
|---|---|---|---|---|
| 1 | 2.6 kVpp | 2 mM | 0 | main |
| 2 | 3.2 kVpp | 2 mM | 0 | main |
| 3 | 3.6 kVpp | 2 mM | 0 | main |
| 4 | 3.2 kVpp | 0 | 0 | **control**: TPA 없을 때 OH 수준·H₂O₂ 비교 |
| 5 | 3.2 kVpp | 2 mM | 100 μM | carbonate 민감도 |
| 6 | 3.2 kVpp | 2 mM | 10 μM | carbonate 민감도 |

**설정**: gas_alpha BC, δ_gas=10 mm (기본), α_b 종별, Monolithic BDF, t_end=600 s, dt_enforce=None.

**출력 (run당)**:
- `[hTPA]_bulk_avg(t)`, `[OH](z, t)`, `[TPA](z, t)`, `pH(z, t)`.
- 파생량: `cumulative_OH_scavenged_by_TPA = 0.35⁻¹ · [hTPA]_bulk(600)` (이론상 [OH]_total 측정량).
- 개별 반응 기여도 (Simpson 적분 기반, `gen_fig2_rate_evolution.py` 재활용).

---

### Phase 5 — 비교 분석 (1–2 days)

#### 5-A. 정량 매칭 지표

| 지표 | 목표 정확도 |
|---|---|
| `[hTPA]_sim(600 s, bulk avg)` vs 실험 | ±30% |
| 전압 rank (3.2>3.6>2.6) 재현 | Must |
| pH 말미 | ≥ 10 |
| 2.6/3.2 비율 | ~4.5× (실험), ±40% 이내 |

#### 5-B. 비단조성 (3.2 > 3.6) 후보 원인 진단

실험은 여러 가설을 제시했는데, 시뮬레이션이 어느 쪽을 지지하는지 확인한다.

1. **Gas input 자체의 비단조성** → OAS 데이터에서 3.6 > 3.2가 O₃/N₂O₅ 모두 성립한다면, 시뮬레이션도 3.6 > 3.2 나와야 함. 이 경우 **측정 artifact (inner filter / H₂O₂ 경쟁)가 지배**.
2. **pH 저하 (3.6 최저)**: 알칼리 완충이 더 빨리 소진 → O₃+OH⁻ 경로 약화 → OH 생성 저하. 시뮬레이션에서 pH 시계열 비교.
3. **H₂O₂ 축적 → OH + HO₂⁻ 재결합**: TPA 포획 이전에 OH가 HO₂⁻/H₂O₂와 반응. `R27 (OH+HO2)`, `R29 (OH+H2O2)` 기여도 비교.
4. **TPA 고갈**: 3.6 kV에서 10분간 TPA가 거의 다 소모되면 후반 OH 비포획 → [hTPA] 감소 플래토. `[TPA](t=600)` 확인.

**진단 sweep** (불일치 시):
- α_b (O₃, H₂O₂): 0.5× / 1× / 2×
- δ_gas: 5 / 10 / 20 mm
- k_TPA+OH: 3.3 / 4.0 / 5.0 × 10⁹
- branching: 0.28 / 0.35
- gas input HONO/H₂O₂ 비율: 0.5× / 1× / 2×

#### 5-C. Control 비교
- TPA 없을 때 `[OH]_peak`, `[H₂O₂]`, `[HO₂⁻]` vs TPA 있을 때 차이.
- Kinetic steady-state에서 `k_TPA·[TPA] ≈ 8×10⁶ s⁻¹` vs OH의 다른 sink 합. TPA가 정말 dominant sink인지 정량.

---

### Phase 6 — Figure 및 문서화 (1 day)

**Figure (신규: `Figures/gen_fig_oh_tpa.py`)**
1. **(a) Bar chart**: `[hTPA]_sim` vs `[hTPA]_exp` @ 3 voltages, error bar (±2σ from sweep).
2. **(b) [OH] spatial profile** @ t = 1, 5, 10 min (3 voltages). Log scale.
3. **(c) 누적 OH scavenged 시계열**: `0.35⁻¹ · [hTPA]_bulk(t)` vs 실험 endpoint.
4. **(d) pH(t)** 표면/bulk 2 line, 실험 기준선(>10) 표시.
5. **(e) TPA depletion** `[TPA](t)/[TPA]₀`.
6. **(f) OH sink contribution (3.2 kVpp)**: TPA, HO₂, H₂O₂, OH⁻, 기타. Pie + 시간평균 bar.

**문서** (`notes/oh_tpa_comparison.md`)
- 방법 요약, 결과 표 (Phase 5-A), 비단조성 원인 결론, 불확실도 범위.
- TPA 포획률 / 실제 OH 생성률 / 시뮬레이션 OH 생성률 3단 비교.
- 한계: 2D/3D 효과, plasma-liquid contact geometry, carbonate.

---

## 5. 위험 요소 · 완화 전략

| # | 위험 | 가능성 | 완화 |
|---|---|---|---|
| R1 | 알칼리 조건에서 수치 안정성 문제 (pH 극단) | 중 | atol=1e-12 충분. OH⁻ 10 mM는 trace species 아님. Monolithic BDF로 안정. Phase 2에 반응 OFF 검증 step 추가. |
| R2 | TPA 반응 k/branching 불확실 (30% 범위) | 고 | Phase 5-B sweep으로 민감도 정량. 실험값과 30% 이내 일치하면 문헌 중앙값 채택. |
| R3 | pH=11.5에서 O₃+OH⁻ 경로 활성화 → 가스상 O₃ 대량 소모 → gas_alpha BC 가정 위반 | 중 | Phase 0에서 Liu 2016 (alkaline) 검토. 필요 시 O₃ gas input에 추가 sink 보정. |
| R4 | Inner filter effect(측정 artifact) 보정 논란 | 중 | 논문 주장대로 ×2 보정 후 값 사용. 보정 전 값도 보조 지표로 병기. |
| R5 | 같은 chamber 가정 틀림 (가스상 OAS는 DBD+wet, TPA는 DBD+alkaline) | 낮 | Phase 0에서 실험자 확인. 가스상 배출은 액상 유무에 둔감하다는 가정 (짧은 residence time). |
| R6 | H₂O₂ 가스상=0 가정의 한계 (이미 알려진 문제) | 중 | Phase 5-B H₂O₂ 비율 sweep. 필요 시 측정 가능한 상한 적용. |
| R7 | 공기 CO₂ 침투에 의한 HCO₃⁻ scavenging 미반영 | 낮 | Phase 4 run #5, #6에서 민감도 확인. Liu 2016 보면 주요 변수 아님. |
| R8 | 1D가 실험의 2D plasma patch footprint 재현 못 함 | 고 (구조적) | **논문 limitation에 명기**. 1D는 면적 평균 flux. |
| R9 | TPA 고갈 시 hTPA가 OH에 의해 분해되어 측정치 과소평가 | 중 | R_TPA3 포함. Phase 5-B에서 `[TPA](600)` 확인. <20% 고갈이면 무시 가능. |

---

## 6. 검증 기준 (Success Criteria)

**Minimum viable**:
- 반응 확장 후 DIW 기존 결과 regression 0% 차이.
- TPA 시나리오 수치 수렴 (pH, charge balance, mass conservation unit tests 통과).

**Primary goal (정량 비교)**:
- [hTPA]_sim이 실험과 같은 자릿수(10–100 μM).
- 전압 rank 재현 (3.2 > 3.6 > 2.6) **또는** 비단조성이 측정 artifact임을 시뮬레이션으로 입증.

**Stretch goal (논문 figure 수준)**:
- ±30% 이내 정량 일치 (파라미터 sweep 포함).
- OH sink 분해 정량 — TPA가 실제 dominant scavenger임을 확인.

---

## 7. 예상 타임라인 (순차 진행 기준)

| Phase | Effort | 누적 |
|---|---|---|
| 0 | 1 d | 1 d |
| 1 | 1–2 d | 3 d |
| 2 | 1 d | 4 d |
| 3 | 1 d | 5 d |
| 4 | 0.5 d | 5.5 d |
| 5 | 1–2 d | 7.5 d |
| 6 | 1 d | 8.5 d |

병렬화 가능: Phase 0 완료 후 1/2/3 병행 가능 → 6.5 d로 단축 가능.

---

## 8. Open questions (Phase 0 전 확인 필요)

1. **동일 chamber?** TPA 실험과 OAS 가스상 측정이 같은 DBD 장치/조건인가?
2. **Wet/Dry?** OAS data는 "Dry" 폴더 → 액상 없는 empty chamber. TPA 실험은 액상 있음. 가스상 농도는 액상 유무에 얼마나 민감한가? (이전 연구에선 10 min 처리 시 가스상은 수초 내 steady → 액상 흡수가 있더라도 feed 농도는 유사.)
3. **처리 부피**: 2 mL TPA solution의 액상 깊이가 현재 1D 설정의 `liquid_depth=10 mm`와 일치하는가? (2 mL / 표면적 = 깊이.)
4. **Inner filter ×2 보정값** 채택 기준 확정.

---

## 9. 즉시 실행 가능한 첫 스텝 (사용자 승인 시)

1. `Ver4_1D/reactions_tpa.yaml` 신규 작성 (Phase 1-a).
2. `config_1d.py`에 TPA/hTPA/Na+ species 추가, diffusivity/charge 테이블 업데이트.
3. `run_tpa_alkaline.py` 초기조건 스캐폴딩.
4. 반응 OFF sanity check 1회 (수치 안정성 확인).

**예상 시간**: 4시간 이내.
