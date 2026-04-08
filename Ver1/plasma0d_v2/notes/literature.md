# Literature Notes — plasma0d_v2

## Core References

### Reaction Set
- **Wang/Snoeckx et al. 2018**, J. Phys. Chem. C 122, 4842
  - CH4/CO2/N2/O2/H2O 시스템, 137종 631반응 — 우리 모델의 기반
  - DOI: 10.1021/acs.jpcc.7b10619
- **De Bie et al. 2015**, J. Phys. Chem. C 119, 22331
  - CH4/CO2 DBD 75종, 이온 화학 pathway 참조
- 상세 문헌 목록: `ion_chemistry_reference.md`

### Cross Section Data
- IST-Lisbon (Guerra group): Viegas 2023, Silva 2024 — CO2 σ 업데이트
- BOLSIG+ 계산: `input/BOLSIG_parameter/`, `input/BOLSIG_EEDF/`

### Thermodynamic Data
- NIST-JANAF: 종별 ΔHf° 값
- 상세 표: `enthalpy_reference.md`

### Solver Approach
- **ZDPlasKin**: SUNDIALS CVODE with non-negative constraints (우리가 scipy로 근사 구현)
- **GlobalKin**: log-transformed variables (trace species 문제로 우리 모델에서는 불가, Trial 6)
- **Patankar-type positive schemes**: 향후 대안 후보
- **ChemPlasKin (2024)**: CVODE + CppBOLOS 통합, operator splitting 없음 — arXiv:2405.04224

### Electron Energy Balance — 코드 간 비교 (2026-03-24 조사)
- **Hurlbatt et al. 2017**, Plasma Process. Polym. 14, 1600138
  - Global model 종합 리뷰. 전자 에너지 방정식 표준 형태 정리
- **Dorai PhD Thesis (~2002)**, UIUC — GlobalKin 전자 에너지 방정식 상세 기술
  - https://cpseg.eecs.umich.edu/pub/theses/rajesh_phd_thesis.pdf
- **Hagelaar & Pitchford 2005**, PSST 14, 722 — BOLSIG+ two-term approximation 방법론
- **Kushner group superelastic cross section** — detailed balance 유도
  - https://cpseg.eecs.umich.edu/classes/pub/eecs517/handouts/super_elastic_cross_section.pdf

## Key Findings from Literature

### Electron Floor (ne_seed = 1e8 m⁻³)
- 수치적 안전장치 (solver가 0으로 떨어지는 것 방지), 물리적 하한이 아님
- 이 값을 올려서 문제를 해결하는 것은 비물리적 (afterglow 분해 불가)
- Afterglow ne 유지는 반응(Penning ionization, 전자 탈착)이 담당해야 함

### Pulsed DBD Characteristics
- Afterglow: ne decay τ_diff = Λ²/D_a ≈ 42 µs (우리 계산)
- 7 orders of magnitude oscillation (1e10 ~ 1e17) per cycle — 문헌과 일치

### ne/Te 에너지 방정식 종속성 (2026-03-24 조사)

#### 세 코드의 전자 에너지 처리 비교

| 항목 | ZDPlasKin | GlobalKin | COMSOL |
|------|-----------|-----------|--------|
| Te 결정 | BOLSIG+ online (E/N 입력) | 에너지 ODE → ε̄ → Te | 에너지 ODE → ε̄ → Te |
| State variable | E/N (사용자 지정) | n_e·ε̄ | ln(n_e·ε̄) (로그) |
| EEDF | 매 step Boltzmann 풀이 | 주기적 Boltzmann 갱신 | Maxwellian/Boltzmann 선택 |
| Rate coeff. 인덱스 | E/N [Td] | ε̄ [eV] | ε̄ [eV] |
| Superelastic | BOLSIG+에 포함 | 에너지 ODE에 양의 소스항 | 명시적 역반응으로 포함 |
| Afterglow | E/N=0 → Te 즉시 열화 | P=0 → Te 점진적 감쇠 | P=0 → Te 점진적 감쇠 |

#### 현재 plasma0d_v2의 위치
- **GlobalKin/COMSOL과 동일한 구조**: 에너지 ODE(ne_eps) + ε̄-indexed LUT
- ZDPlasKin의 E/N 직접 지정 방식보다 afterglow 처리가 이론적으로 우수
- ne/Te의 power 종속은 구조적으로 정상 (GlobalKin/COMSOL도 P=0이면 Te→Tgas)

#### "비물리적 감소 가능성"의 실제 원인 후보 3가지

**(a) Superelastic collision 부재**
- GlobalKin/COMSOL: afterglow에서 e + A* → e + A + ΔE (전자 에너지 회수)
- Detailed balance: σ_super(ε') = (g₀/g*) × (ε'+ΔE)/ε' × σ_excitation(ε'+ΔE)
- 이 항이 afterglow 초기 Te 감쇠를 늦추는 핵심 역할
- **현재 reactions.yaml에 포함 여부 확인 필요**

**(b) 로그 변환 미사용 (ne_eps만)**
- COMSOL: ln(n_ε) 사용 → 10 orders magnitude를 ~23으로 압축
- Trial 6의 "log transform 불가"는 species 농도 전체 변환이었음
- **전자 에너지 밀도만 로그 변환**하는 것은 별도로 시도 가능

**(c) EEDF LUT 방식 — 문제 아님 (2026-03-24 확인)**
- 수학적으로 GlobalKin과 동일: BOLSIG+ two-term → EEDF → ∫σvF0 dε → k_ei
- 미리 계산(LUT) vs runtime 계산의 차이만 있음
- ε̄ 변화에 따른 k_ei 업데이트: 매 RHS 호출마다 LUT log-log 보간으로 정상 작동
- LUT 범위: ε̄ = 0.04~15.51 eV (50포인트), E/N = 0.1~1000 Td — pulse 영역 충분히 커버
- 의도적으로 제외한 의존성:
  - Tgas: 미미 → 300K 고정 (523K 파일 존재하나 미사용)
  - 가스 조성: N2 carrier 지배 + 낮은 전환율 + 계산 비용 tradeoff → 고정
- 이 두 가지는 engineering tradeoff이며 결함이 아님

#### 개선 우선순위 (문헌 기반, 2026-03-24 재조정)

**대기압 Te 감쇠는 물리적으로 정상** (elastic cooling τ ~ 10 ns, 실험적으로도 50-100 ns 내 Te→Tgas). Superelastic에 의한 Te buffering은 저압 현상이며 대기압에서는 무시 가능.

**진짜 문제: ne가 afterglow에서 과도하게 감쇠** — Penning ionization, 전자 탈착 등 afterglow ne 유지 반응이 부족하여 ne가 ne_seed(1e8 m⁻³)까지 떨어짐. 물리적으로는 이 반응들이 ne를 1e14~1e16 수준에서 유지해야 함.

**ne_seed 상향은 비물리적** — ne_seed는 수치적 안전장치(solver가 0으로 떨어지는 것 방지)이며, 이를 올려서 문제를 해결하면 afterglow 물리를 분해할 수 없음. 반응 자체가 ne를 유지해야 함. (2026-03-24 사용자 지적으로 철회)

| 순위 | 항목 | 난이도 | 기대 효과 |
|------|------|--------|----------|
| **1** | **Penning ionization + 전자 탈착 반응 점검/추가** | 중 | afterglow ne 자연 유지, 물리적 정확도 ↑ |
| 2 | ne_eps 로그 변환 (species 아닌 에너지만) | 중 | stiffness 추가 완화 |
| 3 | SUNDIALS CVODE 직접 사용 (ChemPlasKin식) | 고 | 근본적 성능 개선 |
| ~~취소~~ | ~~ne_seed 상향~~ | — | 비물리적: afterglow 분해 불가 (2026-03-24 철회) |
| ~~취소~~ | ~~Superelastic Tier 1~~ | — | 대기압에서 Te buffering 무시 가능 (elastic cooling 압도) |
| ~~취소~~ | ~~EEDF 동적 갱신~~ | — | 불필요: LUT 방식이 수학적으로 동일 (2026-03-24) |

#### ne 유지 메커니즘 — 대기압 afterglow (2026-03-24 조사)

| 메커니즘 | 반응 | k | 비고 |
|----------|------|---|------|
| Penning ionization | N2(A)+N2(A)→N₄⁺+e | ~5×10⁻¹¹ cm³/s | afterglow ne 유지 핵심 |
| 연관 이온화 | N2(A)+N2(a')→N₄⁺+e | | 높은 에너지 준안정 종 간 |
| 전자 탈착 | O⁻+N2(A)→O+N2+e | | 음이온이 전자 저장소 역할 |
| 표면 전하 | sDBD 유전체 전하 잔류 | | memory effect, 재점화 촉진 |

문헌: Zhao et al. (2020) High Voltage — memory effect 종합 리뷰
      Adamovich group — N2(A) kinetics, microwave ne 측정
      PMC 9905790 — 반복 펄스 대기압 N2 vibrational kinetics

#### Te 감쇠 시간 스케일 (2026-03-24 조사)

대기압 N2: τ_elastic ≈ 10 ns, Te→Tgas in ~50-100 ns (실험 확인)
- Thomson scattering (TU/e): afterglow 진입 25ns부터 monotone 감소
- OES: 90 ns 후 Te < 0.5 eV
- 시뮬레이션: ~1 μs 내 Te = Tgas

### Superelastic 반응 상세 (2026-03-24 조사)

#### Tier 1: 즉시 추가 가능 (species 이미 존재)

| 반응 | ΔE [eV] | g₀/g* | excit. cross section |
|------|---------|-------|---------------------|
| e + N2(A) → e + N2 | +6.17 | 1/3 | Re7: N2_excitation_A3_v04_Re7.txt |
| e + N2(a1) → e + N2 | +8.4 | 1/1 | Re10: N2_excitation_a1prime_Re10.txt |
| e + O2(a1Dg) → e + O2 | +0.977 | 3/2 | Re208: O2_excitation_O2a1.txt |

σ_super(ε) = (g₀/g*) × (ε+ΔE)/ε × σ_excit(ε+ΔE)

#### Tier 2: species 추가 필요

| 반응 | ΔE [eV] | 비고 |
|------|---------|------|
| e + N2(v≥1) → e + N2(v-1) | +0.29/quantum | afterglow 장기 지배적 (Guerra 2003) |
| e + CO2(001) → e + CO2 | +0.291 | CO2 feed gas |

문헌: afterglow 초기(μs)=electronic superelastic 지배, 후기(ms)=vibrational 지배

#### 주요 참고문헌
- Kushner group: superelastic cross section 유도 (Klein-Rosseland)
- Guerra & Loureiro (2003), PSST 12: N2 afterglow electron kinetics
- Laporta et al. (2012), arXiv:1208.2582: N2(v) e-V cross section set
- Kozak & Bogaerts (2014), PSST 23: CO2 vibrational model
- BOLSIG+ doc: superelastic 자동 처리 (`<->` 기호)

---
<!-- 기록 규칙:
- 새 문헌/조사결과 추가 시 반드시 날짜(YYYY-MM-DD) 포함
- 형식: "### 제목 (YYYY-MM-DD)" 또는 항목 끝에 "(YYYY-MM-DD 조사)"
- 날짜 없는 기록은 금지.
-->
