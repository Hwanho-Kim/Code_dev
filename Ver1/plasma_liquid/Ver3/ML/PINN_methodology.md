# PINN 기반 기상 미측정종 추정 — 방법론 및 계획

> **상태**: Future work. 설계 완료, 구현 미시작.  
> **최종 수정**: 2026-03-11  
> **프로젝트**: plasma_liquid/Ver3

---

## 1. 문제 정의

OAS로 측정 가능한 기상 종: O₃, NO, NO₂, NO₃, N₂O₅ (5종).  
액상 시뮬레이션에 필요하지만 미측정: **HONO, HONO₂ (HNO₃), H₂O₂** (3종).  
OH, HO₂는 수명이 극히 짧아 OAS로 측정 불가능한 라디칼.

**목표**: 부분 OAS 데이터로부터 HONO, HONO₂, H₂O₂ 기상 농도를  
Physics-Informed Neural Network (PINN) + QSSA 제약 조건을 이용하여 추정.

---

## 2. 실험 데이터

- **측정 방법**: UV-Vis OAS (광흡수 분광법)
- **측정종**: O₃, NO, NO₂, NO₃, N₂O₅ (시간분해)
- **실험 조건**: 총 12개 (습도 4수준 × 전압 3수준)
- **시스템**: Indirect DBD 플라즈마-액체 처리
- **데이터 경로**: `/home/hawn/work/plasma_liquid/empty chamber/`

---

## 3. 검토한 추정 방법론 (9가지)

논문 투고 적합성과 실현 가능성을 기준으로 평가:

| 방법 | 0D 모델 필요? | 실현성 | 논문 적합성 | 비고 |
|------|-------------|--------|-----------|------|
| A. 간이 0D 반응속도론 | O | 높음 | 양호 (신규성 낮음) | 표준 접근법 |
| B. 역방향 피팅 (액상 오차 최소화) | X | 높음 | 보통 | 순환 논증 위험 |
| C. 문헌 비율 + 민감도 분석 | X | 높음 | 미흡 | 엄밀성 부족 |
| D. PINN (Lin식, loss 내 0D) | O | 중간 | 높음 (Lin과 동일 컨셉) | 신규성 우려 |
| E. 0D surrogate + ML | O | 중간 | 양호 | 여전히 0D 필요 |
| F. Bayesian + 0D | O | 높음 | 우수 | 구현 복잡 |
| G. SINDy | X | 낮음 | - | 대상종 신호 너무 약함 |
| H. PCA + manifold | O | 중간 | 양호 | 대규모 데이터 필요 |
| I. Neural ODE / UDE | 부분 | 중간 | 보통 | 학습 불안정 |

**선택: 변형 PINN (D 변형)** — Lin 2024와 달리 해석적 rate residual + QSSA 사용으로 0D 모델 불필요.

---

## 4. Lin et al. 2024와의 비교

### 참고 논문
- Li Lin, Sophia Gershman, Yevgeny Raitses, Michael Keidar
- "Data-Driven Prediction of the Output Composition of an Atmospheric Pressure Plasma Jet"
- J. Phys. D: Appl. Phys. 57, 015203 (2024)
- DOI: 10.1088/1361-6463/acfcc7
- 오픈 액세스: https://www.osti.gov/servlets/purl/2326052
- 코드: https://mpnl.seas.gwu.edu/open-codes/

### Lin 2024 손실 함수 (Eq. 3)

```
J_total = J + W_S * Σ(Sᵢ nᵢ) + W₀ * J₀
```

#### 항 1: 정상상태 반응속도 잔차 (J, weight=1)
- 예측 농도를 0D rate equations에 투입
- 100 ns 시뮬레이션 수행 (dt=10 ns)
- J = Σᵢ [ (max(nᵢ) - min(nᵢ)) / mean(nᵢ) ]
- full 0D mechanism 필요 (~150종, ~1000+ 반응)
- 미분 불가능 → Evolutionary Algorithm 사용 (Adam 불가)

#### 항 2: 엔트로피 최대화 (W_S=1e-7)
- 열역학 제2법칙: 가능한 정상상태 중 엔트로피가 최대인 것이 물리적 해
- Sᵢ: NIST 표준 엔트로피, 여기상태/광자/전자는 별도 유도 (Eq. 9-11)
- 대수적 계산 (ODE solver 불필요)

#### 항 3: 보존법칙 페널티 (W₀=1000, 가장 강한 제약)
```
J₀ = Σᵢ Σⱼ μᵢⱼ |nⱼ - nⱼ⁰| + Σⱼ Zⱼ nⱼ
```
- 원소 보존: 입력 가스(He+air) 대비 N, O, H 원자수 보존 검증
  - μᵢⱼ = 종 j에 포함된 원소 i의 원자 수
  - nⱼ⁰ = 초기 공급 가스 농도
- 전하 중성: Σ(Zⱼ nⱼ) ≈ 0
- 대수적 계산 (ODE solver 불필요)

#### 비음수성 (Non-negativity)
- 손실 항이 아님; 출력 인코딩으로 구조적 보장: nᵢ = 10^(A''), A'' ∈ [9,15]

#### 학습
- Evolutionary Algorithm (비경사 기반)
- population mutation: w += ζ(δ - 0.5), δ ~ U[0,1]
- 50,000 iterations, minibatch=50
- FTIR 데이터(O₃, N₂O, NO₂)는 입력 뉴런으로 투입 (별도 loss 아님)

### 우리 접근법 vs Lin 2024

| | Lin 2024 | 본 연구 |
|---|---|---|
| NN 출력 | 147종 농도 | 미측정종 농도 + 학습가능 source term (Q) |
| Physics loss | 0D full mechanism (~1000 반응) | 13개 반응 rate residual (미분 가능) |
| OH/HO₂ | 0D 시뮬레이션 내부에서 계산 | QSSA 대수 방정식 |
| 보존법칙 | N,O,H 원소 + 전하 중성 (W=1000) | N 보존 (O, H 추가 검토 필요) |
| 열역학 | 엔트로피 최대화 (제2법칙) | 미적용 (시간변화 afterglow에 부적합) |
| Optimizer | Evolutionary Algorithm | Adam (모든 구성요소 미분 가능) |
| 0D 모델 필요 | 필수 (loss에 내장) | 불필요 |

### 신규성 프레이밍
> "Lin et al. (2024)은 loss function 내에 완전한 기상 반응속도 메커니즘(~153종)을
> 필요로 했으나, 본 연구는 잘 확립된 최소 반응 세트(N=13)를 physics 제약으로 사용하고
> 단수명 중간체에 대해 QSSA를 적용한다. 이를 통해 학습 루프 내 0D 시뮬레이션 임베딩의
> 계산 비용을 회피하면서, 해석적으로 미분 가능한 rate residual을 통해 물리적 일관성을
> 유지한다. OH 생성 속도는 학습가능 파라미터로 취급하여, 명시적 방전 모델링 없이
> 복잡한 플라즈마 화학을 암묵적으로 인코딩한다."

---

## 5. PINN 아키텍처

### 입력층 (3 뉴런)
```
(t, V, RH)
t  = 처리 시간 [s]
V  = 방전 전압 [V] (예: 2600, 3200, 3600)
RH = 상대 습도 [%]
```

### 은닉층
```
4층: [64, 128, 128, 64], 활성함수: Tanh
```

### 출력층 (12 뉴런)
```
[0:5]  = 측정종 log₁₀ 농도 (O₃, NO, NO₂, NO₃, N₂O₅)
[5:8]  = 미측정종 log₁₀ 농도 (HONO, HONO₂, H₂O₂)
[8:12] = 학습가능 source terms log₁₀ (Q_OH, Q_O₃, Q_NO₂, Q_NO₃)
```

비음수성은 log₁₀ 인코딩으로 구조적 보장 (Lin 2024와 동일 개념).

---

## 6. QSSA 모듈 (미분 가능)

OH, HO₂는 NN이 예측하지 않음. d[OH]/dt = 0, d[HO₂]/dt = 0으로 놓고 대수적으로 계산.

### OH 정상상태
```
생성: Q_OH + k_NEW2·HO₂·NO + k_NEW3·HO₂·O₃
소비: k8·O₃ + k7·NO₂·M + k_NEW1·NO·M + k12·HONO + k13·HONO₂
      + k14·H₂O₂ + 2·k9·OH·M + k10·HO₂

2차 방정식: a·OH² + b·OH - c = 0
OH_ss = (-b + sqrt(b² + 4ac)) / (2a)
```

### HO₂ 정상상태
```
생성: k8·OH·O₃ + k14·OH·H₂O₂
소비: 2·k11·HO₂ + k10·OH + k_NEW2·NO + k_NEW3·O₃

2차 방정식 (HO₂ 자기반응 항 때문)
```

### N₂O₄ 평형
```
N₂O₄ = Keq · NO₂² · kBT/P
```

---

## 7. 반응 세트 (13개)

초기 설계에는 R1~R4(측정종 간 반응)가 포함되어 있었으나 제거함.  
제거 이유:
- 측정종은 L_data로 구속되므로, 측정종 간 반응의 ODE를 강제할 필요 없음
- R1~R4는 미측정종 ODE에 직접 등장하지 않음
- 불완전한 측정종 ODE를 강제하면 NN에 bias 발생

누락된 핵심 반응 3개를 추가:

### 가교 반응: 측정종 → 미측정종
```
R5:   N₂O₅ + H₂O → 2 HONO₂             γ=0.04          Bertram&Thornton 2009
R6:   N₂O₄ + H₂O → HONO + HONO₂         γ=1e-4          Finlayson-Pitts 2003
R7:   NO₂ + OH + M → HONO₂ + M          삼체반응         JPL-19
NEW1: NO + OH + M → HONO                k₀=7.4e-31 cm⁶/s JPL-19
```

### H₂O₂ 생성
```
R9:   OH + OH + M → H₂O₂ + M            k₀=6.9e-31       JPL-19
R11:  HO₂ + HO₂ → H₂O₂ + O₂            k=2.2e-13        JPL-19
```

### 라디칼 순환 및 소비
```
R8:   OH + O₃ → HO₂ + O₂               k=7.3e-14        JPL-19
R10:  OH + HO₂ → H₂O + O₂              k=1.1e-10        JPL-19
R12:  OH + HONO → NO₂ + H₂O            k=6.0e-12        JPL-19
R13:  OH + HONO₂ → NO₃ + H₂O           k=1.5e-13        JPL-19
R14:  OH + H₂O₂ → HO₂ + H₂O           k=1.7e-12        JPL-19
NEW2: HO₂ + NO → OH + NO₂              k=8.0e-12        JPL-19
NEW3: HO₂ + O₃ → OH + 2 O₂            k=2.0e-15        JPL-19
```

### NEW2 (HO₂+NO)가 필수인 이유
NOx-rich 플라즈마 환경(NO ~ 100 ppb)에서:
```
k_NEW2·[NO] = 8e-12 × 2.5e12 = 20 s⁻¹       ← HO₂ 주 소비 경로 (~80%)
2·k11·[HO₂] = 2×2.2e-13 × 1e9 = 0.44 s⁻¹    ← 자기반응 (비주류)
```
NEW2 없으면 HO₂가 ~5배, H₂O₂가 ~25배 과대추정됨.

또한 OH QSSA에서 주요 production term이 누락됨 (HO₂+NO→OH+NO₂).

### NEW1 (NO+OH→HONO)이 필수인 이유
이 반응 없이는 HONO가 비균질 N₂O₄ 가수분해(γ=1e-4, 매우 느림)에서만 생성됨.
NO+OH+M은 HONO의 주요 균질 생성 경로.

### 제거된 반응
```
R1: O₃ + NO → NO₂ + O₂       (측정종 간 반응, 미측정종 ODE에 미등장)
R2: O₃ + NO₂ → NO₃ + O₂      (동일)
R3: NO₂ + NO₃ + M → N₂O₅ + M (동일)
R4: N₂O₅ → NO₂ + NO₃          (동일)
```

### 검토했으나 제외한 반응
```
NO + NO₂ + H₂O(표면) → 2HONO     반응기 벽면 의존, 불확실 → Q_OH에 흡수
HO₂ + NO₂ + M → HO₂NO₂           298K에서 열분해 빠름 (τ~30s) → 무시 가능
HONO + HONO → NO + NO₂ + H₂O     k~1e-20, 극히 느림 → 무시
```

---

## 8. 손실 함수 설계

```
L_total = λ_d · L_data + λ_p · L_physics + λ_c · L_conservation + λ_s · L_smooth
```

### L_data (λ_d = 1.0)
```
L_data = Σ || ĉ_measured - c_OAS ||²
```
측정종(O₃, NO, NO₂, NO₃, N₂O₅)이 OAS 데이터와 일치하도록 구속.

### L_physics (λ_p = curriculum: 0 → 0.1)
```
L_physics = Σ || dc_unmeasured/dt - R(c) ||²
```
HONO, HONO₂, H₂O₂에 대한 rate residual.
dc/dt는 autograd로 계산; R(c)는 13개 반응 + QSSA에서 산출.
curriculum: 학습 초반에는 0, 점진적으로 증가.

### L_conservation (λ_c = 10.0)
```
질소 원자수 보존: 전체 종에 포함된 N 원자 총합 ≈ 일정
```
참고: O, H 원소 보존 추가 검토 필요 (Lin 2024는 전 원소에 W=1000 적용).

### L_smooth (λ_s = 0.001)
```
시간적 평활성: 큰 d²c/dt² 페널티 부여
```

---

## 9. 학습 계획

### 데이터
- 12개 조건 (습도 4 × 전압 3), 시간분해 OAS 측정
- 각 조건당: ~60-120 시간 포인트 (측정에 따라 다름)
- 총계: ~700-1400 데이터 포인트

### 데이터 증강
- 초기에는 미계획 (Lin 2024는 16 → 304개 조건으로 증강)
- 12개 조건이 부족할 경우 필요할 수 있음

### 학습 절차
1. Phase 1 (워밍업): L_data만 사용, 5000 epochs
2. Phase 2 (물리 도입): λ_p 점진적 증가, 20000 epochs
3. Phase 3 (미세 조정): 모든 loss 활성, LR 감소, 10000 epochs

### Optimizer
- Adam (lr=1e-3, 1e-5까지 감소)
- 모든 구성요소가 미분 가능 → 표준 역전파 적용 가능

### 검증
- Leave-one-out 교차검증 (12개 조건 → 12 fold)
- 물리적 일관성 검사:
  - 모든 농도 ≥ 0 (구조적 보장)
  - HONO, HONO₂, H₂O₂가 물리적으로 타당한 범위 내
  - 보존법칙 충족
  - 예상되는 단조성 (예: 습도 증가 → H₂O₂ 증가)

---

## 10. 계획된 파일 구조

```
Ver3/ML/
  PINN_methodology.md    ← 이 파일
Ver3/pinn/
  __init__.py
  model.py               PlasmaChemPINN 아키텍처
  qssa.py                OH, HO₂ QSSA (미분 가능, PyTorch)
  reactions.py            13개 반응식 + 속도상수
  loss.py                 L_data, L_physics, L_conservation, L_smooth
  train.py                학습 루프 + curriculum 스케줄
  data_loader.py          OAS CSV → torch Dataset
  validate.py             교차검증 + 물리적 일관성 검사
```

---

## 11. 액상 시뮬레이션과의 통합

PINN 학습 완료 후:
1. PINN이 주어진 (t, V, RH)에 대해 HONO, HONO₂, H₂O₂ 예측
2. 측정된 O₃, NO, NO₂, NO₃, N₂O₅와 결합
3. QSSA로 OH, HO₂ 계산
4. 전체 기상 경계조건으로 기존 Ver3 액상 시뮬레이션에 투입
5. Two-film 물질전달 → 101개 수상 반응 → 액상 농도 계산
6. 액상 OAS 측정값(pH, NO₂⁻, NO₃⁻, H₂O₂(aq))과 비교 검증

---

## 12. 미해결 질문

1. **O, H 원소 보존**: L_conservation에 산소, 수소 원자수 보존을 추가해야 하는가?
   Lin 2024는 전 원소에 가장 높은 가중치(1000)를 적용함.
2. **엔트로피 항**: Lin 2024는 열역학 제2법칙을 적용함. 시간변화하는 afterglow
   시스템에 적용 가능한가? 직접 적용은 어려울 것으로 판단되나 조사 필요.
3. **데이터 충분성**: 12개 조건이 PINN 학습에 충분한가?
   Lin 2024는 16개 조건을 304개로 증강함.
4. **학습가능 source terms (Q)**: 현재 Q_OH, Q_O₃, Q_NO₂, Q_NO₃.
   명시적으로 모델링하지 않은 비균질 경로를 위해 Q_HONO 추가 필요?
5. **OAS로 HONO 측정 가능성**: Bae et al. 2024가 DOAS식 피팅(NNLS+CISSIR)으로
   HONO 검출에 성공함. HONO가 측정 가능해지면 PINN 추정 범위가 축소됨.

---

## 13. 주요 참고문헌

- Liu 2015, J. Phys. D 48:495201 — 101개 수상 반응, Henry's law 상수
- Liu 2016, Plasma Processes Polym. — 액상 시뮬레이션 방법론
- Lee 2023, CEJ 458:141425 — Two-film 물질전달 모델
- Lin et al. 2024, J. Phys. D 57:015203 — 플라즈마 종 추정을 위한 PINN (참고 방법)
- Sakiyama et al. 2012, J. Phys. D 45:425201 — 기상 메커니즘 (~150종)
- Bae et al. 2024, Chemosphere 364:143105 — 이중 OAS 기상+액상 측정
- Huh et al. 2024, PSST 33:075007 — OAS 피팅 방법론 (NNLS+CISSIR)
- Bertram & Thornton 2009, Atmos. Chem. Phys. — N₂O₅ 가수분해 γ
- Finlayson-Pitts 2003, Chem. Rev. — N₂O₄ 가수분해
- JPL Publication 19-5 — NASA/JPL 반응속도 데이터 평가 (전체 속도상수)
