# Ion Chemistry Reference for CH₄/CO₂/N₂/O₂ sDBD 0D Model

**작성일**: 2026-03-05 | **최종 수정**: 2026-03-12 (IIR binary+ternary 확장)  
**목적**: 기존 51종/139반응 neutral-only 모델에 이온 화학 추가를 위한 핵심 pathway 조사 결과  
**방침**: 방대한 reaction set (Bogaerts 631개) 대신 핵심 pathway 기준 최소 세트 구성  
**검증**: IST-Lisbon, Kushner, Capitelli, TRINITI 그룹 + 2020-2025 리뷰 14편으로 교차검증 완료

---

## 1. 문헌 조사 대상

### 1.1 초기 조사 (Bogaerts 그룹 중심, 6편)

| 논문 | 시스템 | 총 종 | 이온 반응 수 | DOI |
|------|--------|:-----:|:-----------:|-----|
| De Bie et al. 2011, Plasma Process. Polym. | CH₄ DBD | 36 | 175 | — |
| De Bie et al. 2015, J. Phys. Chem. C 119, 22331 | CH₄/CO₂ DBD | 75 | 386 | 10.1021/acs.jpcc.5b06515 |
| Snoeckx et al. 2016, Energy Environ. Sci. 9, 999 | CO₂/N₂ DBD | ~100 | ~300+ | 10.1039/C5EE03304G |
| Wang et al. 2017, ChemSusChem 10, 2145 | N₂/O₂ GA | ~50 | ~200+ | 10.1002/cssc.201700095 |
| Wang/Snoeckx et al. 2018, J. Phys. Chem. C 122, 4842 | CH₄/CO₂/N₂/O₂/H₂O | **137** | **631** | 10.1021/acs.jpcc.7b10619 |
| Slaets 2024, PhD thesis UAntwerpen | CO₂/CH₄/N₂ GAP | 177 | ~500+ | — |

### 1.2 추가 조사 (독립 그룹 교차검증, 14편+)

| 그룹 | 핵심 논문 | 기여 |
|------|-----------|------|
| **Kushner** (Michigan) | Dorai 2001/2003, Meyer 2025 | N₂/O₂/hydrocarbon DBD 이온 cascade 확인 |
| **IST-Lisbon** (Guerra) | Viegas 2023, Silva 2024, Liu 2025 | 0D 검증, CO₂ σ 업데이트 |
| **Capitelli** (Bari) | Pietanza 2022 | CO₂ 이온 화학, CO₃⁺ cluster |
| **TRINITI** (Pancheshnyi) | Pancheshnyi 2013 | O₂(a¹Δg) detachment |
| **MuroranIT** (Kawaguchi) | Kawaguchi 2021, 2025 | N₂/O₂ σ 업데이트 |
| **독립** | Bang 2023, Sun 2022, Vialetto 2025 | 교차검증, 민감도 분석, 3-body attachment |
| **KIDA/UMIST/Anicich** | 데이터베이스 조회 | CH₄ 이온 rate 확인 |

---

## 2. 대기압 sDBD 이온 캐스케이드 구조

```
전자충돌 이온화 (discharge ON, ns~μs)
         |
    N₂⁺, O₂⁺, CO₂⁺, CH₄⁺              <-- 1차 이온
         |
    +-- charge transfer (k ~ 1e-10 ~ 1e-9, ns 스케일) --+
    |  N₂⁺ + O₂  -> O₂⁺ + N₂                            |
    |  N₂⁺ + CH₄ -> CH₃⁺ + H + N₂  (50%)               |
    |  CO₂⁺ + CH₄ -> CH₃⁺ + OH + CO                     |
    |  CH₄⁺ + CH₄ -> CH₅⁺ + CH₃                         |
    +-----------------------------------------------------+
         |
    +-- 3-body clustering (1 atm에서 near high-P limit) --+
    |  N₂⁺ + N₂ + M -> N₄⁺ + M   (k_eff ~ 8e-10)       |
    |  O₂⁺ + O₂ + M -> O₄⁺ + M   (k_eff ~ 6e-11)       |
    |  BUT: N₄⁺ + O₂ -> O₂⁺ + 2N₂ (N₄⁺는 중간체!)       |
    +------------------------------------------------------+
         |
    Terminal ions:
      dry N₂/O₂:      NO⁺
      with CH₄:        CH₅⁺ -> (+ H₂O) -> H₃O⁺
      with trace H₂O:  H₃O⁺ (dominant)
```

**핵심 물리**:
- CH₄가 1%만 있어도 N₂⁺ + CH₄ (k ~ 1e-9)가 N₂⁺ + O₂ (k ~ 6e-11)보다 **17배 빠름**
- CH₄가 이온 캐스케이드를 완전히 지배
- 1 atm에서 N₄⁺/O₄⁺ cluster ion의 DR이 N₂⁺/O₂⁺보다 5~10배 빠름 -> 생략 불가

---

## 3. 추가할 이온 종 (Minimal Set: 12종)

### 3.1 양이온 (10종)

| 이온 | 역할 | 문헌 출현 (6편 중) |
|------|------|:------------------:|
| N₂⁺ | 1차 이온화, charge transfer 소스 | 4 |
| O₂⁺ | 1차 이온화, N₂⁺ cascade 수신 | 4 |
| CO₂⁺ | CO₂ 이온화 | 4 |
| CH₄⁺ | CH₄ 이온화, CH₅⁺ 전구체 | 4 |
| CH₃⁺ | N₂⁺+CH₄의 주 생성물, DR로 라디칼 생성 | 4 |
| CH₅⁺ | CH₄⁺+CH₄ proton transfer | 4 |
| N₄⁺ | 1 atm clustering, DR 매우 빠름 (2.6e-6) | 4 |
| O₄⁺ | 1 atm clustering, DR 매우 빠름 (1.4e-6) | 4 |
| NO⁺ | terminal ion (dry), DR 느림 | 4 |
| H₃O⁺ | terminal ion (H₂O 존재 시) | 3 |

### 3.2 음이온 (2종)

| 이온 | 역할 | 문헌 출현 (6편 중) |
|------|------|:------------------:|
| O⁻ | 해리 attachment (discharge 중), 빠른 detachment | 4 |
| O₂⁻ | 3-body attachment (1 atm 지배적), afterglow 전자 저장 | 5 |

### 3.3 제외 근거

| 제외된 이온 | 이유 |
|------------|------|
| O⁺, N⁺, H⁺, H₂⁺ | 빠른 중간체 — charge transfer로 즉시 소멸 (ns) |
| C₂Hₓ⁺ (C₂H₂⁺~C₂H₆⁺) | C2 생성물 농도 낮음, 우선순위 낮음 |
| CO⁺ | CO₂ dissociative ionization (threshold 19.5 eV), 2차적 |
| H₂O⁺, OH⁺ | 빠른 중간체 — H₂O⁺ + H₂O -> H₃O⁺ + OH (즉시) |
| O₃⁻, O₄⁻ | O₂⁻가 지배적, minor correction |
| CO₃⁻, CO₄⁻ | CO₂-유래 음이온, 2차적 |
| NH₄⁺, NH₃⁺ | NH₃ 화학 불포함 시 불필요 |
| N₃⁺ | 3-body N₂⁺+N₂+N₂ 경로지만 N₄⁺가 지배적 |
| N₂O⁺, NO₂⁺ | NOx 이온, minor |

---

## 4. 핵심 이온 반응 세트 (38반응)

### 4A. 전자충돌 이온화 (5반응)

BOLSIG+ cross-section 기반, Method B (BOLSIG+ EEDF × sigma 적분)로 처리.
**참고**: BOLSIG+ EEDF 파일 제공 전까지 Maxwellian EEDF를 placeholder로 사용 중.

| ID | 반응 | Threshold (eV) | Cross-section source |
|----|------|:--------------:|----------------------|
| I1 | e + N₂ -> N₂⁺ + 2e | 15.58 | Itikawa 2006, J. Phys. Chem. Ref. Data 35, 31 |
| I2 | e + O₂ -> O₂⁺ + 2e | 12.07 | Itikawa 2009, J. Phys. Chem. Ref. Data 38, 1 |
| I3 | e + CH₄ -> CH₄⁺ + 2e | 12.61 | Phelps/Morgan DB, LXCat |
| I4 | e + CH₄ -> CH₃⁺ + H + 2e | 14.3 | Phelps/Morgan DB, LXCat |
| I5 | e + CO₂ -> CO₂⁺ + 2e | 13.78 | Liu et al. 2025, PSST 34, 035003 |

**참고**: LXCat (lxcat.net) Phelps database에서 cross-section 다운로드 가능.

### 4B. 해리 재결합 — Dissociative Recombination (8반응)

Rate coefficient 형태: **k = A × (300/Tₑ)^n** [cm³/s], Tₑ in Kelvin.
변환: Tₑ(K) = eps_mean(eV) × 11604 × 2/3

| ID | 반응 | A (cm³/s) | n | Ref |
|----|------|:---------:|:---:|-----|
| DR1 | e + N₂⁺ -> N + N | 2.2e-7 | 0.39 | Sheehan & St-Maurice 2004, J. Geophys. Res. 109, A03302 |
| DR2 | e + O₂⁺ -> O + O | 1.95e-7 | 0.70 | Sheehan & St-Maurice 2004 |
| DR3 | e + CO₂⁺ -> CO + O | 3.8e-7 | 0.50 | Florescu-Mitchell & Mitchell 2006, Phys. Rep. 430, 277 |
| DR4 | e + CH₄⁺ -> CH₃ + H | 1.05e-7 | 0.50 | Sheehan & St-Maurice 2004, Adv. Space Res. 33, 216 |
| DR5 | e + CH₃⁺ -> CH₂ + H | **1.1e-8** | 0.50 | Vejby-Christensen et al. 1997, Ap. J. 483, 531 (ASTRID ring) |
| DR6 | e + N₄⁺ -> N₂ + N₂ | **2.6e-6** | 0.50 | Cao & Johnsen 1991, J. Chem. Phys. 95, 7356 |
| DR7 | e + O₄⁺ -> O₂ + O₂ | **1.4e-6** | 0.50 | Florescu-Mitchell & Mitchell 2006 |
| DR8 | e + NO⁺ -> N + O | 3.5e-7 | 0.69 | Sheehan & St-Maurice 2004 |

**주의**: DR6, DR7 (cluster ion)은 DR1~5보다 **5~10배 빠름**.
대기압에서 이걸 빼면 이온 수명을 크게 과대평가.

**⚠️ DR5 보정 (2026-03 확인)**: Florescu-Mitchell 2006의 CH₃⁺ DR 값 (3.75e-7)은 **34× 과대평가**.
CH₃⁺는 closed-shell 이온 → DR이 비정상적으로 느림.
Vejby-Christensen et al. 1997 (ASTRID storage ring 직접 측정)에서 k = 1.1e-8·(300/Tₑ)^0.50.
비교: CH₅⁺ DR = 1.1e-6 (fast), CH₄⁺ DR = 2.8e-7 (normal), CH₃⁺ DR = 1.1e-8 (anomalously slow).

**DR4 branching (참고, 본 모델에서는 주 채널만 사용)**:
- CH₃ + H: 30%
- CH₂ + H₂: 40% (largest)
- CH₂ + 2H: 16%
- CH + H₂ + H: 14%

### 4C. 이온-분자 Charge Transfer (10반응)

Rate coefficient: 상수 k [cm³/s] (약한 온도 의존성 무시).

| ID | 반응 | k (cm³/s) | Ref |
|----|------|:---------:|-----|
| CT1 | N₂⁺ + O₂ -> O₂⁺ + N₂ | 6.0e-11 | Anicich 1993, J. Phys. Chem. Ref. Data 22, 1469 |
| CT2 | N₂⁺ + CO₂ -> CO₂⁺ + N₂ | 7.0e-11 | Fehsenfeld et al. 1966, J. Chem. Phys. 44, 4537 |
| CT3 | N₂⁺ + CH₄ -> CH₃⁺ + H + N₂ | 5.5e-10 | Anicich 1993 (50% branch of total 1.1e-9) |
| CT4 | N₂⁺ + CH₄ -> CH₄⁺ + N₂ | 4.4e-10 | Anicich 1993 (40% branch of total 1.1e-9) |
| CT5 | CO₂⁺ + CH₄ -> CH₃⁺ + OH + CO | 1.0e-9 | Anicich 1993 |
| CT6 | CH₄⁺ + CH₄ -> CH₅⁺ + CH₃ | 1.1e-9 | Smith & Adams 1977, Ap. J. 217, 741 |
| CT7 | O₂⁺ + NO -> NO⁺ + O₂ | 4.4e-10 | Anicich 1993 |
| CT8 | N₄⁺ + O₂ -> O₂⁺ + 2N₂ | 2.5e-10 | Anicich 1993 |
| CT9 | N₄⁺ + CH₄ -> CH₄⁺ + 2N₂ | 1.0e-9 | Anicich 1993 |
| CT10 | CH₅⁺ + H₂O -> H₃O⁺ + CH₄ | 2.0e-9 | Anicich 1993 |

**물리적 참고사항**:
- O₂⁺ + CH₄는 약간 endothermic (IE(O₂)=12.07 < IE(CH₄)=12.61) -> 느림, 제외
- CO₂⁺ + O₂는 endothermic (reverse favored) -> 제외

### 4D. 3-Body Clustering (2반응)

1 atm에서 effective bimolecular rate로 처리.

| ID | 반응 | k_eff at 1 atm (cm³/s) | 비고 | Ref |
|----|------|:----------------------:|------|-----|
| CL1 | N₂⁺ + N₂ (+M) -> N₄⁺ | 8.3e-10 | high-P limit at 1 atm | Troe 2005, PCCP 7, 1560 |
| CL2 | O₂⁺ + O₂ (+M) -> O₄⁺ | 6.0e-11 | intermediate falloff | Dheandhanoo & Johnsen 1983, Planet. Space Sci. 31, 933 |

**구현 참고**: CL1은 이미 high-pressure limit이므로 k_eff = k_inf = 8.3e-10.
CL2는 k₀·[M] ≈ 8.5e-11이므로 falloff 영역. 정밀 구현 시 Troe falloff formula 사용.

**온도 의존성**:
- CL1: k₀ = 6.8e-29 × (300/T)^2.23 cm⁶/s
- CL2: k₀ = 3.4e-30 × (300/T)^3.2 cm⁶/s

### 4E. 전자 Attachment (3반응)

| ID | 반응 | k | 비고 | Ref |
|----|------|---|------|-----|
| AT1 | e + O₂ + M -> O₂⁻ + M | 1.1e-31·(300/Tₑ)²·exp(-70/Tg)·exp(1500·(Tₑ-Tg)/(Tₑ·Tg)) cm⁶/s | **1 atm 지배적** | Kossyi et al. 1992, PSST 1, 207 |
| AT2 | e + O₂ -> O⁻ + O | sigma(eps), BOLSIG+ cross-section | threshold ~4.7 eV | Phelps DB, LXCat |
| AT3 | e + O₃ -> O₂⁻ + O | 1.0e-9 cm³/s | O₃ 축적 시 중요 | Capitelli et al. 2000, Springer |

**AT1 effective 2-body rate at 1 atm (low Te)**:
k₂ = k₃ × [M] ≈ 1.1e-31 × 2.7e19 ≈ 3e-12 cm³/s
-> afterglow에서 전자 소멸의 주 메커니즘

### 4F. Detachment (4반응, afterglow 전자 재생)

| ID | 반응 | k (cm³/s) | 비고 | Ref |
|----|------|:---------:|------|-----|
| DT1 | O⁻ + O -> O₂ + e | 1.4e-10 | 빠름, 핵심 afterglow 소스 | Capitelli et al. 2000 |
| DT2 | O₂⁻ + O -> O₃ + e | 1.5e-10 | O 원자 필요, ⚠️ 아래 참고 | Capitelli et al. 2000 |
| DT3 | O⁻ + N₂(A) -> O + N₂ + e | 2.2e-9 | Penning-type, 매우 빠름 | Capitelli et al. 2000 |
| DT4 | O₂⁻ + N₂(A) -> O₂ + N₂ + e | 2.1e-9 | Penning-type, 매우 빠름 | Capitelli et al. 2000 |

**물리적 참고사항**:
- 300K에서 thermal detachment (O₂⁻ + M -> O₂ + e + M)는 exp(-5590/T) ≈ 0 -> 무시 가능
- Detachment는 **O 원자**와 **N₂(A) 준안정**에 의해 구동됨
- 현재 모델에 N₂(A)가 이미 포함되어 있으므로 DT3, DT4 즉시 사용 가능

**⚠️ DT2 불일치 참고 (2026-03 교차검증)**:
Capitelli 2000: k = 1.5e-10 cm³/s. Kushner group (Dorai & Kushner 2001): k = 3.3e-10 cm³/s.
Factor ~2.2 차이. Kossyi 1992 원문값은 3.3e-10에 가까움.
**권장**: Kossyi 1992 원문 확인 후 결정. 두 값 모두 reasonable range.

**O⁻ + N₂ -> N₂O + e 주의사항**:
Kossyi 1992 값은 과대평가. Shuman et al. 2023 (PCCP 25, 31917) 재측정:
k = 3e-12·exp(-3250/T). 300K에서 ~1e-17 cm³/s -> 무시 가능.

### 4G. Ion-Ion 재결합 — Binary Mutual Neutralization (18반응)

#### 4G-1. Molecular ions (기존 6반응)

Rate: **k = 2.0e-7 × (300/T)^0.5** [cm³/s]

| ID | 반응 | 생성물 | Ref |
|----|------|--------|-----|
| II1 | O⁻ + O₂⁺ -> O + O₂ | neutrals | Capitelli et al. 2000 |
| II2 | O⁻ + N₂⁺ -> O + N₂ | neutrals | Capitelli et al. 2000 |
| II3 | O⁻ + NO⁺ -> O + NO | neutrals | Capitelli et al. 2000 |
| II4 | O₂⁻ + O₂⁺ -> O₂ + O₂ | neutrals | Capitelli et al. 2000 |
| II5 | O₂⁻ + N₂⁺ -> O₂ + N₂ | neutrals | Capitelli et al. 2000 |
| II6 | O₂⁻ + NO⁺ -> O₂ + NO | neutrals | Capitelli et al. 2000 |

#### 4G-2. 추가 binary IIR pairs (12반응, 2026-03 추가)

기존 6반응은 O₂⁺/N₂⁺/NO⁺만 포함. O₄⁺/N₄⁺/CH₄⁺/CH₃⁺/CO₂⁺/CH₅⁺에 대한 IIR이 누락되어 있었음.
특히 **O₄⁺의 유일한 손실 채널이 DR과 D_a/Λ²**뿐이어서 Λ 민감도가 비물리적으로 높았음.

Rate:
- **Cluster ions** (O₄⁺, N₄⁺, CH₅⁺): k = 1.0e-7 cm³/s (flat, 온도 무관)
- **Molecular ions** (CH₄⁺, CH₃⁺, CO₂⁺): k = 2.0e-7 × (300/T)^0.5 cm³/s

| ID | 반응 | 생성물 | k 분류 | Ref |
|----|------|--------|--------|-----|
| II7 | O⁻ + O₄⁺ -> O + 2O₂ | neutrals | cluster | Kossyi 1992 |
| II8 | O⁻ + N₄⁺ -> O + 2N₂ | neutrals | cluster | Kossyi 1992 |
| II9 | O⁻ + CH₄⁺ -> O + CH₄ | neutrals | molecular | Kossyi 1992 |
| II10 | O⁻ + CH₃⁺ -> O + CH₃ | neutrals | molecular | Kossyi 1992 |
| II11 | O⁻ + CO₂⁺ -> O + CO₂ | neutrals | molecular | Kossyi 1992 |
| II12 | O⁻ + CH₅⁺ -> O + CH₄ + H | neutrals | cluster | Kossyi 1992 |
| II13 | O₂⁻ + O₄⁺ -> 3O₂ | neutrals | cluster | Kossyi 1992 |
| II14 | O₂⁻ + N₄⁺ -> O₂ + 2N₂ | neutrals | cluster | Kossyi 1992 |
| II15 | O₂⁻ + CH₄⁺ -> O₂ + CH₄ | neutrals | molecular | Kossyi 1992 |
| II16 | O₂⁻ + CH₃⁺ -> O₂ + CH₃ | neutrals | molecular | Kossyi 1992 |
| II17 | O₂⁻ + CO₂⁺ -> O₂ + CO₂ | neutrals | molecular | Kossyi 1992 |
| II18 | O₂⁻ + CH₅⁺ -> O₂ + CH₄ + H | neutrals | cluster | Kossyi 1992 |

**단위 환산 (reactions.yaml)**:
- Cluster: A_SI = 1e-7 × N_A = 6.022e+10, n=0
- Molecular: A_SI = 2e-7 × 300^0.5 × N_A = 2.08610e+12, n=-0.5

### 4H. Ion-Ion 재결합 — Ternary (3-body) Channel (18반응, 2026-03 추가)

Binary IIR 외에, 대기압에서 제3체(M)가 매개하는 ternary IIR 채널이 존재.
Thomson theory 기반 universal rate.

Rate: **k₃ = 2.0e-25 × (300/T)^2.5** [cm⁶/s]

At 300K, 1 atm: k_eff = k₃ × [M] = 4.9e-6 cm³/s → **binary 대비 ~25× (molecular), ~49× (cluster)**

모든 18개 binary IIR pair에 대해 동일한 ternary 채널 추가:

| ID | 반응 | Ref |
|----|------|-----|
| IIT1 | O⁻ + O₂⁺ + M -> O + O₂ + M | Kossyi 1992 |
| IIT2 | O⁻ + N₂⁺ + M -> O + N₂ + M | Kossyi 1992 |
| IIT3 | O⁻ + NO⁺ + M -> O + NO + M | Kossyi 1992 |
| IIT4 | O⁻ + O₄⁺ + M -> O + 2O₂ + M | Kossyi 1992 |
| IIT5 | O⁻ + N₄⁺ + M -> O + 2N₂ + M | Kossyi 1992 |
| IIT6 | O⁻ + CH₄⁺ + M -> O + CH₄ + M | Kossyi 1992 |
| IIT7 | O⁻ + CH₃⁺ + M -> O + CH₃ + M | Kossyi 1992 |
| IIT8 | O⁻ + CO₂⁺ + M -> O + CO₂ + M | Kossyi 1992 |
| IIT9 | O⁻ + CH₅⁺ + M -> O + CH₄ + H + M | Kossyi 1992 |
| IIT10 | O₂⁻ + O₂⁺ + M -> 2O₂ + M | Kossyi 1992 |
| IIT11 | O₂⁻ + N₂⁺ + M -> O₂ + N₂ + M | Kossyi 1992 |
| IIT12 | O₂⁻ + NO⁺ + M -> O₂ + NO + M | Kossyi 1992 |
| IIT13 | O₂⁻ + O₄⁺ + M -> 3O₂ + M | Kossyi 1992 |
| IIT14 | O₂⁻ + N₄⁺ + M -> O₂ + 2N₂ + M | Kossyi 1992 |
| IIT15 | O₂⁻ + CH₄⁺ + M -> O₂ + CH₄ + M | Kossyi 1992 |
| IIT16 | O₂⁻ + CH₃⁺ + M -> O₂ + CH₃ + M | Kossyi 1992 |
| IIT17 | O₂⁻ + CO₂⁺ + M -> O₂ + CO₂ + M | Kossyi 1992 |
| IIT18 | O₂⁻ + CH₅⁺ + M -> O₂ + CH₄ + H + M | Kossyi 1992 |

**단위 환산 (reactions.yaml)**:
- A_SI = 2e-25 × 300^2.5 × N_A² × 1e-12 = 1.13067e+17, n=-2.5, order=3
- 코드에서 order=3: rate = k × c[A⁻] × c[B⁺] × c_total (M = total gas)

**배경**:
- GlobalKin (Kushner group): binary IIR만 사용, ternary 미포함
- ZDPlasKin detailed (Bogaerts group): binary + ternary 모두 사용
- 본 모델: binary + ternary 모두 포함 (ZDPlasKin 방식)

**Λ 민감도 개선 효과** (303K, Λ=1mm vs 10mm):

| 단계 | ΔTe | Δn_e | ΔCH4 |
|------|-----|------|------|
| IIR 없음 | -14.3% | +50.7% | -0.56%p |
| Binary IIR only | -9.6% | +30.5% | -0.31%p |
| Binary + Ternary IIR | **-1.9%** | **+5.4%** | **-0.27%p** |

Ternary IIR 추가 후 Λ는 사실상 insensitive parameter.

---

## 5. 요약 통계

### 추가할 반응/종 수

| 카테고리 | 반응 수 | Rate 형태 |
|----------|:-------:|-----------|
| A. 이온화 | 5 | BOLSIG+ cross-section (Method B) |
| B. 해리 재결합 | 8 | k = A·(300/Tₑ)^n [Tₑ in K] |
| C. Charge Transfer | 10 | 상수 k [cm³/s] |
| D. Clustering | 2 | 1 atm effective k [cm³/s] |
| E. Attachment | 3 | Tₑ/Tg-dependent + cross-section |
| F. Detachment | 4 | 상수 k [cm³/s] |
| G. Ion-Ion 재결합 (binary) | 18 | molecular: k = 2e-7·(300/T)^0.5, cluster: k = 1e-7 |
| H. Ion-Ion 재결합 (ternary) | 18 | k₃ = 2e-25·(300/T)^2.5 cm⁶/s |
| **합계** | **68** | |

### 모델 규모 변화

| 항목 | 초기 (neutral only) | 이온 추가 후 | IIR 확장 후 (현재) |
|------|:-------------------:|:----------:|:-----------------:|
| Species | 51 | 63 (+12) | **63** |
| Reactions | 139 | 177 (+38) | **207** (+30 IIR) |
| 양이온 | 0 | 10 | **10** |
| 음이온 | 0 | 2 | **2** |
| Electron-impact | 21 | 28 | **28** |
| Arrhenius (order 2) | 118 | 148 | **160** (+12 binary IIR) |
| Arrhenius (order 3) | — | 11 | **29** (+18 ternary IIR) |

---

## 6. 구현 시 주의사항

### 6.1 단위 변환

문헌의 rate coefficient는 **CGS (cm³/s, cm⁶/s)** 단위.
현재 모델은 **SI (m³/mol·s)** 기반. 변환 필요:

```
k [m³/s] = k [cm³/s] × 1e-6
k [m³/s] (concentration basis) = k [cm³/s] × 1e-6 × N_A  (if using mol/m³)
k [m⁶/s] = k [cm⁶/s] × 1e-12  (3-body)
```

### 6.2 Dissociative Recombination의 Te 계산

DR rates는 Te(K) 기반. 현재 모델의 state variable은 eps_mean(eV):

```
Te(K) = eps_mean(eV) × 11604 × 2/3
      = eps_mean(eV) × 7736
```

단, 이것은 Maxwellian EEDF 가정. Non-Maxwellian EEDF에서는 approximate.

### 6.3 Three-body Attachment의 Te/Tg 의존성

AT1 (e + O₂ + M -> O₂⁻ + M)의 rate는 Te와 Tg **모두**에 의존:

```
k = 1.1e-31 × (300/Te)² × exp(-70/Tg) × exp(1500×(Te-Tg)/(Te×Tg))
```

Te 단위: Kelvin. Discharge 중 Te >> Tg이면 attachment 매우 느림.
Afterglow에서 Te -> Tg이면 attachment 빠르게 증가.

### 6.4 Quasi-neutrality 확인

이온 추가 후 반드시 확인:

```
n_e + sum(n_negative) = sum(n_positive)
```

이것은 ODE constraint가 아닌 diagnostic check.
충분히 작은 dt에서 자연스럽게 만족되어야 함.

### 6.5 Ionization Rate의 BOLSIG+ 연동

현재 모델에서 ionization cross-section이 BOLSIG+ 입력에 포함되어 있다면,
BOLSIG+ transport coefficient A23 (growth rate)에 이미 net ionization이 반영됨.
이 경우 explicit ionization reaction과 double-counting 주의.

**권장**: ionization은 explicit reaction으로 처리하고,
BOLSIG+ A23은 electron energy equation의 energy loss term으로만 사용.

---

## 7. 참고문헌 (Full Citation)

### 핵심 참고문헌

1. **Sheehan, C.H. & St-Maurice, J.-P.** (2004). Dissociative Recombination of N₂⁺, O₂⁺, and NO⁺.
   *J. Geophys. Res.* 109, A03302. DOI: 10.1029/2003JA010132
   -> DR1, DR2, DR8 (N₂⁺, O₂⁺, NO⁺)

2. **Sheehan, C.H. & St-Maurice, J.-P.** (2004). Dissociative recombination of the methane family ions.
   *Adv. Space Res.* 33, 216-220. DOI: 10.1016/j.asr.2003.07.019
   -> DR4 (CH₄⁺ branching ratios)

3. **Florescu-Mitchell, A.I. & Mitchell, J.B.A.** (2006). Dissociative Recombination.
   *Physics Reports* 430, 277-374. DOI: 10.1016/j.physrep.2006.04.002
   -> DR3, DR5, DR7 (CO₂⁺, CH₃⁺, O₄⁺)

4. **Cao, Y.S. & Johnsen, R.** (1991). Recombination of N₄⁺ ions with electrons.
   *J. Chem. Phys.* 95, 7356-7359. DOI: 10.1063/1.461361
   -> DR6 (N₄⁺, very fast)

5. **Anicich, V.G.** (1993). Evaluated Bimolecular Ion-Molecule Gas Phase Kinetics.
   *J. Phys. Chem. Ref. Data* 22, 1469-1569. DOI: 10.1063/1.555940
   -> CT1-CT10 (ion-molecule charge transfer)

6. **Kossyi, I.A. et al.** (1992). Kinetic scheme of the non-equilibrium discharge in nitrogen-oxygen mixtures.
   *Plasma Sources Sci. Technol.* 1, 207. DOI: 10.1088/0963-0252/1/3/011
   -> AT1 (three-body attachment)

7. **Capitelli, M. et al.** (2000). *Plasma Kinetics in Atmospheric Gases*. Springer.
   -> DT1-DT4, II1-II6, AT3 (detachment, ion-ion recombination)

8. **Troe, J.** (2005). N₂⁺ + N₂ + M association.
   *Phys. Chem. Chem. Phys.* 7, 1560. DOI: 10.1039/b417945a
   -> CL1 (N₄⁺ formation falloff)

9. **Dheandhanoo, S. & Johnsen, R.** (1983). Three-body association of NO⁺, O₂⁺, N⁺, N₂⁺.
   *Planet. Space Sci.* 31, 933. DOI: 10.1016/0032-0633(83)90145-0
   -> CL2 (O₄⁺ formation)

### Cross-section 데이터베이스

10. **Itikawa, Y.** (2006). Cross Sections for Electron Collisions with Nitrogen Molecules.
    *J. Phys. Chem. Ref. Data* 35, 31-53. DOI: 10.1063/1.1937426
    -> I1 (N₂ ionization cross-section)

11. **Itikawa, Y.** (2009). Cross Sections for Electron Collisions with Oxygen Molecules.
    *J. Phys. Chem. Ref. Data* 38, 1-20. DOI: 10.1063/1.3025886
    -> I2 (O₂ ionization cross-section)

12. **Liu, Y. et al.** (2025). An updated set of electron-impact cross sections for CO₂.
    *Plasma Sources Sci. Technol.* 34, 035003. DOI: 10.1088/1361-6595/adba86
    -> I5 (CO₂ ionization cross-section, latest)

13. **LXCat**: https://lxcat.net — Phelps/Morgan database
    -> I3, I4 (CH₄ ionization), AT2 (O₂ dissociative attachment)

### 모델 참고 논문 (Bogaerts 그룹)

14. **Wang, W., Snoeckx, R., Zhang, X., Cha, M.S. & Bogaerts, A.** (2018).
    Modeling Plasma-based CO₂ and CH₄ Conversion: the Bigger Plasma Chemistry Picture.
    *J. Phys. Chem. C* 122, 4842-4859. DOI: 10.1021/acs.jpcc.7b10619

15. **De Bie, C., van Dijk, J. & Bogaerts, A.** (2015).
    The Dominant Pathways for the Conversion of Methane into Oxygenates and Syngas in a DBD.
    *J. Phys. Chem. C* 119, 22331-22350. DOI: 10.1021/acs.jpcc.5b06515

16. **Snoeckx, R. et al.** (2016). CO₂ conversion in a DBD: N₂ in the mix.
    *Energy Environ. Sci.* 9, 999-1011. DOI: 10.1039/C5EE03304G

### 음이온 관련 최신 보정

17. **Shuman, N.S. et al.** (2023). O⁻ + N₂ -> N₂O + e remeasurement.
    *Phys. Chem. Chem. Phys.* 25, 31917. DOI: 10.1039/D3CP03856D
    -> Kossyi 값 보정: 300K에서 ~1e-17 cm³/s (무시 가능)

---

## 8. Future Work: Gas Temperature 발열항

이온 화학 추가와 별개로, gas temperature 방정식에 화학반응 발열항 추가 필요:

| 반응 | Delta_H (eV) | 비고 |
|------|:------------:|------|
| O + O + M -> O₂ + M | -5.12 | 매우 큰 발열 |
| H + OH + M -> H₂O + M | -5.11 | 매우 큰 발열 |
| H + H + M -> H₂ + M | -4.48 | |
| CO + O + M -> CO₂ + M | -5.51 | |
| CH₃ + H + M -> CH₄ + M | -4.55 | |

현재 gas_thermal.py에는 elastic heating만 포함. 이 발열항이 없으면 Tgas 과소평가.

---

## 9. 독립 그룹 교차검증 결과 (2026-03 추가)

초기 조사가 Bogaerts 그룹 6편에 한정되어 있었으므로, IST-Lisbon, Kushner, Capitelli, TRINITI 그룹
및 2020-2025 최신 리뷰를 추가 조사하여 38-reaction 세트의 구조와 rate를 교차검증.

### 9.1 구조 검증

| 그룹 | 핵심 논문 | 시스템 | 우리 세트 호환 |
|------|-----------|--------|:--------------:|
| Kushner (Michigan) | Dorai & Kushner 2001 (J. Phys. D 34) | N₂/O₂ DBD NOx | ✅ 동일 cascade |
| Kushner | Dorai & Kushner 2003 (J. Phys. D 36) | N₂/O₂/hydrocarbon DBD | ✅ CH₃⁺/CH₄⁺/CH₅⁺ 포함 |
| Kushner | Meyer & Kushner 2025 (J. Appl. Phys. 137) | Ar/CH₄/O₂ ns-DBD | ✅ CH₃⁺/CH₄⁺ primary |
| IST-Lisbon | Viegas et al. 2023 (PSST 32) | O₂ DC glow 0D/1D 비교 | ✅ 0D 검증 |
| IST-Lisbon | Silva et al. 2024 (J. Phys. Chem. A) | N₂/O₂ | ✅ Kossyi 기반 |
| TRINITI | Pancheshnyi 2013 (J. Phys. D 46) | N₂/O₂ air | ✅ 핵심 이온 동일 |
| Capitelli (Bari) | Pietanza et al. 2022 (PSST 31) | CO₂ glow | ✅ CO₂⁺ DR 일치 |
| 독립 (KAUST) | Bang, Snoeckx & Cha 2023 | CH₄/CO₂ DBD | ✅ IST-Lisbon + TRINITI DB 인용 |
| 독립 (ZDPlasKin) | Sun et al. 2022 (J. Phys. D 55) | He/CH₄/O₂ NRP | ✅ rate 민감도 "insignificant" |

**결론**: 4개 독립 그룹 + 3개 독립 논문 모두 동일한 이온 cascade 구조 사용.
38-reaction 세트의 species 선택과 pathway 구조에 누락 없음 확인.

### 9.2 Rate Coefficient 검증

| 반응 | 우리 값 | 독립 그룹 값 | 차이 | 판정 |
|------|---------|-------------|------|------|
| DR1-DR4, DR6-DR8 | Sheehan/Florescu-Mitchell | 모든 그룹 동일 | <20% | ✅ |
| CT1-CT10 | Anicich 1993 | Kushner/IST-Lisbon 동일 | 일치 | ✅ |
| CL1 (N₄⁺ formation, k₀) | 6.8e-29 (Troe 2005) | ~5e-29 (IST-Lisbon) | ~25% | ✅ 불확실성 이내 |
| CL2 (O₄⁺ formation, k₀) | 3.4e-30 (Dheandhanoo) | 2.4e-30 (Kushner) | ~40% | ✅ 불확실성 이내 |
| AT1 (3-body attachment) | Kossyi 1992 | Vialetto & Hara 2025 검증 | 일치 | ✅ |
| II1-II6 (ion-ion recomb.) | 2e-7·(300/T)^0.5 | 모든 그룹 동일 | 일치 | ✅ |
| **DR5 (CH₃⁺)** | ~~3.75e-7~~ → **1.1e-8** | Vejby-Christensen 1997 | **34×** | 🔴 보정 완료 |
| **DT2 (O₂⁻+O)** | 1.5e-10 (Capitelli) | 3.3e-10 (Kushner/Kossyi) | **~2.2×** | ⚠️ 확인 필요 |

### 9.3 Primary Source 현행성

| 출처 | 상태 | 비고 |
|------|------|------|
| Kossyi 1992 | 부분 유효 | N₂/O₂ 사실상 표준. 포괄적 후계자 없음. 선별적 업데이트만 |
| Sheehan 2004 | ✅ 현행 | DR rate 기준으로 유효 |
| Florescu-Mitchell 2006 | ✅ 현행 | 포괄적 DR 리뷰로 표준. **단, CH₃⁺ 값은 과대평가** |
| Capitelli 2000 | 부분 유효 | 이온 rate OK. **σ set은 outdated** |
| Anicich 1993 | ✅ 현행 | 이온-분자 반응 여전히 유효 |

---

## 10. 조건부 추가 반응 (현재 38-reaction 세트 외)

현재 모델의 species 구성에 따라 추가를 고려할 수 있는 반응들.
**38-reaction 핵심 세트에는 불포함 — 해당 종이 모델에 있을 때만 추가.**

### 10.1 H₂O 3-body Attachment

CH₄ 산화 생성물로 H₂O가 축적될 경우:

| ID | 반응 | k | 비고 | Ref |
|----|------|---|------|-----|
| AT4 | e + O₂ + H₂O -> O₂⁻ + H₂O | ~1.75e-29 cm⁶/s | H₂O는 O₂보다 **7× 효과적** | Vialetto & Hara 2025, PSST, 10.1088/1361-6595/adbb18 |

검증 실험: de Urquijo et al. 2023, J. Phys. D 57, 125205.

### 10.2 O₂(a¹Δg) Detachment

O₂(a¹Δg) singlet delta oxygen이 species로 포함된 경우:

| ID | 반응 | k (cm³/s) | 비고 | Ref |
|----|------|:---------:|------|-----|
| DT5 | O⁻ + O₂(a¹Δg) -> O₃ + e | 2.2e-11 | O₂(a¹Δg)에 의한 detachment | Pancheshnyi 2013, J. Phys. D 46, 155201 |

현재 DT3/DT4는 N₂(A) 구동 detachment만 포함.
O₂(a¹Δg)가 축적되는 조건에서는 DT5 추가 필요.

### 10.3 CO₃⁺ Cluster Ion

CO₂ 비율이 높은 경우 (>10%):

| ID | 반응 | k (cm³/s) | 비고 | Ref |
|----|------|:---------:|------|-----|
| — | CO₂⁺ + CO₂ -> CO₃⁺ + CO | 1.0e-11 | 대기압에서 cluster 형성 가능 | Pietanza et al. 2022 (Capitelli group) |

현재는 CO₂ 비율이 낮을 경우 무시 가능. CO₂-rich 조건에서만 고려.

---

## 11. BOLSIG+ 입력 데이터 업데이트 권장사항

### 11.1 CO₂ Cross-Section (🔴 필수)

**Liu et al. 2025** (IST-Lisbon, PSST 34, 035003, DOI: 10.1088/1361-6595/adba86):
- 기존 Phelps 7 eV CS: **15%만 해리** (나머지는 electronic excitation)
- 기존 Phelps 10.5 eV CS: **100% 해리** → O(³P) + CO(a³Πr)
- 기존 Polak & Slovetsky CS: CO₂ 전환율 과소평가
- **권장**: LXCat IST-Lisbon 2025 database에서 업데이트된 CO₂ CS 다운로드

### 11.2 N₂ Cross-Section (⚠️ 선택)

**Kawaguchi et al. 2021** (PSST, DOI: 10.1088/1361-6595/abe1d4):
- MuroranIT database on LXCat
- Phelps & Pitchford 1985 대비 개선된 N₂ CS set

### 11.3 O₂ Cross-Section (⚠️ 선택)

**Kawaguchi et al. 2025** (PSST, DOI: 10.1088/1361-6595/ade626):
- MuroranIT database on LXCat — 최신 O₂ CS set
- 3-body attachment: Biagi/Taniguchi CS set 검증됨 (Vialetto & Hara 2025)

### 11.4 Cross-Section 데이터베이스 정리

| 분자 | 현재 소스 | 권장 소스 | LXCat DB명 |
|------|-----------|-----------|------------|
| CO₂ | Phelps/Polak | **IST-Lisbon 2025** | IST-Lisbon |
| N₂ | Itikawa 2006 | Kawaguchi 2021 (선택) | MuroranIT |
| O₂ | Itikawa 2009 | Kawaguchi 2025 (선택) | MuroranIT |
| CH₄ | Phelps/Morgan | 변경 불필요 | Phelps |

---

## 12. 추가 참고문헌 — 교차검증 (2026-03)

18. **Vejby-Christensen, L. et al.** (1997). Complete Branching Ratios for the Dissociative
    Recombination of H₂O⁺, H₃O⁺, and CH₃⁺.
    *Astrophysical Journal* 483, 531. DOI: 10.1086/304242
    -> DR5 보정 (CH₃⁺ DR, ASTRID storage ring 직접 측정)

19. **Dorai, R. & Kushner, M.J.** (2001). A model for plasma modification of polypropylene
    using atmospheric pressure discharges.
    *J. Phys. D: Appl. Phys.* 34, 574. DOI: 10.1088/0022-3727/34/4/319
    -> Kushner group N₂/O₂ DBD 이온 화학 교차검증

20. **Dorai, R. & Kushner, M.J.** (2003). Consequences of unburned hydrocarbons on
    microstreamer dynamics and chemistry during plasma remediation of NOₓ.
    *J. Phys. D: Appl. Phys.* 36, 1075. DOI: 10.1088/0022-3727/36/9/305
    -> Kushner group hydrocarbon 이온 화학

21. **Meyer, C., Hartman, N.Z. & Kushner, M.J.** (2025). Oxygenates production in
    a microfluidic DBD in Ar/CH₄/O₂.
    *J. Appl. Phys.* 137. DOI: 10.1063/5.0239464
    -> 최신 Kushner group CH₄ 이온 화학

22. **Liu, Y. et al.** (2025). An updated set of electron-impact cross sections for CO₂:
    untangling dissociation.
    *Plasma Sources Sci. Technol.* 34, 035003. DOI: 10.1088/1361-6595/adba86
    -> CO₂ σ 업데이트 (IST-Lisbon 2025)

23. **Vialetto, L. & Hara, K.** (2025). Monte Carlo simulations for three-body attachment
    in humid air.
    *Plasma Sources Sci. Technol.* DOI: 10.1088/1361-6595/adbb18
    -> 3-body attachment 검증, H₂O 효과

24. **Kawaguchi, S. et al.** (2021). Electron collision cross section set for N₂.
    *Plasma Sources Sci. Technol.* DOI: 10.1088/1361-6595/abe1d4
    -> N₂ σ 업데이트 (MuroranIT DB)

25. **Kawaguchi, S. et al.** (2025). Electron collision cross section set of O₂.
    *Plasma Sources Sci. Technol.* DOI: 10.1088/1361-6595/ade626
    -> O₂ σ 업데이트 (MuroranIT DB)

26. **Pancheshnyi, S.** (2013). Effective ionization rate in nitrogen–oxygen mixtures.
    *J. Phys. D: Appl. Phys.* 46, 155201. DOI: 10.1088/0022-3727/46/15/155201
    -> O₂(a¹Δg) detachment, effective ionization

27. **Silva, T. et al.** (2024). Unraveling NO Production in N₂–O₂ Plasmas.
    *J. Phys. Chem. A.* DOI: 10.1021/acs.jpca.4c03323
    -> IST-Lisbon N₂/O₂ 모델, Kossyi 비교

28. **Bang, S., Snoeckx, R. & Cha, M.S.** (2023). CH₄/CO₂ DBD 모델.
    *Plasma Chem. Plasma Process.* DOI: 10.1007/s11090-023-10370-7
    -> 독립 교차검증 (KAUST group)

29. **Sun, J. et al.** (2022). Temperature-dependent ion chemistry in ns discharge
    plasma-assisted CH₄ oxidation.
    *J. Phys. D: Appl. Phys.* 55, 135203. DOI: 10.1088/1361-6463/ac45ac
    -> CH₄ 이온 화학 민감도 분석 ("insignificant sensitivity")

30. **Zhang, Y. et al.** (2022). DRM in NRP discharge.
    *Plasma Sources Sci. Technol.* DOI: 10.1088/1361-6595/ac6bbc
    -> Bogaerts group 최신 ZDPlasKin 모델

31. **Yi, Y., Slaets, J. et al.** (2023). CH₄/O₂ in DBD.
    *ACS Sustainable Chem. Eng.* DOI: 10.1021/acssuschemeng.3c04352
    -> Bogaerts group DBD, 업데이트된 rate set

32. **Pietanza, L.D., Colonna, G. & Capitelli, M.** (2022). Non-equilibrium plasma
    kinetics of CO₂ in glow discharges.
    *Plasma Sources Sci. Technol.* 31. DOI: 10.1088/1361-6595/ac9083
    -> Capitelli group CO₂ 이온 화학 (CO₃⁺ cluster)

33. **Viegas, P. et al.** (2023). Comparison between 1D radial and 0D global models
    for low-pressure oxygen DC glow discharges.
    *Plasma Sources Sci. Technol.* 32, 024002. DOI: 10.1088/1361-6595/acbb9c
    -> IST-Lisbon 0D 모델 검증

34. **Pitchford, L.C. et al.** (2017). LXCat: an open-access, web-based platform.
    *Plasma Processes and Polymers.* DOI: 10.1002/ppap.201600098
    -> LXCat 데이터베이스

35. **de Urquijo, J. et al.** (2023). Electron attachment in humid air.
    *J. Phys. D: Appl. Phys.* 57, 125205.
    -> 3-body attachment 실험 검증

---

*이 문서는 plasma0d_v2 이온 화학 확장 작업의 참고자료로 사용됩니다.*
*최종 업데이트: 2026-03-12 (IIR binary 12반응 + ternary 18반응 추가, Λ 민감도 해소)*
