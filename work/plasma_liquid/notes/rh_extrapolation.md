# RH 80% Gas-Phase Concentration Extrapolation

## 목적

액상 실험은 RH ~80% 환경(petri dish 위 수증기)에서 수행되나, OAS 기상 측정 데이터는 Dry/25%/55%/65%만 존재.
RH 80% 기상 농도를 외삽하여 시뮬레이션 input으로 사용.

## 가용 데이터

- Source: `OAS data/Dry/` (xlsx) + `OAS data/Humid/` (csv)
- 전압: 2.6, 3.2, 3.6 kVpp
- RH: Dry(~0%), 25%, 55%, 65%
- 종: O₃, NO₂, NO₃, N₂O₅, HONO (Humid에서만 HONO 측정)
- 시간: 0~600s, 2s 간격

## Data point 선정

- **Steady-state 값**: 각 시계열의 마지막 100초 평균
- **RH 25% 제외**: 전반적으로 경향에서 벗어나는 이상점 (O₃ 2.6kV에서 증가, N₂O₅ 비단조 등). 문헌의 단조 경향과 불일치.
- **사용 점**: RH = Dry(0%), 55%, 65% (3점)

## 외삽 방법: 비율 기반 + 물리적 fitting 함수

### 원리

절대 농도 대신 **종 간 비율**을 fitting하면:
- 전압 간 일관성 향상 (비율은 전압에 덜 의존)
- 물리적 메커니즘에서 함수형 유도 가능
- 외삽 안정성 증가

### Anchor 종: O₃

O₃는 가장 높은 농도, 가장 안정적 경향 (단조 감소). 직접 선형 외삽.
```
O₃(RH) = a + b × RH          (선형, 단조 감소)
```

### 비율 fitting 함수 및 물리적 근거

#### 1. N₂O₅/NO₂ = A / (1 + B × RH²)

**물리적 근거**: N₂O₅ 정상상태
```
생성: NO₂ + NO₃ → N₂O₅                    (k_f)
소비: N₂O₅ + H₂O → 2HNO₃                  (k_h1 = 2.5×10⁻²² cm³/molec/s)
      N₂O₅ + 2H₂O → 2HNO₃ + H₂O          (k_h2 = 1.8×10⁻³⁹ cm⁶/molec²/s)
```

정상상태: [N₂O₅]/[NO₂] = K_eq × [NO₃] / (k_h1[H₂O] + k_h2[H₂O]²)

RH 50%에서 k_h2[H₂O]² ≈ 3 × k_h1[H₂O] → **수증기 dimer 반응 지배**.
[H₂O] ∝ RH이므로: N₂O₅/NO₂ ∝ 1/(1 + const × RH²)

**문헌**: Wahner 1998 (GRL), JPC A 2014, PNAS 2022.

**Fitting 결과 (3.2kV)**: A=1.39, B=3.90×10⁻³. RH 80%: 비율=0.054.

#### 2. HONO/NO₂ = A × RH

**물리적 근거**: 표면 반응
```
NO₂ + H₂O(surface) → HONO + HNO₃          (heterogeneous)
```

[HONO]/[NO₂] ∝ [H₂O]_surface ∝ RH (저~중 RH에서 선형)

**문헌**: Stutz 2004 (JGR) — HONO/NO₂ < 0.04 at RH 10-30%, up to 0.09 at high RH.
Nature 2024 — kNO₂→HONO = 0.7e-3~2.5e-3 min⁻¹ as RH 5%→79%.

**Fitting 결과 (3.2kV)**: A=8.84×10⁻⁵. RH 80%: 비율=0.0071.

#### 3. NO₂/O₃ = A + B × RH

**물리적 근거**: O₃→NOₓ mode 전환. RH 증가 → OH 증가 → O₃ 소비 + NO₂ 재분배.
경험적 선형 관계.

**Fitting 결과 (3.2kV)**: A=0.015, B=9.54×10⁻⁴. RH 80%: 비율=0.091.

#### 4. NO₃/O₃ = A + B × RH

NO₃ 라디칼. Scatter 크지만 약한 양의 경향.

**Fitting 결과 (3.2kV)**: A=0.0013, B=3.90×10⁻⁵. RH 80%: 비율=0.0044.

### 농도 복원 체인

```
O₃(80%)  = linear fit                       (anchor)
NO₂(80%) = O₃(80%) × (NO₂/O₃ ratio @80%)
N₂O₅(80%)= NO₂(80%)× (N₂O₅/NO₂ ratio @80%)
HONO(80%)= NO₂(80%)× (HONO/NO₂ ratio @80%)
NO₃(80%) = O₃(80%) × (NO₃/O₃ ratio @80%)
```

## 결과: RH 80% 예측 농도

### 3.2 kVpp

| Species | RH 80% (cm⁻³) | ppm | Dry 대비 |
|---------|-------------|-----|---------|
| O₃ | 6.66×10¹⁶ | 2707 | -35% |
| NO₂ | 6.09×10¹⁵ | 248 | +3.4× |
| N₂O₅ | 3.57×10¹⁴ | 14.5 | -86% |
| HONO | 4.72×10¹³ | 1.9 | (Dry=0) |
| NO₃ | 2.94×10¹⁴ | 12.0 | +1.7× |

### 2.6 kVpp

| Species | RH 80% (cm⁻³) | ppm |
|---------|-------------|-----|
| O₃ | 1.66×10¹⁶ | 674 |
| NO₂ | 3.68×10¹⁵ | 150 |
| N₂O₅ | 1.95×10¹⁴ | 7.9 |
| HONO | 4.16×10¹³ | 1.7 |
| NO₃ | 2.97×10¹⁴ | 12.1 |

### 3.6 kVpp

| Species | RH 80% (cm⁻³) | ppm |
|---------|-------------|-----|
| O₃ | 8.47×10¹⁶ | 3445 |
| NO₂ | 8.04×10¹⁵ | 327 |
| N₂O₅ | 3.09×10¹⁴ | 12.6 |
| HONO | 5.59×10¹³ | 2.3 |
| NO₃ | 2.85×10¹⁴ | 11.6 |

## 미측정 종 (HNO₃, H₂O₂)

OAS에서 측정되지 않은 종. 비율 기반 추정 (notes/unmeasured_gas_species.md):
- **HNO₃**: N₂O₅ + H₂O → 2HNO₃ 산물. RH↑ → 증가. HNO₃/N₂O₅ = 0.83 (median 추정)
- **H₂O₂**: OH + OH → H₂O₂. RH↑ → 증가. H₂O₂/O₃ = 0.03 (median 추정)

이 비율은 humid 조건 문헌 기반이므로 RH 80%에서 그대로 적용 가능.

## 스크립트

- `Figures/test/test_rh_ratio_fit.py` — 비율 fitting + 외삽 + plot
- `Figures/test/test_rh_extrapolation.py` — 직접 외삽 (참고용, 비율 방식이 더 우수)

## 문헌

- Wahner et al. (1998) GRL — N₂O₅ + H₂O gas-phase rate constants
- JPC A (2014) — N₂O₅ + H₂O computational kinetics
- PNAS (2022) — amine-promoted N₂O₅ hydrolysis
- Stutz et al. (2004) JGR — HONO/NO₂ vs RH field observations
- Nature (2024) — nocturnal HONO formation vs RH
- Jogi et al. (2023) Plasma Chem. Plasma Process. — sDBD species vs RH

---
<!-- Last updated: 2026-04-14 -->
