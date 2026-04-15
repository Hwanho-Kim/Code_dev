# Unmeasured Gas-Phase Species: HONO, HNO₃, H₂O₂

## 배경

FTIR 측정으로 O₃/NO/NO₂/NO₃/N₂O₅는 시계열 데이터 확보.
HONO/HNO₃/H₂O₂는 미측정 (CSV에 0 또는 미포함).
현재 시뮬레이션에서 이 3종의 가스상 농도 = 0 → 액상 H₂O₂ ≈ 0 (실험 11µM과 큰 괴리).

## 추정 범위

### 방법 1: 실측 종 대비 비율 (우리 시스템 기준)

| Species | 비율 근거 | 기준종 (max) | 추정 범위 | 중간값 |
|---------|----------|-------------|----------|--------|
| H₂O₂ | H₂O₂/O₃ ~ 0.01~0.1 | O₃ = 1.05×10¹⁷ cm⁻³ | 10¹⁵~10¹⁶ cm⁻³ | 3×10¹⁵ cm⁻³ (~122 ppm) |
| HONO | HONO/NO₂ ~ 0.1~1 | NO₂ = 3.0×10¹⁵ cm⁻³ | 3×10¹⁴~3×10¹⁵ cm⁻³ | 1×10¹⁵ cm⁻³ (~41 ppm) |
| HNO₃ | HNO₃/N₂O₅ ~ 0.5~2 | N₂O₅ = 1.2×10¹⁶ cm⁻³ | 6×10¹⁵~2.4×10¹⁶ cm⁻³ | 1×10¹⁶ cm⁻³ (~407 ppm) |

### 방법 2: sDBD humid air 문헌 범위 (deep research 결과)

| Species | 범위 (ppm) | 범위 (cm⁻³) | 비고 |
|---------|-----------|------------|------|
| HONO | 1–100 | 2.5×10¹³~2.5×10¹⁵ | 로그 스케일 sweep 권장 |
| HNO₃ | 10–250 | 2.5×10¹⁴~6.1×10¹⁵ | Jogi 2023 max=142ppm |
| H₂O₂ | 1–100 | 2.5×10¹³~2.5×10¹⁵ | 문헌 불확실성 가장 큼, 넓게 잡기 |

### 우리 실측 종 참고값

| Species | max (cm⁻³) | max (ppm) |
|---------|-----------|-----------|
| O₃ | 1.05×10¹⁷ | 4268 |
| NO₂ | 3.0×10¹⁵ | 122 |
| N₂O₅ | 1.2×10¹⁶ | 488 |
| NO₃ | 1.6×10¹⁴ | 6.6 |

n_total = 2.46×10¹⁹ cm⁻³ (1 atm, 298K). 1 ppm = 2.46×10¹³ cm⁻³.

## Sweep 계획

첫 테스트: 방법 1의 중간값 (H₂O₂=3×10¹⁵, HONO=1×10¹⁵, HNO₃=1×10¹⁶)
이후: 방법 2의 범위에서 로그 스케일 sweep

## 문헌 출처

- Jogi et al. (2023) Plasma Chem. Plasma Process. — sDBD FTIR, HNO₃ max=142ppm
- Hybrid DBD Reactor (2025) MDPI Plasma — HNO₃ 142ppm at RH 40%
- FTIR 검출한계: O₃ 70ppm, NO₂ 15ppm, N₂O₅ 5ppm, HNO₃ 15ppm (Jogi 2023)
- H₂O₂ 기상 측정: sDBD FTIR 사례 거의 없음. CRDS로 plasma jet만 측정됨.

---
<!-- Last updated: 2026-04-10 -->
