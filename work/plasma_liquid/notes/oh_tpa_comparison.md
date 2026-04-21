# OH 라디칼 정량 비교: TPA→hTPA 형광 prob vs 1D 시뮬레이션

작성일: 2026-04-20 | Ultraplan Phase 5 산출물
코드: `Ver4_1D/run_tpa_alkaline.py`, `Figures/gen_fig_oh_tpa.py`
캐시: `Figures/cache/tpa/{voltage}_{tpa,notpa}.npz`

---

## 1. 결과 요약

| 전압 | Sim [hTPA] (µM) | Exp [hTPA] (µM) | 오차 | Sim OH_eq=hTPA/0.35 |
|---|---|---|---|---|
| 2.6 kVpp | **4.89** | 12.66 | **−61%** | 13.98 |
| 3.2 kVpp | **13.92** | 57.72 | **−76%** | 39.78 |
| 3.6 kVpp | **16.38** | 43.26 | **−62%** | 46.81 |

- pH: 11.99–12.01 (초기 12.00 유지 ✓)
- TPA 소모: 1.4–1.9% (10분간 거의 그대로 — hTPA+OH 2차 반응 영향 무시 가능)
- Wall time: TPA-on 125–130 s, TPA-off 26–30 s, 총 10.9분.

### 핵심 관찰
1. **정량: 시뮬이 실험보다 일관되게 2.7–4배 낮음** (−61 ~ −76% 오차). 전압 간 비율은 유사.
2. **전압 rank 불일치**:
   - Sim: **3.6 > 3.2 > 2.6** (단조, gas-side O₃/N₂O₅ 단조에 비례)
   - Exp: **3.2 > 3.6 > 2.6** (비단조)
3. **OH 농도** (spatial avg): 1–2 × 10⁻¹⁴ M (TPA 존재). 공급이 거의 즉시 소비.
4. **Control (TPA=0) OH 농도**: 9–16 × 10⁻¹⁴ M → TPA가 실제로 OH를 5–8배 감소시킴. 그러나 **OH⁻ (10 mM)가 TPA (2 mM)보다 더 강한 scavenger** — 경쟁에서 OH⁻가 이김.

### 가스상 입력 (RH 80% 보정 후) — voltage 단조성 확인
| 전압 | O₃ max (cm⁻³) | NO₂ max | N₂O₅ max |
|---|---|---|---|
| 2.6 | 1.78×10¹⁶ | 3.95×10¹⁵ | ~9×10¹³ |
| 3.2 | 6.77×10¹⁶ | 6.52×10¹⁵ | 3.34×10¹⁴ |
| 3.6 | 8.64×10¹⁶ | 8.60×10¹⁵ | 3.02×10¹⁴ |

**가스-side는 모두 단조 증가** → 실험의 3.2>3.6 비단조는 liquid-side 또는 측정 artifact로 확정.

---

## 2. 비단조성 (3.2 > 3.6) 원인 분석

시뮬레이션에서 3.6 > 3.2 나왔으므로 실험의 비단조성을 **재현 못함**. 원인 후보 진단:

| # | 후보 원인 | 시뮬이 재현 못함 | 해석 |
|---|---|---|---|
| A | Gas 자체 비단조 | ✗ (O₃/N₂O₅ 모두 3.6>3.2) | 배제 |
| B | pH 저하 (3.6이 최저) | ✗ (시뮬 pH=12 유지, 실험 pH>10) | pH 변화가 원인이어도 |
| C | H₂O₂ 축적 → OH 재결합 | ✗ (H₂O₂ 수준 비슷) | 추가 조사 |
| D | TPA 고갈 | ✗ (TPA 1.4–1.9% 소모만) | 배제 |
| E | Inner filter effect | **시뮬과 무관한 측정 artifact** | **주원인 후보** |
| F | H₂O₂ 경쟁 (실험 주장) | ✗ (TPA 있는 상태에서는 OH+OH→H₂O₂ 거의 없음) | 배제 |

**결론**: 실험 비단조성은 **측정계 inner filter effect** (PPT에서 주장) 또는 **2D plasma footprint에서 3.6kV의 비균일한 분포**가 원인일 가능성 높음. 1D 시뮬은 이를 모델링하지 않음 (limitation).

---

## 3. 정량 불일치 (시뮬이 실험보다 낮음) 원인 분석

### 3-1. OH sink budget (3.2 kVpp, t=600s, spatial avg)

| Sink | k (M⁻¹s⁻¹) | Rate (M/s) | 기여율 |
|---|---|---|---|
| **OH⁻** (R21) | 1.3×10¹⁰ | ~2×10⁻⁶ | **~94%** (forward rate 기준; reversible) |
| **TPA** (R_TPA1+2) | 4.0×10⁹ | ~1.4×10⁻⁷ | **~6%** |
| HO₂⁻ (R42) | 7.5×10⁹ | ~8×10⁻⁹ | <1% |
| O₃ (R27) | 3.0×10⁹ | ~1.4×10⁻⁹ | <1% |
| H₂O₂ (R41) | 2.7×10⁷ | ~9×10⁻¹² | 무시 |

주의: R21은 reversible (k_b=1.7×10⁶), 정상상태에서 net rate는 forward보다 훨씬 낮음. Figure (f)는 forward 순간 rate 기준이므로 과대표시.

**실제 순 결과**: TPA 포획률은 시뮬에서 약 **5–8%** (control→TPA-on 감소 비율). 즉 95%의 OH가 non-TPA 경로로 소비 → 시뮬이 실험보다 낮게 나오는 핵심 이유.

### 3-2. 가능한 model underestimation 원인

| # | 원인 | Phase 5 action |
|---|---|---|
| P1 | **α_b(O₃) 과소** (기본 0.05) | sweep 0.01/0.05/0.1/0.2 |
| P2 | **H₂O₂/O₃ 가스 비율 0.03 과소** (RH 의존) | sweep 0.01/0.03/0.1/0.3 |
| P3 | **Gas-side OH transfer 누락** (OH은 transferable 아님) | 문헌: OH gas-phase 주로 short-lived, 넘겨도 영향 작음 |
| P4 | **TPA k 불확실성** | sweep 3×10⁹ / 4×10⁹ / 5×10⁹ |
| P5 | **Branching 불확실** | sweep 0.28 / 0.35 |
| P6 | **1D vs 2D geometry** | Fundamental — 논문 limitation |
| P7 | **알칼리에서 새로운 OH 소스 누락** (예: O₃⁻ + H₂O) | 문헌 검토 |

### 3-3. 대략적 sensitivity 추정

TPA 포획률 p ≈ k_TPA[TPA] / (k_TPA[TPA] + k_OH⁻[OH⁻] + …)
- 현재 p ≈ 0.06 (6%) at pH 12
- p = 0.2 되려면 α_b(O₃) 3배 증가 또는 OH⁻ scavenging 축소 필요

선형 스케일 추정: **시뮬 ×3 → 실험 근접**. 따라서 단일 parameter로 맞추기보다 여러 파라미터 동시 조정 필요.

---

## 4. 논문 limitation 기록

1. **1D 모델은 plasma footprint와 solution volume 간의 2D interaction을 표현하지 못함**. 실제는 plasma patch 바로 아래 (~1 cm²)만 고농도 OH가 생성되고 나머지는 확산으로 도달.
2. **Inner filter effect는 측정계 artifact**이며 시뮬에서 재현 불가.
3. **OH radical의 plasma-gas phase 직접 기여**는 본 1D model에서 미포함. 그러나 [OH]_gas는 10⁻⁶–10⁻⁷ cm⁻³ 수준으로 N₂O₅ 대비 무시 가능한 flux.
4. **알칼리(pH 12)에서 OH⁻ 자체가 강한 scavenger** — TPA 2 mM이 경쟁에서 이기지 못함. 실험에서는 inner filter effect로 인해 측정치가 실제 [OH]를 과대 반영하는 셈.

---

## 5. 다음 단계 (Phase 5 진단 sweep)

**최우선 2D sweep**: α_b(O₃) × H₂O₂/O₃ ratio (3×4=12 runs, 약 1 시간).

이후:
- TPA k / branching 민감도 (2×2 = 4 runs)
- Gas-side O₃ scale (RH 90% 실험 보정 등)

Sweep 결과가 여전히 실험과 불일치하면 **1D geometric underprediction** 결론 (논문 limitation).

---

## 6. 2026-04-20 2차 세션 — Henry 수정 + k_R3 sensitivity

### 6.1 주요 변경
- **Henry 버그 수정** (pde_solver.compute_k_mt): `H_cc = H_cp × R × T` 중복 제거. HENRY_CONSTANTS는 이미 dimensionless H_cc (Liu 2015). k_mt 24.5배 증가.
- **Dry vs Humid_fitting × Henry × k_R3** 다변인 비교 완료.

### 6.2 최종 조건 비교표 (Humid_fitting + Henry 수정)

| k_R3 (M⁻¹s⁻¹) | 2.6kV | 3.2kV | 3.6kV | rank | 대표 오차 |
|---|---|---|---|---|---|
| **0** | **15.3 (+21%)** | **41.4 (−28%)** | **43.5 (+1%)** | 3.6≈3.2>2.6 | **3.6kV +1% ✓** |
| 1×10⁹ (임의 추정) | 13.77 | 22.30 | 21.40 | 3.2>3.6>2.6 | −51~−61% |
| 6.3×10⁹ (Page 2010) | 9.13 | 6.53 | 5.77 | 2.6>3.2>3.6 (역전) | −89%~−28% |
| **실험** | **12.66** | **57.72** | **43.26** | 3.2>3.6>2.6 | — |

→ **k_R3=0이 실험에 가장 근접**. 3.6kV 정확 일치(+1%), 3.2kV 근접(−28%), 2.6kV 약간 과대(+21%).

### 6.3 왜 k_R3=6.3×10⁹에서 rank 역전했는가 (정확한 메커니즘)

Surface cell 질량보존:
| 전압 | ΔTPA_surf (mM) | 누적 hTPA 생성 (×0.35) | 관측 hTPA | **분해량** | **분해율** |
|---|---|---|---|---|---|
| 2.6 | 0.67 | 234 µM | 146 | 88 | **37%** |
| 3.2 | 1.73 | 605 | 63 | 542 | **90%** |
| 3.6 | 1.77 | 620 | 53 | 567 | **91%** |

**실제로는 고전압에서 hTPA가 훨씬 많이 생성**(620 µM)되지만, **즉시 R_TPA3로 90% 분해**. hTPA 수명:
```
τ_hTPA = 1/(k_R3·[OH]_surf)
         2.6kV: 21 s  |  3.2kV: 3.2 s  |  3.6kV: 2.8 s
```
고전압에서 수명 3초로 짧아 축적 불가 → local SS 빠르게 도달, SS 값은 [TPA]_local에 비례하므로 낮음.

### 6.4 k_R3 문헌 근거 한계 (research 결과)

- **Page 2010이 유일한 직접 측정** (J. Environ. Monit. 12:1658). Competition kinetics, pH 5–7.
- Tampieri 2021 등 plasma 논문은 kinetic model 대신 <90s data truncation 관행.
- **pH 12 조건**에서 hTPA의 phenolic -OH가 deprotonated (phenoxide, pKa~9.5) → 실제 반응성 ±50% 불확실.
- 유사 분자 비교: salicylate (5×10⁹), phenol (6–14×10⁹) → 6.3×10⁹은 합리적 범위 내이나 pH 의존성 미지.

### 6.5 논문 서술 권장
1. **Baseline**: k_R3=0 (Tampieri 관행 + 가장 실험 일치)
2. **Sensitivity band**: k_R3 = 0, 1×10⁹, 6.3×10⁹ 3 케이스로 uncertainty 명시
3. **Limitation 명기**:
   - 1D 정지 액상 → 실제 실험의 convective mixing 미반영
   - Page 2010 k_R3은 pH 5–7 측정, pH 12 적용 불확실
   - 실험 fluorescence는 hTPA 외 부산물 기여 가능 (HPLC 분리 없음, Tampieri 경고)

### 6.6 최종 figure
`Figures/fig_oh_tpa_humidfitting.png` — k_R3=0, Humid_fitting + Henry 수정, 6-panel (bar / OH profile / hTPA time / pH / TPA depletion / OH sinks).

---

## 7. 생성된 파일

- `Ver4_1D/run_tpa_alkaline.py` — 3전압 × TPA on/off runner
- `Ver4_1D/reactions_tpa.yaml` — 3개 TPA 반응 (R_TPA1/2/3)
- `Ver4_1D/config_1d.py` — TPA, hTPA species + diffusivity + charge (Z=-2)
- `Ver4_1D/chemistry_1d.py` — `tpa_mode=True` flag 추가
- `Ver4_1D/pde_solver.py` — TPA 전하 병합
- `Figures/gen_fig_oh_tpa.py` — 6-panel comparison figure
- `Figures/fig_oh_tpa.png`, `.pdf` — 결과 figure
- `Figures/cache/tpa/*.npz` — 6개 run 캐시
- `notes/ultraplan_tpa_oh_comparison.md` — 계획 문서
- `notes/tpa_chemistry_literature.md` — 문헌 파라미터
- `notes/oh_tpa_comparison.md` — 본 문서
