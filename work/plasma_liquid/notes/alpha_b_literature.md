# Species-Specific Mass Accommodation Coefficients (α_b)

## Overview

Mass accommodation coefficient α_b: 기체 분자가 액체 표면에 충돌하여 bulk liquid에 흡수될 확률 (0~1).
현재 모델은 **단일 α_b (= 0.03)**를 전 종에 적용하나, 문헌에서는 종별로 **4자릿수 이상** 차이.

### 관련 개념 구분
- **α_s (surface accommodation)**: 표면에 흡착될 확률. α_s ≥ α_b 항상 성립.
- **α_b (bulk/mass accommodation)**: 표면 → bulk liquid 전달 확률. 이 문서에서 다루는 값.
- **γ (reactive uptake)**: 순 비가역 흡수 확률. 용해도+확산+반응 포함. γ ≤ α_b 항상 성립.
- 관계식: 1/γ = 1/α_b + 1/Γ_rxn (Γ_rxn: 확산+반응 한계)

---

## Species-Specific α_b Values

### O₃ (Ozone)

| Source | α_b | Notes |
|--------|-----|-------|
| Davidovits et al. 2006 | ~0.05 | Droplet train 실험 |
| IUPAC (Ammann et al. 2013) | > 0.01 | Poorly constrained |
| Kolb et al. 2010 | > 0.01 | γ_obs ~ 10⁻³ (용해도 제한) |
| Plasma-liquid models | 0.01–0.05 | Bruggeman 2016, Liu 2015 |

- **추천값: 0.01–0.05**
- Henry 상수 낮음 (H ~ 1.1×10⁻² M/atm) → 실제 uptake는 용해도 제한, α_b가 아님
- 온도 의존성: 약한 음의 상관 (T↑ → α_b↓)
- pH 의존성: α_b 자체에 없음. 고 pH에서 γ 증가 (OH⁻ 반응)
- O₃ 반응성 침투깊이 ~34µm → 표면에서 빠르게 소비

### H₂O₂ (Hydrogen Peroxide)

| Source | α_b | Notes |
|--------|-----|-------|
| Kolb et al. 2010 | 0.11 ± 0.02 (273K) | **가장 잘 정량화된 종** |
| Davidovits et al. 2006 | 0.18 (260K) → 0.08 (295K) | 강한 온도 의존성 |
| IUPAC (Ammann et al. 2013) | 0.1 | 추천값 |
| Worsnop et al. 1989 | Original droplet train 측정 | |
| Plasma-liquid models | 0.1 | Lindsay 2015, Bruggeman 2016 |

- **추천값: 0.1 (298K)**
- **가장 잘 측정된 α_b** — 높은 용해도 (H ~ 10⁵ M/atm)로 accommodation이 rate-limiting
- 온도 의존성: **강한 음의 상관** — ΔH‡ ≈ −29 kJ/mol, ΔS‡ ≈ −134 J/(mol·K)
  - 260K: 0.18, 273K: 0.11, 293K: 0.08, 298K: ~0.08
- pH 의존성: α_b에 없음

### NO (Nitric Oxide)

| Source | α_b | Notes |
|--------|-----|-------|
| JPL recommendation | γ < 10⁻³ | 순수 물 |
| Kolb et al. 2010 | Not determined | 용해도 너무 낮아 측정 불가 |
| MD simulations (Garrett 2006) | 10⁻³–10⁻² | 이론적 추정 |
| Graves 2012 | ~10⁻³ | Plasma-liquid 모델 |
| Heirman et al. | 5×10⁻⁴–5×10⁻³ | Plasma-liquid 모델 |

- **추천값: ~10⁻³ (poorly constrained)**
- Henry 상수 극히 낮음 (H ~ 1.9×10⁻³ M/atm) → 용해도 제한으로 α_b 실험 측정 불가
- 실측 γ < 10⁻³는 accommodation이 아닌 용해도 한계 반영

### NO₂ (Nitrogen Dioxide)

| Source | α_b | Notes |
|--------|-----|-------|
| IUPAC (Ammann et al. 2013) | > 0.02 | γ ~ 10⁻⁴–10⁻³ (가수분해 제한) |
| Kolb et al. 2010 | ≥ 0.02 | |
| Mertes & Wahner 1995 | γ = 1.5×10⁻³ | Wetted wall, 278K |
| Lee & Schwartz 1981 | — | 가수분해 반응이 rate-limiting |
| Plasma-liquid models | 0.01–0.04 | Bruggeman 2016, Heirman |

- **추천값: 0.02–0.04**
- 2NO₂ + H₂O → HNO₃ + HONO 가수분해가 rate-limiting (accommodation 아님)
- pH 의존성: 고 pH에서 γ 증가 (OH⁻ 반응)

### N₂O₅ (Dinitrogen Pentoxide)

| Source | α_b | Notes |
|--------|-----|-------|
| IUPAC (Ammann et al. 2013) | γ ≈ 0.02–0.03 (298K) | α_b ≥ γ |
| Davidovits et al. 2006 | α_b ≥ 0.04 | |
| Kolb et al. 2010 | 0.03 (추천) | |

- **추천값: ≥ 0.03**
- 빠른 가수분해 (N₂O₅ + H₂O → 2HNO₃) → α_b와 γ 분리 어려움
- Cl⁻에 의해 γ 증가 (ClNO₂ 생성). NO₃⁻에 의해 γ 감소.
- 우리 모델에서 N₂O₅가 N 유입의 99.3% 지배 → **가장 중요한 종**

### HONO (Nitrous Acid, HNO₂)

| Source | α_b | Notes |
|--------|-----|-------|
| IUPAC (Ammann et al. 2013) | 0.05 | 추천값 |
| Davidovits et al. 2006 | 0.04 (298K) | Droplet train |
| Kolb et al. 2010 | 0.05 | |
| Bongartz et al. 1994 | — | Bubble column 실험 |

- **추천값: 0.04–0.05**
- 약산 (pKa ≈ 3.3) → pH > pKa에서 해리 증가 → γ 증가 (α_b 자체는 pH 무관)
- 온도 의존성: 약한 음의 상관 — 273K: ~0.07, 298K: ~0.04

### HNO₃ (Nitric Acid)

| Source | α_b | Notes |
|--------|-----|-------|
| Davidovits et al. 2006 | 0.07 (298K) | Droplet train (gas-diff 보정 후) |
| JPL Eval. No. 19 (Burkholder 2020) | 0.054 (298K) | |
| IUPAC (Ammann et al. 2013) | ≥ 0.05, 추천 0.07 | |
| Kolb et al. 2010 | 0.05 (하한) | |
| Van Doren et al. 1990 | 0.1–0.17 | 초기 측정 (보정 전) |

- **추천값: 0.05–0.07**
- 강산 (pKa ≈ −1.3) → 비가역적 흡수 (역방향 거의 없음)
- 온도 의존성: 음의 상관 — 273K: ~0.10–0.15, 298K: ~0.05–0.07

### OH (Hydroxyl Radical)

| Source | α_b | Notes |
|--------|-----|-------|
| IUPAC (Ammann et al. 2013) | ≥ 0.01 | Firm recommendation 없음 |
| Davidovits et al. 2006 | 0.01–0.04 | Critical cluster model |
| Takami et al. 1998 | γ ≈ 0.0035 | 순수 물 (self-reaction 한계) |
| MD (Vieceli et al. 2005) | α_s ~ 0.95–1.0, α_b ~ 0.1–0.83 | Surface vs bulk 구분 |
| Heirman & Bogaerts 2012 | 0.04 | Plasma-liquid 모델 |
| Liu et al. 2015 | 1.0 | 상한 가정 |
| Bruggeman et al. 2016 | 0.01–1.0 | "large uncertainty" |

- **추천값: 0.04 (plasma-liquid 모델 기준) / 0.01–1.0 (불확실)**
- OH는 표면에서 계면활성제 거동 → α_s ≫ α_b 가능
- 실제 영향은 제한적: OH가 액상에서 극도로 반응성이 높아 liquid-phase reaction이 rate-limiting

### HO₂ (Hydroperoxyl Radical)

| Source | α_b | Notes |
|--------|-----|-------|
| IUPAC (Ammann et al. 2013) | ≥ 0.5 | |
| Kolb et al. 2010 | ≥ 0.5 | |
| Davidovits et al. 2006 | 0.2 (298K) | Mozurkewich 1987 |
| JPL | > 0.2 | |
| Plasma-liquid models | 0.2–1.0 | Bruggeman 2016, Heirman |

- **추천값: 0.5 (≥ 0.2)**
- pH 의존성 중요: HO₂ ⇌ H⁺ + O₂⁻ (pKa ≈ 4.7). pH > 5에서 γ 급증
- 전이금속 이온 (Cu²⁺, Fe²⁺) 존재 시 γ → 1.0

---

## Summary Table

| Species | α_b 추천값 | 불확실도 | 현재 모델(0.03) 대비 | Rate-limiting step |
|---------|-----------|---------|---------------------|-------------------|
| **N₂O₅** | ≥ 0.03 | Low | **~일치** | 가수분해 (반응) |
| **O₃** | 0.01–0.05 | Moderate | 범위 내 | 용해도 (Henry) |
| **H₂O₂** | ~0.1 | Low | **3× 과소** | Accommodation |
| **NO** | ~10⁻³ | High | **30× 과대** | 용해도 (Henry) |
| **NO₂** | 0.02–0.04 | Moderate | ~일치 | 가수분해 (반응) |
| **HONO** | 0.04–0.05 | Low | 약간 과소 | Accommodation |
| **HNO₃** | 0.05–0.07 | Low | 약간 과소 | Accommodation |
| **OH** | 0.04 (0.01–1.0) | Very High | 범위 내 | 반응 (liquid-side) |
| **HO₂** | 0.5 (≥ 0.2) | Moderate | **17× 과소** | pH 의존 |

---

## Plasma-Liquid Model Approaches (문헌 비교)

### Heirman 2025 (Bogaerts group)
- Film model: k_mt = α_b × D_l / δ_liq
- α_b를 parametric하게 변화 (주로 전체 종에 동일값 적용)
- δ_liq와 α_b가 degenerate parameter — 둘 중 하나만 조절해도 동일 효과

### Liu et al. 2015/2016 (Bruggeman group)
- **Henry's law 직접 적용** (α_b 사용 안 함)
- c_surface = K_H × p_gas (계면 열역학 평형 가정)
- Mass transfer는 확산으로 제한
- Heirman 기준 O₃ 10–20배 과대예측 가능성 지적됨

### Tian & Kushner 2014
- Reactive uptake γ framework
- Flux = γ × v_thermal/4 × n_gas
- 종별 γ 사용 (OH ~0.01–0.1, O₃ ~10⁻³, H₂O₂ ~0.1)

### Norberg et al. 2014 (Kushner group)
- 종별 γ: OH 0.01, O₃ 0.002, H₂O₂ 0.1
- Reactive uptake + Henry's law equilibrium

### Bruggeman et al. 2016 (Review)
- 가장 포괄적 compilation
- α_b 값은 대기화학 실험에서 유래 → 플라즈마 조건(비평형 계면)에 직접 적용 시 주의
- 종별 α_b 범위를 Table로 정리

---

## 우리 모델 적용 시 고려사항

### 1. 가장 민감한 종
- **N₂O₅**: NO₃⁻ 생성의 99.3% 지배. α_b 변화 → NO₃⁻ 직접 변화
- **H₂O₂**: 현재 α_b=0.03 << 문헌 0.1. 종별 α_b 적용 시 H₂O₂ uptake 3배 이상 증가 예상
- **HO₂**: 현재 0.03 << 문헌 0.5. 대폭 과소 추정

### 2. α_b vs δ_liq degeneracy
- k_mt = α_b × D_l / δ_liq에서 α_b/δ_liq가 실질 파라미터
- 종별 α_b를 적용하면 이 degeneracy가 부분적으로 해소

### 3. 실용적 추천값 (298K 기준)

| Species | α_b (모델 적용) | 근거 |
|---------|----------------|------|
| N₂O₅ | 0.03 | Kolb 2010, IUPAC |
| O₃ | 0.05 | Davidovits 2006 |
| H₂O₂ | 0.1 | Kolb 2010, IUPAC |
| NO | 0.001 | Graves 2012, 추정 |
| NO₂ | 0.03 | Bruggeman 2016 |
| HONO | 0.05 | IUPAC, Davidovits 2006 |
| HNO₃ | 0.07 | Davidovits 2006, JPL |
| OH | 0.04 | Heirman 2012 |
| HO₂ | 0.5 | IUPAC, Kolb 2010 |

---

## References

1. **Ammann, M. et al. (2013)** "Evaluated kinetic and photochemical data for atmospheric chemistry: Volume VI – heterogeneous reactions with liquid substrates." *Atmos. Chem. Phys.* 13, 8045–8228. DOI: 10.5194/acp-13-8045-2013
2. **Davidovits, P. et al. (2006)** "Mass Accommodation and Chemical Reactions at Gas−Liquid Interfaces." *Chem. Rev.* 106, 1323–1354. DOI: 10.1021/cr040366k
3. **Kolb, C.E. et al. (2010)** "An overview of current issues in the uptake of atmospheric trace gases by aerosols and clouds." *Atmos. Chem. Phys.* 10, 10561–10605. DOI: 10.5194/acp-10-10561-2010
4. **Bruggeman, P.J. et al. (2016)** "Plasma–liquid interactions: a review and roadmap." *Plasma Sources Sci. Technol.* 25, 053002. DOI: 10.1088/0963-0252/25/5/053002
5. **Graves, D.B. (2012)** "The emerging role of reactive oxygen and nitrogen species in redox biology and some implications for plasma applications to medicine and biology." *J. Phys. D: Appl. Phys.* 45, 263001. DOI: 10.1088/0022-3727/45/26/263001
6. **Heirman, A. & Bogaerts, A. (2025)** *J. Phys. D: Appl. Phys.* 58, 085206.
7. **Liu, Z.C. et al. (2016)** "Chemical Kinetics and Reactive Species in Normal Saline Activated by a Surface Air Discharge." *Plasma Processes Polym.*
8. **Tian, W. & Kushner, M.J. (2014)** *J. Phys. D: Appl. Phys.* 47, 165201.
9. **Norberg, S.A. et al. (2014)** *J. Phys. D: Appl. Phys.* 47, 475203.
10. **Lindsay, A. et al. (2015)** *J. Phys. D: Appl. Phys.* 48, 424007.
11. **Worsnop, D.R. et al. (1989)** *J. Phys. Chem.* 93, 1159–1172.
12. **Vieceli, J. et al. (2005)** "Molecular Dynamics Simulations of Atmospheric Oxidants at the Air−Water Interface." *J. Phys. Chem. B* 109, 15876–15892.
13. **Sander, R. (2015)** "Compilation of Henry's law constants for water as solvent." *Atmos. Chem. Phys.* 15, 4399–4981.
14. **JPL/NASA Evaluation No. 19, Burkholder, J.B. et al. (2020)**
15. **Bongartz, A. et al. (1994)** *J. Atmos. Chem.* 18, 149–169.
16. **Mertes, S. & Wahner, A. (1995)** *J. Phys. Chem.* 99, 14000–14006.
17. **Lee, Y.N. & Schwartz, S.E. (1981)** *J. Phys. Chem.* 85, 840–848.
18. **Van Doren, J.M. et al. (1990)** *J. Phys. Chem.* 94, 3265–3269.
19. **Mozurkewich, M. et al. (1987)** *J. Geophys. Res.* 92, 4163–4170.
20. **Takami, A. et al. (1998)** *J. Phys. Chem. A* 102, 1346–1352.

---
<!-- 
Last updated: 2026-04-09
Source: Atmospheric chemistry evaluations (IUPAC, JPL) + plasma-liquid model papers
Purpose: 종별 α_b 구현 시 참조
-->
