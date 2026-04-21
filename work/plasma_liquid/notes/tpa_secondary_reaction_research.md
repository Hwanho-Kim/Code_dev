# Deep Research: hTPA + OH Secondary Reaction in TPA Fluorescence Probe

작성일: 2026-04-20 | 관련 코드: `Ver4_1D/reactions_tpa.yaml` R_TPA3

---

## Executive Summary

**결론: R_TPA3 제거는 문헌과 반대 방향. 대신 k를 1×10⁹ → 6.3×10⁹로 상향해야 함.**

- **Page et al. 2010 (J. Environ. Monit.)이 k(hTPA + OH) = (6.3 ± 0.1) × 10⁹ M⁻¹s⁻¹를 직접 측정**. TPA+OH보다 43% 더 빠르다.
- **Tampieri 2021 (Anal. Chem.)은 kinetic model에 포함하지 않는 대신, 데이터 분석을 treatment time < 90 s로 제한**하여 hTPA 2차 산화를 empirical하게 회피.
- **우리 실험 조건은 600 s (10분) — Tampieri의 safe window 대비 6.7배 초과** → hTPA 2차 산화가 dominant이므로 시뮬에 반드시 포함 필요.

---

## 1. 직접 측정된 rate constant (Page et al. 2010)

**인용**: Page S.E., Arnold W.A., McNeill K. "Terephthalate as a probe for photochemically generated hydroxyl radical" *J. Environ. Monit.* 12(9), 1658-1665 (2010). DOI: 10.1039/C0EM00160K.

Abstract 직접 인용 (via WebFetch):
> "TPA reacts with hydroxyl radical at (4.4 ± 0.1) × 10⁹ M⁻¹ s⁻¹"
> "hTPA reacts with hydroxyl radical at (6.3 ± 0.1) × 10⁹ M⁻¹ s⁻¹"
> "hTPA was shown to undergo direct photochemical degradation" with quantum yield Φ(365nm) = (6.3 ± 0.1) × 10⁻³

### 의미
- hTPA 2차 산화 rate가 TPA 1차 산화 rate보다 **43% 빠름**
- 따라서 [hTPA]가 [TPA]의 1/6.3 수준(~300 µM)에 도달하면 생성과 분해가 동률. 우리 시뮬 [hTPA]=22 µM는 아직 growth 단계지만 **rate 기반 k×[hTPA]·[OH] 항이 여전히 비선형 영향 큼**.

---

## 2. plasma-liquid 논문들의 처리 방식

### 2.1 Tampieri et al. 2021 (Anal. Chem.) — 대표 plasma-liquid TPA probe 논문

**인용**: Tampieri F. et al. "Quantification of Plasma-Produced Hydroxyl Radicals in Solution and their Dependence on the pH" *Anal. Chem.* **93**, 3666-3673 (2021). DOI: 10.1021/acs.analchem.0c04906.

PMC 전문 fetch에서:
> "For longer treatment times, linearity in Figure 1b is lost because hTPA is oxidized by CAP-generated RS."
>
> "For this reason, to calculate RhTPA, we used only the experimental points obtained for the short treatment times (lower than 90 s) when the degradation of hTPA is negligible."

처리 방식:
- **kinetic model에 hTPA 2차 반응 포함 안 함** (명시적 rate 없음)
- 대신 **데이터 분석을 t < 90 s로 제한**
- 측정계 bias 보정: `[OH]_ss`를 **TPA 농도 0으로 외삽**하여 probe 영향 제거
- 사용 TPA 농도: **15–100 µM (pH 3), 50–5000 µM (pH 7)** — 우리 실험의 2 mM과 비슷

**핵심 함의**: Tampieri가 90 s 이상에서 linearity 잃는다고 명시 → **우리 600 s 실험은 secondary sink 반드시 포함 필요**.

### 2.2 Heirman 2025 (J. Phys. D) — 우리 BC formulation 원조 논문

**인용**: Heirman P., Bogaerts A. "Critical comparison of interfacial boundary conditions in modelling plasma–liquid interaction" *J. Phys. D: Appl. Phys.* **58**, 085206 (2025).

- 전문 검토 결과: **TPA/hTPA/terephthalate 전혀 언급 없음**. H₂O₂, O₃, ·NO, HNO₂ 용해 BC만 다룸.
- 우리가 쓰는 `gas_alpha` BC formula의 근거 논문이지만 probe 선택에는 관여 안 함.

### 2.3 Bruggeman & Frontiera 2016 (PSST) review — plasma-liquid roadmap

**인용**: Bruggeman P. et al. "Plasma-liquid interactions: a review and roadmap" *PSST* **25**, 053002 (2016).

- TPA/HTA probe 사용 사례 언급 있으나 **secondary reaction에 대한 kinetic treatment 구체적 기술 없음**
- 주로 detection 방법론 소개 수준

### 2.4 Charbouillot et al. 2011 (J. Photochem. Photobiol. A)

**인용**: Charbouillot T. et al. *J. Photochem. Photobiol. A* **222**, 70-76 (2011).

- k(TPA + OH) = (4.0 ± 0.1) × 10⁹ M⁻¹s⁻¹ 측정
- pH/T 의존성: Y_TAOH = 0.0248·pH + 0.046 (288 K, pH 3.9–7.5)
- **hTPA secondary oxidation 논의 없음** (단시간 radiolysis 조건)

### 2.5 Gonzalez et al. 2018 (Analytical Letters — PPT 인용 [2])

**인용**: Gonzalez D.H., Cala C.K., Peng Q., Paulson S.E. "Terephthalate Probe for Hydroxyl Radicals: Yield of 2-Hydroxyterephthalic Acid and Transition Metal Interference" *Anal. Lett.* **51**(15), 2488-2497 (2018).

- Y_hTPA = 31.5 ± 7% (재측정, Page 2010의 35%보다 약간 낮음)
- **secondary kinetics 직접 다루지 않음**, 주로 transition metal interference 측정
- PPT는 이 논문의 **35% yield**를 사용 (PPT [2] 인용).

---

## 3. 종합 비교표

| 논문 | k(TPA+OH) | k(hTPA+OH) | 2차 반응 처리 | treatment time |
|---|---|---|---|---|
| Matthews 1980 | ~3×10⁹ | — | 언급 없음 | pulse radiolysis (µs) |
| Saran & Summer 1999 | — | — | 언급 없음 | short-term |
| **Page 2010** | **4.4×10⁹** | **6.3×10⁹** | 측정 보고 (photolysis) | analytical |
| Charbouillot 2011 | 4.0×10⁹ | — | 언급 없음 | radiolysis |
| Gonzalez 2018 | — | — | 언급 없음 | 1 min |
| **Tampieri 2021** | 4.4×10⁹ (Page 인용) | **언급 없음** | **< 90 s data truncation** | 30–300 s |
| Bruggeman 2016 | — | — | 언급 없음 (review) | N/A |
| Heirman 2025 | — | — | **probe 다루지 않음** | N/A |

---

## 4. 현재 시뮬 vs 문헌

| Parameter | 현재 값 | Page 2010 | 정합성 |
|---|---|---|---|
| k(TPA+OH)_total | 4.0×10⁹ | 4.4×10⁹ | ✓ (10% 이내) |
| Branching → hTPA | 0.35 | 0.315–0.35 | ✓ |
| k(hTPA+OH) | **1×10⁹** | **6.3×10⁹** | **6.3× 과소** |

---

## 5. 권고

### 5.1 즉시 조치
**R_TPA3의 k를 1×10⁹ → 6.3×10⁹로 상향** (Page 2010 직접 측정값).

```yaml
# reactions_tpa.yaml
- type: irreversible
  reactants: {hTPA: 1, OH: 1}
  products: {}
  k: 6.3e9        # Page et al. 2010 J. Environ. Monit. 12:1658 direct measurement
  label: "R_TPA3: hTPA + OH → decomposition"
```

### 5.2 예상 결과 (k_R3 = 6.3×10⁹ 시)
Nonlinear coupling으로 결과 예측 어려움. 4가지 가능 시나리오:

| k_R3 | hTPA @ 3.2kV | 메커니즘 |
|---|---|---|
| 0 (제거) | 41.4 µM | Sink 없음 + [OH] 증가 feedback |
| 1×10⁹ (현재) | 22.3 µM | 약한 sink |
| **6.3×10⁹ (문헌)** | **~10–15 µM 예상** | 강한 sink + [hTPA] steady-state 가까이 |
| ∞ | 2 µM (branching·[TPA]) | 즉시 hTPA = generation/destruction |

### 5.3 실험과의 gap 재해석
문헌 k를 적용하면 시뮬 [hTPA]는 **더 낮아짐** → 실험(58 µM)과 차이 **커짐**. 이는:
- 시뮬 [OH] 공급이 여전히 과소 (α_b, δ_gas, 가스 공급 재조정 필요), **또는**
- **실험 측정이 hTPA 이외의 형광 product를 포함** (Tampieri 2021 직접 언급):
  > "additional fluorescent products lead to overestimation, requiring chromatographic separation"
  - PPT 실험은 **fluorescence only** (HPLC 분리 없음) → 다른 형광 생성물(예: 2,5-dihydroxyterephthalate 이성질체, 산화 부산물) 포함 가능
  - 이는 **실험값이 true [hTPA]를 overestimate**한다는 뜻

### 5.4 논문 방어 전략
1. R_TPA3 포함 + Page 2010 문헌 k 사용 (물리/화학 정확)
2. 실험-시뮬 gap은 두 축으로 설명:
   - OH 공급 측 (α_b, δ_gas sweep → 정량 calibration)
   - **측정 측 artifact (fluorescence only → HPLC 분리 없이는 overestimate)** — Tampieri 2021 참조로 문헌 근거 있음
3. **Tampieri처럼 "TPA 농도를 0으로 외삽한 [OH]_ss"** 를 추가 지표로 보고하여 probe 영향을 보정한 값 제시

---

## 6. 인용 문헌 (모두 링크)

- [Page et al. 2010 — J. Environ. Monit. 12:1658](https://pubs.rsc.org/en/content/articlelanding/2010/em/c0em00160k) — **k(hTPA+OH) = 6.3×10⁹ 직접 측정**
- [Page 2010 PDF (eScholarship)](https://escholarship.org/content/qt8g85g04v/qt8g85g04v_noSplash_cae5174e6ea7e3337a2fae7decfb9c62.pdf)
- [Tampieri et al. 2021 — Anal. Chem. (ACS)](https://pubs.acs.org/doi/10.1021/acs.analchem.0c04906) — **< 90 s data truncation 전략**
- [Tampieri 2021 PMC free](https://pmc.ncbi.nlm.nih.gov/articles/PMC7931173/)
- [Charbouillot et al. 2011 — J. Photochem. Photobiol. A](https://www.sciencedirect.com/science/article/abs/pii/S1010603011002085)
- [Gonzalez et al. 2018 — Analytical Letters](https://www.tandfonline.com/doi/abs/10.1080/00032719.2018.1431246)
- [Bruggeman et al. 2016 — PSST review](https://strathprints.strath.ac.uk/90263/1/Bruggeman-etal-PSST-2016-Plasma-liquid-interactions-a-review-and-roadmap.pdf)
- Heirman & Bogaerts 2025 — J. Phys. D 58:085206 (로컬 PDF 확인, TPA probe 언급 없음)
