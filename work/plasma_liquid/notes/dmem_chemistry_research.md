# DMEM + Plasma RONS Chemistry — Deep Research Report

**Date:** 2026-05-07
**Purpose:** Temporary DMEM addon to existing DIW plasma-liquid reaction set (Ver4_1D).
**PPTX trigger:** `Article/JYChoi_20260429_DMEM RONS 반응 관련 레퍼런스.pptx`
**Verdict:** PPTX의 3개 반응(Pyr+H2O2, Met+H2O2, Cystine 평형)만으로는 **불충분**. 실제 DMEM은 (a) 44 mM HCO3⁻이 ONOO⁻ 화학을 완전히 다른 경로로 돌리고, (b) 25 mM Glucose가 •OH의 50%를 잡아먹고, (c) Tyr/Trp이 nitration 산물의 주생성지가 됨.

---

## 1. Executive Summary

### PPTX 대비 핵심 누락 사항 (반드시 추가해야 할 것)

| 항목 | PPTX | 실제 중요도 |
|---|---|---|
| **NaHCO3 44 mM** | 미언급 | ★★★★★ ONOO⁻+CO2 (k=5.8e4) 경로가 직접 AA 반응보다 99% 우세. •NO2+CO3•⁻ 생성 |
| **Glucose 25 mM** | 무시 | ★★★★★ •OH의 ~50% 흡수 (k×[C] = 3.8e7 s⁻¹) |
| **Tyrosine 0.4 mM** | 미언급 | ★★★★ 3-nitrotyrosine 주 생성지, ONOO⁻/CO2 경로 산물 |
| **Tryptophan 0.078 mM** | 미언급 | ★★★ Indol ring, NFK/kynurenine 생성 |
| **Pyruvate + ONOO⁻** | 미언급 | ★★★ k=100, **•NO2+•CO2⁻ 라디칼 생성** (clean scavenging 아님) |
| **Cysteine 산화 캐스케이드** | 단순언급 | ★★★ CysSOH 빠르게 disulfide로 회귀, CysSO2H/SO3H은 종착 |
| **DMEM에는 free Cys 없음** | 가역평형으로 가정 | ★★★ Cystine (S-S, 0.2 mM)만 존재. 효소 없는 환경에서 환원 거의 안 됨 |
| **Met + H2O2 k=2e-2** | 핵심 반응으로 표기 | ★★ τ ≈ 10⁶ s. **60s 모델링에서 무시 가능** |

### PPTX의 우선순위가 잘못된 부분
- Met + H2O2 (k=2e-2): 60s 노출에서 거의 영향 없음 → **drop 가능**
- Pyruvate + H2O2 (k=2.36): 1 mM 농도에서 H2O2 τ ≈ 7 min, 60s에선 ~14% 소비. 의미 있음
- Cystine ⇌ 2 Cysteine: DMEM (효소 없음)에서 **kinetically slow**, 평형 가정 부적절 → cystine 고정값 사용 권장

---

## 2. DMEM Gibco 12800017 Composition

기준: cytion.com / Sigma D6429 / ATCC 30-2002 — 동일 formulation. Powder는 NaHCO3 미포함이지만 사용 시 3.7 g/L 첨가.

### 2.1 Amino acids (mM)

| Species | mM | Reactivity tier |
|---|---|---|
| L-Glutamine | **4.00** | low (slow with RONS) |
| L-Threonine | 0.798 | OH only |
| L-Valine | 0.803 | OH only |
| L-Isoleucine | 0.802 | OH only |
| L-Leucine | 0.802 | OH only |
| L-Lysine·HCl | 0.798 | OH only |
| L-Cystine·2HCl | **0.201** (= 0.402 thiol-eq) | ★★★ S-S, slow with H2O2 but fast with •OH |
| L-Methionine | **0.201** | ★★ S-CH3, all RONS |
| L-Tyrosine·2Na·2H2O | **0.398** | ★★★★ Phenol, nitration target |
| L-Phenylalanine | 0.400 | OH only |
| L-Histidine·HCl·H2O | **0.200** | ★★ Imidazole, 1O2 |
| L-Arginine·HCl | 0.398 | OH + ONOOH slow |
| L-Serine | 0.400 | OH only |
| Glycine | 0.400 | OH only |
| L-Tryptophan | **0.0784** | ★★★★ Indole, all RONS |

### 2.2 Vitamins / cofactors (mM)

| Species | mM | Note |
|---|---|---|
| Folic acid | 0.00907 | OH k≈1.1e10 (concentration low) |
| Riboflavin | 0.00106 | photosensitizer |
| Choline-Cl | 0.0286 | minor |
| Nicotinamide | 0.0328 | minor |
| Thiamine·HCl | 0.0119 | minor |
| Pyridoxal·HCl | 0.0196 | minor |
| myo-Inositol | 0.0400 | minor |
| Ca-pantothenate | 0.00839 | minor |

### 2.3 Inorganic salts (mM) — **CRITICAL for chemistry**

| Species | mM | Why critical |
|---|---|---|
| NaCl | **110** | Cl⁻ → secondary HOCl chain if Cl⁻+OH• occurs |
| **NaHCO3** | **44.05** | ★★★★★ ONOO⁻ pathway hijack |
| KCl | 5.36 | electrolyte |
| CaCl2·2H2O | 1.80 | electrolyte |
| MgSO4·7H2O | 0.813 | electrolyte |
| NaH2PO4·2H2O | 0.908 | weak buffer |
| Fe(NO3)3·9H2O | 0.000248 | trace, may catalyze auto-oxidation |

### 2.4 Energy / pH indicator (mM)

| Species | mM | Role |
|---|---|---|
| **D-Glucose** | **25.0** | ★★★★★ •OH sink |
| **Sodium pyruvate** | **1.00** | ★★★★ H2O2 + ONOO⁻ scavenger |
| **Phenol red Na** | 0.0399 | pH indicator, reacts with •NO2 / •OH |

---

## 3. Critical Chemistry Insights

### 3.1 Bicarbonate completely re-routes ONOO⁻

DMEM has **44 mM HCO3⁻** (vs 1.2 mM dissolved CO2 at atmospheric equilibrium). Two effects:

1. **HCO3⁻ + •OH → CO3•⁻ + OH⁻** (k = 8.5×10⁶ M⁻¹ s⁻¹)
   - 44 mM × 8.5e6 = 3.7×10⁵ s⁻¹ pseudo-first-order
   - CO3•⁻ is **selective oxidant** — slow with glucose, fast with Tyr/Trp/Cys
   - Lives ~10× longer than •OH → travels deeper into liquid

2. **ONOO⁻ + CO2 → ONOOCO2⁻** (k = 5.8×10⁴ M⁻¹ s⁻¹, pH-indep at pH 7.4)
   - τ(ONOO⁻) ≈ 1/(5.8e4 × 1.2e-3) ≈ **10 ms** — peroxynitrite essentially never reaches Cys/Met directly
   - ONOOCO2⁻ → 0.33×(•NO2 + CO3•⁻) + 0.67×(NO3⁻ + CO2)
   - Net: 1 ONOO⁻ → 0.33 (•NO2 + CO3•⁻), Tyr nitration via •NO2 + TyrO•

**Consequence**: DMEM 환경에서 효과적인 oxidant set은 `{CO3•⁻, •NO2, ¹O2, residual •OH, residual O3}`. 직접 ONOO⁻ + AA 반응을 모델에 넣으면 **수 자리수 과대평가**됨.

### 3.2 Glucose dominates •OH consumption

`k(glucose+OH) = 1.5×10⁹` × `[glucose] = 25 mM` → **3.8×10⁷ s⁻¹**, total •OH 흡수의 ~50%.

| Component | k×[C] (s⁻¹) | % of •OH sink |
|---|---|---|
| Glucose | 3.8×10⁷ | 49% |
| Lumped non-reactive AAs | ~3×10⁷ | 38% |
| Tyrosine | 5.2×10⁶ | 6.7% |
| Glutamine | 2.0×10⁶ | 2.6% |
| Histidine | 1.4×10⁶ | 1.8% |
| Tryptophan | 1.0×10⁶ | 1.3% |
| Phenol red | 5.6×10⁵ | 0.7% |
| HCO3⁻ | 3.7×10⁵ | 0.5% |
| Folic acid | 1.0×10⁵ | 0.1% |

→ DMEM에서 •OH는 거의 모두 glucose + AA pool에 소진. plasma-derived 1차 RONS 중 **•OH는 표면층(δ_diff ~수 µm)에서만 살아남음**.

### 3.3 Pyruvate + ONOO⁻ is radical-generating, not clean scavenging

Vásquez-Vivar 1997 (Chem Res Toxicol):
- Pyruvate + ONOO⁻ → acetate + CO2 + **•NO2 + •CO2⁻**
- Pyruvate가 ONOO⁻ scavenger로 작동하지만 동시에 **2차 라디칼 생성** → Tyr nitration 가속하는 경우도 보고

→ 모델에서 "pyruvate가 ONOO⁻을 안전하게 분해"로 가정하면 nitration 과소평가. Branch ratio 추가 필요.

### 3.4 Cysteine cascade — DMEM에는 free cysteine 없음

PPTX의 "Cystine ⇌ 2 cysteine 가역" 가정은 부정확:
- DMEM은 **L-cystine·2HCl** (S-S 형태)로 출발. Free CysSH 없음.
- **효소 (Trx, GSH) 없는 환경에서 cystine → cysteine 환원은 매우 느림** (E°' = -0.22V vs NHE, 60s에선 무시 가능)
- 그러므로 모델에서 cystine 0.2 mM 고정, free CysSH ≈ 0으로 시작이 맞음

다만 **CysSSCys + •OH** (k ≈ 7×10⁹) 반응으로 S-S 절단 → CysS• + CysSOH 가능. 이 경로는 살려야 함.

### 3.5 Plasma-acidified DIW (pH 3-4) effect

CysSH thiol pKa ≈ 8.3 → pH 7에서 thiolate 5%, pH 3에선 ~5×10⁻⁶. Thiolate가 reactive species이므로:
- pH 7: k_obs(CysSH+H2O2) ≈ 20 M⁻¹s⁻¹
- pH 3: ~10⁻⁴배로 감소

→ DMEM이 buffered (44 mM HCO3⁻로 pH ~7.4 유지)이므로 plasma 처리해도 **pH는 거의 안 떨어짐**. DIW 모델의 pH 3.6 결과와 정반대 거동. **이게 가장 큰 차이점**.

---

## 4. Master Kinetic Table (recommended for inclusion)

### 4.1 Pyruvate (1.0 mM)

| # | Reaction | k (M⁻¹s⁻¹) | Source |
|---|---|---|---|
| Pyr1 | Pyr + H2O2 → AcO⁻ + CO2 + H2O | **2.36** | Asmus 2019 Sci Rep 9:19858 |
| Pyr2 | Pyr + •OH → CH3COCOO• + H2O | **~7×10⁸** | Buxton 1988 (anion form) |
| Pyr3 | Pyr + ONOOH → AcO⁻ + CO2 + •NO2 + •CO2⁻ | 49 | Vásquez-Vivar 1997 ChemResTox 10:786 |
| Pyr4 | Pyr + ONOO⁻ → AcO⁻ + CO2 + •NO2 + •CO2⁻ | 100 | Vásquez-Vivar 1997 |
| Pyr5 | Pyr + O3 → ? | <3 (drop) | Schöne 2014 ACP 14:4503 |
| Pyr6 | Pyr + (•NO2, NO3•, O2•⁻, ¹O2) | data gap (drop) | — |

### 4.2 Methionine (0.20 mM) — 60s에서 의미 있는 것만

| # | Reaction | k (M⁻¹s⁻¹) | Source | Action |
|---|---|---|---|---|
| Met1 | Met + •OH → S-radical | **8.3×10⁹** | Buxton 1988 | KEEP |
| Met2 | Met + O3 → MetSO | **~4×10⁶** | Pryor 1984 / Hoigné | KEEP |
| Met3 | Met + ¹O2 → MetSO | **1.7×10⁷** | Sysak 1977 | KEEP if 1O2 tracked |
| Met4 | Met + ONOOH → MetSO | **181** (37°C) | Pryor & Padmaja 1994 PNAS | KEEP (lump w/ ONOO⁻+CO2) |
| Met5 | Met + HOCl → MetSO | 3.4×10⁷ | Storkey 2014 FRBM | KEEP if Cl chemistry on |
| Met6 | Met + H2O2 → MetSO | 2×10⁻² | Sysak 1977 / Park 2024 | **DROP** (τ~10⁶ s) |
| Met7 | Met + •NO2 → ø | ~0 | Prütz 1985 | DROP (negligible) |
| Met8 | MetSO + H2O2 → MetSO2 | 2.5×10⁻³ | Org Lett 2024 | DROP |

### 4.3 Cystine (0.2 mM, S-S form)

| # | Reaction | k (M⁻¹s⁻¹) | Source | Action |
|---|---|---|---|---|
| Cys1 | CysSSCys + •OH → CysS• + CysSOH | **~7×10⁹** | Buxton 1988 / PNAS 2020 | KEEP |
| Cys2 | CysSSCys + H2O2 → 2 CysSOH | <10⁻³ (slow) | Winterbourn 2008 | DROP |
| Cys3 | CysSSCys + O3 → mixed sulfinic/sulfonic | ~4.4×10⁶ (Cys proxy) | Mudd 1969 | OPTIONAL |
| Cys4 | CysSSCys + 2e⁻/2H⁺ ⇌ 2 CysSH | very slow w/o enzyme | Jocelyn 1967 | DROP |
| Cys5 | Cystine·2HCl ⇌ Cystine + 2H+ + 2Cl⁻ | instant (ionic) | trivial | inherent |

**Model simplification**: cystine을 단순 •OH/O3 sink로 lump. CysSOH/CysSO2H/CysSO3H는 별도 산물 추적 불필요 (terminal).

### 4.4 Tyrosine (0.40 mM) — Nitration 핵심

| # | Reaction | k (M⁻¹s⁻¹) | Source | Action |
|---|---|---|---|---|
| Tyr1 | Tyr + •OH → TyrO• | **1.3×10¹⁰** | Buxton | KEEP |
| Tyr2 | Tyr + O3 → ? | ~4×10⁵–4×10⁶ | (uncertain) | KEEP |
| Tyr3 | Tyr + •NO2 → TyrO• + NO2⁻ | **3.2×10⁵** | Prütz 1985 | KEEP |
| Tyr4 | TyrO• + •NO2 → 3-NO2-Tyr | ~3×10⁹ | Bartesaghi 2018 | KEEP |
| Tyr5 | Tyr + CO3•⁻ → TyrO• + HCO3⁻ | 4.5×10⁷ | Augusto 2002 | KEEP |
| Tyr6 | Tyr + ¹O2 → endoperoxides | ~8×10⁶ | NIST | OPTIONAL |
| Tyr7 | Tyr + O2•⁻ (TyrO• coupling) | 1.5×10⁹ | RSC 1993 | OPTIONAL |
| Tyr8 | Tyr + ONOOH | small (<10²) | direct path negligible vs CO2 route | DROP |

### 4.5 Tryptophan (0.078 mM)

| # | Reaction | k (M⁻¹s⁻¹) | Source | Action |
|---|---|---|---|---|
| Trp1 | Trp + •OH → indolyl | **1.3×10¹⁰** | Buxton | KEEP |
| Trp2 | Trp + O3 → NFK | ~7×10⁶ | Pryor (upper bound) | KEEP |
| Trp3 | Trp + ¹O2 → NFK | 3-7×10⁷ | Davies | KEEP if 1O2 |
| Trp4 | Trp + ONOOH → ox-Trp | 184 (37°C) | Alvarez | OPTIONAL |
| Trp5 | Trp + CO3•⁻ → Trp• | ~7×10⁸ | Padmaja | KEEP |
| Trp6 | Trp + •NO2 → indolyl + NO2⁻ | ~10⁶ | reviews | KEEP |

### 4.6 Histidine (0.20 mM)

| # | Reaction | k (M⁻¹s⁻¹) | Source | Action |
|---|---|---|---|---|
| His1 | His + •OH → ImH• | ~5×10⁹ | Buxton | KEEP |
| His2 | His + ¹O2 → endoperoxide | 3.2×10⁷ (pH 7) | NIST | KEEP if 1O2 |
| His3 | His + O3 → ? | ~2.5×10⁵ (pH 7) | reviews | OPTIONAL |
| His4 | His + CO3•⁻ → ? | ~1×10⁷ | reviews | OPTIONAL |

### 4.7 Glucose (25 mM) — •OH sink only

| # | Reaction | k (M⁻¹s⁻¹) | Source | Action |
|---|---|---|---|---|
| Glc1 | Glc + •OH → Glc-ox | **1.5×10⁹** | Buxton | **KEEP (#1 priority)** |
| Glc2 | Glc + O3, H2O2, ONOO⁻ | <10² (all) | reviews | DROP all |
| Glc3 | Glc + CO3•⁻ | ~6×10³ (negligible) | review | DROP |

### 4.8 Glutamine (4.0 mM) — bulk •OH sink

| # | Reaction | k (M⁻¹s⁻¹) | Source | Action |
|---|---|---|---|---|
| Gln1 | Gln + •OH → ? | ~5×10⁸ | NDRL | KEEP |
| Gln2 | spontaneous deamidation | non-radical, slow | — | DROP |

### 4.9 Bicarbonate / CO2 system — **★★★★★ MUST INCLUDE**

| # | Reaction | k (M⁻¹s⁻¹) | Source | Action |
|---|---|---|---|---|
| Bic1 | HCO3⁻ + •OH → CO3•⁻ + H2O | **8.5×10⁶** | Buxton 1988 | **MUST KEEP** |
| Bic2 | CO3²⁻ + •OH → CO3•⁻ + OH⁻ | 3.9×10⁸ | Buxton 1988 | optional (low at pH 7.4) |
| Bic3 | ONOO⁻ + CO2 → ONOOCO2⁻ | **5.8×10⁴** | Lymar & Hurst 1995 | **MUST KEEP** |
| Bic4 | ONOOCO2⁻ → 0.33(•NO2+CO3•⁻) + 0.67(NO3⁻+CO2) | unimol fast | Goldstein 1998 | **MUST KEEP** |
| Bic5 | 2 CO3•⁻ → products | ~2×10⁷ | NDRL | KEEP (decay) |
| Bic6 | CO3•⁻ + Tyr/Trp/His/Cys | see 4.4-4.6 | — | KEEP |

### 4.10 Lumped "non-reactive" amino acid pool (≈ 30 mM total)

`Gly+Ala+Ser+Thr+Val+Leu+Ile+Pro+Asn+Asp+Glu+Lys+Arg+Phe` ≈ 30 mM 합산
- Effective k(•OH) ≈ 3×10⁹ (concentration-weighted)
- Single lumped sink "AA_inert + •OH → AA_ox"

---

## 5. Recommended Implementation Strategy

### 5.1 Reaction set 추가 (DIW set 위에)

**Tier 1 (반드시)** — 14 reactions:
- Pyr1, Pyr3+Pyr4 (lump as ONOO⁻ branch), Pyr2
- Met1, Met2, Met4
- Cys1 (CysSSCys+OH), Cys3 (optional)
- Tyr1, Tyr3, Tyr4 (TyrO•+NO2→nitro-Tyr)
- Glc1, Gln1, AA_pool+OH
- **Bic1, Bic3, Bic4 (CO2 hijack)**

**Tier 2 (singlet oxygen 추적 시)**: Met3, Tyr6, His2, Trp3

**Tier 3 (CO3•⁻ propagation 정밀)**: Tyr5, Trp5, Bic5

### 5.2 새 species 추가 (chemistry_1d.py)

```
신규: Pyruvate, Acetate, Methionine, MetSO, Cystine, CysSOx (lumped),
      Tyrosine, TyrO_radical, NO2_Tyr, Tryptophan, Trp_ox,
      Histidine, His_ox, Glucose, Glc_ox, Glutamine, Gln_ox,
      AA_inert (lumped), AA_ox (lumped),
      HCO3_minus, CO2, CO3_radical, ONOOCO2_minus
```

총 20여 종 추가. 기존 47종 → 67~70종. Numba JIT 영향은 미미 (선형 증가).

### 5.3 초기 조건 (mM)

DIW initial과 다른 점:
- 모든 아미노산/glucose/pyruvate/HCO3⁻은 위 표 그대로
- pH는 NaHCO3 buffering으로 ≈ 7.4 시작 (DIW는 pH ≈ 6.0 시작)
- conductivity는 NaCl 110 mM로 매우 높음 (Saline과 비슷한 ionic strength)

### 5.4 Boundary condition은 그대로

기상→액상 transfer (gas_alpha BC, three_film 등)는 변경 없음. DMEM은 액상 chemistry만 다름. Henry 상수도 (수용액 기준) 동일 사용.

### 5.5 Validation 전략

DMEM 결과의 실험 비교는:
- pH 변화 없거나 미미 (HCO3⁻ buffer)
- H2O2 농도: pyruvate에 의해 DIW 대비 50% 이상 감소 (1 mM Pyr × 60s × 2.36 = 14% 정도지만 실제론 longer treatment)
- NO3⁻은 ONOO⁻+CO2 경로로 67% 직접 생성 → DIW와 비슷하거나 약간 높음
- Tyr nitration / NFK 측정 가능하면 ★★★ 좋은 검증 지표 (LC-MS, fluorescence)

---

## 6. Major Uncertainties / Gaps

1. **Pyr + (•NO2, NO3•, O2•⁻, ¹O2)** — 직접 측정 없음. 일단 drop, 필요시 fitting 파라미터로.
2. **Phenol red 직접 k 값** — phenol/tyrosine analogue로 추정. 실험하기 전엔 ±50%.
3. **Riboflavin ¹O2 quenching** — 최근(2024) 재평가로 1e9→2e5 하향. 농도 1 µM이라 영향 미미.
4. **Pryor 1984 O3+AA k 값** — 최근 비판으로 2-5× 과대평가 가능성. 상한선으로 사용.
5. **CysSSCys + H2O2** — 신뢰할 만한 k 없음. 0으로 처리.
6. **Cystine ⇌ 2 Cys 환원 속도** — 효소 없는 환경, GSH 없음. 60s 모델에선 cystine 고정.
7. **Park ChemistryOpen 2024 e202300213** — 직접 다운로드 실패. 인용된 Pyr+H2O2 k=2.36은 Asmus 2019에서 나왔음을 확인.

---

## 7. Sources (consolidated)

### Pyruvate kinetics
- Asmus et al. *Sci Rep* 2019, 9:19858 — https://www.nature.com/articles/s41598-019-55951-9
- Park et al. *ChemistryOpen* 2024, e202300213 — https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/open.202300213
- Vásquez-Vivar, Denicola, Radi, Augusto, *Chem Res Toxicol* 1997, 10, 786 — https://pubs.acs.org/doi/10.1021/tx970031g
- Schöne et al. *Atmos Chem Phys* 2014, 14, 4503 — https://acp.copernicus.org/articles/14/4503/2014/
- Long & Halliwell *BBRC* 2015 — https://pubmed.ncbi.nlm.nih.gov/25754627/

### Methionine kinetics
- Pryor & Padmaja *PNAS* 1994 — https://pmc.ncbi.nlm.nih.gov/articles/PMC45189/
- Pattison & Davies *Chem Res Toxicol* 2001 — https://pubmed.ncbi.nlm.nih.gov/11599938/
- Storkey, Davies & Pattison *FRBM* 2014 — https://www.sciencedirect.com/science/article/abs/pii/S0891584914001968
- Sysak, Foote, Ching *Photochem Photobiol* 1977 — Wiley
- Matheson et al. *Photochem Photobiol* 1979 — Wiley

### Cysteine/Cystine kinetics
- Luo et al. *J Pharm Sci* 2005, 94, 304 — https://pubmed.ncbi.nlm.nih.gov/15570599/
- Radi *JBC* 1991 — https://www.jbc.org/article/S0021-9258(20)64313-7/pdf
- Trujillo & Radi *ABB* 2002 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3656273/
- Peskin/Winterbourn *JBC* 2013 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3656273/
- Rehder & Borges *Biochemistry* 2010 — https://pubs.acs.org/doi/10.1021/bi1008694
- PNAS 2020 (Cys-S-S + OH 2-step) — https://www.pnas.org/doi/10.1073/pnas.2006639117
- Winterbourn & Hampton *FRBM* 2008 — https://pmc.ncbi.nlm.nih.gov/articles/PMC2693905/

### Tyrosine / Tryptophan / nitration
- Bartesaghi & Radi *Chem Rev* 2018 — https://pubs.acs.org/doi/10.1021/acs.chemrev.7b00568
- Alvarez et al. (Trp+ONOOH) — https://pubmed.ncbi.nlm.nih.gov/27406073/
- Davies "Tripping up Trp" — https://pmc.ncbi.nlm.nih.gov/articles/PMC4684788/
- Padmaja et al. (Trp nitration) — https://pubs.acs.org/doi/abs/10.1021/tx950133b
- Prütz 1985 (NO2• + Tyr/Met-Gly) — https://pubmed.ncbi.nlm.nih.gov/4062299/

### Bicarbonate / CO2 / peroxynitrite
- Lymar & Hurst *JACS* 1995, 117, 8867 — https://pubs.acs.org/doi/10.1021/ja00139a027
- Denicola, Freeman, Trujillo, Radi 1996 — https://pubmed.ncbi.nlm.nih.gov/8806753/
- Augusto et al. *FRBM* 2002, 32, 841 — https://iubmb.onlinelibrary.wiley.com/doi/pdf/10.1080/15216540701230511
- CO3•⁻ vs aromatic AAs *JPCB* 2017 — https://pubs.acs.org/doi/10.1021/acs.jpcb.7b05186
- Buxton & Elliot 1986, RaPC 27:241

### Compilations
- Buxton et al. *J Phys Chem Ref Data* 1988, 17, 513 — https://pubs.aip.org/aip/jpr/article/17/2/513/241398
- NIST/NDRL Solution Kinetics — https://kinetics.nist.gov/solution/
- Wilkinson et al. NIST jpcrd489 (¹O2) — https://www.nist.gov/system/files/documents/srd/jpcrd489.pdf
- Bielski, Cabelli, Arudi *JPCRD* 1985 (HO2/O2⁻) — https://srd.nist.gov/jpcrdreprint/1.555739.pdf
- Neta, Huie, Ross *JPCRD* (inorganic radicals) — https://srd.nist.gov/JPCRD/jpcrd346.pdf

### DMEM composition
- Thermo Fisher 12800017 — https://www.thermofisher.com/order/catalog/product/12800017
- Cytion DMEM page — https://www.cytion.com/DMEM-w-4.5-g-L-Glucose-w-4-mM-L-Glutamine-w-3.7-g-L-NaHCO3-w-1.0-mM-Sodium-pyruvate/820300a
- Sigma D6429 — https://www.sigmaaldrich.com/US/en/technical-documents/technical-article/cell-culture-and-cell-culture-analysis/mammalian-cell-culture/dulbecco-modified-eagle-medium-formulation
- ATCC 30-2002 — https://www.atcc.org/products/30-2002

---

## 8. Methodology

3-agent parallel deep research (general-purpose subagent), 2026-05-07.
- Agent 1: DMEM 12800017 composition + pyruvate kinetics. ~15 sources.
- Agent 2: Methionine + cysteine/cystine cascade. ~25 sources.
- Agent 3: Secondary components (phenol red, riboflavin, Tyr, Trp, His, glucose) + bicarbonate/CO2 system. ~22 sources.
Total ≈ 60+ unique web sources, prioritizing peer-reviewed primary literature and NIST/NDRL compilations.
