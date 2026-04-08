# Literature Notes — plasma_liquid

## Core References

### Aqueous Chemistry
- **193 reactions, 47 species** — 반응 세트 출처는 reactions_full.yaml 주석 참조

### Gas-Liquid Interface (BC) — 최대 Black Box
- **Liu et al. 2015**: Dirichlet BC (C_surface = H·p_gas)
  - 단순하지만 실제 mass transfer 무시
- **Lee et al. 2023**: D_adj (adjusted diffusivity) approach
  - 현재 모델 채택. gas-side-only Robin 시도 → FAILED (HNO3 C_eq=8.36M divergence)
- **Silsby et al. 2021**: 또 다른 BC 접근법
  - NOT TRIED — 비교 대상 후보

### Saline Chemistry
- **(Liu 2016)** Chemical Kinetics and Reactive Species in Normal Saline Activated by a Surface Air Discharge
  - 파일: `Article/Article/(Liu2016)...pdf`
  - Cl 반응 경로 참조
- **Saline_reaction.pdf**: Cl 반응 추가 참조

### Henry's Law Constants
- config_1d.py에 종별 Henry 상수 정리됨
- 출처: Sander 2015 compilation + 개별 문헌

## Key Findings

### HONO/HONO2/H2O2 Fitting 실패 (2026-03-23)
- Saline + DIW 동시 만족하는 해 없음
- 원인: 이 세 종의 기상 농도가 측정값(FTIR)이 아닌 추정값
- 결론: fitting parameter에서 제외, 문헌값 또는 0으로 처리

### D_adj BC 선택 근거
- Robin BC (gas-side-only) 시도 → HNO3의 C_eq = 8.36M로 divergence
- D_adj는 물리적으로 questionable하지만 현재 유일하게 작동하는 방식

### pH Gap
- DI water: 예측 pH vs 실험 pH 차이 → 179 µM missing anions 추정
- 원인 미확인 (carbonate? trace impurities?)

---
<!-- 새 문헌을 읽을 때마다 여기에 추가. 형식: 저자 연도, 제목, 핵심 내용 1-2줄 -->
