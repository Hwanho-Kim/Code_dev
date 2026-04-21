# TPA–OH Chemistry — 문헌 파라미터 확정

작성일: 2026-04-20 | Phase 0 산출물

## 반응 상수 (채택값)

| 반응 | k (M⁻¹s⁻¹) | 근거 |
|---|---|---|
| **TPA²⁻ + OH → hTPA²⁻** | 1.40 × 10⁹ | = 0.35 × 4.0×10⁹ (branching × k_total) |
| **TPA²⁻ + OH → 비형광 부산물** | 2.60 × 10⁹ | = 0.65 × 4.0×10⁹ |
| **hTPA²⁻ + OH → 분해** | 1.0 × 10⁹ | 보수적 추정, 10 min 내 영향 <5% |

- k(TPA+OH)_total = **4.0 × 10⁹ M⁻¹s⁻¹** (Matthews 1980, Fang 1996, Charbouillot 2011 범위 3.3–4.4×10⁹)
- **branching = 0.35** (PPT와 일치, Page 2010, Saran 1999)

## 알칼리 조건 (pH 11.5) 주요 경로 — 기존 `reactions_full.yaml`에 이미 존재
- R22: O₃+OH⁻ → O₂+HO₂⁻ (k=40)
- R23: O₃+OH⁻ → O₂⁻+HO₂ (k=70)
- **R26: O₃+HO₂⁻ → O₂⁻+O₂+OH (k=5.5×10⁶)** — autocatalytic OH 공급원
- R27: O₃+OH → HO₂+O₂ (k=3.0×10⁹) — OH sink (TPA와 경쟁)
- R29: O₃+HO₂ → OH+2O₂
- R30: O₃+H₂O₂ → OH+HO₂+O₂
- R45: 2OH → H₂O₂

## TPA²⁻ 단일종 근사
- pKa₁=3.51, pKa₂=4.82 → pH 11.5에서 99.99% dianion. 단일 species `TPA`로 처리.
- 가수분해/이성질화 없음 (비휘발성).

## 경쟁 scavenger 순위 (2 mM TPA 기준)
- k[TPA] = 4×10⁹ × 2×10⁻³ = **8×10⁶ s⁻¹** (dominant sink of OH)
- k[OH⁻](R21) = 1.2×10¹⁰ × 3×10⁻³ = 3.6×10⁷ s⁻¹ (competing!)
- k[HO₂⁻](R42) = 7.5×10⁹ × ≤1×10⁻⁵ = ≤7.5×10⁴ s⁻¹
- k[CO₃²⁻](optional) = 3.9×10⁸ × 100×10⁻⁶ = 3.9×10⁴ s⁻¹

→ **OH⁻ 자체 scavenging (R21, k=1.2×10¹⁰)** 이 TPA와 동급. 시뮬레이션에서 TPA 포획률 ≈ 18–30%로 예상.

## 가스상 입력 데이터 확인 (OAS `(P-L) 가스활성종 농도.xlsx`)
- 시트: `2.6kV`, `3.2kV`, `3.6kV` (300 rows, 2–600 s, 2 s 간격)
- 열: t, O₃, NO₂, NO₃, N₂O₅ (단위 cm⁻³)
- O₃ 종료값 (평탄 도달):
  - 2.6 kV: 3.34 × 10¹⁶
  - 3.2 kV: 1.05 × 10¹⁷ (3× 증가)
  - 3.6 kV: 1.13 × 10¹⁷ (추가 7% 증가 — 비단조성은 **gas-side 아님**)

## 물리 파라미터
- TPA diffusivity: 7.5 × 10⁻¹⁰ m²/s (Stokes-Einstein, radius ≈ 3.2 Å)
- hTPA: 7.0 × 10⁻¹⁰
- 전하: 둘 다 Z = -2

## Phase 1 구현 요약
1. `config_1d.py`: `AQUEOUS_SPECIES`에 `TPA`, `hTPA`, `Na+` 추가. diffusivity/charge 테이블 업데이트.
2. 신규 `reactions_tpa.yaml` (3개 반응).
3. `chemistry_1d.py`: saline_mode와 동일 패턴으로 `tpa_mode` flag 추가하여 TPA 반응 병합.
4. `run_tpa_alkaline.py`: 초기조건 + 3전압 runner.

## 결정된 민감도 sweep 범위 (Phase 5)
- k_TPA+OH: 3.3 / 4.0 / 5.0 × 10⁹
- branching: 0.28 / 0.35
- Initial [OH⁻]: 3 / 6 / 10 mM (NaOH 중화 후 평형)
- Carbonate: 0 / 10 / 100 μM
- α_b(O₃): 0.025 / 0.05 / 0.1
