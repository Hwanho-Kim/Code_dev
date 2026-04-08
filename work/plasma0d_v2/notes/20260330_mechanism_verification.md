# 반응 메커니즘 검증 (2026-03-30)

## 목적
V_eff=4.9cm³ 모델에서 실험값과의 gap(RMSE~2.95)이 반응 메커니즘의 문제인지 확인.
- (옵션 2) Rate coefficient 불확실성
- (옵션 3) 누락 반응 경로

## 1. Rate Coefficient 검증

### 검증 대상 (7개 핵심 radical 반응)
| id | 반응 | 코드 source | 300K 비교 | 판정 |
|----|------|------------|----------|------|
| 22 | CH₄+O→CH₃+OH | Burgess&Manion 2021 | GRI 대비 1.11x | 양호 |
| 23 | CH₄+OH→CH₃+H₂O | ref 없음 | JPL 대비 0.64x | 주의(36%↓) |
| 42 | O+O₂+M→O₃ | ref 없음 | JPL 대비 0.97x | 양호 |
| 47 | HO₂+O→O₂+OH | ref 없음 | JPL 대비 0.98x | 양호 |
| 48 | HO₂+HO₂+M→H₂O₂+O₂ | ref 없음 | k₂만: JPL~10%↓ | **bimol항 누락(~2x)** |
| 90 | OH+OH+M→H₂O₂ | ref 없음 | JPL 대비 **44~120x↓** | **심각** |
| 91 | HO₂+OH→H₂O+O₂ | ref 없음 | JPL 완벽 일치 | 양호 |

### 보정 테스트 결과
```
Scenario         30°C  100°C  180°C  250°C   RMSE
Baseline         4.35   5.84  10.33  16.45   2.95
Fix90(Troe)      4.35   5.83  10.32  16.45   2.95
Fix48(2x)        4.31   5.78  10.24  16.41   3.00
Fix90+48         4.31   5.77  10.24  16.40   3.01
```
**결론**: 최대 -0.09%p. OH+OH 44~120x 보정해도 전환율 무영향.
이유: OH+OH→H₂O₂는 OH 소비 경로이나, OH 자기반응은 CH₄+OH 대비 매우 마이너.

## 2. 누락 반응 검토

### 식별된 누락 반응 (우선순위순)
| # | 반응 | k(300K) cm³/(molec·s) | 중요도 | 근거 |
|---|------|:----:|:---:|---|
| 1 | O+OH→O₂+H | 3.3e-11 | CRITICAL | 가장 기본적 radical 반응, 완전 누락 |
| 2 | N+NO→N₂+O | 2.5e-11 | HIGH | N원자 주요 소멸(gas-kinetic), Zeldovich 2 |
| 3 | N+OH→NO+H | 2.9e-11 | HIGH | Extended Zeldovich 3 |
| 4 | NO₂+O→NO+O₂ | 9.7e-12 | MOD-HIGH | NOx 순환 필수 |
| 5 | N+O₂→NO+O | 8.2e-17 | MOD | 300K에서 느림, 500K에서 활성화 |
| 6 | O+O+M→O₂ | 1.1e-33* | MOD-LOW | O+O₂→O₃가 이미 있음 |

(*) cm⁶/(molecule²·s)

### N원자 화학 문제
현재 메커니즘에서 ground state N(⁴S)의:
- **생성**: e+N₂⁺→N+N (DR), e+NO⁺→N+O (DR), N₂(A)+O→NO+N(²D)
- **소멸**: N+O₃→NO+O₂ **단 1개**

N+NO(k=2.5e-11)와 N+OH(k=2.9e-11)가 없어 N원자가 축적될 수 있음.
단, [N]≈4e-15 mol/m³로 극히 낮아 실질 영향 미미.

### 추가 테스트 결과
```
Scenario         30°C  100°C  180°C  250°C   RMSE
Experiment       5.26   8.05  14.36  20.02
Baseline         4.35   5.84  10.33  16.45   2.95
+O+OH            4.35   5.82  10.28  16.34   3.00
+N_chem(3개)     4.24   5.84  10.33  16.45   2.95
+All(6개)        4.35   5.82  10.28  16.34   3.00

Delta from baseline:
  +O+OH          -0.00  -0.02  -0.05  -0.11
  +N_chem        -0.11  -0.00  -0.00  -0.00
  +All           -0.00  -0.02  -0.05  -0.11
```

### Radical 농도 변화 (303K)
```
Scenario       [O] mol/m³  [OH] mol/m³  [N] mol/m³   [NO] mol/m³
Baseline       5.067e-06   1.719e-06    4.179e-15    1.572e-10
+O+OH          5.064e-06   1.713e-06    4.183e-15    1.572e-10
+N_chem        5.097e-06   5.721e-07    3.964e-16    1.195e-10
+All           5.064e-06   1.713e-06    3.758e-16    2.778e-09
```

### 해석
- **O+OH→O₂+H**: [O]·[OH]=5e-6×1.7e-6=8.5e-12 mol²/m⁶ — 반응속도 R=1.08e7×8.5e-12≈9.2e-5 mol/(m³·s). 매우 작아 영향 없음.
- **N 화학**: [N] 10x 감소(4.2e-15→4.0e-16), [OH] 3x 감소(1.72e-6→0.57e-6). N+OH가 OH 소비하나 전환율 영향 -0.11%p만(303K only).
- **NOx**: NO 농도 변화하나 전환율 무영향.

## 3. 종합 결론

### 메커니즘은 CH₄ 전환 예측에 충분
- Rate coefficient 보정: 최대 -0.09%p (무의미)
- 누락 반응 추가: 최대 -0.11%p (무의미, 오히려 감소)
- 218개 반응 세트는 N₂/O₂/CO₂/CH₄ 조성에서 CH₄ 전환에 필요한 모든 주요 경로를 포함

### Gap의 원인은 0D+V_eff 체적 모델의 구조적 한계
- V_eff=4.9cm³에서 radical 농도가 높아져 2차 자기소멸 증가
- EI/N₂* 1차 반응은 V_eff 무감(ratio=1.0x)이나, OH/O 2차 반응은 비선형 scaling
- V_reactor=250cm³에서는 radical 농도가 낮아 2차 반응이 자연 억제 → 실험과 일치
- 이는 chemistry 문제가 아니라 **0D intensive formulation + sDBD geometry** 간 불일치

### 향후 가능 방향
1. **Radical diffusion/mixing** — V_eff 밖으로의 radical 확산/대류를 모델에 포함
2. **Multi-zone model** — discharge zone(V_eff) + post-discharge zone(V_reactor-V_eff) 커플링
3. **V_reactor=250cm³ 기존 baseline 유지** — 물리적 근거 약하나 실용적 (RMSE~1.1)

## 참고 파일
- 테스트 스크립트: `Log_script/test_rate_corrections.py`, `Log_script/test_missing_reactions.py`
- V_eff 민감도 분석: `notes/20260330_veff_sensitivity.md`
