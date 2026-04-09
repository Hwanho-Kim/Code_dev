# Boundary Condition Formulation for 1D Diffusion-Reaction Model

## 핵심 원리

경계조건은 **계산 도메인 외부 또는 경계 자체의 미해석 물리만** 인코딩해야 한다.
1D PDE가 전체 액상(z=0→L)의 확산+반응을 해석하므로, BC에 액상 막 저항을 포함하면 이중 계산이다.

## 직렬 저항 모델 (Schwartz 1986)

기-액 물질전달의 총 저항:

```
1/K_G = 1/k_G + 4/(α·v̄) + 1/(H·β·k_L)
         ───     ────────    ──────────
         기상     계면(α_b)    액상(PDE가 처리)
```

- k_G = D_g / δ_gas : 기상 경계층 확산
- α·v̄/4 : Hertz-Knudsen 계면 동역학 flux (α_b = mass accommodation coefficient)
- k_L = D_l / δ_liq : 액상 경계층 확산 → **PDE가 해석하므로 BC에서 제외**

## 올바른 BC: 동역학적 Robin BC

```
-D_l ∂C/∂z|_{z=0} = k_gi × (C_eq - C(0,t))
```

여기서:
```
1/k_gi = 4/(α_b·v̄) + 1/k_g       (기상 단위, m/s)
k_g = D_g / δ_gas
v̄ = sqrt(8RT/(πM))                (Maxwell-Boltzmann mean speed)
```

액상 농도 단위로 변환:
```
k_mt = k_gi / H_cc                 (H_cc = dimensionless Henry constant)
H_cc = H_cp × R × T               (H_cp: M/atm, R = 0.08206 L·atm/(mol·K))
```

최종 코드 형태:
```
flux = k_mt × (C_eq - C_surface)   where C_eq = H_cp × c_gas_molar
```

## BC 유형별 비교

| BC | 기상 | 계면(α_b) | 액상 | 1D PDE에서 |
|-----|------|----------|------|-----------|
| Dirichlet | ✗ | ✗ | ✗ | 고용해도 종에 적합 (α≈1) |
| two_film | ✓ | ✗ | **✓ (이중계산)** | 부적합 |
| film | ✗ | ✗ | **✓ (이중계산)** | 부적합 |
| film_alpha | ✗ | ✓ | **✓ (이중계산)** | 부적합 |
| **gas_alpha** | **✓** | **✓** | **✗** | **올바름** |

## 이중 계산의 수학적 증명 (정상 상태)

반응 없는 정상 상태 확산: D_l·d²C/dz² = 0, C(L) = C_bulk

**올바른 경우 (Dirichlet):**
- C(0) = C_eq
- J = D_l·(C_eq - C_bulk)/L
- 유효 k_eff = D_l/L

**잘못된 경우 (Robin with D_l/δ):**
- -D_l·dC/dz|₀ = (D_l/δ)·(C_eq - C(0))
- J = D_l·(C_eq - C_bulk)/(L + δ)
- 유효 저항 = L/D_l + **δ/D_l** ← 존재하지 않는 추가 저항

## δ_gas 값

| 조건 | δ_gas | 근거 |
|------|-------|------|
| 정체 기상 | 1~5 mm | 자연 대류 경계층 |
| 플라즈마 제트 | 0.1~1 mm | 강제 대류 |
| Surface DBD (우리 시스템) | ~1~3 mm | 약한 대류 |
| 현재 코드 기본값 | 10 mm | **과대 (gas gap 전체)** |

δ_gas가 크면 gas-side 저항이 지배 → α_b 무감.
δ_gas가 작으면 계면 저항 비중 증가 → α_b 효과 나타남.

## 테스트 결과 (2026-04-09)

### gas_alpha (δ_gas = 10mm)
| α_b | pH | NO₃⁻ (µM) |
|-----|-----|----------|
| 0.01~0.05 | 3.99 | 45.6 |
→ α_b에 무감. gas-side 지배 (1/k_gas = 667 >> 1/k_int ≈ 0.5)

### gas_alpha (δ_gas = 1mm)
| α_b | pH | NO₃⁻ (µM) |
|-----|-----|----------|
| 0.01~0.1 | 3.00 | 987 |
→ 여전히 α_b 무감. NO₃⁻ 16배 과다. gas-side 여전히 지배.

### film_alpha (기존, δ_gas = 10mm)
| α_b | pH | NO₃⁻ (µM) |
|-----|-----|----------|
| 0.03 | 4.42 | 38.4 |
→ α_b 민감 (이중 계산된 액상 저항이 α_b와 결합)

실험값: pH=3.61, NO₃⁻=63µM

## Heirman 2025와의 차이

Heirman의 Eq.7: flux = α_b·D_l/Δx·(H_cc·c_g - c_l)

- 2D 축대칭 모델 + 대류 포함
- c_l = liquid phase **average** concentration (bulk average)
- Δx = convective boundary layer 두께 (PDE가 미해석)
- → D_l/Δx는 미해석 대류 경계층을 매개변수화

우리 모델:
- 1D, 대류 없음
- C_surface = surface cell 값 (PDE가 해석)
- PDE가 surface→bulk 확산 전체 해석
- → D_l/δ_liq는 PDE와 역할 중복

## 문헌 근거

- Schwartz (1986): 직렬 저항 분해. 각 저항은 독립적 물리 과정에 대응.
- Zheng, Wang & Bruggeman (2020, JVST A): 1D 액상 모델에서 α·v̄/4 동역학적 BC 사용. δ_liq 없음.
- Liu et al. (2021, AIP Advances): 3가지 BC 비교 — 열역학/확산/동역학. 독립적 대안이지 곱셈 결합 아님.
- Silsby et al. (2021): "δ는 대류 경계층과 연결. 대류 없으면 매개변수화 불필요."
- PRA/KM-SUB (Pöschl 2007, Shiraiwa 2010): bulk 전달이 해석되면 BC에는 α_s만 필요.
- Lewis & Whitman (1924): 막 이론의 δ는 "전적으로 허구적, 모델링 편의".

---
<!-- Last updated: 2026-04-09 -->
