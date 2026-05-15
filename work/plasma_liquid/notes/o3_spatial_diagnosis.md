# O3 Spatial / Temporal Diagnosis (2026-05-04 ~ 05-06)

## 배경

3.6 kV / 3.2 kV / 2.6 kV DIW Humid_fitting (three_film BC, HONOvar) 결과에서 사용자가
지적한 **3가지 문제**:

| # | 위치 | 증상 |
|---|---|---|
| 1 | fig1c (concentration time series) | **2.6 kV에서만** O3 (×1000 µM scale) 시계열에 1~5분 oscillation. 3.2/3.6 kV smooth. Henry 평형 도달 시점부터 노이즈 |
| 2 | fig2b_radical_rate (panel a) | O atom rate budget이 ΔC/Δt와 안 맞음. R28/R27 ±2e-14 mirror, dC/dt ≈ 0 |
| 3 | fig5_spatial (panel d, O3) | 3.6 kV 8분에 surface 7e-7 → 0.2 mm 8e-22 → 1 mm atol-noise → **4 mm peak 1.83e-10 (jump up)** → 9.8 mm decay. 비물리적 U-shape |

핵심 질문 (사용자 비판): **10 mm DIW 내 O3 농도가 10²⁰배 차이가 1~2 mm 안에서 발생하는 게 가능한가?**

---

## 진행한 부수 작업

**HONO/NO2 voltage-specific fine-tune** (`Figures/test/test_hono_finetune.py`):
- 이전 uniform HONO/NO2 = 0.10 → 2.6 kV NO2⁻ 19.9 µM (exp 0), 3.2 kV 14.7 µM (exp 3.58) over-prediction
- Sweep 결과: **2.6 kV → 0.005, 3.2 kV → 0.055, 3.6 kV → 0.097**
- 새 결과: 2.6 kV NO2⁻ 0.05 (baseline floor R19/R95), 3.2 kV 4.11 (+15%), 3.6 kV 20.68 (−0.3%) ✓
- **NO2⁻ voltage trend 단조 증가 회복** (이전 19.9/14.7/21.7 → 0.05/4.11/20.68)
- `gen_all_figures.py`에 `--label-suffix HONOvar` 옵션 추가 (기존 폴더 보존)

**fig5에 NO2⁻ panel 영구 추가**:
- `SPATIAL_SPECIES`에 `('HONO_total', 'NO2-', 'uM', 1e6)` 추가
- `gen_fig5`에 `_SPECIATE_IONIC` dict + pH-dependent speciation
  (NO2⁻ = HONO_total × Ka/(H+ + Ka), per-cell + per-snapshot)
- 향후 'HO2-', 'O2-' 추가 시 자동 speciation

---

## 문제 1 진단 (해결): R32 × NO2⁻ Lotka-Volterra limit cycle

### Phase A — 현상 정량
`Figures/test/diag_o3_henry_oscillation.py`

| Voltage | NO2⁻ µM | τ_R32 = 1/(k·[NO2⁻]) | τ_MT = L/k_mt | regime |
|---|---|---|---|---|
| **2.6 kV** | 0.05 | **43 s** | 573 s | **underdamped** |
| 3.2 kV | 3.77 | 0.53 s | 573 s | overdamped (1080×) |
| 3.6 kV | 19.97 | 0.10 s | 573 s | overdamped (5730×) |

- driving force `(C_eq − c_surf)/C_eq`는 모든 voltage에서 0.7+ 유지 — **Henry 평형 절대 도달 안 함, PDE 액상 저항 dominant**
- 2.6 kV만 fig1b MT flux 8e-9 plateau saturate, 3.2/3.6 kV monotonic 증가
- 즉 사용자 지적 "Henry 평형 도달 시점"은 실은 **MT flux convergence 시점**

### Phase B — chemistry isolation
초기 시도 (`diag_o3_phase_b1_r32_off.py`, `diag_o3_phase_b2_b4.py`)에서 R32 disable이 무효함을 발견.

**★ 결정적 버그 발견**: chemistry 수정 후 `chem._precompute_numba_arrays()` 호출 안 하면 Numba JIT compute_rates_batch가 원본 k 값 사용. `_rxn_data['k']=0` 만 수정하는 건 실제로 작동하지 않음.

### Phase B redo (`diag_o3_phase_b_redo.py`) — 버그 수정 후
| Case | bulk-only mean | detrended std (CV%) |
|---|---|---|
| baseline (R32 active) | 2.06 nM | **24.1%** |
| **R32 OFF** | 298.6 nM (×145) | **2.7%** ← 10× 감소 |
| R22-R32 ALL OFF | 471.4 nM | 2.2% |

**R32 disable 시 oscillation 거의 사라짐 → R32 가설 부활 확정**.

### Phase B5 — numerical artifact 배제
`Figures/test/diag_o3_phase_b5.py` (5 cases × atol/rtol/max_step sweep):

| Case | nfev | bulk_std% |
|---|---|---|
| baseline (1e-15, 1e-6, 1.0) | 25636 | 24.125 |
| atol 1e-18 | 31360 | 24.124 |
| rtol 1e-9 | 78053 | 24.114 |
| max_step 0.1 s | 25329 | 24.125 |
| all tight | 88946 | **24.125** |

**std 4 자리 소수점까지 동일** — BDF 정확도 1000× 강화해도 oscillation amplitude 변화 0.

### 결론
**R32 (O3 + NO2⁻ → O2 + NO3⁻, k=5×10⁵) ↔ NO2⁻ predator-prey type limit cycle**:
- 2.6 kV는 NO2⁻ 0.05 µM이라 R32 weakly damped → **underdamped 영역에서 limit cycle**
- 3.2/3.6 kV는 NO2⁻가 100~400× 커서 R32 overdamped → smooth
- Lotka-Volterra type: constant (smooth) input에서도 nonlinear chemistry feedback이 자체 limit cycle 생성

**물리적 chemistry feature** — input smoothness, BDF 정확도와 무관.
사용자 직관 "smooth input + saturating system → smooth output"은 linear system에서만 성립.

---

## 문제 3 진단 (부분 해결): 4 mm O3 peak

### Phase C (`diag_o3_phase_c_diff_chem_off.py`)
3가지 시뮬 비교:

| z (mm) | baseline | chem_off | diff_off |
|---|---|---|---|
| 0.003 | 7.08e-7 | 2.87e-5 | 1.16e-5 |
| 0.04 | 4.06e-10 | 2.79e-5 | 5.88e-15 |
| 0.2 | 7.98e-22 | 2.39e-5 | 5.88e-15 |
| 1.3 | 1.13e-22 | 6.49e-6 | 5.88e-15 |
| 4.0 | **1.83e-10** | **1.93e-8** | 5.88e-15 |
| 9.8 | 5.82e-15 | 1.14e-15 | 5.88e-15 |

**확인 사실**:
- chem_off (chemistry 모두 OFF): **diffusion-erfc analytical과 일치** (4 mm 1.93e-8 vs 예측 2.5e-8). **SG diffusion solver는 정상**.
- diff_off (D=0): 모든 deep cells 5.88e-15 균일 — chemistry가 trace 1e-30에서 시작해서 6e-15 수준 자체 생성. 작은 효과.
- baseline = diffusion in − chemistry consumption. 4 mm에서 chemistry가 99% 소비 (1.93e-8 → 1.83e-10).

### Phase D (`diag_o3_phase_d_atol_sweep.py`) — atol 가설 검증

| Case | z=4 mm | wall | nfev |
|---|---|---|---|
| chem_off atol=1e-15 | 1.93e-8 | 31s | 14k |
| chem_off atol=1e-30 | 1.93e-8 | 110s | 76k |
| baseline atol=1e-15 | **1.83e-10** | 42s | 18k |
| baseline atol=1e-30 | **1.83e-10** | 560s | **209k** |

**atol을 1e-15 → 1e-30 (10¹⁵배 강화)에도 baseline 4 mm O3 변화 없음**. nfev ×11.5, wall ×13배 cost 증가에도 결과 동일.

→ **atol-floor 가설 완전 기각**. baseline 4 mm = 1.83e-10은 **BDF가 atol 1e-30 정확도에서도 안정적으로 추적하는 PDE의 실제 해**.

### Phase E (`diag_o3_phase_e_cell40_dydt.py`) — cell 40 dydt budget 분해

| t (s) | O3 (M) | diff_in (M/s) | chem_net (M/s) | total (M/s) |
|---|---|---|---|---|
| 60 | 2.93e-15 | +1.17e-19 | **+2.98e-17** ← positive (R20_f) | +3.0e-17 |
| 120 | 1.94e-14 | +1.72e-15 | -1.15e-17 | +1.6e-15 |
| 180 | 1.21e-12 | +6.95e-14 | -1.22e-15 | +6.0e-14 |
| 240 | 1.15e-11 | +4.04e-13 | -8.03e-15 | +3.21e-13 |
| 300 | 4.25e-11 | +1.01e-12 | -2.81e-14 | +7.19e-13 |
| 420 | **1.63e-10** | **+1.81e-12 (peak)** | -5.32e-14 | +9.05e-13 |
| 480 | 1.83e-10 | +6.63e-13 | -2.22e-13 | -3.6e-13 |
| 600 | 4.45e-11 | -3.05e-13 | -8.88e-13 | -1.05e-12 |

**Top reactions at cell 40 (peak |rate|)**:
1. R32 (O3+NO2⁻): 9.3e-13 M/s
2. R25 (O3+O2⁻): 3.1e-14
3. R27 (O3+OH): 3.1e-14
4. R28 (O3+HO2): 2.8e-15
5. R20 (O+O2 ↔ O3): 5.6e-16

**Mechanism**:
- **diff_in이 dominant positive source** (peak 1.81e-12 M/s at t=420s)
- chem_net 소량 negative (R32 lifetime 850s at 4 mm with NO2⁻=2.35 nM)
- O3 chain: surface → cell 1 → ... → cell 39 → cell 40 (diffusion 누적 over 480s)

### 부분 결론
- BDF/atol/diffusion solver/chemistry batch 모두 numerical level 정상
- **baseline 4 mm O3 = 1.83e-10은 우리 PDE의 정확한 해**

**잔여 의문 (미해결)**:
- mid cells (0.5 ~ 2 mm)에서 NO2⁻ µM~mM 수준이라 R32 lifetime <2 s
- Diffusion transit time through 1~2 mm = (1mm)²/D ≈ 1000 s
- Timescale gap에도 O3가 mid cells을 통과해서 4 mm까지 leak
- 단순 reactive-penetration 분석 vs 실제 PDE 해 사이 **1~2 orders 불일치**

가능 원인:
- **(A)** Early-time (t < 60 s) NO2⁻ build-up 전에 free diffusion으로 O3가 deep까지 도달. NO2⁻ 누적 후에는 surface drain. Deep의 잔여 O3는 NO2⁻ 작아 천천히 소비.
- **(B)** SG scheme이 extreme gradient (10¹⁵ ratio across 0.2 mm)에서 미세 numerical artifact
- **(C)** 미발견 subtle bug (예: chemistry RHS와 transport coupling, BC 적용 순서 등)

**결정적 verification**: 독립 reference solver (단순 FD + explicit RK4)로 동일 PDE 풀어 비교. 미진행.

---

## 문제 2 (O atom rate budget) — 미진행

Phase E 시점에서 사용자가 문제 3 우선 진행으로 결정. 진단 직전 상태:
- O atom 농도 ~10⁻¹⁸ M (atto-Molar)
- atol=1e-15 한참 이하라 BDF 정확도 제어 영역 밖
- spatial avg가 numerical zero (부호 random) 수준
- Phase B5에서 입증된 BDF 정확도 1000× 강화도 무효 가능 (농도 자체가 atto-Molar)

**예상 fix 방향** (보류):
- 'O' atom QSSA 적용 (chemistry_1d.py `apply_qssa` 인프라 활용)
- 또는 chemistry network에서 O 관련 6개 reactions (R20, R73, R106-R109) 비활성화 (기여도 미미)
- 또는 fig2b panel에서 O 제외 + NO3 radical로 대체

---

## 코드 발견 사실

### Chemistry는 모든 cell에 적용됨
`pde_solver.py:894-895`:
```python
y_2d_clipped = np.clip(y_safe.reshape(N_z, N_s), trace, max_conc)
dydt[:] = self.chem.compute_rates_batch(y_2d_clipped).ravel()
```
`chemistry_1d.py:1325-1327`:
```python
def compute_rates_batch(self, y_2d):
    for j in range(N_z):  # 49 cells loop
        result[j] = self.compute_rates_numba(y_2d[j])
```

### Diffusion: Scharfetter-Gummel finite-volume flux
`pde_solver.py:760-762`:
```
J_{j+1/2,i} = (D_i / h_{j+1/2}) × [B(α) × C_{j,i} − B(−α) × C_{j+1,i}]
α_{j+1/2,i} = −Z_eff,i × E_{j+1/2} × h / V_T
```

- **Poisson OFF (E=0)이라 α=0 → 표준 Fickian flux**: J = D/h × (c_j − c_{j+1})
- 학계 표준 (Scharfetter & Gummel 1969, semiconductor device physics)
- 농도 구배 10²⁰에 대한 수치 정확도는 정상 (Phase C/D 검증)

고려되는 것: Fickian diffusion, acid-base speciation 효과적 charge, drift (E-field 있을 때만, 현재 비활성).
고려 안 되는 것: convection/advection, activity coefficient, multi-component cross-diffusion.

### atol species-specific 이미 적용
`pde_solver.py:998-1005`:
```python
_tight_species = ['OH', 'O-', 'O3-', 'HO3', 'NO3']
for sp_name in _tight_species:
    atol_arr[j * N_s + si] = atol_base * 0.01  # 100× tighter
```
즉 OH/O-/O3-/HO3/NO3는 atol = 1e-17. 다른 trace species (예: O atom)는 default 1e-15.

### Numba precompute 필수 (★ 중요 버그)
chemistry 수정 시 다음 두 가지 모두 필요:
```python
chem.reactions[i]['k'] = new_k       # Python-level dict
chem._rxn_data[i]['k'] = new_k       # _precompute_reaction_data 결과
chem._precompute_numba_arrays()      # ★ Numba JIT 배열 재구성
```

`compute_rates_batch`는 Numba JIT로 precomputed 배열 (`_nb_*`)만 사용하므로,
`_rxn_data`만 수정하면 작동 안 함. Phase B1 초기 시도가 이 버그로 실패한 이유.

기존 `Ver4_1D/run_kR3_sweep.py`도 같은 패턴 (CLAUDE.md 2026-04-22 entry 참조).

### initial condition
`pde_solver.py:835-863` `build_initial_condition`:
- 모든 cells trace = 1e-30
- `O2 = 2.5e-4 M` (Henry 평형)
- `N2 = 5e-4 M`
- `H+ = 10^-pH` (default pH 7), `OH- = 10^-14 / H+`
- `OH = 1e-12` (radical seed)
- saline 모드: `Cl- = 0.154 M`
- `± 1e-6` random 곱연산 perturbation 후 trace floor clip

### Macro step 후 clip
`pde_solver.py:1075-1076`:
```python
np.clip(y, DEFAULTS.trace_concentration, ODE_CONFIG.max_concentration, out=y)
```
`trace = 1e-30`, `max = 1.0`. 이 floor가 1e-30이라 atol=1e-15보다 한참 아래
→ BDF가 atol 이하 영역에서 만든 음수/garbage가 그대로 macro step 결과에 저장됨.

---

## 변경/생성 파일 목록 (이 진단 동안)

**신규 진단 스크립트** (모두 `Figures/test/`):
- `test_hono_finetune.py`
- `diag_o3_henry_oscillation.py` (Phase A)
- `diag_o3_phase_b1_r32_off.py` (Phase B1, Numba bug 발견 직전)
- `diag_o3_phase_b2_b4.py` (Phase B2/B4)
- `diag_o3_phase_b_redo.py` (Phase B redo, fixed)
- `diag_o3_phase_b5.py` (Phase B5 BDF accuracy)
- `diag_o3_phase_c_diff_chem_off.py` (Phase C)
- `diag_o3_phase_d_atol_sweep.py` (Phase D)
- `diag_o3_phase_e_cell40_dydt.py` (Phase E)

**산출물**:
- 위 스크립트들의 `fig_*.png/pdf`, `diag_*.txt` (numerical summaries)

**메인 코드 변경**:
- `Figures/gen_all_figures.py`:
  - `RH80_RATIOS` HONO_NO2 voltage-specific (0.005/0.055/0.097)
  - `--label-suffix` argparse + `out_folder` 적용
  - `gen_fig1` suptitle 동적 HONO/NO2 표시
  - `SPATIAL_SPECIES`에 NO2⁻ 추가
  - `gen_fig5`에 `_SPECIATE_IONIC` dict + per-cell speciation
- `Figures/DIW results/gen_voltage_comparison_HONOvar.py` (신규, gen_voltage_comparison_HONO010.py 기반)

**figure 산출**:
- `Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film_HONOvar/` 3 폴더 × 9 figure (fig1, 1b, 1c, 2, 2b, 3, 4, 5, 6) + cache
- `Figures/DIW results/fig_voltage_comparison_HONOvar.{png, pdf}`

---

## 결론 요약

| 문제 | 상태 | 결론 |
|---|---|---|
| 1 (2.6 kV O3 noise) | **해결** | R32 × NO2⁻ Lotka-Volterra limit cycle, physical |
| 2 (O atom budget) | 미진행 | atto-Molar 농도, atol 한참 이하 의심 |
| 3 (fig5 4 mm peak) | **부분 해결** | Numerical 정상 검증 통과. 단순 RD analysis와 1~2 orders 잔여 의문 |

**다음 단계 후보**:
1. Reference solver (FD+RK4) 작성 — 문제 3 결정적 verification
2. NO2⁻ 시공간 추적 — early-time free diffusion 가설 검증
3. R20 OFF 시뮬 — chem_net positive 영향 정량
4. 문제 2 진단 (O atom QSSA 또는 chemistry 제거)
5. 문제 1 fix 옵션 (limit cycle 시각화 처리, plot smoothing 등)
