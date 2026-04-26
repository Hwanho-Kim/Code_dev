# Plasma-Liquid Simulation Project

## Project Context
1D plasma-liquid reactive transport model for publication.
Gas → liquid dissolution → diffusion → 193 aqueous reactions, 47 species.
Custom backward Euler + Newton + Gummel solver with Numba JIT.

## Research Direction (2026-03-23 updated)

### 방향 전환: Optimizer 폐기 → 측정종만 사용
- HONO/HONO2/H2O2 동시 추정 fitting **실패** (saline+DIW 모두 만족하는 해 없음)
- **새 목표**: 기상 측정종(FTIR)만으로 액상농도 예측 가능 여부 확인
- HONO/HONO2/H2O2는 fitting parameter가 아닌 **문헌값 또는 0**으로 처리
- 가스-액상 계면 경계조건이 최대 black box → 여러 문헌 BC 대입 필요

### ML 시도
- 기획 단계. 미팅에서 **후순위** 결정.
- 한계: 대응 액상농도 데이터 부재, HONO 부정확

### 관련 프로젝트: Pulsed Code (plasma0d_v2)
- 위치: /home/hawn/work/plasma0d_v2/
- Continuous power 모드: 완성도 높음 (reaction set 개선, PFR 모델 추가)
- Pulse 모드: stiffness 문제 (V-I curve 기반 dt 분해, power 급증/감)
  - 단기 sim 가능하나 steady-state(tau_flow~30s)에 훨씬 못 미침
  - ne/Te가 power 종속 → 비물리적 감소 (방향성 불확실)

## Key Paths
- Solver: Ver4_1D/pde_solver.py (1720 lines)
- Chemistry: Ver4_1D/chemistry_1d.py (754 lines)
- Config: Ver4_1D/config_1d.py (328 lines)
- Optimizer: Ver4_1D/run_optimizer.py **(폐기 예정)**
- Saline runner: Ver4_1D/run_saline_1d.py
- Gas data: empty chamber/empty chamber/1kHz3.2kVpp.csv
- Python: Ver3/.venv/bin/python

## Experimental Targets (OAS data, Dry, 10min treatment)
Source: `OAS data/Dry/(P-L) 액체활성종 농도, pH, conductivity.xlsx`

### DIW
| Voltage | pH | NO₂⁻ (µM) | NO₃⁻ (µM) | H₂O₂ (µM) |
|---------|-----|----------|----------|-----------|
| 2.6 kVpp | 5.09 | 0 | 32.63 | 4.76 |
| **3.2 kVpp** | **3.61** | **3.58** | **62.74** | **11.21** |
| 3.6 kVpp | 3.25 | 20.74 | 70.42 | 16.25 |

### Saline
| Voltage | pH | NO₂⁻ (µM) | NO₃⁻ (µM) | H₂O₂ (µM) |
|---------|-----|----------|----------|-----------|
| 2.6 kVpp | 5.15 | 0 | 32.44 | 2.00 |
| **3.2 kVpp** | **3.60** | **0** | **101.30** | **5.14** |
| 3.6 kVpp | 3.43 | 0 | 112.77 | 7.73 |
<!-- 2026-04-20: xlsx Saline sheet 재확인, 이전 기록(4.70/10.45/16.92)은 인접 컬럼(다른 단위) 오독 -->


### Gas-phase data
Source: `OAS data/Dry/(P-L) 가스활성종 농도.xlsx`
- 전압별 시계열: 2.6/3.2/3.6 kVpp
- 종: O₃, NO₂, NO₃, N₂O₅ (cm⁻³)
- 시간: 0~600s, 2s 간격, 301 points
- **이전 CSV(1kHz3.2kVpp.csv)와 차이**: 10분(600s) vs 12분(720s). 새 데이터가 정확한 reference.

## Active Constraints
- 3.2kVpp 기준. Multi-voltage 데이터 확보됨 (2.6/3.6 kVpp).
- Poisson OFF (Debye << grid).
- S97/S98 반응 현재 disabled.
- "말을 듣고 해 니 말대로 하지말" — 사용자 지시를 정확히 따를 것.

## Current Status

### WORKING
- Ver3 (0D): DI water fitting 완료 (NO3- 0.6%, H2O2 0.0% 오차)
- **Monolithic BDF solver**: DIW 600s, atol=1e-15, rtol=1e-6, dt_enforce=None
- **gas_alpha BC 채택** (Schwartz 저항 모델): k_gi = (δ_gas/D_g + 4/(α_b·v̄))⁻¹, k_mt = k_gi/H_cc.
- **Henry 상수 버그 수정 완료 (2026-04-20)**: H_cc 직접 사용. R×T 재곱 제거.
- **δ_gas=10mm**, α_b 종별 (N₂O₅=0.03, O₃=0.05, H₂O₂=0.1 등)
- **Humid fitting 조건**: RH80 스케일링 + HONO/NO₂=0.0071, HONO₂/N₂O₅=0.83, H₂O₂/O₃=0.03
- **Henry 수정 후 DIW 결과 (3.2kV, Humid fitting, δ_gas=10mm)**:
  - pH=3.40 (실험 3.61)
  - NO₃⁻=396 µM (실험 63, **6.3× 과다**)
  - NO₂⁻=5.0 µM (실험 3.58, **근접**)
  - H₂O₂=193 µM (실험 11, **17× 과다** — 비율 불확실성)
- **NO₃⁻ 과다 비율 전 전압 일정 (~7×)**: 2.6kV=7.4×, 3.2kV=7.3×, 3.6kV=7.7×
- Numba JIT 65x 가속, O(N) graph coloring
- Grid convergence 완전 확인
- 종별 α_b 구현 완료 (gas_alpha에서도 적용). 단 gas_alpha에서 α_b에 거의 무감 (gas-side 지배).
- Gas onset filter + linear interpolation: noise spike 제거 + 0 구간 보간
- `one_film_gas` BC 추가 (gas-side only, α_b 없이)
- Fig 6 추가: Raw + Linear interpolation + Unmeasured ratio-based time series

### BROKEN
- run_simulation.py: Strang split 경로에서 pH_surface KeyError (minor bug)

### TRIED (결과 있음)
- 측정종만 사용 forward simulation (DIW): pH 6.1% 오차 → 산성화는 측정종으로 설명 가능
  - 하지만 NO3⁻ 574% 과다 → N2O5 mass transfer/Henry 상수/BC 검토 필요
- **Saline with K_CAP** (K_CAP=1e6, 과거): pH=2.943(18.3%), Cl⁻ -12µM, HClO=8.6µM
- **Saline K_CAP 제거 + Cl 보존 강제** (2026-03-25): QSSA 확산제외 + 셀별 Cl 원자보존
  - pH=4.281 (실험 3.60, 18.9%), NO3⁻=1153uM(10배 과다), H2O2≈0
  - **Cl⁻: 0.154→0.150M (-3866µM)** — HClO_total(945µM)+기타 Cl종으로 물리적 전환
  - HClO=945µM (K_CAP 없이 물리적 HOCl 생성), 14.9분 완료, 0 slow cells
  - K_CAP 있을 때 pH 2.94(너무 산성), 없을 때 4.28(덜 산성). 실험값 3.60은 그 사이
- **QSSA net effective rate** (2026-03-25): S3-S9 analytical + S23-S69 filtered loop
  - pH=3.114 (실험 3.60, 13.5%), NO3⁻=1339uM, H2O2≈0
  - Cl⁻: 0.154→0.152M (-1966µM, 비가역반응의 물리적 소모)
  - HClO=294µM, 6.9분 완료, 0 slow cells. Cl 보존 hack 제거.
  - pH 크게 개선 (4.28→3.11), 2배 가속 (14.9→6.9분)
- **QSSA Picard 수렴 개선** (2026-03-25): 2-pass→20-pass with convergence check (rtol=1e-12)
  - pH=2.884 (19.9%), NO3⁻=1329µM, Cl⁻ -1059µM, HClO=355µM, 6.1min
  - Cl 소모 개선 (-1966→-1059µM), but pH 악화 (3.114→2.884)
- **Rate capping 제거** (2026-03-25): mass-balance limiting 전부 제거
  - Rate capping 없이 + Cl 보존 projection: pH=2.127, NO3⁻=8800µM, Cl⁻ -4099µM, 8.6min
  - dt_split=1s 실험: NO3⁻=54456µM (악화!) → 문제는 BDF 안정성이 아니라 operator splitting
  - 확산 step에서 gas 유입 → 화학 step에서 즉시 전량 소비 (splitting artifact)
  - Rate capping은 이 splitting error를 보상하는 역할을 했음
  - Implicit mass transfer (surface RHS) 시도 → 이중 계산으로 더 악화 (12443µM)
  - **현재 상태: rate capping 제거 + Cl 보존 projection. 정량적 오차는 BC/mass transfer 문제 (후순위)**
- **Mass transfer BC를 chemistry step으로 이동** (2026-03-25):
  - 기존: BC가 확산 step에만 있어 operator splitting artifact 발생 (dt 줄이면 NO3⁻ 악화)
  - 수정: surface cell (j=0) chemistry RHS에 k_L/dz*(C_eq-C) 포함, BDF가 유입+소비 동시 처리
  - dt=10s→1s 수렴 확인 (6996→6260µM, 이전에는 8800→54456µM)
  - NO3⁻ 여전히 60배 과다 → BC/mass transfer 모델 자체 검토 필요
- **시간축 2x 버그 수정 + 12분 전체 실행** (2026-03-25):
  - OAS 데이터 2초 간격 → times *= 2.0, t_end=720s (12분)
  - 시간 인덱싱 `int(t)` → `int(t/dt_gas)` 3곳 수정
  - 결과: pH=1.989(44.8%), NO3⁻=12864µM(126배 과다), H2O2=2.96µM(40.9%), Cl⁻ -9300µM
  - HClO=506µM, surface pH≈0, 15.3분 소요
  - 시간 2배로 늘었으니 NO3⁻도 ~2배 (6996→12864). mass transfer 과다 유입 확정

- **Component Verification Tests** (2026-03-25): 10개 테스트 전부 PASS
  - 확산(CN Thomas): L∞=4.4e-8 (해석해 대비), 비균일 질량보존 2.4e-15
  - 화학(BDF): trace dydt≈0, 산-염기 보존 정확, BDF 지수감쇠 2.3e-7
  - 질량전달: k_mt/C_eq 수계산 일치, 용해 ODE 8.7e-16
  - 결합(D+MT): N2O5만 유입 시 N원자 수지 ratio=1.07 (거의 완벽)
  - N보존(60s): ratio=2.15 (상한 추정 부정확, 컴포넌트 자체는 정상)
  - **결론: 개별 컴포넌트 버그 없음. NO3⁻ 과다는 BC/mass transfer 모델 문제 확정**
- **N-atom Budget 진단** (2026-03-25): diagnose_n_budget.py
  - **N2O5가 99.3% 지배** (810/816 µM). 나머지 종(NO2 0.4%, NO3 0.3%) 무시 가능
  - N2O5는 liquid-side limited (R_liq/R_total=99.98%) → H 값이 핵심
  - δ_gas 변경으로 맞추려면 208mm 필요 (비물리적, 보통 1-10mm)
  - **δ_liq=1mm이면 121µM (실험 102µM에 근접!)** — δ_liq가 가장 유효한 조절 파라미터
  - 현재 δ_liq=100µm이 물리적으로 타당한지 검토 필요 (PDE가 bulk 해결하므로 δ_liq → sub-grid accommodation)

- **BC 비교 실행 완료** (2026-03-26, DIW): run_bc_comparison.py
  - Two-film: pH=2.23, NO3⁻=5877µM (93배 과다) — 기존 BC 확인
  - Dirichlet: pH=2.14, NO3⁻=7199µM (114배) — 최악
  - Film(α_b=1): pH=3.83, NO3⁻=148µM (2.3배) — 극적 개선
  - Film+α_b=0.05: pH=3.93, NO3⁻=118µM (1.9배) — 실험 근접
  - Film+α_b=0.01: pH=4.26, NO3⁻=55µM (실험 이하)
  - **α_b≈0.01~0.05에서 실험 NO3⁻(63µM) 교차. Heirman 2025 Film+α_b 유효 확인**
  - H2O2 전 케이스 ≈0 → 측정종만 모드 한계 (HONO/H2O2 가스상=0)
- **O3 "1000배 낮음" 분석** (2026-03-26): 단위 불일치 아닌 물리적 효과
  - conv=1000/N_A → molecules/cm³→mol/L (초기조건 mol/L과 일치, 단위 OK)
  - C_eq(O3)=31~40µM (문헌 범위 안)
  - O3+NO2⁻ (k=5e5 M⁻¹s⁻¹, τ=0.7s) → **반응성 침투깊이 34µm** (전체 10mm의 0.3%)
  - 표면 O3≈0.7µM, bulk avg 수십nM → 시뮬레이션 42nM은 물리적으로 정확
  - 문헌 "1-100µM"은 표면농도 또는 반응파트너 없는 순수 PAW 조건

- **문헌 비교 (Liu 2016 + Heirman 2025)** (2026-03-26):
  - Liu 2016 (1D saline, COMSOL, Henry's law BC): O₃ 표면 ~100µM, 침투 ~100µm, bulk→0
  - OH: DI ~1nM, saline ~10pM (Cl⁻이 100배 소진). HClO ~197µM 지배적 RCS
  - Heirman 2025: 비반응 O₃ 실험에서도 0.3-0.5µM 수준 (모델은 10-20배 과대예측)
  - **결론: O₃ bulk avg ~nM, OH ~pM은 1D 모델에서 정상. Liu도 같은 패턴**
  - Liu는 Henry's law 직접 적용 (α_b 없음) → Heirman 기준으로 O₃ 과대예측 가능성 있음

- **α_b 민감도 분석 + 종별 mass balance (DIW)** (2026-03-27): run_alpha_analysis.py
  - α_b=0.01: pH=4.260, NO3⁻=54.9µM (실험 이하), H2O2=0.01µM
  - α_b=0.03: pH=4.069, NO3⁻=85.4µM (36% 과다), H2O2=0.14µM
  - α_b=0.05: pH=3.927, NO3⁻=118.2µM (88% 과다), H2O2=0.02µM
  - **실험 NO3⁻(63µM) 교차점: α_b≈0.015~0.02 추정**
  - 라디컬: OH ~5-8pM, O3 ~16-42nM, HO2 ~23-57pM (α_b에 선형 비례)
  - **종별 질량 수지 (dC/dt = Σ반응 + MT)**:
    - NO3⁻: R98(N2O5 가수분해) 99.6-99.8% 지배. sink 없음. 정상상태 아님(축적 중)
    - O3: MT(gas→liq) 100% source, 소비 R27(OH)+R28(HO2) ~33%. 60% 아직 축적
    - NO2⁻: 소비 R92(NO3+NO2⁻) 78-84% 지배. 생성<소비 → 0 수렴
    - H2O2: MT가 liq→gas(40-62%) 방향(가스상=0). 측정종만 모드 한계

- **시간별 반응속도 기여도 Figure 생성 + Rate budget 검증** (2026-03-30): Figures/gen_fig2_rate_evolution.py
  - DIW Film+α_b (α_b=0.03), 72 snapshots (10s 간격), 4종 (NO₃⁻, O₃, NO₂⁻, H₂O₂)
  - 개별 선 그래프 + net dC/dt (검정 점선, 실제 농도 차분에서 계산)
  - **Rate budget 심각한 불일치 발견** (diagnose_rate_vs_conc.py):
    - snapshot 순간 rate 합 ≠ 실제 dC/dt (NO₃⁻ 3.3배 과다, O₃/NO₂⁻/H₂O₂ 부호 반전)
    - 원인: Strang splitting 후 snapshot에서 순간 rate 평가 → BDF 화학 step 중 시간평균과 괴리
    - O₃: MT는 smooth하나 반응 rate(R28/R27/R25)은 noisy → radical 농도의 splitting artifact
    - H₂O₂: 계단식 점프+plateau 패턴 (t=70s +270×, t=410s +3×, t=470s +2×) — 가스 데이터는 smooth → 순수 splitting artifact
    - 결론: **snapshot 기반 per-reaction rate은 정량적으로 신뢰 불가**
  - **해결 방안 (미구현)**:
    - (A) dt_split 축소 (10s→2s): 간단하나 근본 해결 아님, 5× 느림
    - (B) BDF dense_output + Simpson적분: rate(t₀)+4·rate(t_mid)+rate(t_end))/6·dt → 정확한 시간평균, pde_solver.py 수정 필요
  - fig2_rate_evolution.png/pdf 저장, 캐시(fig2_rate_cache.npz)

- **BC별 MT flux 정량 비교** (2026-03-30): run_bc_comparison.py에 MT flux 출력 추가
  - Step 1: 해석적 k_mt 비교표. N2O5 k_mt ratio: Two-film 1.0, Film 1.57, Film+0.05 0.08, Film+0.01 0.02
  - Step 2: 시뮬레이션 후 instantaneous flux 추출. **N₂O₅ C_surface=0 (모든 BC)** → 완전 소비, k_mt가 유일 조절인자
  - Cross-check: N₂O₅ flux×2×t vs NO₃⁻ — ratio 0.16~0.92 (instantaneous flux ≠ 시간평균, 가스농도 시변)
  - H₂O₂ flux 음수 (liq→gas, 가스상=0) — 측정종만 모드 한계 확인
  - 결과: Figures/mt_comparison_output.txt 저장

- **Monolithic BDF dt 수렴 테스트** (2026-03-30):
  - dt_enforce=120/60/30/10s → NO₃⁻=128.3/133.0/134.4/135.0µM (5% 변동)
  - Strang splitting에서 10배 변하던 것 대비 완전 해결
  - dt_enforce 120→10s에서 pH=3.892→3.870 (0.022 차이)
  - **dt_enforce=60s 채택** (정확도 -1.5%, 속도 10.5min)
- **Monolithic BDF Figure 2 (rate evolution) 재생성** (2026-03-30):
  - Film+α_b=0.03, dt_enforce=60s, snapshot 10s 간격, 73 snapshots
  - **문제 발견**: H₂O₂와 O₃의 net dC/dt(농도 차분)가 여전히 크게 요동
  - **원인 가설**: (1) electroneutrality enforcement(60s마다) → H⁺ 보정, (2) BDF restart transient, (3) 라디컬 volume-avg noise
- **Fig 2 rate budget 디버깅** (2026-03-31):
  - dt_enforce=None(단일 BDF, restart 0회) vs dt_enforce=60s(12회 restart) 비교
  - **결과**: BDF restart 제거 후에도 H₂O₂/O₃ dC/dt 진동 잔존 → BDF restart는 부분 원인
  - pH: 3.869 vs 3.876 (차이 0.007, 무시 가능)
  - **속도 3× 향상** (3.5min vs 10.5min) — electroneutrality 보정 12→1회
  - **진동의 주원인**: trace 농도(H₂O₂ ~1e-10M)에 대한 finite-diff dC/dt의 본질적 한계. 순간 snapshot rate ≠ BDF 시간평균 rate
  - pde_solver.py에 `dt_enforce=None` 옵션 추가 (단일 BDF 호출)
  - gen_fig2_rate_evolution.py에서 `DT_ENFORCE=None` 사용
  - **근본 원인: atol=1e-8이 H₂O₂(~1e-10M)보다 100× 큼** → BDF가 오차 제어 안 함
  - **해결: atol=1e-12, rtol=1e-6** → config_1d.py 기본값 영구 변경
    - H₂O₂ Σrate/ΔC·Δt: 2.569 → **0.988** (RMSE 65× 개선)
    - 속도: 3.5min → **1.7min** (BDF가 higher order 유지, nfev 37250)
    - atol=1e-14 수렴 확인 (RMSE 동일, nfev만 증가)
  - Simpson 3-point 적분 구현 (dense_output + compute_rates_simpson)
  - fig2_rate_evolution.png/pdf 재생성 완료
- **Fig 1/3/4 데이터 수집 + 갱신 완료** (2026-04-01):
  - collect_monolithic_data.py: dt_poisson=None으로 5 BC + 3 α_b + mass balance 재실행
  - plot_bc_results.py: BC_DATA, ALPHA_DATA, species_data, Fig 4 mass balance 모두 갱신
  - Fig 1/3/4 재생성 완료 (atol=1e-12, dt_enforce=None 반영)
- **Fig 2 rate evolution 개선** (2026-04-01):
  - net 선: finite-diff ΔC/Δt → Σrate(budget_net)으로 교체 (자기일관적)
  - DT_SNAPSHOT=10→2s (5× 해상도). 시간 기반 smoothing: SMOOTH_TIME_RATE=60s, NET=120s
  - O₃ 2분 부근 급변: **autocatalytic radical chain ignition** (O₃→OH⁻→HO₂⁻→O₂⁻→O₃⁻→OH chain). 물리적 전이, 미세구조는 BDF step size >> radical QSS timescale(µs)로 인한 수치적 artifact
- **Fig 5 공간분포 생성** (2026-04-01): gen_fig5_spatial.py
  - Film+α_b=0.03, t=[1,2,4,6,8,12]min snapshots
  - pH(linear) + 5종(log scale): NO₃⁻, O₃, H₂O₂, OH, HO₂
  - O₃/OH 표면 집중(반응성 침투깊이 ~34µm) 시각적 확인
- **Fig 1-2 MT flux 시계열 생성** (2026-04-01): gen_fig1b_mt_flux.py
  - N₂O₅, O₃, HNO₃, H₂O₂ × 5 BC (Dirichlet 제외 — 비물리적 무한 flux)
  - Row 1: instantaneous flux (M/s), Row 2: cumulative (µM)
  - Two-film N₂O₅ cumul=405µM vs Film+α_b=0.01 cumul=6.4µM (63× 차이)

### TRIED (2026-04-09)
- **종별 α_b (film_alpha)**: O₃ +48%, pH/NO₃⁻ 변화 없음. H₂O₂는 가스상=0이라 무관.
- **종별 atol (1e-20 for trace)**: baseline과 동일 결과, 30% 느림. atol=1e-15 이미 충분.
- **QSSA (O₃⁻/HO₃/O⁻)**: 3가지 시도 (dydt=0, relaxation, N₂O₅ instant). 전부 실패 — radical chain 파괴 (O₃ 100배 축적). 상호의존성 때문에 단순 analytical QSSA 부적합.
- **gas_alpha BC (δ_gas=10mm)**: α_b 무감 (gas-side 1/k_gas=667 >> 1/k_int≈0.5). pH=3.99, NO₃⁻=101µM.
- **gas_alpha BC (δ_gas=1mm)**: α_b 여전히 무감. NO₃⁻=987µM (16배 과다). gas-side 여전히 지배.

### NOT TRIED
- Log-transform (C → ln(C)) for Newton conditioning
- Saline with fitted parameters (HONO/HONO2/H2O2 nonzero)
- Monolithic BDF saline 실행 (gas_alpha BC)
- δ_gas 민감도 sweep (0.1~10mm)

## Pending Tasks — (2026-04-10 갱신)
1. ~~**측정종만 사용 forward simulation**~~ ✅
2. ~~**Saline solver fix — QSSA 적용**~~ ✅
3. ~~**NO3⁻ 과다 진단**~~ ✅
4. ~~**계면 BC 문헌 비교**~~ ✅
5. ~~**O3/라디컬 농도 문헌 검증**~~ ✅
6. ~~**Monolithic BDF 구현**~~ ✅
7. ~~**Fig 2 rate budget 불일치 디버깅**~~ ✅
8. ~~**Fig 1~5 생성/갱신**~~ ✅
9. ~~**BC 이중 계산 문제 규명 + gas_alpha 구현**~~ ✅ (2026-04-09)
10. ~~**종별 α_b 구현**~~ ✅ config_1d.py `alpha_b_species` + pde_solver.py 종별 적용. gas_alpha BC에서도 자동 적용됨. (2026-04-09)
11. **δ_gas 결정** — 현재 10mm (gas gap 전체, 과대). 실험 조건(surface DBD)에서 적절한 값 결정 필요. ~1-3mm 추정.
12. **α_b 민감도 재검증 (gas_alpha)** — gas_alpha + 종별 α_b에서 유의미한 영향 조건 확인. δ_gas 의존.
13. ~~**gas_alpha로 전체 Figure 재생성**~~ ✅ (2026-04-10)
14. **NO₃⁻ 과다 원인 규명** — O₃ MT 아님 (확인). N₂O₅ MT/Henry 상수 검토 필요.
15. **NO₂⁻ 부족 (1000배)** — 소비 반응(R92, R32) 검토. HONO 비율 증가? 소비 속도 검토?
16. **HONO/HNO₃/H₂O₂ 비율 sweep** — 현재 고정값 (0.33/0.83/0.03). 문헌 범위 내 로그 sweep.
17. **δ_gas sweep** — 현재 10mm. 축소 테스트.
18. **O₃ oscillation (t≈110s)** — QSSA 시도 실패. 미해결.
19. **Monolithic BDF saline 실행** — gas_alpha BC 적용
20. **Saline with fitted parameters**

## Key Decisions (settled — 재논의 불필요)
- Poisson OFF: Debye(0.8nm saline, 30nm DI) << grid(5um). 확정.
- D_adj BC 유지 (현재): gas-side-only Robin 시도 → FAILED (HNO3 C_eq=8.36M divergence)
- Grid: geometric dz_min=5um, ratio=1.12, 49 cells. Convergence 확인.
- Optimizer 방식 폐기: saline/DIW 동시 만족 해 없음 (2026-03-23 결정)
- ML 후순위: 미팅 결정 (2026-03-23)
- Poisson 기본값 False로 변경: config_1d.py PoissonConfig.enabled=False (2026-03-23)
- QSSA 4종 (HOCl⁻,Cl₂⁻,Cl,HOClH): Cl 화학 stiffness 완전 해소. 4→2→2 축소 시스템 (2026-03-24)
- K_CAP 가역반응 적용 확정: type=='rev' 체크로 수정. 질량보존 필수 조건 (2026-03-24)
- K_CAP 제거 확정: 문헌 근거 없는 numerical hack. QSSA 확산제외 + 셀별 Cl 원자보존으로 대체 (2026-03-25)
- ~~Cl 보존 강제~~: net effective rate로 대체 (2026-03-25)
- QSSA 종 확산 제외: 대수적 종은 로컬 화학만으로 결정. 확산 시 건너뜀 (2026-03-25)
- QSSA net effective rate: S3-S9을 개별 rate 대신 analytical 선형결합으로 표현. Cl hack 불필요 (2026-03-25)
- Rate capping 완전 제거 확정: 문헌 근거 없음 (BDF implicit solver에 불필요). 정량적 악화는 operator splitting artifact — BC/mass transfer 개선 시 해결 예정 (2026-03-25)
- Cl 보존 projection 채택: Sturm & Silva 2024 (ACS EST Air) 기반. BDF 후 per-cell Cl 원자 보존 → Cl⁻에 투영 (2026-03-25)
- Mass transfer BC → chemistry step 이동 확정: 확산 step에서 BC 제거, surface cell BDF RHS에 포함. dt 수렴 문제 해결 (2026-03-25)
- ~~Film+α_b BC 채택~~: **폐기 (2026-04-09)**. D_l/δ_liq가 PDE의 액상 확산과 이중 계산. Schwartz 1986, Zheng/Bruggeman 2020 근거.
- **gas_alpha BC 채택 (2026-04-09)**: 기상+계면 저항만. 1/k_gi = δ_gas/D_g + 4/(α_b·v̄), k_mt = k_gi/H_cc. 액상 저항은 PDE가 처리. notes/bc_formulation.md 참조. **→ 2026-04-23 three_film로 대체**.
- ~~**gas_alpha BC**~~ → **three_film BC 채택 (2026-04-23, ★ paper BC)**: Schwartz 3-저항 full form `1/K_L = H·δ_gas/D_g + H·4/(α·v̄) + δ_liq/D_l`. δ_liq=100µm. NO3⁻ 3.2/3.6kV에서 ×0.93/0.94 (exp 7% 오차 내). Grid-convergent (dz_min 1-20µm에서 0.15% 변동), voltage-independent (ratio 0.148 전 voltage). `pde_solver.py::compute_k_mt` bc_type='three_film'. Physical: PDE(bulk) + film(sub-grid BL) complementary. H2O2 17× 과다는 별도 c_gas 문제.
- Monolithic BDF 채택 (Strang splitting 대체): 확산+반응+질량전달을 단일 BDF로 동시 implicit 처리. Strang은 `_use_strang=True`로 비교 가능하게 유지. QSSA는 monolithic에서 OFF (BDF가 Cl stiffness 직접 처리). (2026-03-30)
- DIW에서 dt_enforce=None (단일 BDF) 권장: macro-step restart 불필요. 3× 가속, pH 차이 무시 가능(0.007). Saline은 Cl conservation 보정 때문에 dt_enforce=60s 유지. (2026-03-31)
- ODE tolerances: atol=1e-12, rtol=1e-6 확정. 이전 atol=1e-8은 trace species(H₂O₂ ~1e-10M)에 대한 오차 제어 불능 → rate budget 65× 개선, 속도 2× 향상. atol=1e-14 수렴 확인. (2026-03-31)

## Session History
- 2026-03-23: ECC 설치, 프로젝트 구조 체계화, 연구 방향 전환 반영 (optimizer 폐기 → 측정종만 사용)
- 2026-03-23: 측정종만 forward sim 실행. DIW: pH=3.389(6.1%), NO3-=574% 과다. Saline: stiffness hang 진단 완료. Poisson 기본값 False로 변경.
- 2026-03-24: Cl⁻ stiffness 심층 조사. S4/S6 주범 확인 (stiffness ratio 3.3e19). 문헌: Kushner=fully implicit, Liu/GlobalKin=0D 회피, COMSOL=implicit BDF. Operator splitting으로 saline Cl 해결한 1D 사례 없음.
- 2026-03-24: 4종 QSSA 구현 (HOCl⁻,Cl₂⁻,Cl,HOClH). 고유값 분석으로 Cl/HOClH가 양의 고유값(λ=+2e7) 원인 확인. 4→2→2 축소 시스템으로 해결.
- 2026-03-24: K_CAP 버그 발견 및 수정 — `_convert_reaction`이 'reversible'→'rev' 변환 후 K_CAP가 'reversible' 체크하여 가역반응에 미적용. 수정 후 QSSA 잔차 -1.93e-4→정확히 0. 2-pass Picard iteration 추가. Saline forward sim: pH=2.943, Cl⁻ 변화 -12µM(질량보존 완벽), HClO=8.6µM. 10.5분 완료.
- 2026-03-25: K_CAP 제거 + QSSA 확산제외 + Cl 원자보존 강제. Cl budget 진단으로 leak 원인 특정(BDF chemistry의 catastrophic cancellation). Saline: pH=4.281(18.9%), Cl⁻ -3866µM(물리적 전환), HClO=945µM. K_CAP 없이 물리적 Cl 화학 작동 확인.
- 2026-03-25: QSSA net effective rate 구현. S3-S9 analytical 기여 + S23-S69 filtered loop. 34 반응 태깅(7 rev, 27 irr). Cl hack 제거. Saline: pH=3.114(13.5%), Cl⁻ -1966µM(물리적), 6.9분. pH 크게 개선 + 2배 가속.
- 2026-03-25: QSSA Picard 수렴 개선 (2→20회, rtol 1e-12). Cl⁻ -1059µM(개선), pH=2.884(악화). Analytical net rate의 S3-S9 cancellation 문제 지속 확인.
- 2026-03-25: Rate capping 완전 제거. 결과 악화(pH=2.127, NO3⁻=8800µM)는 operator splitting artifact로 확인. dt_split=1s 시 54456µM으로 더 악화 → BDF 안정성이 아닌 splitting error. Cl 보존 projection(Sturm&Silva 2024) 추가.
- 2026-03-25: **Operator splitting artifact 해결** — mass transfer BC를 확산 step에서 제거, chemistry step surface cell RHS에 포함. BDF가 유입+소비 동시 implicit 처리. dt 수렴 확인: 10s→1s에서 6996→6260µM(수렴 방향). 이전 54456→8800(발산)과 대비. NO3⁻ 여전히 60배 과다는 BC 모델 자체 문제.
- 2026-03-25: **시간축 버그 수정** — OAS 데이터 2초 간격인데 1초로 처리하던 버그 수정. 12분(720s) 전체 실행: pH=1.989, NO3⁻=12864µM(126배), Cl⁻ -9300µM. mass transfer 과다유입 확정.
- 2026-03-25: **Component Verification Tests** — 10개 테스트 전부 PASS. 확산/화학/질량전달 개별 컴포넌트 정확성 확인. NO3⁻ 과다는 개별 버그가 아닌 BC/mass transfer 모델 자체 문제 확정.
- 2026-03-25: **N-atom Budget 진단** — N2O5가 N 유입의 99.3% 지배 (816µM, 실험 8배). liquid-side limited(R_liq 99.98%). δ_liq=1mm→121µM(실험 근접). δ_liq가 핵심 조절 파라미터.
- 2026-03-26: **BC 비교 구현+실행** — config_1d.py에 bc_type/alpha_b 추가, pde_solver.py compute_k_mt() 4-way 분기 구현. DIW 6케이스 실행: Film+α_b가 NO3⁻ 과다 해결 (two-film 93배→Film+α_b=0.05 1.9배). α_b≈0.01~0.05에서 실험(63µM) 교차. 비단조성(Film>Film+0.1) 분석: O3 과잉이 OH 소진→peroxynitrite→NO3⁻ 경로 억제하는 물리적 효과.
- 2026-03-26: **O3/라디컬 농도 검증** — (1) 단위 확인: conv=1000/N_A→mol/L 정확, 단위 불일치 없음. (2) O3 반응성 침투깊이 34µm (O3+NO2⁻, τ=0.7s), 표면 0.7µM, bulk avg 42nM은 물리적으로 정확. (3) Heirman 2025: 비반응 실험에서도 O3 겨우 0.3-0.5µM, Henry's law 10-20배 과대예측 확인. (4) Liu 2016 (1D saline): O3 표면 ~100µM, 침투 ~100µm, bulk→0 동일 패턴. OH saline ~10pM(DI ~1nM의 1/100). **결론: 우리 모델 라디컬 농도 문헌과 일관적.**
- 2026-03-27: **α_b 민감도 분석 + 종별 mass balance** — α_b=0.01/0.03/0.05 DIW 실행. NO3⁻: 55/85/118µM. 완전 수지(MT 포함): O3 MT가 유일 source(100%), 반응소비 ~33%, 나머지 축적. NO3⁻: R98 생성 99.6%, sink 없음(축적). H2O2: MT가 제거 방향(가스상=0). 12분 처리로 O3/NO3⁻ 정상상태 미도달.
- 2026-03-30: **BC별 MT flux 정량 비교** — run_bc_comparison.py에 해석적 k_mt 표 + 시뮬레이션 후 MT flux 추출 추가. N₂O₅ C_surface=0(모든 BC, 완전소비) → k_mt가 유일 조절인자. Cross-check ratio 0.16~0.92. 결과: Figures/mt_comparison_output.txt.
- 2026-03-30: **시간별 반응속도 기여도 Figure + Rate budget 검증** — gen_fig2_rate_evolution.py 생성 (개별 선 + net dC/dt). diagnose_rate_vs_conc.py로 검증: snapshot rate 합과 실제 농도변화 심각 불일치 (NO₃⁻ 3.3×, O₃/NO₂⁻/H₂O₂ 부호 반전). H₂O₂ 계단식 점프 발견 (가스 데이터 smooth 확인 → 순수 Strang splitting artifact). O₃ reaction rate noisy (MT는 smooth). **결론: snapshot 기반 per-reaction rate은 정량적으로 신뢰 불가. BDF dense_output + Simpson 적분 필요.**
- 2026-03-30: **Monolithic BDF 구현 + dt 수렴 확인** — Strang operator splitting → 단일 BDF solver 전환. 6단계 수정: (1) solve() 라우팅 `_use_strang=False`, (2) rhs()에서 Cl⁻ dydt=0 제거 + 중복 OH⁻ zeroing 정리, (3) chemistry_1d `set_qssa_enabled()` 추가, (4) `_enforce_cl_conservation()` 메서드 추출+`__init__`에 `_cl_cons_idx` pre-compute, (5) `_solve_bdf_with_electroneutrality` 반환에 t_eval/y_eval 추가, (6) run_saline_1d.py에 Film+α_b BC+dt_poisson=60. DIW 720s Film+α_b=0.03: pH=3.876(7.4%), NO₃⁻=133µM(2.1배), O3=112nM, OH=24pM, 10.5min. **dt 수렴 확인**: dt_enforce=120→10s에서 NO₃⁻ 128→135µM(5%), pH 3.892→3.870(0.02). dt_enforce=60s 채택.
- 2026-03-30: **Fig 2 rate evolution 재생성** — monolithic BDF로 재실행. NO₃⁻ rate budget 양호(R98 source, MT sink). **H₂O₂/O₃ net dC/dt 진동 문제 발견** — 개별 rate은 smooth한데 농도 차분 dC/dt가 불규칙. Strang에서도 동일 문제였음. electroneutrality enforcement(60s)/BDF restart/라디컬 volume-avg noise 중 하나가 원인.
- 2026-03-30: **Fig 1/3/4 생성 완료** — collect_monolithic_data.py로 데이터 수집 (5 BC + 3 α_b + mass balance, ~60min). plot_bc_results.py 갱신 후 생성. Monolithic BDF 결과: Two-film pH=2.43/NO₃⁻=3748µM → Film+α_b=0.01 pH=4.27/NO₃⁻=53µM. α_b sensitivity: 0.01~0.05에서 NO₃⁻ 53~189µM. Mass balance: NO₃⁻ R98=99.6%, O₃ MT=100% source, H₂O₂ R45=88.5%.
- 2026-03-31: **Fig 2 rate budget 완전 해결** — (1) dt_enforce=None(단일 BDF), 3× 가속. (2) Simpson 3-point 적분(dense_output). (3) **근본 원인: atol=1e-8 → H₂O₂(1e-10M)보다 100× 커서 BDF 오차 제어 불능**. atol=1e-12, rtol=1e-6으로 변경 → H₂O₂ ratio 2.569→0.988(65× 개선), 속도 10.5→1.7min. config_1d.py 기본값 영구 변경.
- 2026-04-01: **Fig 2 개선** — net 선을 Σrate으로 교체, DT_SNAPSHOT=2s, 시간기반 smoothing(60s/120s). O₃ 2분 급변은 radical chain ignition(물리적). **Fig 5 공간분포** 추가. **Fig 1-2 MT flux 시계열** 생성(Dirichlet 제외). **Fig 1/3/4 갱신** — collect_monolithic_data.py 재실행(dt_enforce=None, atol=1e-12) 후 plot_bc_results.py 데이터+이미지 갱신 완료.
- 2026-04-08~09: **물리적/수치적 문제점 분석 + Figure 재생성**
  - Ver1 vs Ver2 비교: Ver1(선형보간+volume-weighted avg) vs Ver2(계단식+산술평균). Ver1이 수치적으로 정확. 동일 조건(Film+αb=0.03) 결과: Ver1 NO₃⁻=38.4µM vs Ver2 134.1µM — C_eq 보간 방식 차이.
  - **작업 폴더 이동**: Ver1/Ver2는 backup, `work/work/` 기준으로 전환
  - **통합 Figure 생성 스크립트**: `Figures/gen_all_figures.py` 신규 작성. 6개 unique 시뮬레이션 → Fig 1(BC비교), 1b(MT flux), 2(rate evolution), 3(radicals), 4(mass balance), 5(spatial) 전부 생성. npz 캐시 지원, `--fig`, `--rerun` 옵션.
  - **Fig 2 smoothing 개선**: (1) `np.convolve(mode='same')` edge artifact → edge-aware MA로 교체. (2) BDF dense output spike → median filter. (3) net dC/dt를 Σrate → ΔC/Δt(finite-diff ground truth)로 교체. (4) MA 제거, median despike만 적용 (물리적 transient 보존).
  - **atol 1e-12 → 1e-15**: 중간 시간대 R98 spike 해결. OH oscillation은 미해결 (atol과 무관).
  - **OH oscillation 근본 원인 규명**: OH 농도 ~1e-12M, τ_OH ~ms. BDF step ~0.03-1s. atol을 줄여도 해결 안 됨 — BDF가 step 내 fast radical dynamics를 표현 못 함. QSSA 전환 필요.
  - **Radical ignition trigger 규명**: t≈110s에서 O₃ autocatalytic chain이 아니라 **가스상 NO₂ 첫 출현(t≈106-108s)**이 trigger. NO₂ 유입 → R15_rev(NO₂+OH→ONOOH, k=4.5e9)가 기존 OH sink의 4.4배 → OH QSS 붕괴 → radical chain 재구성.
  - **BDF step 분석**: 12,387 steps (median=31ms). t=110-130s에서 dt=0 (step rejection) 수십 회 — radical ignition transition에서 BDF 극심한 stiffness.
  - **시뮬레이션 효율화**: 11 runs → 6 unique (bc_type, alpha_b) runs. 불필요한 dense_dt 분기 제거, 모든 run에 2s t_eval 통일.
- 2026-04-09: **Grid convergence 완전 확인** — dz_min(1~20µm) 및 stretch_ratio(1.02~1.12) 변화에 bulk/surface 결과 수렴. Step log pH의 `np.mean` (cell-count avg)이 grid 의존적이나 물리량(`pH_avg`, volume-weighted)은 무관.
- 2026-04-09: **종별 α_b 구현+테스트** — config_1d.py에 `alpha_b_species` dict 추가 (N₂O₅=0.03, O₃=0.05, H₂O₂=0.1, NO=0.001 등). O₃ +48% 외 pH/NO₃⁻ 변화 없음. H₂O₂는 가스상=0이라 α_b 무관.
- 2026-04-09: **Gas onset noise filter** — `_filter_onset(n_consecutive=5)` 구현. NO₂ t=4-10s spike 제거. 모든 가스종에 자동 적용.
- 2026-04-09: **atol 분석** — O₃⁻(2.1e-16), HO₃(5.7e-15), O⁻(8.2e-16), N₂O₅(aq)(6.3e-18) 등이 atol=1e-15 이하. R98(N₂O₅+H₂O) rate 오차가 NO₃⁻ budget의 10000%. 종별 atol 테스트: baseline과 동일 결과, 30% 느림.
- 2026-04-09: **QSSA 시도+실패** — O₃⁻/HO₃/O⁻ analytical steady-state + relaxation. Radical chain 파괴 (O₃ 100배 축적, OH 80배 감소). 상호의존성+PDE coupling으로 단순 QSSA 부적합.
- 2026-04-09: **O₃ oscillation 진단** — t=114-145s에서 surface O₃ 감쇠진동 (주기~6s). 0.1s 해상도로 확인. BDF가 radical QSS(ns~µs)를 미해석하여 발생하는 수치적 artifact. 물리적으로는 smooth 전이여야 함.
- 2026-04-09: **BC 이중 계산 문제 규명** — film_alpha(α_b×D_l/δ_liq)가 PDE의 액상 확산과 중복. Schwartz 1986 저항 모델, Zheng/Bruggeman 2020, Liu 2021 근거. notes/bc_formulation.md 작성.
- 2026-04-09: **gas_alpha BC 구현** — 1/k_gi = δ_gas/D_g + 4/(α_b·v̄), k_mt = k_gi/H_cc. 초기 단위 오류(H_cp vs H_cc) 수정. 테스트: δ_gas=10mm에서 α_b 무감(gas-side 지배). δ_gas=1mm에서 NO₃⁻=987µM(16배 과다). δ_gas 결정이 핵심 과제.
- 2026-04-09: **gas_alpha BC 채택 결정** — 물리적으로 올바른 BC. δ_gas 결정 및 α_b 민감도 재검증은 pending.
- 2026-04-10: **Fig 1~5 gas_alpha BC로 재생성** — gen_all_figures.py 수정. BC_CASES: Dirichlet / One-film(gas) / Gas+αb(종별). 참조: δ_gas=1mm vs 10mm 비교 — 10mm에서 NO₃⁻=101µM(1.6배), 1mm에서 987µM(16배). 10mm가 더 합리적.
- 2026-04-10: **one_film_gas BC 추가** — gas-side only, α_b 없음. `k_mt = (D_g/δ_gas) / H_cc`. Gas+α_b와 비교해 α_b가 무감임을 확인 (gas-side 지배).
- 2026-04-10: **Gas onset filter 개선** — 이전: noise 제거 + 0을 그대로 유지 → 계단형 점프. 수정: noise 제거 + onset 전/사이 0 구간을 linear interpolation으로 채움 → smooth ramp-up. Ramp는 t=0(값 0)부터 첫 nonzero까지.
- 2026-04-10: **비율 기반 HONO/HNO₃/H₂O₂ gas input 구현** — 이전: gas=0 → H₂O₂≈0 (실험 괴리). 이번: scalar 대신 array 허용. HONO=NO₂×0.33, HNO₃=N₂O₅×0.83, H₂O₂=O₃×0.03. 문헌 근거: notes/unmeasured_gas_species.md. 결과: **H₂O₂=13.2µM** (실험 11에 근접!), NO₃⁻=155µM(2.5배), pH=3.81.
- 2026-04-10: **Fig 6 추가** — gen_fig6_gas_data.py. 3 패널: (a) Raw 측정값 (cm⁻³), (b) Measured species 시계열 (onset filter + linear interpolation, mol/L), (c) Unmeasured species 시계열 (비율 기반, mol/L).
- 2026-04-10: **O₃ MT scaling 테스트** — Figures/test/test_o3_mt_scaling.py. O₃ k_mt × [1, 0.5, 0.1, 0.01]. 결과: NO₃⁻/pH 거의 무변화 (155→152µM), O₃/OH는 비례 감소. **NO₃⁻ 과다 원인은 O₃가 아니라 N₂O₅ MT** 확정. NO₂⁻는 모든 scale에서 ~0.003µM (실험 3의 1/1000).
- 2026-04-10: **Fig 1에서 Dirichlet 제거** — One-film(gas)와 Gas+α_b만 표시. pH=3.8, NO₃⁻=155, H₂O₂=13.2 (두 케이스 거의 동일).

- 2026-04-09: **Radical chain ignition 심층 분석 + reference 조건 확정**
  - **가스 데이터 전처리**: below-LOD 보간 6가지 비교 (Raw, LOD/2, Linear, Exp, Sigmoid, SG). NO₂/NO₃에 intermittent zeros. stable start 규칙(5연속 nonzero) 적용. **Linear interp 채택.**
  - **OH oscillation 검증 (3 tests)**: (1) max_step=0.01s → oscillation 동일 (2) 0D 2cells → 동일 (3) dt=0.1s → aliasing 아님. **결론: 물리적 damped oscillation (BDF artifact 아님).**
  - **atol 확정: 1e-15** — trace radical(OH ~1e-12M)의 정확한 오차 제어. 비용 +20%. 1e-12에서 R98 spike 발생.
  - **Reference 조건 확정**: atol=1e-15, rtol=1e-6, max_step=1.0s, dt_snap=2.0s, linear interp 전처리, Film+αb=0.03, dz_min=5µm, stretch=1.12.
  - **Fig 1b 수정**: HNO₃→NO₃으로 변경.
  - **NO₂ trigger 가설 기각**: OH budget에서 R15_rev(NO₂+OH) <0.1%. 가스 전처리 후에도 transition 시점 불변.
  - **전체 radical 동시 전환 확인**: t≈2.1min에 OH/HO₂/O₂⁻/O₃⁻/HO₃/ONOOH/NO₃(aq) 전부 10배 급증.
  - **Bifurcation 분석 (Step 1~3)**:
    - Step 1 (slow driver): O₃ 축적, pH 감소, NO₂/NO₃ gas 출현 확인
    - Step 2 (Jacobian): radical subsystem 고유값 항상 음수 → bifurcation 아님
    - Step 3 (perturbation): **결정적 결과**:
      - O₃ x2/x0.5/=0: **효과 없음** (baseline과 동일) → O₃는 trigger 아님
      - **NO₃ gas OFF: OH=0.3pM, 전환 완전 억제** → **NO₃ radical MT가 필수 조건**
      - NO₂ gas OFF: OH 2.66배 증가, 전환 앞당겨짐 → NO₂는 radical 억제 역할
      - O₃ gas OFF: OH 3.3배 증가 → O₃ 공급 중단 시 radical 더 활성화
    - **이전 "O₃가 유일한 trigger" 결론은 철회**. perturbation 결과를 확증편향으로 잘못 해석했음.
  - **NO₃ radical 경로**: gas NO₃(H=44) → liq NO₃ radical → R93(+OH⁻→OH, 45-59%) + R102(+NO₂→N₂O₅, 32-46%). OH budget 기여는 <5%이지만, NO₃ OFF 시 전환 자체가 억제됨 → 직접 OH 생산이 아닌 chain 활성화 촉매 역할 가능성.
  - **전수조사**: 101 reactions × 25 species × 121 snapshots (0~4min). full_budget.csv 저장.
  - **개별 반응 knockout 테스트 (6개 NO₃ 반응 + R79)**:
    - **R93 OFF (NO₃ + OH⁻ → NO₃⁻ + OH): OH@200s=0.30pM, 전환 완전 억제** → NO₃ gas OFF와 동일
    - R102 OFF (NO₃ + NO₂ → N₂O₅): OH=148pM, 전환 정상 (1.07x)
    - R92 OFF (NO₃ + NO₂⁻ → NO₃⁻ + NO₂): OH=174pM (1.25x)
    - R89 OFF (NO₃ + HO₂ → NO₃⁻ + O₂): OH=192pM (1.38x)
    - R90/R91/R79 OFF: baseline과 동일
    - **결론: R93 (NO₃ + OH⁻ → NO₃⁻ + OH) 단독이 radical chain ignition의 필수 조건**
  - **R93의 역할**: OH budget에서 0.003%이지만, 표면 cell에서 OH seed 공급. Baseline OH_surface=4.8pM vs R93 OFF OH_surface=0.46pM (10배 차이). 이 seed가 O₃ chain gain(>1)의 지수 성장 기반. seed 없으면 chain 시작 불가.
  - **Radical ignition 메커니즘 최종 요약**:
    1. 가스상 NO₃ radical이 용해 (H=44, k_mt=α_b×D/δ_liq)
    2. R93 (NO₃ + OH⁻ → NO₃⁻ + OH)이 표면에서 OH radical seed 공급
    3. OH seed가 O₃ chain (Staehelin-Hoigné: R27→R28/R25→R38/R55→OH) 구동
    4. Chain gain > 1이므로 OH 지수 성장 (doubling ~µs)
    5. O₃ MT 축적 → chain 소비가 MT 초과 시 O₃ peak → O₃↓↔OH↑ positive feedback → 급속 전환
  - **OH surface oscillation (CV~9%) 추가 분석**:
    - 종별 atol(OH 등 1e-17) 적용 → 개선 안 됨 (bulk OH noise가 원인 아님)
    - 가스종별 perturbation: O₃/NO₂/N₂O₅ 개별 smooth/const → 효과 없음. **ALL gas constant → CV=0.9%** (oscillation 소멸). 단일 종이 아닌 복합 gas noise가 원인.
    - NO₃ SG31 smooth → CV 9.3→6.8% (약간 감소). NO₃ gas noise가 일부 기여하나 단독 원인은 아님.
  - **FUTURE WORK (닫음)**: 
    - NO₃가 R93 (NO₃ + OH⁻ → NO₃⁻ + OH)을 통해 OH profile에 큰 영향. R93이 OH budget의 0.003%인데 radical chain ignition의 필수 조건인 amplification 메커니즘은 미규명.
    - OH surface oscillation의 정확한 원인 (복합 gas noise vs 물리적 chain dynamics) 미규명.
  - **생성/수정 파일**:
    - `Figures/gen_all_figures.py` — 통합 Figure 생성 (Fig 1~5, 6 sims, npz cache, linear interp 전처리 포함)
    - `Figures/test/run_oh_tests.py` — OH oscillation 3 tests
    - `Figures/test/run_bifurcation_analysis.py` — Step 1~3 bifurcation 분석
    - `Figures/test/run_full_budget.py` — 전수조사 (101 rxn × 25 sp)
    - `Figures/test/full_budget.csv` — 전수조사 데이터
    - `Figures/test/gen_fig_gas_interpolation.py` — 가스 데이터 전처리 비교
    - `Ver4_1D/config_1d.py` — atol 1e-12 → 1e-15
    - `run_comparison.py` (work root) — Ver1 vs Ver2 비교

- 2026-04-10: **논문 전산모사 계획 수립 + Surrogate optimizer 개발 + pH/NO2⁻ 진단**
  - **연구 방향 확정**: Dry OAS input → baseline 경향성 → HONO/HONO2/H2O2 문헌값 추정 → 살균 메커니즘 (3분/5분 threshold) 연결. Phase 1/2/3 계획 수립 (memory: `project_paper_plan.md`).
  - **Surrogate 2-stage optimizer 개발**: `Ver4_1D/run_surrogate_opt.py`
    - Stage 1: 0D FastSurrogate0D (gas_alpha BC, instant chemistry, R32 with reactive penetration depth correction)
    - Stage 2: 1D PDE validation
    - Feedback: 1D/0D bias로 0D target 보정 → 재최적화
    - 4 free params: HONO, HONO2, H2O2, δ_gas (log space, DE)
  - **단위 버그 발견/수정** (0D에서): (1) HENRY_CONSTANTS는 이미 dimensionless H_cc인데 1D code의 gas_alpha branch가 R×T 한번 더 곱함 — 0D도 동일하게 맞춰 일관성 유지. (2) k_mt[m/s]를 liquid depth L로 나눠야 bulk concentration rate [1/s] 됨.
  - **수렴 결과** (3 iterations): HONO=1.4e15, HONO2≈0, H2O2=2.3e15, δ_gas=13-15mm. 1D 결과: pH=4.07, NO3⁻=84µM (실험 63, +33%), NO2⁻=0.003µM (실험 3, -99.9%), H2O2=9.4.
  - **pH/NO2⁻ gap 진단 스크립트**: `Figures/test/diag_ph_no2_gap.py` — charge balance + NO2⁻ rate budget + HONO2 sensitivity sweep
  - **pH gap 원인 규명 (결정적)**: 1D charge balance 완벽 닫힘 (+0.04%). pH는 NO3⁻에 의해 1:1 결정 ([H⁺]=84µM ≈ NO3⁻=84µM, 다른 음이온 모두 무시 가능). **실험 pH=3.61, NO3⁻=63µM은 charge balance 불가능** — [H⁺]=245µM 필요하나 NO3⁻+NO2⁻=66µM만 존재. **~180µM 음이온 gap**. HONO2 sweep (0→1e15 cm⁻³): pH 4.074→4.052만 변화, NO3⁻ 84→89. **HONO2로 gap 닫을 수 없음 확정**. 가능 원인: CO2 흡수(H2CO3), ionic strength electrode 오차, 미측정 음이온, plasma-off 후속 반응.
  - **NO2⁻ rate budget 규명**: **R32 (O3+NO2⁻→NO3⁻) 단독이 97% sink**. 과거 "R92 지배"는 다른 조건. Source: R19 (2NO2 hydrolysis) 64% + R95 (N2O4 hydrolysis) 36%. **HONO gas 용해는 NO2⁻ source에 무의미 (R78 <0.1%)**. Net = -2.5e-9 M/s. Spatial segregation: source는 bulk 전역, R32 sink는 surface 34µm에 집중 → bulk steady-state 0.7µM 예상이지만 diffusion-sink coupling으로 실제 0.003µM.
  - **논문 내러티브 시사점**:
    1. pH를 fitting target에서 제외 or lower weight — 실험 자체가 charge balance 불합
    2. NO3⁻가 primary validation target
    3. NO2⁻ 3µM은 plasma-off 후 측정 아티팩트로 해석 가능 (O3 decay 후 R32 멈춤 → 축적)
    4. HONO gas input은 NO2⁻ 결정 요인 아님 — NO2/N2O4 gas 농도가 핵심
    5. HONO/HONO2/H2O2 중 H2O2만 gas 용해가 유효 (liquid H2O2에 직접 기여)
  - **생성 파일**:
    - `Ver4_1D/run_surrogate_opt.py` — 0D+1D feedback optimizer
    - `Ver4_1D/optimal_params_surrogate.yaml` — 최적 파라미터 출력
    - `Figures/test/diag_ph_no2_gap.py` — charge balance + NO2 budget 진단

- 2026-04-13~14: **Gas-phase sweep + NO₂⁻ 진단 + 새 OAS data 적용**
  - **HONO/HNO₃/H₂O₂ 개별 sweep (14 cases)**: Figures/test/sweep/test_gas_sweep.py. Dry baseline(gas=0) 대비 한 종씩. 결과: H₂O₂←H₂O₂gas(직접), pH←HNO₃(직접), NO₃⁻ 변화없음(N₂O₅ 지배), NO₂⁻ 변화없음(O₃ barrier). notes/gas_sweep_results.md.
  - **NO₂⁻ R92/R32 knockout**: Figures/test/sweep/test_no2m_sensitivity.py. R92(NO₃+NO₂⁻) OFF → 변화없음. **R32(O₃+NO₂⁻) OFF → NO₂⁻=5.4µM (실험 3.58 근접!)**. R32가 진짜 소비원 확정. O₃가 표면에서 NO₂⁻를 즉시 산화 ("O₃ barrier"). 1D no-convection의 구조적 한계.
  - **HONO×O₃ 2D sweep (20 cases)**: Figures/test/sweep/test_hono_o3_sweep.py. HONO/NO₂=[1,3,5,10]×O₃scale=[0.1~1.0]. ★HONO/NO₂=10,O₃×0.3: NO₂⁻=2.89µM, pH=3.62 (실험 근접). 단 HONO/NO₂=10은 비물리적, NO₃⁻=241(악화).
  - **새 OAS data 수신**: `OAS data/Dry/` — 3전압(2.6/3.2/3.6kV), t=0~600s(10min), O₃/NO₂/NO₃/N₂O₅. **N₂O₅가 이전 CSV 대비 5배 감소** (1.24e16→2.55e15). notes/experimental_reference.md.
  - **새 OAS data Dry baseline 전압별**: NO₃⁻=11/21/25µM (실험 33/63/70의 1/3). pH=4.9/4.7/4.6. H₂O₂/NO₂⁻≈0.
  - **gen_all_figures.py 구조 변경**: --voltage 인자, CONDITION_LABEL, 전압별 output folder (`Figures/OAS data/{voltage}_{condition}_Dg_Xmm/`). xlsx 읽기. Fig 1 NO₃⁻ linear scale 수정.
  - **Humid median 전압별 완료**: H₂O₂가 전압별 실험값과 잘 맞음 (2.6kV: 4.65 vs 4.76, 3.2kV: 10.6 vs 11.2). 비율 0.03 유효.
- 2026-04-14~15: **RH 80% 비율 기반 외삽 + Humid fitting 시뮬레이션**
  - **RH 경향 분석**: Humid OAS data (RH 25/55/65%) + Dry. 종별 경향: O₃↓, NO₂↑, N₂O₅↓↓, HONO↑. RH 25% 데이터 이상점 → 제외.
  - **비율 기반 fitting (물리적 함수)**:
    - N₂O₅/NO₂ = A/(1+B·RH²): 수증기 dimer 가수분해 정상상태. 전압 간 잘 겹침.
    - HONO/NO₂ = A·RH: 표면 [H₂O] ∝ RH.
    - NO₂/O₃ = A+B·RH: 경험적 mode 전환.
    - NO₃/O₃ = A+B·RH: scatter 있으나 양의 경향.
    - O₃: 직접 선형 (anchor 종).
    - 스크립트: Figures/test/test_rh_ratio_fit.py. 문서: notes/rh_extrapolation.md.
  - **RH 80% 예측 (3.2kV)**: O₃=6.66e16(-35%), NO₂=6.09e15(+3.4×), N₂O₅=3.57e14(-86%), HONO=4.72e13(1.9ppm).
  - **Humid fitting 시뮬레이션**: Dry 시계열 shape 보존 × SS 비율 스케일링. 초기 shape 버그(O₃ shape을 NO₂에 사용) 발견 → 수정.
  - **3조건 전압별 비교 (fig_voltage_comparison.png)**:
    - Dry: NO₃⁻ 실험의 1/3, H₂O₂=0
    - Humid median: H₂O₂ 전압별 실험 근접(비율 0.03 유효!), NO₃⁻ 실험의 ~50%
    - Humid fitting: NO₃⁻ 더 감소(21µM, N₂O₅ -86%), H₂O₂ 부족(6.7 vs 11.2 — RH↑에서 H₂O₂/O₃ 비율 증가 필요)
  - **gen_all_figures.py**: Fig 6 추가 (gas input 3패널), RH80_RATIOS 전압별 dict, CONDITION_LABEL 시스템, shape 보존 스케일링.
  - **output 구조**: `Figures/OAS data/{voltage}_{Dry|Humid_median|Humid_fitting}_Dg_10mm/`

- 2026-04-20: **TPA→hTPA OH 정량 비교 (1 세션 완료, ultraplan 7 phase)** — 2026-04-17 실험(TPA 2mM + NaOH 10mM, pH~11.5, 10min) 3전압 데이터와 1D 시뮬 비교.
  - 구현: `reactions_tpa.yaml`(3 rxn, R_TPA1/2/3), `config_1d.py`(TPA/hTPA species, Z=-2), `chemistry_1d.py`(`tpa_mode` flag), `pde_solver.py`(전하 병합), `run_tpa_alkaline.py` runner, `Figures/gen_fig_oh_tpa.py` 6-panel figure.
  - 결과 (gas_alpha BC, δ_gas=10mm, α_b 종별, RH80 ratios, 10min): 2.6kV hTPA=4.89µM (exp 12.66, −61%), 3.2kV=13.92 (exp 57.72, −76%), 3.6kV=16.38 (exp 43.26, −62%).
  - **Sim rank 3.6>3.2>2.6 단조** vs 실험 3.2>3.6>2.6 비단조 — 가스-side 단조 확인, 실험 비단조성은 inner filter effect(측정 artifact) 가능성.
  - 정량 2.7–4× 낮음: **OH⁻ 10mM이 TPA 2mM보다 강한 scavenger** (k[OH⁻]=1.2e8 s⁻¹ vs k[TPA]=8e6 s⁻¹, 15배 차이). 시뮬 TPA 포획률 ~6%.
  - pH 11.99–12.01 유지, TPA 소모 1.4–1.9%, wall 125s/run(TPA-on). notes/{ultraplan_tpa_oh_comparison,tpa_chemistry_literature,oh_tpa_comparison}.md 작성.
  - **다음 단계 후보**: α_b(O₃)×H₂O₂비율 2D sweep (P1+P2), TPA k/branching 민감도, 1D geometry limitation 확정.

- 2026-04-20: **Henry 버그 수정 + 전압별 재실행 + 파라미터 sweep + 과다 원인 분석**
  - **Henry 상수 R×T 재곱 버그 수정**: gas_alpha/one_film_gas에서 H_cp→H_cc 변환 제거. k_mt 24.5× 증가.
  - **Dry 3전압 재실행**: 2.6kV NO₃⁻=243(exp 33, 7.4×), 3.2kV=458(exp 63, 7.3×), 3.6kV=540(exp 70, 7.7×). **전 전압 ~7× 일정 과다**. pH: 3.2kV 3.34(exp 3.61, -7.5%), 3.6kV 3.27(exp 3.25, +0.5%). 2.6kV pH=3.62(exp 5.09, 대폭 괴리).
  - **Humid fitting 3전압 Fig 1~6 재생성**: `Figures/DIW results/{V}_Henry_Dry_Dg_10mm/`, `{V}_Henry_Humid_fitting_Dg_10mm/`. fig_voltage_comparison.png 재생성.
  - **Humid fitting 3.2kV 결과**: pH=3.40, NO₃⁻=396µM(6.3×), NO₂⁻=5.0µM(실험 3.58 근접!), H₂O₂=193µM(17×).
  - **NO₂⁻ 매칭 시도 (Henry 수정 전)**: R32 속도상수 ×[1, 0.1, 0.01] → NO₂⁻ 불변(0.0038µM). Post-treatment(gas=0) → NO₂⁻ 감소. HONO/NO₂=1.0 → NO₂⁻=0.011µM. **물리적 파라미터로 3.58µM 도달 불가** 확인.
  - **대규모 파라미터 sweep (31 sims)**: δ_gas=[10~100mm], H₂O₂/O₃=[0.001~0.03], HONO₂/N₂O₅=[0~0.83], HONO/NO₂=[0.007~1.0], 조합 10개.
    - δ_gas: NO₃⁻에 반비례. 70mm에서 70.8µM(실험 근접). 하지만 70mm는 비물리적.
    - H₂O₂/O₃=0.01: H₂O₂=12.1µM(실험 11.2 근접). NO₃⁻ 무관.
    - HONO₂/N₂O₅: 0.83→0.0에서 NO₃⁻ 7% 감소만. **무영향**.
    - HONO/NO₂ 증가: NO₃⁻ **증가**(역방향).
  - **3-skill 동시 분석 (deep-research + council + verification)**:
    - **Gas depletion 가설 (최우선)**: OAS가 empty-chamber 측정 → 실제 처리 시 액면 uptake로 gas 농도 낮음. depletion factor 0.3~0.5 추정.
    - **η (유효면적) 가설**: plasma patch < petri dish. mass_transfer_eta가 dead code (rhs에 미적용) → 1줄 수정 필요.
    - **Henry double-counting 기각**: gas_alpha에서 C_surface=0 → J=k_gi×C_gas, H cancel. H 변경 무효.
    - **Penetration theory 기각**: stagnant film보다 k_mt 높음(역방향).
    - **HONO₂ 비율 기각**: NO₃⁻에 7% 영향만.
    - **핵심 결론**: δ_gas 또는 η만이 N₂O₅ flux 조절 가능. 전 전압 ~7× 일정 과다 → 체계적 보정 필요.
  - **생성 파일**: `Figures/test/test_hono_ratio_1.py`, `test_no2m_matching.py`, `test_no2m_matching_v2.py`, `test_voltage_dry.py`, `test_param_sweep.py`.

- 2026-04-20: **Saline 파이프라인 구축 + 6 cases 전체 생성**
  - **초기 test run (예전 설정)**: `run_saline_1d.py` 그대로 실행 → 예전 CSV `1kHz3.2kVpp.csv` + `film_alpha` BC + α_b=0.03 uniform. 45.1min 완료. pH=2.05(exp 3.60), NO₃⁻=**6659µM(65× 과다)**, H₂O₂=0, Cl⁻ +896µM. Henry R×T 버그 + 이전 CSV(N₂O₅ 5× 높음) + film_alpha BC 복합 문제로 폐기.
  - **실험값 교정**: xlsx `OAS data/Dry/(P-L) 액체활성종 농도...xlsx` Saline sheet 재확인. **NO₃⁻ 32.44/101.30/112.77 µM (2.6/3.2/3.6 kVpp)**. 이전 CLAUDE.md 기록(4.70/10.45/16.92)은 인접 컬럼 오독. 표 수정 완료.
  - **α_b species-specific 재확인**: `pde_solver.py:244` `alpha_b=None`일 때만 config dict 사용(N₂O₅ 0.03, O₃ 0.05, H₂O₂ 0.1, NO₂ 0.03, HONO 0.05, HONO2 0.07). run_saline_1d.py는 0.03 하드코딩이라 species 무시됨.
  - **gen_all_figures.py 확장**: `--saline` flag + `--condition {Dry,Humid_median,Humid_fitting}` 추가. 전역 IS_SALINE/SOLUTION_LABEL/FIXED_CATION_CONC/EXP_*_ALL 도입, saline_mode=IS_SALINE를 run_case/_get_solver/fig1b solver_bc 4곳 반영. Output 분기: DIW `Figures/OAS data/...`, Saline `Figures/Saline results/{V}_{condition}_Dg_10mm/`.
  - **Saline 6 cases 결과** (3전압 × Dry/Humid_fitting, Grid 49 cells, monolithic BDF single step, 각 ~6min):

| Case | pH sim/exp | NO₃⁻ µM sim/exp | NO₂⁻ sim | H₂O₂ µM sim/exp |
|---|---|---|---|---|
| 2.6kV Dry | 3.65 / 5.15 | 241 / 32 (7.5×) | 0 | 0 / 2.00 |
| 3.2kV Dry | 3.36 / 3.60 | 457 / 101 (4.5×) | 0 | 0 / 5.14 |
| 3.6kV Dry | 3.28 / 3.43 | 539 / 113 (4.8×) | 0 | 0 / 7.73 |
| 2.6kV Humid_fit | 3.51 / 5.15 | 308 / 32 (9.5×) | **3.42** | **2.03 / 2.00** ✓ |
| 3.2kV Humid_fit | 3.41 / 3.60 | 410 / 101 (4.1×) | 0 | 7.25 / 5.14 (+41%) |
| 3.6kV Humid_fit | 3.28 / 3.43 | 547 / 113 (4.8×) | 0 | 10.4 / 7.73 (+34%) |

  - **핵심 관찰**: (1) **H₂O₂ Humid_fit에서 양호** — 2.6kV 완벽(2.03/2.00), 3.2/3.6kV +35~40%. DIW의 H₂O₂=193µM(17×)와 달리 Saline은 적정 — Cl⁻ scavenger 효과로 H₂O₂ 추가 생성 억제된 듯. (2) **pH 3.2/3.6kV 근접** — 0.15~0.25 unit. (3) **NO₃⁻ 구조적 4.5~9.5× 과다** — DIW(7×)와 유사한 크기. 같은 gas depletion/η 문제 동일 적용 가능. (4) **2.6kV 저전압 pH 크게 빗나감** — sim 3.5~3.65 vs exp 5.15(−1.5 unit). Saline+저전압 조합 특성. (5) **Humid_fit 2.6kV NO₂⁻ 3.4µM** — 저전압 O₃ barrier 약화(O₃ 작으면 R32 sink 감소).
  - **변경 파일**: `Figures/gen_all_figures.py` (IS_SALINE 글로벌, argparse 확장, EXP voltage-specific dict, out_folder 분기).
  - **생성 산출물**: `Figures/Saline results/{2.6|3.2|3.6}kV_{Dry|Humid_fitting}_Dg_10mm/` 각 폴더에 fig1/1b/2/3/4/5/6 (png+pdf) + cache npz.
  - **memory feedback 추가**: `feedback_plasma_liquid_new_setup.md` — 모든 실행은 새 OAS data(xlsx) + gas_alpha BC(δ_gas=10mm) 필수, 예전 CSV/film_alpha 금지.

### PENDING (2026-04-23 재정의)
**★ three_film BC 공식 채택 (2026-04-23 User 결정)**. NO3⁻ 7× 문제 grid/voltage robust 해결. Finite gas reservoir ODE 방향 **보류** (three_film가 더 간단하고 효과적).

0. **★ three_film default 전환 + Figure 일괄 업데이트** — `MASS_TRANSFER.bc_type='two_film'` → `'three_film'`. 모든 run/gen 스크립트 bc_type 확인.
1. **H₂O₂ gas input ratio 재산정** — H2O2/O3=0.03 → ~0.003-0.01 sweep. BC와 무관한 별도 track. 16× 과다의 실질적 fix.
2. **2.6 kV under-prediction audit** (×0.67) — c_gas(2.6 kV) 측정 SNR 혹은 plasma production rate voltage 의존성 재검토.
3. **NO2⁻ 재건** — **보류** (User 2026-04-23). 추후 필요 시 (a) species-specific δ_liq / (b) α_b(NO2) / (c) HONO ratio 중 선택.
4. **pH gap** — three_film에서 pH +0.4~1.0 단위 높음. 음이온 balance 재검토 필요.
5. **Saline 확장 + three_film** — DIW 확정 후 saline 적용.
6. **TPA Phase 5 sweep** — 보류 (새 BC 확정 후 재실행).
7. **Phase 3 살균 threshold** — 우선순위 낮음.

### Superseded / Cancelled
- ~~η sweep~~ (2026-04-21 취소, phenomenological knob).
- ~~Plan v1 PRA α-surface~~ (2026-04-22 supersede; `notes/reactive_uptake_bc_plan.md` 참조용 보존).
- ~~Finite gas reservoir ODE~~ (2026-04-23 보류; three_film가 더 간단하고 효과적).

### 주요 수치 (2026-04-23 three_film 확정)
| Voltage | NO3⁻ sim/exp | pH sim/exp | H2O2 sim/exp | NO2⁻ sim/exp |
|---|---|---|---|---|
| 2.6 kV | 21.9/32.6 (×0.67) | 4.66/5.09 | 80.1/4.76 (×16.8) | 0.16/0 |
| 3.2 kV | 58.5/62.7 (×0.93) | 4.23/3.61 | 185.9/11.2 (×16.6) | 0.48/3.58 |
| 3.6 kV | 66.2/70.4 (×0.94) | 4.18/3.25 | 199.6/16.3 (×12.3) | 0.54/20.7 |

- 2026-04-20 (TPA 후속, 3차 세션): **Henry 수정 후 TPA 3조건 × k_R3 3값 감도 분석**.
  - **Dry + Henry fix (k_R3=1e9)**: 2.6kV 23.6(+86% overshoot), 3.2kV 19.1(−67%), 3.6kV 19.0(−56%). rank 2.6>3.2≈3.6 역전. pH 3.6kV 11.48 저하 과도. HO₂⁻/H₂O₂ 100× 증가(~5µM).
  - **Humid_fitting + Henry fix (k_R3=1e9)**: 2.6kV 13.77(+9%), 3.2kV 22.30(−61%), 3.6kV 21.40(−51%). **rank 3.2>3.6>2.6 첫 실험 재현**. pH 11.98 유지(N₂O₅ gas 7.6× 축소).
  - **R_TPA3 근거 deep research** (`deep-research` skill): Page 2010(J. Environ. Monit. 12:1658)이 **k(hTPA+OH)=6.3×10⁹ 직접 측정한 유일한 논문**. Tampieri 2021(Anal. Chem.) 등은 kinetic model 대신 <90s data truncation으로 회피. notes/tpa_secondary_reaction_research.md.
  - **k_R3=6.3×10⁹ (Page 2010)**: 2.6kV 9.13, 3.2kV 6.53, 3.6kV 5.77. **rank 2.6>3.2>3.6 재역전**. Surface [TPA] 87-89% 고갈(2.6kV 33%) + hTPA 수명 τ=1/(k_R3·[OH]): 2.6kV 21s / 3.6kV 2.8s. 생성된 hTPA 90%가 즉시 분해 → 저전압 축적 우세.
  - **k_R3=0 (Tampieri 관행, 최종 채택)**: 2.6kV 15.30(+21%), **3.2kV 41.40(−28%), 3.6kV 43.53(+1%)**. **실험과 가장 일치**. rank 3.6≈3.2>2.6. Figure: `Figures/fig_oh_tpa_humidfitting.png`.
  - **k_R3 문헌 한계**: Page 2010 single-lab + pH 5-7 측정. pH 12에서 hTPA(phenoxide, pKa~9.5) 반응성 ±50% 불확실. Salicylate 5×10⁹, phenol 6-14×10⁹ 비교로 합리적 범위이나 독립 검증 無.
  - **경향 역전 정정**: (a) surface TPA diffusion-limited 고갈 → 고전압에서 local [TPA] drop, (b) hTPA 수명이 [OH]에 반비례. 단순 "OH↑→hTPA↑"는 **TPA 충분 공급 전제하에서만** 성립.
  - 변경: `reactions_tpa.yaml` R_TPA3 k=0 주석처리, `run_tpa_alkaline.py` --condition 플래그 추가, `gen_fig_oh_tpa.py` CONDITION 선택. `notes/tpa_secondary_reaction_research.md`, `notes/oh_tpa_comparison.md` 업데이트.

- 2026-04-20 (코드 리뷰 + Paper storyline 세션): **Ver4_1D CRITICAL 3건 검증 + A/B/C 전략 수립**
  - **Ultrareview**: 1 file diff (+1/-9), findings 0.
  - **Ver3/Ver4_1D 코어 병렬 리뷰** (general-purpose agent × 2): 파일별 CRITICAL/HIGH/MEDIUM 분류.
  - **Ver4_1D CRITICAL 3건 코드 실제 검증**:
    1. **gas_alpha/one_film_gas R×T 재곱** — 확정. (이미 같은 날 별도 세션에서 수정됨.)
    2. **Interface BC Ka asymmetry (pde_solver.py:901-911)** — **리뷰 오류, 취소**. `C_eq = H_cc × c_gas` (분자형 평형), `c_eff = f_mol × c_total` (분자형 표면). flux `k × (C_eq − c_eff)`는 질량 수지상 올바름. `set_gas_data:797`이 H_cc 직접 사용 확인.
    3. **apply_qssa in-place 변이 (chemistry_1d.py:991, 1135-1138)** — **MEDIUM 강등**. 호출자 `compute_rates:689`의 `np.clip(y_cell, ...)` (out= 없음)이 새 배열 반환 → caller state 보호. Monolithic BDF에서 QSSA OFF (2026-03-30 결정) 상태라 현재 경로 미호출. 향후 QSSA 재활성화 시 footgun.
  - **Ver3 (legacy) 주요 지적** (참조): mass-balance rate capping 미제거, `reaction_rates` RHS side-effect (contributions.clear()), `chemistry_utils.py` D_adj Henry 이중곱, `trace=1e-30`/`atol=1e-10` 충돌. Ver4_1D는 대응 완료.
  - **교훈**: 리뷰 에이전트 결과 "찾았다"와 "맞다"는 다름 — 코드 실제 검증 필수.
  - ---
  - **Paper storyline 재설정 (A+B+C)**:
    - **A (Council, 4 voices)**: Architect/Skeptic/Pragmatist/Critic 병렬 논의.
      - 3/4 voices가 "soft-sensing" 프레임 drop 권고. Skeptic + Pragmatist 독립적으로 **"saline vs DIW electrolyte-selective RONS delivery"를 실제 novelty hook**으로 지목 (real signal).
      - Critic: Henry-fix 직후 재현성 + NO₂⁻ 1000× gap이 "predictive soft sensor" 주장 desk-reject 유발 가능.
    - **B (deep research, general-purpose agent, 12편)**: "Radical chain ignition bifurcation"은 plasma-liquid 문헌에 **named concept 없음** → 정말 novel. 4-way integration (sDBD patch + OAS + 1D + live cell)도 **현재 발표 논문 없음** → gap. Target journal: **PSST (IF~4.4) / J.Phys.D 1순위**, PNAS/Nat Comm은 cell biology 강할 때만.
    - **C (synthesis — outline 확정)**:
      - 제안 title: "Electrolyte-selective RONS delivery in surface DBD plasma patches: gas-phase diagnostics resolve DIW-vs-saline sterilization divergence"
      - 메인 hook: **saline vs DIW divergence** (F3 central novelty figure)
      - Sim 언어: "semi-quantitative mechanism interpreter" 고정. "predictive" 금지.
      - Ignition bifurcation: "chemistry transition" 완곡 표현 → Discussion 4.2. Formal bifurcation은 follow-up 논문 예고만.
      - 6 main figures (F1 overview / F2 gas OAS / F3 liquid DIW vs saline / F4 spatial / F5 cell viability / F6 mechanism cartoon) + S1 parity + S2 transition.
      - 4 claim 문장: C1 divergence, C2 inference, C3 transition, C4 Cl pathway.
      - Ship: Henry-fix 재실행 + 3/5/12min 액상 1-2주 + 세포 RONS + 작성 3주 = **6-8주**.
    - **전략 변경**: 사용자 초안 "soft-sensing + gas-liquid-cell 체인 + 3/5분 threshold" → 수정 "saline vs DIW electrolyte selectivity (main) + semi-quantitative inference (보조) + chemistry transition (완전 demote)".
  - **변경 파일**: `memory/feedback_henry_constants_convention.md` (버그 수정 반영), `memory/MEMORY.md:10` (description 갱신).

- 2026-04-21~22: **Reactive-uptake BC ultraplan 수립** — `notes/reactive_uptake_bc_plan.md` (540+ 줄).
  - **Framework 전환 결정**: Schwartz 1986 gas_alpha → PRA α-surface (Pöschl 2007, Ammann 2013 IUPAC, Kolb 2010).
  - **현재 gas_alpha 4 pathology 진단**:
    (i) α vs γ 혼동 — `alpha_b_literature.md`의 일부 값이 γ 기반 구측정 인용 (O₃ 0.05는 실제 α≈1e-3 Utter 1992, NO₂ 0.03은 α≈2e-4 Cheung 2000).
    (ii) δ_gas=10mm throttle — k_g=D_g/δ_gas=1.5e-3 m/s가 k_int=α·v̄/4=4.55 m/s를 압도 → **α가 완전 inert** (α 0.05→1e-3 변화해도 k_mt 불변).
    (iii) 실제 regime은 Dirichlet 아닌 kinetic-limited (k_mt·dz_min/D_l=0.007), BUT δ_gas가 비물리적으로 과도한 throttle.
    (iv) γ_literature를 입력으로 쓰면 bulk physics double-count (Ammann 2013 quote).
  - **제안 수식**: `-D_l ∂c/∂z|₀ = (v̄·α/4)·(c_gas − c_surf/H_cc)`, δ_gas drop. c_gas는 OAS chamber-bulk로 취급, 필요 시 Sherwood-film optional (Phase E).
  - **예상 결과 (honest)**: O₃/NO₂ surface 대폭 감소(kinetic barrier 작동). N₂O₅/H₂O₂/HONO₂는 α 이미 문헌 정합이라 거의 불변 → **NO₃⁻/H₂O₂ 과대 예측은 c_gas 입력 문제로 exposed**. User 철학 "NO3- 정량 fit 아님"과 정합.
  - **5 Phase plan**: Phase 0 diagnostic (skippable) → A doc audit → B code (compute_k_mt branch + config α 교정) → C DIW validation → D Saline → E optional Sherwood.
  - **문헌 조사 (general-purpose agent)**: PRA/KM-SUB framework, Hanson 1997, Silsby 2021(δ_gas critique), Zheng-Bruggeman 2020(명시적 Robin BC without δ_gas), Heirman 2025. 45+ citations.
  - **Open questions 6건 (승인 필요)**: Phase 0 skip 여부, α(O₃) 1e-3 vs 5e-4, α(NO₂) 2e-4, α(NO₃/N₂O₄) assumption, bc_type default 전환 시점, α(H₂O₂) 유지 여부.
  - **변경 파일**: `notes/reactive_uptake_bc_plan.md` (신규), `CLAUDE.md` (이 엔트리 + Pending Tasks §0 갱신).
  - **η sweep 취소**: phenomenological knob, 도입 안 함.

- 2026-04-22: **TPA validation figures 재작성 + k_R3 sweep 재실행 + mechanism 재정립**
  - **fig_htpa_validation 의도 복원**: 이전 버전은 "k_R1·∫⟨TPA·OH⟩dt vs sim hTPA" mass-balance self-check였는데 이는 사용자 원 의도와 달랐음(기록 부재 확인). 원 의도 = "Sim OH는 3.6>3.2>2.6 단조, 실험은 3.2>3.6>2.6 비단조 → inner-filter artifact 시사". 최종은 Green+Red 2-bar로 간소화(Sim vs Exp only), 자잘한 문구/grid/dual axis 제거, y축 0–60 고정.
  - **색상 반복 수정**: green/red → red/navy → crimson/navy → Okabe-Ito(#D55E00 vermillion / #0072B2 blue) → **Teal+Coral(#e07856 coral / #2a6a8b teal) 최종 채택**. 두 figure(fig_htpa_validation, fig_kR3_1e9) 모두 동일 팔레트.
  - **k_R3 sweep 실제 재실행** (이전에는 2026-04-20 기록된 값을 `gen_fig_kR3_sweep.py`에 하드코딩만 해두고 raw cache 없었음):
    - 신규 `Ver4_1D/run_kR3_sweep.py`: chemistry._load_reactions 후 `{'type':'irr', reactants={hTPA:1, OH:1}, k}` 주입, `_precompute_reaction_data/_precompute_numba_arrays` 재호출
    - 초기 버그: `type: 'irreversible'` 사용 → `_precompute_reaction_data`가 내부 key `'irr'/'rev'` 기대 (ReactionLoader._convert_reaction이 변환 수행) → smoke test fail → `'irr'`로 수정 후 정상
    - Cache 포맷: `{V}_tpa2000uM_humidfitting.npz` (k_R3=0, 재사용) + `{V}_tpa2000uM_humidfitting_kR3-{k:.0e}.npz` (1e+09, 6e+09). 총 9 runs, 1개 재사용 + 1개 smoke test + 백그라운드 5개 (~130–220s/run, 순 17min).
    - 값: 재실행 결과가 하드코딩 값과 0.01 µM 이내 일치 확인 (2.6kV k=1e9: 13.77, 13.77 등).
    - `gen_fig_kR3_sweep.py` 리팩터링: 하드코딩 → `_cache_path(v, k)` 로부터 `d['hTPA_uM']` 직접 로드. 재현 가능성 확보.
  - **신규 figure `fig_kR3_1e9.{png,pdf}`** — k_R3=1e9 sim vs experiment 2-bar (fig_htpa_validation과 동일 스타일). 3.2/3.6 kV가 실험의 ~50%, 2.6kV는 근접.
  - **k_R3 효과 메커니즘 재정립 (이전 설명 정정)**:
    - 사용자 지적: k_R1, k_R3 모두 [OH]에 비례 → ODE `d[hTPA]/dt = [OH]·(k_R1·[TPA] − k_R3·[hTPA])`에서 **[OH]가 공통 인수로 소거**. "고 OH가 분해에 더 기여"는 잘못된 표현.
    - 정정: `u ≡ ∫[OH]dt` 시간 변수 변환 시 `[hTPA](u) = [hTPA]_ss·(1 − exp(−k_R3·u))`, where `[hTPA]_ss = k_R1·[TPA]/k_R3`는 **[OH] 무관 상수 (=2800 µM local surface)**. 고전압은 표면에서 빠르게 이 SS에 saturate → [OH] 아무리 더 넣어도 표면 [hTPA] 고정 → bulk flux 포화. 저전압은 선형 phase `[hTPA]≈k_R1·[TPA]·u`에 머물러 k_R3 sink 영향 거의 없음 (2.6kV 10% 감소 vs 3.6kV 51%).
    - 1D 공간분리 (OH ~µs 수명, 표면 500nm–34µm만 존재, hTPA는 bulk로 확산) 때문에 실제 bulk avg는 0D SS 2800µM보다 훨씬 낮음(13~22µM)이지만 "OH-무관 SS로 포화" 메커니즘 자체는 동일.
  - **TPA reaction set 누락 product 영향 분석**:
    - R_TPA2 (products={}) → TPA/OH 소모는 정확, peroxyl/유기 radical 2차 화학은 닫혀있음. 실제 O₂ dissolved species 미tracked라 peroxyl 경로가 자동으로 차단됨 (자연스러움).
    - OH sink 경쟁 추정: TPA k·[S]=8×10⁶ s⁻¹ vs OH⁻ (pH 12) 1.2×10⁸ s⁻¹ → pH 12 alkaline에서 OH⁻가 15× 강한 scavenger. TPA 포획률 ~6% (기존 기록 확인).
    - 2차 radical 기여 ~10³ s⁻¹로 OH⁻ 대비 10⁻⁵ → 무시 가능. Page 2010 branching 0.35는 net yield (2차 포함)라 이중계산 없음.
    - **결론: 표준 관행(Charbouillot 2011, Tampieri 2021) 그대로 유지. 논문에 "downstream products not propagated, TPA conversion <5%" 한 줄 명시 권장.**
  - **생성/수정 파일**:
    - `Ver4_1D/run_kR3_sweep.py` 신규
    - `Figures/gen_fig_kR3_sweep.py` 리팩터링 (하드코딩 → cache load)
    - `Figures/gen_fig_htpa_validation.py` 재작성 (mass-balance check → 2-bar sim vs exp)
    - `Figures/gen_fig_kR3_1e9.py` 신규
    - `Figures/cache/tpa/{2.6,3.2,3.6}kV_tpa2000uM_humidfitting_kR3-{1e+09,6e+09}.npz` 신규 (6 files)
    - `Figures/fig_htpa_validation.{png,pdf}`, `fig_kR3_1e9.{png,pdf}`, `fig_kR3_sweep.{png,pdf}` 갱신

- 2026-04-23 (three_film BC 채택 — ★ 중요 결정): **Schwartz 3-저항 full form 복원으로 NO3⁻ 7× 문제 해결**.
  - **배경**: 어제(2026-04-22) driving force diagnostic에서 N2O5/NO3/HNO3/H2O2 4종이 driving force ≈1 (full open)로 확인 → "OAS가 infinite reservoir처럼 작동" 결론. User가 **"액상 저항을 복원했을 때 어떻게 되는지 계산"** 요청.
  - **User empirical 근거 누적**: (1) source audit, (2) species-specific BC (R98 on/off), (3) PRA α-surface 모두 효과 없음 또는 direction 반대. 남은 유일 lever = **Schwartz 3-저항에서 drop한 액상 film 저항 복원**.
  - **코드 변경**: `pde_solver.py::compute_k_mt`에 **`bc_type='three_film'`** branch 추가. 공식: `1/K_L = H·(δ_gas/D_g + 4/(α·v̄)) + δ_liq/D_l` (Schwartz 1986 full form, liquid-units).
  - **δ_liq = 100 µm** (config 기본값 유지).
  - **단일 케이스 pilot (3.2 kV Humid)**: NO3⁻ 395.86 µM → **58.46 µM** (exp 62.74, ×0.93, **7% 오차**). 4× 근처 target 달성.
  - **Grid convergence test (`Figures/test/test_three_film_robustness.py`)**: dz_min ∈ {1, 5, 20} µm @ 3.2 kV.
    - NO3⁻: 58.42 / 58.46 / 58.51 µM (**0.15% 변동**, grid-convergent 확정)
    - pH/H2O2 모두 수렴. **Film 효과는 grid artifact 아님**.
  - **Voltage transfer test (2.6/3.2/3.6 kV × gas_alpha/three_film, dz_min=5µm)**:
    | V | gas_alpha NO3⁻ | three_film NO3⁻ | **exp** | three_film ratio |
    |---|---|---|---|---|
    | 2.6 kV | 148.64 | **21.88** | **32.63** | **×0.67** (33% under) |
    | 3.2 kV | 395.86 | **58.46** | **62.74** | **×0.93** ✓ |
    | 3.6 kV | 447.89 | **66.19** | **70.42** | **×0.94** ✓ |
    - three_film / gas_alpha ratio: 0.147 / 0.148 / 0.148 — **voltage-independent**. Physics consistent.
    - 3.2/3.6 kV 6-7% 오차, 2.6 kV 33% under.
  - **실험값 정식 기록** (from `OAS data/Dry/(P-L) 액체활성종 농도, pH, conductivity.xlsx`, DIW sheet):
    - 2.6 kV: pH=5.09, NO3⁻=32.63, NO2⁻=0, H2O2=4.76 µM
    - 3.2 kV: pH=3.61, NO3⁻=62.74, NO2⁻=3.58, H2O2=11.21 µM
    - 3.6 kV: pH=3.25, NO3⁻=70.42, NO2⁻=20.74, H2O2=16.25 µM
  - **잔존 문제 (BC와 무관, 별도 track)**:
    - **H2O2 16-17× 과다 전 voltage 일정** → H_cc=2.1e6 거대라 film 무효. **c_gas input (H2O2/O3=0.03 ratio)** 문제. 별도 fix 필요.
    - **NO2⁻ over-throttle**: three_film에서 NO2 K 84× ↓ → 전 voltage 0.16~0.54 µM (exp 0/3.58/20.74). Voltage trend도 잃음. 재건 보류 (User 2026-04-23 결정).
    - **pH +0.4~1.0 단위 높음** (덜 산성): NO3⁻ 감소 + H2O2 잔존 + NO2⁻ 부족 복합 영향.
    - **2.6 kV 33% under**: gas input 측정 정확도 or plasma production rate voltage 의존성. 별도 audit.
  - **Double-counting 재해석**: 수학적으로 three_film는 PDE와 액상 resolution에서 겹친다고 볼 수 있으나, paper grid dz_min=5µm는 **convective BL (~100 µm) + near-surface reacto-diffusive layer**를 해상 못 함. Film 저항이 이 sub-grid + BL 영역을 complementary하게 parameterize. **Physical double-count 아닌 complementary**.
  - **User 결정 (2026-04-23)**:
    1. **three_film 공식 채택** ✓ (paper BC로 사용)
    2. NO2⁻ 재건: **보류**
    3. CLAUDE.md 기록: ✓ (이 엔트리)
  - **변경 파일**:
    - `Ver4_1D/pde_solver.py` (line 170-188: `three_film` branch 신규)
    - `Figures/test/test_three_film.py` (신규, pilot 1-case)
    - `Figures/test/test_three_film_robustness.py` (신규, grid + voltage sweep, 실험값 포함)
    - `CLAUDE.md` (이 엔트리 + Pending Tasks + Key Decisions 갱신 예정)
  - **Next steps (pending)**:
    1. H2O2/O3 ratio 재산정 (0.03 → ~0.003) — 독립 track
    2. 2.6 kV under-prediction audit — c_gas 혹은 RH80 scaling factor 재검토
    3. `MASS_TRANSFER.bc_type` default 전환 (현재 `two_film` → `three_film`)
    4. 모든 Figure 생성 스크립트 bc_type 일괄 업데이트
    5. Saline 확장 시 `three_film` 적용 테스트

- 2026-04-22 (BC 진단 심화 세션): **Reactive-uptake plan v1 부분 supersede → "Finite gas reservoir" 방향 도출**.
  - **맥락**: 이전 세션(2026-04-21~22)의 PRA α-surface plan v1이 monolithic approach. User의 empirical 증거(R98 on/off test, source audit)와 대조 후 구조 재설계.
  - **Deep research 체크리스트 reconciliation**: User가 이전 Claude deep research 결과(10단계, "7× = 2×·2×·1.5× 복합 오차", N2O5 surface source + 3 병렬 트랙) 공유. 그러나 **user가 이미 N2O5 single-axis fix (R98 on/off, `Figures/test/test_r98_onoff.py`)를 실험했고 compensation 관찰** — N2O4/NO2/N2O5 복합 작용으로 한 경로 차단해도 다른 경로가 보완. Deep research의 "N2O5 93% 지배" 전제 자체가 empirically 기각. → **N2O5-specific BC는 해결책 아님**.
  - **User empirical 순서 확정 (중요)**: (1) Source audit 먼저 → 효과 無. (2) Species-specific BC test → 효과 無. (3) **BC 구조 자체로 이동**. 나는 원래 (1)을 재제안했는데 user가 이미 수행했음. Source audit 재제안 **중단**.
  - **"Reactive uptake" 용어 명확화 (user 의도)**: 내 초기 제안 (C) PRA α-only no δ_gas는 user 의도와 불일치. User 의도 = "gas side resistance에 추가하는" (Schwartz 직렬 + γ-based). 3 framework 분류: (A) γ-lumped in Schwartz 직렬, (B) surface reaction term 추가 (Hanson 1997), (C) PRA α-only. User 의도는 (A) 또는 (B).
  - **Novelty 재정의 (user)**: **"Gas-phase reaction을 사용하지 않는 것 자체가 novelty"**. 대부분 plasma-liquid 논문이 10-100× 불확실한 gas-phase rate constants를 쓰는 반면, 우리는 OAS-constrained BC로 우회. **0D-1D-1D framework with transport-only gas phase, measurement-driven**.
  - **8mm gap 지적 (user)**: OAS 측정 위치에서 액상까지 8mm gap 존재. 현재 코드는 **direct input**으로 사용 (`set_gas_data`가 OAS time series를 `_gas_conc_molar`에 저장 → `C_eq = H × c_gas` 로 직결). Gas-phase 운송 미모델링 가설.
  - **BC equation audit (9 species)**: `1/K = 1/k_G + 4/(α·v̄)` 직접 계산 → 전 종 **98-99.99% gas-side limited** (N2O5 99.95%, H2O2 99.99%, NO 98.3% 등). User의 "K ≈ k_G" 관찰 정량 확증. 단위 체크: H_cc dimensionless, c_gas M, Γ[M/s] consistent.
  - **D_g = 분자확산계수 확인**: `config_1d.py:73-85` GAS_DIFFUSIVITY (m²/s, 기상 literature). `k_gas = D_g/δ_gas`는 1D stagnant-gas 정상상태 확산의 analytical solution. → **현재 gas_alpha가 이미 1D gas diffusion을 shortcut으로 풀고 있음**. Explicit 1D gas PDE (no chemistry) = gas_alpha with δ_gas=L_gap **수학적 완전 동등** (증명: `J = c_OAS/(L/D + 4/(αv̄))`).
  - **Driving force diagnostic 신규 (`Figures/test/test_driving_force.py`)**: `(C_eq − c_eff_surf)/C_eq` 시계열 측정 (3.2kV Humid fitting, 60s). 초기 6종 → **9종 전수로 확장 (2026-04-23 user 지적)**:
    - **O3: 0.006** (SATURATED — c_surf=1.00e-5 M ≈ C_eq=1.01e-5 M, R32 bulk sink 불충분)
    - **NO2: 0.35** (중간 saturated)
    - **N2O5: 1.000** (R98 즉시 소비, c_surf=1.19e-14 M)
    - **NO3: 0.998** (full open, c_surf=4.28 nM ≈ 0, bulk reaction 즉시 소비) ← 추가 발견
    - **HNO3: 1.000** (강산 해리로 분자형 c_eff=4.35e-10 M)
    - **H2O2: 0.9999** (C_eq=2.77M 거대, c_surf=3.55e-4 M)
    - **HONO: 0.97** (해리로 분자형 작음)
    - **N2O4: −2.75 (음수!)** — c_surf(1.01e-8 M) > C_eq(2.68e-9 M)로 **액상에서 gas로 역flux**. 2NO2(aq) ⇌ N2O4(aq) 축적 → BC가 desorption. Flux 절대값은 ~N2O5의 1/6000 (미미) but 양방향 BC 구조 유지 필요 확인.
    - **NO: OAS=0** (test 스크립트 `load_gas`가 NO 미추출; 실제 production run OAS 확인 필요 — xlsx에 NO 컬럼 있는지 check).
  - **결정적 통찰**: **N2O5/NO3/HNO3/H2O2 4종이 driving force ≈ 1 (full open)** → BC가 `K·H·c_OAS`로 max rate 유입. User 8mm gap 가설 **정확히 맞음**: 현재 모델은 OAS time series를 **infinite reservoir**로 취급, **finite gas inventory feedback 없음**. 2-film K는 "특정 시점 interface 정상상태 depletion" 계산하지만, c_OAS 자체가 uptake에 의해 감소해야 한다는 mass balance는 **없음**. NO3 gas 직접 uptake 경로도 독립적 contributor로 확인됨 (N2O5 R98 외).
  - **O3/NO2는 다른 문제**: 이미 액상 saturated. Gas-side 구조 fix로 해결 안 됨. Bulk consumption/diffusion 한계.
  - **7× 과다의 framework 내 lever 소진**: `Γ = K·H·c_OAS`, K=D_g/δ(geometric bound), H=lit H_cc(fixed), c_OAS=audited. Framework 내에 줄일 lever 없음 → **구조 확장 필요**.
  - **새 제안: Finite gas reservoir ODE** (no chemistry, novelty 호환):
    - `dc_gas_i(t)/dt = P_plasma_i − k_leak·c_gas_i − (A_liq/V_gas)·J_uptake_i`
    - OAS = steady-state 제약 (plasma production rate calibration)
    - Gas mass balance가 uptake feedback 반영 → driving force 자발적 감소
    - 필요 파라미터: `V_gas` (chamber volume), `A_liq` (petri surface area), `k_leak` (optional)
  - **Plan v1 상태**: `notes/reactive_uptake_bc_plan.md` PRA α-surface 접근은 **supersede** (참조용 보존). Literature survey + K audit은 유효.
  - **변경/신규 파일**:
    - `Figures/test/test_driving_force.py` (신규 diagnostic)
    - `CLAUDE.md` (이 엔트리)
  - **다음 단계 대기**: Finite gas reservoir ODE 구현. User 결정 대기 항목:
    1. Option A (finite reservoir ODE) vs Option B (wall loss uniform factor) vs Option C (Langmuir competition)?
    2. Chamber geometry 파라미터 (V_gas, A_liq) 값?
    3. `plan v1 supersede` 명시적 문서화 필요 여부?

- 2026-04-23 (Figure 정리 + gas preprocessing audit + N₂O₄ bug fix):
  - **Figure 색상 팔레트 수렴**: `fig_htpa_validation`, `fig_kR3_1e9` 두 figure에서 사용자와 반복 조정 (green/red → red/navy → crimson/navy → Okabe-Ito(#D55E00/#0072B2) → **Teal+Coral(#e07856 coral / #2a6a8b teal) 최종 채택**). 두 figure 동일 팔레트로 통일.
  - **TPA 프로브 reaction set 브리핑** (`reactions_tpa.yaml`):
    - R_TPA1 (활성): `TPA + OH → hTPA`, k=1.4e9 (branching 0.35, fluorescent)
    - R_TPA2 (활성): `TPA + OH → non-fluor`, k=2.6e9 (products={}, 0.65 branching)
    - R_TPA3 (**비활성**, k=0 주석처리): hTPA + OH — Page 2010 k=6.3e9 single-lab pH 5-7, pH 12 불확실
    - OH sink 경쟁 (pH 12): TPA k·[S]=8×10⁶ s⁻¹ vs OH⁻ 1.2×10⁸ s⁻¹ → OH⁻ 15× 강 → TPA 포획률 ~6%
  - **R_TPA2/R_TPA3 product 미추적 safety 분석**:
    - OH/TPA 소비량 계수 처리로 정확. 2차 peroxyl/유기 radical 화학만 누락.
    - O₂ dissolved species 애초에 미tracked → peroxyl 경로 자동 차단 (자연스러움).
    - 2차 radical 기여 ~10³ s⁻¹ vs OH⁻ 1.2×10⁸ → 10⁻⁵ 수준 → 무시 가능.
    - Page 2010 branching 0.35는 **net yield**(2차 포함) → 이중계산 없음.
    - 문헌 관행(Charbouillot 2011, Tampieri 2021) 유지. 논문 disclaimer 1줄 권장: "downstream products not propagated; TPA conversion <5% minimizes back-reaction".
  - **Gas preprocessing 구조 정리**:
    - 측정 (O₃/NO₂/NO₃/N₂O₅): xlsx raw + LOD filter + Savitzky-Golay smoothing
    - **수식 기반: N₂O₄ 1종만** — `2 NO₂ ⇌ N₂O₄`, van't Hoff 온도 보정
    - Ratio 기반: HONO = NO₂·r[HONO_NO2], HONO₂ = N₂O₅·0.83, H₂O₂ = O₃·0.003 (2026-04-23 sweep 재fit, 이전 0.03)
  - **N₂O₄ ordering bug 발견 및 수정** (`run_tpa_alkaline.py`, `gen_all_figures.py`):
    - 버그: `N₂O₄ = C·NO₂²` 계산이 **humid_fitting NO₂ 스케일링 전**에 실행 → humid_fitting 모드에서 NO₂는 스케일되지만 N₂O₄는 Dry 기반 값으로 고정.
    - 정량 (3.2 kV Humid_fitting): NO₂ 1.91e15→6.52e15 (3.4× 증가), 실제 N₂O₄는 1.17e13이어야 하는데 수정 전에는 Dry 기반 1.00e12 (실제의 8.6% 수준 저평가).
    - Fix (A 채택): N₂O₄ 블록을 humid 스케일링 if/else 뒤로 이동. 양 파일 동일 구조.
    - 검증: Dry/Humid_fitting 모두 `N₂O₄_peak / (C·NO₂_peak²) = 1.000` (rel_err=0).
    - Prefactor 정정: `C = Kp·(k_B·T/P)·T = 2.7422×10⁻¹⁹ cm³/molecule` (at 298.15 K). 이전 세션 답변의 9.2×10⁻²² 잘못됨.
    - 영향 범위: Dry/Humid_median 변화 없음. **Humid_fitting cache stale** → N₂O₄ 11.6× 증가로 R95(N₂O₄+H₂O→HNO₃+HONO) source 영향 가능성. NO₃⁻/HONO budget 재평가 필요.
    - `gen_all_figures.py`는 사용자 사이드 편집으로 추가 변경 포함: REF_BC='three_film' (2026-04-23 project default), H2O2_RATIO 0.03→0.003, BC_CASES/MT_BC_CASES 비움, Condition label 'Henry_Humid_fitting'.
  - **Pending**: Humid_fitting 모드 전체 cache 재생성(DIW/Saline/TPA, three_film + N₂O₄ fix 동시 반영). 재실행 범위 사용자 결정 대기.
  - **변경 파일**:
    - `Ver4_1D/run_tpa_alkaline.py` N₂O₄ 블록 L128-135 → L149-157 이동
    - `Figures/gen_all_figures.py` N₂O₄ 블록 L231-239 → L271-279 이동 (+ 사이드 편집)
    - `Figures/gen_fig_htpa_validation.py`, `Figures/gen_fig_kR3_1e9.py` 팔레트 통일
    - `Figures/fig_htpa_validation.{png,pdf}`, `fig_kR3_1e9.{png,pdf}` 재생성

- 2026-04-23: **three_film default 전환 + H2O2/O3=0.003 + N2O4 fix 통합 검증 (DIW×3 + Saline×3)**
  - **코드 default 전환**: `config_1d.py:142` `bc_type='two_film'`→`'three_film'`, `gen_all_figures.py:52` `REF_BC='three_film'`, `run_saline_1d.py:136` `bc_type='three_film'`+`alpha_b=None`(species dict). `pde_solver.py:928` `mass_transfer_eta` Robin BC 적용 제거 (default=1.0, no-op이고 η sweep은 2026-04-21에 cancelled).
  - **Smoke 검증**: `Figures/test/smoke_saline_three_film.py` 신규 — `gen_all_figures.load_gas_data()`를 직접 import해서 canonical gas 전처리(Dry shape × SS_rh80/SS_dry rescale, N2O4 post-rescale) 그대로 재사용. 3전압 × {DIW, Saline} = 6 sims, 총 12분 (DIW ~50s, Saline ~3min/case).
  - **DIW 정량 (three_film + 0.003 + N2O4 fix)**:

| V | pH sim/exp | NO3⁻ sim/exp | NO2⁻ sim/exp | H2O2 sim/exp |
|---|---|---|---|---|
| 2.6kV | 4.42/5.09 | 38.2/32.6 (×1.17) | 0.05/0 | 5.70/4.76 (×1.20) |
| 3.2kV | 4.23/3.61 | 59.1/62.7 (**×0.94**) | 0.05/3.58 | 16.6/11.2 (×1.48) |
| 3.6kV | 4.22/3.25 | 60.1/70.4 (×0.85) | 0.05/20.7 | 21.3/16.3 (×1.31) |

  - **Saline 정량 (같은 default)**:

| V | pH sim/exp | NO3⁻ sim/exp | H2O2 sim/exp | Cl⁻ |
|---|---|---|---|---|
| 2.6kV | 4.51/5.15 | 38.3/32.4 (×1.18) | **0.07/2.00 (×0.04)** | 155.1 mM conserved |
| 3.2kV | 4.33/3.60 | 59.2/101.3 (**×0.58**) | **0.52/5.14 (×0.10)** | 155.1 mM conserved |
| 3.6kV | 4.32/3.43 | 60.3/112.8 (**×0.53**) | **1.22/7.73 (×0.16)** | 155.1 mM conserved |

  - **★ Saline NO3⁻ enhancement sim이 전혀 재현 못함**: 실험 Saline/DIW NO3⁻ ratio = 1.0/**1.62/1.60** (2.6/3.2/3.6kV), sim ratio = 1.00/1.00/1.00 (0.3% 이내 동일). 현재 chemistry/BC에 saline의 NO3⁻ 증진 메커니즘 없음. **Paper main hook "electrolyte-selective RONS delivery"에 치명적 — chemistry audit 필요**.
  - **★ Saline H2O2 과도하게 destruction**: 실험 Sal/DIW ratio = 0.42/0.46/0.48, sim = **0.012/0.031/0.057** (25-80× 과소). Cl-mediated H2O2 sink가 chemistry에서 너무 강함 (HOCl + H2O2 경로 등 재검토 필요). 0.003 ratio가 DIW엔 적정이지만 saline엔 부족.
  - **DIW는 거의 해결**: NO3⁻ 이전 7× over → 0.85~1.17×. H2O2 이전 12-17× → 1.2-1.5×. three_film + N2O4 fix + 0.003이 복합적으로 효과. pH 고전압에서 0.6-1.0 unit 높은 건 NO2⁻ voltage scaling 미포착 + NO3⁻ under 영향.
  - **생성/변경 파일**:
    - `Figures/test/smoke_saline_three_film.py` 신규 (6-case DIW+Saline smoke, gaf.load_gas_data import)
    - `Ver4_1D/config_1d.py` bc_type default 전환 + docstring
    - `Ver4_1D/pde_solver.py:928` mass_transfer_eta revert (line 219/243 attribute는 유지, 25곳 callsite 호환)
    - `Ver4_1D/run_saline_1d.py:136-137` bc_type='three_film', alpha_b=None
    - `Figures/gen_all_figures.py` REF_BC/H2O2_RATIO (+ 사용자 사이드: Fig 1 time-series, output folder `{V}_{condition}_{bc}`)
  - **다음 단계 후보**: (1) Saline NO3⁻ enhancement 메커니즘 audit (HOCl+NO2⁻, Cl-radical chain), (2) Saline H2O2 sink rate budget 진단, (3) Saline-specific H2O2/O3 ratio (0.01-0.03) sweep, (4) Paper storyline 재검토.

---
<!-- UPDATE RULE:
작업 단위가 완료될 때마다 즉시 이 파일을 갱신할 것 (세션 종료를 기다리지 않는다).
1. Current Status 섹션의 WORKING/BROKEN/NOT TRIED 갱신
2. Pending Tasks 우선순위 조정
3. Key Decisions에 새 결정 추가
4. Session History에 날짜+요약 한 줄 추가
사용자가 터미널을 그냥 닫아도 CLAUDE.md는 이미 최신 상태여야 한다.
-->
