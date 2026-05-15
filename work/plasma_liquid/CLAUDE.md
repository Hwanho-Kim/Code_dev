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

- 2026-04-23 (이어서): **H2O2 sink 진단 → S36 (Cl2⁻ + H2O2, k=4.2e8) disable + Saline 검증**
  - **rate budget 진단** (`Figures/test/diag_h2o2_rate_budget.py` 신규): DIW vs Saline 3.2 kV Humid_fitting, 600 s final state. compute_per_reaction_rates + compute_mass_transfer_flux 활용.
    - DIW H2O2 sink 8.57e-9 M/s: R41(OH+H2O2) 75.7%, R91(NO3+H2O2) 24.2%
    - **Saline H2O2 sink 4.28e-8 M/s: S36 단독 99.4% (4.25e-8 M/s)**, S35 0.3%
    - k_eff: DIW 5.2e-4 s⁻¹, Saline 82 s⁻¹ → **15만 배 차이** (S36 단독 원인 확정)
  - **S36 출처 audit** (`Article/Article/Saline_reaction.pdf`, Liu 2016 SI ppap.201600113):
    - Liu 2016 #36: Cl2⁻ + H2O2 → OH + OH⁻ + Cl2, k=4.2×10⁸, ref [44] = **Lundström, Christensen, Sehested, Radiat. Phys. Chem. 61, 109 (2001)** (방사선화학)
    - Liu 2016 #35는 동일 반응물인데 k=1.4×10⁶ (300× 차이, 동일 분기 두 경로 부여 — 비정상)
    - Buxton 1988 (JPCRD 17:513), Jayson 1973, Yu 2004, Verlackt 2018, Heirman 2019 등 주류 plasma-liquid/radical compilation에 OH+OH⁻+Cl2 분기 **부재**
    - 열역학: S36 ΔE ≈ +0.01 V (thermoneutral), S35 ΔE ≈ +0.63 V (favorable). S36이 더 어려운 경로인데 300× 빠르다는 것은 비물리적
    - 의심: Lundström 2001은 radiation chemistry context의 "effective" rate constant일 가능성 → plasma-liquid에 elementary로 옮기면 double-counting
  - **S36 disable** (`Ver4_1D/reactions_saline.yaml` L394-405 주석화) 후 6 cases 재실행 (`smoke_saline_three_film.py` 그대로):

| V | metric | DIW sim | DIW exp | Sal sim (S36 off) | Sal exp |
|---|---|---|---|---|---|
| 2.6kV | pH | 4.42 | 5.09 | 4.45 | 5.15 |
|       | H2O2 µM | 5.70 | 4.76 | **5.35** (vs S36 on=0.07) | 2.00 |
|       | NO3⁻ µM | 38.2 | 32.6 | 38.3 | 32.4 |
| 3.2kV | pH | 4.23 | 3.61 | 4.27 | 3.60 |
|       | H2O2 µM | 16.6 | 11.2 | **15.5** (vs 0.52) | 5.14 |
|       | NO3⁻ µM | 59.1 | 62.7 | 59.2 | 101.3 |
| 3.6kV | pH | 4.22 | 3.25 | 4.26 | 3.43 |
|       | H2O2 µM | 21.3 | 16.3 | **19.6** (vs 1.22) | 7.73 |
|       | NO3⁻ µM | 60.1 | 70.4 | 60.2 | 112.8 |

  - **Saline H2O2 30× 회복 확정** (3.2kV: 0.52 → 15.5 µM). **그러나 sim Sal ≈ DIW로 수렴** (실험 Sal/DIW=0.46 vs sim Sal/DIW=0.93) — S36 완전 제거가 과도, 실제 saline destruction 일부는 존재해야 함. Intermediate k 검토 필요.
  - **Saline NO3⁻ enhancement 여전히 미포착**: sim Sal/DIW=1.00, exp=1.62/1.60. S36과 무관한 별도 chemistry gap.
  - **Figure 산출** (`Figures/test/fig_s36_comparison.py` → `fig_s36off_vs_exp.{png,pdf}`): Saline 단독 4-panel (pH/H2O2/NO2⁻/NO3⁻) × 3 voltage bar chart. DIW 비교 제거.
  - **결정 기록**: S36 disable은 진단용 임시 조치 — Lundström 2001 원문 검토 후 elementary vs effective 판정. 잠정 옵션은 (a) k=1.4e6 (S35와 동일), (b) k=1e7~1e8 fitting, (c) Liu 2016 set 외 reference set로 교체.

- 2026-04-24 ~ 2026-04-26: **추가 sweep 분석 (사용자 사이드 작업)**
  - **`Figures/test/test_h2o2_ratio_sweep.py`** — H2O2/O3 ∈ {0.001, 0.003, 0.01, 0.03}, 3전압 × 4 ratio = 12 sims. 결과로 **DIW 기준 H2O2/O3 = 0.003 채택 근거** 확보 (이전 0.03은 12-17× over). gen_all_figures.py default도 0.003으로 갱신됨.
  - **`Figures/test/test_hono_hono2_sweep.py` + `hono_hono2_sweep_tables.py`** — HONO/NO2 ∈ {0.007, 0.03, 0.07, 0.1}, HONO2/N2O5 ∈ {0.83, 2, 3, 5}. 24 sims. 결과 (DIW, three_film, H2O2/O3=0.003 고정):
    - **HONO/NO2 ↑ → NO2⁻ 크게 증가** (3.6kV: 0.05 → 2.6 → 12.4 → 21.7 µM, 실험 20.74에 근접). NO3⁻도 ~30% 증가. **DIW NO2⁻ voltage scaling 미포착 문제의 핵심 lever 발견**
    - HONO2/N2O5 ↑ → NO3⁻와 pH 영향 (HNO3 직접 dissolution)
    - **결과 npz/pdf**: `hono_hono2_sweep_tables.{png,pdf}` (8 panel, sweep × 4 metric)
  - **`Figures/test/test_no2m_budget_hono_sweep.py`** — HONO sweep 시 NO2⁻ rate budget 변화 추적. 결과 `no2m_rate_budget_hono_sweep.pdf`. (세부 결과 미검토 — 추후 audit 필요)
  - **사용자 사이드 commit `8f41fb0` (2026-04-26)**: 위 모든 변경사항 통합 commit. 3780 lines 추가, 75 files. Working tree clean 상태.

- 2026-04-28 (오늘 진행 상황 점검): **Git commit 완료, CLAUDE.md 미기록 작업 정리**
  - 2026-04-23~26 작업 모두 commit됨, 본 entry로 미기록 항목 보강.
  - **현재 Pending (재정의)**:
    1. **★ Saline NO3⁻ enhancement chemistry audit** — sim Sal/DIW=1.00 vs exp 1.6, paper main hook 직결, 미수행
    2. **★ S36 k 재조정 또는 Lundström 2001 원문 audit** — 현재 disable이 임시, intermediate k(1e6~1e8) 또는 alternative reference set 조사
    3. **DIW NO2⁻ voltage scaling**: HONO/NO2 ↑ 시 sweep 결과로 0.07~0.1 적용 시 실험 근사. ratio 정상화 필요 (현재 voltage-dependent default 0.007~0.009)
    4. **Saline H2O2 fine-tune** — S36 off로 sim Sal=DIW이지만 exp Sal=0.46×DIW, 적정 saline-specific sink 부족
    5. **Paper storyline 재검토** — saline NO3⁻ 미재현은 "electrolyte-selective RONS delivery" hook 약화

- 2026-04-28: **Saline RONS=DIW 원인 규명 — Cl 화학 0.00004% 활성도 정량 확증**
  - **NO3⁻ rate budget 진단** (`Figures/test/diag_no3_saline_vs_diw.py` 신규): DIW vs Saline 3.2 kV.
    - DIW production 8.31e−8 M/s: R98(44%) + R32(24%) + R19(15%) + R92(9%) + R95(9%)
    - Saline production 9.22e−8 M/s: R98(39%) + R32(21%) + **S55(18%)** + R19(13%) + R95(8%) + R92(0.1%)
    - **★ NO3 radical sink swap 발견**: DIW의 R92(NO3+NO2⁻ → NO3⁻+NO2)가 Saline에서 S55(NO3+Cl⁻ → NO3⁻+Cl)로 갈아탐. 둘 다 1:1 NO3⁻ 생성, 154 mM Cl⁻이 0.05 µM NO2⁻를 200만배 압도해 S55가 R92를 가로챔.
    - S56(N2O5+Cl⁻) bulk pathway = 4.9e−11 (0.05%): N2O5 bulk≈0이라 무력
    - **Saline NO3⁻ enhancement (실험 1.6×) 미포착의 정량적 원인 확정**
  - **NO2⁻/H2O2 smoke vs diag 270× 불일치 의문 해결**: HONO_NO2 ratio가 04-23(voltage-specific 0.00707/0.00707/0.00662) → 04-26(uniform 0.10)으로 commit 8f41fb0에서 변경됨. 코드 정상, 입력 파라미터 변화. `check_spatial_avg.py` 직접 dump로 검증 완료. SWEEP1 데이터 (HONO=0.007 → NO2⁻=0.054, HONO=0.10 → NO2⁻=14.7) 일치.
  - **Cl⁻ 보존 메커니즘 정리** (`pde_solver.py:298-311, 634-653`): _enforce_cl_conservation은 **총 Cl 원자 질량 보존** (21개 Cl 함유 종 합, n_Cl 가중치). Sturm & Silva 2024 projection method, BDF 수치오차를 dominant pool(Cl⁻)에 흡수. 화학 변환 자체는 보존 강제와 무관하게 정상 진행.
  - **Cl 진화/활성화 진단** (`Figures/test/diag_cl_evolution.py` 신규):
    - 이전 diag의 +1112 µM Cl⁻ drift는 **vol_avg 계산 bias** (Σdz=10.07mm vs L=10.0mm geometric grid mismatch). **실제 drift = 0.0000%** (5자리수 보존)
    - **Cl 화학 활성도 = 0.00004%** — 154 mM Cl⁻ 중 단지 65 nM만 transient species로 변환
    - 종별: Cl3⁻ 21 nM (가장 큼), HClO_total 531 pM, ClO3⁻ 334 pM, ClNO2 13 pM, Cl2 2.1 pM, Cl2⁻ 0.34 pM
    - **HClO_total = 531 pM**, 실험 plasma-activated saline의 mM 보고와 ~10⁹배 괴리
    - per-cell Cl⁻ 변화: surface ~ 중간 영역 균일하게 -0.45 µM, bottom 0
  - **Cl 활성화 차단 메커니즘 (4가지 동시)**:
    1. Chain initiation 부족 — Cl atom 생성 single-source (S55, 1.68e−8 M/s)
    2. ★ Catalytic short-circuit — S54(Cl2⁻+NO2⁻ → 2Cl⁻) + S35(Cl2⁻+H2O2 → 2Cl⁻)이 Cl2⁻ sink 45% 차지하며 즉시 Cl⁻ 회수
    3. Quadratic dependence trap — Cl2 생성 모두 [Cl]² 또는 [Cl2⁻]² 의존, trace 농도(fM/pM)에서 완전 차단
    4. OH 고갈 — S4(OH+Cl⁻ → HOCl⁻, k_eff=6.6e8 s⁻¹)가 OH 즉시 scavenge, [OH]_saline = 1.8e−15 M (DIW 16× 작음). S39(Cl2⁻+OH → HClO) 죽음
  - **Cl⁻ 소비 budget**:
    - Production 1.52e−8 M/s: S54(97.5%) + S35(1.7%) + S58(0.4%) + 기타
    - Consumption 3.37e−8 M/s: S55(49.9%) + S6(49.8%) + S56(0.1%) + S4(0.1%) + S96(0.0%)
    - **S55 ≈ S6 정확 일치** (Cl atom SS 강제, τ_Cl ≈ 1 ns) → Cl atom은 순간 매개체
    - **★ Catalytic cycle 등가성**: S55 + S6 + S54 net = NO3 + NO2⁻ → NO3⁻ + NO2 = R92와 stoichiometrically 동일. **Cl⁻은 R92 catalyst 역할만 함**
    - Cycle turnover: 600s 동안 Cl⁻ 1개당 평균 6.0e−5회 cycle (15,000개 중 1개만 한 번 돌아봄) — 효율 극저
  - **활성화 가능성 평가 (현재 reaction set 외부 메커니즘 필요)**:
    - (A) Plasma-direct surface oxidation: e⁻/O atom/OH 표면 직접 작용. 모델에 없음
    - (B) Gas-phase Cl 입력 (Cl2/HClO MT): GAS_TO_AQUEOUS_MAP에 Cl 종 일체 없음. 측정값 있어야 추가 가능
    - (C) OH 고갈 우회: S4 효율 낮추거나 다른 OH source 추가
  - **결론**: bulk reaction-only chemistry로는 plasma-activated saline의 mM HClO를 설명 불가. paper main hook "electrolyte-selective RONS delivery"가 sim 미포착. saline novelty 재정의 또는 surface chemistry 도입 필요.
  - **CLAUDE.md 갱신 + commit `6241f86`**: 04-23~28 session log (+48 lines).
  - **변경/생성 파일**:
    - `Figures/test/diag_no3_saline_vs_diw.py` 신규 (NO3⁻/NO2⁻/H2O2/Cl⁻ 4-budget + Saline-specific Cl chemistry ranking)
    - `Figures/test/check_spatial_avg.py` 신규 (spatial_avg key dump 검증)
    - `Figures/test/diag_cl_evolution.py` 신규 (Cl 시간 진화 + 보존 검증 + per-cell 분포 + activation 정량)
    - `CLAUDE.md` 04-23~28 entry 추가 (commit 6241f86)
  - **다음 단계 후보**: (1) Cl 활성화 메커니즘 결정 (plasma-direct/gas Cl 입력/OH 우회 중 하나 도입), (2) Paper storyline 재정의 (saline novelty hook 약화 반영), (3) 다른 priority pending로 전환 (DIW NO2⁻ scaling, S36 audit, 등).

- 2026-04-28 (HONO uniform 확정 + fig 1c/2b 신규 + 진단·문헌조사 세션):
  - **★ 핵심 결정: HONO/NO2 = 0.10 uniform 채택**. voltage-dependent 폐기 (이전 RH80_RATIOS 0.00915/0.00707/0.00662 → 전 voltage 0.10).
    - 결정 근거: 3.6 kV NO2⁻ ×1.05 perfect (실험 20.74 vs sim 21.7), 2.6 kV overshoot (19.9 vs 0)은 uniform의 한계로 수용. NO2⁻ trend 부분 fit, paper에서 first-order surrogate로 서술.
  - **HONO ∝ NO2 문헌 deep research 완료**: 두 채널 모두 NO2에 first-order 의존:
    - **Homogeneous**: OH + NO → HONO (NOx mode에서 [NO]∝[NO2]) — Sakiyama 2012, JPhysD 45 425201
    - **Heterogeneous**: 2 NO2(g) + H2O(ads) → HONO + HNO3 (γ_NO2→HONO ~ 10⁻⁵-10⁻⁴) — Finlayson-Pitts 2003 PCCP, Liu 2022 ES&T
    - 결론: 비례식 OK, 단 k는 RH/power/τ_res 의존 → "single operating point first-order surrogate"로 paper 서술 권장
    - 핵심 reference: Sakiyama 2012, Pavlovich/Clark/Graves 2014 PSST 23 065036 (FTIR HNO2 정량), Bruggeman 2016 PSST 25 053002, Liu 2022 ES&T 56
  - **3 voltage 폴더 일괄 재생성** (HONO=0.10): `Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film/` 삭제 후 재생성. fig1, fig1b, **fig1c (신규)**, fig2, **fig2b (신규)**, fig3-6.
  - **Figure 변경 사항 종합**:
    - **fig1_bc_comparison**: 2x2 subplot, **Sim+Exp 막대 2개** per metric (이전 dashed line + bar → bar+bar).
    - **fig1b_mt_flux**: three_film만 표시 (MT_BC_CASES = [('three_film',...)]). **윗 열 post-flux SG smoothing window=75 (cosmetic)**, 아래 열 cumulative는 raw 적분 유지 (mass balance 보존, bias <0.15%).
    - **fig1c_concentration_timeseries (신규)**: 1×2 subplot. (a) Long-lived: NO3⁻/NO2⁻/**O3×1000**/H2O2 (linear µM). (b) Short-lived 10종: OH, **HO2/O2⁻ acid-base 분리** (pKa=4.8), **ONOOH/ONOO⁻** (pKa=6.6), **O2NOOH/O2NOO⁻** (pKa=5.9), O3⁻, HO3, O (log scale pM, solid=분자 / dashed=이온).
    - **fig2_rate_evolution**: median(10s) + **SG smoothing window=75 (cosmetic)**. **MT 라벨 reaction-style** (`HONO2 → H+ + NO3-`, `HONO → H+ + NO2-`, `O3(g) → O3(aq)`, `H2O2(g) → H2O2(aq)`) — 이온 직접 MT가 아닌 **분자형 가스 + 강산 해리** 경로임을 명시.
    - **fig2b_radical_rate (신규)**: O, OH, HO2, H+ rate budget. Top-10 contributors per panel, median+SG smoothing 동일 적용. H+ panel은 5% 임계 (다른 panel 1%).
  - **`gen_all_figures.py` 코드 변경**:
    - `RH80_RATIOS`: HONO_NO2 → 0.10 전 voltage
    - `MT_BC_CASES`: `[('three_film', 'three_film', None, 0.01)]` 추가 (이전 빈 리스트)
    - `gen_fig1`: bar Sim+Exp redesign
    - `gen_fig1b`: post-flux SG smoothing (window=75) 적용
    - `gen_fig1c`: 신규 함수
    - `gen_fig2`: post-flux median+SG (window=75) 적용
    - `gen_fig2b`: 신규 함수
    - `species_contribution`: MT 라벨 → reaction-style mapping
    - `FIGURE_MAP`: `'1c'`, `'2b'` 등록
    - `_preprocess_below_lod`: SG window 31 (62s) 유지 — input은 그대로, post-flux smoothing만 강화
  - **진단 분석 (이번 세션):**
    - **fig1b noise propagation diagnostic** (`Figures/test/diag_fig1b_noise.py` 신규): 단계별 CV% 추적. NO3 raw 9% → SG31 후 0.5% (×18 감소). 0.5%가 floor — C_s noise는 driving force cancellation으로 영향 미미. **SG window 31→75 post-flux smoothing이 시각적으로 가장 효과적**.
    - **NO3⁻ sink 0개 확정** (Liu 2015 mechanism): 14 sources, 0 sinks → terminal accumulator. 물리적으로 합리적 (UV photolysis 외 reduction agent 거의 없음).
    - **H2O2 9 sinks, O3 11 sinks 확인** (이전 "sink 거의 없음" 정정). 단 라디칼 (OH ~0.04 pM, HO2 ~ pM) 농도 부족으로 sink rate ~source의 0.4% → 실질 inactive. R32 (O3+NO2⁻)가 dominant O3 sink.
    - **Radical chain ignition 시간 패턴**: ~3분 peak (radical-rich), 4분 이후 NO2⁻ 빌드업 → R77 (OH+NO2⁻ k=1e10) 활성화 → OH 100× 감소 → quasi-steady (radical-poor regime). Bruggeman 2016 review의 known transient.
    - **fig1b cumulative vs fig2 dC/dt 일관성 확인**: 3.6 kV t=600s에서 MT_in (5e-9 M/s) = R32_out → bulk O3 0.4 nM 안정, dC/dt≈0, but cumulative ↗ 5e-9·600s = ~3 µM throughput. **Mass conservation 정확히 작동**.
  - **현재 fit (HONO=0.10, three_film, H2O2/O3=0.003, HONO2/N2O5=0.83)**:
    | V | NO3⁻ s/e | NO2⁻ s/e | H2O2 s/e | pH s/e |
    |---|---|---|---|---|
    | 2.6 | 42.9/32.6 (×1.31) | 19.9/0 (over) | 6.43/4.76 (×1.35) | 4.25/5.09 |
    | 3.2 | 74.5/62.7 (×1.19) | 14.7/3.58 (×4.1) | 19.8/11.2 (×1.77) | 4.09/3.61 |
    | 3.6 | 78.5/70.4 (×1.12) | **21.7/20.7 (×1.05) ✓** | 25.1/16.3 (×1.54) | 4.05/3.25 |
  - **잔존 문제**:
    - 라디칼 (OH/HO2/O) 농도 underestimate — gas-phase OH/HO2 input 누락 가능성 (해결 후보: Sakiyama 비율 ~10⁻⁴ × O3로 추정 입력)
    - pH gap +0.4-0.8 단위 (charge balance, sim H+ < exp H+ at 3.2/3.6 kV)
    - NO2⁻ trend (uniform HONO 한계) — 2.6 kV overshoot, 3.2 kV undershoot
  - **변경/신규 파일** (이번 세션):
    - `Figures/gen_all_figures.py` 수정 (HONO=0.10, fig 1/1b/1c/2/2b, MT label, smoothing)
    - `Figures/test/diag_fig1b_noise.py` 신규 (단계별 noise propagation 진단)
    - 3 voltage figures 재생성 (`Figures/DIW results/{V}_Humid_fitting_three_film/fig*.{png,pdf}`)
    - `MEMORY.md` + `project_no2m_o3_feedback.md` 신규 + `project_reactive_uptake_bc_plan.md` 갱신 (이전 세션 마지막)
  - **다음 권장**: (1) Git commit, (2) gas-phase OH/HO2 input 추가 (라디칼 농도 보강), (3) Saline 동일 framework 적용 검증.

- 2026-04-29: **OAS gas input 진단 + Dry condition figure 생성 + fig2 acid-base speciation 검토 + Liu/Kong 2016 reference 추출**
  - **OAS gas input 진단** (`Figures/test/diag_gas_input_voltage_trend.py`, `diag_gas_raw_vs_processed.py` 신규):
    - Raw OAS vs RH80-processed 종별 변환 인자 (last 100s SS):
      | 종 | 2.6kV | 3.2kV | 3.6kV | 효과 |
      |---|---|---|---|---|
      | O₃ PROC/RAW | 0.493 | 0.647 | 0.762 | RH80 감쇠 |
      | NO₂ PROC/RAW | 4.66× | 3.41× | 3.23× | RH80 증폭 |
      | N₂O₅ PROC/RAW | 0.137 | 0.132 | 0.115 | ÷7-8 (수증기 가수분해) |
      | NO₃ PROC/RAW | 1.64× | 1.98× | 1.59× | 약한 증폭 |
    - Voltage scaling 변화 (3.6/2.6 ratio): O₃ RAW 3.30× → PROC 5.10× (×1.55 amplified), NO₂ 3.14× → 2.18× (dampened), NO₃ 0.99× → 0.96× (preserved), N₂O₅ 2.24× → 1.88×
    - **사용자 지적**: RH80 fitting에서 voltage 증가 시 O₃가 NO₂보다 더 sharp scaling — 직관(NOx-mode 전환)과 반대 방향
  - **Dry condition figure 생성** (사용자 요청, raw OAS shape 유지 + HONO/HONO2/H2O2 ratio 적용):
    - `gen_all_figures.py` Dry 분기 임시 patch (HONO=0.10·NO₂, HONO2=0.83·N₂O5, H2O2=0.003·O₃ 적용, RH80 미적용) → 3 voltage 실행 → 즉시 원복 (영구 변경 X)
    - 신규 폴더: `Figures/DIW results/{2.6, 3.2, 3.6}kV_Dry_three_film/` 각 9 figure (fig1/1b/1c/2/2b/3/4/5/6)
    - 신규 스크립트: `Figures/DIW results/gen_voltage_comparison_Dry.py`
    - 신규 figure: `fig_voltage_comparison_Dry.{png,pdf}` (Dry vs Humid_fitting vs Exp 3-bar)
    - **결과**: Dry NO₃⁻ 187/362/404 µM vs Humid 43/75/78 → **N₂O₅ ÷7 효과로 5-6× 폭증** 확인 (RH80 가수분해 감쇠가 N₂O₅-mediated NO₃⁻ 생성을 결정)
  - **fig2 acid-base speciation 처리 검토 (long discussion + 결국 원복)**:
    - 사용자 의문: HONO MT가 NO₂⁻ panel에 그대로 그려지고 f_ion 적용 안 됨. pH 의존이 어디서 들어가는지?
    - 코드 audit:
      - `_compute_single_rate` + `_get_conc` (chemistry_1d.py:750-790): rate 계산 시 [NO₂⁻] = f_ion × HONO_total로 정확히 분리된 농도 사용 ✓
      - `_apply_rate` + `_get_target_idx` (chemistry_1d.py:792-814): mass conservation은 *_total slot 단일 처리 — NO₂⁻이든 HONO든 reactant이면 dydt[HONO_total] -= rate
      - `_enforce_electroneutrality` (pde_solver.py:570-632): HONO_total 변화 후 charge balance Newton iteration으로 H⁺ 자동 보정 (Σ Ka/(H+Ka)·C_total 항)
    - **결론**: HONO MT 1 mol → HONO_total +1 → NO₂⁻ +f_ion mol, HONO 분자형 +f_mol mol, H⁺ +f_ion mol (instantaneous re-eq via electroneutrality solve)
    - 잘못된 첫 시도: `mt_flux`를 acid_form/base_form으로 split (surface f_ion) + reaction rate에 vol-avg f_ion 곱 → surface(0.71) ≠ vol-avg(0.95) f_ion으로 mass balance 깨짐
    - 사용자 정정 후 **원복**: 시뮬은 conservative variable HONO_total 단위로 풀고, NO₂⁻ 단독 budget은 derived view. **이전 fig2가 mass pool 기준 mass balance 정확** — 라벨만 misleading
    - 최종: 라벨만 reaction-style 명확화 — `'HONO(aq) → H+ + NO2-'`, `'HONO2(aq) → H+ + NO3-'`, `'H2O2(g) → H2O2(aq)'`, `'O3(g) → O3(aq)'`
    - fig2/2b/4 재생성 (Humid_fitting 3 voltage + Dry 3 voltage)
  - **NO₂⁻ voltage trend 모순 분석**:
    - 사용자 모순 정리: 액상 NO₂⁻ (0/3.58/20.74 µM, 5.8× 점프 3.2→3.6 kV) → 고전압=NOx-mode vs OAS RH80 NO₂/O₃ ratio (0.222/0.091/0.095) → 저전압=NOx-mode. 정반대.
    - 사용자 명시 제약: NO 기각 (detection limit 이하), gas-phase 1D PDE 안 함, OAS data 그대로 사용
    - 차원분석: source/sink ratio = NO₂(g)²/O₃(g) ∝ 4.75×/5.10× = **0.93×** → 시뮬에서 voltage 증가 시 NO₂⁻ 거의 평탄 (실제 sim 19.9/14.7/21.7와 일치, exp 5.8× 점프 못 잡음)
    - Humid OAS 직접 측정 외삽 (`test_rh_ratio_fit.py` 실행): HONO/NO₂ = 0.0091/0.0071/0.0066 (voltage-별 RH80 외삽) vs 우리 default 0.10 → **14× 차이** + voltage 거의 평탄. 직접 측정값 쓰면 NO₂⁻ 절대값 더 작아져 absolute level 격차만 커짐
    - 결론: **현재 framework lever 모두 소진**. NO₂⁻ 5.8× voltage 점프는 액상 chemistry only로는 자연 발생 불가. 후보:
      - (R5) Liquid surface heterogeneous chemistry (현 모델 부재) — 코드 큰 수정 필요
      - 또는 framework limit으로 paper에서 "first-order surrogate, 3.6 kV fit, voltage trend는 surface chemistry 미고려 한계"로 명시
  - **Article folder 1D 논문 review**:
    - **현규 1D (Lee 2023 CEJ 458:141425)**: 우리와 거의 동일 framework (two-film, δ_gas=10mm, δ_liq=100µm). **표면 heterogeneous chemistry 없음**. UV photolysis 7개만 추가 (Table 1), 외부 6W UV lamp (UV-C+UV-A) 사용 — 우리 setup에 부적절. NO 생성의 51%가 HONO + hν → OH + NO photolysis. Sim ×0.25 to fit experiment (4× over-prediction, 우리도 비슷한 NO₃⁻ over)
    - **Liu/Kong 2016 (Sci Rep 6:23737, our reaction set 원본)**: SDBD+DIW, V_pp=11kV, **power 0.05 W/cm² 고정**, air gap **L_g=0.1~2cm 변수**. Critical L_g=0.5cm: S1 (MT dominant) vs S2 (liquid chemistry dominant). 우리 setup L_g=10mm=1cm = **S2 영역**. 실험 검증: pH 0.6 unit gap, H₂O₂ 1.2~7.2× over, nitrite/nitrate 1.6~2.9× over (우리 three_film+ratio fit 후 NO₃⁻ 1.12-1.31×, H₂O₂ 1.2-1.8×로 더 좋음)
  - **Photolysis 처리 방식 정리** (UV photolysis 추가 검토용 reference): Q = ∫ Φ σ F dλ (effective 1차 rate constant, s⁻¹). hν는 reactant 안 적고 k_eff에 흡수. YAML 'irr' 형식으로 추가 가능. Spatial uniform 또는 Beer-Lambert. **우리 setup에 외부 UV 없으면 적용 부적절**.
  - **Memory 작성/정리**:
    - 신규: `reference_liu2016_pathway.md` — Liu/Kong 2016 reference. 첫 작성 시 Fig 7-8 정량 % 분석 포함, 사용자가 화살표 convention 정정 (시작단=loss 분배 vs 끝단=기여도) → NO₂⁻만 정확 재작성 후 **사용자 요청으로 Fig 7-8 분석 모두 제거** + OH 농도 reference도 단위 환산 부정확으로 제거. 최종: 시스템 setup, S1/S2 scenario, 실험 검증 결과, voltage 모순 해석만 유지
    - `MEMORY.md` 업데이트
  - **변경/생성 파일**:
    - `Figures/gen_all_figures.py`: Dry 분기 임시 patch + 원복, mt_flux split 시도 후 원복, MT 라벨 reaction-style (`HONO(aq) → H+ + NO2-` 등)
    - `Figures/test/diag_gas_raw_vs_processed.py` 신규
    - `Figures/test/diag_gas_input_voltage_trend.py` 신규
    - `Figures/DIW results/{2.6, 3.2, 3.6}kV_Dry_three_film/` 신규 폴더 (각 9 figure)
    - `Figures/DIW results/gen_voltage_comparison_Dry.py` 신규
    - `Figures/DIW results/fig_voltage_comparison_Dry.{png, pdf}` 신규
    - `Figures/DIW results/{V}_{Humid_fitting, Dry}_three_film/fig2/2b/4` 재생성 (라벨 정확화)
    - `memory/reference_liu2016_pathway.md` 신규 (정정 거치며 보수적 내용으로 수렴)
    - `memory/MEMORY.md` 업데이트
  - **다음 단계 후보**:
    1. Liquid surface heterogeneous chemistry 추가 (NO₂⁻ voltage trend 재현) — 코드 큰 수정
    2. R11 (NO₃ + H₂O₂ → HO₂ + H⁺ + NO₃⁻) 우리 model에서 활성화 정도 확인 (Liu에선 H₂O₂ sink 86-92% dominant)
    3. Paper storyline 갱신 — voltage trend는 framework limit으로 명시, electrolyte-selective hook 약화 인정
    4. Saline 확장 검증

- 2026-05-04: **HONO/NO₂ voltage-specific fine-tune sweep + 3 voltage figure 재생성 (HONOvar)**
  - **목적**: 이전 uniform HONO/NO₂=0.10이 3.6 kV NO₂⁻만 매칭하고 2.6/3.2 kV는 over-prediction → voltage별 ratio 최적화로 NO₂⁻ trend 재현 시도.
  - **이전 sweep 참조 (2026-04-26 SWEEP1, NO₂⁻ µM)**:
    - HONO=0.007/0.030/0.070/0.100 → 2.6kV: 0.045/2.567/12.447/19.943, 3.2kV: 0.054/0.328/7.447/14.683, 3.6kV: 0.051/0.740/11.788/21.666
    - exp NO₂⁻: 2.6kV=0, 3.2kV=3.58, 3.6kV=20.74
  - **Log-linear interp 추정**: 2.6kV ≤0.005, 3.2kV ≈0.057, 3.6kV ≈0.097
  - **Fine-tune sweep 신규** (`Figures/test/test_hono_finetune.py`, 9 sims, ~12min):
    - 2.6kV {0.000, 0.001, 0.003, 0.005} → NO₂⁻=0.069/0.068/0.058/0.049 (모두 baseline floor ~0.05 µM)
    - 3.2kV {0.045, 0.055, 0.060} → NO₂⁻=2.191/4.111/5.179
    - 3.6kV {0.090, 0.097} → NO₂⁻=18.377/20.682
  - **Best HONO/NO₂ voltage-specific 확정**:

| V | HONO/NO₂ | NO₂⁻ sim/exp | NO₃⁻ sim/exp | pH sim/exp | H₂O₂ sim/exp |
|---|---|---|---|---|---|
| 2.6 kV | **0.005** | 0.049/0.00 (+0.05 floor) | 36.94/32.63 (×1.13) | 4.43/5.09 | 5.16/4.76 (×1.08) |
| 3.2 kV | **0.055** | 4.11/3.58 (+15%) | 71.77/62.74 (×1.14) | 4.13/3.61 | 19.59/11.21 (×1.75) |
| 3.6 kV | **0.097** | 20.68/20.74 (−0.3%) ✓ | 78.03/70.42 (×1.11) | 4.05/3.25 | 25.03/16.25 (×1.54) |

  - **2.6 kV NO₂⁻ floor ~0.05 µM**: HONO=0 일 때 sim NO₂⁻=0.069 → R19 (2NO₂+H₂O→HONO+HONO₂) + R95 (N₂O₄+H₂O 가수분해) 두 source가 baseline 결정. exp=0 도달 불가 (실험 detection limit 이하 가정).
  - **NO₂⁻ trend 매칭 결과**: voltage 단조 증가 0.05 → 4.11 → 20.68 µM, 실험 0/3.58/20.74과 일치. 이전 uniform 0.10에서는 19.9/14.7/21.7 (단조 깨짐).
  - **NO₃⁻/H₂O₂는 거의 불변**: HONO/NO₂ ratio 변경이 N₂O₅-mediated NO₃⁻에 영향 미미 (HONO direct dissolution path 미미). H₂O₂는 HONO 무관.
  - **pH gap +0.4 ~ +0.8 unit 잔존**: NO₃⁻ ×1.1 정도라 H⁺ 부족, charge balance gap 미해결.
  - **변경 파일**:
    - `Figures/test/test_hono_finetune.py` (신규, voltage-specific sweep, 9 sims)
    - `Figures/test/hono_finetune_results.txt` (신규)
    - `Figures/gen_all_figures.py` — `RH80_RATIOS[V]['HONO_NO2']`을 voltage-specific으로 변경 (0.005/0.055/0.097), `--label-suffix` argparse 추가, Fig 1 suptitle 동적 ratio 표시
    - `Figures/DIW results/gen_voltage_comparison_HONOvar.py` (신규, gen_voltage_comparison_HONO010.py 기반)
    - `Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film_HONOvar/` (3 폴더, 각 9 figure + cache)
    - `Figures/DIW results/fig_voltage_comparison_HONOvar.{png, pdf}` (신규)
  - **이전 폴더 보존**: `_three_film/` (HONO=0.10 uniform) 그대로 유지. 비교 가능.
  - **다음 단계 후보**:
    1. 3.2kV 추가 fine-tune (HONO=0.052) — 현재 +15% over를 더 줄이려면
    2. NO₃⁻ ×1.1 audit — 더 정밀 fit 위해 HONO₂/N₂O₅ ratio 재조정 검토
    3. pH gap 진단 (charge balance 닫힘 여부 재확인)
    4. Saline 동일 voltage-specific HONO 적용 검증

---
<!-- UPDATE RULE:
작업 단위가 완료될 때마다 즉시 이 파일을 갱신할 것 (세션 종료를 기다리지 않는다).
1. Current Status 섹션의 WORKING/BROKEN/NOT TRIED 갱신
2. Pending Tasks 우선순위 조정
3. Key Decisions에 새 결정 추가
4. Session History에 날짜+요약 한 줄 추가
사용자가 터미널을 그냥 닫아도 CLAUDE.md는 이미 최신 상태여야 한다.
-->

- 2026-05-04: **사용자 지적 3가지 문제 진단 + HONO/NO2 voltage-specific fine-tune**
  - **PART 1 — HONO/NO2 voltage-specific sweep**:
    - 이전 uniform HONO/NO2=0.10이 3.6kV NO2⁻만 매칭, 2.6/3.2kV over-prediction
    - `Figures/test/test_hono_finetune.py` 신규 (9 sims, ~12min). voltage-specific sweep 결과:
      - 2.6 kV → **0.005** (NO2⁻=0.049 µM, baseline floor R19/R95 ~0.05 µM, exp 0)
      - 3.2 kV → **0.055** (NO2⁻=4.11 µM, exp 3.58, +15%)
      - 3.6 kV → **0.097** (NO2⁻=20.68 µM, exp 20.74, −0.3%) ✓
    - `gen_all_figures.py` 수정: RH80_RATIOS HONO_NO2 voltage-specific, `--label-suffix` argparse 추가 (기존 폴더 보존), Fig 1 suptitle 동적 ratio 표시
    - `Figures/DIW results/gen_voltage_comparison_HONOvar.py` 신규 (HONO010 기반)
    - `Figures/DIW results/{2.6,3.2,3.6}kV_Humid_fitting_three_film_HONOvar/` 3 폴더 × 9 figure 신규 + comparison figure
    - 결과: 2.6kV NO2⁻ 19.9→0.05, 3.2kV 14.7→4.11, 3.6kV 21.7→20.68 µM. **NO2⁻ voltage trend 단조 회복**
  - ---
  - **PART 2 — fig5에 NO2⁻ panel 추가 (영구)**:
    - `gen_all_figures.py`:
      - `SPATIAL_SPECIES`에 `('HONO_total', 'NO2-', 'uM', 1e6)` 추가
      - `gen_fig5`에 pH-dependent speciation 로직 (`_SPECIATE_IONIC` dict): NO2⁻ = HONO_total × Ka/(H+ + Ka), per-cell snapshot별. HO2⁻/O2⁻도 향후 자동 적용
    - 3 voltage 폴더 fig5 재생성. 7-panel layout (pH + NO3⁻ + NO2⁻ + O3 + H2O2 + OH + HO2)
  - ---
  - **PART 3 — 사용자 지적 3가지 문제 진단**:
    - **문제 1**: 2.6 kV fig1c의 O3 시계열 noise (Henry 평형 도달 시점부터 oscillation, 다른 voltage는 smooth)
    - **문제 2**: fig2b_radical_rate의 O atom panel rate budget mismatch (R28/R27 ±2e-14 mirror, ΔC/Δt ≈ 0)
    - **문제 3**: fig5_spatial 3.6 kV의 O3 비물리적 U-shape (surface 7e-7 → 0.2mm 8e-22 → 1.3mm zero/atol-noise → 4mm 1.83e-10 peak → deep decay)
  - ---
  - **문제 1 진단 (Phase A → B → B5 → 결론)**:
    - **Phase A** (`Figures/test/diag_o3_henry_oscillation.py`): driving force는 모든 voltage 0.7+ 유지 (Henry eq 절대 도달 안 함, PDE 액상 저항 dominant). 2.6 kV만 MT flux 8e-9 plateau saturate, 3.2/3.6 kV는 monotonic 증가
    - **τ_chem (R32) 비교**: 2.6 kV = 43s (NO2⁻ 0.05µM), 3.2 kV = 0.53s (NO2⁻ 3.77µM), 3.6 kV = 0.10s (NO2⁻ 19.97µM). τ_MT ~573s 모든 voltage 동일. **2.6 kV만 underdamped 영역**.
    - **Phase B1 초기 시도** (`diag_o3_phase_b1_r32_off.py`): R32 disable로 baseline와 결과 동일. 가설 기각으로 보임.
    - **Phase B5** (`diag_o3_phase_b5.py`): BDF atol 1e-15 → 1e-18, rtol 1e-6 → 1e-9, max_step 1.0 → 0.1, 5 cases. **detrended std 4 자리 소수점까지 완전 동일** (10.740, 24.125). nfev ×3.5 cost 증가. **Numerical artifact 완전 배제 확정**.
    - **★ Phase B redo** (`diag_o3_phase_b_redo.py`): 이전 Phase B1/B2 disable이 `chem._rxn_data['k']=0`만 수정하고 `_precompute_numba_arrays()` 호출 안 해서 **Numba JIT compute_rates_batch가 원본 k 사용** 버그 발견. 수정 후:
      - baseline bulk 2.06 nM, **R32 OFF bulk 298.6 nM (145× 증가, R32 매우 active 확인)**, R22-R32 ALL OFF 471 nM
      - **detrended std: baseline 24.1% → R32 OFF 2.7% (10× 감소)** — R32 가설 부활 확정
    - **결론**: **R32 ↔ NO2⁻ Lotka-Volterra-type chemistry-driven limit cycle**. 2.6 kV에서 NO2⁻ 작아 R32 weakly damped → underdamped limit cycle. 3.2/3.6 kV는 R32가 1000~5000× 강한 damping → overdamped, smooth. **물리적 chemistry feature** (numerical 아님), input smoothness와 무관.
    - 사용자 비판: "smooth input + saturating system → fluctuation 비물리적" — linear system에서만 성립. R32 (2-species product nonlinear) limit cycle은 constant input에서도 자연 발생.
    - **Fix 미적용** (mechanism 인정으로 마무리, plotting smoothing 등 옵션 보류)
  - ---
  - **문제 3 진단 (Phase C → D → E → 부분 결론)**:
    - **Phase C** (`diag_o3_phase_c_diff_chem_off.py`): chem_off (k=0 모든 reactions) vs diff_off (D=0 모든 species) vs baseline 비교
      - **chem_off**: diffusion-only erfc analytical 거의 정확히 일치 (4mm 1.93e-8 vs erfc 예측 2.5e-8). **SG diffusion solver 정상 작동 확인**.
      - **diff_off**: 모든 deep cells 5.88e-15 균일 (atol-floor + chemistry 자체 trace 생성). 작은 효과.
      - baseline: surface→0.1mm chemistry-dominated dip → 4mm peak (1.83e-10) → deep decay.
    - **Phase D** (`diag_o3_phase_d_atol_sweep.py`, 8 sims): atol 1e-15 / 1e-20 / 1e-25 / 1e-30 × {chem_off, baseline}.
      - **atol을 10¹⁵× 강화해도 baseline 4mm O3 = 1.83e-10 그대로** (음수 cells 일부 정상화 -1e-21 → +3e-26 정도만)
      - chem_off도 4mm = 1.93e-8 그대로
      - **atol-floor 가설 완전 기각**. baseline 4mm peak는 BDF의 정확한 PDE 해.
      - cost: nfev ×11.5 (atol 1e-30), wall ×13배. 매우 비쌈.
    - **Phase E** (`diag_o3_phase_e_cell40_dydt.py`): cell 40 (z=4.07mm) dydt budget 시간별 분해.
      - Top reactions affecting O3 at cell 40: R32 (peak 9.3e-13), R25 (3.1e-14), R27 (3.1e-14), R28 (2.8e-15), R20 (5.6e-16)
      - **diff_in이 dominant positive source** (peak 1.81e-12 M/s at t=420s). chem_net 소량 negative (R32 lifetime 850s at 4mm with NO2⁻=2.35nM)
      - cell 40 history: t=60s 2.93e-15 → t=180s 1.21e-12 (60×) → t=420s 1.63e-10 peak → t=600s 4.45e-11 decay
      - chain mechanism: surface → cell 1 → ... → cell 39 → cell 40 (diffusion 누적 over 480s)
    - **부분 결론**: BDF/atol/diffusion solver/chemistry batch 모두 numerical level은 정확. baseline 4mm = 1.83e-10은 우리 PDE 해 자체. 단:
      - mid cells (0.5-2mm)에서 R32 lifetime <2s vs diffusion transit ~1000s timescale gap에도 O3가 deep까지 leak — 단순 reactive-penetration estimate와 1-2 orders 불일치
      - 가능 원인: (A) early-time free diffusion (NO2⁻ build-up 전) + late-time slow consumption, (B) SG scheme이 extreme gradient 10¹⁵에서 미세 numerical artifact, (C) 미발견 subtle bug
      - **미해결**: 사용자 비판 "10¹⁵ in 0.2mm 비물리적" 정량 정당화 부족. Reference solver (FD+RK4) 비교가 결정적 verification으로 남음
    - 사용자 비판 "physical 우기지 말고 debug": 인정. atol/diffusion/chemistry 모두 검증 통과했고 BDF가 정확히 추적하는 PDE 해이므로 numerical artifact 아님 확정. 그러나 simple analytical estimate와 차이가 크고 transient 해석만으로 정당화 약함 — 진짜 physical인지 1-2 orders 잔여 의문.
  - ---
  - **문제 2 (O atom rate budget) 진단 미진행** — Phase E 시점에서 사용자가 문제 3로 우선 이동. 이전 진단 (atol 1e-15 floor 한참 위가 1e-18 농도)만 정리됨.
  - ---
  - **변경/생성 파일** (이번 세션):
    - `Figures/test/test_hono_finetune.py` 신규 (HONO sweep)
    - `Figures/test/hono_finetune_results.txt`
    - `Figures/test/diag_o3_henry_oscillation.py` 신규 (Phase A)
    - `Figures/test/diag_o3_phase_b1_r32_off.py` 신규 (Phase B1, Numba bug)
    - `Figures/test/diag_o3_phase_b2_b4.py` 신규 (Phase B2/B4, Numba bug)
    - `Figures/test/diag_o3_phase_b_redo.py` 신규 (Phase B redo, fixed)
    - `Figures/test/diag_o3_phase_b5.py` 신규 (Phase B5 BDF atol sweep)
    - `Figures/test/diag_o3_phase_c_diff_chem_off.py` 신규 (Phase C)
    - `Figures/test/diag_o3_phase_d_atol_sweep.py` 신규 (Phase D atol sweep)
    - `Figures/test/diag_o3_phase_e_cell40_dydt.py` 신규 (Phase E cell 40 budget)
    - `Figures/test/fig_diag_o3_oscillation.{png,pdf}`, `fig_diag_o3_b1.{png,pdf}`, `fig_diag_o3_b2b4.{png,pdf}`, `fig_diag_o3_b_redo.{png,pdf}`, `fig_diag_o3_b5.{png,pdf}`, `fig_diag_o3_c.{png,pdf}`, `fig_diag_o3_d.{png,pdf}`, `fig_diag_o3_e_cell40.{png,pdf}` 신규
    - `Figures/test/diag_o3_oscillation.txt`, `diag_o3_b1.txt`, `diag_o3_b2b4.txt`, `diag_o3_b_redo.txt`, `diag_o3_b5.txt`, `diag_o3_c.txt`, `diag_o3_d.txt` 신규
    - `Figures/gen_all_figures.py`:
      - `RH80_RATIOS` HONO_NO2 voltage-specific (0.005/0.055/0.097)
      - `--label-suffix` argparse + `out_folder` 적용
      - `gen_fig1` suptitle 동적 HONO/NO2 표시
      - `SPATIAL_SPECIES`에 NO2⁻ 추가
      - `gen_fig5`에 `_SPECIATE_IONIC` dict + speciation logic
    - `Figures/DIW results/gen_voltage_comparison_HONOvar.py` 신규
    - `Figures/DIW results/fig_voltage_comparison_HONOvar.{png, pdf}` 신규
    - `Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film_HONOvar/` 3 폴더 신규 (각 9 figure + cache)
  - ---
  - **확인된 코드 사실**:
    - **Chemistry는 모든 N_z=49 cells에 적용** (`compute_rates_batch` 49-cell loop, `pde_solver.py:894-895`)
    - **Diffusion은 SG (Scharfetter-Gummel) finite-volume flux**: `J_{j+1/2} = D/h × (B(α)·c_j − B(−α)·c_{j+1})`. Poisson OFF로 α=0이라 표준 Fickian flux로 환원
    - **atol species-specific 이미 적용**: OH/O-/O3-/HO3/NO3는 atol_base × 0.01 = 1e-17 (`pde_solver.py:1000-1005`)
    - **Numba precompute 필수**: chemistry 수정 후 `chem._precompute_numba_arrays()` 호출 안 하면 `compute_rates_batch`가 원본 k 사용 (Phase B1 버그 원인)
  - ---
  - **다음 단계 후보**:
    1. **Reference solver 작성**: 단순 explicit FD+RK4로 동일 reaction-diffusion PDE 풀어 우리 결과와 비교 (문제 3 결정적 verification)
    2. **NO2⁻ 시공간 추적**: mid cells (0.5-2mm)의 NO2⁻ build-up 시점 분석. 문제 3의 "early-time free diffusion" 가설 정량 검증
    3. **R20 OFF 시뮬**: deep cells에서 chem_net positive (R20 forward) 영향 정량 (Phase E에서 t=60s chem_net=+3e-17 발견)
    4. **문제 2 진단**: O atom rate budget 본격 분석 (atol-tight species list에 'O' 추가 또는 QSSA)
    5. **문제 1 fix 옵션**: chemistry-driven limit cycle 시각화 — caption annotation 또는 plotting smoothing

- 2026-05-06: **Problem 1/2/3 통합 진단 (외부 AI 협업) — 핵심 narrative 정정**
  - 2026-05-04 entry 후속. 사용자 비판으로 제 LV narrative 완전 기각, mechanism 재식별.
  - ---
  - **Problem 1 (2.6 kV oscillation) — chemistry network linear analysis로 LV 가설 정량 기각**:
    - 사용자 비판: R32 (mutual annihilation, dx/dt=dy/dt=-kxy) → 2-species Jacobian trace<0/det>0 → eigenvalue 항상 음의 실수부. **LV 수학적으로 불가**. 제 "underdamped regime sustained oscillation" narrative는 damped와 양립 불가.
    - **Phase H** (`Figures/test/diag_o3_phase_h_jacobian.py`): 0D chemistry Jacobian (25×25) eigenvalue 직접 계산. **2.6 kV / 3.6 kV bulk-only 및 surface 모두 Hopf candidates 0**. Most-positive Re eigenvalue: 2.6 kV bulk +4.35e-11 (essentially conservation), surface +1.11e-2 (Im=0 purely real, exponential growth only — oscillation 아님)
    - **Phase β** (`diag_o3_phase_beta_no2_clamp.py`): 2.6 kV에서 `solver.rhs` monkey patch로 HONO_total dydt=0 강제 (NO2⁻ 시간 변화 없음). 결과: bulk-only [O3] std 11.5% → 15.8% (오히려 증가). **NO2⁻은 dynamical variable 아님. R32-NO2⁻ pair는 oscillator 아님 — amplification channel만**.
    - **Phase γ** (`diag_o3_phase_gamma_pde_jacobian.py`): **Full 1D PDE Jacobian (1225×1225) eigenvalue**. 2.6 kV: Hopf 0 (top-5 모두 Im=0). 3.6 kV: Hopf candidates 4 (period 5811s = 97분, growth τ 774s — 시뮬 600s 한 cycle 못 끝냄, visible 안 보임). **2.6 kV oscillation은 chemistry 자연 limit cycle 아님 — 0D + 1D PDE 모두 Hopf 0**.
    - **Phase Radau** (`diag_o3_phase_radau.py`): scipy BDF → Radau (5차 implicit RK). 4mm peak ratio 1.029× (3% 차이만). **ODE solver method 무관**.
    - **결론 정정**: **forced response from gas BC residual variation (raw OAS ±2.4% 변동) + R32 nonlinear amplification**. 2.6 kV는 bulk 농도 작아 relative noise 큼. 다른 voltage는 chemistry damping 강해 absorbed.
  - ---
  - **Problem 2 (O atom rate budget mismatch) — 진단 + plot fix 적용**:
    - **Phase η** (`diag_o3_phase_eta_species_atol.py`): 모든 25 species의 bulk-only avg vs atol 직접 비교. **8개 species (32%) atol-band noise 영역**:
      - H, H2, NO, NO3, N2O, N2O3 (대부분 voltages), 2.6 kV에서 O atom (1/3 voltages)
      - 음수 농도 빈출 (-1e-19 ~ -1e-22 사이 numerical zero)
      - N2O는 initial trace 1e-30 그대로 (chemistry inactive)
    - **즉 O atom뿐 아니라 25 species 중 32%가 numerical noise level**. fig2b의 O atom rate budget (R28/R27 ±2e-14 mirror, ΔC/Δt≈0)은 atol-band 양쪽 noise 곱한 garbage.
    - **★ Fix (δ) 적용**: `gen_all_figures.py:124-126` `TARGET_SPECIES_RADICAL: ['O', 'OH', 'HO2', 'H+'] → ['NO3', 'OH', 'HO2', 'H+']`. NO3 radical은 H_cc=44 깊은 침투 + R93 catalyst 역할로 의미 있는 budget. 3 voltage HONOvar 폴더 fig2b 재생성 완료.
  - ---
  - **Problem 3 (3.6 kV 4mm O3 peak) — 모든 numerical knob 무효, mechanism 미식별**:
    - **Phase F** (`diag_o3_phase_f_o_atom_removal.py`): O atom 6 reactions OFF (R20, R73, R106-R109) + Numba precompute 호출. 4mm peak 0.2% 변화. 외부 AI 첫 가설 (noise-fed R20 via O atom × O2 reservoir) 기각.
    - **Phase G** (`diag_o3_phase_g_dz_sweep.py`): dz_min sweep 1, 5, 10, 20 µm. 4mm peak max/min ratio 1.08 (8% 변동). **grid-independent (numerical leak via stiff Jacobian conditioning 가설 기각)**.
    - **Phase α** (`diag_o3_phase_alpha_wall_dump.py`): 49 cells full dump. wall (z=0.4-1.3mm) atol-noise (음수 -1e-19까지). Deep zone (z=1.4mm 이후) **monotonic 증가** to peak at cell 39 (z=3.63mm)=2.6e-10. cell-by-cell smooth gradient (internal source 형태).
    - **Phase J** (`diag_o3_phase_j_wall_qss.py`): QSS reactive-penetration 분석. NO2⁻(z) profile로 ∫₀^z dz/λ(z) 직접 적분 (∫=15 / 81 / 235 for 2.6/3.2/3.6 kV). QSS 예측 c(4mm)/c(0) = e⁻²³⁵ ≈ **5×10⁻¹⁰³**. 관측 sim/QSS = **2×10⁹⁸** at 4mm. **deep cells 90-100 orders over physical**.
    - **Phase I** (`diag_o3_phase_i_external_verify.py`, 외부 AI 제안): 6-case verification:
      - **Case I (R20+R35 OFF + initial OH=trace)**: 4mm peak 1.83e-10 → 1.92e-10 (×1.05). **chemistry source 0인데도 peak 살아남음 — 외부 AI Case I 결과 우리 setup에서 정확 재현**.
      - Case I × N_z 25/49/57: ±11% 변동 (grid-independent)
      - **atol_tight (1e-30 + max_step 0.1s)**: 4mm peak 0.997× (변화 없음, 외부 AI atol fix 추천 무효)
      - **trace_1e50** (chemistry RHS clip floor 1e-50): 4mm peak 0.997× (외부 AI atol-trace floor mismatch 가설 기각)
    - **Phase Radau**: BDF → Radau도 4mm peak 1.029× (solver method 무관)
    - **결론**: **Numerical artifact 확정** (Phase J 90+ orders over). **그러나 atol/trace/dz/chemistry sources/solver method 모두 무관**. Mechanism 미식별. 외부 AI atol-trace floor mismatch 가설도 정량 기각. 진짜 source 후보 (모두 미검증):
      1. SG flux scheme의 extreme gradient (10¹⁵ ratio across 0.2mm) round-off
      2. BDF Newton inner iteration의 numerical limit at stiff coupling
      3. Float64 cumulative error
      4. Initial perturbation (y0 *= 1+1e-6 noise) seeding deep cells × chemistry
  - ---
  - **fig5_spatial visualization 개선**:
    - X-axis log scale 적용 (gen_all_figures.py `gen_fig5`에 `ax.set_xscale('log')`). wall structure (0.01-1mm 구간) fine resolution.
    - **atol=1e-30 fig5 재생성** (`regen_fig5_atol1e30.py`): 3 voltage `_HONOvar_atol1e30/` 폴더. 결과:
      - **Type 1 (atol-bound)**: NO3⁻, NO2⁻, H2O2, HO2 deep floor 4-6 orders 더 깊게 (10⁻¹¹ → 10⁻¹⁵ µM 등)
      - **Type 2 (initial seed, atol 무관)**: pH=7.0 (deep, initial H+=1e-7), OH ~10⁻¹³ M (initial 1e-12 seed)
      - **Type 3 (mechanism 미식별)**: O3 4mm peak, HO2 1mm wave — atol/initial 둘 다 무관
    - **사용자 직관 정확**: OH plateau at 1e-13 (atol-tight 1e-32보다 19 orders 위)이 atol 가설로 설명 안 됨 → initial OH=1e-12 seed의 잔재 (build_initial_condition `pde_solver.py:854`)
  - ---
  - **Initial conditions 분석** (`build_initial_condition` line 835-863):
    - Explicit seeds (모든 cells): O2=2.5e-4 M, N2=5e-4 M, H+=1e-7 (pH=7), OH-=1e-7, **OH=1e-12 (radical seed)**
    - 나머지 모든 species: trace=1e-30
    - O2/N2: air-saturated water 가정 (atmosphere Henry 평형, standard plasma-liquid 모델 가정)
    - 단 **O2 huge reservoir (2.5e-4 M, 모든 cells, 거의 무변화)**가 deep cells에서 R10/R20/R25 등 O2-involved reactions의 fictitious source 가능성 (Phase F R20만 OFF로 부분 검증, R10 등 미검증)
  - ---
  - **외부 AI 협업 결과**:
    - Case I 제안 → numerical leak 결정적 확증 (chemistry source 0에도 peak 살아남음)
    - atol-trace floor mismatch 가설 → 우리 검증 → **기각** (외부 AI 추천 fix 4mm peak에 영향 없음)
    - Reference solver (FD+RK4) 제안 → 미진행
  - ---
  - **변경/생성 파일** (이번 세션):
    - `Figures/test/diag_o3_phase_f_o_atom_removal.py` (Phase F)
    - `Figures/test/diag_o3_phase_g_dz_sweep.py` (Phase G)
    - `Figures/test/diag_o3_phase_alpha_wall_dump.py` (Phase α 49 cells dump)
    - `Figures/test/diag_o3_phase_h_jacobian.py` (Phase H 0D Jacobian)
    - `Figures/test/diag_o3_phase_beta_no2_clamp.py` (Phase β NO2⁻ clamp)
    - `Figures/test/diag_o3_phase_i_external_verify.py` (Phase I 외부 AI Case I + atol/trace)
    - `Figures/test/diag_o3_phase_j_wall_qss.py` (Phase J QSS analysis)
    - `Figures/test/diag_o3_phase_gamma_pde_jacobian.py` (Phase γ 1225-dim PDE Jacobian)
    - `Figures/test/diag_o3_phase_radau.py` (Phase Radau)
    - `Figures/test/diag_o3_phase_eta_species_atol.py` (Phase η 25 species atol audit)
    - `Figures/test/regen_fig5_atol1e30.py` (atol=1e-30 fig5)
    - `Figures/test/regen_fig5_seedmin.py` (initial seeds → trace fig5, in progress)
    - `Figures/test/fig_diag_o3_*.{png, pdf}`, `diag_o3_*.txt` (각 Phase 산출)
    - `Figures/gen_all_figures.py`:
      - `TARGET_SPECIES_RADICAL: 'O' → 'NO3'` (line 124-126)
      - `gen_fig5`에 `ax.set_xscale('log')` 추가 (line 1073-1074)
    - `Figures/DIW results/{V}_Humid_fitting_three_film_HONOvar_atol1e30/fig5_spatial.{png,pdf}` (3 voltage)
    - `Figures/DIW results/{V}_Humid_fitting_three_film_HONOvar_seedmin/fig5_spatial.{png,pdf}` (3 voltage, in progress)
    - `Figures/DIW results/{V}_Humid_fitting_three_film_HONOvar/fig2b_radical_rate.{png,pdf}` (재생성)
  - ---
  - **종합 상태**:
    | Problem | 진단 상태 | Fix |
    |---|---|---|
    | 1 (oscillation) | ✓ chemistry network linear analysis로 LV 기각, **forced response from gas BC variation 확정** | 미적용 |
    | 2 (O atom budget) | ✓ atol-band noise (32% species 동일 문제) | ✓ plot fix (O→NO3) |
    | 3 (4mm peak) | ✓ Numerical artifact (QSS 90+ orders over), ✗ mechanism 미식별 | 미적용 |
  - ---
  - **다음 단계 후보**:
    1. **Reference solver (FD+RK4)**: Problem 3 mechanism 식별 결정적. 미진행.
    2. **Initial seed minimal fig5**: O2/N2/OH=trace로 변경 후 deep floor 변화 검증 (in progress)
    3. **O2-involved reactions sweep**: R10, R20, R25 등 모두 OFF로 deep peak 영향 검증
    4. **External AI 다음 round**: Phase I 결과 (atol-trace fix 무효) 보내서 다른 mechanism 가설 요청
    5. **Bottom BC 변경 시도** (no-flux → absorbing c=0): mass trap 효과 검증
  - ---
  - **Article folder 논문들의 bottom BC 일관성 확인**: 모든 plasma-liquid 1D/2D 논문 (Liu 2015, Liu 2016, Liu 2017 saline, Liu 2021, Lee 2023 현규, Heirman 2025) bottom BC = **no-flux (closed wall)** 사용. Liu 2015만 명시 (Γ=0, "100s 동안 deep cells 도달 안 함" 가정). 우리 600s 시뮬은 이 가정 위반 — 우리 numerical 해가 **Liu 2015 가정 violate** 하는 상태. 정통 BC지만 deep cells에 mass trap 부작용.

- 2026-05-07: **fig5 deep cells horizontal floor 분류 + 영구 default 변경 (★ 중요 결정)**
  - 2026-05-06 entry 후속. Problem 3 (deep cells anomaly) 후속 진단 + 영구 fix 적용.
  - ---
  - **사용자 지적 1 — fig5 deep cells horizontal floor 분류**:
    - 사용자가 fig5에서 일부 species가 특정 depth부터 일정 농도로 horizontal하게 유지되는 점 지적
    - 검증된 분류 (verification 완료):
      - **Type 1 (atol-band)**: NO3⁻, NO2⁻, H2O2, HO2 등 trace species. atol=1e-30로 변경 시 4-6 orders 더 깊어짐 (atol-floor 효과)
      - **Type 2A (initial pH=7 seed)**: H+/OH- 1e-7. atol 무관, deep cells 도달 못 한 영역 그대로 유지
      - **Type 2B (initial OH=1e-12 seed)**: OH plateau ~1e-13 M (initial 1e-12에서 약간 drain). atol 무관 (atol 1e-30로도 1e-13 stuck → atol 가설 결정적 기각)
      - **Type 3 (mechanism 미식별)**: O3 4mm peak, HO2 1mm wave (Phase D/F/G/I/Radau 모든 numerical knob 무관)
    - 신규: `Figures/test/regen_fig5_atol1e30.py` (3 voltage), `Figures/test/regen_fig5_seedmin.py` (3 voltage), `Figures/test/diag_o3_phase_eta_species_atol.py`
  - ---
  - **사용자 지적 2 — fig5 panel (f) OH cliff (single-cell negative spikes)**:
    - 사용자: 1분 라인 cell 37 (z=2.88mm), 8분 라인 cell 47 (z=9.04mm) 단 한 cell 음수 spike. 다른 시간엔 다른 cell.
    - 첫 가설 (random initial perturbation, seed=42): pde_solver.py:858-861의 `y0 *= (1 + 1e-6 * rng.standard_normal)` → cell-specific deterministic random sign이 atol-band sign-flip 시드라 의심
    - 검증 (`regen_fig5_no_perturb.py`): perturbation 완전 제거 후에도 cell 37, 47 동일하게 음수 → **Random perturbation 가설 정량 기각**
    - 두 번째 가설 (grid resolution): dz_min sweep test
      - dz_min=5µm (default, stretch 1.12) → cell 47 음수
      - dz_min=1µm → 음수 cells 0개 (Phase G dz=1µm 결과와 일관)
      - **Mechanism 확정**: SG flux scheme의 face-by-face round-off가 large dz cells (default stretch 1.12에서 dz_max 1028µm)에서 atol-band 농도의 sign flip 유발
  - ---
  - **★ 사용자 결정 — 영구 default 변경 (★)**:
    1. **`pde_solver.py:858-861` random perturbation 블록 제거**: y0 *= (1+1e-6 random) 영구 삭제. uniform-trace IC 의도와 충돌.
    2. **`gen_all_figures.py:48` STRETCH = 1.12 → 1.02**:
       - N_z: 49 → 188 cells
       - dz 범위: 5-1028µm → 5-199µm (deep cells 5× smoother)
       - cell-specific 음수 spike 해결 main driver
    3. **`config_1d.py:327` ODE_CONFIG.atol = 1e-15 → 1e-20**:
       - atol-band noise zone 5 orders 축소
       - Phase D 측정: cost ×1.5 (vs default), atol=1e-25 대비 ×8 빠름
    4. **`pde_solver.py:842-856` build_initial_condition full seedmin**:
       - O2 = 2.5e-4 → trace 1e-30
       - N2 = 5e-4 → trace 1e-30
       - OH = 1e-12 (radical seed) → trace 1e-30
       - H+/OH- = 10⁻⁷ (pH=7, pH-physical) 유지
       - Cl⁻ = 0.154 (saline mode only) 유지
       - **Side effect**: chemistry network 일부 변화 — air-saturated O2/N2 reservoir 없어 R10/R20/R25 등 reactions의 deep-cell behavior 다름. **pH 결과 4.05 → 3.47 (0.6 unit acidic 변화) 측정됨.**
  - ---
  - **시뮬 wall time 측정** (3.6 kV):

| Setup | N_z | dz 범위 | atol | Wall | nfev | pH (final) |
|---|---|---|---|---|---|---|
| Default 이전 (stretch 1.12, atol 1e-15, with seeds) | 49 | 5-514 µm† | 1e-15 | 42 s | 25,636 | 4.05 |
| stretch 1.02 + atol 1e-30 + seedmin | 188 | 5-107 µm | 1e-30 | >1 hour | not converged | (kill) |
| stretch 1.02 + atol 1e-25 + seedmin | 188 | 5-107 µm | 1e-25 | 1771 s (29.5분) | 223,081 | 3.47 |
| **★ stretch 1.02 + atol 1e-20 + seedmin (영구 default)** | **188** | **5-107 µm** | **1e-20** | **656 s (10.9분)** | **83,307** | **3.47** |

    † 시뮬 print message에서 dz_max=514µm 표시되나 직접 build 시 1028µm — overshoot trim 처리 차이로 추정.
  - ---
  - **3 voltage HONOvar_v2 생성 완료**:
    - `Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film_HONOvar_v2/` 3 폴더
    - 각 폴더 9 figures (fig1, 1b, 1c, 2, 2b, 3, 4, 5, 6) + cache
    - 기존 `_HONOvar` 폴더는 그대로 보존 (이전 default 비교용)
  - ---
  - **결과 — 영구 변경 효과 (3.6 kV fig5 직접 확인)**:
    - ✓ Panel (f) OH cliff 완전 제거 (cell-specific 음수 spike 모든 시점 0개)
    - ✓ Panel (b) NO3⁻, (c) NO2⁻, (e) H2O2, (g) HO2 모두 매끈한 monotonic profile
    - ✓ Panel (a) pH: z>3mm pH=7 horizontal (initial H+/OH- seed로 인한 정상 영역, chemistry 도달 못한 영역)
    - ✗ **Panel (d) O3 4mm deep peak 여전히 존재** (~10⁻⁵ µM): grid/atol/initial seeds/perturbation 모두 무관 (별도 mechanism)
    - ✗ Panel (g) HO2 1mm 부근 wave-like pattern 여전: 같은 mechanism class
  - ---
  - **변경/생성 파일** (이번 세션):
    - **★ 영구 코드 변경**:
      - `Ver4_1D/pde_solver.py:842-856` build_initial_condition (full seedmin, perturbation 제거)
      - `Ver4_1D/config_1d.py:327` ODE_CONFIG.atol = 1e-20
      - `Figures/gen_all_figures.py:48` STRETCH = 1.02
    - **신규 진단 스크립트**:
      - `Figures/test/regen_fig5_atol1e30.py` (3 voltage atol=1e-30 검증)
      - `Figures/test/regen_fig5_seedmin.py` (3 voltage seedmin 검증)
      - `Figures/test/regen_fig5_no_perturb.py` (random perturbation 가설 검증, 기각)
      - `Figures/test/regen_fig5_smalldz.py` (dz=1µm test, 음수 cells 0개 확인)
      - `Figures/test/test_3.6kV_full_combo.py` (영구 default verification, atol 1e-20/1e-25/1e-30 sweep)
      - `Figures/test/diag_o3_phase_eta_species_atol.py` (25 species atol 비교, Phase η)
    - **재생성 폴더**:
      - `Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film_HONOvar_atol1e30/` (atol 검증)
      - `Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film_HONOvar_seedmin/` (seedmin 검증)
      - `Figures/DIW results/3.6kV_Humid_fitting_three_film_HONOvar_seedmin_noperturb/` (perturbation 가설 검증)
      - `Figures/DIW results/3.6kV_Humid_fitting_three_film_HONOvar_seedmin_dz1um/` (dz 가설 검증)
      - `Figures/DIW results/3.6kV_Humid_fitting_three_film_HONOvar_FULL_atol1e25/` (atol 1e-25)
      - `Figures/DIW results/3.6kV_Humid_fitting_three_film_HONOvar_FULL_atol1e20/` (atol 1e-20)
      - **`Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film_HONOvar_v2/`** (★ 영구 default 적용 결과, 3 voltage × 9 figures)
  - ---
  - **종합 상태**:
    | Problem | 진단 상태 | Fix |
    |---|---|---|
    | 1 (oscillation) | ✓ chemistry network linear analysis로 LV 기각 (Phase H/γ) | 미적용 (gas BC variation forced response) |
    | 2 (O atom budget) | ✓ atto-Molar atol-band noise (32% species 동일 issue) | ✓ plot fix (O→NO3, 2026-05-06) |
    | 3 (deep cells anomaly) | **부분 해결**: cell-specific 음수 spike는 grid/perturbation/atol/seedmin 영구 변경으로 제거 (2026-05-07). 그러나 **O3 4mm deep peak는 여전히 별도 mechanism** | ★ 영구 fix 4건 적용 (cell anomaly 부분), O3 4mm peak는 미해결 |
  - ---
  - **다음 단계 후보**:
    1. **panel (d) O3 4mm peak 별도 진단**: 모든 simple knob 무관 확인됨 → reference solver (FD+RK4) 작성이 결정적. 또는 Bottom BC 변경 (absorbing c=0) 시도.
    2. **HONOvar (이전 default) vs HONOvar_v2 직접 비교 figure**: 영구 변경 효과 시각화 (pH, NO3⁻, H2O2 정량 비교)
    3. **Saline 결과 영구 default 적용 재생성**: DIW와 동일 setting으로 saline 6 cases 재시뮬
    4. **External AI 다음 round**: O3 4mm peak가 모든 시도에 무관함을 보고하고 다른 mechanism 가설 요청
    5. **memory MEMORY.md 갱신** (영구 변경 사항 + "Always verify, never speculate" rule 반영)

- 2026-05-07 (afternoon): **O3 V-shape 결정적 진단 — Phase K1~K6 + R32 OFF + D sweep + Poisson ON**
  - **fig5b 시도 (단명)**: per-cell rate × 5 time × 7 species 시각화 시도 → 너무 복잡으로 폐기. `gen_all_figures.py`에서 `gen_fig5b`/`compute_rates_snapshot_2d`/`species_contribution_2d` 모두 제거 (residual 0).
  - ---
  - **Phase K1~K6 — V-shape이 numerical artifact인지 PDE 해인지 결정 검증** (`Figures/test/diag_o3_phase_k*.py` 6개 신규):
    - K1: SG flux Gaussian manufactured solution. 중간 cells 정확, deep cells (1e−26~1e−68) underflow 영역에서 deviation. SG implementation 자체 OK.
    - K2: Pure diffusion (chemistry OFF) + 15-order step IC + BDF. **Monotonic** — chemistry 없으면 V-shape 안 형성.
    - K3: Single-species O3 toy + imposed NO2⁻(z, t) (cache interpolation). **V-shape 정확 재현** (recovery x5.7e15). multi-species coupling 무관 확정.
    - K4: K3 toy를 BDF/Radau/LSODA/RK45 4개 integrator. 모두 동일 V-shape (recovery x4.83e15) — time integrator 무관.
    - K5: SG vs simple central FD. 정확 일치 (FD/SG = 1.00). E=0이라 SG = Fickian, Bernoulli 책임 없음.
    - K6: Bernoulli function precision. α 모든 범위 machine precision OK.
    - **결론**: **V-shape은 PDE의 정확한 해**. 8 numerical scheme 조합 모두 동일.
  - ---
  - **R32 OFF 검증** (`Figures/test/diag_r32_off_v2.py`):
    - 49-cell HONOvar baseline에서 R32 (O3+NO2⁻→NO3⁻)만 disable → numba precompute → 시뮬.
    - Baseline: recovery x1.63e13 / R32 OFF: **x1.0 (monotonic 회복)**.
    - **V-shape 100% R32 단독 책임 확정**. 이전 추측 ("다른 sinks도 partner 부족으로 약함")이 검증 결과 틀림 — 12개 다른 sinks는 V-shape 형성에 기여 안 함.
  - ---
  - **사용자 의문 정량 검증** (`diag_o3_param_dump.py`, `diag_o3_surface_balance.py`, `diag_o3_deep_source_v2.py`):
    - Surface (j=0) mass balance: MT (1.13e−4 M/s) ≈ R32 sink (8.35e−5) + diff outflow (2.95e−5). **Closure 100.5%**.
    - 모든 시간에서 max R32 sink rate at cell 0 (surface). "mid에서 sink"는 시각적 인상.
    - Mid c_O3 underflow 이유: surface 100µm 안에서 mass 16 orders 소진 → mid 도달 자체가 차단.
    - Deep cells dCdt > 0 (mass gaining) — chem_net all sink, **div_diff > 0 = diffusion이 유일 source**. NO2⁻ wall ahead의 free-diffusion 잔재가 deep으로 propagate.
  - ---
  - **D sweep + dz refinement** (`diag_d_sweep_dzrefine.py`, `diag_d_hono_sweep.py`):
    - D_O3 ×{0.1, 1, 10, 100}: recovery 1→1e15→1e12→1e4. **D_O3 ×100에도 V 잔존**.
    - D_HONO_total ×{0.1, 1, 10, 100}: recovery 7.6e15→5.3e14→2.7e6→**1.0**. **D_HONO ×100에서 V 완전 제거**.
    - dz_min 0.5µm + stretch 1.20 (50 cells, surface refined): baseline과 동일 recovery x5.3e14. **mesh refinement 영향 없음**.
    - 결론: D_HONO_total이 D_O3보다 V-shape에 훨씬 sensitive.
  - ---
  - **Poisson ON 시뮬 (의외 결과)** (`diag_poisson_on.py`, `diag_poisson_no_enforce.py`):
    - `POISSON.enabled=True`로 quasi-neutrality + ambipolar coupling 기대 (D_amb = 2D+D−/(D++D−) = 3.17e−9 for H+/NO2⁻).
    - Poisson ON: V-shape **더 강해짐** (recovery 1.63e13 → 5.34e14, mid valley x100 더 깊고 narrow).
    - `_enforce_electroneutrality` monkey-patch로 OFF 추가 시뮬 → 결과 정확히 동일 (5.34e14). enforce는 dt_enforce=None이라 시뮬 끝 1번만 호출.
    - 즉 Poisson ON의 효과는 진짜 PDE level. Multi-ion 상황에서 ambipolar coupling이 V 약화 안 시킴.
    - **E field magnitude per cell dump 미실시** (다음 단계).
  - ---
  - **Mid sink 제거 옵션 비교 표**:
    | 방법 | recovery factor | V 제거 |
    |---|---|---|
    | R32 OFF | x1.0 | 완전 |
    | D_HONO_total ×100 | x1.0 | 완전 |
    | D_HONO_total ×10 | x2.7e6 | 8 orders |
    | D_O3 ×100 | x1e4 | 11 orders 잔존 |
    | dz_0.5µm refinement | x5.3e14 | 변화 없음 |
    | Poisson ON | x5.3e14 | 강화 (×33) |
    | Poisson ON + enforce OFF | x5.3e14 | 동일 |
  - ---
  - **NO2⁻ propagation front 메커니즘 (정량 확정)**:
    - D_HONO_total = 1.85e−9 ≈ D_O3 = 1.75e−9 → 비슷한 속도로 propagate.
    - NO2⁻ front 안쪽 (R32 active): O3 빠르게 0으로 (sink wall, λ_react = 3.5µm at surface).
    - NO2⁻ front 바깥쪽 (deep): sink 부재 → free-diffusion 잔재.
    - Bottom no-flux trap → deep mass 잔류.
  - ---
  - **Memory rule 추가**: `feedback_always_verify.md` 신규 — "추측 금지, 모든 답변 데이터 검증 후". MEMORY.md 업데이트. 사용자 비판 반복 ("뇌피셜말고 검증해봤음?", "해봤냐고^^")으로 강화.
  - ---
  - **변경/생성 파일**:
    - `Figures/test/diag_o3_phase_k{1..6}_*.py` (6 phase scripts)
    - `Figures/test/diag_o3_param_dump.py`, `diag_o3_surface_balance.py`, `diag_o3_deep_source_v2.py`
    - `Figures/test/diag_r32_off_v2.py` + `r32_off_v2.npz`
    - `Figures/test/diag_d_sweep_dzrefine.py` + `d_sweep_*.npz` (5 caches)
    - `Figures/test/diag_d_hono_sweep.py` + `d_hono_sweep_*.npz` (4 caches)
    - `Figures/test/diag_poisson_on.py`, `diag_poisson_no_enforce.py` + caches
    - `Figures/test/diag_sg_flux_v2.py` (face flux + mass conservation)
    - `Figures/test/plot_o3_no2_product_v2.py`, `plot_o3_rhs_terms_v2.py`
    - `Figures/test/fig_*.{png,pdf}` 약 12개 figures
    - `gen_all_figures.py` — fig5b 추가 후 제거
    - `memory/feedback_always_verify.md` 신규
    - `memory/MEMORY.md` 업데이트
  - ---
  - **다음 단계 후보** (미진행):
    1. **E_half(z, t) + ρ(z, t) dump** (Poisson ON 정량 검증)
    2. **D_HONO_total → D_amb=3.17e−9 substitution** (×1.7 ambipolar 단순)
    3. **NO2⁻ 침투 깊이 실험 비교** (Liu 2016 등)
    4. **Multi-component ambipolar (Maxwell-Stefan)** — H+/OH−/NO2−/NO3−/Cl− 동시
    5. **NO2⁻ surface generation 자체 의심** (HONO MT 너무 강한지)
    6. **Liu 2015 100s 가정 violation 영향 정량**

- 2026-05-07 (evening): **DMEM RONS 임시 프로젝트 — Deep research 리포트 작성**
  - 사용자가 `Article/JYChoi_20260429_DMEM RONS 반응 관련 레퍼런스.pptx` 추가. DIW 반응 set에 DMEM 약식 반응 추가하는 임시 과제.
  - PPTX 핵심 (Pyr+H2O2 k=2.36, Met+H2O2 k=2e-2, Cystine⇌2Cys 가역 평형, 2 references)이 **불충분**하다고 판단 → 3-agent 병렬 deep research 수행 (Web).
  - **Agent 1**: DMEM Gibco 12800017 정확 조성 (cytion/Sigma D6429/ATCC 30-2002 cross-check) + Pyruvate kinetics. Vásquez-Vivar 1997 ChemResTox: Pyr+ONOOH k=49, Pyr+ONOO⁻ k=100, **products=AcO⁻+CO2+•NO2+•CO2⁻ (radical 생성)**. Pyr+•OH k≈7e8 (Buxton). Pyr+(O3, NO2•, NO3•, O2•⁻, ¹O2) data gap.
  - **Agent 2**: Met + Cys/Cystine cascade. Met+H2O2 k=2e-2 → **τ~10⁶s, 60s 모델에서 무시 가능 (drop)**. Met+•OH=8.3e9, +O3=4e6, +ONOOH=181, +¹O2=1.7e7, +HOCl=3.4e7, **Met+NO2• ≈ 0** (Prütz 1985). Cys cascade (CysSH→CysSOH→CysSO2H→CysSO3H): 각 단계 k 결정. **DMEM에는 free CysSH 없음 (cystine·2HCl 0.2 mM만)**, 효소 없는 환경에서 환원 거의 안 됨. CysSSCys+•OH=7e9이 주 반응.
  - **Agent 3**: 부차 성분 + 결정적 발견. **HCO3⁻ 44 mM이 ONOO⁻ 화학을 완전히 hijack**: ONOO⁻+CO2 (k=5.8e4) → ONOOCO2⁻ → 0.33(•NO2+CO3•⁻) + 0.67(NO3⁻+CO2). τ(ONOO⁻) ≈ **10ms**, AA에 직접 도달 못 함. **Glucose 25 mM이 •OH의 49% 흡수** (k×[C]=3.8e7 s⁻¹). Tyr (0.4 mM) → 3-NO2-Tyr nitration이 주 신호. Trp (0.078 mM) → NFK/kynurenine.
  - **PPTX 대비 핵심 누락 8개** 식별: NaHCO3, Glucose, Tyr, Trp, Pyr+ONOO⁻ radical 생성, Cys 산화 캐스케이드, free Cys 부재, Met+H2O2 우선순위 오류.
  - **권장 reaction set**: Tier 1 14개 (Pyr×3, Met×3, Cys×1, Tyr×3, Glc×1, Gln+AA_pool×2, Bic×3 ★MUST). Tier 2/3 optional. 신규 species ~20종 (총 67~70).
  - **DMEM의 가장 큰 차이점**: pH가 거의 안 떨어짐 (HCO3⁻ buffer로 ~7.4 유지). DIW의 pH 3.6 결과와 정반대. 산성 thiolate 효과 (CysSH pKa 8.3) 무력화.
  - **변경/생성 파일**:
    - `notes/dmem_chemistry_research.md` 신규 (8 sections, ~60 sources, full kinetic table + recommended reaction set + initial conditions)
  - **다음 단계 (미진행)**: (1) Park ChemistryOpen 2024 e202300213 본문 직접 다운로드 (Pyr+H2O2 k=2.36은 Asmus 2019 인용 확인), (2) `reactions_dmem.yaml` 스켈레톤 작성, (3) 신규 species (Pyruvate, Cystine, Tyrosine, Glucose, HCO3, CO3_radical 등) chemistry_1d.py 등록, (4) DMEM initial condition (NaHCO3 buffered pH=7.4) 별도 시뮬 케이스 추가.

- 2026-05-12: **★ 2.6 kV O3 oscillation 해결 — NO2/NO3 species-aware SG smoothing 영구 default**
  - 배경: User가 `2.6kV_Humid_fitting_three_film_HONOvar_v2/fig1c` O3 시계열 oscillation 재지적. 2026-05-06 외부 AI Phase γ에서 "forced response from gas BC residual variation"로 정성 결론났으나 정량 확증 + production fix 미적용 상태였음. Phase F1~F3로 mechanism 확정 + 영구 default 변경.
  - ---
  - **Phase F1 — All-gas-const 검증** (`Figures/test/diag_o3_phase_f1_gas_const.py`):
    - 모든 gas species → 시간평균 const (NO2/N2O4/O3/N2O5/NO3 + HONO/HONO2/H2O2). 2 sims (baseline vs all-const).
    - Result detrended std (t>180s): bulk **23.31% → 5.09% (−78%)**, vol 10.67 → 1.54 (−86%), surf 5.49 → 0.43 (−92%). pH/NO3⁻/H2O2 동일.
    - Wall time 552s → 33305s (60× 증가, all-const step input의 BDF stiffness 영향).
    - **결론**: gas-side variation이 oscillation 단독 원인 정량 확증. 2026-05-06 외부 AI Phase γ 결론과 정확 일치.
  - ---
  - **Phase F2 — Leave-one-out species 격리** (`Figures/test/diag_o3_phase_f2_species_iso.py`):
    - 5 cases: baseline + (O3/NO2/N2O5/NO3 각각 const, 나머지 변동).
    - bulk std: baseline 23.31, **NO3 const 11.54 (−51%)**, O3 22.36 (−4%), N2O5 22.91 (−2%), **NO2 const 52.44 (+125%)**.
    - **NO2 const +125%는 step input artifact**: NO2 평균 const → HONO/N2O4도 t=0부터 큰 값 inject (정상은 ramp-up) → huge transient → detrend 잔차 폭증. NO2 진짜 변동 기여는 F2로 측정 불가.
    - **NO3 const −51%은 깨끗**: NO3는 derived species 없음 → step input artifact 작음 → 진짜 변동 제거 효과만 측정.
    - **User 직관 (NO2가 화학적으로 더 중요)이 정확**: step input artifact 인정.
  - ---
  - **NO2/NO3 raw CV% 비교** (시뮬 없이, `Figures/test/plot_no2_no3_smoothing.py`):
    - **NO2 raw 14.98%** → w=31 (62s) 5.81% → w=151 (302s) 4.94% → w=251 (502s) 4.12%
    - **NO3 raw 6.83%** → w=31 3.36% → w=151 3.13% → w=251 3.16%
    - **NO2가 NO3보다 raw measurement noise 2.2배 큼**. User 직관 확증.
    - w=151 → NO2에 유의미한 추가 reduction, NO3는 diminishing returns. w=151 채택.
  - ---
  - **Phase F3 — NO2/NO3 SG w=151 시뮬 검증** (`Figures/test/diag_o3_phase_f3_strong_sg.py`):
    - baseline (w=31 all) vs strong (NO2/NO3 w=151, O3/N2O5 default w=31). 2 sims.
    - Result bulk std: **23.31 → 12.52 (−46%)**, surf 5.49 → 2.28 (−58%). pH/NO3⁻/NO2⁻/H2O2 **0.2% 이내 동일**.
    - Wall time 563s → 469s (**−17%**, smooth input → BDF stiffness 감소).
    - **결론**: NO2+NO3 strong smoothing이 oscillation 절반 잡음. F1 5% floor 대비 F3 12.5% — 잔여 ~7%는 O3 자체 변동 + BDF numerical noise.
  - ---
  - **★ 영구 default 변경 (2026-05-12)**:
    - `Figures/gen_all_figures.py::_preprocess_below_lod(vals, species=None)`: **species-aware** SG window
      - NO2/NO3 → **window=151** (302s averaging, forced-response noise reduction)
      - Others → window=31 (62s, transient dynamics 보존)
    - 호출 site: `gas_conc[col] = _preprocess_below_lod(raw, species=col)` (line 245)
    - Sanity check: NO2 CV 1.67% → 0.44%, N2O4 CV 3.33% → 0.89% (NO2-derived), O3/N2O5 변경 없음, mean 모두 보존.
  - ---
  - **3 voltage HONOvar_v3 재생성 결과**:

| V | NO3⁻ sim/exp | NO2⁻ sim/exp | H2O2 sim/exp | pH sim/exp | wall |
|---|---|---|---|---|---|
| 2.6 kV | 36.92/32.63 (×1.13) | 0.049/0 | 5.17/4.76 (×1.09) | 4.43/5.09 | 488s |
| 3.2 kV | 71.86/62.74 (×1.15) | 4.15/3.58 (+16%) | 19.59/11.21 (×1.75) | 4.13/3.61 | 494s |
| 3.6 kV | 78.15/70.42 (×1.11) | 20.55/20.74 ✓ | 25.04/16.25 (×1.54) | 4.05/3.25 | 2235s |

    - HONOvar_v2 대비 변화 0.2% 이내 (NO3⁻/NO2⁻/H2O2/pH 모두 보존). 정량 fit 무손실.
    - **★ 2.6 kV fig1c O3 oscillation 완전 소멸** — 1~3.5분 사이 3-peak 패턴이 single monotonic ramp-up으로 변환. 3.2/3.6 kV는 원래 smooth였으나 일관성 위해 동일 적용.
  - ---
  - **Mechanism 종합 (2026-04-09 ~ 2026-05-12)**:
    1. 2026-04-09: O3 oscillation "BDF artifact" 잠정 결론 (틀림)
    2. 2026-05-04 Phase B5: BDF tolerance 10¹⁵× 강화에도 std% 동일 → numerical 배제
    3. 2026-05-04 Phase B redo: R32 OFF로 std 24→2.7% → "Lotka-Volterra limit cycle" 가설 (틀림)
    4. 2026-05-06 Phase H/γ: 0D + 1D PDE Jacobian 모두 Hopf candidates 0 → LV 수학적 기각, "forced response from gas BC variation" 정성 결론
    5. **2026-05-12 Phase F1 정량 확증**: all-const → 78% reduction. forced response 확정.
    6. 2026-05-12 Phase F2/F3: NO2 raw noise > NO3 (user 직관), production fix (NO2/NO3 SG w=151) 50% reduction with 0% 정량 변화.
  - ---
  - **변경/생성 파일** (이번 세션):
    - ★ 영구 변경: `Figures/gen_all_figures.py` `_preprocess_below_lod` species-aware (line 183-235, 245)
    - 진단 스크립트 신규: `Figures/test/diag_o3_phase_f1_gas_const.py`, `diag_o3_phase_f2_species_iso.py`, `diag_o3_phase_f3_strong_sg.py`, `plot_no2_no3_smoothing.py`
    - 진단 figure 신규: `Figures/test/fig_diag_o3_{f1,f2,f3}.{png,pdf}`, `fig_no2_no3_smoothing_2.6kV.{png,pdf}`
    - 진단 텍스트: `Figures/test/diag_o3_{f1,f2,f3}.txt`
    - **신규 폴더 (★ 새 default 결과)**: `Figures/DIW results/{2.6,3.2,3.6}kV_Humid_fitting_three_film_HONOvar_v3/` (각 9 figures + cache)
  - ---
  - **종합 상태 (Problem 1/2/3 통합 — 2026-05-04~12)**:
    | Problem | 진단 | Fix |
    |---|---|---|
    | 1 (2.6 kV oscillation) | ✓ forced response from gas BC variation (F1 78% 정량 확증) | **✓ NO2/NO3 SG w=151 영구 default (F3 46% in production, 0% 정량 변화)** |
    | 2 (O atom budget) | ✓ atto-Molar atol-band noise (2026-05-06) | ✓ plot fix (O→NO3, 2026-05-06) |
    | 3 (deep cells anomaly) | 부분 해결 (2026-05-07 영구 default fix) | ✓ cell anomaly 부분, O3 4mm peak는 미해결 |
  - ---
  - **잔존 Pending (재정의)**:
    1. ~~2.6 kV O3 oscillation~~ ✅ (이번 세션)
    2. **3.2/3.6 kV H2O2 1.5-1.75× 과다** — H2O2/O3 ratio voltage-별 재산정? (현재 uniform 0.003)
    3. **pH gap +0.4-0.8 unit** — charge balance gap, `project_le_chatelier_h_pending.md` 연관
    4. **Saline에 동일 NO2/NO3 SG default 적용 검증** — paper main hook
    5. **Voltage_comparison HONOvar_v3 figure** — 3-voltage bar comparison 신규 생성
    6. **O3 4mm deep peak** — reference solver (FD+RK4) 미진행

- 2026-05-14: **fig2b 재구성 (5 radicals) + R27/R32/R77 OFF sweep + Acid-base speciation framework 확정**
  - **Fig2b TARGET_SPECIES_RADICAL 변경**: `['NO3', 'OH', 'HO2', 'H+']` → `['HO2', 'HO3', 'O2-', 'O3-', 'OH']`. Layout 2×2 → 3×2 (6번째 hidden), abcde 라벨, suptitle `Radical & H+` → `Radical`. 3 voltage `_HONOvar_v3/` 폴더 fig2b 재생성.
  - **OH voltage scaling 의문 (user 출발점)**: fig2b OH panel에 R27 (O3+OH→HO2+O2)이 main sink로 보이는데, 저전압에서 O3 더 많으니 R27 sink 강해야 → OH 더 낮아야. 그러나 실측 sim: 2.6 kV [OH]=0.91 pM vs 3.6 kV [OH]=5.2×10⁻⁴ pM (1700× 감소).
  - **확증편향 정정 (반복)**: 첫 답변에서 R77이 lever라고 surface 농도 기반으로 주장 (k_R27=1.1e8/k_R77=1e10 잘못 사용). yaml 확인 결과 R27 k=3.0e9, R77 k=1.0e9. 두번째 답변에서 R77 dominant 다시 주장 (확증편향 user 비판). 세번째 답변에서도 R77 lever 가설 유지. **User: "확증편향적으로 R77이 우세할거라고 얘기하는 이유는?"** → 정량 분해로 정정.
  - **source vs sink decomposition (3.6 vs 2.6 kV)**: [OH] 1740× 감소 = source 356× 감소 × k_sink_eff 4.9× 증가 곱 (≈1740). **Source 감소가 main driver**, sink intensification 보조.
  - **R27/R32/R77 OFF sweep 실행** (`Figures/test/diag_r32_r77_off_sweep.py`, N_z=49 stretch 1.12, 2 voltages × 5 cases = 10 sims, ~10분):

| Case | 2.6 kV [OH] | 3.6 kV [OH] | 2.6/3.6 gap |
|---|---|---|---|
| baseline | 9.08×10⁻¹³ | 3.26×10⁻¹⁶ | **2780×** |
| R27 OFF | 3.98×10⁻¹² (×4.4) | 1.13×10⁻¹⁴ (×34.5) | 353× |
| R32 OFF | **5.21×10⁻¹⁴ (×0.057↓)** | 3.48×10⁻¹⁵ (×10.7) | 15× |
| R77 OFF | 1.13×10⁻¹² (×1.25) | 1.48×10⁻¹⁵ (×4.5) | 767× |
| R32+R77 OFF | 1.73×10⁻¹³ (×0.19↓) | 3.71×10⁻¹⁴ (×114) | **4.67×** |

  - **결정적 발견 1**: **R32+R77 합작이 voltage gap 2780× → 4.67× (600× 좁힘)**. 즉 voltage scaling 1700×의 거의 전부가 NO2⁻ 매개 chemistry (R32+R77)에서 옴.
  - **결정적 발견 2**: **R32 single이 voltage 따라 정반대 부호**. 3.6 kV에서 R32 OFF → OH ×10.7 회복 (NO2⁻이 O3 잡아먹는 lever). 2.6 kV에서 R32 OFF → OH ×0.057 (17× 감소!). [O3] 50× 폭증하면 R28 (O3+HO2→OH)이 [HO2] 540× crash시켜 source 자체 죽음. 비선형 chemistry.
  - **결정적 발견 3**: **R27 OFF는 voltage gap 거의 못 줄임** (2780→353, 8×). R27이 fig2b vol-int rate에서 dominant인 건 맞지만 voltage-independent — k·[O3]가 voltage 따라 약함. fig2b rate가 voltage 무관하게 ~3e-9으로 비슷한 이유: [OH]·[O3] cancel.
  - **잘못된 분석 정정 (또)**: 사용자 "R32 왜 끄고 검증함. R27 끈 게 아니라" → R27 OFF 추가. 사용자 "기여도 순서가 fig2b에 있는데 1% 안에도 없는 R78/R15 ㅇㅈㄹ" → t=600s degenerate state budget 들고 온 것 정정, fig2b 직접 읽으면 R27 main sink 인정.
  - **Sweep figure**: `Figures/test/fig_r27_r32_r77_off_sweep.{png,pdf}` (2.6/3.6 kV × OH/O3 시계열 4-panel + OH recovery bar chart).
  - ---
  - **Acid-base speciation framework 확정**:
    1. **사용자 첫 지적**: O2⁻ panel에 R27/R28/R41/R91 (HO2-direct rxns) 나오고 HO2 panel에 R25 (O2⁻-direct rxn) 나옴 → 헷갈림. 원인: `_total_match_names`가 acid-base pair를 mass pool (HO2_total)로 묶음.
    2. **strict matching 시도**: O2⁻ panel에 R25만 나오게 변경 → user 거부 ("생성 소멸에 관련된 건 다 표기하라는 의미"). Mass-pool aggregation 복원.
    3. **사용자 두번째 지적**: "근데 그렇다고 사라지는 O2⁻의 농도가 모두 (1:1로) sink에 기여하지는 않잖아" → speciation factor f_mol/f_ion 적용 필요. Per-cell rate × f(z) vol-avg 구현. HO2/O2⁻ panel 동일 rxn list, magnitude만 f_mol vs f_ion 비율로 분리.
    4. **사용자 세번째 지적 (핵심)**: "HO2 ↔ H+ + O2⁻를 통해 HO2 source sink를 계산하면 안됨? 그러면 딱 맞을거 아니야 budget" → AB equilibrium을 explicit line으로 표시.
    5. **최종 framework**:
       - Strict-direct rxns: 해당 species가 reactant/product에 명시된 경우만, full stoich × rate (no f-weighting)
       - MT line: `f_self(surface) × MT_pool` (transferable 종만)
       - **AB equilibrium line** (`HO2 ⇌ H+ + O2-` 등): `residual = ΔC/Δt − strict − MT` ← 정확한 정의
       - Panel title 농도: f_mol/f_ion으로 speciation 분리
  - **Mass balance 검증** (`Figures/test/diag_mass_balance_final.py`, 3 voltages):
    | 종 종류 | ratio Σ/FD | 평가 |
    |---|---|---|
    | Acid-base (HO2, O2⁻, NO2⁻, NO3⁻, H2O2) | **1.000000** | by construction 완벽 closure |
    | Non-pair O3 | 0.96-1.04 | <3% FD timing 잔차 |
    | OH (일반 시점) | 0.99-1.01 | <1% |
    | atto-Molar trace (HO3, O3⁻ 후반) | noise | atol-band, absolute 무시 |
  - **Pool-level 솔버 보존 확인**: `d⟨HO2_total⟩/dt(FD) vs ⟨chem⟩` ratio 0.99-1.00. **솔버는 pool 수준에서 직접 보존**, species 수준은 derived (f × pool). Panel closure는 후처리 residual approach.
  - **Ratio<1 잔차 원인 (non-pair)**: FD-시점 mismatch (ΔC/Δt 2s 평균 vs snapshot 순간 rate). dense_output Simpson 적분으로 완벽 closure 가능하나 미구현.
  - ---
  - **변경/생성 파일**:
    - `Figures/gen_all_figures.py`:
      - `TARGET_SPECIES_RADICAL` 변경 (5 radicals)
      - `gen_fig2b`: 3×2 layout, per-cell rate, AB-residual closure
      - `gen_fig2`: 동일 framework (strict-direct + MT speciated + AB residual)
      - `species_contribution`: `strict` 옵션 추가 (기본 False 유지)
    - 신규: `Figures/test/diag_r32_r77_off_sweep.py` (+ 10 npz caches `r27_r32_r77_sweep_{V}_{case}.npz`)
    - 신규: `Figures/test/plot_r32_r77_off_sweep.py` → `fig_r27_r32_r77_off_sweep.{png,pdf}`
    - 신규: `Figures/test/diag_fig2b_mass_balance.py`, `diag_fig2_mass_balance_full.py`, `diag_mass_balance_final.py`, `diag_oh_budget_voltage.py`
    - 재생성: `Figures/DIW results/{2.6, 3.2, 3.6}kV_Humid_fitting_three_film_HONOvar_v3/fig2_rate_evolution.{png,pdf}`, `fig2b_radical_rate.{png,pdf}`
  - **다음 단계 후보**:
    1. Saline에 동일 framework 적용 검증
    2. Voltage_comparison HONOvar_v3 figure 생성 (3 voltage bar 비교)
    3. dense_output Simpson 적분으로 non-pair 종 mass balance 완벽 closure (선택)
    4. R27/R32/R77 sweep에서 잔존 4.67× gap의 추가 lever 식별 (현재 미진행)
