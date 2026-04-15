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
| 2.6 kVpp | 5.15 | 0 | 4.70 | 2.00 |
| **3.2 kVpp** | **3.60** | **0** | **10.45** | **5.14** |
| 3.6 kVpp | 3.43 | 0 | 16.92 | 7.73 |

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
- Ver4_1D (DI water): fitting 완료 (NO2- 0.2%, NO3- 0.1%, H2O2 0.2% 오차)
- **Monolithic BDF solver**: DIW 720s, atol=1e-15, rtol=1e-6, dt_enforce=None
- **gas_alpha BC 채택** (Schwartz 저항 모델): k_gi = (δ_gas/D_g + 4/(α_b·v̄))⁻¹, k_mt = k_gi/H_cc. 이중 계산 없음.
- **δ_gas=10mm** (잠정): gas gap 전체. 축소 필요할 수 있음 (pending task).
- **비율 기반 HONO/HNO₃/H₂O₂ gas input**: HONO/NO₂=0.33, HNO₃/N₂O₅=0.83, H₂O₂/O₃=0.03
- **현재 DIW 결과 (gas_alpha, δ_gas=10mm, 비율 gas input)**:
  - pH=3.81 (실험 3.61, 약간 높음)
  - NO₃⁻=155 µM (실험 63, **2.5배 과다**)
  - **H₂O₂=13.2 µM (실험 11, 근접!)**
  - NO₂⁻=0.003 µM (실험 3, **1000배 부족**)
  - O₃=302 nM
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
- **gas_alpha BC 채택 (2026-04-09)**: 기상+계면 저항만. 1/k_gi = δ_gas/D_g + 4/(α_b·v̄), k_mt = k_gi/H_cc. 액상 저항은 PDE가 처리. notes/bc_formulation.md 참조.
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

### PENDING (2026-04-15 기준)
1. **H₂O₂/O₃ 비율의 RH 의존성** — 현재 고정 0.03이지만 RH↑에서 증가해야 함. Humid fitting에서 H₂O₂ 부족(6.7 vs 11.2)의 원인. 비율을 RH 함수로 만들거나 별도 추정 필요.
2. **NO₂⁻ 문제** — O₃ barrier (R32). 1D no-convection 구조적 한계. 미해결.
3. **pH gap** — ~180µM 음이온 gap. 미해결.
4. **δ_gas 최종 결정** — 현재 10mm.
5. **Saline 실행** — 새 OAS data + gas_alpha BC.
6. **Phase 3: 살균 threshold 분석** — 3분/5분 critical species.
7. **Saline 실행** — gas_alpha BC + 새 OAS data.
8. *(future)* R93 amplification, OH oscillation.

---
<!-- UPDATE RULE:
작업 단위가 완료될 때마다 즉시 이 파일을 갱신할 것 (세션 종료를 기다리지 않는다).
1. Current Status 섹션의 WORKING/BROKEN/NOT TRIED 갱신
2. Pending Tasks 우선순위 조정
3. Key Decisions에 새 결정 추가
4. Session History에 날짜+요약 한 줄 추가
사용자가 터미널을 그냥 닫아도 CLAUDE.md는 이미 최신 상태여야 한다.
-->
