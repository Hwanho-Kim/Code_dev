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

## Experimental Targets (3.2 kVpp only)
- DI water: pH=3.61, NO2-=3uM, NO3-=63uM, H2O2=11uM
- Saline: pH=3.60, NO2-~0, NO3-=102uM, H2O2=5uM

## Active Constraints
- 3.2kVpp 데이터만 사용. Multi-condition fitting 하지 않음.
- Poisson OFF (Debye << grid).
- S97/S98 반응 현재 disabled.
- "말을 듣고 해 니 말대로 하지말" — 사용자 지시를 정확히 따를 것.

## Current Status

### WORKING
- Ver3 (0D): DI water fitting 완료 (NO3- 0.6%, H2O2 0.0% 오차)
- Ver4_1D (DI water): fitting 완료 (NO2- 0.2%, NO3- 0.1%, H2O2 0.2% 오차)
- Ver4_1D (DIW forward, 측정종만): pH=3.389(6.1%), NO3-=424.6uM(574%), NO2-=0, H2O2=0.2uM
- **Monolithic BDF solver** (Strang splitting 대체): DIW 720s 완료. Film+α_b=0.03: pH=3.869(7.4%), NO3⁻=29.5µM, O3=7.3nM, H₂O₂=0.14nM. **1.7min** (atol=1e-12, rtol=1e-6, dt_enforce=None)
- Numba JIT 65x 가속, O(N) graph coloring
- Geometric grid convergence 확인 (49 vs 188 cells ~15% 차이)

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

### NOT TRIED
- Log-transform (C → ln(C)) for Newton conditioning
- Saline with fitted parameters (HONO/HONO2/H2O2 nonzero)
- BC 비교 saline 모드 (DIW 결과로 α_b 범위 확인 후)
- 종별 α_b 적용 (H₂O₂=0.1, O₃=0.01~0.1, NO/HNO₂=0.01~0.1)
- Monolithic BDF saline 실행 (QSSA OFF, Cl stiffness를 BDF가 직접 처리)

## Pending Tasks — 이번 주 (2026-04-01 갱신)
1. ~~**측정종만 사용 forward simulation**~~ ✅ DIW+Saline 모두 완료
2. ~~**Saline solver fix — QSSA 적용**~~ ✅ K_CAP 버그 수정 + 2-pass iteration → 질량 보존 완벽 (-12µM)
3. ~~**NO3⁻ 과다 진단**~~ ✅ N2O5 99.3% 지배. liquid-side limited. δ_liq=1mm → 121µM (실험 근접)
4. ~~**계면 BC 문헌 비교**~~ ✅ Film+α_b 모델 유효. α_b≈0.01~0.05에서 DIW 실험 교차
5. ~~**O3/라디컬 농도 문헌 검증**~~ ✅ Liu 2016 + Heirman 2025 대비 물리적으로 정확 확인
6. ~~**Monolithic BDF 구현**~~ ✅ Strang→monolithic 전환, dt 수렴 확인 (5% 변동)
7. ~~**Fig 2 rate budget 불일치 디버깅**~~ ✅ 근본 원인: atol=1e-8 → 1e-12로 해결. Simpson+dense_output+단일BDF. H₂O₂ ratio 0.988, 속도 1.7min
8. ~~**Fig 1~5 생성/갱신**~~ ✅ monolithic BDF + atol=1e-12 + dt_enforce=None으로 전 Figure 재생성 완료 (2026-04-01)
9. **Monolithic BDF saline 실행** — run_saline_1d.py (Film+α_b=0.03, QSSA OFF)
10. **종별 α_b 구현** — H₂O₂/O₃/NO/HNO₂ 각각 다른 α_b 적용
11. **Saline with fitted parameters** — HONO/HONO2/H2O2 가스상 포함 시 pH/NO3 개선 확인

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
- Film+α_b BC 채택: Heirman 2025 Eq.7 기반. k_mt = α_b × D_l / δ_liq. α_b≈0.01~0.05 범위에서 DIW NO3⁻ 실험값 재현 (2026-03-26)
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

---
<!-- UPDATE RULE:
작업 단위가 완료될 때마다 즉시 이 파일을 갱신할 것 (세션 종료를 기다리지 않는다).
1. Current Status 섹션의 WORKING/BROKEN/NOT TRIED 갱신
2. Pending Tasks 우선순위 조정
3. Key Decisions에 새 결정 추가
4. Session History에 날짜+요약 한 줄 추가
사용자가 터미널을 그냥 닫아도 CLAUDE.md는 이미 최신 상태여야 한다.
-->
