# plasma0d_v2 — 0D Plasma Chemistry Simulation

## Project Context
CH4/CO2 dry reforming in DBD plasma (N2/O2 carrier).
0D global model: 63 species, BOLSIG+ electron kinetics, BDF solver with constrained stepping.
Pulsed sDBD: 1333 Hz, vi_envelope power mode.

## Key Paths
- Entry: main.py (run via `python -m plasma0d_v2 --config config.yaml`)
- Solver: solver.py (~1780 lines) — constrained BDF manual stepping + CVODE ctypes
- CVODE wrapper: cvode_wrapper.py (~500 lines) — SUNDIALS 6.x ctypes binding
- Reactions: reactions.py (782 lines) + input/reactions.yaml
- Boltzmann: boltzmann.py (782 lines) + bolsig_parser.py (500 lines)
- Power: power.py (556 lines) — vi_envelope/vi_curve/pulsed modes
- Config: config.py (225 lines) + config.yaml
- Species: species.py (133 lines) + input/species.yaml
- Numba JIT: numba_core.py (~340 lines) — rhs_numba + pulsed_power_numba
- Electron kinetics: electron_kinetics.py (294 lines)
- Gas thermal: gas_thermal.py (159 lines)
- Flow: flow.py (87 lines) — PFR model
- Output: output.py (167 lines) → output/
- Test: test_jacobian.py (163 lines)
- Trial runner: trial_runner.py (222 lines)
- V-I data: input/Pulsed_VI_curve.csv
- BOLSIG+ data: input/BOLSIG_parameter/, input/BOLSIG_EEDF/
- Python: /home/hawn/work/plasma_liquid/Ver3/.venv/bin/python

## Current Status

### WORKING
- Continuous power (vi_envelope): 63 species, 3ms simulation, ne=2.08e16, Te=1.54eV, Tgas=302K
- Constrained BDF stepping: CVODE-style clamp, 6689 clamps / 101018 steps (6.6%)
- Multi-cycle pulsed discharge: reignition 성공 (Trial 10)
  - ne peak: 1.3e17 → 4.2e17 (cycle-to-cycle 증가)
  - Te: ~1.85eV (pulse) ↔ 0.026eV (afterglow)
- **CVODE ctypes wrapper 통합 완료** (2026-03-25) — method='CVODE'로 사용 가능
  - CVodeSetConstraints 비음수 보장, DQ Jacobian
  - Continuous PFR 1τ 검증: 4.95/7.18/13.21/19.94% (scipy BDF와 일치)
  - Pulsed trapezoidal: ReInit 제거 → 연속 적분 (2026-03-25)
    - Trapezoidal은 연속 함수이므로 ReInit 불필요, 단일 segment로 처리
    - Rectangular waveform만 pulse-edge ReInit 유지
    - 4 periods: CVODE 0.68s vs BDF 1.55s (2.3x), RHS 4830
    - ON phase ne/Te 정확 일치, afterglow 동작 기존과 동일
    - t=87.6µs CVODE 수렴 실패(h=53fs) — afterglow stiffness가 진짜 병목
    - **ReInit 제거 효과 미미** (0.71→0.68s, ~4%): BDF order 유지해도 stiff 구간에서 step 축소 불가피
  - Numba fast RHS: rhs_numba 직접 호출 + pulsed_power_numba 인라인 (Python 콜백 제거)
  - Analytic Jacobian은 과도 stiffness 해석으로 비효율 (DQ가 76x 적은 스텝)
- **Pulsed trapezoidal 1τ 시뮬레이션 완료** (2026-04-02, OFF phase EI 활성화)
  - CVODE ON/OFF operator splitting: ON=full RHS(Numba), OFF=rhs_off(EI+TE+Arrhenius, P_dep=0)
  - OFF phase에서 EI 반응 CX 기반 정상 계산 (이전: k_ei_conc=None으로 EI 비활성이었음)
  - V_eff=4.9cm³, PRF=1333Hz, dc=20%, P_peak=32.5W, P_avg=6.5W
  - 883 pulses, 31.2min, ON=0 fails, OFF=26 fails, 0.47 pulses/s
  - CH4 전환: **5.52%** (이전 EI-off 버전 5.56%와 재현)
  - ne_peak: ~1e14 (quasi-steady), ne_valley: ~3.2e6 m⁻³ (이전 6.6e5 대비 5x↑)
  - ne_eps thermal reset: ON→OFF 전환 시 ne_eps=n_e×ε_th
  - ne re-seeding: 매 pulse 시작 시 ne≥1e8 m⁻³
  - 프로파일: output/pulsed_last10_nofreeze.png
- **Pulsed P_avg=6.5W 4온도 검증** (2026-03-30, EI-off 버전)
  - P_peak=32.5W, dc=20%, P_avg=6.5W
  - 303K: 5.56%, 373K: 6.91%, 453K: 13.08%, 523K: 20.41% (RMSE=0.89)
  - Continuous 6.5W (RMSE=0.87)과 거의 동등한 예측력
  - ※ EI 활성화 버전으로 4온도 재검증 필요
- Reaction set: mature (reactions.yaml)
- PFR flow model: 작동
- **V_eff/V_reactor dilution 제거 완료** (2026-03-24, 커밋 a0247c3)
  - V_eff 무관: PFR에서 dilution 유무 관계없이 동일 결과 (수학적 동치 증명)
  - PFR(1τ) baseline: 4.95/7.18/13.21/19.94% — dilution 제거 전과 동일 (2026-03-25 재확인)
  - ※ 이전 3.09% 결과는 LUT clamp 실험 중 코드 불완전 복원 아티팩트
- **V_eff 기반 volume 모델 실험** (2026-03-26, 미커밋)
  - sDBD geometry: 70mm×70mm×1mm → V_eff=4.9e-6 m³ (4.9 cm³)
  - PowerSource + FlowModel 모두 V_eff 사용 (V_reactor 미사용)
  - config.py: `PowerSource(V_eff=V_eff)`, `FlowModel(V_reactor=V_eff)`
  - config.yaml: `V_eff: 4.9e-6`
  - Pulsed 2 periods: ne=2.3e13(ON), 2e10(afterglow) — 물리적으로 합리적
    - clamp rate 9.5% (이전 V_reactor 기준 대비 대폭 개선)
    - afterglow ne가 seed까지 추락하지 않음 (비선형 탈착 효과)
  - Continuous 4온도: 3.40/4.93/9.00/14.46% (이전 baseline 4.95/7.18/13.21/19.94% 대비 낮음)
  - V_eff sweep 결과: 두께 0.05~5mm 전 범위에서 실험값보다 낮음 (RMSE 최소=4.03 at 0.2mm)
  - Rate 분석: 1차 반응(EI)은 ~51x 스케일링+τ 1/51로 동치. 차이 원인은 비선형 radical recycling
    - OH: 18.4x (51x 아님) — 높은 radical 밀도에서 상호 소비 증가
    - 장수명 종(O₃, CO, H₂): ~1x — τ_eff=0.66s 안에 포화
    - 이온-이온 재결합: ~1300-3400x (51² 스케일링)

### BROKEN / LIMITATIONS
- ~~Pulsed 장시간 시뮬레이션: 993s for 3ms~~ → **해결** (2026-03-30): CVODE ON+OFF operator splitting으로 1τ 완료
- **★ ne afterglow 과도 감쇠 — 최우선 해결 과제** (2026-03-24 식별, 2026-03-30 freeze 제거 후 재확인):
  - **현상**: ON phase ne~2.5e12 m⁻³ → OFF phase ne~6.6e5 m⁻³ (**6자릿수 감쇠**, 600µs 이내)
  - 문헌의 DBD afterglow ne: ~1e16~1e19 m⁻³ (1e10~1e13 cm⁻³) — 우리 모델보다 10자릿수 이상 높음
  - **원인**: 3체 O₂ 부착(e + O₂ + M → O₂⁻ + M, 반응 165)이 전자 손실 97% 지배, τ~40ns
  - DR(해리 재결합)은 2%에 불과
  - 탈착 SOURCE > DR LOSS (15배)이나, 3체 부착이 압도하여 net 음수
  - N2(A)가 O₂ 15% quenching으로 0.4µs 소멸 → Penning/N2(A)-driven detachment 무효
  - **LUT→Maxwellian 전환에서 k_att 5배 불연속** (eps=0.0399→0.0400)
  - **현재 workaround**: 매 pulse 시작 시 ne re-seeding (1e8 m⁻³) — 물리적 근거 약함
  - **영향**: afterglow 동안 전자 관련 반응(EI, attachment, DR) 사실상 중단 → pulsed 특유의 afterglow chemistry 미반영
  - 프로파일 시각화: output/pulsed_last10_nofreeze.png
- **V_eff 모델에서 continuous 전환율 감소** (2026-03-26):
  - 4.95→3.40%(303K), 19.94→14.46%(523K) — 실험값보다 더 낮아짐
  - 원인: 높은 radical 밀도에서 OH 등 상호 소비 증가 (비선형 효과)
  - V_eff sweep(0.05~5mm)으로도 실험값 도달 불가 (RMSE 최소 4.03)
- **PFR 초기조건 편향 문제** (2026-03-27 식별, 2026-03-30 PFR(1τ) 유지 결정):
  - 핵심 문제: 플라즈마 PFR에서 반응 조건(ne, Te, 라디컬)이 해의 일부 → τ 동안 변화 → thermal PFR과 본질적 차이
  - Parcel cycling 시도(2026-03-30): 출구 종을 다음 parcel 초기조건으로 이식 → recycle reactor 해에 수렴 (303K: 17.82% vs 실험 5.26%) → PFR 프레임워크 내에서 해결 곤란
  - **결정: PFR(1τ) 그대로 유지** — 실험과 잘 일치 (4.95/7.18/13.21/19.94% vs 5.26/8.05/14.36/20.02%, RMSE~1.1%p)
  - 근본적 물리 문제는 미해결이나, 현재 접근법이 실용적으로 충분

### NOT TRIED
- ~~SUNDIALS CVODE 직접 사용~~ — 완료 (2026-03-25, ctypes wrapper)
- ~~Operator splitting (Patankar-type positive scheme)~~ — CVODE ON/OFF splitting으로 대체 (2026-03-30)
- Adaptive time stepping optimization for pulsed mode
- 장시간 시뮬레이션을 위한 cycle-averaging 기법
- **Penning ionization + 전자 탈착 반응 점검/추가** (afterglow ne 유지, 2026-03-24 우선순위 1)
  - 단, N2(A) 경로는 O₂ quenching으로 무효 → 다른 메커니즘 필요
- ~~ne_seed 상향~~ — 철회: afterglow 물리를 분해할 수 없는 비물리적 접근 (2026-03-24)
- ne_eps만 로그 변환 (Trial 6은 species 전체 변환이었음, 에너지만 별도 가능)
- ~~Superelastic collision~~ — 대기압에서 Te buffering 무시 가능 (elastic cooling 압도, 2026-03-24 확인)
- ~~LUT boundary clamp 시도~~ — 실패 (2026-03-25)
- ~~LUT smooth blending 시도~~ — 실패 (2026-03-25): emin-eth=0.00075eV 너무 좁아 BDF 실패
- **LUT 하한 확장** (BOLSIG+ 추가 E/N 포인트로 ε̄<0.04 EEDF 확보) — 불연속 해결의 올바른 접근
- ~~V_eff 모델 최적화~~ (2026-03-26): V_eff sweep(0.05~5mm) RMSE 단조감소, 최솟값 없음. V_eff→V_reactor 극한이 최선 (2026-03-26 재확인)
- ~~PFR ne pre-conditioning 검증~~ (2026-03-27→2026-03-30): Parcel cycling 시도 → recycle reactor 해 산출. PFR(1τ) 유지 결정.
- CSTR vs PFR 비교 (2026-03-27) — PFR(1τ) 유지 결정으로 우선순위 낮춤. 필요 시 향후 재검토.

## Key Decisions (settled)
- **V_eff/V_reactor dilution 제거 확정** (2026-03-24) — PFR(1τ)에서 수학적 동치 증명. ne가 자체 조정(~1/f)되어 chemistry 결과 동일. dilution은 ne 해석만 바꿀 뿐. 코드 변경 적용 상태(미커밋).
- **PFR t_end=1×τ** (2026-03-24) — PFR에서 flow source=0이므로, 5τ면 5배 과노출. CSTR은 5τ OK.
- **CVODE ctypes wrapper: DQ Jacobian + scalar atol 사용** (2026-03-25) — analytic Jacobian은 이 시스템에서 76x 더 많은 스텝(20K vs 266/100ns). DQ가 실용적으로 최적.
- Constrained BDF manual stepping 채택 (Trial 10, 2026-03-09)
- Electron floor: ce_floor=1e-30, ne_eps_floor=1e-20 (ZDPlasKin 스타일, continuous에서 무영향) (2026-03-25)
- Vector atol: electron/ne_eps at floor*0.1, rest at 1e-10
- Log transform 불가: trace species에서 stiffness 증가 (Trial 6)
- Radau 불가: Jacobian overflow (Trial 3)
- **Maxwellian fallback 유지 결정** (2026-03-25) — LUT boundary clamp은 afterglow 악화 (BOLSIG+ EEDF가 부착 resonance에서 Maxwellian보다 k_att 5x 높음). Thermal Te에서 Maxwellian이 더 물리적.
- **V_eff 기반 volume 모델 실험** (2026-03-26) — sDBD 70mm×70mm×1mm, V_eff=4.9cm³. P_dep+τ 모두 V_eff 기준. Pulsed ne 물리적 개선(2e13 ON, 2e10 afterglow), continuous 전환율은 비선형 효과로 감소. 커밋 1419f37.
- **V_eff sweep RMSE 단조감소 확인** (2026-03-26) — 0.05~5mm 전 범위에서 V_eff↑ → conversion↑ → RMSE↓. 최솟값 없음. 비선형 반응(radical 상호 소비, 이온-이온 재결합)이 V_eff 민감도의 원인.
- **V_eff 방식 확정, V_reactor 방식 폐기** (2026-03-30) — V_eff=4.9cm³가 물리적으로 올바름. V_reactor=250cm³ baseline(4.95/7.18/13.21/19.94%)은 참고 기록으로만 보존. 현재 baseline: V_eff=4.9cm³ → 4.35/5.84/10.33/16.45%.
- **반응 메커니즘 검증 완료** (2026-03-30) — 218개 반응 세트가 CH₄ 전환 예측에 충분함을 확인. (1) Rate coefficient 불확실성: 7개 핵심 radical 반응을 JPL/NIST 대조 → OH+OH+M이 44~120x 과소평가였으나 전환율 영향 <0.01%p. (2) 누락 반응: O+OH→O₂+H(CRITICAL), N+NO/N+OH(HIGH) 등 6개 추가 테스트 → 최대 -0.11%p, 전환율 gap 원인 아님. Gap은 0D+V_eff 체적 모델의 구조적 한계(radical 비선형 스케일링).
- **V_eff 모델 pulsed 개선의 진짜 원인 = P_dep 증가** (2026-03-27) — dilution 제거(a0247c3)에서 PowerSource(V_eff=V_reactor)로 설정됨. V_eff 모델(1419f37)에서 PowerSource(V_eff=4.9cm³)로 변경 → P_dep 51배 증가가 pulsed ne 개선 원인. τ 변화는 PFR pulsed에서 무관(t_end=n_periods). Continuous 전환율 감소는 τ 단축이 원인.
- **OFF phase electron freeze 제거 + ne_eps thermal reset** (2026-03-30) — rhs_off에서 전자/이온/ne_eps 동결 삭제. ne_eps가 ne와 비례 감쇠(eps_mean 보존) + ON→OFF 전환 시 ne_eps=n_e×ε_thermal(대기압 elastic cooling ~100ns로 정당화). 매 pulse 시작 ne re-seed(1e8 m⁻³, 배경 이온화 모사). Power-scaled 전환율이 continuous와 동일(4.32% vs 4.35%).
- **OFF phase EI 반응 활성화** (2026-04-02) — rhs_off에서 k_ei_conc=None(EI 비활성) → CX 기반 정상 계산으로 변경. 이전 Kossyi thermal attachment 하드코딩 제거. rhs_off = full RHS with P_dep=0. CH4 전환율 재현(5.52% vs 5.56%), ne_valley 5x 상승(3.2e6 vs 6.6e5).
- **PFR(1τ) 유지 결정** (2026-03-30) — 플라즈마 PFR에서 반응 조건(ne, Te, 라디컬)이 τ 동안 변화하는 근본적 차이는 미해결이나, PFR(1τ) baseline이 실험과 잘 일치(RMSE~1.1%p)하므로 현재 접근법 유지. Parcel cycling 시도 → recycle reactor 해에 수렴(303K: 17.82%)하여 PFR 프레임워크 내 해결 곤란 확인.

## Troubleshooting History
- Pulsed 음수 ne 문제: notes/20260306_pulsed_negative_ne.md (Trial 1~10)
- V_eff 민감도 분석: notes/20260330_veff_sensitivity.md (radical 비선형 스케일링)
- 반응 메커니즘 검증: notes/20260330_mechanism_verification.md (rate 불확실성 + 누락 반응)

## Pending Tasks
1. **★ Afterglow ne 감쇠** (2026-04-02 업데이트)
   - **현황**: OFF phase EI 활성화 완료. ne_peak~1e14 → ne_valley~3.2e6 m⁻³ (8 decades, 문헌은 2-3 decades)
   - **근본 원인**: volume-averaging (ne_peak=2.5e6 cm⁻³ vs 문헌 10¹⁴ cm⁻³, 8자릿수 차이)
   - **코드 수정 완료**: rhs_off에서 k_ei_conc 정상 계산 (CX 기반, 이전 k_ei_conc=None 제거)
   - **Kossyi 하드코딩 제거**: 3/31에 추가했던 thermal attachment injection 삭제
   - **CH4 전환율**: 5.52% (이전 EI-off 5.56%와 재현), ne_valley 5x 상승
   - **성능**: 31.2min/883p (이전 3.9min 대비 8x 느림, 3체 부착 12ns timescale)
   - **관련 파일**: solver.py rhs_off(), output/pulsed_last10_nofreeze.png
2. **LUT 하한 확장** — BOLSIG+에서 ε̄<0.04 EEDF 추가 계산하여 LUT→Maxwellian k_att 5배 불연속 제거
3. **Pulsed OFF phase 음수 종 처리 개선** — rhs_off에서 trace species(CH3+, O-, N2(A) 등) 음수 발생, 현재 clamp으로 처리 중

## Related Project
- plasma_liquid (Ver4_1D): /home/hawn/work/plasma_liquid/
  - 1D 액상 반응 수송 모델, 이 코드의 가스상 결과를 boundary condition으로 사용

## Session History
- 2026-03-06~09: Pulsed discharge 음수 ne 문제 해결 (Trial 1~10)
- 2026-03-23: CLAUDE.md 생성, 프로젝트 상태 체계화
- 2026-03-24: ZDPlasKin/GlobalKin/COMSOL 문헌조사 → ne afterglow 과도 감쇠가 핵심 문제. 3체 O₂ 부착 97%, DR 2%, N2(A) O₂ quench 0.4µs.
- 2026-03-24: V_eff/V_reactor dilution 제거 실험 → PFR(1τ)에서 수학적 동치 증명 (ne 자체 조정). dilution 유무 관계없이 CH4 4.95/7.18/13.21/19.92% 동일. 코드 변경 적용(미커밋).
- 2026-03-25: GlobalKin solver 구현 문헌조사. Lietz PhD thesis(2019) 확인: DVODE 사용, rtol=1e-8~1e-6, density<1e6 cm⁻³이면 tolerance 적용 제외. ZDPlasKin: DVODE_F90 + 내장 non-negative constraint(clower=0), rtol=1e-5/atol=1e-10. ChemPlasKin: CVODE BDF, rtol=1e-5~1e-9/atol=1e-10~1e-15, 음수 mass fraction→0 클리핑. CVODE CVodeSetConstraints(1.0=non-negative) 기능 확인.
- 2026-03-25: CVODE ctypes wrapper 통합. cvode_wrapper.py 확장(SVtolerances, Jacobian 콜백, custom constraints). solver.py에 _solve_cvode() 추가, method='CVODE' 디스패치. CSTR 5τ에서 scipy BDF와 완벽 일치 (5.50/7.51/12.12/17.79%). 10x 빠름, 음수 없음.
- 2026-03-25: CVODE pulsed trapezoidal 검증. pulse-edge ReInit(ON/OFF), Numba fast RHS 최적화. 4 periods: CVODE 0.71s vs BDF 1.55s (2.2x). t=50µs ne→0 anomaly는 afterglow 과도감쇠(물리적). Continuous PFR 1τ 회귀 PASSED.
- 2026-03-25: CVODE pulsed ReInit 제거 → 연속 적분 구현. Trapezoidal은 단일 segment, rectangular만 ReInit 유지. 결과: 0.68s vs 0.71s (~4% 개선만). afterglow stiffness가 진짜 병목이므로 ReInit 제거 효과 미미.
- 2026-03-25: LUT→Maxwellian fallback 불연속 분석. LUT boundary clamp 시도 → 실패: BOLSIG+ EEDF(ε̄=0.04)가 부착 resonance에 전자 더 많아 k_att이 Maxwellian보다 높음. Clamp 시 afterglow ne 5x 악화 (BDF 83% clamp). Maxwellian fallback이 thermal Te에서 더 물리적. 올바른 fix: BOLSIG+에서 ε̄<0.04 범위 EEDF 추가 계산하여 LUT 확장.
- 2026-03-25: Baseline 재현 조사. 이전 3.09% 결과는 LUT clamp 실험 중 numba_core.py kth 소스 변경 불완전 복원 아티팩트. 현재 코드에서 floor 값(1e-30 or ne_seed/NA) 무관하게 4.95/7.18/13.21/19.94% 재현 확인. Continuous 모드에서 ne~1e11이므로 floor(1e8) 도달 없음.
- 2026-03-25: Pulsed 장시간 시뮬레이션 문헌조사 완료. 핵심 발견: Bogaerts 그룹은 DBD 0D 모델에서 모든 microdischarge를 brute-force 시뮬레이션(ns→s). Snoeckx 2013: 662 pulses/0.037s, Lietz/Kushner 2016: 5000 pulses @10kHz/0.5s+5min afterglow. Ning 2023: GlobalKin 10kHz 100s(=100만 pulses). 가속 기법: (1) deep learning surrogate(30h→수초), (2) ChemPlasKin unified ODE(operator splitting 대비 3x), (3) time-averaged rate coefficients. Cycle-averaging/extrapolation은 표준 기법 아님 — 대부분 brute-force.
- 2026-03-26: Meyer, Hartman, Kushner (2025) J.Appl.Phys. 논문 리뷰. GlobalKin 0D, Ar/CH₄/O₂, 122종 3265반응. ne=1.8e13 cm⁻³(ON), ~1e11(afterglow, seed까지 안 떨어짐). 단, microfluidic channel(500µm gap)이라 V_eff≈V_reactor — sDBD volume 문제 없음.
- 2026-03-26: 0D 모델 volume 처리 문헌조사. 핵심: GlobalKin/ZDPlasKin/ChemPlasKin 모두 intensive formulation(per unit volume), volume이 방정식에 없음. Volume은 P_dep=P/V 계산에서만 영향. VDBD는 V_eff≈V_reactor로 문제 없음. sDBD 전용 0D 모델은 거의 없음.
- 2026-03-26: V_eff 기반 volume 모델 실험. sDBD 70mm×70mm×1mm, V_eff=4.9cm³. PowerSource+FlowModel 모두 V_eff 사용. Pulsed: ne=2.3e13(ON)/2e10(afterglow), clamp 9.5%(이전 92%). Continuous PFR(1τ) V_eff=4.9cm³: 4.35/5.84/10.33/16.45%. 이전 기록(3.40/4.93/9.00/14.46)은 미커밋 중간 상태에서 측정된 것으로 재현 불가(커밋 1419f37 후 확인).
- 2026-03-26: V_eff sweep 재실행(0.05~5mm). RMSE 단조감소(4.18→2.35), 최솟값 없음. V_reactor=250cm³ baseline(4.95/7.18/13.21/19.94, RMSE~1.1)이 최선. CH4 소비 반응 분석: 저온(30°C) EI+N2*=72%, 고온(250°C) OH+O radical=79%. CH4+OH가 10→67%로 T↑시 지배. 생성은 CH5⁺+H2O→H3O⁺+CH4(97~99%)뿐, prod/cons=2.2→0.3%.
- 2026-03-27: V_eff sweep PFR 1τ 재실행(0.49~250cm³, 9점×4온도). 이전 결과 완벽 재현(V_eff=4.9cm³: 4.35/5.84/10.33/16.45%, V_reactor=250cm³: 4.95/7.18/13.21/19.94%). RMSE 단조감소 재확인(3.78→0.74).
- 2026-03-27: V_eff 모델 pulsed 개선 원인 분석. 커밋 a0247c3에서 PowerSource(V_eff=V_reactor)로 변경 확인 → 1419f37에서 PowerSource(V_eff=4.9cm³)로 변경 시 P_dep 51배 증가가 pulsed ne 유지의 진짜 원인. dilution 제거 자체는 동치이므로 무관. Continuous 전환율 감소는 τ 단축(33.8→0.66s)이 원인.
- 2026-03-27: PFR 초기조건 편향 문제 식별. ne_seed에서 시작하는 PFR은 ignition 과도구간(~수ms)의 낮은 반응속도가 ∫₀^τ R(c)dt에 포함 → 전환율 하방 편향. 실제 PFR 정상운전에서는 ne가 이미 정상. 표준 0D 코드(Bogaerts/GlobalKin)는 pulsed batch에서 자연 해결(수 pulse 후 quasi-periodic). Continuous+PFR+짧은τ 조합이 비표준적 — CSTR 또는 ne pre-conditioning 필요.
- 2026-03-30: V_eff 방식 확정, V_reactor=250cm³ 폐기. V_eff 민감도 분석: radical 비선형 스케일링이 원인. EI/N₂* 반응은 V_eff 무감(ratio=1.0x), OH 경로(303K: 0.28x, Δ=−1.20%p), O 경로(523K: 0.57x, Δ=−1.34%p). HO₂ ratio=7.0≈√51(교과서적 2차 자기소멸). 상세: notes/20260330_veff_sensitivity.md.
- 2026-03-30: PFR 초기조건 편향 문제 검토. Parcel cycling 3가지 구현 시도: (1) 전 종 carry-over → 생성물 무한 축적. (2) 몰분율 정규화 → 비feed 종 공간 0. (3) c_room 질량 보존 → recycle reactor 해 수렴(303K: 17.82%). PFR 프레임워크 내에서 해결 곤란 확인. 핵심 문제(플라즈마 반응 조건이 τ 동안 변화)는 미해결이나, PFR(1τ) baseline이 실험과 잘 일치(RMSE~1.1%p)하므로 PFR(1τ) 유지 결정.
- 2026-03-30: 반응 메커니즘 검증 — rate coefficient 불확실성 + 누락 반응 검토. (1) Rate: 7개 핵심 radical 반응 JPL/NIST 대조. OH+OH+M(id90) 44~120x 과소, HO₂+HO₂(id48) bimol항 누락(~2x), 나머지 양호. 보정해도 전환율 변화 <0.1%p. (2) 누락 반응 6개 추가 테스트: O+OH→O₂+H(CRITICAL), N+NO/N+OH/N+O₂(Zeldovich), NO₂+O, O+O+M. 최대 -0.11%p. 메커니즘은 CH₄ 전환에 충분, gap은 0D+V_eff 체적 모델 구조적 한계. 상세: notes/20260330_mechanism_verification.md.
- 2026-03-30: Pulsed trapezoidal 1τ 시뮬레이션 완료. CVODE ON/OFF operator splitting 개발: ON phase=full RHS(Numba, constrained), OFF phase=rhs_off(frozen electrons, unconstrained). V_eff=4.9cm³, 883 pulses, 4.6min, 0 failures. CH4 전환 1.51% (P_avg=1.62W). ne quasi-steady 2.4e12. Afterglow stiffness 우회 성공. OFF phase에서 trace species 음수 → clamp으로 처리.
- 2026-03-30: Pulsed P_avg=6.5W 4온도 검증. P_peak=32.5W, dc=20%. 결과: 5.56/6.91/13.08/20.41% (RMSE=0.89). Continuous 6.5W (5.63/7.57/13.40/21.34%, RMSE=0.87)과 거의 동등. Pulsed/Cont ratio 0.91~0.99. 전 온도 0 failures, 총 7.1min.
- 2026-03-30: rhs_off electron freeze 제거. 사용자 요청으로 OFF phase 전자/이온/ne_eps 동결 코드 삭제. (1) freeze 없이 constraints='none' → ne=-4.7e14 폭주. (2) full RHS P_dep=0 → afterglow stiffness 15s/pulse. (3) rhs_off + ne_eps proportional tracking → Te 발산. (4) 최종 해법: ne_eps thermal reset(ON→OFF 시 ne_eps=n_e×ε_th) + ne re-seeding(1e8 m⁻³). 883 pulses 3.9min, 0 fails, CH4=1.40%. Power-scaled: 4.32%≈continuous 4.35%.
- 2026-03-30: Pulsed reference 확정 — P_avg=6.5W, P_peak=32.5W, dc=20%, PRF=1333Hz. ★ ne afterglow 6자릿수 급감(2.5e12→6.6e5)을 최우선 과제로 재설정. 상세 맥락 기록: memory/pulsed_afterglow_ne.md.
- 2026-04-02: rhs_off EI 활성화. (1) 이전 rhs_off는 k_ei_conc=None으로 EI 반응 전체 비활성 — 사실상 부분 freeze였음. (2) 3/31 추가한 Kossyi thermal attachment 하드코딩 제거. (3) rhs_off에서 full RHS와 동일하게 k_ei_conc를 CX 기반으로 정상 계산 (3-stage afterglow transition). P_dep=0만 차이. (4) P_peak=32.5W 1τ 검증: CH4=5.52%(이전 5.56% 재현), ne_valley=3.2e6(이전 6.6e5 대비 5x↑), 31.2min, OFF 26 fails. 프로파일: output/pulsed_last10_nofreeze.png.

---
<!-- UPDATE RULE:
작업 단위가 완료될 때마다 즉시 이 파일을 갱신할 것 (세션 종료를 기다리지 않는다).
1. Current Status 섹션의 WORKING/BROKEN/NOT TRIED 갱신
2. Pending Tasks 우선순위 조정
3. Key Decisions에 새 결정 추가
4. Session History에 날짜+요약 한 줄 추가
사용자가 터미널을 그냥 닫아도 CLAUDE.md는 이미 최신 상태여야 한다.

DATE RULE (모든 기록에 적용):
- CLAUDE.md, notes/*.md 등 모든 문서 기록 시 반드시 날짜(YYYY-MM-DD)를 포함한다.
- 새 섹션/항목 추가 시: "(YYYY-MM-DD)" 또는 "(YYYY-MM-DD 조사/확인/추가)" 형태로 날짜 표기
- Session History: "YYYY-MM-DD: 내용" 형식
- Key Decisions: "결정 내용 (YYYY-MM-DD)" 형식
- 날짜 없는 기록은 금지.
-->
