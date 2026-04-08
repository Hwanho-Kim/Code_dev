# Numba @njit RHS Rewrite for plasma0d_v2

## TL;DR

> **Quick Summary**: RHS 핵심 연산을 Numba @njit 컴파일된 순수 함수로 재작성하여 126µs → ~6µs (20× 가속)
> 
> **Deliverables**:
> - `plasma0d_v2/numba_core.py` — @njit 컴파일된 RHS 함수들
> - `solver.py` 수정 — numba RHS를 기본으로 사용, Python fallback 유지
> - 벤치마크: 10,000회 호출 비교
> 
> **Estimated Effort**: Medium (1-2일)
> **Parallel Execution**: NO — sequential
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request
pulsed long-time simulation (340k pulses)이 Python RHS 오버헤드로 인해 비실용적 (1 period = 10s, 340k = 39일). 
RHS 자체는 126µs이지만 solver loop의 Python 오버헤드가 9s. 
Numba로 RHS를 컴파일하면 RHS 비용이 ~6µs로 줄어 전체 wall time 감소.

### Research Findings
- Numba LUT interpolation만 JIT: 8× 가속 (7µs → 1µs) 하지만 RHS의 6%만 차지
- 전체 RHS JIT 필요: compute_reaction_rates (30%), source_terms (20%), LUT (6%), 기타 (44%)
- @njit은 Python 객체(class instance, dict) 불가 → 모든 데이터를 numpy array로 추출

---

## Work Objectives

### Core Objective
solver.rhs()의 핵심 연산을 @njit 순수 함수로 재작성

### Concrete Deliverables
- `plasma0d_v2/numba_core.py`: @njit 함수 + extract_numba_params() 헬퍼
- `solver.py`: rhs()에서 numba RHS 호출, fallback 유지

### Definition of Done
- [ ] rhs_numba(t, y, params) == solver.rhs(t, y) within 1e-6 relative error
- [ ] rhs_numba < 10µs per call (10,000 iterations 벤치마크)
- [ ] continuous PFR 4온도 regression 통과 (4.95/7.18/13.18/19.92%)
- [ ] pulsed 1 period 결과 일치

### Must Have
- LUT interpolation @njit
- compute_reaction_rates @njit (EI + Arrhenius + TE-dependent)
- compute_source_terms @njit
- compute_gas_heating @njit
- electron energy RHS @njit
- gas temperature RHS @njit
- flow source @njit
- diffusion rate @njit
- 3-stage afterglow transition @njit
- sigma_over_N 지원 (N_gas_cm3 곱셈)
- Maxwellian fallback (thermal cache)

### Must NOT Have
- 기존 Python RHS 삭제 (fallback으로 유지)
- Jacobian의 Numba화 (이 계획 범위 밖)
- solver loop 자체의 변경

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (벤치마크 스크립트)
- **Automated tests**: Tests-after

### QA Policy
모든 task에서 Python RHS와 Numba RHS 비교 검증

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (기반):
├── Task 1: numba_core.py 파일 생성 — @njit 헬퍼 함수들 [quick]

Wave 2 (핵심 함수):
├── Task 2: rhs_numba() 메인 함수 + extract_numba_params() [deep]

Wave 3 (통합):
├── Task 3: solver.py에 numba RHS 통합 [quick]

Wave 4 (검증):
├── Task 4: 정확성 검증 + 벤치마크 [unspecified-high]
├── Task 5: continuous + pulsed regression [unspecified-high]
```

---

## TODOs

- [ ] 1. numba_core.py — @njit 헬퍼 함수 구현

  **What to do**:
  - `/home/hawn/work/plasma0d_v2/numba_core.py` 새 파일 생성
  - 아래 @njit 함수들 구현:
    - `lut_rate_coefficients(log_eps, log_eps_grid, log_k_table, k_dead_mask)` — LUT 보간
    - `_lut_interp_scalar(log_eps, grid, data)` — 단일 transport coefficient 보간
    - `compute_rates(c, T_gas, c_total, k_ei, has_ei, Te_eV, P_gas, ...)` — 모든 반응 속도
    - `compute_source_terms(rates, stoich_matrix, veff_mask, n_sp)` — S_veff, S_bulk
    - `compute_gas_heating(rates, heat_idx, delta_h_J, heat_veff_mask)` — Q_veff, Q_bulk
    - `compute_e_energy_loss(rates, ei_global, ei_energy_loss, n_ei)` — P_inel fallback
    - `compute_e_loss_rate(rates, e_loss_indices, e_loss_stoich_e)` — S_e_loss
    - `compute_diffusion_rate(T_gas, Te_eV, Lambda_sq)` — D_a/Λ²
    - `compute_energy_rhs(ne_eps, c_e, T_gas, ...)` — electron energy equation
    - `compute_Tgas_rhs(T_gas, Q_elastic, tau, ...)` — gas temperature
    - `compute_flow_source(c, T_gas, x_inlet, ...)` — CSTR/PFR flow
  - 상수: NA, QE, KB, R_GAS, T_STP, P_STP를 모듈 레벨에서 정의 (numba에서 사용 가능)
  - @njit 제약: np.clip 대신 수동 min/max, dict/class 불가, np.searchsorted 가능

  **Must NOT do**:
  - 기존 Python 코드 수정
  - Jacobian 함수의 Numba화

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - `plasma0d_v2/boltzmann.py:364-420` — get_rate_coefficients, get_transport 구현
  - `plasma0d_v2/reactions.py:340-430` — compute_reaction_rates 벡터화 버전
  - `plasma0d_v2/reactions.py:455-465` — compute_gas_heating_split
  - `plasma0d_v2/reactions.py:470-490` — compute_electron_energy_loss, compute_electron_loss_rate
  - `plasma0d_v2/electron_kinetics.py:28-86` — compute_energy_rhs
  - `plasma0d_v2/gas_thermal.py:27-60` — compute_Tgas_rhs
  - `plasma0d_v2/flow.py:60-80` — compute_flow_source

  **Acceptance Criteria**:
  - [ ] 모든 @njit 함수가 컴파일 에러 없이 import됨
  - [ ] 각 함수가 Python 버전과 동일 입력에 동일 출력 (1e-6 이내)

  **QA Scenarios**:
  ```
  Scenario: @njit 함수 컴파일 확인
    Tool: Bash
    Steps:
      1. python -c "from plasma0d_v2.numba_core import rhs_numba; print('OK')"
    Expected: 'OK' 출력, 에러 없음
  ```

  **Commit**: NO

---

- [ ] 2. rhs_numba() 메인 함수 + extract_numba_params()

  **What to do**:
  - `numba_core.py`에 `rhs_numba(t, y, ...)` 메인 @njit 함수 구현
    - solver.py rhs() (L530-729)의 전체 로직을 @njit으로 재현
    - 3-stage afterglow transition 포함
    - QN Mode B는 제외 (거의 사용 안 함)
    - energy_source는 'constant' 모드만 지원
    - P_dep_Wm3는 외부에서 전달 (power.get_power_density는 Python에서 호출)
  - `extract_numba_params(solver)` 함수 구현
    - solver, lut, rxn, flow, gth 객체에서 모든 numpy array 추출
    - dict로 반환
    - 한 번만 호출 (solver 초기화 시)
  
  **Key data to extract**:
  ```
  From boltzmann.py (LUT):
    log_eps_grid, log_k_table, k_dead_mask, sigma_over_N_mask
    _eps_grid, _power_N, _elastic_power_N, _inelastic_power_N
    _k_thermal_cache
    
  From reactions.py:
    stoich_matrix, _veff_mask
    _ei_global_idx, _ei_bolsig_idx, _ei_target_idx
    _arr_global_idx, _arr_A, _arr_n, _arr_E, _arr_order, _arr_idx_a, _arr_idx_b, _arr_n_reactants
    _te_global_idx, _te_A_cgs, _te_n_Te, _te_k3_cgs, _te_idx_a, _te_idx_b, _te_n_reactants, _te_target_idx
    _gas_heating_idx_arr, _delta_h_J, _gas_heat_veff_mask
    _electron_loss_indices
    각 EI reaction의 energy_loss_eV
    각 TE reaction의 subtype (int: 0=DR/constant, 1=AT1_KOSSYI)
    
  From flow.py:
    x_inlet, V_reactor, Q_slm, P_gas, flow_model=='PFR'
    
  From solver.py:
    _f_species, _ce_floor, _ne_eps_floor, _concentration_floor
    _eps_min_lut, _A21_at_boundary, _A22_at_boundary
    _positive_ion_indices, _negative_ion_indices
    _vol_ratio
    
  From electron_kinetics.py:
    Lambda_sq
    
  From gas_thermal.py:
    T_wall, wall_loss_freq, M_avg, cp_avg, P_gas
  ```

  **Recommended Agent Profile**:
  - **Category**: `deep`

  **Parallelization**:
  - **Blocked By**: Task 1

  **References**:
  - `plasma0d_v2/solver.py:530-729` — rhs() 전체
  - Task 1의 모든 헬퍼 함수

  **Acceptance Criteria**:
  - [ ] extract_numba_params(solver)가 에러 없이 dict 반환
  - [ ] rhs_numba(0.0, y0, **params) 호출 성공

  **Commit**: NO

---

- [ ] 3. solver.py에 numba RHS 통합

  **What to do**:
  - solver.py에 `_setup_numba_params(self)` 메서드 추가
    - `extract_numba_params(self)`를 호출하여 `self._nb_params` 저장
    - solver 초기화 시 한 번 호출
  - `rhs(self, t, y)` 수정:
    - `self._nb_params`가 있으면 rhs_numba 사용
    - P_dep_Wm3는 `self.power.get_power_density(t)`로 Python에서 계산 후 전달
    - fallback: `self._nb_params`가 없으면 기존 Python rhs 사용
  - config.py 또는 solver 생성 시 `_setup_numba_params()` 호출 추가

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Blocked By**: Task 2

  **References**:
  - `plasma0d_v2/solver.py:519-729` — 현재 rhs()
  - `plasma0d_v2/numba_core.py` — rhs_numba, extract_numba_params

  **Acceptance Criteria**:
  - [ ] solver.rhs(0.0, y0)가 numba 경로로 실행됨
  - [ ] Python fallback도 여전히 작동

  **Commit**: NO

---

- [ ] 4. 정확성 검증 + 벤치마크

  **What to do**:
  - 검증 스크립트:
    ```python
    dy_py = solver_python_rhs(0.0, y0)
    dy_nb = solver_numba_rhs(0.0, y0)
    rel_err = np.max(np.abs(dy_py - dy_nb) / (np.abs(dy_py) + 1e-30))
    assert rel_err < 1e-6
    ```
  - 벤치마크:
    ```python
    # Warm up (첫 호출은 JIT 컴파일 포함)
    solver.rhs(0.0, y0)
    # 10,000 iterations
    t0 = time.time()
    for _ in range(10000): solver.rhs(0.0, y0)
    dt = (time.time()-t0)/10000*1e6  # µs
    print(f'Numba RHS: {dt:.1f} µs (target: <10 µs)')
    ```
  - 여러 상태점에서 검증:
    - 초기 상태 (ne=1e8, thermal)
    - ON plateau (ne=2.87e14, Te=2.5eV)
    - OFF afterglow (ne=1e9, Te=0.026eV)
    - Steady state (continuous 5W)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Blocked By**: Task 3

  **Acceptance Criteria**:
  - [ ] 4개 상태점에서 rel_err < 1e-6
  - [ ] Numba RHS < 10 µs/call

  **QA Scenarios**:
  ```
  Scenario: 정확성 검증
    Tool: Bash
    Steps:
      1. python -c "..." (위 검증 코드)
    Expected: rel_err < 1e-6 출력

  Scenario: 속도 벤치마크
    Tool: Bash
    Steps:
      1. 10,000회 반복 벤치마크
    Expected: < 10 µs/call
  ```

  **Commit**: NO

---

- [ ] 5. Continuous + Pulsed regression

  **What to do**:
  - Continuous PFR 4온도 sweep (303, 373, 453, 523K)
    - 기대값: 4.95, 7.18, 13.18, 19.92%
    - 허용 오차: ±0.1%p
  - Pulsed 2 cycles (200µs) n_e/Te profile
    - ON plateau ne ≈ 2.87e14, Te ≈ 2.5eV
    - OFF decay ne → ~1e8 (3-body AT)
  - 결과가 Python RHS와 동일한지 확인

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 4)
  - **Blocked By**: Task 3

  **Acceptance Criteria**:
  - [ ] 4온도 conversion 일치 (±0.1%p)
  - [ ] Pulsed profile 일치

  **Commit**: YES
  - Message: `feat(numba): add Numba @njit RHS for 20x speedup`
  - Files: `numba_core.py, solver.py, config.py`

---

## Final Verification Wave

- [ ] F1. **정확성 최종 검증** — `unspecified-high`
  4개 상태점 × Python vs Numba 비교. rel_err < 1e-6.

- [ ] F2. **성능 최종 벤치마크** — `unspecified-high`
  Numba RHS < 10µs. Pulsed 1 period wall time 측정.

---

## Success Criteria

### Verification Commands
```bash
# 컴파일 확인
python -c "from plasma0d_v2.numba_core import rhs_numba; print('OK')"

# 정확성
python -c "
from plasma0d_v2 import ...
assert max_rel_err < 1e-6
"

# 벤치마크
python -c "
# 10000회 벤치마크 → <10µs 확인
"

# Regression
python -c "
# 4온도 PFR → 4.95/7.18/13.18/19.92%
"
```

### Final Checklist
- [ ] Numba RHS 작동
- [ ] Python fallback 유지
- [ ] 정확성 1e-6 이내
- [ ] 속도 <10µs
- [ ] 4온도 regression 통과
- [ ] Pulsed profile 일치
