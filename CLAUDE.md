# Project Instructions

## Session Logging (MANDATORY)

매 세션에서 수행한 작업을 해당 프로젝트의 CLAUDE.md Session History에 일자별로 기록할 것.
- 날짜 + 핵심 변경사항/발견 요약 (한 줄~여러 줄)
- 코드 변경, 시뮬레이션 결과, 발견한 문제, 결정사항 포함
- compact 전에 반드시 기록 완료
- 사용자가 터미널을 그냥 닫아도 CLAUDE.md는 이미 최신 상태여야 한다

## Auto Skills (MANDATORY - 반드시 따를 것)

아래 조건에 해당하면 해당 skill 파일(~/.claude/skills/)을 읽고 반드시 적용해야 한다. 선택이 아닌 필수 규칙이다.

### 항상 적용
- 코드 작성 시 → ~/.claude/skills/python-patterns.md 읽고 적용
- 컨텍스트 70% 이상 차면 → ~/.claude/skills/strategic-compact.md 읽고 실행
- 토큰 소모 큰 작업 전 → ~/.claude/skills/context-budget.md 읽고 예산 판단

### 코드 변경 시
- Python 코드 수정 후 → ~/.claude/skills/security-review.md 읽고 실행
- 새 기능 추가 시 → ~/.claude/skills/tdd-workflow.md, ~/.claude/skills/python-testing.md 읽고 적용
- 수정 후 → ~/.claude/skills/verification-loop.md 읽고 결과 검증

### 복잡한 작업 시
- 파일 3개 이상 동시 수정 → ~/.claude/skills/dmux-workflows.md 읽고 병렬 처리
- 설계 판단 필요 시 → ~/.claude/skills/council.md 읽고 다중 관점 검토
- 버그 원인 불명 시 → ~/.claude/skills/deep-research.md 읽고 조사
- 장시간 반복 작업 → ~/.claude/skills/autonomous-loops.md 읽고 자율 실행

### 프로젝트 시작 시
- 새 프로젝트 진입 → ~/.claude/skills/codebase-onboarding.md 읽고 파악
- 저장소 상태 확인 → ~/.claude/skills/repo-scan.md 읽고 실행

### Git 작업 시
- 커밋/PR → ~/.claude/skills/git-workflow.md 읽고 적용

### 과학 계산 / 시뮬레이션
- PyTorch/PINN 관련 → ~/.claude/skills/pytorch-patterns.md 읽고 적용
- 수치 발산/NaN 발생 시 → deep-research + council 읽고 원인 분석
- solver 성능 이슈 → verification-loop 읽고 반복 프로파일링
- 실험 데이터 비교 시 → verification-loop 읽고 오차 범위 검증
