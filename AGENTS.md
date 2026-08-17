# AGENTS.md

이 repository는 Physical AI 시스템을 완성하기 위한 UR3 capability입니다.

최종 목표는 repository 자체의 완성이 아니라 다음 폐루프를 가장 빠르고 정확하게 만드는 것입니다.

현실 인식 → 판단 → 계획 → 물리 행동 → 실제 결과 측정 → 학습/개선

- 실제 repository/source/Git/test evidence를 자연어 보고보다 우선합니다.
- 확인되지 않은 것은 `UNKNOWN` / `UNRESOLVED` / `NOT_VERIFIED` / `HYPOTHESIS`로 유지합니다.
- 한 번에 첫 미해결 경계 하나만 처리합니다.
- 기존 구현 재사용과 최소 변경을 우선합니다.
- Gate와 무관한 refactor, 새 framework, 새 abstraction, 새 manager/orchestrator, 새 simulator를 만들지 않습니다.
- 사용자 미커밋 변경을 덮어쓰거나 제거하지 않습니다.
- 실제 UR3 motion, 새 calibration, robot/network/controller activation, 실제 카메라/물리 장비 조작은 사용자의 별도 명시 승인 없이는 수행하지 않습니다.
- 한국어 존댓말로 간결하게 보고합니다.

Gate가 닫히면 evidence를 확인하고 STOP합니다. 다음 Gate 구현까지 자동으로 확장하지 않습니다.
