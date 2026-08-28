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

## 프로젝트 전역 프로그램 / 시퀀스 재사용 사전확인

`GLOBAL_PROGRAM_SEQUENCE_PRECHECK = MANDATORY`

새 logic, program, runtime, adapter, controller, sequence, task/recovery flow 또는 integration을 구현하기 전에 반드시 기존 PROGRAM / SEQUENCE를 먼저 확인합니다.

확인 순서:

```text
현재 milestone / NEXT_ACTION
→ 기존 VERIFIED/CLOSED capability
→ 프로젝트 전역 PROGRAM / SEQUENCE registry
→ 이 repository 실제 source / entrypoint
→ 필요 시 관련 Physical AI repository 실제 source
→ 기존 VERIFIED runtime / research / maintained upstream
→ 없는 최초 연결만 최소 구현
```

프로젝트 전역 기준표는 현재 다음 source를 사용합니다.

```text
SportyRain/ur3_visual_servoing/docs/PHYSICAL_AI_PROGRAM_REGISTRY.md
```

단, registry는 탐색 지도이며 실제 repository/source가 최종 권위입니다.

다음을 금지합니다.

- 이 repository만 확인하고 capability가 없다고 판정
- 기존 프로그램/시퀀스 확인 없는 신규 구현
- 기존 VERIFIED sequence 재구현
- 기존 기능을 새 manager/orchestrator/framework로 대체

source에는 기존 program/sequence가 있는데 registry에 없으면 `PROGRAM_REGISTRY = STALE`로 판정하고 registry 동기화와 재사용 판정을 신규 구현보다 먼저 수행합니다.

새 executable/module runtime/canonical sequence/runtime contract가 추가·변경된 경우 registry 동기화 전 milestone을 CLOSED 처리하지 않습니다.

## 프로젝트 전역 히스토리 / 기록 동기화

```text
GLOBAL_RECORD_SYNC = MANDATORY
HISTORY_RECORD_FOR_MEANINGFUL_EVENT = MANDATORY
```

상세 정책은 다음 중앙 source를 따릅니다.

```text
SportyRain/ur3_visual_servoing/docs/PHYSICAL_AI_RECORD_POLICY.md
```

의미 있는 milestone 성공, 중요한 실패/root cause, 수정 후 개선, 실제 physical runtime 검증, 중요한 Sim/Real 차이, program/sequence 도입·폐기, 재사용/안전/운영 정책 변경은 history 대상입니다.

이 repository에서 발생한 사건도 Physical AI 프로젝트 전체에 의미가 있으면 중앙 history continuity 대상입니다.

기술적 SUCCESS가 있어도 필요한 기록이 동기화되지 않았다면 milestone을 CLOSED로 보고하지 않습니다.

```text
MILESTONE_STATUS = RECORD_SYNC_PENDING
CLOSED = NO
```

최종 evidence에는 최소한 다음을 포함합니다.

```text
HISTORY_RECORD_REQUIRED = YES/NO
HISTORY_EVENT_SUMMARY =
HISTORY_EVIDENCE =
CURRENT_STATE_UPDATE_REQUIRED = YES/NO
PROGRAM_REGISTRY_UPDATE_REQUIRED = YES/NO
RUNTIME_COMMAND_UPDATE_REQUIRED = YES/NO
RECORD_SYNC_CHECK = PASS/PENDING
```

중앙 history를 직접 수정할 수 없으면 `PENDING`을 숨기지 않고 Supervisor가 동기화할 수 있게 위 evidence를 제출합니다.

Gate가 닫히면 evidence와 기록 동기화를 확인하고 STOP합니다. 다음 Gate 구현까지 자동으로 확장하지 않습니다.