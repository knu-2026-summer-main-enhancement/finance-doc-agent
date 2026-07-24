# 개발자 가이드

이 문서는 처음 프로젝트를 인계받은 개발자가 코드의 위치와 변경 순서를 빠르게 찾기 위한 안내서입니다. 사용자 기능은 루트 `README.md`, 실행 방법은 `backend/README.md`를 먼저 참고하세요.

## 10분 안에 구조 파악하기

사용자 질문은 다음 순서로 처리됩니다.

```text
static/app.js
  → main.py (/chat)
  → datastore/scope.py (선택 문서 제한)
  → rag/question_decision.py (질문의 의도 계약)
  → rag/deterministic_query_plan.py 또는 rag/query_planner.py
  → pandas_engine/plan_validator.py
  → pandas_engine/query_executor.py
  → pandas_engine/formatter.py + interactive.py
  → 구조화된 API 응답
  → static/app.js (본문, 인물 카드, 금액 상세 렌더링)
```

표로 답할 수 없는 문서 설명 질문은 `rag/vector.py`로 분기합니다. 브라우저는 한국어 답변 문자열을 다시 분석하지 않으며, 인물·금액·페이지 정보는 API의 구조화 필드를 사용합니다.

문서 적재는 다음 순서입니다.

```text
main.py (/ingest)
  → utils/ingest.py
  → utils/parsers/*
  → utils/table_ingest_pipeline.py
  → utils/semantic_schema.py
  → Parquet + semantic sidecar + ChromaDB
  → datastore/state.py의 메모리 스냅샷 교체
```

## 변경 목적별 시작 파일

| 바꾸려는 기능 | 먼저 볼 파일 | 함께 확인할 파일 |
|---|---|---|
| API 계약·응답 필드 | `main.py` | `static/app.js`, `tests/test_main_question_routing.py` |
| 질문 operation 판정 | `rag/question_decision.py` | `question_engine.py`, `question_detectors.py` |
| 정규식 기반 빠른 계획 | `rag/deterministic_query_plan.py` | `pandas_engine/query_plan.py` |
| LLM 기반 계획 | `rag/query_planner.py` | `rag/prompts.py`, `plan_validator.py` |
| 날짜 범위 | `pandas_engine/date_filter.py` | `deterministic_query_plan.py`, `tests/test_date_query.py` |
| 금액·순위·집계 | `pandas_engine/aggregation.py`, `money.py` | `query_executor.py`, `formatter.py` |
| 이름·마스킹 이름 | `utils/table_parser.py` | `query_grounding.py`, `interactive.py` |
| 인물 카드·금액 상세 | `pandas_engine/interactive.py` | `main.py`, `static/app.js` |
| 자동완성 | `rag/question_suggestions.py` | `main.py`, `static/app.js` |
| 새 파일 형식 | `utils/parsers/` | `table_ingest_pipeline.py`, `semantic_schema.py` |
| 컬럼 의미 추론 | `utils/semantic_schema.py` | `datastore/schema.py`, `plan_validator.py` |
| 벡터 문서 검색 | `rag/vector.py` | `core/llm.py`, `rag/prompts.py` |
| 보안·개인정보 | `core/security.py`, `privacy.py` | `main.py`, 관련 테스트 |

## 디렉터리별 책임

- `core/`: 환경 설정, LLM·벡터 저장소 연결, 인증과 개인정보 로그 정책
- `datastore/`: 현재 적재된 표의 공유 상태, 요청별 문서 범위, 런타임 스키마
- `rag/`: 질문의 의미 판정, 빠른 계획·LLM 계획 선택, 벡터 검색
- `pandas_engine/`: 실행 계약인 QueryPlan, 검증, 필터·집계 실행, 출력 생성
- `utils/`: 원본 파일 파싱, 표 정리, 의미 스키마 생성, 저장
- `static/`: 구조화된 API 응답을 보여주는 브라우저 UI
- `tests/`: 단위·통합 테스트, 골드셋 평가 도구

각 디렉터리의 세부 흐름은 해당 `README.md`에 있습니다.

## 반드시 유지할 설계 규칙

1. 특정 문서명, 사람 이름, 연도, 골드셋 ID 또는 정답을 production 코드에 하드코딩하지 않습니다.
2. 컬럼명 자체보다 semantic sidecar와 런타임 스키마를 우선 사용합니다.
3. LLM이 만든 계획도 `plan_validator.py`를 통과한 뒤에만 실행합니다.
4. 인물과 금액 링크는 렌더링된 문장을 파싱하지 않고 구조화 응답을 사용합니다.
5. 동명이인은 임의로 합치지 않습니다. 이름이 같아도 서로 다른 원본 행은 구분합니다.
6. 전화번호와 이메일은 요청된 상세 카드 외의 로그·보고서에 남기지 않습니다.
7. “몇 명”은 행 수가 아니라 고유 인물 수인지 확인합니다.
8. 날짜 컬럼 후보가 여러 개면 암묵적으로 하나를 고르지 말고 명시된 해석 정책을 따릅니다.
9. 적재 중 공유 DataFrame을 먼저 비우지 않습니다. 완성된 새 스냅샷으로 원자적으로 교체합니다.

## 오류를 추적하는 순서

1. 요청이 올바른 문서 범위로 제한됐는지 확인합니다.
2. 질문 판정 결과의 operation과 상태를 확인합니다.
3. 생성된 QueryPlan의 필터, 대상 컬럼, 정렬, limit을 확인합니다.
4. validator가 계획을 바꾸거나 거절한 이유를 확인합니다.
5. executor 결과를 원본 Excel 또는 Parquet 행과 비교합니다.
6. formatter와 interactive payload가 같은 행을 가리키는지 확인합니다.
7. 마지막으로 브라우저 렌더링을 확인합니다.

이 순서를 지키면 라우팅 실패, 계획 실패, 실행 실패, 화면 표시 실패를 섞어서 디버깅하는 일을 줄일 수 있습니다.

## 변경 후 확인

백엔드 루트에서 다음을 기본 확인으로 사용합니다.

```powershell
.\venv\Scripts\python.exe -m pytest -q
node --check static/app.js
```

골드셋은 이 workspace의 격리 서버를 별도 포트로 실행한 뒤 평가합니다. 이미 떠 있는 `localhost:8080` 서버가 다른 checkout일 수 있으므로 프로세스 경로와 작업 디렉터리를 먼저 확인하세요.
