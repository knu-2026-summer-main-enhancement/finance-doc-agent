# Testing and Goldset Evaluation

이 문서는 기존 README의 검증 방향과 현재 테스트 실행 방법을 정리합니다.

## 검증 원칙

정답률은 키워드 포함 여부만으로 판단하지 않습니다.

- 질문이 올바른 엔진으로 라우팅되었는지
- QueryPlan이 실제 컬럼과 조건을 사용했는지
- 반환 행이 모든 조건을 만족하는지
- 금액이 원본 데이터와 같은 단위로 계산되었는지
- 사람 수가 행 수가 아니라 고유 인원 수인지
- 개인별 합계와 개별 지급 행을 구분했는지
- 동점 순위 정책이 실행기와 답변에서 일치하는지
- 평가기의 오탐인지 실제 프로그램 오류인지

## 전체 단위 테스트

`backend/`에서 실행합니다.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

현재 기준 전체 회귀 테스트는 310개입니다.

## 주요 테스트 범위

| 테스트 | 검증 내용 |
|---|---|
| `test_semantic_schema.py` | 컬럼 의미, 단위, 개인정보 |
| `test_table_ingest_pipeline.py` | 공통 적재와 다중 시트 |
| `test_deterministic_query_plan.py` | 스키마 기반 빠른 계획 |
| `test_query_planner.py` | LLM QueryPlan 생성·복구 |
| `test_plan_validator.py` | 컬럼·자료형·근거 검증 |
| `test_query_executor.py` | 필터·집계·순위 |
| `test_date_query.py` | 월, 연도, 기간 |
| `test_interactive_result.py` | 인물 카드와 계산 상세 |
| `test_question_suggestions.py` | 자동완성 |
| `test_privacy.py` | 질문 로그와 오류 정보 노출 |

## Goldset

활성 원본은 `Result_1.xlsx`, 평가 질문은 `tests/goldset.json`입니다.

평가 전에 현재 작업공간 서버인지 확인하고 8081 같은 격리 포트를 사용합니다.

```powershell
.\venv\Scripts\python.exe tests\eval.py `
  --url http://127.0.0.1:8081 `
  --tag schema_review
```

특정 항목:

```powershell
.\venv\Scripts\python.exe tests\eval.py `
  --url http://127.0.0.1:8081 `
  --id R005 `
  --tag debug `
  -v
```

## 실패 분류

실패는 다음 순서로 분리합니다.

1. Routing failure
2. Question planning failure
3. Plan validation failure
4. Execution failure
5. Formatting or interactive payload failure
6. Evaluator false negative

수정 전에는 원본 Excel 행과 기대값을 직접 비교합니다. 골드셋의 이름, 연도, 금액, 테스트 ID를 production 코드에 하드코딩하지 않습니다.

## 스키마 변형 테스트

새 파일 구조를 검증할 때 다음 변형을 함께 테스트합니다.

- 컬럼 순서 변경
- 불필요한 컬럼 추가
- 사람 이름 헤더 변경
- 금액 헤더와 단위 표현 변경
- 완전한 날짜와 연도·월 분리
- 다중 금액 컬럼
- 다중 날짜 컬럼
- 동일 이름과 마스킹 이름
- 200건 이상의 목록
- 재적재 중 동시 질문

