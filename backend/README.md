# Backend Guide

Finance Document Agent 백엔드의 실행 방법, API, 데이터 처리 구조를 설명합니다. 사용자에게 보이는 기능은 루트 [README](../README.md), 세부 구현은 각 모듈 문서를 참고하세요.

## 현재 지원 범위

- Excel (`.xlsx`)
- 텍스트·표·스캔 PDF
- HWP/HWPX 본문과 표
- 표 이미지 (`.png`, `.jpg`, `.jpeg` 등)

표 데이터는 스키마 기반 결정적 실행 경로로 조회하고, 설명형 문서는 섹션 기반 Vector RAG 경로로 검색합니다. 파일명, 문서명, 사람 이름, 연도 또는 골드셋 정답을 production 코드에 하드코딩하지 않습니다.

## 질문 처리 구조

### 구조화 데이터

명단, 금액, 날짜, 집계, 순위처럼 정확한 실행이 필요한 질문은 QueryPlan으로 처리합니다.

```text
질문
→ Schema-Grounded Planner
→ QueryPlan 검증
→ 결정적 필터·집계·정렬 실행
→ 답변 + 인물 entity + 계산 reference
```

- 자주 쓰는 질문은 LLM 없이 현재 문서 스키마에 직접 연결합니다.
- 빠른 계획을 안전하게 만들 수 없는 질문만 LLM 분석으로 넘깁니다.
- LLM은 임의 Python 코드를 실행하지 않고 제한된 QueryPlan만 제안합니다.
- 실행 전 실제 DataFrame, 컬럼, 자료형, 연산자와 질문 근거를 검증합니다.
- 행별 금액 순위와 인물별 누적 금액 순위를 구분합니다.
- 동명이인과 마스킹 이름을 임의로 병합하지 않습니다.

### 설명형 문서

목적, 대상, 기준, 절차, 일정, 조건처럼 문맥 이해가 필요한 질문은 선택 문서 범위 안에서 검색합니다.

```text
질문
→ 검색 질의 구성
→ 선택 문서의 child chunk 검색
→ reranking
→ parent section 확장
→ 근거 기반 LLM 답변
```

- PDF/HWP의 제목, 상위 섹션, 본문, 표 행을 함께 저장합니다.
- 각 문서에는 `document_type`, 섹션 목록, 지원 capability가 기록됩니다.
- 명시한 섹션 제목과 일반적인 표현을 모두 검색할 수 있습니다.
- 검색된 근거가 부족하면 파일명이나 일반 지식으로 답을 추측하지 않습니다.
- 수치 조건 질문은 검색된 문서 기준과 사용자가 제시한 값을 함께 비교합니다.

## 문서 적재

```text
업로드
→ 형식별 파서
→ 본문·표·섹션 추출
→ 공통 표 정제
→ 의미 스키마 생성
→ Parquet + semantic sidecar + ChromaDB + manifest
→ 공유 DataFrame 갱신
```

### 형식별 동작

| 형식 | 처리 방식 |
|---|---|
| Excel | 시트·병합 셀·다단 헤더를 복원하고 표로 저장 |
| PDF | 본문과 표를 추출하고 스캔 페이지는 가능한 경우 OCR 수행 |
| HWP/HWPX | 본문 구조와 표를 분리하고 실제 상위 섹션을 식별 |
| 표 이미지 | 셀 위치를 감지한 뒤 OCR 결과를 원본 열에 맞춰 구조화 |

원본 컬럼은 유지하고 이름, 금액, 날짜, 범주, 식별자, 개인정보, 품질 메타데이터 역할을 sidecar에 기록합니다. 컬럼 이름, 순서, 개수가 달라도 런타임 스키마를 다시 구성할 수 있습니다.

적재 중에는 기존 DataFrame 스냅샷을 유지하고 새 스냅샷이 완성된 뒤 공유 상태를 교체합니다. UI는 상태 API를 폴링해 새로고침 없이 `in_progress`에서 완료 상태로 전환합니다.

## 구조화 응답

`/chat`은 답변 문자열 외에도 UI가 직접 사용할 수 있는 구조화 데이터를 반환합니다.

- 답변 내 인물·금액 위치를 나타내는 inline segment
- 동명이인을 분리한 person entity
- 인물의 원본 컬럼과 원본 행
- 집계 operation, 대상 컬럼, 필터, 유효·제외 행 수
- 계산 기여 행 상세조회 reference
- 목록 및 상세 데이터 페이지 정보
- 사용 문서와 검색 근거

브라우저는 한국어 답변 문자열에서 이름이나 금액을 다시 추출하지 않습니다. 인물 카드와 금액 계산 근거는 API가 제공한 reference로 조회합니다.

개인정보 정책:

- 일반 답변과 로그에는 전화번호·이메일을 노출하지 않습니다.
- 연락처는 사용자가 요청한 인물 상세 카드에서만 제공합니다.
- OCR confidence, 내부 행 식별자, 검색용 파생 컬럼은 인물 카드에서 제외합니다.
- 표시할 추가 원본 정보가 있을 때만 상세 UI의 더보기 기능을 제공합니다.

## 주요 API

| Method | Endpoint | 역할 |
|---|---|---|
| `GET` | `/`, `/ui` | 모바일 중심 정적 채팅 UI |
| `GET` | `/health` | 서버·LLM·Vector 저장소 상태 |
| `POST` | `/chat` | 질문 분류, 실행, 구조화 응답 |
| `POST` | `/chat/stream` | 설명형 문서 답변 스트리밍 |
| `POST` | `/chat/cancel/{request_id}` | 진행 중인 스트리밍 요청 취소 |
| `POST` | `/chat/suggestions` | 문서 스키마·섹션 기반 자동완성 카탈로그 |
| `POST` | `/chat/person-suggestions` | 대규모 인물 접두사 검색 |
| `GET` | `/chat/details/{reference}` | 계산 기여 행 페이지 조회 |
| `GET` | `/chat/results/{reference}/person/{row_index}` | 선택한 원본 행의 인물 상세 |
| `POST` | `/ingest` | 서버에 존재하는 파일 적재 |
| `POST` | `/ingest/upload` | 업로드 파일 저장 및 비동기 적재 |
| `POST` | `/ingest/all` | 데이터 폴더의 지원 파일 일괄 적재 |
| `GET` | `/status` | 파일별 적재 진행 상태 |
| `GET` | `/documents` | 적재 문서와 capability 목록 |
| `GET` | `/documents/{source}/sections` | 문서의 상위 섹션 목록 |
| `GET` | `/documents/{source}/sections/{section_id}` | 섹션 메타데이터와 상세 내용 |
| `PATCH` | `/documents/{source}` | 문서 표시 이름 변경 |
| `DELETE` | `/documents/{source}` | 문서와 관련 저장 데이터 삭제 |
| `GET` | `/summary` | 적재 데이터 요약 |

## 자동완성

문서를 선택할 때 서버가 실행 가능한 질문 카탈로그를 제공합니다. 이후 일반적인 입력 필터링은 브라우저에서 수행하므로 키 입력마다 API나 LLM을 호출하지 않습니다.

- 전체 목록·기록 수·합계
- 문서에 존재하는 인물의 금액·기록·원본 필드
- 연도·월·기간 목록과 집계
- 평균·중앙값·최댓값·최솟값·순위
- 설명형 문서의 전체 섹션 및 주요 섹션 질문
- 여러 문서에 공통으로 존재하는 스키마 질문

화면에는 통합 순위가 높은 후보를 최대 3개만 표시합니다. 문서 규모가 큰 경우 인물 접두사 API를 별도로 사용합니다.

## 데이터 저장

| 저장소 | 용도 |
|---|---|
| Parquet | 명단, 금액, 날짜, 식별자 등 구조화 표 |
| Semantic sidecar | 원본 컬럼 의미, 자료형, 단위, 개인정보·품질 분류 |
| ChromaDB | PDF/HWP 본문, 섹션 child chunk, 표 행 검색 |
| PostgreSQL manifest | 파일 해시, 적재 상태, 파서·스키마 버전 |

## 주요 모듈

| 파일 | 역할 |
|---|---|
| `main.py` | FastAPI endpoint, 문서 범위, 모드 dispatch |
| `datastore/state.py` | DataFrame snapshot과 source metadata |
| `datastore/scope.py` | 요청별 선택 문서 범위 |
| `datastore/schema.py` | 런타임 스키마와 LLM 전달 계약 |
| `rag/question_engine.py` | LLM 질문 분류 |
| `rag/deterministic_query_plan.py` | 스키마 기반 빠른 QueryPlan |
| `rag/query_planner.py` | LLM QueryPlan 생성 |
| `rag/vector.py` | 문서 검색, reranking, parent section 확장 |
| `rag/question_suggestions.py` | 자동완성 후보 생성 |
| `pandas_engine/` | 계획 검증, 실행, 답변, interactive payload |
| `utils/` | 형식별 파서, 공통 적재, 의미 스키마 |
| `static/` | 모바일 채팅 UI |

세부 문서:

- [Developer Guide](DEVELOPER_GUIDE.md)
- [Document Ingestion](utils/README.md)
- [Question Routing](rag/README.md)
- [Query Execution](pandas_engine/README.md)
- [Testing and Goldset](tests/README.md)

## 환경 준비

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env.example`을 참고해 PostgreSQL, ChromaDB, Ollama 설정을 준비합니다. 실제 `.env`는 커밋하지 않습니다.

기본 개발 환경:

```text
API/UI: http://localhost:8080
Local LLM: qwen2.5:3b
Embedding: bge-m3
```

필요한 인프라와 모델:

```powershell
docker compose up -d
ollama pull qwen2.5:3b
ollama pull bge-m3
```

## 서버 실행

```powershell
cd backend
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8080
```

브라우저에서 `http://localhost:8080/ui`를 엽니다.

격리 평가에는 8081처럼 별도 포트를 사용합니다.

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8081
```

평가 전에 해당 포트의 프로세스 경로와 작업 디렉터리가 이 저장소인지 확인해야 합니다. 다른 checkout이 사용하는 `localhost:8080` 결과를 현재 코드의 근거로 사용하지 않습니다.

## 테스트

`backend/`에서 실행합니다.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

격리 서버에 대한 Excel goldset:

```powershell
.\venv\Scripts\python.exe tests\eval.py `
  --url http://127.0.0.1:8081 `
  --tag backend_readme_check
```

Excel goldset은 `tests/goldsets/goldset.json`, 문서별 Vector goldset은
`tests/goldsets/vector_goldset_*.json`에서 관리합니다. 실행 결과는
`tests/results/`에 저장합니다. 자세한 실행법과 실패 분류 기준은
[Testing and Goldset](tests/README.md)을 참고하세요. 정답률은 키워드
포함만으로 판단하지 않고 route, filter, 반환 행, scalar, 인원 수,
근거 섹션을 원본과 비교합니다.

## 운영 안전장치

- 질문 원문, 이름, 전화번호, 이메일을 로그에 기록하지 않습니다.
- 질문은 해시 기반 `question_id`와 글자 수로만 추적합니다.
- 내부 예외 경로와 DB 접속 정보는 API 응답에 노출하지 않습니다.
- 상세조회 저장소는 TTL과 최대 항목 수 제한을 둡니다.
- 계산 기여 행은 요청한 페이지만 JSON으로 변환합니다.
- 문서 범위가 선택되면 다른 문서의 검색 근거를 섞지 않습니다.
