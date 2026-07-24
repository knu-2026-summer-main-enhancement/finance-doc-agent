# Backend Guide

이 문서는 Finance Document Agent 백엔드의 실행 방법과 구성 요소를 설명합니다. 사용자 기능 소개는 루트 [README](../README.md)를 참고하세요.

## 제공 기능

- `/chat`: 자동 질문 분류 후 표 조회 또는 문서 검색
- `/chat/stream`: 문서 검색 스트리밍 답변
- `/chat/suggestions`: 문서 스키마 기반 질문 자동완성 카탈로그
- `/chat/details/{reference}`: 인물 카드와 계산 기여 행 페이지 조회
- `/ingest`: 문서 적재와 데이터프레임 갱신
- `/summary`: 적재 문서 요약
- `/health`: 서버, LLM, 벡터 저장소 상태 확인
- `/ui`: 정적 채팅 UI

## 데이터 저장

표와 문서 본문은 목적에 따라 분리해 저장합니다.

| 저장소 | 용도 |
|---|---|
| Parquet | 명단, 금액, 날짜, 식별자 등 표 데이터 |
| Semantic sidecar | 원본 컬럼의 의미, 자료형, 단위, 개인정보 여부 |
| ChromaDB | 문서 설명과 행 텍스트의 의미 검색 |
| PostgreSQL manifest | 파일 적재 상태, 해시, 스키마 버전 |

문서 적재 중에는 기존 DataFrame 스냅샷을 유지하고, 새 파일 로딩이 끝난 뒤 공유 상태를 교체합니다. 따라서 큰 파일을 다시 적재하는 동안 질문이 빈 데이터셋을 보지 않습니다.

## 주요 모듈

| 파일 | 역할 |
|---|---|
| `main.py` | FastAPI 엔드포인트와 질문 처리 흐름 |
| `datastore/state.py` | 적재된 DataFrame과 소스 메타데이터 |
| `datastore/scope.py` | 요청별 선택 문서 범위 |
| `datastore/schema.py` | LLM에 전달할 런타임 스키마 |
| `datastore/query.py` | 하위 호환 직접 조회 |
| `rag/` | 질문 분류, 라우팅, 문서 검색 |
| `pandas_engine/` | QueryPlan 검증과 결정적 실행 |
| `utils/` | 문서 파서, 적재, 의미 스키마 |
| `static/` | 채팅 화면 |

상세 설명:

- [Document Ingestion](utils/README.md)
- [Question Routing](rag/README.md)
- [Query Execution](pandas_engine/README.md)
- [Testing](tests/README.md)

## 환경 준비

Python 가상환경:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env.example`을 참고해 PostgreSQL, ChromaDB, Ollama 설정을 준비합니다. 실제 `.env`는 커밋하지 않습니다.

주요 기본값:

```text
API/UI: http://localhost:8080
Ollama: qwen2.5:3b
Embedding: bge-m3
```

## 인프라 실행

프로젝트에 포함된 Compose 설정을 사용하는 경우:

```powershell
docker compose up -d
```

Ollama에 필요한 모델이 없다면 먼저 내려받습니다.

```powershell
ollama pull qwen2.5:3b
ollama pull bge-m3
```

## 서버 실행

```powershell
cd backend
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8080
```

UI:

```text
http://localhost:8080/ui
```

격리된 평가 서버는 8081 같은 별도 포트를 사용합니다.

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8081
```

평가 전에 해당 포트의 프로세스 실행 경로와 작업 디렉터리가 현재 저장소인지 확인해야 합니다.

## 운영 안전장치

- 질문 원문, 이름, 전화번호, 이메일을 로그에 기록하지 않습니다.
- 질문은 해시 기반 `question_id`와 글자 수로만 추적합니다.
- 내부 예외 경로와 DB 정보는 API 응답에 노출하지 않습니다.
- 상세조회 캐시는 15분 TTL과 최대 256개 제한을 사용합니다.
- 계산 기여 행은 요청한 페이지에 대해서만 JSON으로 변환합니다.
- 연락처는 일반 응답이 아닌 사용자 요청 상세 카드에서만 제공합니다.

