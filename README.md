# Finance Document Agent

장학금·후원금·지원금·예산 명세처럼 **금액과 표가 중심인 문서**를 적재하고, 자연어로 검색·계산·조회할 수 있는 로컬 문서 질의 에이전트입니다.

Excel, PDF, HWP/HWPX뿐 아니라 세로로 긴 표 이미지도 직접 처리합니다. 표 데이터는 Parquet에 구조화해 저장하고, 문서 설명과 행 텍스트는 ChromaDB에 임베딩합니다. 질문이 들어오면 하나의 질문 분석 결과를 기준으로 문서 목록 조회, 결정론적 PANDAS 계산, 근거 기반 VECTOR 검색 중 적절한 경로를 선택합니다.

이 프로젝트의 핵심 목표는 단순한 문서 요약 챗봇이 아니라 다음과 같은 재정 문서 업무를 안전하게 처리하는 것입니다.

- 장학재단 지급·후원 문서 조회
- 지자체 예산·지원금 명세 내부 검색
- 수혜자·기부자·기관 명단 관리
- 금액 합계, 평균, 최댓값, 기간별 집계
- 원본 문서와 계산 근거를 확인할 수 있는 답변

> 현재는 로컬 개발·검증 단계입니다. 역할 기반 접근 제어와 감사 로그는 향후 구현 예정이며, 민감한 실데이터를 운영 환경에 배포하기 전 별도의 보안 검토가 필요합니다.

---

## 핵심 특징

### 1. 여러 문서 형식의 통합 적재

| 입력 형식 | 처리 방식 | 구조화 저장 | 벡터 저장 |
|---|---|---:|---:|
| Excel | 시트별 표 추출 및 공통 정제 | Parquet | ChromaDB |
| PDF | 표와 본문을 분리 추출, 스캔 페이지 OCR 보완 | Parquet | ChromaDB |
| HWP/HWPX | `pyhwpx`로 임시 HTML 변환 후 표 추출 | Parquet | ChromaDB |
| PNG/JPG 등 | OpenCV 표 구조 탐지 + 셀 단위 PaddleOCR | Parquet | ChromaDB |

지원 확장자:

```text
xlsx, pdf, hwp, hwpx,
png, jpg, jpeg, webp, bmp, tif, tiff
```

### 2. 셀 단위 이미지 표 OCR

세로로 길고 글자가 작은 표는 이미지를 일정 길이로만 자르는 방식으로는 행·열과 병합 셀을 안정적으로 복원하기 어렵습니다. 이 프로젝트의 이미지 파서는 다음 순서로 처리합니다.

```text
이미지 로드
→ OpenCV로 가로선·세로선 검출
→ 행·열 경계와 병합 셀 구조 복원
→ 각 셀을 개별 이미지로 분리
→ PaddleOCR로 셀 단위 인식
→ OCR 신뢰도와 보정 이력 기록
→ 첫 번째 표 행을 실제 헤더로 사용
→ DataFrame 정제 및 품질 검증
→ Parquet + ChromaDB 적재
```

특정 연도, 특정 컬럼명, 고정 5열 구조를 코드에 넣지 않습니다. 감지된 열 개수와 실제 헤더를 사용하므로 서로 다른 표 구조를 같은 적재 흐름으로 처리할 수 있습니다.

현재 이미지 파서는 **격자선이 분명한 단일 표 이미지**에 최적화되어 있습니다. 테두리가 없는 표, 한 이미지의 여러 표, 심한 기울기·왜곡은 추가 개선 대상입니다.

### 3. 원본을 보존하는 의미 스키마

모든 문서의 컬럼을 하나의 고정 컬럼 집합에 강제로 맞추지 않습니다. 원본 컬럼을 유지하면서 별도의 의미 스키마를 생성합니다.

예를 들어 `출연금액`, `후원액`, `지급액`은 원본 이름을 보존하되, 신뢰도가 충분하면 금액 의미와 KRW 단위 정보를 메타데이터에 기록합니다. 판단하기 어려운 컬럼은 `unknown`으로 남겨 잘못된 자동 매핑을 피합니다.

스키마에는 다음 정보가 포함됩니다.

- 문서·표·행 식별자
- 원본 컬럼명과 추론된 의미
- 데이터 타입과 단위
- 민감정보·개인정보 여부
- 매핑 신뢰도
- 컬럼 구조 지문과 스키마 버전

DataFrame은 Parquet로, 스키마는 `.schema.json` 사이드카로 저장됩니다. 현재 스키마 버전은 `2.2`입니다.

### 4. 하나의 질문 분석 결과를 공유하는 라우팅

기존처럼 Guard와 Router가 각각 다른 정규식으로 질문을 다시 판단하지 않습니다.

```text
사용자 질문
→ Question Detectors: 표현과 작업 신호 탐지
→ Question Analyzer: 의도·집계·날짜·대상 통합 분석
→ Guard/Guide: 모호하거나 충돌하는 요청 안내
→ Router: DOCUMENTS / PANDAS / VECTOR 실행 경로 선택
```

| 경로 | 용도 | 예시 |
|---|---|---|
| `DOCUMENTS` | 적재 문서 목록 조회 | `전체 문서 보여줘` |
| `PANDAS` | 표 필터링·명단·금액 계산 | `3월 출연금액 합계 알려줘` |
| `VECTOR` | 본문 내용·절차·목적 검색 | `이 장학금의 지급 기준이 뭐야?` |
| `GUIDE` | 모호하거나 복합적인 질문 재작성 안내 | `금액이랑 규정 전부 비교해줘` |

### 5. LLM 대신 검증된 함수로 처리하는 기본 집계

합계나 최솟값 같은 기본 계산을 LLM이 즉석에서 Pandas 코드로 생성하게 두면, `1,000,000` 같은 문자열 금액을 연결하거나 문자열 사전순으로 비교할 수 있습니다. 현재는 다음 연산을 전용 집계 엔진이 처리합니다.

- 인원·행 개수
- 합계
- 평균
- 중앙값
- 최빈값
- 1인당 금액
- 최댓값·최솟값
- 상위 N개·하위 N개
- 개인별 누적 금액 최대·최소

금액 파서는 쉼표, 원, 천원, 만원, 음수와 0을 처리하며, 손상되거나 해석이 모호한 값은 임의로 계산에 포함하지 않습니다.

### 6. 날짜 조건과 집계의 결합

표에서 날짜 컬럼을 찾아 다음과 같은 질문을 처리합니다.

```text
3월에 낸 사람 리스트 알려줘
3~4월 출연금액 합계 알려줘
2025년 4월에 가장 많이 낸 사람은?
3월에 가장 적게 지급된 금액은?
```

하이픈·점·슬래시·한글 날짜·`YYYYMMDD`·Excel 날짜 일련번호를 지원합니다. 여러 연도가 섞였는데 질문에 연도가 없거나 날짜 컬럼이 여러 개라서 판단하기 어려우면 임의로 선택하지 않고 사용자에게 확인을 요청합니다.

### 7. 개인·기관·마스킹 이름 검색

표에 있는 값을 개인 이름으로만 가정하지 않습니다.

- 개인·기관·단체·학과 구분
- 마스킹 이름 정규화 및 패턴 검색
- 기수·발행번호·식별번호 검색
- 같은 이름을 기수·발행번호·행 문맥으로 구분
- 여러 문서에서 같은 이름이 발견되면 문서 선택 요청

현재의 사람 식별자는 문서 안에서 후보를 구분하기 위한 값입니다. 여러 문서를 관통하는 확정 인물 ID는 아직 만들지 않았습니다.

### 8. 문서 범위 격리

`/chat` 요청의 `sources`에 파일명을 전달하면 PANDAS와 VECTOR가 동일한 문서 범위만 사용합니다.

```json
{
  "question": "가장 많이 낸 사람 누구야?",
  "sources": ["test2025.png"]
}
```

선택하지 않은 문서의 DataFrame은 LLM이 생성하는 제한적 Pandas 코드에서도 접근할 수 없습니다. 문서를 선택하지 않은 상태에서 여러 문서의 동명이인이 발견되면 임의로 한 사람을 고르지 않고 문서 선택을 안내합니다.

### 9. 답변 근거와 계산 내역

표 계산 결과에는 가능한 경우 다음 근거가 함께 표시됩니다.

- 사용한 원본 문서
- 계산 대상 컬럼
- 수행한 연산
- 조건에 일치한 행 수
- 실제 계산에 사용한 유효 행 수
- 제외된 행 수
- 적용한 날짜 컬럼과 기간

```text
총 출연금액은 977,070,000원입니다.

계산 근거
- 문서: test2025.png
- 대상 컬럼: 출연금액
- 계산: 합계
- 유효 행: 158행
```

VECTOR 답변은 검색된 문서에 직접적인 근거가 없으면 일반 지식이나 파일명 추측으로 내용을 채우지 않도록 제한합니다.

### 10. 별도 웹 UI와 Swagger UI

FastAPI의 Swagger UI뿐 아니라 문서 업무에 맞춘 간단한 웹 화면을 제공합니다.

- 웹 UI: `http://localhost:8080/ui`
- Swagger UI: `http://localhost:8080/docs`

웹 UI에서 다음 작업을 할 수 있습니다.

- 문서 업로드
- 적재 상태 확인
- 전체 문서 목록 조회
- 질문에 사용할 문서 선택
- 질문 및 답변
- 적재 문서 삭제

---

## 전체 구조

```text
파일 업로드
  ├─ Excel parser
  ├─ PDF text/table parser
  ├─ HWP/HWPX → HTML parser
  └─ Image grid detector + cell OCR
          ↓
공통 표 정제 + 의미 스키마 생성
  ├─ Parquet + schema sidecar
  ├─ ChromaDB text chunks + metadata
  └─ PostgreSQL ingestion manifest

질문 입력 + 선택 문서
          ↓
Question Detectors → Question Analyzer
          ↓
Guard / Guide
          ↓
Router
  ├─ DOCUMENTS → 문서 목록
  ├─ PANDAS    → 필터·집계·날짜·이름 검색
  └─ VECTOR    → 관련도 검색 + 근거 기반 LLM 답변
          ↓
답변 + 출처 + 계산 근거
```

---

## 기술 구성

| 영역 | 기술 |
|---|---|
| API/UI | FastAPI, Pydantic, HTML, CSS, JavaScript |
| 표 처리 | Pandas, PyArrow, OpenPyXL |
| 이미지 OCR | OpenCV, PaddleOCR, PaddlePaddle |
| PDF/HWP | pdfplumber, pdf2image, Tesseract, pyhwpx, BeautifulSoup |
| LLM/Embedding | Ollama, Qwen2.5, BGE-M3, LangChain |
| 저장소 | Parquet, ChromaDB, PostgreSQL |
| 자동화 | Docker Compose, n8n, Slack 연동 워크플로 |

---

## 폴더 구조

```text
finance-doc-agent/
├─ backend/
│  ├─ main.py                       # FastAPI 엔드포인트와 UI 연결
│  ├─ core/
│  │  ├─ config.py                  # 실행 환경 설정
│  │  ├─ llm.py                     # Ollama·Embedding·VectorStore
│  │  └─ security.py                # API Key 검사
│  ├─ datastore/
│  │  ├─ state.py                   # DataFrame·스키마 메모리 상태
│  │  ├─ scope.py                   # 요청별 선택 문서 범위
│  │  ├─ schema.py                  # LLM용 안전한 DataFrame 스키마
│  │  └─ query.py                   # 이름·기관·식별번호·집계 조회
│  ├─ pandas_engine/
│  │  ├─ aggregation.py             # 결정론적 집계
│  │  ├─ date_filter.py             # 날짜 표현과 기간 필터
│  │  ├─ money.py                   # 금액 파싱과 단위 처리
│  │  ├─ executor.py                # 제한된 Pandas 코드 실행
│  │  └─ formatter.py               # 사용자 답변과 계산 근거
│  ├─ rag/
│  │  ├─ question_detectors.py      # 질문 신호 탐지
│  │  ├─ question_analyzer.py       # 공통 질문 분석 결과
│  │  ├─ guard.py                   # 처리 가능성과 충돌 검사
│  │  ├─ guide.py                   # 질문 재작성 안내
│  │  ├─ router.py                  # 실행 경로 결정
│  │  ├─ pandas_rag.py              # 구조화 데이터 답변 흐름
│  │  ├─ vector.py                  # 관련도 기반 벡터 검색
│  │  └─ prompts.py                 # 근거 제한 LLM 프롬프트
│  ├─ utils/
│  │  ├─ ingest.py                  # 파일 형식별 적재 진입점
│  │  ├─ semantic_schema.py         # 의미·단위·민감도 스키마
│  │  ├─ table_parser.py            # 표 정제와 엔티티 처리
│  │  ├─ parquet_store.py           # Parquet·스키마 저장
│  │  ├─ chroma_store.py            # ChromaDB 저장·삭제
│  │  ├─ manifest.py                # 중복 적재와 상태 관리
│  │  └─ parsers/
│  │     ├─ xlsx_parser.py
│  │     ├─ pdf_parser.py
│  │     ├─ hwp_parser.py
│  │     ├─ image_table_extractor.py
│  │     └─ image_table_ocr_parser.py
│  ├─ static/                       # 문서 관리·채팅 웹 UI
│  ├─ tests/                        # 집계·날짜·스키마·OCR 테스트
│  ├─ data/                         # 원본 문서, Git 제외
│  └─ dataframes/                   # Parquet·schema sidecar, Git 제외
├─ docker-compose.yml
├─ my_workflow.json                 # 선택적 n8n·Slack 연동
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## 설치 및 실행

### 1. 저장소와 환경변수 준비

```powershell
git clone https://github.com/goyojin/finance-doc-agent.git
cd finance-doc-agent
Copy-Item .env.example .env
Copy-Item .env.example backend/.env
```

최소한 다음 값은 실제 환경에 맞게 변경합니다.

```dotenv
POSTGRES_PASSWORD=change_me_secure_password
API_KEY=change_me_api_key
```

`.env`에는 실제 비밀번호와 API Key가 들어가므로 Git에 커밋하지 않습니다.

### 2. 인프라 실행

Docker Desktop이 로컬 Docker context로 실행 중인지 확인한 후 다음 명령을 실행합니다.

```powershell
docker compose up -d
```

| 서비스 | 호스트 포트 | 용도 |
|---|---:|---|
| Ollama | 11434 | LLM·Embedding 서버 |
| PostgreSQL | 5433 | 적재 manifest |
| ChromaDB | 8000 | 벡터 저장소 |
| n8n | 5678 | 선택적 자동화 워크플로 |

모델을 준비합니다.

```powershell
docker exec ollama_server ollama pull qwen2.5:3b
docker exec ollama_server ollama pull bge-m3
```

### 3. Python 환경 준비

```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
```

HWP/HWPX 적재에는 Windows에 한글 프로그램이 설치되어 있어야 합니다. 스캔 PDF OCR에는 Tesseract와 Poppler가 필요합니다. PaddleOCR 모델은 최초 이미지 적재 시 내려받기 때문에 첫 실행은 오래 걸릴 수 있습니다.

### 4. 백엔드 실행

```powershell
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

접속 주소:

```text
웹 UI      http://localhost:8080/ui
Swagger UI http://localhost:8080/docs
상태 확인   http://localhost:8080/health
```

---

## API

`API_KEY`가 설정된 경우 보호된 엔드포인트에 `X-API-Key` 헤더가 필요합니다.

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/health` | API·Ollama·ChromaDB 상태 확인 |
| `GET` | `/ui` | 문서 관리 및 채팅 화면 |
| `POST` | `/chat` | 질문과 선택 문서를 받아 답변 반환 |
| `POST` | `/chat/stream` | 텍스트 스트리밍 답변 |
| `GET` | `/documents` | 적재 문서와 상태 목록 |
| `GET` | `/status?source=...` | 파일별 적재 상태 |
| `POST` | `/ingest/upload` | multipart 파일 업로드 및 비동기 적재 |
| `POST` | `/ingest` | 허용된 서버 경로의 파일 적재 |
| `POST` | `/ingest/all` | `backend/data` 문서 일괄 적재 |
| `DELETE` | `/documents/{source}` | 원본·Parquet·Chroma·manifest 삭제 |
| `GET` | `/summary` | 적재 문서 요약 정보 |

### 문서 업로드

```powershell
curl.exe -X POST "http://localhost:8080/ingest/upload" `
  -H "X-API-Key: change_me_api_key" `
  -F "file=@C:\documents\test2025.png"
```

업로드 API는 먼저 `accepted`를 반환합니다. 이후 `/status`에서 완료 여부를 확인합니다.

### 선택 문서 질문

```powershell
$body = @{
  question = "3~4월 출연금액 합계 알려줘"
  sources  = @("test2025.png")
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8080/chat" `
  -Headers @{ "X-API-Key" = "change_me_api_key" } `
  -ContentType "application/json" `
  -Body $body
```

### 문서 삭제

```powershell
Invoke-RestMethod `
  -Method Delete `
  -Uri "http://localhost:8080/documents/test2025.png" `
  -Headers @{ "X-API-Key" = "change_me_api_key" }
```

---

## 질문 예시

### 명단과 조건 검색

```text
선택한 문서의 전체 명단을 보여줘
58기 기부자 명단을 알려줘
발행번호 2025-061 기록을 찾아줘
김*수와 일치하는 마스킹 이름을 찾아줘
3~4월에 낸 사람 리스트 알려줘
```

### 금액 집계

```text
출연금액 총액은 얼마야?
평균 지급액 알려줘
중앙값과 최빈값 중 하나만 알려줘
가장 많이 낸 사람은 누구야?
개인별 누적 금액이 가장 적은 사람은?
4월 출연금액 상위 5명을 보여줘
```

### 문서 내용 검색

```text
이 장학금의 지급 기준을 설명해줘
문서에 적힌 신청 절차가 뭐야?
지원 대상 조건을 근거와 함께 알려줘
```

---

## 테스트

백엔드 가상환경에서 실행합니다.

```powershell
cd backend
venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

현재 기준 자동화 테스트 **77개가 모두 통과**합니다.

현재 테스트는 다음 영역을 다룹니다.

- 질문 분석, Guard, Router
- 집계 표현과 금액 단위 처리
- 평균·중앙값·최빈값·최대·최소·상하위 N개
- 날짜 단일 월·범위·연도 모호성
- 문서 범위 격리
- 개인·기관·마스킹 이름·기수·발행번호 검색
- 의미 스키마와 민감도
- 표 정제
- 동적 이미지 헤더·열 개수·OCR 보정 이력

OCR 정확도는 문서 해상도와 표 형태에 따라 달라지므로 고정 백분율을 프로젝트 전체 성능으로 주장하지 않습니다. 실제 운영 전에는 사용할 문서 유형별 골드셋으로 셀 정확도, 금액 정확도, 구조 정확도를 각각 측정해야 합니다.

---

## 현재 제한 사항

- 이미지 OCR은 격자선이 분명한 단일 표에서 가장 안정적입니다.
- 공통 스키마는 의미를 보조하는 메타데이터이며 모든 컬럼을 강제로 표준화하지 않습니다.
- 여러 문서에서 동일 인물임을 확정하는 전역 사람 ID는 아직 없습니다.
- 기본 집계는 결정론적 함수로 처리하지만, 지원하지 않는 복잡한 질의는 제한된 LLM Pandas 경로를 사용할 수 있습니다.
- `/summary`의 일부 문서 목적·기간 정보는 파일명 규칙에 의존하므로 범용 문서에서는 `미확정`일 수 있습니다.
- API Key는 단일 키 방식입니다. 역할별 권한과 사용자별 문서 접근 제어는 구현 전입니다.
- 개인정보 조회 감사 로그와 보존 정책은 구현 전입니다.

---

## 다음 개발 계획

우선순위는 기능을 무작정 늘리는 것보다 실제 문서와 질문으로 신뢰성을 측정하는 것입니다.

1. Router·Guard·집계·날짜 질의 골드셋 구축
2. 표 유형별 OCR 정확도와 실패 기준 측정
3. `query.py`, `formatter.py`, `table_parser.py` 책임 분리
4. 역할 기반 권한, 문서별 접근 제어, 감사 로그
5. 테두리 없는 표·복수 표 이미지 지원
6. 검증된 집계 결과를 이용한 시각화
7. 여러 문서의 동일 인물 통합이 실제로 필요한 경우에만 전역 ID 도입

---

## Git 작업 규칙

```text
feat: 새 기능
fix: 오류 수정
refactor: 동작 변화 없는 구조 개선
test: 테스트 추가·수정
docs: README와 문서 수정
```

`.env`, 원본 재정 문서, 생성된 Parquet, ChromaDB 데이터와 개인정보가 포함된 테스트 파일은 Git에 올리지 않습니다.
