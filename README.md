# Finance Document Agent

장학금·후원금·지원금·예산 명세처럼 **금액과 표가 중심인 문서**를 적재하고, 자연어로 검색·계산·조회할 수 있는 로컬 문서 질의 에이전트

Excel, PDF, HWP/HWPX뿐 아니라 세로로 긴 표 이미지까지 직접 처리하는 구조. 표 데이터는 Parquet에 구조화하고 문서 설명과 행 텍스트는 ChromaDB에 임베딩하는 이중 저장 방식. 검증된 직접 조회와 안전한 QueryPlan JSON 실행, 근거 기반 VECTOR 검색을 결합한 질의 구조

핵심 목표는 단순 문서 요약을 넘어 다음 재정 문서 업무를 안전하게 처리하는 업무형 에이전트 구축

- 장학재단 지급·후원 문서 조회
- 지자체 예산·지원금 명세 내부 검색
- 수혜자·기부자·기관 명단 관리
- 금액 합계, 평균, 최댓값, 기간별 집계
- 원본 문서와 계산 근거를 확인할 수 있는 답변

> 현재 상태는 로컬 개발·검증 단계. 역할 기반 접근 제어와 감사 로그는 향후 구현 예정. 민감한 실데이터의 운영 배포 전 별도 보안 검토 필요

---

## 출발점과 고도화 방향

기존 로컬 LLM 기반 하이브리드 RAG 프로젝트를 출발점으로 삼아, 재정 문서의 표 조회와 계산 정확도에 초점을 맞춰 고도화한 프로젝트

원본 프로젝트의 기반 요소:

- Ollama 기반 로컬 LLM과 Embedding
- Parquet 구조화 데이터와 ChromaDB 벡터 데이터의 이중 저장
- Excel·PDF·HWP/HWPX 파서
- PANDAS 또는 VECTOR의 이중 라우팅
- FastAPI API와 n8n·Slack 자동화 흐름

`project1`에서 직접 고도화한 요소:

| 원본 구조 | 확인된 한계 | 현재 고도화 구조 |
|---|---|---|
| Excel·PDF·HWP/HWPX 중심 적재 | 이미지 파일 자체의 표 적재 불가 | OpenCV와 PaddleOCR 기반 셀 단위 이미지 표 파서 추가 |
| 특정 컬럼과 표 형태 중심 정제 | 새로운 컬럼·열 개수에 대한 확장성 부족 | 실제 헤더와 동적 열 개수를 사용하는 범용 이미지 표 처리 |
| DataFrame형 입력의 중복된 표 적재 과정 | 새 표 어댑터마다 정제·저장 코드가 반복될 가능성 | Excel과 후속 표 어댑터용 Table Ingest Pipeline 분리 |
| 빈칸을 이전 행 값으로 자동 채움 | 일반 결측값을 병합 셀로 오인해 원본 값 변조 | XLSX 병합 범위·PDF 셀 좌표·HWP rowspan·이미지 경계선에 근거한 병합 복원 |
| 원본 컬럼명만 사용하는 DataFrame | 같은 의미의 서로 다른 컬럼을 연결하기 어려운 구조 | 원본 보존형 의미 스키마와 `.schema.json` 사이드카 추가 |
| Router 내부의 분산된 키워드 판정 | Guard·Router·집계 판단 간 불일치 가능성 | Question Detectors와 Question Analyzer를 통한 단일 분석 결과 공유 |
| PANDAS와 VECTOR의 양자 분기 | 모호한 질문과 문서 목록 요청의 잘못된 분기 가능성 | DOCUMENTS·PANDAS·VECTOR·GUIDE 경로 분리 |
| 기본 집계도 LLM Pandas 코드에 의존 | 문자열 금액 합산·최대·최소 계산 오류 가능성 | 금액 파서와 결정론적 집계 엔진 추가 |
| 미지원 표 질문을 Python 코드로 생성 | 잘못된 컬럼·연산·자료형 코드의 실행 위험 | 제한된 QueryPlan JSON 생성·검증·실행 구조로 교체 |
| 날짜 표현의 별도 처리 부족 | 지급월 단독 컬럼과 분리된 년·월 검색 실패 | 날짜·연월·연도·월·일 의미 분리와 월·기간 필터 추가 |
| 전체 적재 문서에 대한 묵시적 조회 | 여러 문서의 동명이인과 데이터 혼합 가능성 | `sources` 기반 문서 범위 격리 |
| 결과값 중심 응답 | 어떤 문서와 행으로 계산했는지 확인 곤란 | 문서·컬럼·연산·사용 행·제외 행 근거 표시 |
| Swagger UI 중심 테스트 | 반복적인 업로드와 질문의 불편 | 문서 업로드·목록·선택·삭제·채팅 웹 UI 추가 |

고도화의 중심 원칙:

- 특정 연도나 테스트 문서명 대신 데이터 구조와 의미를 이용한 처리
- 계산 가능한 질문의 LLM 의존 최소화
- 원본 데이터 보존과 추론 메타데이터 분리
- 문서 범위와 답변 근거를 확인할 수 있는 조회
- 지원하지 못하는 질문에 대한 임의 답변 대신 재질문 안내

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

표 형식의 공통 처리 원칙:

```text
파일별 파서에서 원시 표 추출
→ 실제 병합 구조 복원
→ 공통 헤더 탐지와 데이터 행 정제
→ 원본 보존형 의미 스키마 생성
→ 형식별 저장 어댑터에서 Parquet + schema sidecar 저장
→ 행 단위 검색 청크 생성
→ ChromaDB 저장
```

병합 셀은 빈칸 형태로 추측하지 않고 파일에서 확인한 물리적 근거만 사용. 일반 공란과 결측값을 원본 그대로 유지하는 정책

### 2. 셀 단위 이미지 표 OCR

세로로 길고 글자가 작은 표를 단순 분할할 때 발생하는 행·열 및 병합 셀 복원 실패를 줄이기 위한 다음 처리 구조

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

특정 연도·컬럼명·고정 5열 구조를 코드에 넣지 않고, 감지된 열 개수와 실제 헤더를 사용하는 동적 적재 방식

현재 최적 처리 대상은 **격자선이 분명한 단일 표 이미지**. 테두리 없는 표, 한 이미지의 여러 표, 심한 기울기·왜곡은 추가 개선 대상

### 3. 원본을 보존하는 의미 스키마

모든 컬럼을 고정 표준에 강제로 맞추는 대신 원본 컬럼과 별도 의미 스키마를 함께 보존하는 방식

예를 들어 `출연금액`, `후원액`, `지급액`의 원본 이름을 보존하고, 신뢰도가 충분한 경우에만 금액 의미와 KRW 단위를 메타데이터에 기록. 판단하기 어려운 컬럼은 `unknown`으로 유지해 잘못된 자동 매핑 방지

스키마 구성 정보:

- 문서·표·행 식별자
- 원본 컬럼명과 추론된 의미
- 데이터 타입과 단위
- 민감정보·개인정보 여부
- 매핑 신뢰도
- 컬럼 구조 지문과 스키마 버전

날짜 계열 공통 의미:

| 의미 | 예시 | 스키마 역할 |
|---|---|---|
| 완전한 날짜 | `출연일자`, `결제 등록 날짜` | `date` |
| 연월 | `2025-03`, `졸업연월` | `year_month` |
| 연도 | `년`, `지급연도` | `year` |
| 월 | `월`, `지급월` | `month` |
| 일 | `일` | `day` |

DataFrame은 Parquet, 스키마는 `.schema.json` 사이드카로 저장. 현재 스키마 버전 `2.3`

### 4. 하나의 질문 분석 결과를 공유하는 라우팅

Guard와 Router의 중복 판정 대신 하나의 분석 결과를 공유하는 구조. 기존 규칙 기반 분석과 LLM 질문 엔진을 `legacy`, `shadow`, `llm` 모드로 비교·전환할 수 있는 구성

```text
사용자 질문
→ Question Engine 또는 Question Analyzer: 작업 유형과 대상 분석
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

LLM의 즉석 Pandas 코드가 `1,000,000` 같은 문자열 금액을 연결하거나 사전순으로 비교하는 오류를 막기 위한 전용 집계 엔진

- 인원·행 개수
- 합계
- 평균
- 중앙값
- 최빈값
- 1인당 금액
- 최댓값·최솟값
- 상위 N개·하위 N개
- 개인별 누적 금액 최대·최소

쉼표, 원, 천원, 만원, 음수와 0을 지원하고 손상되거나 모호한 값은 계산에서 제외하는 금액 파서

### 6. 검증 가능한 QueryPlan

직접 조회기로 처리하지 못하는 일반 표 질문을 Python 코드가 아닌 제한된 JSON 계획으로 변환하는 구조

```text
질문 + 선택 문서의 안전한 스키마
→ LLM QueryPlan JSON 생성
→ JSON 규격과 operation 계약 검사
→ 질문 원문의 숫자·단위·AND/OR 조건 대조
→ 선택 문서와 실제 컬럼·자료형 검증
→ 허용된 필터·정렬·집계 함수만 실행
→ 결과와 계산 근거 출력
```

지원 operation:

```text
list, count, sum, mean, median, mode, min, max
```

검증 대상은 존재하지 않는 DataFrame·컬럼, 선택 문서 범위 이탈, 문자열/숫자/금액/날짜 자료형 불일치, 질문에 없는 필터값, 잘못된 비교 연산자, 임의의 필터 논리와 정렬 조건. 검증 실패 시 코드 실행 없이 안내 또는 안전한 검색 경로 선택

### 7. 날짜 조건과 집계의 결합

표의 날짜 컬럼을 찾아 처리하는 질문 예시:

```text
3월에 낸 사람 리스트 알려줘
3~4월 출연금액 합계 알려줘
2025년 4월에 가장 많이 낸 사람은?
3월에 가장 적게 지급된 금액은?
지급월이 12월인 사람 알려줘
2024년 12월부터 2025년 2월까지 지급된 명단 알려줘
```

하이픈·점·슬래시·한글 날짜·`YYYYMMDD`·Excel 날짜 일련번호 지원. 완전한 날짜가 없어도 월 컬럼만으로 검색 가능하며, 연도와 월이 별도 컬럼이면 조회 시점에 조합하는 방식. 월 정보만 있는 문서에 특정 연도 조건이 들어오거나 날짜 기준이 여러 개인 경우 임의 선택 대신 사용자 확인 요청

### 8. 개인·기관·마스킹 이름 검색

표의 이름 값을 개인으로만 가정하지 않는 엔티티 처리

- 개인·기관·단체·학과 구분
- 마스킹 이름 정규화 및 패턴 검색
- 기수·발행번호·식별번호 검색
- 같은 이름을 기수·발행번호·행 문맥으로 구분
- 여러 문서에서 같은 이름이 발견되면 문서 선택 요청

현재 사람 식별자는 문서 안의 후보 구분용 값. 여러 문서를 관통하는 확정 인물 ID는 미구현 상태

### 9. 문서 범위 격리

`/chat` 요청의 `sources`에 지정한 동일 문서 범위를 PANDAS와 VECTOR에 함께 적용하는 방식

```json
{
  "question": "가장 많이 낸 사람 누구야?",
  "sources": ["test2025.png"]
}
```

선택하지 않은 문서의 DataFrame은 QueryPlan에서도 접근 불가. 문서 미선택 상태에서 여러 문서의 동명이인이 발견되면 임의 선택 대신 문서 선택 안내

### 10. 답변 근거와 계산 내역

표 계산 결과와 함께 제공하는 근거:

- 사용한 원본 문서
- 계산 대상 컬럼
- 수행한 연산
- 조건에 일치한 행 수
- 실제 계산에 사용한 유효 행 수
- 제외된 행 수
- 적용한 날짜 컬럼과 기간

```text
총 출연금액: 977,070,000원

계산 근거
- 문서: test2025.png
- 대상 컬럼: 출연금액
- 계산: 합계
- 유효 행: 158행
```

검색 문서에 직접 근거가 없을 때 일반 지식이나 파일명 추측으로 내용을 채우지 않도록 제한한 VECTOR 답변

### 11. 별도 웹 UI와 Swagger UI

FastAPI Swagger UI와 별도로 제공하는 문서 업무용 웹 화면

- 웹 UI: `http://localhost:8080/ui`
- Swagger UI: `http://localhost:8080/docs`

웹 UI 지원 작업:

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
파일별 실제 병합 셀 복원
          ↓
공통 표 정제 + 의미 스키마 2.3 생성
  ├─ Parquet + schema sidecar
  ├─ ChromaDB text chunks + metadata
  └─ PostgreSQL ingestion manifest

질문 입력 + 선택 문서
          ↓
Question Engine 또는 Question Detectors → Question Analyzer
          ↓
Guard / Guide
          ↓
Router
  ├─ DOCUMENTS → 문서 목록
  ├─ PANDAS
  │    ├─ DIRECT    → 검증된 집계·날짜·이름 검색
  │    └─ QUERY_PLAN → JSON 계획 검증·제한 실행
  └─ VECTOR    → 관련도 검색 + 근거 기반 LLM 답변
          ↓
답변 + 출처 + 계산 근거
```

---

## 사용 기술 스택

| 영역 | 기술 | 프로젝트 내 역할 |
|---|---|---|
| Language | Python | 파서·질문 분석·조회 엔진·API 구현 |
| API/UI | FastAPI, Pydantic, HTML, CSS, JavaScript | REST API, Swagger UI, 문서 관리·채팅 UI |
| 표 처리 | Pandas, PyArrow, OpenPyXL | 표 정제, 집계, Parquet 저장, Excel 처리 |
| 이미지 OCR | OpenCV, PaddleOCR, PaddlePaddle | 표 격자 탐지, 셀 분리, 한국어·숫자 인식 |
| PDF/HWP | pdfplumber, pdf2image, Tesseract, pyhwpx, BeautifulSoup | PDF 표·본문 추출, 스캔 OCR, HWP HTML 변환 |
| LLM | Ollama, Qwen2.5, LangChain | 질문 유형 분류, QueryPlan JSON 생성, 문서 근거 답변 |
| Embedding | BGE-M3 | 문서·행 텍스트 임베딩 |
| 저장소 | Parquet, ChromaDB, PostgreSQL | 구조화 표, 벡터 청크, 적재 manifest 저장 |
| 자동화 | Docker Compose, n8n, Slack 워크플로 | 로컬 인프라와 선택적 업무 자동화 |

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
│  │  ├─ query_plan.py               # 제한된 JSON 조회 계약
│  │  ├─ query_grounding.py          # 질문 숫자·단위·조건 근거 대조
│  │  ├─ plan_validator.py           # 문서·컬럼·자료형·연산 검증
│  │  ├─ query_executor.py           # 검증된 QueryPlan 결정론적 실행
│  │  └─ formatter.py               # 사용자 답변과 계산 근거
│  ├─ rag/
│  │  ├─ question_detectors.py      # 질문 신호 탐지
│  │  ├─ question_analyzer.py       # 공통 질문 분석 결과
│  │  ├─ question_engine.py         # LLM 기반 작업 유형 분류
│  │  ├─ question_decision.py       # 분류 결과 계약
│  │  ├─ guard.py                   # 처리 가능성과 충돌 검사
│  │  ├─ guide.py                   # 질문 재작성 안내
│  │  ├─ router.py                  # 실행 경로 결정
│  │  ├─ pandas_rag.py              # 구조화 데이터 답변 흐름
│  │  ├─ query_planner.py           # QueryPlan 생성·형식 복구
│  │  ├─ vector.py                  # 관련도 기반 벡터 검색
│  │  └─ prompts.py                 # 근거 제한 LLM 프롬프트
│  ├─ utils/
│  │  ├─ ingest.py                  # 파일 형식별 적재 진입점
│  │  ├─ semantic_schema.py         # 의미·단위·민감도 스키마
│  │  ├─ table_ingest_pipeline.py    # DataFrame형 표 어댑터 공통 적재 과정
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

실제 환경에 맞게 변경할 최소 설정:

```dotenv
POSTGRES_PASSWORD=change_me_secure_password
API_KEY=change_me_api_key
QUESTION_ENGINE_MODE=legacy
```

`QUESTION_ENGINE_MODE` 선택값:

- `legacy`: 기존 Question Detectors·Analyzer 결과 사용
- `shadow`: 기존 결과로 실행하면서 LLM 분류 결과를 로그로 비교
- `llm`: LLM 질문 엔진의 구조화된 operation 결과를 실제 라우팅에 사용

실제 비밀번호와 API Key가 포함된 `.env`는 Git 커밋 제외 대상

### 2. 인프라 실행

Docker Desktop의 로컬 Docker context 실행 확인 후 다음 명령 사용

```powershell
docker compose up -d
```

| 서비스 | 호스트 포트 | 용도 |
|---|---:|---|
| Ollama | 11434 | LLM·Embedding 서버 |
| PostgreSQL | 5433 | 적재 manifest |
| ChromaDB | 8000 | 벡터 저장소 |
| n8n | 5678 | 선택적 자동화 워크플로 |

Ollama 모델 준비:

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

HWP/HWPX 적재 조건은 Windows 한글 프로그램 설치. 스캔 PDF OCR 조건은 Tesseract와 Poppler 설치. PaddleOCR 모델 다운로드가 발생하는 최초 이미지 적재는 추가 시간 소요 가능

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

`API_KEY` 설정 시 보호된 엔드포인트에 필요한 `X-API-Key` 헤더

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

업로드 API의 최초 응답은 `accepted`. 이후 `/status`를 통한 완료 여부 확인

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

백엔드 가상환경 기준 실행 명령:

```powershell
cd backend
venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

현재 기준 자동화 테스트 **205개 전체 통과**

현재 테스트 범위:

- 질문 분석, Guard, Router
- LLM 질문 엔진의 JSON 계약·형식 복구·shadow 비교
- QueryPlan 생성·질문 근거 대조·문서 범위·컬럼·자료형 검증
- QueryPlan 필터·AND/OR·정렬·상하위 N개·집계 실행
- 집계 표현과 금액 단위 처리
- 평균·중앙값·최빈값·최대·최소·상하위 N개
- 날짜·연월·연도·월·일 의미 분류
- 월 단독·분리된 년/월·월 범위·교차 연도·날짜 기준 모호성
- 문서 범위 격리
- 개인·기관·마스킹 이름·기수·발행번호 검색
- 의미 스키마와 민감도
- 표 정제와 실제 병합 셀/일반 공란 구분
- XLSX·PDF 병합 셀 근거 기반 복원과 DataFrame형 공통 적재 파이프라인
- 동적 이미지 헤더·열 개수·OCR 보정 이력

문서 해상도와 표 형태에 따라 달라지는 OCR 정확도. 고정 백분율을 프로젝트 전체 성능으로 사용하지 않고, 실제 운영 전 문서 유형별 골드셋으로 셀·금액·구조 정확도를 각각 측정하는 방침

---

## 현재 제한 사항

- 격자선이 분명한 단일 표에서 가장 안정적인 이미지 OCR
- 모든 컬럼의 강제 표준화가 아닌 의미 보조 메타데이터 방식의 공통 스키마
- 여러 문서에서 동일 인물임을 확정하는 전역 사람 ID 미구현
- 기본 집계는 결정론적 함수, 일반 표 질의는 검증된 QueryPlan 실행 구조
- 월만 존재하는 문서는 연도를 구분할 수 없으며 날짜 기준이 여러 개면 사용자 지정 필요
- QueryPlan 생성과 질문 엔진의 정확도는 사용 LLM과 문서 스키마 품질의 영향 존재
- 파일명 규칙에 일부 의존해 범용 문서에서 `미확정`이 될 수 있는 `/summary`의 목적·기간 정보
- 단일 API Key 방식이며 역할별 권한과 사용자별 문서 접근 제어는 구현 전
- 개인정보 조회 감사 로그와 보존 정책은 구현 전

---

## 다음 개발 계획

새 기능의 무조건적인 확대보다 실제 문서와 질문을 통한 신뢰성 측정을 우선하는 계획

1. 실제 사용 문서 유형별 Excel·PDF·이미지 표 확보
2. 질문 유형별 골드셋과 기대 QueryPlan 구축
3. 문서별·질문별 정답률, 라우팅 정확도, 계산 정확도 측정
4. 실패 사례를 기준으로 Question Engine·QueryPlan·직접 조회 보완
5. 반복 검증을 위한 웹 UI 결과 기록과 테스트 편의 개선
6. 표 유형별 OCR 정확도와 실패 기준 측정
7. 테두리 없는 표·복수 표 이미지 지원

---

## Git 작업 규칙

```text
feat: 새 기능
fix: 오류 수정
refactor: 동작 변화 없는 구조 개선
test: 테스트 추가·수정
docs: README와 문서 수정
```

`.env`, 원본 재정 문서, 생성된 Parquet, ChromaDB 데이터, 개인정보 포함 테스트 파일은 Git 업로드 제외 대상
