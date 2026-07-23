from __future__ import annotations #타입힌트를 더 편하게 사용할 수 있도록 하는 기능
                                   #아직 정의되지않은 클래스 또는 자기 자신을 타입으로 사용할 때 따옴표를 안써도 됨

import glob #파일 탐색
import logging #로그 출력
import os #파일 폴더 제어
import re #문자열 정규식
import shutil  #파일 폴더 제어
import sys
from contextlib import asynccontextmanager #비동기 처리
from datetime import datetime, timezone #시간 계산 
from typing import AsyncIterator, Literal #비동기 처리
from urllib.request import urlopen #웹 요청
from urllib.error import URLError

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
#.env 라는 외부 파일에서 데이터베이스 비밀번호나 API 키 같은 환경 변수를 읽어와 프로그램에 주입
#백엔드 보안 절차

import chromadb #벡터 데이터베이스
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
#웹서버 프레임워크 , HTTPExecption 에러가 났을때 404과 같은에러를 띄움, 백그라운드에서 따로 돌림
# 의존성 주입용 도구 로그인 체크나 API 키 검사를 미리 수행, 
#웹으로 주고받는 데이터의 데이터 타입과 형식을 강제하고 검증하는 라이브러리
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_ollama import OllamaEmbeddings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#현재 실행 중인 폴더 경로를 파이썬 시스템 경로에 추가하여, 같은 폴더 내의 다른 .py 파일들을 
#자유롭게 import할 수 있음 

# 도메인 모듈 (config → security/llm/datastore/rag 순으로 의존)
from utils.ingest import process_file, ensure_manifest_table
from utils.parsers.image_table_ocr_parser import IMAGE_EXTS
from utils.manifest import get_manifest_status, get_all_manifest_entries, delete_manifest
from core.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, EMBED_MODEL,
    CHROMA_HOST, CHROMA_PORT, DATA_FOLDER,
    QUESTION_ENGINE_MODE,
)
from core.security import _verify_api_key, _validate_ingest_path
from core.llm import get_llm_rag
from datastore.state import _df_namespace, _df_sources, _load_dataframes
from datastore.schema import _get_df_schema
from datastore.scope import document_scope, scoped_mapping
from datastore.query import (
    _count_valid_name_rows,
    _extract_total_from_source,
    _extract_recipient_from_dfs,
    _extract_month_from_source,
)
from rag.router import (
    _route,
    pandas_strategy_for_operations,
    route_operations,
)
from rag.guard import check_question, check_question_decision
from rag.guide import build_guide_response
from rag.vector import _answer_vector, _stream_vector
from rag.pandas_rag import _answer_pandas, current_interactive_result
from pandas_engine.interactive import get_interactive_detail
from rag.question_engine import (
    QuestionEngineError,
    compare_shadow_decision,
    decide_question,
)
from rag.deterministic_query_plan import (
    ambiguous_person_lookup_candidates,
    build_schema_grounded_plan,
    has_unmatched_person_amount_reference,
    has_unmatched_person_field_reference,
    is_grounded_person_amount_lookup_question,
    is_grounded_person_payment_existence_question,
)
from rag.question_suggestions import build_person_autocomplete_catalog, build_question_suggestions
from rag.question_decision import QuestionDecision

logger = logging.getLogger("uvicorn.error")


def _route_with_guard(
    question: str,
    guard_result,
    mode: Literal["auto", "natural"] = "auto",
) -> str:
    if mode == "natural":
        return "VECTOR"
    return _route(question, analysis=guard_result.analysis)


def _schedule_shadow_question_engine(
    background_tasks: BackgroundTasks,
    question: str,
    legacy_route: str,
    legacy_operations: list[str] | tuple[str, ...],
) -> bool:
    """Queue an observation-only LLM decision while document scope is active."""

    if QUESTION_ENGINE_MODE != "shadow":
        return False
    schema = _get_df_schema()
    background_tasks.add_task(
        compare_shadow_decision,
        question,
        legacy_route,
        tuple(legacy_operations),
        schema,
    )
    return True


async def _resolve_llm_question(question: str):
    """Return validated LLM operations, engine, and optional PANDAS strategy."""

    normalized = re.sub(r"\s+", "", question)
    # Common IME typo/variant: "두번쨰", "두번째" should mean "두번째".
    normalized = normalized.replace("번쨰", "번째")
    dataframes = scoped_mapping(_df_namespace, _df_sources)
    deterministic_operation = None
    # A plain whole-table list has no semantic ambiguity and should not wait
    # for a local model. File/document inventories use a separate route and do
    # not match this table-record expression.
    if (
        is_grounded_person_payment_existence_question(question, dataframes=dataframes)
        or is_grounded_person_amount_lookup_question(question, dataframes=dataframes)
    ):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="lookup_amount"
        )
        if candidate is not None:
            deterministic_operation = "lookup_amount"
    elif re.fullmatch(
        r"(?:표의?)?(?:전체|모든|전부)(?:데이터|기록|행|명단|목록|리스트)?"
        r"(?:보여줘|보여|알려줘|조회해줘|확인해줘)?[?!.]*",
        normalized,
    ):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="list_records"
        )
        if candidate is not None:
            deterministic_operation = "list_records"
    # A missing-value phrase is a filter condition even when it names a field
    # that can otherwise be returned by a lookup (for example an email).
    elif re.search(r"(?:비어있|안적|미입력|누락|공백|없(?:는|어|어?))", normalized):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="structured_query"
        )
        if candidate is not None:
            deterministic_operation = "structured_query"
    # A payment-time request can naturally contain "돈 냈어".  Resolve its
    # explicit temporal projection before the payment-existence total below.
    elif any(token in normalized for token in ("등록날짜", "지급일", "날짜", "언제", "시기")):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="lookup_field"
        )
        if candidate is not None:
            deterministic_operation = "lookup_field"
    elif re.search(r"(?:돈|금액|회비|결제|납부|후원|기부).{0,8}?(?:냈|내었|냈어|냈나요|했어|했나|했나요)", normalized):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="lookup_amount"
        )
        if candidate is not None:
            deterministic_operation = "lookup_amount"
    elif (
        any(token in normalized for token in ("전화번호", "이메일"))
        and re.search(r"(?:얼마|돈|금액|회비|결제|납부|후원|기부)", normalized)
    ):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="lookup_amount"
        )
        if candidate is not None:
            deterministic_operation = "lookup_amount"
    elif any(token in normalized for token in ("전화번호", "이메일", "전공", "학과")):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="lookup_field"
        )
        if candidate is not None:
            deterministic_operation = "lookup_field"
    elif re.search(r"(?:사람|인원|회원).*?(?:몇명|수)|몇명", normalized):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="count_records"
        )
        if candidate is not None:
            deterministic_operation = "count_records"
    # Explicit scalar extremes must take precedence over the generic money-total
    # shortcut below. "가장 큰 금액" asks for max, not sum.
    elif (
        re.search(
            r"(?:최댓값|최대(?:값|액|금액)?|최고(?:값|액|금액)?|"
            r"(?:가장|제일)(?:큰|높은|많은)(?:값|금액|돈|액)|"
            r"(?:값|금액|돈|액).{0,8}?(?:가장|제일)(?:큰|높은|많은))",
            normalized,
        )
        and not re.search(r"(?:사람|회원|인원|누구)", normalized)
    ):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="max_amount"
        )
        if candidate is not None:
            deterministic_operation = "max_amount"
    elif re.search(
        r"(?:최솟값|최소(?:값|액|금액)?|최저(?:값|액|금액)?|"
        r"(?:가장|제일)(?:작은|낮은)(?:값|금액|돈|액))",
        normalized,
    ):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="min_amount"
        )
        if candidate is not None:
            deterministic_operation = "min_amount"
    # Ordering and ordinal ranking must take precedence over the generic
    # money-total shortcut below.  "금액을 큰 순서대로" is a list request,
    # while "누적 금액이 두 번째로 큰 사람" is a grouped rank request.
    elif re.search(
        r"(?:오름차순|내림차순|순서대로|큰순|작은순|많은순|적은순|"
        r"\d+번째|첫번째|두번째|세번째|네번째|다섯번째|최신|가장이른)",
        normalized,
    ):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="structured_query"
        )
        if candidate is not None:
            deterministic_operation = "structured_query"
    # A grounded person plus a short money noun is a person-scoped total.
    # The QueryPlan builder keeps explicit average/mode/ranking precedence and
    # declines this route unless the subject is grounded in the dataframe.
    elif re.search(r"(?:총합|합계|총액|얼마|금액|돈)", normalized):
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="sum_amount"
        )
        if candidate is not None:
            deterministic_operation = "sum_amount"
    else:
        candidate = build_schema_grounded_plan(
            question, dataframes=dataframes, operation_hint="structured_query"
        )
        if candidate is not None:
            deterministic_operation = "structured_query"

    if deterministic_operation:
        decision = QuestionDecision.model_validate(
            {
                "status": "ready",
                "requests": [{"source_text": question, "operation": deterministic_operation}],
                "reason": "스키마와 질문 원문으로 검증 가능한 표 조회입니다.",
            }
        )
        guard_result = check_question_decision(decision)
        logger.info("[QUESTION_ENGINE] 스키마 기반 분류 | operation=%s", deterministic_operation)
        return guard_result, "PANDAS", pandas_strategy_for_operations(decision.operations)

    decision = await decide_question(
        question,
        schema=_get_df_schema(),
    )
    guard_result = check_question_decision(decision)
    if guard_result.status == "GUIDE":
        return guard_result, "GUIDE", None
    route = route_operations(decision.operations)
    strategy = pandas_strategy_for_operations(decision.operations)
    return guard_result, route, strategy


def _document_list_answer(entries: list[dict]) -> tuple[str, list[str]]:
    """Manifest의 공개 필드만 사용해 적재 문서 목록을 읽기 쉽게 만든다."""
    if not entries:
        return "현재 적재 기록에 등록된 문서가 없습니다.", []

    status_labels = {
        "SUCCESS": "적재 완료",
        "IN_PROGRESS": "처리 중",
        "FAILED": "적재 실패",
    }
    lines = [f"현재 등록된 문서는 총 {len(entries):,}개입니다."]
    sources: list[str] = []
    for entry in entries:
        source = str(entry.get("source") or "").strip()
        if not source:
            continue
        status = str(entry.get("status") or "").upper()
        lines.append(f"- {source} ({status_labels.get(status, status or '상태 미확인')})")
        sources.append(source)
    return "\n".join(lines), sources


# ---------------------------------------------------------------------------
# 파일 탐색 (재귀)
# ---------------------------------------------------------------------------
def _find_files(folder: str) -> list[str]:
    paths = []
    for ext in ("xlsx", "pdf", "hwp", "hwpx", *IMAGE_EXTS): #튜플 순회
        paths.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
        #** -> 모든 하위 폴더
    return [p for p in paths if not os.path.basename(p).startswith(".")]
    #.으로 시작하는 파일 말고 모든 xlsx, pdf, hwp, hwpx 파일 


# ---------------------------------------------------------------------------
# 앱 수명 주기 #앱 시작하기 전 초기화 코드
# ---------------------------------------------------------------------------
@asynccontextmanager #데코레이터 시작과 종료를 표기
async def lifespan(app: FastAPI): #비동기 함수 (돌아가는 동안 다른것도 돌아갈 수 있게 해줌)
    ensure_manifest_table()
    logger.info("manifest 테이블 확인 완료")

    _load_dataframes()
    logger.info("DataFrame 로드 완료 | count=%d", len(_df_namespace))

    try:
        logger.info("LLM 워밍업 중... (model=%s)", OLLAMA_MODEL) #llm
        await get_llm_rag().ainvoke("안녕") #워밍업
        logger.info("LLM 워밍업 완료") 
    except Exception as e:
        logger.warning("LLM 워밍업 실패 | model=%s err=%s", OLLAMA_MODEL, e)
    try:
        logger.info("임베딩 모델 워밍업 중... (model=%s)", EMBED_MODEL) #임베딩
        OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL).embed_query("안녕") #워밍업
        logger.info("임베딩 워밍업 완료")
    except Exception as e:
        logger.warning("임베딩 워밍업 실패 | model=%s err=%s", EMBED_MODEL, e)

    yield

app = FastAPI(title="Local RAG Chatbot API", version="2.0.0", lifespan=lifespan)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel): #베이스모델 -> 제이슨으로 return
    question: str #질문 
    sources: list[str] = Field(default_factory=list) #선택한 원본 문서명
    mode: Literal["auto", "natural"] = "auto" #자동 분기 또는 자연어 의미 검색

class ChatResponse(BaseModel):
    answer: str #답변,
    source: str  #소스,
    sources: list[str] = Field(default_factory=list) #출처
    result: dict | None = None

class ChatSuggestionRequest(BaseModel):
    query: str = ""
    sources: list[str] = Field(default_factory=list)
    limit: int = Field(default=6, ge=1, le=50)
    catalog: bool = False

class IngestRequest(BaseModel):
    file_path: str #파일 업로드 요청

class StatusResponse(BaseModel):
    status: str
    message: str
    filename: str | None = None #상태

# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root_ui():
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def chatbot_ui():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health") #서버 상태 확인
def health():
    result: dict = {
        "status":      "ok",
        "llm_model":   OLLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "dataframes":  len(_df_namespace),
        "question_engine": QUESTION_ENGINE_MODE,
    }
    try:
        urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        result["ollama"] = "ok"
    except URLError:
        result["ollama"] = "unreachable"
        result["status"] = "degraded"
    try:
        chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT).heartbeat()
        result["chromadb"] = "ok"
    except Exception:
        result["chromadb"] = "unreachable"
        result["status"] = "degraded"
    return result


@app.get("/chat/details/{reference}")
def chat_detail(reference: str, offset: int = 0, limit: int = 50, _: None = Depends(_verify_api_key)):
    if offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="offset/limit 범위가 올바르지 않습니다.")
    detail = get_interactive_detail(reference, offset=offset, limit=limit)
    if detail is None:
        raise HTTPException(status_code=404, detail="상세 조회 정보가 없거나 만료되었습니다.")
    return detail


@app.post("/chat/suggestions")
def chat_suggestions(
    req: ChatSuggestionRequest,
    _: None = Depends(_verify_api_key),
):
    if len(req.query) > 200:
        raise HTTPException(status_code=400, detail="query가 너무 깁니다.")
    with document_scope(req.sources):
        dataframes = scoped_mapping(_df_namespace, _df_sources)
        suggestions = build_question_suggestions(
            req.query,
            dataframes=dataframes,
            limit=req.limit if req.catalog else min(req.limit, 3),
        )
        person_catalog = build_person_autocomplete_catalog(dataframes) if req.catalog else {"names": [], "actions": []}
    return {
        "suggestions": suggestions,
        "person_names": person_catalog["names"],
        "person_actions": person_catalog["actions"],
    }


@app.get("/summary") #모든 적재 문서의 요약 정보 반환 
def summary(_: None = Depends(_verify_api_key)):
    """모든 적재 문서의 명세 요약: 문서별 목적·인원·총액 + 전체 합산.
    n8n·Slack 연동 시 명세서 자동 작성에 활용."""
    _AMOUNT_RE = re.compile(r"(\d[\d,]*)만원")

    seen_sources: list[str] = []
    docs: list[dict] = []

    for alias in sorted(_df_namespace.keys()):
        source = _df_sources.get(alias, alias)
        if source in seen_sources:
            continue
        seen_sources.append(source)

        same_src = [a for a in _df_namespace if _df_sources.get(a) == source]
        total_count = sum(_count_valid_name_rows(_df_namespace[a]) for a in same_src)

        amount_str = _extract_total_from_source(alias)
        amount_int = 0
        if amount_str:
            m = _AMOUNT_RE.search(amount_str)
            if m:
                amount_int = int(m.group(1).replace(",", ""))

        # 목적: 파일명에서 번호·금액·괄호 제거
        core = re.sub(r"\s*[-–]\s*\d[\d,]*만원.*$", "", source)
        core = re.sub(r"\s*\.[a-zA-Z]+$", "", core)
        core = re.sub(r"\s*\([^)]*\)\s*", " ", core).strip()
        core = re.sub(r"^\d+\.\s*", "", core).strip()

        recipient = _extract_recipient_from_dfs(same_src)
        month_str = _extract_month_from_source(source)

        docs.append({
            "문서명": source,
            "목적": core,
            "인원": total_count,
            "총액": amount_str or "미확인",
            "총액_만원": amount_int,
            "지급처": recipient,
            "지출월": month_str,
        })

    total_people = sum(d["인원"] for d in docs)
    total_amount = sum(d["총액_만원"] for d in docs)

    return {
        "생성일시": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "전체합산": {
            "총인원": total_people,
            "총지원금액": f"{total_amount:,}만원",
        },
        "문서_목록": [
            {k: v for k, v in d.items() if k != "총액_만원"}
            for d in docs
        ],
        "전체합산_지급처": list(dict.fromkeys(
            d["지급처"] for d in docs if d["지급처"]
        )),
    }


@app.post("/chat", response_model=ChatResponse) #답변 하기
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(_verify_api_key),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question이 비어있습니다.")
    try:
        with document_scope(req.sources) as selected:
            if selected:
                logger.info("[SCOPE] 선택 문서 | sources=%s", list(selected))
            use_llm_engine = (
                QUESTION_ENGINE_MODE == "llm"
                and req.mode == "auto"
            )
            if use_llm_engine:
                scoped_dataframes = scoped_mapping(_df_namespace, _df_sources)
                candidates = ambiguous_person_lookup_candidates(
                    req.question,
                    dataframes=scoped_dataframes,
                )
                if candidates:
                    return ChatResponse(
                        answer=(
                            "동일하거나 유사한 이름의 회원이 여러 명입니다. "
                            "전체 이름을 알려 주세요. 후보: " + ", ".join(candidates)
                        ),
                        source="pandas",
                        sources=[],
                    )
                if has_unmatched_person_field_reference(
                    req.question,
                    dataframes=scoped_dataframes,
                ):
                    return ChatResponse(
                        answer="\uc870\ud68c\ub41c \uc815\ubcf4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.",
                        source="pandas",
                        sources=[],
                    )
                if has_unmatched_person_amount_reference(
                    req.question,
                    dataframes=scoped_dataframes,
                ):
                    return ChatResponse(
                        answer="조회된 금액이 없습니다.",
                        source="pandas",
                        sources=[],
                    )
                try:
                    guard_result, route, pandas_strategy = (
                        await _resolve_llm_question(req.question)
                    )
                except QuestionEngineError as exc:
                    logger.warning(
                        "[QUESTION_ENGINE] 실제 분류 실패 | err=%s",
                        exc,
                    )
                    return ChatResponse(
                        answer=(
                            "질문의 처리 유형을 안전하게 결정하지 못했습니다. "
                            "질문을 조금 더 명확하게 입력해 주세요."
                        ),
                        source="guide",
                        sources=[],
                    )
            else:
                guard_result = check_question(req.question)
                route = _route_with_guard(
                    req.question,
                    guard_result,
                    req.mode,
                )
                pandas_strategy = None

            if guard_result.status == "GUIDE":
                _schedule_shadow_question_engine(
                    background_tasks,
                    req.question,
                    "GUIDE",
                    guard_result.operations,
                )
                logger.info("[GUARD] GUIDE | reason=%s", guard_result.reason_code)
                return ChatResponse(
                    answer=build_guide_response(guard_result),
                    source="guide",
                    sources=[],
                )
            if not use_llm_engine:
                _schedule_shadow_question_engine(
                    background_tasks,
                    req.question,
                    route,
                    guard_result.operations,
                )
            logger.info("[ROUTE] %s | mode=%s question=%s", route, req.mode, req.question[:50])
            if route == "DOCUMENTS":
                answer, sources = _document_list_answer(get_all_manifest_entries())
                actual_route = "documents"
            elif route == "PANDAS":
                answer, sources, actual_route = await _answer_pandas(
                    req.question,
                    allow_vector_fallback=not use_llm_engine,
                    analysis=guard_result.analysis,
                    strategy=pandas_strategy or "AUTO",
                    operation_hint=(
                        guard_result.operations[0]
                        if use_llm_engine and len(guard_result.operations) == 1
                        else None
                    ),
                )
                interactive_result = current_interactive_result()
            else:
                answer, sources, actual_route = await _answer_vector(
                    req.question,
                    allow_pandas_fallback=(
                        req.mode != "natural"
                        and not use_llm_engine
                    ),
                    analysis=guard_result.analysis,
                )
                interactive_result = None
            return ChatResponse(answer=answer, source=actual_route, sources=sources, result=interactive_result if route == "PANDAS" else None)
    except Exception as e:
        logger.exception("[CHAT] 처리 오류 | question=%s", req.question[:50])
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream") #답변 하기 (글씨가 조금씩 써내려져가는 stream 방식)
async def chat_stream(req: ChatRequest, _: None = Depends(_verify_api_key)):
    """스트리밍 응답 — n8n 없이 프론트에서 직접 붙일 때 사용."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question이 비어있습니다.")

    async def generate() -> AsyncIterator[str]:
        try:
            with document_scope(req.sources):
                use_llm_engine = (
                    QUESTION_ENGINE_MODE == "llm"
                    and req.mode == "auto"
                )
                if use_llm_engine:
                    try:
                        guard_result, route, pandas_strategy = (
                            await _resolve_llm_question(req.question)
                        )
                    except QuestionEngineError:
                        yield (
                            "질문의 처리 유형을 안전하게 결정하지 못했습니다. "
                            "질문을 조금 더 명확하게 입력해 주세요."
                        )
                        return
                else:
                    guard_result = check_question(req.question)
                    route = _route_with_guard(
                        req.question,
                        guard_result,
                        req.mode,
                    )
                    pandas_strategy = None

                if guard_result.status == "GUIDE":
                    logger.info("[GUARD] GUIDE(stream) | reason=%s", guard_result.reason_code)
                    yield build_guide_response(guard_result)
                    return
                if route == "DOCUMENTS":
                    answer, _ = _document_list_answer(
                        get_all_manifest_entries()
                    )
                    yield answer
                elif route == "PANDAS":
                    answer, _, _ = await _answer_pandas(
                        req.question,
                        allow_vector_fallback=not use_llm_engine,
                        analysis=guard_result.analysis,
                        strategy=pandas_strategy or "AUTO",
                        operation_hint=(
                            guard_result.operations[0]
                            if use_llm_engine and len(guard_result.operations) == 1
                            else None
                        ),
                    )
                    yield answer
                else:
                    async for chunk in _stream_vector(req.question):
                        yield chunk
        except Exception as e:
            logger.exception("Stream 처리 오류")
            yield f"\n[오류] {e}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


def _process_and_reload(file_path: str): #파일 처리 후 DataFrame 저장소를 다시 로드
    """인제스트 후 DataFrame 저장소를 갱신한다."""
    process_file(file_path)
    _load_dataframes()


@app.post("/ingest", response_model=StatusResponse) #서버에 존재하는 파일 경로 받아 색인
def ingest(req: IngestRequest, background_tasks: BackgroundTasks, _: None = Depends(_verify_api_key)):
    safe_path = _validate_ingest_path(req.file_path)
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail=f"파일 없음: {safe_path}")
    background_tasks.add_task(_process_and_reload, safe_path) #백그라운드 작업으로 넘기기
    return StatusResponse(status="accepted", message=f"'{safe_path}' 처리를 시작했습니다.")


_ALLOWED_INGEST_EXTS = {"xlsx", "pdf", "hwp", "hwpx", *IMAGE_EXTS} #업로드 가능한 파일 확장자 목록


@app.post("/ingest/upload", response_model=StatusResponse) #파일 바이너리 전체 받기
def ingest_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    filename_override: str | None = None,
    _: None = Depends(_verify_api_key),
):
    """파일 바이너리를 직접 업로드받아 data 폴더에 저장 후 색인한다 (Slack 첨부 등).
    filename_override 쿼리 파라미터로 저장 파일명을 지정할 수 있다."""
    filename = os.path.basename(filename_override or file.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_INGEST_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 형식: .{ext} (허용: {', '.join(sorted(_ALLOWED_INGEST_EXTS))})",
        )
    os.makedirs(DATA_FOLDER, exist_ok=True)
    dest = _validate_ingest_path(os.path.join(DATA_FOLDER, filename))
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()
    background_tasks.add_task(_process_and_reload, dest)
    return StatusResponse(status="accepted", message=f"'{filename}' 업로드 완료, 색인을 시작했습니다.", filename=filename)


@app.get("/documents") #색인된 문서 목록 반환함.
def documents(_: None = Depends(_verify_api_key)):
    """색인된 문서 전체 목록과 상태를 반환한다. n8n·Slack 문서목록 명령용."""
    entries = get_all_manifest_entries()
    return {"count": len(entries), "files": entries}


@app.get("/status") #특정 파일의 색인상태를 조회한다.
def ingest_status(source: str, _: None = Depends(_verify_api_key)):
    """파일명(source)으로 색인 상태를 조회한다. 업로드 후 n8n 폴링용.
    반환 status: IN_PROGRESS | SUCCESS | FAILED, 기록 없으면 404."""
    st = get_manifest_status(os.path.basename(source))
    if st is None:
        raise HTTPException(status_code=404, detail=f"'{source}' 색인 기록이 없습니다.")
    return st


@app.delete("/documents/{source}") #색인된 문서 삭제
def delete_document(source: str, _: None = Depends(_verify_api_key)):
    """색인된 문서를 완전히 삭제한다 — ChromaDB·Parquet·manifest·data 파일 모두 제거."""
    from utils.chroma_store import delete_from_chroma
    from utils.parquet_store import drop_dataframe_by_source

    source = os.path.basename(source)

    chroma_deleted = delete_from_chroma(source)
    drop_dataframe_by_source(source)
    manifest_deleted = delete_manifest(source)

    if not manifest_deleted and not os.path.exists(os.path.join(DATA_FOLDER, source)):
        raise HTTPException(status_code=404, detail=f"'{source}' 문서를 찾을 수 없습니다.")

    # Parquet·manifest 삭제 후 즉시 메모리 갱신 (파일 락 여부와 무관)
    _load_dataframes()

    # data 파일 삭제 — Windows 파일 락 시 조용히 건너뜀
    data_path = os.path.join(DATA_FOLDER, source)
    file_existed = os.path.exists(data_path)
    if file_existed:
        try:
            os.remove(data_path)
        except PermissionError:
            logger.warning("data 파일 삭제 실패 (파일 락) — 다음 재시작 시 제거 필요: %s", data_path)

    return {
        "source": source,
        "chroma_deleted": chroma_deleted,
        "file_deleted": file_existed,
        "manifest_deleted": manifest_deleted,
    }


@app.post("/ingest/all", response_model=StatusResponse) #data_folder 안의 모든 문서를 한번에 색인하는 api
def ingest_all(background_tasks: BackgroundTasks, _: None = Depends(_verify_api_key)):
    if not os.path.exists(DATA_FOLDER):
        raise HTTPException(status_code=404, detail="data 폴더를 찾을 수 없습니다.")
    files = _find_files(DATA_FOLDER)
    if not files:
        return StatusResponse(status="ok", message="처리할 파일이 없습니다.")

    def _run():
        for fp in files:
            process_file(fp)
        _load_dataframes()

    background_tasks.add_task(_run)
    return StatusResponse(status="accepted", message=f"{len(files)}개 파일 처리를 시작했습니다.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

#fastAPI 파일을 실행
