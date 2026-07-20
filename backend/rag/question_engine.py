from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from core.llm import get_llm_code
from rag.prompts import (
    _QUESTION_ENGINE_REPAIR_TEMPLATE,
    _QUESTION_ENGINE_TEMPLATE,
)
from rag.question_decision import QuestionDecision
from rag.router import (
    pandas_strategy_for_operations,
    route_operations,
)


logger = logging.getLogger("uvicorn.error")

_EXPLICIT_DOCUMENT_INVENTORY = re.compile(
    r"(?:파일\s*(?:목록|리스트)|"
    r"(?:적재|업로드|등록|저장)(?:된|한)?\s*(?:문서|파일|목록)|"
    r"(?:문서|파일)\s*(?:목록|리스트)|"
    r"(?:무슨|어떤)\s*(?:문서|파일))",
)


class QuestionEngineError(RuntimeError):
    def __init__(self, message: str, responses: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.responses = responses


@dataclass(frozen=True)
class ShadowComparison:
    question: str
    legacy_operations: tuple[str, ...]
    llm_operations: tuple[str, ...]
    legacy_route: str
    llm_route: str
    status: str
    llm_strategy: str | None
    engine_matched: bool
    operation_matched: bool
    reason: str


def compact_question_schema(schema: str, max_chars: int = 5000) -> str:
    """Keep only routing-relevant schema lines, excluding row samples."""

    kept: list[str] = []
    for line in str(schema or "").splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("파일:")
            or stripped.startswith("데이터프레임:")
            or stripped.startswith("컬럼(")
            or stripped.startswith("컬럼:")
            or stripped.startswith("검증된 컬럼 의미:")
        ):
            kept.append(line)
    compact = "\n".join(kept).strip()
    return (compact or "(조회 가능한 표 없음)")[:max_chars]


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", None)
    if content is not None:
        return str(content).strip()
    return str(response).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("응답에서 완전한 JSON 객체를 찾을 수 없습니다.")


def parse_question_decision(
    response: Any,
    *,
    fallback_retrieval_query: str | None = None,
) -> QuestionDecision:
    text = _response_text(response)
    if not text:
        raise ValueError("LLM이 빈 응답을 반환했습니다.")
    payload = _extract_json_object(text)
    candidates = payload.get("candidates")
    if isinstance(candidates, str):
        payload["candidates"] = [candidates]
    operations = payload.get("operations")
    if isinstance(operations, str):
        payload["operations"] = [operations]
    elif operations is None and isinstance(payload.get("operation"), str):
        payload["operations"] = [payload.pop("operation")]
    requests = payload.get("requests")
    if isinstance(requests, dict):
        payload["requests"] = [requests]
        requests = payload["requests"]
    if requests and not payload.get("operations"):
        payload["operations"] = list(dict.fromkeys(
            request.get("operation")
            for request in requests
            if isinstance(request, dict) and request.get("operation")
        ))
    # Previous route/intent fields and volunteered expressions are never
    # executed. Removing them makes the operation contract migration tolerant
    # without guessing a missing operation from a broad engine label.
    for obsolete in ("route", "intent", "query", "filters"):
        payload.pop(obsolete, None)
    if payload.get("status") == "ready":
        operations = payload.get("operations") or []
        has_document_operation = any(
            str(operation).startswith("document_")
            for operation in operations
        )
        if has_document_operation and not payload.get("retrieval_query"):
            if fallback_retrieval_query:
                # The original question is always a safe retrieval query and
                # avoids trusting an invented rewrite from the classifier.
                payload["retrieval_query"] = fallback_retrieval_query
        elif not has_document_operation:
            payload.pop("retrieval_query", None)
    return QuestionDecision.model_validate(payload)


def _validate_request_evidence(
    decision: QuestionDecision,
    question: str,
) -> QuestionDecision:
    for request in decision.requests:
        if request.source_text not in question:
            raise ValueError(
                "독립 요청의 source_text가 실제 질문에 존재하지 않습니다: "
                f"{request.source_text}"
            )
    return decision


def _align_document_inventory_operation(
    decision: QuestionDecision,
    question: str,
) -> QuestionDecision:
    """Reserve list_documents for explicit file/document inventory requests."""

    if decision.status != "ready" or not decision.requests:
        return decision

    explicit_inventory = bool(_EXPLICIT_DOCUMENT_INVENTORY.search(question))
    normalized_requests: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    changed = False

    for request in decision.requests:
        operation = request.operation
        if operation == "list_documents" and not explicit_inventory:
            operation = "list_records"
            changed = True
        key = (request.source_text, operation)
        if key in seen:
            changed = True
            continue
        seen.add(key)
        normalized_requests.append(
            {
                "source_text": request.source_text,
                "operation": operation,
            }
        )

    if not changed:
        return decision

    payload = decision.model_dump(mode="python")
    payload["requests"] = normalized_requests
    payload.pop("operations", None)
    logger.warning(
        "[QUESTION_ENGINE] 문서 목록 operation 계약 보정 | question=%s",
        question[:80],
    )
    return QuestionDecision.model_validate(payload)


def _validation_message(error: Exception) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for issue in error.errors(include_url=False):
            location = ".".join(str(item) for item in issue.get("loc", ()))
            message = str(issue.get("msg") or "규격 오류")
            parts.append(f"{location}: {message}" if location else message)
        return "\n".join(parts)[:2000]
    return str(error)[:2000]


async def decide_question(
    question: str,
    *,
    schema: str,
    llm: Any | None = None,
) -> QuestionDecision:
    """Classify one question without executing either retrieval engine."""

    clean_question = str(question or "").strip()
    if not clean_question:
        raise QuestionEngineError("빈 질문은 분류할 수 없습니다.")

    model = llm or get_llm_code()
    prompt = _QUESTION_ENGINE_TEMPLATE.format(
        schema=compact_question_schema(schema),
        question=clean_question,
    )
    responses: list[str] = []

    raw = await model.ainvoke(prompt)
    responses.append(_response_text(raw))
    try:
        return _align_document_inventory_operation(
            _validate_request_evidence(
                parse_question_decision(
                    raw,
                    fallback_retrieval_query=clean_question,
                ),
                clean_question,
            ),
            clean_question,
        )
    except (ValueError, TypeError, ValidationError) as first_error:
        error_message = _validation_message(first_error)
        logger.warning(
            "[QUESTION_ENGINE] 첫 응답 규격 오류, 형식 수정 재시도 | err=%s",
            error_message[:500],
        )

    repair_prompt = _QUESTION_ENGINE_REPAIR_TEMPLATE.format(
        question=clean_question,
        error=error_message,
        response=responses[0][:2500],
    )
    repaired = await model.ainvoke(repair_prompt)
    responses.append(_response_text(repaired))
    try:
        return _align_document_inventory_operation(
            _validate_request_evidence(
                parse_question_decision(
                    repaired,
                    fallback_retrieval_query=clean_question,
                ),
                clean_question,
            ),
            clean_question,
        )
    except (ValueError, TypeError, ValidationError) as second_error:
        raise QuestionEngineError(
            "LLM 응답을 안전한 질문 결정으로 변환하지 못했습니다.",
            tuple(responses),
        ) from second_error


async def compare_shadow_decision(
    question: str,
    legacy_route: str,
    legacy_operations: list[str] | tuple[str, ...],
    schema: str,
    *,
    llm: Any | None = None,
) -> ShadowComparison | None:
    """Run the LLM decision for observation only and never affect an answer."""

    try:
        decision = await decide_question(
            question,
            schema=schema,
            llm=llm,
        )
    except Exception as exc:
        logger.warning(
            "[QUESTION_ENGINE:SHADOW] 분류 실패, 기존 응답 유지 | "
            "legacy=%s err=%s question=%s",
            legacy_route,
            exc,
            question[:80],
        )
        return None

    llm_operations = tuple(decision.operations)
    llm_route = (
        route_operations(llm_operations)
        if decision.status == "ready"
        else "GUIDE"
    )
    llm_strategy = pandas_strategy_for_operations(llm_operations)
    normalized_legacy_operations = tuple(legacy_operations)
    engine_matched = legacy_route.upper() == llm_route.upper()
    operation_matched = (
        set(normalized_legacy_operations) == set(llm_operations)
    )
    comparison = ShadowComparison(
        question=question,
        legacy_operations=normalized_legacy_operations,
        llm_operations=llm_operations,
        legacy_route=legacy_route.upper(),
        llm_route=llm_route.upper(),
        status=decision.status,
        llm_strategy=llm_strategy,
        engine_matched=engine_matched,
        operation_matched=operation_matched,
        reason=decision.reason,
    )
    logger.info(
        "[QUESTION_ENGINE:SHADOW] legacy_ops=%s llm_ops=%s "
        "legacy_engine=%s llm_engine=%s strategy=%s "
        "engine_match=%s operation_match=%s status=%s "
        "reason=%s question=%s",
        list(comparison.legacy_operations),
        list(comparison.llm_operations),
        comparison.legacy_route,
        comparison.llm_route,
        comparison.llm_strategy or "-",
        comparison.engine_matched,
        comparison.operation_matched,
        comparison.status,
        comparison.reason[:200],
        question[:80],
    )
    return comparison
