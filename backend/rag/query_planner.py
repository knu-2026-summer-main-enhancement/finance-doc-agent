from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping

import pandas as pd
from pydantic import ValidationError

from core.llm import get_llm_code
from datastore.schema import _get_df_schema_filtered
from pandas_engine.plan_validator import (
    PlanValidationResult,
    validate_query_plan,
)
from pandas_engine.query_plan import QueryPlan
from rag.prompts import _QUERY_PLAN_REPAIR_TEMPLATE, _QUERY_PLAN_TEMPLATE


logger = logging.getLogger("uvicorn.error")

_EXPLICIT_OR = re.compile(
    r"(?:또는|혹은|아니면|(?:이|하)?거나|중\s*(?:하나|하나라도)|\bor\b|\|\|)",
    re.IGNORECASE,
)
_EXPLICIT_AND = re.compile(
    r"(?:그리고|동시에|모두|(?:이|하)고(?:\s|$)|이며|면서|\band\b|&&)",
    re.IGNORECASE,
)
_RANKED_LIST = re.compile(
    r"(?:큰|작은|높은|낮은|많은|적은)\s*순서(?:대로)?\s*"
    r"(?P<limit>\d+)\s*(?:개|건|명|행)?",
    re.IGNORECASE,
)
_EXPLICIT_FILTER_SCOPE = re.compile(
    r"(?:이상|이하|초과|미만|같은?|동일|포함|제외|아닌|없는|있는|"
    r"중(?:에서|에)?|가운데|부터|까지|사이|>=|<=|>|<)",
    re.IGNORECASE,
)


class QueryPlannerError(RuntimeError):
    def __init__(self, message: str, responses: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.responses = responses


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", None)
    if content is not None:
        return str(content).strip()
    return str(response).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Return the first complete JSON object without using permissive eval."""

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


def parse_query_plan_response(response: Any) -> QueryPlan:
    text = _response_text(response)
    if not text:
        raise ValueError("LLM이 빈 응답을 반환했습니다.")
    payload = _extract_json_object(text)
    # Small local models sometimes emit a single collection item as an object
    # instead of a one-element array. Normalize shape only; never invent or
    # alter columns, operators, values, or operations.
    for field in ("filters", "sort"):
        value = payload.get(field)
        if isinstance(value, dict):
            payload[field] = [value]
    for field in ("select", "distinct_by", "candidates"):
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = [value]
    return QueryPlan.model_validate(payload)


def _align_filter_logic_with_question(plan: QueryPlan, question: str) -> QueryPlan:
    """Keep flat multi-filter logic faithful to explicit question connectors."""

    if plan.status != "ready" or len(plan.filters) < 2:
        return plan

    has_or = bool(_EXPLICIT_OR.search(question))
    has_and = bool(_EXPLICIT_AND.search(question))
    desired_logic = "any" if has_or and not has_and else "all"
    if plan.filter_logic == desired_logic:
        return plan

    logger.warning(
        "[QUERY_PLAN] 질문 연결어 기준 필터 논리 보정 | generated=%s corrected=%s",
        plan.filter_logic,
        desired_logic,
    )
    return plan.model_copy(update={"filter_logic": desired_logic})


def _remove_ungrounded_rank_filters(plan: QueryPlan, question: str) -> QueryPlan:
    """Drop invented filters from an otherwise explicit sorted-list request."""

    if plan.status != "ready" or not plan.filters:
        return plan
    ranked = _RANKED_LIST.search(question)
    if ranked is None or _EXPLICIT_FILTER_SCOPE.search(question):
        return plan

    requested = int(ranked.group("limit"))
    planned = plan.effective_limit or plan.effective_top_n
    if planned != requested or not plan.sort:
        return plan

    logger.warning(
        "[QUERY_PLAN] 정렬 목록 질문에 없는 필터 제거 | filters=%d",
        len(plan.filters),
    )
    return plan.model_copy(update={"filters": (), "filter_logic": "all"})


def _align_plan_with_question(plan: QueryPlan, question: str) -> QueryPlan:
    plan = _align_filter_logic_with_question(plan, question)
    return _remove_ungrounded_rank_filters(plan, question)


def _validation_message(error: Exception) -> str:
    if isinstance(error, ValidationError):
        items: list[str] = []
        for issue in error.errors(include_url=False):
            location = ".".join(str(part) for part in issue.get("loc", ()))
            message = str(issue.get("msg") or "규격 오류")
            items.append(f"{location}: {message}" if location else message)
        return "\n".join(items)[:3000]
    return str(error)[:3000]


async def generate_query_plan(
    question: str,
    *,
    schema: str | None = None,
    llm: Any | None = None,
) -> QueryPlan:
    """Generate a QueryPlan and retry once for JSON/schema repair only."""

    clean_question = str(question or "").strip()
    if not clean_question:
        raise QueryPlannerError("빈 질문으로는 QueryPlan을 생성할 수 없습니다.")

    resolved_schema = (
        _get_df_schema_filtered(clean_question)
        if schema is None
        else str(schema)
    )
    original_prompt = _QUERY_PLAN_TEMPLATE.format(
        schema=resolved_schema or "(조회 가능한 DataFrame 없음)",
        question=clean_question,
    )
    model = llm or get_llm_code()
    responses: list[str] = []
    first_error_message = ""

    raw = await model.ainvoke(original_prompt)
    responses.append(_response_text(raw))
    try:
        plan = _align_plan_with_question(
            parse_query_plan_response(raw),
            clean_question,
        )
        logger.info(
            "[QUERY_PLAN] 생성 성공 | status=%s operation=%s dataframe=%s",
            plan.status,
            plan.operation,
            plan.dataframe,
        )
        return plan
    except (ValueError, TypeError, ValidationError) as first_error:
        first_error_message = _validation_message(first_error)
        logger.warning(
            "[QUERY_PLAN] 첫 응답 규격 오류, 형식 수정 재시도 | err=%s",
            first_error_message[:500],
        )

    repair_prompt = _QUERY_PLAN_REPAIR_TEMPLATE.format(
        error=first_error_message,
        response=responses[0][:3000],
        question=clean_question,
    )
    repaired = await model.ainvoke(repair_prompt)
    responses.append(_response_text(repaired))
    try:
        plan = _align_plan_with_question(
            parse_query_plan_response(repaired),
            clean_question,
        )
        logger.info(
            "[QUERY_PLAN] 형식 수정 성공 | status=%s operation=%s dataframe=%s",
            plan.status,
            plan.operation,
            plan.dataframe,
        )
        return plan
    except (ValueError, TypeError, ValidationError) as second_error:
        logger.error(
            "[QUERY_PLAN] 형식 수정 실패 | err=%s",
            _validation_message(second_error)[:500],
        )
        raise QueryPlannerError(
            "LLM 응답을 안전한 QueryPlan으로 변환하지 못했습니다.",
            tuple(responses),
        ) from second_error


async def generate_validated_query_plan(
    question: str,
    *,
    schema: str | None = None,
    llm: Any | None = None,
    dataframes: Mapping[str, pd.DataFrame] | None = None,
    source_by_alias: Mapping[str, str] | None = None,
    explicit_dataframe_aliases: set[str] | frozenset[str] | None = None,
) -> PlanValidationResult:
    """Generate a plan and immediately enforce runtime DataFrame validation."""

    plan = await generate_query_plan(
        question,
        schema=schema,
        llm=llm,
    )
    if explicit_dataframe_aliases is None and dataframes is None:
        # 파일명·표시 레이블이 질문에 직접 나타난 경우에만 여러 문서 중 하나를
        # 명시적으로 선택한 것으로 인정한다.
        from datastore.query import _find_dfs_by_source_label

        explicit_dataframe_aliases = set(_find_dfs_by_source_label(question))
    return validate_query_plan(
        plan,
        dataframes=dataframes,
        source_by_alias=source_by_alias,
        explicit_dataframe_aliases=explicit_dataframe_aliases,
    )
