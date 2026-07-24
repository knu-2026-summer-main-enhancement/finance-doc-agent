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
from pandas_engine.query_grounding import ground_query_plan_filters
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


def _schema_column_names(schema: str) -> tuple[str, ...]:
    columns: list[str] = []
    for line in schema.splitlines():
        if "컬럼(이 이름만 사용)" not in line and not line.strip().startswith("컬럼:"):
            continue
        for column in re.findall(r'"([^"]+)"', line):
            if column not in columns:
                columns.append(column)
    return tuple(columns)


def _lookup_field_candidates(question: str, schema: str) -> tuple[str, ...]:
    return tuple(
        column
        for column in _schema_column_names(schema)
        if column in question and not column.startswith("_")
    )


def _operation_hint_text(
    operation_hint: str | None,
    *,
    question: str = "",
    schema: str = "",
) -> str:
    if operation_hint == "lookup_field":
        text = (
            "lookup_field: 특정 대상의 금액이 아닌 컬럼값 조회. "
            "list 연산으로 대상을 필터링하고 요청 컬럼을 선택"
        )
        candidates = _lookup_field_candidates(question, schema)
        if candidates:
            text += ". 질문과 실제 스키마에 함께 존재하는 조회 컬럼: " + ", ".join(candidates)
        return text
    if operation_hint == "structured_query":
        return "structured_query: 범용 표 조건·정렬·선택 조회"
    operation_contracts = {
        "list_records": "list_records: operation=list으로 전체 행 목록을 반환",
        "filter_records": "filter_records: operation=list과 질문의 모든 조건 filters를 사용",
        "count_records": "count_records: operation=count. 사람/인원 질문이면 distinct_by에 사람 식별 컬럼 필수",
        "sum_amount": "sum_amount: operation=sum, target은 금액 컬럼, 질문의 조건은 모두 filters에 보존",
        "average_amount": "average_amount: operation=mean, target은 금액 컬럼",
        "median_amount": "median_amount: operation=median, target은 금액 컬럼",
        "mode_amount": "mode_amount: operation=mode, target은 금액 컬럼",
        "lookup_amount": "lookup_amount: 특정 대상의 금액 조회. 대상 식별 filters와 금액 target을 사용",
        "max_amount": "max_amount: operation=max, target은 금액 컬럼",
        "min_amount": "min_amount: operation=min, target은 금액 컬럼",
    }
    if operation_hint in operation_contracts:
        return operation_contracts[operation_hint]
    return "없음: 질문과 스키마만으로 계획 결정"


def _validate_operation_hint_contract(
    plan: QueryPlan,
    operation_hint: str | None,
) -> QueryPlan:
    """Ensure a specialized classifier decision survives plan generation."""

    if operation_hint != "lookup_field":
        return plan
    if plan.status != "ready":
        raise ValueError("lookup_field requires an executable ready plan")
    if plan.operation != "list":
        raise ValueError("lookup_field requires operation=list")
    if not plan.filters:
        raise ValueError("lookup_field requires a target-identifying filter")
    if not plan.select:
        raise ValueError("lookup_field requires at least one selected field")
    return plan


def _align_plan_with_operation_hint(
    plan: QueryPlan,
    operation_hint: str | None,
    question: str,
    schema: str,
) -> QueryPlan:
    """Fill only an unambiguous field selection grounded in question/schema."""

    if (
        operation_hint == "lookup_field"
        and plan.status == "ready"
        and plan.operation == "list"
        and plan.filters
    ):
        filter_columns = {condition.column for condition in plan.filters}
        candidates = tuple(
            column
            for column in _lookup_field_candidates(question, schema)
            if column not in filter_columns
        )
        if candidates:
            selected_filter_columns = tuple(
                column for column in plan.select if column in filter_columns
            )
            grounded_select = tuple(dict.fromkeys(
                (*selected_filter_columns, *candidates)
            ))
            if plan.select == grounded_select:
                return _validate_operation_hint_contract(plan, operation_hint)
            logger.info(
                "[QUERY_PLAN] lookup_field 조회 컬럼 안전 교정 | columns=%s",
                list(candidates),
            )
            plan = plan.model_copy(update={"select": grounded_select})
    return _validate_operation_hint_contract(plan, operation_hint)


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
    plan = ground_query_plan_filters(plan, question)
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
    operation_hint: str | None = None,
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
        operation_hint=_operation_hint_text(
            operation_hint,
            question=clean_question,
            schema=resolved_schema,
        ),
    )
    model = llm or get_llm_code()
    responses: list[str] = []
    first_error_message = ""

    raw = await model.ainvoke(original_prompt)
    responses.append(_response_text(raw))
    try:
        plan = _align_plan_with_operation_hint(
            _align_plan_with_question(
                parse_query_plan_response(raw),
                clean_question,
            ),
            operation_hint,
            clean_question,
            resolved_schema,
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
            "[QUERY_PLAN] 첫 응답 규격 오류, 형식 수정 재시도 | error_type=%s",
            type(first_error).__name__,
        )

    repair_prompt = _QUERY_PLAN_REPAIR_TEMPLATE.format(
        error=first_error_message,
        response=responses[0][:3000],
        question=clean_question,
        operation_hint=_operation_hint_text(
            operation_hint,
            question=clean_question,
            schema=resolved_schema,
        ),
    )
    repaired = await model.ainvoke(repair_prompt)
    responses.append(_response_text(repaired))
    try:
        plan = _align_plan_with_operation_hint(
            _align_plan_with_question(
                parse_query_plan_response(repaired),
                clean_question,
            ),
            operation_hint,
            clean_question,
            resolved_schema,
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
            "[QUERY_PLAN] 형식 수정 실패 | error_type=%s",
            type(second_error).__name__,
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
    operation_hint: str | None = None,
) -> PlanValidationResult:
    """Generate a plan and immediately enforce runtime DataFrame validation."""

    plan = await generate_query_plan(
        question,
        schema=schema,
        llm=llm,
        operation_hint=operation_hint,
    )
    if explicit_dataframe_aliases is None and dataframes is None:
        # 파일명·표시 레이블이 질문에 직접 나타난 경우에만 여러 문서 중 하나를
        # 명시적으로 선택한 것으로 인정한다.
        from datastore.query import _find_dfs_by_source_label

        explicit_dataframe_aliases = set(_find_dfs_by_source_label(question))
    return validate_query_plan(
        plan,
        question=question,
        dataframes=dataframes,
        source_by_alias=source_by_alias,
        explicit_dataframe_aliases=explicit_dataframe_aliases,
        operation_hint=operation_hint,
    )
