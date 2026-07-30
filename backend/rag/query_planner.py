from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping

import pandas as pd
from pydantic import ValidationError

from core.llm import get_llm_code, response_text
from datastore.schema import _get_df_schema_filtered
from rag.cancellation import await_cancellable
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
_CONTAINS_LANGUAGE = re.compile(
    r"(?:\ud3ec\ud568|\ub4e4\uc5b4\uac00|\ub4e4\uc5b4\uac04|\uc2dc\uc791|\ub05d\ub098|\ub05d\ub0a8|\ub05d\ub098\ub294)"
)
_OR_BETWEEN_VALUES = re.compile(
    r"(?:\ub610\ub294|\ud639\uc740|\uc544\ub2c8\uba74|\uc774\uac70\ub098|\uac70\ub098|\uc774\ub098|\ub098)"
)
_LLM_LITERAL_WRAPPERS = ".^$*+?\\"


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
        "max_person_by_amount": (
            "max_person_by_amount: operation=group_sum, target은 금액 컬럼, "
            "group_by는 사람 식별 컬럼, group_order=desc, top_n=1"
        ),
        "min_person_by_amount": (
            "min_person_by_amount: operation=group_sum, target은 금액 컬럼, "
            "group_by는 사람 식별 컬럼, group_order=asc, top_n=1"
        ),
    }
    if operation_hint in operation_contracts:
        return operation_contracts[operation_hint]
    return "없음: 질문과 스키마만으로 계획 결정"


def _validate_operation_hint_contract(
    plan: QueryPlan,
    operation_hint: str | None,
) -> QueryPlan:
    """Ensure a specialized classifier decision survives plan generation."""

    if operation_hint in {"max_person_by_amount", "min_person_by_amount"}:
        expected_order = "desc" if operation_hint == "max_person_by_amount" else "asc"
        if (
            plan.status != "ready"
            or plan.operation != "group_sum"
            or not plan.target
            or not plan.group_by
            or plan.group_order != expected_order
        ):
            raise ValueError(
                f"{operation_hint} requires ranked person group_sum with {expected_order} order"
            )
        return plan
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
    text = response_text(response)
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

    same_column = len({condition.column for condition in plan.filters}) == 1
    scalar_values = [
        str(condition.value)
        for condition in plan.filters
        if isinstance(condition.value, (str, int, float))
    ]
    value_positions = sorted(
        (
            question.find(value),
            question.find(value) + len(value),
        )
        for value in scalar_values
        if value and question.find(value) >= 0
    )
    has_value_or = bool(
        same_column
        and len(value_positions) >= 2
        and any(
            _OR_BETWEEN_VALUES.search(
                question[left_end:right_start]
            )
            for (_, left_end), (right_start, _) in zip(
                value_positions,
                value_positions[1:],
            )
        )
    )
    has_or = bool(_EXPLICIT_OR.search(question)) or has_value_or
    has_and = bool(_EXPLICIT_AND.search(question))
    desired_logic = (
        "any"
        if has_or and (same_column or not has_and)
        else "all"
    )
    if plan.filter_logic == desired_logic:
        return plan

    logger.warning(
        "[QUERY_PLAN] 질문 연결어 기준 필터 논리 보정 | generated=%s corrected=%s",
        plan.filter_logic,
        desired_logic,
    )
    return plan.model_copy(update={"filter_logic": desired_logic})


def _literal_in_question(value: object, question: str) -> str | None:
    if not isinstance(value, str):
        return None
    literal = value.strip()
    if literal and literal in question:
        return literal
    stripped = literal.strip(_LLM_LITERAL_WRAPPERS)
    if stripped and stripped in question:
        return stripped
    return None


def _explicit_filter_column(
    condition: Any,
    question: str,
    schema: str,
    literal: str | None,
) -> str | None:
    if not literal:
        return None
    value_position = question.find(literal)
    if value_position < 0:
        return None
    candidates = [
        (
            question.rfind(column, 0, value_position),
            question.rfind(column, 0, value_position) + len(column),
            column,
        )
        for column in _schema_column_names(schema)
        if column in question
    ]
    candidates = [
        (start, end, column)
        for start, end, column in candidates
        if start >= 0
        and not any(
            other_start <= start
            and end <= other_end
            and (other_end - other_start) > (end - start)
            for other_start, other_end, _ in candidates
            if other_start >= 0
        )
    ]
    if not candidates:
        return None
    _, _, closest = max(candidates)
    return closest if closest != condition.column else None


def _ground_string_filters(
    plan: QueryPlan,
    question: str,
    schema: str,
) -> QueryPlan:
    """Repair only string-filter details directly recoverable from the question."""

    if plan.status != "ready" or not plan.filters:
        return plan

    same_column_alternatives = bool(
        len(plan.filters) >= 2
        and len({condition.column for condition in plan.filters}) == 1
        and _align_filter_logic_with_question(plan, question).filter_logic == "any"
    )
    changed = False
    grounded = []
    for condition in plan.filters:
        if condition.operator not in {"eq", "ne", "contains", "in"}:
            grounded.append(condition)
            continue
        source_text = str(condition.source_text or "").strip()
        if condition.operator == "in" and isinstance(condition.value, tuple):
            literals = [
                _literal_in_question(value, question)
                for value in condition.value
            ]
            if all(literals):
                positions = [
                    (question.find(literal), question.find(literal) + len(literal))
                    for literal in literals
                ]
                start = min(position[0] for position in positions)
                end = max(position[1] for position in positions)
                updates: dict[str, Any] = {}
                normalized_values = tuple(str(literal) for literal in literals)
                if condition.value != normalized_values:
                    updates["value"] = normalized_values
                if not source_text or source_text not in question:
                    updates["source_text"] = question[start:end]
                explicit_column = _explicit_filter_column(
                    condition,
                    question,
                    schema,
                    str(literals[0]),
                )
                if explicit_column is not None:
                    updates["column"] = explicit_column
                corrected = (
                    condition.model_copy(update=updates)
                    if updates
                    else condition
                )
                grounded.append(corrected)
                changed = changed or corrected != condition
                continue
            if not source_text or source_text not in question:
                changed = True
                continue
            grounded.append(condition)
            continue

        literal = _literal_in_question(condition.value, question)
        if literal is None:
            if not source_text or source_text not in question:
                changed = True
                continue
            grounded.append(condition)
            continue

        updates: dict[str, Any] = {}
        if condition.value != literal:
            updates["value"] = literal
        if not source_text or source_text not in question:
            updates["source_text"] = literal
        if (
            condition.operator == "contains"
            and same_column_alternatives
            and not _CONTAINS_LANGUAGE.search(question)
        ):
            updates["operator"] = "eq"
        explicit_column = _explicit_filter_column(
            condition,
            question,
            schema,
            literal,
        )
        if explicit_column is not None:
            updates["column"] = explicit_column

        corrected = condition.model_copy(update=updates) if updates else condition
        grounded.append(corrected)
        changed = changed or corrected != condition

    if not changed:
        return plan
    logger.warning("[QUERY_PLAN] 질문 원문으로 문자열 필터 안전 보정")
    return plan.model_copy(update={"filters": tuple(grounded)})


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


def _align_plan_with_question(
    plan: QueryPlan,
    question: str,
    schema: str,
) -> QueryPlan:
    plan = _ground_string_filters(plan, question, schema)
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

    raw = await await_cancellable(model.ainvoke(original_prompt))
    responses.append(response_text(raw))
    try:
        plan = _align_plan_with_operation_hint(
            _align_plan_with_question(
                parse_query_plan_response(raw),
                clean_question,
                resolved_schema,
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
    repaired = await await_cancellable(model.ainvoke(repair_prompt))
    responses.append(response_text(repaired))
    try:
        plan = _align_plan_with_operation_hint(
            _align_plan_with_question(
                parse_query_plan_response(repaired),
                clean_question,
                resolved_schema,
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
