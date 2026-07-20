from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Literal, Mapping

from pandas_engine.money import MONEY_TEXT_UNITS, money_multiplier_for_unit
from pandas_engine.query_plan import FilterCondition, QueryPlan, ScalarValue


logger = logging.getLogger("uvicorn.error")

_OPERATOR_BY_TEXT = {
    "이상": "gte",
    ">=": "gte",
    "초과": "gt",
    ">": "gt",
    "이하": "lte",
    "<=": "lte",
    "미만": "lt",
    "<": "lt",
}
_NON_MONEY_UNITS = ("기", "명", "개", "건", "점", "%")
_UNIT_PATTERN = "|".join(
    re.escape(unit)
    for unit in (*MONEY_TEXT_UNITS, *_NON_MONEY_UNITS)
)
_NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_SUFFIX_COMPARISON_RE = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN})\s*"
    rf"(?P<unit>{_UNIT_PATTERN})?\s*"
    r"(?P<operator>이상|이하|초과|미만|>=|<=|>|<)"
)
_PREFIX_COMPARISON_RE = re.compile(
    r"(?P<operator>>=|<=|>|<)\s*"
    rf"(?P<number>{_NUMBER_PATTERN})\s*"
    rf"(?P<unit>{_UNIT_PATTERN})?"
)


@dataclass(frozen=True)
class GroundedComparison:
    operator: Literal["gt", "gte", "lt", "lte"]
    value: ScalarValue
    value_text: str
    value_kind: Literal["money", "number", "unspecified"]
    source_text: str


def parse_grounded_comparisons(source_text: str) -> tuple[GroundedComparison, ...]:
    """Parse explicit numeric comparisons from an exact question span."""

    text = str(source_text or "").strip()
    matches = list(_SUFFIX_COMPARISON_RE.finditer(text))
    matches.extend(_PREFIX_COMPARISON_RE.finditer(text))
    matches.sort(key=lambda match: match.start())

    comparisons: list[GroundedComparison] = []
    occupied: list[tuple[int, int]] = []
    for match in matches:
        span = match.span()
        if any(span[0] < end and start < span[1] for start, end in occupied):
            continue
        occupied.append(span)

        number_text = match.group("number")
        unit = match.group("unit") or ""
        literal = f"{number_text}{unit}"
        if money_multiplier_for_unit(unit) is not None:
            value: ScalarValue = literal
            value_kind: Literal["money", "number", "unspecified"] = "money"
        else:
            numeric = float(number_text.replace(",", ""))
            value = int(numeric) if numeric.is_integer() else numeric
            value_kind = "number" if unit else "unspecified"
        comparisons.append(
            GroundedComparison(
                operator=_OPERATOR_BY_TEXT[match.group("operator")],
                value=value,
                value_text=literal,
                value_kind=value_kind,
                source_text=match.group(0).strip(),
            )
        )
    return tuple(comparisons)


def ground_query_plan_filters(plan: QueryPlan, question: str) -> QueryPlan:
    """Correct only filters backed by one exact, unambiguous question span."""

    if plan.status != "ready" or not plan.filters:
        return plan

    changed = False
    grounded_filters: list[FilterCondition] = []
    for condition in plan.filters:
        source_text = str(condition.source_text or "").strip()
        if not source_text or source_text not in question:
            grounded_filters.append(condition)
            continue

        comparisons = parse_grounded_comparisons(source_text)
        if len(comparisons) != 1:
            grounded_filters.append(condition)
            continue

        grounded = comparisons[0]
        updates = {
            "operator": grounded.operator,
            "value": grounded.value,
            "source_text": source_text,
        }
        corrected = condition.model_copy(update=updates)
        changed = changed or corrected != condition
        grounded_filters.append(corrected)

    if not changed:
        return plan

    logger.warning("[QUERY_PLAN] 질문 원문 근거로 숫자 필터 안전 보정")
    return plan.model_copy(update={"filters": tuple(grounded_filters)})


def ground_query_plan_filters_by_type(
    plan: QueryPlan,
    question: str,
    column_types: Mapping[str, str],
) -> QueryPlan:
    """Recover a comparison only when its DataFrame type identifies one filter."""

    if plan.status != "ready" or not plan.filters:
        return plan

    comparisons = parse_grounded_comparisons(question)
    if not comparisons:
        return plan

    filters = list(plan.filters)
    used_filters: set[int] = set()
    changed = False

    for comparison in comparisons:
        compatible: list[int] = []
        evidence_matches: list[int] = []
        for index, condition in enumerate(filters):
            if index in used_filters or condition.operator not in {"gt", "gte", "lt", "lte"}:
                continue
            data_type = column_types.get(condition.column, "")
            if comparison.value_kind == "money" and data_type != "money":
                continue
            if comparison.value_kind == "number" and data_type != "number":
                continue
            if comparison.value_kind == "unspecified" and data_type not in {"money", "number"}:
                continue
            compatible.append(index)

            source_text = str(condition.source_text or "").strip()
            if source_text and source_text in question:
                evidence = parse_grounded_comparisons(source_text)
                if (
                    len(evidence) == 1
                    and evidence[0].source_text == comparison.source_text
                ):
                    evidence_matches.append(index)

        if len(evidence_matches) == 1:
            selected = evidence_matches[0]
        elif len(compatible) == 1:
            selected = compatible[0]
        else:
            continue

        condition = filters[selected]
        corrected = condition.model_copy(
            update={
                "operator": comparison.operator,
                "value": comparison.value,
                "source_text": comparison.source_text,
            }
        )
        filters[selected] = corrected
        used_filters.add(selected)
        changed = changed or corrected != condition

    if not changed:
        return plan

    logger.warning("[QUERY_PLAN] 컬럼 자료형과 질문 원문으로 숫자 필터 안전 보정")
    return plan.model_copy(update={"filters": tuple(filters)})
