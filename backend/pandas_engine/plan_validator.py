from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Hashable, Literal, Mapping
import unicodedata

import pandas as pd

from datastore.scope import scoped_mapping
from datastore.state import _df_namespace, _df_sources
from pandas_engine.money import money_unit_for_column, parse_money_value
from pandas_engine.query_grounding import (
    ground_query_plan_filters_by_type,
    parse_grounded_comparisons,
)
from pandas_engine.query_plan import FilterCondition, QueryPlan
from utils.semantic_schema import SYSTEM_COLUMNS, infer_data_type
from utils.table_parser import IDENTITY_INTERNAL_COLS


ValidationStatus = Literal[
    "valid",
    "clarification",
    "not_applicable",
    "invalid",
]

_INTERNAL_COLUMNS = set(SYSTEM_COLUMNS) | set(IDENTITY_INTERNAL_COLS)
_NUMERIC_TYPES = {"number", "money"}
_ORDERABLE_TYPES = _NUMERIC_TYPES | {"date", "year_month"}
_STRING_TYPES = {"string"}
_PLAN_COMPARISON_OPERATORS = {"gt", "gte", "lt", "lte", "between"}


@dataclass(frozen=True)
class _GroundedNumericCondition:
    operator: str
    value: float
    value_kind: Literal["money", "number", "unspecified"]
    original: str


@dataclass(frozen=True)
class PlanValidationIssue:
    code: str
    message: str
    field: str = ""
    column: str = ""


@dataclass(frozen=True)
class PlanValidationResult:
    status: ValidationStatus
    plan: QueryPlan
    issues: tuple[PlanValidationIssue, ...] = ()
    dataframe: pd.DataFrame | None = None
    source_file: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @property
    def is_accepted(self) -> bool:
        return self.status in {"valid", "clarification", "not_applicable"}

    @property
    def is_executable(self) -> bool:
        return self.status == "valid" and self.dataframe is not None


def _issue(
    code: str,
    message: str,
    *,
    field: str = "",
    column: str = "",
) -> PlanValidationIssue:
    return PlanValidationIssue(
        code=code,
        message=message,
        field=field,
        column=column,
    )


def column_data_type(df: pd.DataFrame, column: Hashable) -> str:
    schema = df.attrs.get("semantic_schema")
    if isinstance(schema, dict):
        columns = schema.get("columns")
        mapping = columns.get(str(column)) if isinstance(columns, dict) else None
        if isinstance(mapping, dict):
            data_type = str(mapping.get("data_type") or "").strip().casefold()
            if data_type:
                return data_type
    return infer_data_type(df[column])


def _is_internal_column(column: str) -> bool:
    return column in _INTERNAL_COLUMNS or column.startswith("_")


def _referenced_columns(plan: QueryPlan) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    if plan.target:
        references.append(("target", plan.target))
    references.extend(("select", column) for column in plan.select)
    references.extend(("filters", condition.column) for condition in plan.filters)
    references.extend(("sort", condition.column) for condition in plan.sort)
    references.extend(("distinct_by", column) for column in plan.distinct_by)
    return references


def _condition_values(condition: FilterCondition) -> list[object]:
    if condition.value is None:
        return []
    if isinstance(condition.value, tuple):
        return list(condition.value)
    return [condition.value]


def _values_match_type(condition: FilterCondition, data_type: str) -> bool:
    values = _condition_values(condition)
    if not values:
        return True

    if data_type == "money":
        return all(parse_money_value(value) is not None for value in values)
    if data_type == "number":
        converted = pd.to_numeric(pd.Series(values), errors="coerce")
        return bool(converted.notna().all())
    if data_type in {"date", "year_month"}:
        converted = pd.to_datetime(pd.Series(values), errors="coerce")
        return bool(converted.notna().all())
    if data_type == "boolean":
        return all(isinstance(value, bool) for value in values)
    return True


def _validate_filter_type(
    condition: FilterCondition,
    data_type: str,
) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    operator = condition.operator

    if operator == "contains" and data_type not in _STRING_TYPES:
        issues.append(
            _issue(
                "incompatible_operator",
                f"contains 연산은 문자열 컬럼에만 사용할 수 있습니다: {condition.column}",
                field="filters",
                column=condition.column,
            )
        )
        return issues

    if operator in {"gt", "gte", "lt", "lte", "between"} and data_type not in _ORDERABLE_TYPES:
        issues.append(
            _issue(
                "incompatible_operator",
                f"{operator} 연산을 적용할 수 없는 컬럼 형식입니다: {condition.column}",
                field="filters",
                column=condition.column,
            )
        )
        return issues

    if not _values_match_type(condition, data_type):
        issues.append(
            _issue(
                "incompatible_value",
                f"필터 값이 컬럼 형식과 맞지 않습니다: {condition.column}",
                field="filters",
                column=condition.column,
            )
        )
    return issues


def _question_numeric_conditions(question: str) -> list[_GroundedNumericCondition]:
    conditions: list[_GroundedNumericCondition] = []
    for comparison in parse_grounded_comparisons(question):
        if comparison.value_kind == "money":
            parsed = parse_money_value(comparison.value)
            if parsed is None:
                continue
            value = parsed
        else:
            value = float(comparison.value)
        conditions.append(
            _GroundedNumericCondition(
                operator=comparison.operator,
                value=value,
                value_kind=comparison.value_kind,
                original=comparison.source_text,
            )
        )
    return conditions


def _validate_filter_source_text(
    plan: QueryPlan,
    question: str,
) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    for condition in plan.filters:
        source_text = str(condition.source_text or "").strip()
        if not source_text:
            continue
        if source_text not in question:
            issues.append(
                _issue(
                    "invalid_filter_evidence",
                    (
                        "필터의 원문 근거가 실제 질문에 존재하지 않습니다: "
                        f"{source_text}"
                    ),
                    field="filters",
                    column=condition.column,
                )
            )
            continue
        if (
            condition.operator in _PLAN_COMPARISON_OPERATORS
            and len(parse_grounded_comparisons(source_text)) != 1
        ):
            issues.append(
                _issue(
                    "ambiguous_filter_evidence",
                    (
                        "숫자 필터의 원문 근거가 하나의 명확한 비교 조건이 "
                        f"아닙니다: {source_text}"
                    ),
                    field="filters",
                    column=condition.column,
                )
            )
    return issues


def _normalized_string_evidence(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", "", text)


def _validate_string_filter_grounding(
    plan: QueryPlan,
    question: str,
    df: pd.DataFrame,
    actual_columns: Mapping[str, Hashable],
) -> list[PlanValidationIssue]:
    """Reject LLM-created string literals absent from their question evidence."""

    issues: list[PlanValidationIssue] = []
    grounded_operators = {"eq", "ne", "contains", "in"}
    for condition in plan.filters:
        column = actual_columns.get(condition.column)
        if (
            column is None
            or column_data_type(df, column) != "string"
            or condition.operator not in grounded_operators
        ):
            continue

        source_text = str(condition.source_text or "").strip()
        evidence = source_text if source_text and source_text in question else question
        normalized_evidence = _normalized_string_evidence(evidence)
        values = _condition_values(condition)
        missing = [
            str(value)
            for value in values
            if _normalized_string_evidence(value) not in normalized_evidence
        ]
        if missing:
            issues.append(
                _issue(
                    "ungrounded_string_filter",
                    (
                        "질문 원문에서 확인되지 않은 문자열 조건이 "
                        f"조회 계획에 포함됐습니다: {condition.column}={missing}"
                    ),
                    field="filters",
                    column=condition.column,
                )
            )
    return issues


def _condition_numeric_parts(
    condition: FilterCondition,
    *,
    df: pd.DataFrame,
    column: Hashable,
    data_type: str,
) -> list[tuple[str, float]] | None:
    if condition.operator not in _PLAN_COMPARISON_OPERATORS:
        return []

    raw_values = _condition_values(condition)
    parsed: list[float] = []
    for value in raw_values:
        if data_type == "money":
            numeric = parse_money_value(
                value,
                money_unit_for_column(df, str(column)),
            )
        elif data_type == "number":
            converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            numeric = None if pd.isna(converted) else float(converted)
        else:
            return []
        if numeric is None:
            return None
        parsed.append(float(numeric))

    if condition.operator == "between":
        if len(parsed) != 2:
            return None
        return [("gte", parsed[0]), ("lte", parsed[1])]
    if len(parsed) != 1:
        return None
    return [(condition.operator, parsed[0])]


def _numeric_values_equal(left: float, right: float) -> bool:
    tolerance = max(abs(left), abs(right), 1.0) * 1e-12
    return abs(left - right) <= tolerance


def _validate_numeric_filter_grounding(
    plan: QueryPlan,
    question: str,
    df: pd.DataFrame,
    actual_columns: Mapping[str, Hashable],
) -> list[PlanValidationIssue]:
    """Reject numeric comparison filters that alter literals from the question."""

    expected = _question_numeric_conditions(question)
    if not expected:
        return []

    actual: list[tuple[str, float, str, str]] = []
    for condition in plan.filters:
        column = actual_columns.get(condition.column)
        if column is None:
            continue
        data_type = column_data_type(df, column)
        parts = _condition_numeric_parts(
            condition,
            df=df,
            column=column,
            data_type=data_type,
        )
        if parts is None:
            continue
        actual.extend(
            (operator, value, data_type, condition.column)
            for operator, value in parts
        )

    unmatched_actual = set(range(len(actual)))
    issues: list[PlanValidationIssue] = []
    for condition in expected:
        matched_index: int | None = None
        for index in unmatched_actual:
            operator, value, data_type, _ = actual[index]
            kind_matches = (
                condition.value_kind == "unspecified"
                or condition.value_kind == data_type
            )
            if (
                kind_matches
                and operator == condition.operator
                and _numeric_values_equal(value, condition.value)
            ):
                matched_index = index
                break
        if matched_index is None:
            issues.append(
                _issue(
                    "literal_mismatch",
                    (
                        "질문의 숫자·단위·비교 조건이 조회 계획에 정확히 "
                        f"보존되지 않았습니다: {condition.original}"
                    ),
                    field="filters",
                )
            )
        else:
            unmatched_actual.remove(matched_index)

    for index in sorted(unmatched_actual):
        operator, value, _, column = actual[index]
        issues.append(
            _issue(
                "ungrounded_numeric_filter",
                (
                    "질문 원문에서 확인되지 않은 숫자 비교 조건이 "
                    f"조회 계획에 포함됐습니다: {column} {operator} {value:g}"
                ),
                field="filters",
                column=column,
            )
        )
    return issues


def validate_query_plan(
    plan: QueryPlan,
    *,
    question: str | None = None,
    dataframes: Mapping[str, pd.DataFrame] | None = None,
    source_by_alias: Mapping[str, str] | None = None,
    explicit_dataframe_aliases: set[str] | frozenset[str] | None = None,
) -> PlanValidationResult:
    """Validate a parsed QueryPlan against the current document scope.

    This function never repairs an LLM plan. Any mismatch is returned as an
    explicit issue so an invalid or hallucinated reference cannot be executed.
    """

    if plan.status == "clarification":
        return PlanValidationResult(status="clarification", plan=plan)
    if plan.status == "not_applicable":
        return PlanValidationResult(status="not_applicable", plan=plan)

    all_dataframes = _df_namespace if dataframes is None else dataframes
    all_sources = _df_sources if source_by_alias is None else source_by_alias
    available = scoped_mapping(all_dataframes, all_sources)
    alias = str(plan.dataframe)

    if alias not in available:
        if alias in all_dataframes:
            issue = _issue(
                "dataframe_out_of_scope",
                f"선택한 문서 범위 밖의 데이터프레임입니다: {alias}",
                field="dataframe",
            )
        else:
            issue = _issue(
                "unknown_dataframe",
                f"존재하지 않는 데이터프레임입니다: {alias}",
                field="dataframe",
            )
        return PlanValidationResult(
            status="invalid",
            plan=plan,
            issues=(issue,),
        )

    df = available[alias]
    source_file = all_sources.get(alias, alias)

    available_sources = {
        str(all_sources.get(available_alias, available_alias))
        for available_alias in available
    }
    if len(available_sources) > 1:
        explicit_aliases = set(explicit_dataframe_aliases or ())
        if explicit_aliases != {alias}:
            issue = _issue(
                "ambiguous_document_scope",
                "여러 문서가 조회 대상입니다. 조회할 문서를 하나 선택해 주세요.",
                field="dataframe",
            )
            return PlanValidationResult(
                status="clarification",
                plan=plan,
                issues=(issue,),
            )

    actual_columns = {str(column): column for column in df.columns}
    issues: list[PlanValidationIssue] = []
    valid_references: dict[str, Hashable] = {}

    for field, column in _referenced_columns(plan):
        if _is_internal_column(column):
            issues.append(
                _issue(
                    "internal_column",
                    f"내부 시스템 컬럼은 조회할 수 없습니다: {column}",
                    field=field,
                    column=column,
                )
            )
        elif column not in actual_columns:
            issues.append(
                _issue(
                    "unknown_column",
                    f"실제 표에 존재하지 않는 컬럼입니다: {column}",
                    field=field,
                    column=column,
                )
            )
        else:
            valid_references[column] = actual_columns[column]

    if issues:
        return PlanValidationResult(
            status="invalid",
            plan=plan,
            issues=tuple(issues),
            dataframe=df,
            source_file=source_file,
        )

    if question:
        plan = ground_query_plan_filters_by_type(
            plan,
            question,
            {
                column: column_data_type(df, actual_column)
                for column, actual_column in valid_references.items()
            },
        )

    if plan.operation in {"sum", "mean", "median"} and plan.target:
        target_type = column_data_type(df, valid_references[plan.target])
        if target_type not in _NUMERIC_TYPES:
            issues.append(
                _issue(
                    "incompatible_target",
                    f"{plan.operation} 연산에는 숫자 또는 금액 컬럼이 필요합니다: {plan.target}",
                    field="target",
                    column=plan.target,
                )
            )

    if plan.operation in {"min", "max"} and plan.target:
        target_type = column_data_type(df, valid_references[plan.target])
        if target_type not in _ORDERABLE_TYPES:
            issues.append(
                _issue(
                    "incompatible_target",
                    f"{plan.operation} 연산에 사용할 수 없는 컬럼 형식입니다: {plan.target}",
                    field="target",
                    column=plan.target,
                )
            )

    for condition in plan.filters:
        data_type = column_data_type(df, valid_references[condition.column])
        issues.extend(_validate_filter_type(condition, data_type))

    if question:
        issues.extend(_validate_filter_source_text(plan, question))
        issues.extend(
            _validate_string_filter_grounding(
                plan,
                question,
                df,
                actual_columns,
            )
        )
        issues.extend(
            _validate_numeric_filter_grounding(
                plan,
                question,
                df,
                actual_columns,
            )
        )

    if issues:
        return PlanValidationResult(
            status="invalid",
            plan=plan,
            issues=tuple(issues),
            dataframe=df,
            source_file=source_file,
        )

    return PlanValidationResult(
        status="valid",
        plan=plan,
        dataframe=df,
        source_file=source_file,
    )
